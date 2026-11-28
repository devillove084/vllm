# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare and select native serving layouts from workload and capacity."""

import json
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from vllm.entrypoints.serve.plait_deploy import Deployment

from .calibration import Calibrator, Settings, Target
from .group import (
    DIRECTORY_POLL_SECONDS,
    Control,
    ControlError,
    Endpoint,
    endpoint_ready,
)
from .load import Load
from .mode_state import ProfileStore, fingerprint

VERSION = re.compile(r"[a-f0-9]{32}@[1-9][0-9]*")
GROUP_ID = re.compile(r"[a-f0-9]{32}")


@dataclass(frozen=True)
class Capacity:
    versions: frozenset[str]
    devices: int
    rps: float
    valid_until_ms: int

    @classmethod
    def read(cls, profile: dict) -> "Capacity":
        versions = profile["versions"]
        if (
            not isinstance(versions, list)
            or not versions
            or any(not isinstance(v, str) or not VERSION.fullmatch(v) for v in versions)
            or len({v.split("@", 1)[0] for v in versions}) != len(versions)
        ):
            raise ValueError("invalid versioned SLO profile")
        return cls(frozenset(versions), *profile_limits(profile, len(versions)))


def profile_limits(profile: dict, groups: int) -> tuple[int, float, int]:
    slo = profile["slo_profile"]
    devices = profile["devices"]
    expires = slo["valid_until_ms"]
    if (
        type(devices) is not int
        or devices < groups
        or type(expires) is not int
        or expires <= 0
    ):
        raise ValueError("invalid device count or lifetime in SLO profile")
    return devices, number(slo["rps"], positive=True), expires


def group_id(text: str) -> str:
    if not isinstance(text, str) or not GROUP_ID.fullmatch(text):
        raise ValueError("expected a lowercase 32-digit Plait identity")
    return text


@dataclass(frozen=True)
class PlanMember:
    group: str
    hosts: tuple[str, ...]
    workload: dict
    calibration: str | None

    @classmethod
    def read(cls, item: dict, parent: Path, model: str | None) -> "PlanMember":
        deployment = Deployment.read(parent / item["deployment"])
        workload = item["workload"]
        if not isinstance(workload, dict) or (
            model is not None and deployment.model_alias != model
        ):
            raise ValueError("invalid provision workload or deployment")
        calibration = item.get("calibration")
        return cls(
            deployment.group_id,
            tuple(sorted(deployment.hosts)),
            workload,
            group_id(calibration) if calibration is not None else None,
        )

    def request(self) -> dict:
        return {
            "group": list(bytes.fromhex(self.group)),
            "workload": self.workload,
            "hosts": [list(bytes.fromhex(host)) for host in self.hosts],
            "calibration": (
                list(bytes.fromhex(self.calibration))
                if self.calibration is not None
                else None
            ),
        }


@dataclass(frozen=True)
class Provision:
    members: tuple[PlanMember, ...]
    devices: int
    rps: float
    valid_until_ms: int
    bridge: str | None = None

    @classmethod
    def read(cls, profile: dict, parent: Path, model: str | None) -> "Provision":
        members = tuple(
            PlanMember.read(item, parent, model) for item in profile["groups"]
        )
        if not members or len({member.group for member in members}) != len(members):
            raise ValueError("provision needs unique execution groups")
        bridge = profile.get("bridge")
        if bridge is not None and (
            not isinstance(bridge, str)
            or not VERSION.fullmatch(bridge)
            or bridge.split("@", 1)[0] in {member.group for member in members}
        ):
            raise ValueError("a provision bridge needs a separate versioned group")
        return cls(members, *profile_limits(profile, len(members)), bridge)


def number(value: object, *, positive: bool = False) -> float:
    if (
        type(value) not in (float, int)
        or not math.isfinite(value)
        or value < 0
        or (positive and value == 0)
    ):
        raise ValueError("expected a finite nonnegative policy amount")
    return float(value)


@dataclass(frozen=True)
class Policy:
    route_revision: int
    window_seconds: float
    stable_windows: int
    horizon_seconds: float
    cooldown_seconds: float
    device_cost_requests_per_second: float
    switch_loss_requests: float
    minimum_gain_requests: float
    modes: tuple[Capacity, ...]
    provision: tuple[Provision, ...]
    calibration: Settings | None

    @classmethod
    def read(cls, path: Path, model: str | None = None) -> "Policy":
        config = json.loads(path.read_text(encoding="utf-8"))
        revision = config["route_revision"]
        stable = config["stable_windows"]
        modes = tuple(Capacity.read(profile) for profile in config["modes"])
        provision = tuple(
            Provision.read(profile, path.parent, model)
            for profile in config.get("provision", [])
        )
        calibration = Settings.read(config["calibration"]) if "calibration" in config else None
        bridges = [
            frozenset([target.bridge]) for target in provision if target.bridge is not None
        ]
        mode_versions = {mode.versions for mode in modes}
        if (
            type(revision) is not int
            or revision < 1
            or type(stable) is not int
            or stable < 1
            or len(mode_versions) != len(modes)
            or len(set(bridges)) != len(bridges)
            or (calibration is None and not mode_versions.issuperset(bridges))
        ):
            raise ValueError("invalid serving mode policy")
        return cls(
            revision,
            number(config["window_seconds"], positive=True),
            stable,
            number(config["horizon_seconds"], positive=True),
            number(config["cooldown_seconds"]),
            number(config["device_cost_requests_per_second"]),
            number(config["switch_loss_requests"]),
            number(config["minimum_gain_requests"]),
            modes,
            provision,
            calibration,
        )


def gain(
    policy: Policy, incumbent: Capacity, target: Capacity | Provision, demand: float
) -> float:
    baseline = min(demand, incumbent.rps)
    served = min(demand, target.rps)
    if served < baseline:
        return 0.0
    devices_saved = incumbent.devices - target.devices
    return (
        policy.horizon_seconds
        * (served - baseline + devices_saved * policy.device_cost_requests_per_second)
        - policy.switch_loss_requests
    )


def endpoints_for(
    versions: frozenset[str], routes: tuple[tuple[Endpoint, ...], ...]
) -> list[Endpoint]:
    endpoints = {
        endpoint
        for path in routes
        if all(endpoint.version in versions for endpoint in path)
        for endpoint in path
    }
    if (
        len(endpoints) != len(versions)
        or {endpoint.version for endpoint in endpoints} != versions
    ):
        return []
    return sorted(endpoints, key=lambda endpoint: endpoint.version)


@dataclass(frozen=True)
class ModeView:
    revision: int
    routes: tuple[tuple[Endpoint, ...], ...]
    groups: tuple[str, ...]
    active: frozenset[str]

    def projection_routes(self, target: Provision) -> tuple[tuple[Endpoint, ...], ...]:
        groups = {member.group for member in target.members}
        routes = tuple(
            route
            for route in self.routes
            if all(
                endpoint.version not in self.active
                and endpoint.version.split("@", 1)[0] in groups
                for endpoint in route
            )
        )
        versions = {endpoint.version for route in routes for endpoint in route}
        if (
            len(versions) != len(groups)
            or {v.split("@", 1)[0] for v in versions} != groups
        ):
            return ()
        return routes


class ModeController:
    def __init__(
        self,
        policy: Path,
        load: Load,
        control: Control,
        model: str | None = None,
        calibrator: Calibrator | None = None,
        profile_store: ProfileStore | None = None,
    ):
        self.path = policy
        self.load = load
        self.control = control
        self.model = model
        self.calibrator = calibrator
        self.profile_store = profile_store
        self.restore_pending: list[dict] = []
        self.measured: dict[frozenset[str], tuple[Target, Capacity]] = {}
        self.previous: dict | None = None
        self.last_policy: Policy | None = None
        self.proposal: tuple[Capacity, Capacity | Provision] | None = None
        self.windows = 0
        self.cooldown_until = 0.0
        self.recover_at = 0.0

    def reset_evidence(self) -> None:
        self.previous = None
        self.proposal = None
        self.windows = 0

    def watch(
        self, stop: threading.Event, snapshot: Callable[[], ModeView | None]
    ) -> None:
        last_error = None
        try:
            while not stop.is_set():
                try:
                    view = snapshot()
                    if view is None:
                        self.reset_evidence()
                        if self.calibrator is not None:
                            self.calibrator.cancel()
                    else:
                        self.reconcile(view, snapshot)
                    last_error = None
                except (
                    ControlError,
                    OSError,
                    ValueError,
                    KeyError,
                    TypeError,
                    OverflowError,
                ) as error:
                    self.reset_evidence()
                    if self.calibrator is not None:
                        self.calibrator.cancel()
                    if str(error) != last_error:
                        print(f"Plait mode policy: {error}", flush=True)
                    last_error = str(error)
                stop.wait(DIRECTORY_POLL_SECONDS)
        finally:
            if self.calibrator is not None:
                self.calibrator.cancel()
            self.control.close()

    def sample(self, window_seconds: float) -> float | None:
        current = self.load.snapshot()
        previous = self.previous
        if (
            previous is None
            or current["session"] != previous["session"]
            or current["uptime_ms"] < previous["uptime_ms"]
            or current["arrivals"] < previous["arrivals"]
        ):
            self.previous = current
            self.proposal = None
            self.windows = 0
            return None
        elapsed = (current["uptime_ms"] - previous["uptime_ms"]) / 1000
        if elapsed < window_seconds:
            return None
        self.previous = current
        return (current["arrivals"] - previous["arrivals"]) / elapsed

    def reconcile(
        self, view: ModeView, snapshot: Callable[[], ModeView | None]
    ) -> None:
        routes, active = view.routes, view.active
        policy = Policy.read(self.path, self.model)
        if policy != self.last_policy:
            if (
                self.last_policy is None
                or self.state_context(policy) != self.state_context(self.last_policy)
            ):
                self.measured.clear()
                self.restore_pending = self.load_profiles(policy)
            if self.calibrator is not None:
                self.calibrator.cancel()
            self.last_policy = policy
            self.reset_evidence()
        if view.revision == policy.route_revision:
            self.reap_failed_sets(policy)
            self.restore_profiles(view, policy)
        self.finish_calibration(view, policy, snapshot)
        demand = self.sample(policy.window_seconds)
        if (
            demand is None
            or view.revision != policy.route_revision
            or time.monotonic() < self.cooldown_until
        ):
            return
        now_ms = time.time_ns() // 1_000_000
        self.measured = {
            versions: (target, profile)
            for versions, (target, profile) in self.measured.items()
            if profile.valid_until_ms > now_ms
            and target.revision == view.revision
            and all(path in routes for path in target.routes)
        }
        profiles = {
            versions: profile for versions, (_, profile) in self.measured.items()
        }
        profiles.update(
            (mode.versions, mode)
            for mode in policy.modes
            if mode.valid_until_ms > now_ms
        )
        eligible = [
            mode for mode in profiles.values() if endpoints_for(mode.versions, routes)
        ]
        incumbent = next((mode for mode in eligible if mode.versions == active), None)
        if incumbent is None:
            self.proposal = None
            self.windows = 0
            return
        bridges = {
            frozenset([target.bridge]): target
            for target in policy.provision
            if target.bridge is not None
        }
        if target := bridges.get(active):
            self.advance_reuse(view, snapshot, policy, demand, incumbent, target, profiles)
            return
        candidates = sorted(
            (
                (gain(policy, incumbent, mode, demand), mode)
                for mode in eligible
                if mode != incumbent and mode.versions not in bridges
            ),
            key=lambda item: (-item[0], item[1].devices, sorted(item[1].versions)),
        )
        chosen: tuple[float, Capacity | Provision] | None = next(
            (
                (score, mode)
                for score, mode in candidates
                if math.isfinite(score)
                and score > 0
                and score >= policy.minimum_gain_requests
                and self.ready(mode, routes, active)
            ),
            None,
        )
        if chosen is None:
            known = {mode.versions for mode in eligible}
            projections = (
                (gain(policy, incumbent, target, demand), target)
                for target in policy.provision
                if target.valid_until_ms > now_ms
                and frozenset(
                    endpoint.version
                    for path in view.projection_routes(target)
                    for endpoint in path
                )
                not in known
            )
            chosen = max(
                (
                    (score, target)
                    for score, target in projections
                    if math.isfinite(score)
                    and score > 0
                    and score >= policy.minimum_gain_requests
                ),
                key=lambda item: item[0],
                default=None,
            )
        proposal = (incumbent, chosen[1]) if chosen is not None else None
        self.windows = self.windows + 1 if proposal == self.proposal else 1
        self.proposal = proposal
        if chosen is None or self.windows < policy.stable_windows:
            return
        score, target = chosen
        if (
            not self.ready(incumbent, routes, active, verify_http=False)
            or not self.current(view, snapshot, policy, incumbent, target)
        ):
            self.proposal = None
            self.windows = 0
            return
        if isinstance(target, Provision):
            if not self.calibrate(view, view.projection_routes(target), policy.calibration):
                decision = self.prepare(view, target)
                if decision == "prepare_set":
                    print(
                        f"Plait prepared a candidate layout at {demand:.2f} requests/s",
                        flush=True,
                    )
                elif decision == "no_placement" and target.bridge is not None:
                    self.begin_reuse(view, snapshot, policy, demand, incumbent, target, profiles)
        else:
            self.switch(view.groups, active, target.versions)
            if self.calibrator is not None:
                self.calibrator.cancel()
            self.cooldown_until = time.monotonic() + policy.cooldown_seconds
            print(
                f"Plait serving mode switched at {demand:.2f} requests/s "
                f"with projected gain {score:.2f} requests",
                flush=True,
            )
        self.proposal = None
        self.windows = 0

    def begin_reuse(
        self,
        view: ModeView,
        snapshot: Callable[[], ModeView | None],
        policy: Policy,
        demand: float,
        incumbent: Capacity,
        target: Provision,
        profiles: dict[frozenset[str], Capacity],
    ) -> None:
        assert target.bridge is not None
        bridge = profiles.get(frozenset([target.bridge]))
        if bridge is not None and bridge.rps < demand:
            return
        request = self.plan_request(view, target)
        request["after_releasing"] = [
            {"group": list(bytes.fromhex(group)), "epoch": epoch}
            for group, epoch in sorted(active_epochs(view.active).items())
        ]
        preview = self.control.request(request)
        if (
            preview["decision"] != "after_release"
            or sum(len(plan["workers"]) for plan in preview["plans"]) != target.devices
            or not self.current(view, snapshot, policy, incumbent, target)
        ):
            return
        if bridge is None:
            bridge_routes = tuple(
                path
                for path in view.routes
                if len(path) == 1 and path[0].version == target.bridge
            )
            self.calibrate(view, bridge_routes, policy.calibration)
            return
        if not self.ready(bridge, view.routes, view.active) or not self.current(
            view, snapshot, policy, incumbent, target, bridge
        ):
            return
        self.switch(view.groups, view.active, bridge.versions)
        self.cooldown_until = time.monotonic() + policy.cooldown_seconds
        print("Plait activated a temporary layout for device reuse", flush=True)

    def advance_reuse(
        self,
        view: ModeView,
        snapshot: Callable[[], ModeView | None],
        policy: Policy,
        demand: float,
        bridge: Capacity,
        target: Provision,
        profiles: dict[frozenset[str], Capacity],
    ) -> None:
        if not self.current(view, snapshot, policy, bridge, target):
            return
        groups = {member.group for member in target.members}
        mode = next(
            (
                candidate
                for candidate in profiles.values()
                if candidate.devices == target.devices
                and candidate.rps >= min(demand, bridge.rps)
                and candidate.versions.isdisjoint(view.active)
                and active_epochs(candidate.versions).keys() == groups
                and self.ready(candidate, view.routes, view.active)
            ),
            None,
        )
        if mode is not None:
            if self.current(view, snapshot, policy, bridge, target, mode):
                self.switch(view.groups, view.active, mode.versions)
                self.cooldown_until = time.monotonic() + policy.cooldown_seconds
                print("Plait activated the layout prepared after device release", flush=True)
        elif not self.calibrate(view, view.projection_routes(target), policy.calibration):
            self.prepare(view, target)

    def current(
        self,
        view: ModeView,
        snapshot: Callable[[], ModeView | None],
        policy: Policy,
        *profiles: Capacity | Provision,
    ) -> bool:
        return (
            snapshot() == view
            and Policy.read(self.path, self.model) == policy
            and min(profile.valid_until_ms for profile in profiles)
            > time.time_ns() // 1_000_000
        )

    def reap_failed_sets(self, policy: Policy) -> None:
        now = time.monotonic()
        if not policy.provision or now < self.recover_at:
            return
        self.recover_at = now + 1.0
        groups = {member.group for plan in policy.provision for member in plan.members}
        visited = set()
        for group in sorted(groups):
            if group in visited:
                continue
            preparation = self.control.request(
                {
                    "request": "preparation",
                    "group": list(bytes.fromhex(group)),
                }
            )["preparation"]
            if preparation is None:
                continue
            members = preparation["members"]
            versions = [version for version, _ in members]
            phases = {phase for _, phase in members}
            if (
                preparation["activated"]
                or not phases.intersection(("Revoked", "Retired"))
                or not phases.intersection(("Preparing", "Ready"))
            ):
                continue
            visited.update(bytes(version["group"]).hex() for version in versions)
            request = {
                "request": "apply",
                "operation": {
                    "id": self.control.next_operation_id(),
                    "action": {
                        "AbortFailedSet": {
                            "prepare": preparation["prepare"],
                            "versions": versions,
                        }
                    },
                },
            }
            try:
                result = self.control.request(request)["result"]
            except ControlError:
                result = self.control.request(request)["result"]
            if "Ok" not in result:
                if result["Err"] in ("PreparationActivated", "PreparationHealthy"):
                    continue
                raise ControlError(
                    f"Plait rejected candidate rollback: {result['Err']}"
                )
            if result["Ok"]["AbortedSet"] != versions:
                raise ControlError("Plait returned a different aborted candidate set")
            print("Plait revoked an incomplete candidate set", flush=True)

    def state_context(self, policy: Policy, *, legacy: bool = False) -> str:
        settings = (
            asdict(policy.calibration) if policy.calibration is not None else None
        )
        if settings is not None and not legacy:
            # A completed trial measures the timed workload of an exact live
            # layout. Warmup count controls how subsequent trials prepare it.
            del settings["warmups"]
        return fingerprint(
            {
                "model": self.model,
                "revision": policy.route_revision,
                "settings": settings,
            }
        )

    def load_profiles(self, policy: Policy) -> list[dict]:
        if self.profile_store is None or policy.calibration is None:
            return []
        contexts = {self.state_context(policy), self.state_context(policy, legacy=True)}
        try:
            return [
                record
                for record in self.profile_store.read()
                if record["context"] in contexts
            ]
        except (OSError, ValueError, KeyError, TypeError) as error:
            print(f"Plait could not restore calibration profiles: {error}", flush=True)
            return []

    def restore_profiles(self, view: ModeView, policy: Policy) -> None:
        if not self.restore_pending:
            return
        context = self.state_context(policy)
        waiting = []
        now_ms = time.time_ns() // 1_000_000
        for record in self.restore_pending:
            try:
                capacity = Capacity.read(record)
                manifest = record["manifest"]
                binding = record["binding"]
            except (ValueError, KeyError, TypeError, OverflowError):
                continue
            if capacity.valid_until_ms <= now_ms:
                continue
            routes = tuple(
                path
                for path in view.routes
                if all(endpoint.version in capacity.versions for endpoint in path)
            )
            if not endpoints_for(capacity.versions, routes):
                waiting.append(record)
                continue
            if route_fingerprint(routes) != binding:
                continue
            layout = self.inspect_layout(capacity.versions, routes, view.active)
            if layout is None:
                waiting.append(record)
            elif layout == (capacity.devices, manifest) and manifest is not None:
                if self.profile_store is not None and record["context"] != context:
                    try:
                        capacity = Capacity.read(
                            self.profile_store.save(
                                {**record, "context": context}, now_ms
                            )
                        )
                    except (OSError, KeyError, TypeError, ValueError) as error:
                        print(
                            f"Plait could not migrate a capacity profile: {error}",
                            flush=True,
                        )
                target = Target(
                    view.revision, view.active, routes, capacity.devices, manifest
                )
                current = self.measured.get(capacity.versions)
                if (
                    current is None
                    or capacity.valid_until_ms > current[1].valid_until_ms
                ):
                    self.measured[capacity.versions] = (target, capacity)
        self.restore_pending = waiting

    def calibrate(
        self,
        view: ModeView,
        routes: tuple[tuple[Endpoint, ...], ...],
        settings: Settings | None,
    ) -> bool:
        if settings is None or self.calibrator is None or self.model is None:
            return False
        if not routes:
            return False
        versions = frozenset(endpoint.version for path in routes for endpoint in path)
        layout = self.inspect_layout(versions, routes, view.active)
        if layout is None:
            return False
        devices, manifest = layout
        self.calibrator.start(
            Target(view.revision, view.active, routes, devices, manifest),
            settings,
            self.model,
        )
        return True

    def finish_calibration(
        self, view: ModeView, policy: Policy, snapshot: Callable[[], ModeView | None]
    ) -> None:
        calibrator = self.calibrator
        if calibrator is None or calibrator.run is None:
            return
        target = calibrator.run.target
        if policy.calibration is None or not target.matches(
            view.revision, view.active, view.routes
        ):
            calibrator.cancel()
            return
        result = calibrator.poll(policy.calibration)
        if result is None:
            return
        capacity = Capacity.read(
            {
                "versions": sorted(target.versions),
                "devices": target.devices,
                "slo_profile": result,
            }
        )
        layout = self.inspect_layout(target.versions, target.routes, view.active)
        if layout != (target.devices, target.manifest):
            return
        current = snapshot()
        if (
            current is None
            or not target.matches(current.revision, current.active, current.routes)
            or Policy.read(self.path, self.model) != policy
            or capacity.valid_until_ms <= time.time_ns() // 1_000_000
        ):
            return
        if self.profile_store is not None:
            if target.manifest is None:
                raise ControlError(
                    "calibration state requires committed group manifests"
                )
            persisted = self.profile_store.save(
                {
                    "context": self.state_context(policy),
                    "versions": sorted(target.versions),
                    "devices": target.devices,
                    "slo_profile": result,
                    "binding": route_fingerprint(target.routes),
                    "manifest": target.manifest,
                },
                time.time_ns() // 1_000_000,
            )
            capacity = Capacity.read(persisted)
        self.measured[target.versions] = (target, capacity)
        print(
            f"Plait measured candidate capacity {capacity.rps:.2f} requests/s",
            flush=True,
        )

    def plan_request(self, view: ModeView, target: Provision) -> dict:
        incumbent = active_epochs(view.active)
        targets = [member.group for member in target.members]
        guarded = sorted(set(view.groups).union(targets))
        return {
            "request": "plan_set",
            "guards": [
                [list(bytes.fromhex(group)), incumbent.get(group)] for group in guarded
            ],
            "groups": [member.request() for member in target.members],
        }

    def prepare(self, view: ModeView, target: Provision) -> str:
        request = self.plan_request(view, target)
        request["prepare"] = self.control.next_operation_id()
        try:
            response = self.control.request(request)
        except ControlError:
            response = self.control.request(request)
        decision = response["decision"]
        if decision in ("pending", "no_placement"):
            return decision
        result = response["result"]
        if "Ok" not in result:
            raise ControlError(f"Plait layout preparation rejected: {result['Err']}")
        versions = result["Ok"]["PreparedSet"]
        if [bytes(version["group"]).hex() for version in versions] != [
            member.group for member in target.members
        ] or any(
            type(version["epoch"]) is not int or version["epoch"] < 1
            for version in versions
        ):
            raise ControlError("Plait returned a different candidate set")
        return decision

    def ready(
        self,
        mode: Capacity,
        routes: tuple[tuple[Endpoint, ...], ...],
        active: frozenset[str],
        *,
        verify_http: bool = True,
    ) -> bool:
        layout = self.inspect_layout(
            mode.versions, routes, active, verify_http=verify_http
        )
        if layout is None or layout[0] != mode.devices:
            return False
        measured = self.measured.get(mode.versions)
        return (
            measured is None
            or measured[1] != mode
            or measured[0].manifest is None
            or measured[0].manifest == layout[1]
        )

    def inspect_layout(
        self,
        versions: frozenset[str],
        routes: tuple[tuple[Endpoint, ...], ...],
        active: frozenset[str],
        *,
        verify_http: bool = True,
    ) -> tuple[int, str | None] | None:
        endpoints = endpoints_for(versions, routes)
        if not endpoints:
            return None
        devices = 0
        manifests = []
        for endpoint in endpoints:
            group, epoch = endpoint.version.split("@", 1)
            version = {"group": list(bytes.fromhex(group)), "epoch": int(epoch)}
            view = self.control.request({"request": "view", "version": version})[
                "group"
            ]
            phase = "Active" if endpoint.version in active else "Ready"
            if not view or view["version"] != version or view["phase"] != phase:
                return None
            if verify_http and not endpoint_ready(endpoint, 1.0):
                return None
            devices += len(view["grants"])
            manifests.append(view.get("manifest"))
        identity = fingerprint(manifests) if all(manifests) else None
        return devices, identity

    def switch(
        self, groups: tuple[str, ...], active: frozenset[str], target: frozenset[str]
    ) -> None:
        expected = active_epochs(active)
        replacement = active_epochs(target)
        request = {
            "request": "apply",
            "operation": {
                "id": self.control.next_operation_id(),
                "action": {
                    "SwitchActiveSet": {
                        "transitions": [
                            {
                                "group": list(bytes.fromhex(group)),
                                "expected": expected.get(group),
                                "target": replacement.get(group),
                            }
                            for group in groups
                        ]
                    }
                },
            },
        }
        try:
            result = self.control.request(request)["result"]
        except ControlError:
            result = self.control.request(request)["result"]
        if "Ok" not in result:
            raise ControlError(f"Plait mode switch rejected: {result['Err']}")
        if result["Ok"]["ActiveSet"] != [
            [list(bytes.fromhex(group)), replacement.get(group)] for group in groups
        ]:
            raise ControlError("Plait returned a different active set after switching")


def active_epochs(versions: frozenset[str]) -> dict[str, int]:
    return {
        group: int(epoch)
        for version in versions
        for group, epoch in (version.split("@", 1),)
    }


def route_fingerprint(routes: tuple[tuple[Endpoint, ...], ...]) -> str:
    return fingerprint([[asdict(endpoint) for endpoint in path] for path in routes])
