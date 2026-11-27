# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare and select native serving layouts from workload and capacity."""

import json
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
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

    @classmethod
    def read(cls, profile: dict, parent: Path, model: str | None) -> "Provision":
        members = tuple(
            PlanMember.read(item, parent, model) for item in profile["groups"]
        )
        if not members or len({member.group for member in members}) != len(members):
            raise ValueError("provision needs unique execution groups")
        return cls(members, *profile_limits(profile, len(members)))


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
        if (
            type(revision) is not int
            or revision < 1
            or type(stable) is not int
            or stable < 1
            or not modes
            or len({mode.versions for mode in modes}) != len(modes)
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
            Settings.read(config["calibration"]) if "calibration" in config else None,
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
    ):
        self.path = policy
        self.load = load
        self.control = control
        self.model = model
        self.calibrator = calibrator
        self.measured: dict[frozenset[str], tuple[Target, Capacity]] = {}
        self.previous: dict | None = None
        self.last_policy: Policy | None = None
        self.proposal: tuple[Capacity, Capacity | Provision] | None = None
        self.windows = 0
        self.cooldown_until = 0.0

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
                or policy.route_revision != self.last_policy.route_revision
                or policy.calibration != self.last_policy.calibration
            ):
                self.measured.clear()
            if self.calibrator is not None:
                self.calibrator.cancel()
            self.last_policy = policy
            self.reset_evidence()
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
        candidates = sorted(
            (
                (gain(policy, incumbent, mode, demand), mode)
                for mode in eligible
                if mode != incumbent
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
            or snapshot() != view
            or Policy.read(self.path, self.model) != policy
            or min(incumbent.valid_until_ms, target.valid_until_ms)
            <= time.time_ns() // 1_000_000
        ):
            self.proposal = None
            self.windows = 0
            return
        if isinstance(target, Provision):
            if not self.calibrate(view, target, policy.calibration) and self.prepare(
                view, target
            ):
                print(
                    f"Plait prepared a candidate layout at {demand:.2f} requests/s",
                    flush=True,
                )
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

    def calibrate(
        self, view: ModeView, projection: Provision, settings: Settings | None
    ) -> bool:
        if settings is None or self.calibrator is None or self.model is None:
            return False
        routes = view.projection_routes(projection)
        if not routes:
            return False
        versions = frozenset(endpoint.version for path in routes for endpoint in path)
        devices = self.device_count(versions, routes, view.active)
        if devices is not None:
            self.calibrator.start(
                Target(view.revision, view.active, routes, devices),
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
        if (
            self.device_count(target.versions, target.routes, view.active)
            != target.devices
        ):
            return
        current = snapshot()
        if (
            current is None
            or not target.matches(current.revision, current.active, current.routes)
            or Policy.read(self.path, self.model) != policy
            or capacity.valid_until_ms <= time.time_ns() // 1_000_000
        ):
            return
        self.measured[target.versions] = (target, capacity)
        print(
            f"Plait measured candidate capacity {capacity.rps:.2f} requests/s",
            flush=True,
        )

    def prepare(self, view: ModeView, target: Provision) -> bool:
        incumbent = active_epochs(view.active)
        targets = [member.group for member in target.members]
        guarded = sorted(set(view.groups).union(targets))
        request = {
            "request": "plan_set",
            "prepare": self.control.next_operation_id(),
            "guards": [
                [list(bytes.fromhex(group)), incumbent.get(group)] for group in guarded
            ],
            "groups": [member.request() for member in target.members],
        }
        try:
            response = self.control.request(request)
        except ControlError:
            response = self.control.request(request)
        if response["decision"] in ("pending", "no_placement"):
            return False
        result = response["result"]
        if "Ok" not in result:
            raise ControlError(f"Plait layout preparation rejected: {result['Err']}")
        versions = result["Ok"]["PreparedSet"]
        if [bytes(version["group"]).hex() for version in versions] != targets or any(
            type(version["epoch"]) is not int or version["epoch"] < 1
            for version in versions
        ):
            raise ControlError("Plait returned a different candidate set")
        return True

    def ready(
        self,
        mode: Capacity,
        routes: tuple[tuple[Endpoint, ...], ...],
        active: frozenset[str],
        *,
        verify_http: bool = True,
    ) -> bool:
        return (
            self.device_count(mode.versions, routes, active, verify_http=verify_http)
            == mode.devices
        )

    def device_count(
        self,
        versions: frozenset[str],
        routes: tuple[tuple[Endpoint, ...], ...],
        active: frozenset[str],
        *,
        verify_http: bool = True,
    ) -> int | None:
        endpoints = endpoints_for(versions, routes)
        if not endpoints:
            return None
        devices = 0
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
        return devices

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
