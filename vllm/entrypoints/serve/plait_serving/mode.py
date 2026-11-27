# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Select a ready aggregate, P/D, or mixed layout from measured capacity."""

import json
import math
import re
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .group import (
    DIRECTORY_POLL_SECONDS,
    Control,
    ControlError,
    Endpoint,
    endpoint_ready,
)
from .load import Load

VERSION = re.compile(r"[a-f0-9]{32}@[1-9][0-9]*")


@dataclass(frozen=True)
class Capacity:
    versions: frozenset[str]
    devices: int
    rps: float
    valid_until_ms: int

    @classmethod
    def read(cls, profile: dict) -> "Capacity":
        versions = profile["versions"]
        devices = profile["devices"]
        slo = profile["slo_profile"]
        valid_until_ms = slo["valid_until_ms"]
        if (
            not isinstance(versions, list)
            or not versions
            or any(not isinstance(v, str) or not VERSION.fullmatch(v) for v in versions)
            or len({v.split("@", 1)[0] for v in versions}) != len(versions)
            or type(devices) is not int
            or devices < len(versions)
            or type(valid_until_ms) is not int
            or valid_until_ms <= 0
        ):
            raise ValueError("invalid versioned SLO profile")
        return cls(
            frozenset(versions),
            devices,
            number(slo["rps"], positive=True),
            valid_until_ms,
        )


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

    @classmethod
    def read(cls, path: Path) -> "Policy":
        config = json.loads(path.read_text(encoding="utf-8"))
        revision = config["route_revision"]
        stable = config["stable_windows"]
        modes = tuple(Capacity.read(profile) for profile in config["modes"])
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
        )


def gain(policy: Policy, incumbent: Capacity, target: Capacity, demand: float) -> float:
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
    mode: Capacity, routes: tuple[tuple[Endpoint, ...], ...]
) -> list[Endpoint]:
    endpoints = {
        endpoint
        for path in routes
        if all(endpoint.version in mode.versions for endpoint in path)
        for endpoint in path
    }
    if (
        len(endpoints) != len(mode.versions)
        or {endpoint.version for endpoint in endpoints} != mode.versions
    ):
        return []
    return sorted(endpoints, key=lambda endpoint: endpoint.version)


@dataclass(frozen=True)
class ModeView:
    revision: int
    routes: tuple[tuple[Endpoint, ...], ...]
    groups: tuple[str, ...]
    active: frozenset[str]


class ModeController:
    def __init__(self, policy: Path, load: Load, control: Control):
        self.path = policy
        self.load = load
        self.control = control
        self.previous: dict | None = None
        self.last_policy: Policy | None = None
        self.proposal: tuple[Capacity, Capacity] | None = None
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
                    if str(error) != last_error:
                        print(f"Plait mode policy: {error}", flush=True)
                    last_error = str(error)
                stop.wait(DIRECTORY_POLL_SECONDS)
        finally:
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
        policy = Policy.read(self.path)
        if policy != self.last_policy:
            self.last_policy = policy
            self.reset_evidence()
        demand = self.sample(policy.window_seconds)
        if (
            demand is None
            or view.revision != policy.route_revision
            or time.monotonic() < self.cooldown_until
        ):
            return
        now_ms = time.time_ns() // 1_000_000
        eligible = [
            mode
            for mode in policy.modes
            if mode.valid_until_ms > now_ms and endpoints_for(mode, routes)
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
        chosen = next(
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
        proposal = (incumbent, chosen[1]) if chosen is not None else None
        self.windows = self.windows + 1 if proposal == self.proposal else 1
        self.proposal = proposal
        if chosen is None or self.windows < policy.stable_windows:
            return
        score, target = chosen
        if (
            not self.ready(incumbent, routes, active, verify_http=False)
            or snapshot() != view
            or Policy.read(self.path) != policy
            or min(incumbent.valid_until_ms, target.valid_until_ms)
            <= time.time_ns() // 1_000_000
        ):
            self.proposal = None
            self.windows = 0
            return
        self.switch(view.groups, active, target.versions)
        self.cooldown_until = time.monotonic() + policy.cooldown_seconds
        self.proposal = None
        self.windows = 0
        print(
            f"Plait serving mode switched at {demand:.2f} requests/s "
            f"with projected gain {score:.2f} requests",
            flush=True,
        )

    def ready(
        self,
        mode: Capacity,
        routes: tuple[tuple[Endpoint, ...], ...],
        active: frozenset[str],
        *,
        verify_http: bool = True,
    ) -> bool:
        devices = 0
        for endpoint in endpoints_for(mode, routes):
            group, epoch = endpoint.version.split("@", 1)
            version = {"group": list(bytes.fromhex(group)), "epoch": int(epoch)}
            view = self.control.request({"request": "view", "version": version})[
                "group"
            ]
            phase = "Active" if endpoint.version in active else "Ready"
            if not view or view["version"] != version or view["phase"] != phase:
                return False
            if verify_http and not endpoint_ready(endpoint, 1.0):
                return False
            devices += len(view["grants"])
        return devices == mode.devices

    def switch(
        self, groups: tuple[str, ...], active: frozenset[str], target: frozenset[str]
    ) -> None:
        expected = {
            group: int(epoch)
            for version in active
            for group, epoch in (version.split("@", 1),)
        }
        replacement = {
            group: int(epoch)
            for version in target
            for group, epoch in (version.split("@", 1),)
        }
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
