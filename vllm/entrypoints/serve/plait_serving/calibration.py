# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run a bounded serving benchmark against a private candidate ingress."""

import contextlib
import json
import math
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .group import Endpoint


@dataclass(frozen=True)
class Settings:
    model: str
    input_tokens: int
    output_tokens: int
    rates: tuple[float, ...]
    seeds: tuple[int, ...]
    ttft_ms: float
    tpot_ms: float
    vllm: str = "vllm"
    prompts: int = 64
    warmups: int = 3
    attainment: float = 0.95
    valid_for: int = 1800
    timeout_seconds: float = 600
    retry_seconds: float = 30

    @classmethod
    def read(cls, raw: dict) -> "Settings":
        settings = cls(
            **{**raw, "rates": tuple(raw["rates"]), "seeds": tuple(raw["seeds"])}
        )
        if (
            any(
                not isinstance(text, str) or not text
                for text in (settings.model, settings.vllm)
            )
            or any(
                type(value) is not int or value < minimum
                for value, minimum in (
                    (settings.input_tokens, 1),
                    (settings.output_tokens, 2),
                    (settings.prompts, 1),
                    (settings.warmups, 0),
                    (settings.valid_for, 1),
                )
            )
            or not settings.rates
            or not settings.seeds
            or any(type(seed) is not int or seed < 0 for seed in settings.seeds)
            or len(set(settings.seeds)) != len(settings.seeds)
            or any(
                type(value) not in (float, int)
                or not math.isfinite(value)
                or value <= 0
                for value in (
                    *settings.rates,
                    settings.ttft_ms,
                    settings.tpot_ms,
                    settings.timeout_seconds,
                    settings.retry_seconds,
                    settings.attainment,
                )
            )
            or len(set(settings.rates)) != len(settings.rates)
            or settings.attainment > 1
        ):
            raise ValueError("invalid Plait calibration settings")
        return settings

    def command(
        self, url: str, alias: str, target: "Target", output: Path
    ) -> list[str]:
        options = {
            "url": url,
            "model": self.model,
            "served-model": alias,
            "placement": ",".join(sorted(target.versions)),
            "input-tokens": self.input_tokens,
            "output-tokens": self.output_tokens,
            "vllm": self.vllm,
            "prompts": self.prompts,
            "warmups": self.warmups,
            "ttft-ms": self.ttft_ms,
            "tpot-ms": self.tpot_ms,
            "attainment": self.attainment,
            "valid-for": self.valid_for,
            "output": output,
        }
        command = [
            sys.executable,
            "-m",
            "vllm.entrypoints.serve.plait_serving.benchmark",
        ]
        for flag, value in options.items():
            command.extend([f"--{flag}", str(value)])
        return [
            *command,
            "--rates",
            *map(str, self.rates),
            "--seeds",
            *map(str, self.seeds),
        ]


@dataclass(frozen=True)
class Target:
    revision: int
    active: frozenset[str]
    routes: tuple[tuple[Endpoint, ...], ...]
    devices: int
    manifest: str | None = None

    @property
    def versions(self) -> frozenset[str]:
        return frozenset(endpoint.version for path in self.routes for endpoint in path)

    def matches(
        self,
        revision: int,
        active: frozenset[str],
        routes: tuple[tuple[Endpoint, ...], ...],
    ) -> bool:
        return (
            revision == self.revision
            and active == self.active
            and self.versions.isdisjoint(active)
            and all(path in routes for path in self.routes)
        )


@dataclass
class Run:
    target: Target
    process: subprocess.Popen
    cleanup: contextlib.ExitStack
    output: Path
    deadline: float


def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            with contextlib.suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait()


class Calibrator:
    def __init__(
        self, open_ingress: Callable[[Target], contextlib.AbstractContextManager[str]]
    ) -> None:
        self.open_ingress = open_ingress
        self.run: Run | None = None
        self.retry_at = 0.0

    def start(self, target: Target, settings: Settings, alias: str) -> None:
        if self.run is not None or time.monotonic() < self.retry_at:
            return
        with contextlib.ExitStack() as cleanup:
            work = Path(
                cleanup.enter_context(
                    tempfile.TemporaryDirectory(prefix="vllm-plait-calibration-")
                )
            )
            output = work / "capacity.json"
            url = cleanup.enter_context(self.open_ingress(target))
            process = subprocess.Popen(
                settings.command(url, alias, target, output),
                start_new_session=True,
                stdout=subprocess.DEVNULL,
                env={**os.environ, "TMPDIR": str(work)},
            )
            cleanup.callback(stop_process, process)
            self.run = Run(
                target,
                process,
                cleanup.pop_all(),
                output,
                time.monotonic() + settings.timeout_seconds,
            )

    def poll(self, settings: Settings) -> dict | None:
        run = self.run
        if run is None:
            return None
        status = run.process.poll()
        now = time.monotonic()
        if status is None and now < run.deadline:
            return None
        self.retry_at = time.monotonic() + settings.retry_seconds
        try:
            if now >= run.deadline:
                raise ValueError("candidate benchmark timed out")
            if status != 0:
                raise ValueError(f"candidate benchmark exited with status {status}")
            document = json.loads(run.output.read_text(encoding="utf-8"))
            if document["placement"] != ",".join(sorted(run.target.versions)):
                raise ValueError("candidate benchmark returned another layout")
            profile = document["slo_profile"]
            if profile is None:
                raise ValueError("candidate benchmark did not meet its SLO at any rate")
            return profile
        finally:
            self.cancel()

    def cancel(self) -> None:
        run, self.run = self.run, None
        if run is not None:
            run.cleanup.close()
