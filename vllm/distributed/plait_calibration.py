# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import time
from dataclasses import dataclass

import torch

_DTYPES = {"float16": torch.float16, "bfloat16": torch.bfloat16}
_WARMUP = 3
_SAMPLES = 7
_REPETITIONS = 8


@dataclass(frozen=True)
class GemmCalibration:
    """Local GEMM shape and arithmetic contract for placement measurements."""

    m: int
    n: int
    k: int
    dtype: str

    def __post_init__(self) -> None:
        if any(type(dim) is not int or dim <= 0 for dim in (self.m, self.n, self.k)):
            raise ValueError(
                "Plait GEMM calibration requires positive integer dimensions"
            )
        if self.dtype not in _DTYPES:
            raise ValueError("Plait GEMM calibration requires float16 or bfloat16")

    @property
    def id(self) -> str:
        spec = f"plait-torch-mm-v1:{self.dtype}:{self.m}:{self.n}:{self.k}"
        return hashlib.sha256(spec.encode()).hexdigest()[:32]

    @torch.inference_mode()
    def measure(self, device: torch.device) -> int:
        """Return effective operations per second for warm, device-local GEMM."""
        dtype = _DTYPES[self.dtype]
        backend = torch.get_device_module(device)
        with torch.accelerator.device_index(device.index):
            left = torch.full((self.m, self.k), 0.125, dtype=dtype, device=device)
            right = torch.full((self.k, self.n), 0.0625, dtype=dtype, device=device)
            out = torch.empty((self.m, self.n), dtype=dtype, device=device)
            for _ in range(_WARMUP):
                torch.mm(left, right, out=out)
            backend.synchronize(device)
            elapsed = []
            for _ in range(_SAMPLES):
                started = time.perf_counter_ns()
                for _ in range(_REPETITIONS):
                    torch.mm(left, right, out=out)
                backend.synchronize(device)
                elapsed.append(time.perf_counter_ns() - started)
        median = sorted(elapsed)[_SAMPLES // 2]
        work = 2 * self.m * self.n * self.k * _REPETITIONS
        return max(1, work * 10**9 // max(1, median))
