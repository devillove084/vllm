# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass, field, replace
from datetime import timedelta
from typing import TYPE_CHECKING

from torch.distributed import DistStoreError, PrefixStore, TCPStore
from torch.distributed.constants import default_pg_timeout

from vllm import envs
from vllm.distributed.plait_calibration import GemmCalibration
from vllm.distributed.utils import create_tcp_store

if TYPE_CHECKING:
    import torch
    from plait_nccl import CommHandle
    from plait_nccl import NCCLLibrary as PlaitLibrary

    from vllm.config import ParallelConfig


@dataclass(frozen=True)
class CommittedGroup:
    version: str
    seeds: list[str]
    devices: list[tuple[str, int]]
    compute_calibration: GemmCalibration | None = None
    replicated_egress_bytes: int | None = None

    @classmethod
    def load(
        cls, path: str, ranks: list[int], slot: int, axis: str = "tp"
    ) -> "CommittedGroup | None":
        """Resolve a worker's model-parallel group against its device binding."""
        with open(path, encoding="utf-8") as file:
            config = json.load(file)
        key = f"{axis}_groups"
        if key not in config and ("tp_groups" in config or "pp_groups" in config):
            return None
        matches = [group for group in config.get(key, []) if group["ranks"] == ranks]
        if len(matches) != 1:
            raise ValueError(
                f"Expected one committed Plait {axis.upper()} group for ranks {ranks}"
            )
        group = matches[0]
        binding = cls.from_descriptor(config, group)
        if (
            len(binding.devices) != len(ranks)
            or binding.devices[slot][0] != config["host_id"].lower()
        ):
            raise ValueError(
                f"Plait {axis.upper()} binding does not match worker {ranks[slot]}"
            )
        return binding

    @classmethod
    def from_descriptor(cls, config: dict, group: dict) -> "CommittedGroup":
        devices = [(host.lower(), ordinal) for host, ordinal in group["devices"]]
        calibration = group.get("compute_calibration")
        if calibration is not None:
            calibration = GemmCalibration(**calibration)
        egress = group.get("replicated_egress_bytes")
        if egress is not None and (type(egress) is not int or not 0 <= egress < 2**64):
            raise ValueError(
                "Plait replicated_egress_bytes must be an unsigned integer"
            )
        return cls(group["version"], config["seeds"], devices, calibration, egress)


@dataclass
class CommittedModelGroup:
    """Hold both native axes until vLLM takes ownership of each handle."""

    binding: CommittedGroup
    rank: int
    tensor_parallel: int
    pipeline_parallel: int
    _axes: dict[str, "tuple[PlaitLibrary, CommHandle]"] | None = field(
        init=False, default=None
    )

    @classmethod
    def load(cls, path: str, rank: int, tp: int, pp: int) -> "CommittedModelGroup":
        with open(path, encoding="utf-8") as file:
            config = json.load(file)
        model = config.get("model_group")
        if (
            not isinstance(model, dict)
            or "tp_groups" in config
            or "pp_groups" in config
            or tp < 2
            or pp < 2
            or model.get("tensor_parallel") != tp
            or model.get("pipeline_parallel") != pp
        ):
            raise ValueError("committed Plait model group differs from vLLM TP × PP")
        layout = list(range(tp * pp))
        binding = CommittedGroup.from_descriptor(config, model)
        if (
            model.get("ranks") != layout
            or not 0 <= rank < len(layout)
            or len(set(binding.devices)) != len(layout)
            or len(binding.devices) != len(layout)
            or binding.devices[rank][0] != config["host_id"].lower()
        ):
            raise ValueError("committed Plait device layout differs from vLLM workers")
        return cls(binding, rank, tp, pp)

    def take_axis(
        self, axis: str, ranks: list[int], device: "torch.device"
    ) -> "tuple[CommittedGroup, tuple[PlaitLibrary, CommHandle]]":
        from vllm.distributed.device_communicators.pynccl import (
            join_committed_model_group,
        )

        stage, slot = divmod(self.rank, self.tensor_parallel)
        if axis == "tp":
            begin = stage * self.tensor_parallel
            expected = range(begin, begin + self.tensor_parallel)
            binding = self.binding
        elif axis == "pp":
            width = self.tensor_parallel * self.pipeline_parallel
            expected = range(slot, width, self.tensor_parallel)
            binding = replace(self.binding, compute_calibration=None)
        else:
            raise ValueError("expected a TP or PP communicator")
        if ranks != list(expected):
            raise ValueError(f"committed Plait {axis.upper()} slots differ from vLLM")
        if self._axes is None:
            library, tensor, pipeline = join_committed_model_group(
                device, self.binding, self.rank, self.tensor_parallel
            )
            self._axes = {"tp": (library, tensor), "pp": (library, pipeline)}
        return binding, self._axes.pop(axis)


def _open_store(config: "ParallelConfig", *, is_master: bool) -> TCPStore:
    timeout_seconds = config.cpu_distributed_timeout_seconds
    if timeout_seconds is None:
        timeout_seconds = config.distributed_timeout_seconds
    return create_tcp_store(
        config.master_addr,
        config.master_port,
        is_master=is_master,
        wait_for_workers=False,
        timeout=(
            default_pg_timeout
            if timeout_seconds is None
            else timedelta(seconds=timeout_seconds)
        ),
        use_libuv=False,
    )


def agree_on_plait_layout(
    config: "ParallelConfig",
    *,
    seeded_resume: bool = False,
    checkpoint_admission: bool = False,
    checkpoint_peer_budget: bool = False,
) -> TCPStore:
    """Rendezvous once per host before assigning global worker ranks."""
    assert config.plait_tp_layout is not None
    launch_store = _open_store(config, is_master=config.node_rank == 0)
    store = PrefixStore("plait-tp-layout", launch_store)
    keys = [f"node/{node}" for node in range(config.nnodes)]
    key = keys[config.node_rank]
    if store.add(f"{key}/claims", 1) != 1:
        raise ValueError(f"Duplicate node rank {config.node_rank} in Plait TP layout")

    layout = ",".join(str(count) for count in config.plait_tp_layout).encode()
    if (
        config.pipeline_parallel_size > 1
        or seeded_resume
        or checkpoint_admission
        or checkpoint_peer_budget
    ):
        plan = {
            "workers": config.plait_tp_layout,
            "tp": config.tensor_parallel_size,
            "pp": config.pipeline_parallel_size,
            "layers": envs.VLLM_PP_LAYER_PARTITION,
        }
        if seeded_resume:
            plan["seeded_resume"] = True
        if checkpoint_admission:
            plan["checkpoint_admission"] = True
        if checkpoint_peer_budget:
            plan["checkpoint_peer_budget"] = True
        if config.kv_cache_memory_bytes_by_stage is not None:
            plan["kv_cache_memory_bytes_by_stage"] = (
                config.kv_cache_memory_bytes_by_stage
            )
        layout = json.dumps(plan).encode()
    store.set(key, layout)
    try:
        store.wait(keys)
    except DistStoreError as exc:
        raise RuntimeError(
            f"Timed out waiting for all {config.nnodes} hosts to register "
            "their --plait-tp-layout"
        ) from exc
    observed = [store.get(key) for key in keys]
    if any(other != layout for other in observed):
        raise ValueError(
            "Inconsistent --plait-tp-layout, TP/PP sizes, layer partition "
            "or KV configuration: " + ", ".join(plan.decode() for plan in observed)
        )
    return launch_store


def plait_worker_store(config: "ParallelConfig") -> PrefixStore:
    return PrefixStore("plait-tp-workers", _open_store(config, is_master=False))
