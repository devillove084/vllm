# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from dataclasses import dataclass
from datetime import timedelta
from typing import TYPE_CHECKING

from torch.distributed import DistStoreError, PrefixStore, TCPStore
from torch.distributed.constants import default_pg_timeout

from vllm import envs
from vllm.distributed.plait_calibration import GemmCalibration
from vllm.distributed.utils import create_tcp_store

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


@dataclass(frozen=True)
class CommittedTpGroup:
    version: str
    seeds: list[str]
    devices: list[tuple[str, int]]
    compute_calibration: GemmCalibration | None = None

    @classmethod
    def load(cls, path: str, ranks: list[int], slot: int) -> "CommittedTpGroup":
        """Resolve this worker's TP group against the launcher's device bindings."""
        with open(path, encoding="utf-8") as file:
            config = json.load(file)
        matches = [group for group in config["tp_groups"] if group["ranks"] == ranks]
        if len(matches) != 1:
            raise ValueError(f"Expected one committed Plait TP group for ranks {ranks}")
        group = matches[0]
        devices = [(host.lower(), ordinal) for host, ordinal in group["devices"]]
        if len(devices) != len(ranks) or devices[slot][0] != config["host_id"].lower():
            raise ValueError(
                f"Plait TP device binding does not match worker {ranks[slot]}"
            )
        calibration = group.get("compute_calibration")
        if calibration is not None:
            calibration = GemmCalibration(**calibration)
        return cls(group["version"], config["seeds"], devices, calibration)


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
            "or KV capabilities: "
            + ", ".join(plan.decode() for plan in observed)
        )
    return launch_store


def plait_worker_store(config: "ParallelConfig") -> PrefixStore:
    return PrefixStore("plait-tp-workers", _open_store(config, is_master=False))
