# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from datetime import timedelta
from typing import TYPE_CHECKING

from torch.distributed import DistStoreError, PrefixStore, TCPStore
from torch.distributed.constants import default_pg_timeout

from vllm import envs
from vllm.distributed.utils import create_tcp_store

if TYPE_CHECKING:
    from vllm.config import ParallelConfig


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


def agree_on_plait_layout(config: "ParallelConfig") -> TCPStore:
    """Rendezvous once per host before assigning global worker ranks."""
    assert config.plait_tp_layout is not None
    launch_store = _open_store(config, is_master=config.node_rank == 0)
    store = PrefixStore("plait-tp-layout", launch_store)
    keys = [f"node/{node}" for node in range(config.nnodes)]
    key = keys[config.node_rank]
    if store.add(f"{key}/claims", 1) != 1:
        raise ValueError(f"Duplicate node rank {config.node_rank} in Plait TP layout")

    layout = ",".join(str(count) for count in config.plait_tp_layout).encode()
    if config.pipeline_parallel_size > 1:
        layout = json.dumps(
            {
                "workers": config.plait_tp_layout,
                "tp": config.tensor_parallel_size,
                "pp": config.pipeline_parallel_size,
                "layers": envs.VLLM_PP_LAYER_PARTITION,
            }
        ).encode()
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
            "Inconsistent --plait-tp-layout, TP/PP sizes or layer partition: "
            + ", ".join(plan.decode() for plan in observed)
        )
    return launch_store


def plait_worker_store(config: "ParallelConfig") -> PrefixStore:
    return PrefixStore("plait-tp-workers", _open_store(config, is_master=False))
