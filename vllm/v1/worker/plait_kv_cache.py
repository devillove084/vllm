# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import math
from collections.abc import Sequence

import torch
import torch.distributed as dist

from vllm.v1.attention.backend import AttentionBackend, MultipleOf

BlockSizePreferences = tuple[int, list[list[int]]]


def _candidate_block_sizes(preferences: Sequence[BlockSizePreferences]) -> list[int]:
    # Each backend supplies alternative kernel sizes. A cache page can hold
    # several kernel blocks, so try the smallest common multiples first.
    common = {1}
    for _, backends in preferences:
        for sizes in backends:
            if sizes:
                common = {math.lcm(size, base) for size in common for base in sizes}
    if common == {1}:
        common.clear()
    return sorted(common | {preferred for preferred, _ in preferences})


def negotiate_block_size(
    cpu_group: dist.ProcessGroup,
    preferred: int,
    backends: Sequence[type[AttentionBackend]],
) -> int:
    """Agree on the serving group's cache page size before computing cache specs."""
    local_sizes = [
        [
            size.base if isinstance(size, MultipleOf) else size
            for size in backend.get_supported_kernel_block_sizes()
        ]
        for backend in backends
    ]
    preferences: list[BlockSizePreferences] = [
        (0, []) for _ in range(dist.get_world_size(cpu_group))
    ]
    dist.all_gather_object(preferences, (preferred, local_sizes), group=cpu_group)
    if all(size == preferred for size, _ in preferences):
        return preferred

    candidates = _candidate_block_sizes(preferences)
    # Ask each loaded backend to check the candidates. Some override the
    # default divisibility rule to require an exact size.
    accepted = torch.tensor(
        [
            all(backend.supports_block_size(size) for backend in backends)
            for size in candidates
        ],
        dtype=torch.int32,
        device="cpu",
    )
    dist.all_reduce(accepted, op=dist.ReduceOp.MIN, group=cpu_group)
    for size, is_accepted in zip(candidates, accepted.tolist()):
        if is_accepted:
            return size
    raise ValueError(
        "Plait has no common KV cache block size. "
        f"Worker preferences and kernel sizes: {preferences}; "
        f"checked cache sizes: {candidates}"
    )
