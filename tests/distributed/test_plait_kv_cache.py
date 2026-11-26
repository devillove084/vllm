# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch.distributed as dist
import torch.multiprocessing as mp

from vllm.v1.attention.backend import AttentionBackend, MultipleOf
from vllm.v1.worker.plait_kv_cache import negotiate_block_size


class MultipleOf16(AttentionBackend):
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [MultipleOf(16)]


class Exact128(AttentionBackend):
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [128]

    @classmethod
    def supports_block_size(cls, block_size):
        return block_size == 128


class Exact16(AttentionBackend):
    @staticmethod
    def get_supported_kernel_block_sizes():
        return [16]

    @classmethod
    def supports_block_size(cls, block_size):
        return block_size == 16


def _exercise_negotiation(rank: int, rendezvous: str) -> None:
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        group = dist.group.WORLD
        assert negotiate_block_size(group, 32, [MultipleOf16]) == 32

        preferred, backend = (16, MultipleOf16) if rank == 0 else (128, Exact128)
        assert negotiate_block_size(group, preferred, [backend]) == 128

        backend = Exact16 if rank == 0 else Exact128
        with pytest.raises(ValueError, match="no common KV cache block size"):
            negotiate_block_size(group, preferred, [backend])
    finally:
        dist.destroy_process_group()


def test_plait_kv_block_size(tmp_path):
    mp.spawn(_exercise_negotiation, args=((tmp_path / "store").as_uri(),), nprocs=2)
