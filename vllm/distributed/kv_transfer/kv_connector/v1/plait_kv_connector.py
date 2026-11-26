# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse full-attention KV blocks across Plait tensor-parallel placements."""

from __future__ import annotations

import hashlib
import json
import socket
import struct
import uuid
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import torch

from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
)
from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
from vllm.logger import init_logger
from vllm.utils.gpu_sync_debug import gpu_sync_allowed
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.forward_context import ForwardContext
    from vllm.v1.attention.backend import AttentionMetadata
    from vllm.v1.core.kv_cache_manager import KVCacheBlocks
    from vllm.v1.core.sched.output import SchedulerOutput
    from vllm.v1.kv_cache_interface import KVCacheConfig
    from vllm.v1.request import Request

logger = init_logger(__name__)
_HEADER = struct.Struct("<8sc32s16sIIII")
_OK = b"\x01"
_MISS = b"\x00"
_MAX_BLOCK_BYTES = 64 << 20


@dataclass(frozen=True)
class Block:
    key: bytes
    block_id: int


@dataclass
class PlaitKvMetadata(KVConnectorMetadata):
    loads: list[Block] = field(default_factory=list)
    saves: list[tuple[bytes, Block]] = field(default_factory=list)


class Store:
    def __init__(self, address: str, heads: int, head_bytes: int, timeout: float):
        origin = urlsplit("tcp://" + address)
        if (
            not origin.hostname
            or not origin.port
            or origin.path
            or origin.username is not None
            or origin.query
            or origin.fragment
        ):
            raise ValueError("Plait KV address must specify a host and port")
        self.address = (origin.hostname, origin.port)
        self.heads = heads
        self.head_bytes = head_bytes
        self.timeout = timeout

    def exchange(
        self,
        verb: bytes,
        key: bytes,
        first: int = 0,
        count: int = 0,
        writer: bytes = bytes(16),
        data: bytes = b"",
    ) -> bytes | None:
        header = _HEADER.pack(
            b"PLAITKV2", verb, key, writer, self.heads, first, count, self.head_bytes
        )
        with socket.create_connection(self.address, self.timeout) as connection:
            connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            connection.sendall(header)
            if data:
                connection.sendall(data)
            with connection.makefile("rb") as reply:
                state = reply.read(1)
                if state in (_MISS, b"\x02"):
                    return None
                if state != _OK:
                    raise OSError("invalid Plait KV store response")
                expected = count * self.head_bytes if verb == b"G" else 0
                payload = reply.read(expected)
                if len(payload) != expected:
                    raise OSError("incomplete Plait KV block")
                return payload


class PlaitKvConnector(KVConnectorBase_V1):
    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig,
    ):
        super().__init__(vllm_config, role, kv_cache_config)
        extra = self._kv_transfer_config.kv_connector_extra_config
        namespace = extra.get("namespace")
        address = extra.get("address")
        if (
            not isinstance(namespace, str)
            or not namespace
            or not isinstance(address, str)
        ):
            raise ValueError("PlaitKvConnector requires an address and model namespace")
        groups = kv_cache_config.kv_cache_groups
        parallel = vllm_config.parallel_config
        if (
            len(groups) != 1
            or parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
            or vllm_config.speculative_config is not None
            or not vllm_config.model_config.enforce_eager
        ):
            raise ValueError(
                "Plait KV requires one eager attention group without context sharding"
            )
        spec = groups[0].kv_cache_spec
        total_heads = vllm_config.model_config.get_total_num_kv_heads()
        if (
            type(spec) is not FullAttentionSpec
            or spec.sliding_window is not None
            or spec.attention_chunk_size is not None
            or spec.non_causal
            or spec.kv_quant_mode != KVQuantMode.NONE
            or spec.head_size != spec.head_size_v
            or spec.num_kv_heads * parallel.tensor_parallel_size != total_heads
            or spec.dtype not in (torch.float16, torch.bfloat16)
        ):
            raise ValueError(
                "Plait KV requires unreplicated, dense FP16 or BF16 attention"
            )
        self.layers = sorted(groups[0].layer_names)
        self.block_size = spec.block_size
        self.heads = spec.num_kv_heads
        self.content = spec.head_size + spec.head_size_v
        self.dtype = spec.dtype
        self.first_head = (
            get_tensor_model_parallel_rank() * self.heads
            if role == KVConnectorRole.WORKER
            else 0
        )
        head_bytes = len(self.layers) * self.block_size * self.content * 2
        if total_heads * head_bytes > _MAX_BLOCK_BYTES:
            raise ValueError("Plait KV block exceeds the store limit")
        timeout = float(extra.get("timeout", 3.0))
        if not 0 < timeout <= 15:
            raise ValueError("invalid Plait KV store timeout")
        self.store = Store(address, total_heads, head_bytes, timeout)
        self.domain = json.dumps(
            [
                "plait-kv2",
                namespace,
                self.layers,
                str(self.dtype),
                self.block_size,
                total_heads,
                self.content,
            ]
        ).encode()
        self.session = uuid.uuid4().bytes
        self.requests: dict[str, Request] = {}
        self.found: dict[str, tuple[int, list[bytes]]] = {}
        self.loads: list[Block] = []
        self.invalid_blocks: set[int] = set()
        self.caches: list[tuple[torch.Tensor, torch.Tensor]] = []

    def _eligible(self, request: Request) -> bool:
        return (
            request.prompt_token_ids is not None
            and not request.mm_features
            and request.prompt_embeds is None
            and request.lora_request is None
        )

    def _keys(self, request: Request, count: int) -> list[bytes]:
        digest = hashlib.sha256(self.domain)
        salt = (request.cache_salt or "").encode()
        digest.update(struct.pack("<Q", len(salt)))
        digest.update(salt)
        tokens = request.all_token_ids
        keys = []
        for index in range(count):
            start = index * self.block_size
            block = tokens[start : start + self.block_size]
            digest.update(struct.pack(f"<{len(block)}I", *block))
            keys.append(digest.digest())
        return keys

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int, bool]:
        self.found.pop(request.request_id, None)
        if not self._kv_transfer_config.is_kv_consumer or not self._eligible(request):
            return 0, False
        first = num_computed_tokens // self.block_size
        # Keep one input token for the forward that produces the next logits.
        count = (len(request.all_token_ids) - 1) // self.block_size
        hits = []
        try:
            for key in self._keys(request, count)[first:]:
                if self.store.exchange(b"H", key) is None:
                    break
                hits.append(key)
        except OSError:
            logger.warning_once("Plait KV store unavailable during prefix lookup")
        if hits:
            self.found[request.request_id] = (first, hits)
        return len(hits) * self.block_size, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        if self._kv_transfer_config.is_kv_producer and self._eligible(request):
            self.requests[request.request_id] = request
        match = self.found.pop(request.request_id, None)
        if num_external_tokens:
            assert match is not None
            first, keys = match
            count = num_external_tokens // self.block_size
            (block_ids,) = blocks.get_block_ids()
            self.loads.extend(
                Block(key, block_id)
                for key, block_id in zip(
                    keys[:count], block_ids[first : first + count], strict=True
                )
            )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> PlaitKvMetadata:
        metadata = PlaitKvMetadata(self.loads)
        self.loads = []
        block_state = scheduler_output.kv_connector_block_state
        assert block_state is not None
        for req_id, scheduled in scheduler_output.num_scheduled_tokens.items():
            request = self.requests.get(req_id)
            if request is None:
                continue
            before = request.num_computed_tokens
            first = before // self.block_size
            count = (
                min(before + scheduled, len(request.all_token_ids)) // self.block_size
            )
            if count <= first:
                continue
            block_ids = block_state.get_block_ids(req_id)
            assert block_ids is not None
            writer = hashlib.sha256(self.session + req_id.encode()).digest()[:16]
            metadata.saves.extend(
                (writer, Block(key, block_id))
                for key, block_id in zip(
                    self._keys(request, count)[first:],
                    block_ids[0][first:count],
                    strict=True,
                )
            )
        return metadata

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self.requests.pop(request.request_id, None)
        self.found.pop(request.request_id, None)
        return False, None

    def register_kv_caches(
        self,
        kv_caches: dict[str, torch.Tensor | tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        head_size = self.content // 2
        caches = []
        for layer in self.layers:
            cache = kv_caches[layer]
            if isinstance(cache, (tuple, list)) and len(cache) == 2:
                # Ascend stores separate K and V pages in block/token/head order.
                key, value = cache
                if not all(
                    isinstance(tensor, torch.Tensor) and tensor.ndim == 4
                    for tensor in (key, value)
                ):
                    raise ValueError(f"unsupported Plait KV layout for {layer}")
                key, value = key.transpose(1, 2), value.transpose(1, 2)
            elif (
                isinstance(cache, torch.Tensor)
                and cache.ndim == 4
                and cache.shape[3] == self.content
            ):
                # Triton stores packed K/V pages in block/head/token order.
                key, value = cache.split(head_size, dim=-1)
            else:
                raise ValueError(f"unsupported Plait KV layout for {layer}")
            if (
                key.shape != value.shape
                or key.shape[1] != self.heads
                or not key.shape[2]
                or self.block_size % key.shape[2] != 0
                or key.shape[3] != head_size
                or key.dtype != self.dtype
                or value.dtype != self.dtype
            ):
                raise ValueError(
                    f"unsupported Plait KV layout for {layer}: "
                    f"{key.shape}, {value.shape}"
                )
            caches.append((key, value))
        self.caches = caches

    def _cache_page(self, cache: torch.Tensor, block_id: int) -> torch.Tensor:
        chunks = self.block_size // cache.shape[2]
        return cache[block_id * chunks : (block_id + 1) * chunks]

    def start_load_kv(self, forward_context: ForwardContext, **kwargs: Any) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, PlaitKvMetadata)
        for block in metadata.loads:
            try:
                data = self.store.exchange(b"G", block.key, self.first_head, self.heads)
                if data is None:
                    raise OSError("Plait KV block is missing")
                with gpu_sync_allowed():
                    value = torch.frombuffer(bytearray(data), dtype=self.dtype).reshape(
                        self.heads, len(self.layers), self.block_size, self.content
                    )
                    for index, (key_cache, value_cache) in enumerate(self.caches):
                        key_page = self._cache_page(key_cache, block.block_id)
                        value_page = self._cache_page(value_cache, block.block_id)
                        chunks, _, states, head_size = key_page.shape
                        source = (
                            value[:, index]
                            .reshape(self.heads, chunks, states, self.content)
                            .transpose(0, 1)
                        )
                        key, value_part = source.split(head_size, dim=-1)
                        key_page.copy_(key)
                        value_page.copy_(value_part)
            except (OSError, RuntimeError, ValueError):
                logger.warning("Plait KV load failed for block %d", block.block_id)
                self.invalid_blocks.add(block.block_id)

    def wait_for_layer_load(self, layer_name: str) -> None:
        pass

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: AttentionMetadata,
        **kwargs: Any,
    ) -> None:
        pass

    def wait_for_save(self) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, PlaitKvMetadata)
        if self.invalid_blocks:
            return
        for writer, block in metadata.saves:
            try:
                with gpu_sync_allowed():
                    layers = []
                    for key_cache, value_cache in self.caches:
                        key = self._cache_page(key_cache, block.block_id).cpu()
                        value = self._cache_page(value_cache, block.block_id).cpu()
                        combined = torch.cat((key, value), dim=-1).transpose(0, 1)
                        layers.append(
                            combined.reshape(self.heads, self.block_size, self.content)
                        )
                    value = torch.stack(layers, dim=1)
                    data = value.contiguous().view(torch.uint8).numpy().tobytes()
                self.store.exchange(
                    b"P", block.key, self.first_head, self.heads, writer, data
                )
            except OSError:
                logger.warning_once("Plait KV store unavailable during publication")
                break

    def get_block_ids_with_load_errors(self) -> set[int]:
        failures, self.invalid_blocks = self.invalid_blocks, set()
        return failures
