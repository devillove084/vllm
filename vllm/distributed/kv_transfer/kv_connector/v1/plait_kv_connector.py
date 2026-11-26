# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reuse and hand off full-attention KV across Plait TP and PP placements."""

from __future__ import annotations

import hashlib
import json
import re
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
from vllm.v1.request import RequestStatus

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
_RANGE = struct.Struct("<II")
_OK = b"\x01"
_MISS = b"\x00"
_MAX_BLOCK_BYTES = 64 << 20
_MAX_PARTS = 65536
_MAX_RANGES = 4096


@dataclass(frozen=True)
class Block:
    key: bytes
    block_id: int


@dataclass(frozen=True)
class LayerRanges:
    part_bytes: int
    ranges: tuple[tuple[int, int], ...]

    @classmethod
    def for_worker(
        cls,
        global_layers: tuple[str, ...],
        local_layers: tuple[str, ...],
        total_heads: int,
        first_head: int,
        heads: int,
        part_bytes: int,
    ) -> LayerRanges:
        indices = {name: index for index, name in enumerate(global_layers)}
        layers = [indices[name] for name in local_layers]
        ranges: list[tuple[int, int]] = []
        for head in range(first_head, first_head + heads):
            for layer in layers:
                offset = head * len(global_layers) + layer
                if ranges:
                    first, count = ranges[-1]
                    if first + count == offset:
                        ranges[-1] = (first, count + 1)
                        continue
                ranges.append((offset, 1))
        if total_heads * len(global_layers) > _MAX_PARTS or len(ranges) > _MAX_RANGES:
            raise ValueError("Plait KV layout exceeds the store range limits")
        return cls(part_bytes, tuple(ranges))


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
        *,
        parts: LayerRanges | None = None,
    ) -> bytes | None:
        if parts is None:
            header = _HEADER.pack(
                b"PLAITKV2",
                verb,
                key,
                writer,
                self.heads,
                first,
                count,
                self.head_bytes,
            )
            size = count * self.head_bytes
        else:
            header = _HEADER.pack(
                b"PLAITKV3",
                verb,
                key,
                writer,
                self.heads,
                len(parts.ranges),
                parts.part_bytes,
                self.head_bytes,
            ) + b"".join(_RANGE.pack(*item) for item in parts.ranges)
            size = sum(count for _, count in parts.ranges) * parts.part_bytes
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
                expected = size if verb == b"G" else 0
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
        self.layers = tuple(sorted(groups[0].layer_names))
        global_layers = kv_cache_config.plait_layer_names
        if global_layers is None:
            if parallel.pipeline_parallel_size > 1:
                raise ValueError("Plait pipeline KV requires the global layer order")
            global_layers = self.layers
        if (
            not global_layers
            or tuple(sorted(set(global_layers))) != global_layers
            or not set(self.layers).issubset(global_layers)
        ):
            raise ValueError("Plait KV has an invalid model layer order")
        self.block_size = spec.block_size
        self.heads = spec.num_kv_heads
        self.content = spec.head_size + spec.head_size_v
        self.dtype = spec.dtype
        self.first_head = (
            get_tensor_model_parallel_rank() * self.heads
            if role == KVConnectorRole.WORKER
            else 0
        )
        part_bytes = self.block_size * self.content * self.dtype.itemsize
        head_bytes = len(global_layers) * part_bytes
        if total_heads * head_bytes > _MAX_BLOCK_BYTES:
            raise ValueError("Plait KV block exceeds the store limit")
        timeout = float(extra.get("timeout", 3.0))
        if not 0 < timeout <= 15:
            raise ValueError("invalid Plait KV store timeout")
        self.store = Store(address, total_heads, head_bytes, timeout)
        self.checkpoint_decode = extra.get("checkpoint_decode", False) is True
        self.layer_ranges = (
            LayerRanges.for_worker(
                global_layers,
                self.layers,
                total_heads,
                self.first_head,
                self.heads,
                part_bytes,
            )
            if role == KVConnectorRole.WORKER and self.layers != global_layers
            else None
        )
        self.domain = json.dumps(
            [
                "plait-kv2",
                namespace,
                global_layers,
                str(self.dtype),
                self.block_size,
                total_heads,
                self.content,
            ]
        ).encode()
        self.session = uuid.uuid4().bytes
        self.requests: dict[str, tuple[Request, int]] = {}
        self.found: dict[str, tuple[int, list[bytes]]] = {}
        self.unavailable_requests: set[str] = set()
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

    def _request_digest(self, request: Request):
        digest = hashlib.sha256(self.domain)
        salt = (request.cache_salt or "").encode()
        digest.update(struct.pack("<Q", len(salt)))
        digest.update(salt)
        return digest

    def _keys(self, request: Request, count: int) -> list[bytes]:
        digest = self._request_digest(request)
        tokens = request.all_token_ids
        keys = []
        for index in range(count):
            start = index * self.block_size
            block = tokens[start : start + self.block_size]
            digest.update(struct.pack(f"<{len(block)}I", *block))
            keys.append(digest.digest())
        return keys

    def _handoff_receipt(self, request: Request, ticket: str) -> dict[str, Any]:
        tokens = request.prompt_token_ids
        assert tokens
        digest = self._request_digest(request)
        digest.update(struct.pack(f"<{len(tokens)}I", *tokens))
        pages = (len(tokens) - 1) // self.block_size
        host, port = self.store.address
        return {
            "ticket": ticket,
            "store": f"[{host}]:{port}" if ":" in host else f"{host}:{port}",
            "prompt_sha256": digest.hexdigest(),
            "cached_tokens": pages * self.block_size,
        }

    @staticmethod
    def _handoff_ticket(params: Any) -> str | None:
        ticket = params.get("ticket") if isinstance(params, dict) else None
        if isinstance(ticket, str) and re.fullmatch("[0-9a-f]{32}", ticket):
            return ticket
        return None

    def _reject_handoff(self, request: Request) -> tuple[None, bool]:
        self.unavailable_requests.add(request.request_id)
        return None, False

    def take_unavailable_requests(self) -> set[str]:
        unavailable, self.unavailable_requests = self.unavailable_requests, set()
        return unavailable

    def get_num_new_matched_tokens(
        self,
        request: Request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        self.found.pop(request.request_id, None)
        if not self._kv_transfer_config.is_kv_consumer:
            return 0, False
        if request.request_id in self.unavailable_requests:
            return None, False
        params = request.kv_transfer_params or {}
        required = "plait_handoff" in params
        if not self._eligible(request):
            return self._reject_handoff(request) if required else (0, False)
        if required and not request.prompt_token_ids:
            return self._reject_handoff(request)
        first = num_computed_tokens // self.block_size
        required_count = 0
        # Leave one known input for the forward that produces the next logits.
        count = (len(request.all_token_ids) - 1) // self.block_size
        if required:
            receipt = params["plait_handoff"]
            ticket = self._handoff_ticket(receipt)
            if (
                ticket is None
                or type(receipt.get("cached_tokens")) is not int
                or receipt != self._handoff_receipt(request, ticket)
                or self._kv_transfer_config.kv_load_failure_policy != "fail"
            ):
                return self._reject_handoff(request)
            required_count = receipt["cached_tokens"] // self.block_size
            if first < required_count and request.skip_reading_prefix_cache:
                return self._reject_handoff(request)
            if not params.get("plait_resume_token_ids"):
                count = required_count
        if count <= first:
            return 0, False
        hits = []
        try:
            for key in self._keys(request, count)[first:]:
                if self.store.exchange(b"H", key) is None:
                    break
                hits.append(key)
        except OSError:
            logger.warning_once("Plait KV store unavailable during prefix lookup")
        if first + len(hits) < required_count:
            return self._reject_handoff(request)
        if hits:
            self.found[request.request_id] = (first, hits)
        return len(hits) * self.block_size, False

    def update_state_after_alloc(
        self,
        request: Request,
        blocks: KVCacheBlocks,
        num_external_tokens: int,
    ) -> None:
        params = request.kv_transfer_params or {}
        checkpoint = self.checkpoint_decode and params.get("plait_checkpoint") is True
        if (self._kv_transfer_config.is_kv_producer or checkpoint) and self._eligible(
            request
        ):
            self.requests.setdefault(request.request_id, (request, 0))
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
            tracked = self.requests.get(req_id)
            if tracked is None:
                continue
            request, first = tracked
            # Async scheduling can schedule a page boundary before the token IDs
            # are available. Revisit pages once both the IDs and work are staged.
            count = (
                min(request.num_computed_tokens + scheduled, len(request.all_token_ids))
                // self.block_size
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
            self.requests[req_id] = (request, count)
        return metadata

    def request_finished(
        self,
        request: Request,
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        self.requests.pop(request.request_id, None)
        self.found.pop(request.request_id, None)
        params = (request.kv_transfer_params or {}).get("plait_handoff")
        ticket = self._handoff_ticket(params)
        if (
            not self._kv_transfer_config.is_kv_producer
            or ticket is None
            or set(params) != {"ticket"}
            or request.status
            not in (
                RequestStatus.FINISHED_STOPPED,
                RequestStatus.FINISHED_LENGTH_CAPPED,
            )
            or not self._eligible(request)
            or not request.prompt_token_ids
        ):
            return False, None
        receipt = self._handoff_receipt(request, ticket)
        count = receipt["cached_tokens"] // self.block_size
        if request.num_computed_tokens < receipt["cached_tokens"]:
            return False, None
        try:
            if any(
                self.store.exchange(b"H", key) is None
                for key in self._keys(request, count)
            ):
                return False, None
        except OSError:
            logger.warning_once("Plait KV store unavailable during handoff publication")
            return False, None
        return False, {"plait_handoff": receipt}

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
        if not self.layers:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, PlaitKvMetadata)
        for block in metadata.loads:
            try:
                data = self.store.exchange(
                    b"G",
                    block.key,
                    self.first_head,
                    self.heads,
                    parts=self.layer_ranges,
                )
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
        if self.invalid_blocks or not self.layers:
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
                    b"P",
                    block.key,
                    self.first_head,
                    self.heads,
                    writer,
                    data,
                    parts=self.layer_ranges,
                )
            except OSError:
                logger.warning_once("Plait KV store unavailable during publication")
                break

    def get_block_ids_with_load_errors(self) -> set[int]:
        failures, self.invalid_blocks = self.invalid_blocks, set()
        return failures
