# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.plait_kv_connector import (
    Block,
    PlaitKvConnector,
)


def test_exports_a_page_after_async_tokens_become_known() -> None:
    connector = object.__new__(PlaitKvConnector)
    connector.block_size = 4
    connector.domain = b"test-model"
    connector.session = bytes(16)
    connector.loads = []
    request = SimpleNamespace(
        num_computed_tokens=3,
        all_token_ids=[11, 12, 13],
        cache_salt=None,
    )
    connector.requests = {"request": (request, 0)}
    step = SimpleNamespace(
        num_scheduled_tokens={"request": 1},
        kv_connector_block_state=SimpleNamespace(get_block_ids=lambda _: ([42, 43],)),
    )

    # The scheduler crosses the page boundary with a still-pending token ID.
    assert connector.build_connector_meta(step).saves == []

    request.num_computed_tokens = 4
    request.all_token_ids.append(14)
    [(writer, block)] = connector.build_connector_meta(step).saves
    assert len(writer) == 16
    assert block == Block(connector._keys(request, 1)[0], 42)

    request.num_computed_tokens = 5
    request.all_token_ids.append(15)
    assert connector.build_connector_meta(step).saves == []
