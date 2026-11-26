# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

from vllm.distributed.kv_transfer.kv_connector.v1.plait_kv_connector import (
    Block,
    PlaitKvConnector,
)
from vllm.v1.request import RequestStatus


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


def test_handoff_checks_all_pages_and_binds_the_prompt_tail() -> None:
    connector = object.__new__(PlaitKvConnector)
    connector.block_size = 4
    connector.domain = b"test-model"
    connector.found = {}
    connector.requests = {}
    connector.unavailable_requests = set()
    connector._kv_transfer_config = SimpleNamespace(
        is_kv_producer=True, is_kv_consumer=False, kv_load_failure_policy="fail"
    )
    request = SimpleNamespace(
        request_id="request",
        status=RequestStatus.FINISHED_LENGTH_CAPPED,
        cache_salt="tenant",
        kv_transfer_params={"plait_handoff": {"ticket": "a" * 32}},
        prompt_token_ids=[11, 12, 13, 14, 15, 16, 17, 18, 19],
        num_computed_tokens=9,
        mm_features=[],
        prompt_embeds=None,
        lora_request=None,
        skip_reading_prefix_cache=False,
    )
    request.all_token_ids = request.prompt_token_ids
    present = set(connector._keys(request, 2))
    connector.store = SimpleNamespace(
        address=("192.0.2.1", 7952),
        exchange=lambda verb, key: b"" if verb == b"H" and key in present else None,
    )
    delayed, transferred = connector.request_finished(request, [1, 2])
    assert not delayed and transferred["plait_handoff"]["cached_tokens"] == 8

    connector._kv_transfer_config.is_kv_producer = False
    connector._kv_transfer_config.is_kv_consumer = True
    request.kv_transfer_params = transferred
    assert connector.get_num_new_matched_tokens(request, 4) == (4, False)
    request.prompt_token_ids[-1] = 20
    assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
    assert connector.take_unavailable_requests() == {request.request_id}
    request.prompt_token_ids[-1] = 19

    present.remove(connector._keys(request, 2)[-1])
    assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
    assert connector.found == {}
    assert connector.take_unavailable_requests() == {request.request_id}
    request.kv_transfer_params = None
    assert connector.get_num_new_matched_tokens(request, 0) == (4, False)
