# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from struct import Struct
from types import SimpleNamespace
from unittest import TestCase

from vllm.distributed.kv_transfer.kv_connector.v1.plait_kv_connector import (
    Block,
    LayerRanges,
    PlaitKvConnector,
    PlaitKvPeer,
    _encode_peer_manifest,
)
from vllm.exceptions import VLLMValidationError
from vllm.sampling_params import RequestOutputKind, SamplingParams
from vllm.v1.engine.input_processor import InputProcessor
from vllm.v1.request import RequestStatus


def test_worker_manifest_charges_each_address_once_for_its_full_share() -> None:
    workers = [
        PlaitKvPeer("192.0.2.2", 4),
        PlaitKvPeer("::ffff:192.0.2.2", 4),
        PlaitKvPeer("192.0.2.3", 12),
    ]
    entry = Struct("<16sQ")
    manifest = _encode_peer_manifest(workers, 20)
    assert list(entry.iter_unpack(manifest)) == [
        (bytes(10) + b"\xff\xff\xc0\x00\x02\x02", 8),
        (bytes(10) + b"\xff\xff\xc0\x00\x02\x03", 12),
    ]
    with TestCase().assertRaises(ValueError):
        _encode_peer_manifest(workers[1:], 20)


def test_pipeline_stage_merges_only_adjacent_global_head_and_layer_slots() -> None:
    layers = ("model.layers.0", "model.layers.1", "model.layers.10", "model.layers.2")
    local_layers = (layers[0], layers[1], layers[3])
    ranges = LayerRanges.for_worker(layers, local_layers, 4, 1, 2, 128)
    assert ranges.part_bytes == 128
    assert ranges.ranges == ((4, 2), (7, 3), (11, 1))


def test_seeded_stream_requires_the_explicit_sampler_contract() -> None:
    processor = object.__new__(InputProcessor)
    processor.model_config = SimpleNamespace(
        get_vocab_size=lambda: 32, max_model_len=64, use_fp64_gumbel=False
    )
    extra = {"seeded_resume": True}
    processor.vllm_config = SimpleNamespace(
        model_config=processor.model_config,
        kv_transfer_config=SimpleNamespace(
            kv_connector="PlaitKvConnector",
            is_kv_consumer=True,
            kv_connector_extra_config=extra,
        ),
        use_v2_model_runner=True,
        speculative_config=None,
        watermark_config=None,
    )
    params = SamplingParams(
        temperature=0.8, seed=42, max_tokens=4, output_kind=RequestOutputKind.DELTA
    )
    mode = {"plait_sampling_mode": "seeded-v2"}
    processor._validate_plait_stream(mode, params)
    assert processor._validate_plait_resume_tokens([7, 8], params, 12) == [7, 8]

    with TestCase().assertRaises(VLLMValidationError):
        processor._validate_plait_stream({}, params)
    extra["seeded_resume"] = False
    with TestCase().assertRaises(VLLMValidationError):
        processor._validate_plait_stream(mode, params)
    extra["seeded_resume"] = True
    params.seed = None
    with TestCase().assertRaises(VLLMValidationError):
        processor._validate_plait_stream(mode, params)
    params.temperature = 0
    processor._validate_plait_stream({}, params)


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
    connector.checkpoint_only = False
    connector.checkpoint_admission = False
    connector.checkpoint_peer_budget = False
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
    connector.checkpoint_only = True
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    assert connector.found == {}
    request.kv_transfer_params = {"plait_checkpoint": True}
    assert connector.get_num_new_matched_tokens(request, 0) == (4, False)
    connector.checkpoint_only = False

    # Resume keeps the promised prompt pages mandatory and can load complete
    # pages containing generated history. An absent optional tail is a miss.
    output = list(range(20, 29))
    request.all_token_ids = request.prompt_token_ids + output
    keys = connector._keys(request, 4)
    present = set(keys[:3])
    request.kv_transfer_params = {**transferred, "plait_resume_token_ids": output}
    assert connector.get_num_new_matched_tokens(request, 0) == (12, False)
    connector.checkpoint_admission = True
    exchanges = []

    def limited(verb, key, writer=None):
        exchanges.append((verb, key, writer))
        return b"" if verb == b"H" and key in present else None

    original = connector.store.exchange
    connector.store.exchange = limited
    assert connector.get_num_new_matched_tokens(request, 0) == (8, False)
    assert [verb for verb, _, _ in exchanges] == [b"H", b"H", b"h"]
    assert exchanges[-1][2] is not None and len(exchanges[-1][2]) == 16
    assert connector.found[request.request_id] == (0, [(keys[0], None), (keys[1], None)])
    exchanges.clear()

    def admit(verb, key, writer=None):
        exchanges.append((verb, key, writer))
        return b"" if key in present else None

    connector.store.exchange = admit
    assert connector.get_num_new_matched_tokens(request, 0) == (12, False)
    blocks = SimpleNamespace(get_block_ids=lambda: ([7, 8, 9],))
    connector.checkpoint_decode = False
    connector.loads = []
    connector.update_state_after_alloc(request, blocks, 12)
    assert connector.loads == [
        Block(keys[0], 7), Block(keys[1], 8), Block(keys[2], 9, exchanges[2][2]),
    ]
    connector.store.exchange = original
    connector.checkpoint_admission = False
    present.remove(keys[1])
    assert connector.get_num_new_matched_tokens(request, 0) == (None, False)
    assert connector.take_unavailable_requests() == {request.request_id}


def test_decode_checkpoint_publication_requires_profile_and_request_opt_in() -> None:
    connector = object.__new__(PlaitKvConnector)
    connector.requests = {}
    connector.found = {}
    connector._kv_transfer_config = SimpleNamespace(is_kv_producer=False)
    request = SimpleNamespace(
        request_id="stream",
        prompt_token_ids=[1],
        prompt_embeds=None,
        mm_features=[],
        lora_request=None,
        kv_transfer_params={"plait_checkpoint": True},
    )
    connector.checkpoint_decode = False
    connector.update_state_after_alloc(request, None, 0)
    assert connector.requests == {}
    connector.checkpoint_decode = True
    request.kv_transfer_params = {}
    connector.update_state_after_alloc(request, None, 0)
    assert connector.requests == {}
    request.kv_transfer_params = {"plait_checkpoint": True}
    connector.update_state_after_alloc(request, None, 0)
    assert connector.requests == {"stream": (request, 0)}
