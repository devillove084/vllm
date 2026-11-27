# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Huanbing
"""Check that interrupted SSE attempts share one verified output prefix."""

import io
import json
import unittest

from vllm.entrypoints.serve.plait_serving.replay import (
    RESUME_KV_MODE,
    CompletionReplay,
    ReplayMismatch,
    UpstreamInterrupted,
    sse,
)

DONE = b"data: [DONE]\n\n"


def frame(
    ids, text, request_id="first", *, prompt=None, finish=None, resumed=None, seed=None
):
    choice = {"index": 0, "token_ids": ids, "text": text, "finish_reason": finish}
    if prompt is not None:
        choice["prompt_token_ids"] = prompt
    if resumed is not None:
        choice["plait_resumed_tokens"] = resumed
    if seed is not None:
        choice["plait_sampled_seed"] = seed
    return sse({"id": request_id, "created": 10, "model": "test", "choices": [choice]})


def completion(return_token_ids=True, **options):
    return CompletionReplay(
        {
            "model": "test",
            "prompt": [7],
            "stream": True,
            "temperature": 0,
            "return_token_ids": return_token_ids,
        },
        **options,
    )


class CompletionReplayTest(unittest.TestCase):
    def test_splits_at_the_visible_prefix_even_when_chunk_boundaries_change(self):
        replay = completion()
        out = []
        interrupted = frame([1, 2], "Aβ", prompt=[7]) + frame([3], "C", finish="length")
        with self.assertRaises(UpstreamInterrupted):
            replay.drain_attempt(io.BytesIO(interrupted), out.append)
        self.assertEqual(len(out), 1)
        self.assertEqual(replay.tokens, [1, 2])
        replacement = (
            frame([1], "AβC", "second", prompt=[7])
            + frame([2, 3], "", "second")
            + frame([4], "D", "second", finish="length")
            + sse(
                {
                    "id": "second",
                    "model": "test",
                    "choices": [],
                    "usage": {"completion_tokens": 4},
                }
            )
            + DONE
        )
        replay.drain_attempt(io.BytesIO(replacement), out.append)
        events = b"".join(out).split(b"\n\n")
        bodies = [
            json.loads(line[6:]) for line in events if line and line != DONE.strip()
        ]
        self.assertEqual({body["id"] for body in bodies}, {"first"})
        choices = [choice for body in bodies for choice in body["choices"]]
        self.assertEqual(
            [token for choice in choices for token in choice["token_ids"]], [1, 2, 3, 4]
        )
        self.assertEqual("".join(choice["text"] for choice in choices), "AβCD")
        self.assertEqual(sum("prompt_token_ids" in choice for choice in choices), 1)
        self.assertEqual(
            sum(choice["finish_reason"] is not None for choice in choices), 1
        )
        self.assertEqual(b"".join(out).count(DONE), 1)

    def test_discards_the_unverified_suffix_on_token_text_or_prompt_divergence(self):
        for replacement in (
            frame([99, 2], "AB", prompt=[7]),
            frame([1, 2], "XB", prompt=[7]),
            frame([1, 2], "AB", prompt=[8]),
        ):
            with self.subTest(replacement=replacement):
                replay = completion()
                out = []
                with self.assertRaises(UpstreamInterrupted):
                    replay.drain_attempt(
                        io.BytesIO(frame([1], "A", prompt=[7])), out.append
                    )
                with self.assertRaises(ReplayMismatch):
                    replay.drain_attempt(io.BytesIO(replacement + DONE), out.append)
                self.assertEqual(out, [frame([1], "A", prompt=[7])])

    def test_collects_ids_upstream_without_adding_them_to_the_client_stream(self):
        replay = completion(return_token_ids=False)
        self.assertIs(json.loads(replay.payload)["return_token_ids"], True)
        out = []
        replay.drain_attempt(
            io.BytesIO(frame([1], "A", prompt=[7], finish="length") + DONE), out.append
        )
        self.assertNotIn(b"token_ids", b"".join(out))

    def test_kv_resume_sends_latest_emitted_tokens_on_each_replacement(self):
        replay = completion(mode=RESUME_KV_MODE)
        out = []
        self.assertNotIn("kv_transfer_params", json.loads(replay.attempt_payload()))
        with self.assertRaises(UpstreamInterrupted):
            replay.drain_attempt(io.BytesIO(frame([1], "A", prompt=[7])), out.append)
        params = json.loads(replay.attempt_payload())["kv_transfer_params"]
        self.assertEqual(params, {"plait_resume_token_ids": [1]})
        with self.assertRaises(UpstreamInterrupted):
            replay.drain_attempt(
                io.BytesIO(frame([1, 2], "AB", prompt=[7], resumed=1)), out.append
            )
        params = json.loads(replay.attempt_payload())["kv_transfer_params"]
        self.assertEqual(params, {"plait_resume_token_ids": [1, 2]})
        final = frame([1, 2, 3], "ABC", prompt=[7], resumed=2, finish="length")
        replay.drain_attempt(io.BytesIO(final + DONE), out.append)
        frames = [
            json.loads(line[6:])
            for line in b"".join(out).splitlines()
            if line.startswith(b"data: {")
        ]
        self.assertEqual(
            [choice["token_ids"] for body in frames for choice in body["choices"]],
            [[1], [2], [3]],
        )
        self.assertNotIn(b"plait_resumed_tokens", b"".join(out))

    def test_kv_resume_rejects_an_unacknowledged_checkpoint(self):
        replay = completion(mode=RESUME_KV_MODE)
        out = []
        with self.assertRaises(UpstreamInterrupted):
            replay.drain_attempt(io.BytesIO(frame([1], "A", prompt=[7])), out.append)
        for claimed in (None, 2, True):
            with self.subTest(claimed=claimed), self.assertRaises(ReplayMismatch):
                replay.drain_attempt(
                    io.BytesIO(frame([1, 2], "AB", prompt=[7], resumed=claimed)),
                    out.append,
                )
        self.assertEqual(out, [frame([1], "A", prompt=[7])])

    def test_seeded_resume_requires_a_seed_acknowledgment_before_emitting(self):
        request = {
            "model": "test",
            "prompt": [7],
            "stream": True,
            "temperature": 0.8,
            "return_token_ids": True,
            "seed": 42,
        }
        with self.assertRaises(ValueError):
            CompletionReplay(request)
        for seed in (None, -1, True, 1 << 63, -(1 << 63) - 1):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                CompletionReplay({**request, "seed": seed}, RESUME_KV_MODE)

        replay = CompletionReplay(request, RESUME_KV_MODE)
        out = []
        with self.assertRaises(ReplayMismatch):
            replay.drain_attempt(io.BytesIO(frame([1], "A", prompt=[7])), out.append)
        self.assertEqual(out, [])
        with self.assertRaises(UpstreamInterrupted):
            replay.drain_attempt(
                io.BytesIO(frame([1], "A", prompt=[7], seed=42)), out.append
            )
        for claimed in (None, 41, True):
            with self.subTest(claimed=claimed), self.assertRaises(ReplayMismatch):
                replay.drain_attempt(
                    io.BytesIO(
                        frame([1, 2], "AB", prompt=[7], resumed=1, seed=claimed)
                    ),
                    out.append,
                )
        self.assertEqual(out, [frame([1], "A", prompt=[7])])


if __name__ == "__main__":
    unittest.main()
