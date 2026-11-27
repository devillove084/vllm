# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample import prompt_logprob


def test_prompt_logits_submits_the_next_chunk_before_wait(monkeypatch):
    monkeypatch.setattr(prompt_logprob, "CHUNK_SIZE", 2)

    def score(logits, num_logprobs, token_ids, *, logits_mode):
        return LogprobsTensors(
            logprob_token_ids=token_ids.unsqueeze(-1),
            logprobs=logits.gather(-1, token_ids.unsqueeze(-1)),
            selected_token_ranks=torch.ones_like(token_ids),
        )

    monkeypatch.setattr(prompt_logprob, "compute_topk_scores", score)
    hidden = torch.arange(20, dtype=torch.float32).reshape(5, 4)
    tokens = torch.tensor([2, 0, 1, 3, 2], dtype=torch.int64)
    events = []

    def submit(rows):
        first = int(rows[0, 0])
        events.append(("submit", first))
        projected = rows + 3

        def wait():
            events.append(("wait", first))
            return projected

        return wait

    def blocking(rows):
        return rows + 3

    def unexpected_blocking(rows):
        raise AssertionError("overlapping chunks must use the submitter")

    expected = prompt_logprob.compute_prompt_logprobs_with_chunking(
        tokens, hidden, blocking, 2
    )
    result = prompt_logprob.compute_prompt_logprobs_with_chunking(
        tokens, hidden, unexpected_blocking, 2, submit_logits_fn=submit
    )
    assert all(
        torch.equal(actual, baseline) for actual, baseline in zip(result, expected)
    )
    assert events == [
        ("submit", 0),
        ("submit", 8),
        ("wait", 0),
        ("submit", 16),
        ("wait", 8),
        ("wait", 16),
    ]

    events.clear()
    prompt_logprob.compute_prompt_logprobs_with_chunking(
        tokens[:2], hidden[:2], blocking, 2, submit_logits_fn=submit
    )
    assert events == []
