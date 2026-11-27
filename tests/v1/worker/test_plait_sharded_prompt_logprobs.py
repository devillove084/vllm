# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import torch

from vllm.model_executor.layers.logits_processor import PromptLogitsShard
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample import sharded_prompt_logprob


class _Collectives:
    def __init__(self, world_size):
        self.barrier = Barrier(world_size, timeout=10)
        self.inputs = [None] * world_size
        self.history = []

    def exchange(self, rank, kind, value, dim):
        self.inputs[rank] = (kind, value)
        self.barrier.wait()
        assert all(entry[0] == kind for entry in self.inputs)
        tensors = [entry[1] for entry in self.inputs]
        if rank == 0:
            self.history.append((kind, tuple(value.shape), dim))
        if kind == "gather":
            result = torch.cat(tensors, dim=dim)
        else:
            result = torch.stack(tensors).sum(dim=0, dtype=value.dtype)
        self.barrier.wait()
        return result


class _Communicator:
    def __init__(self, collectives, rank):
        self.collectives = collectives
        self.rank = rank
        self.world_size = len(collectives.inputs)

    def all_gather(self, value, dim=-1):
        return self.collectives.exchange(self.rank, "gather", value, dim)

    def all_reduce(self, value):
        return self.collectives.exchange(self.rank, "sum", value, 0)


def _dense_scores(logits, k, targets, *, logits_mode):
    ids = torch.cat((targets[:, None], logits.topk(k, dim=-1).indices), dim=-1)
    rank = (logits >= logits.gather(-1, targets[:, None])).sum(dim=-1)
    scores = logits.float() if logits_mode else logits.float().log_softmax(dim=-1)
    return LogprobsTensors(ids, scores.gather(-1, ids), rank)


def test_sharded_prompt_scores_gather_only_tied_rows(monkeypatch):
    monkeypatch.setattr(sharded_prompt_logprob, "compute_topk_scores", _dense_scores)
    logits = torch.tensor(
        [
            [5, -2, 8, -1, 1, 2, 0, 6, 4, 3],
            [-1, 1, 10, 2, 3, 10, 4, 5, 6, 7],
            [11, 7, 0, 2, 3, 6, 5, 4, 7, 1],
        ],
        dtype=torch.float16,
    )
    targets = torch.tensor([9, 1, 5])
    padded = torch.nn.functional.pad(logits, (0, 2), value=200)
    world = 3
    width = 4
    for logits_mode in (True, False):
        for k in (0, 2):
            collectives = _Collectives(world)

            def score(rank, *, collectives=collectives, k=k, logits_mode=logits_mode):
                start = rank * width
                shard = PromptLogitsShard(
                    project=lambda _hidden: padded[:, start : start + width],
                    communicator=_Communicator(collectives, rank),
                    vocab_size=logits.shape[-1],
                    shard_width=width,
                    first_token=start,
                )
                return sharded_prompt_logprob.compute_sharded_prompt_scores(
                    shard, logits, targets, k, logits_mode=logits_mode
                )

            with ThreadPoolExecutor(max_workers=world) as pool:
                actual = list(pool.map(score, range(world)))
            reference = _dense_scores(logits, k, targets, logits_mode=logits_mode)
            for result in actual:
                torch.testing.assert_close(
                    result.logprob_token_ids, reference.logprob_token_ids
                )
                torch.testing.assert_close(
                    result.selected_token_ranks, reference.selected_token_ranks
                )
                torch.testing.assert_close(
                    result.logprobs, reference.logprobs, atol=2e-6, rtol=0
                )
            packet_width = (2 if logits_mode else 4) + (2 * (k + 1) if k else 0)
            expected = [("gather", (3, packet_width), 0), ("sum", (3,), 0)]
            if k:
                expected.append(("gather", (2, width), -1))
            assert collectives.history == expected
