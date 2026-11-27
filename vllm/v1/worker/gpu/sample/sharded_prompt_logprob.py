# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch.nn.functional import pad

from vllm.v1.outputs import LogprobsTensors
from vllm.v1.worker.gpu.sample.logprob import compute_topk_scores

if TYPE_CHECKING:
    from vllm.model_executor.layers.logits_processor import PromptLogitsShard


def prepare_sharded_prompt_scores(
    shard: "PromptLogitsShard",
    hidden_states: torch.Tensor,
    token_ids: torch.Tensor,
    k: int,
    *,
    logits_mode: bool,
) -> "PreparedShardedPromptScores":
    """Project and pack a local prompt block without issuing collectives."""
    projected = shard.project(hidden_states)
    logits = projected[:, : shard.vocab_size - shard.first_token]
    width = logits.shape[-1]
    targets = token_ids.clamp(0, shard.vocab_size - 1)
    local_ids = (targets - shard.first_token).clamp(0, width - 1)
    local_scores = logits.gather(-1, local_ids[:, None]).float()
    local_max = logits.amax(dim=-1, keepdim=True).float()
    finite = logits.amin(dim=-1, keepdim=True).isfinite() & local_max.isfinite()
    fields = [local_scores, finite.float()]
    if not logits_mode:
        local_sum = (logits.float() - local_max).exp().sum(dim=-1, keepdim=True)
        fields.extend((local_max, local_sum))

    # Retain one extra candidate to detect a tie at the global cutoff.
    candidates = k + 1 if k else 0
    if candidates:
        local_top, local_top_ids = logits.topk(min(width, candidates), dim=-1)
        padding = candidates - local_top.shape[-1]
        fields.extend(
            (
                pad(local_top.float(), (0, padding), value=-float("inf")),
                pad((local_top_ids + shard.first_token).float(), (0, padding)),
            )
        )

    # Global vocabulary IDs are exactly representable in this FP32 packet.
    packet = torch.cat(fields, dim=-1)
    return PreparedShardedPromptScores(
        shard, projected, logits, token_ids, targets, packet, k, logits_mode
    )


@dataclass
class PreparedShardedPromptScores:
    shard: "PromptLogitsShard"
    projected: torch.Tensor
    logits: torch.Tensor
    token_ids: torch.Tensor
    targets: torch.Tensor
    packet: torch.Tensor
    k: int
    logits_mode: bool

    def try_submit_metadata(self) -> Callable[[], torch.Tensor] | None:
        return self.shard.communicator.try_all_gather(self.packet, dim=0)

    def finish(
        self,
        wait: Callable[[], torch.Tensor] | None = None,
        *,
        overlap_rank_reduction: bool = False,
    ) -> LogprobsTensors:
        shard = self.shard
        comm = shard.communicator
        logits = self.logits
        token_ids, targets = self.token_ids, self.targets
        k, logits_mode = self.k, self.logits_mode
        rows = logits.shape[0]
        metadata_width = 2 if logits_mode else 4
        candidates = k + 1 if k else 0
        gathered = comm.all_gather(self.packet, dim=0) if wait is None else wait()
        packets = gathered.view(comm.world_size, rows, -1)
        indices = torch.arange(rows, device=logits.device)
        selected_scores = packets[targets // shard.shard_width, indices, 0]
        fallback = (token_ids != targets) | ~packets[:, :, 1].bool().all(dim=0)
        output_ids = token_ids[:, None]
        output_scores = selected_scores[:, None]

        if candidates:
            values, ids = packets[:, :, metadata_width:].split(candidates, dim=-1)
            all_values = values.transpose(0, 1).flatten(1)
            all_ids = ids.transpose(0, 1).flatten(1)
            top, order = all_values.topk(candidates, dim=-1)
            fallback |= (top[:, :-1] == top[:, 1:]).any(dim=-1)
            output_ids = torch.cat(
                (output_ids, all_ids.gather(-1, order[:, :k]).long()), dim=-1
            )
            output_scores = torch.cat((output_scores, top[:, :k]), dim=-1)

        # The gathered metadata makes every worker take the same fallback branch.
        fallback_rows = fallback.nonzero().flatten()
        if fallback_rows.numel() == rows:
            dense = comm.all_gather(self.projected)[..., : shard.vocab_size]
            return compute_topk_scores(dense, k, token_ids, logits_mode=logits_mode)

        threshold = selected_scores.to(logits.dtype)[:, None]
        local_ranks = (logits >= threshold).sum(dim=-1, dtype=torch.int32)
        wait_ranks = (
            comm.try_all_reduce(local_ranks)
            if overlap_rank_reduction and (not logits_mode or fallback_rows.numel())
            else None
        )
        reduced_ranks = comm.all_reduce(local_ranks) if wait_ranks is None else None
        if not logits_mode:
            maxima = packets[:, :, 2]
            global_max = maxima.amax(dim=0)
            partition = ((maxima - global_max).exp() * packets[:, :, 3]).sum(dim=0)
            output_scores = (output_scores - global_max[:, None]) - partition.log()[
                :, None
            ]

        fallback_input = (
            self.projected.index_select(0, fallback_rows)
            if fallback_rows.numel()
            else None
        )
        if wait_ranks is not None:
            reduced_ranks = wait_ranks()
        assert reduced_ranks is not None
        output_ranks = reduced_ranks.long()
        if fallback_input is not None:
            dense = comm.all_gather(fallback_input)
            fallback_scores = compute_topk_scores(
                dense[..., : shard.vocab_size],
                k,
                token_ids.index_select(0, fallback_rows),
                logits_mode=logits_mode,
            )
            output_ids.index_copy_(0, fallback_rows, fallback_scores.logprob_token_ids)
            output_scores.index_copy_(0, fallback_rows, fallback_scores.logprobs)
            output_ranks.index_copy_(
                0, fallback_rows, fallback_scores.selected_token_ranks
            )

        return LogprobsTensors(output_ids, output_scores, output_ranks)
