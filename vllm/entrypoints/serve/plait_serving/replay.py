# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Huanbing
"""Checkpoint and replay the visible prefix of one vLLM completion stream."""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from http.client import HTTPResponse

from .group import HTTP_ERRORS, MAX_JSON_BYTES
from .launch import SEEDED_KV_RESUME

RECOVERY_HEADER = "X-Plait-Stream-Recovery"
RECOVERY_MODE = "replay-prefix"
RESUME_KV_MODE = "resume-kv"
RESUME_PREFIX_MODE = "resume-prefix"
RESUME_MODES = (RESUME_KV_MODE, RESUME_PREFIX_MODE)


class ReplayMismatch(ValueError):
    pass


class UpstreamInterrupted(Exception):
    pass


class PlannedHandoff(Exception):
    pass


def sse(payload: dict) -> bytes:
    return b"data: " + json.dumps(payload, separators=(",", ":")).encode() + b"\n\n"


def events(response: HTTPResponse) -> Iterator[bytes]:
    lines = []
    size = 0
    while True:
        try:
            line = response.readline(MAX_JSON_BYTES + 1)
        except HTTP_ERRORS as error:
            raise UpstreamInterrupted from error
        if not line:
            raise UpstreamInterrupted
        size += len(line)
        if size > MAX_JSON_BYTES:
            raise ReplayMismatch("upstream SSE event exceeds the size limit")
        if line in (b"\n", b"\r\n"):
            if lines:
                yield b"\n".join(lines)
                lines = []
            size = 0
        elif line.startswith(b"data:"):
            lines.append(line[5:].removeprefix(b" ").rstrip(b"\r\n"))


def validated_request(body: dict, mode: str = RECOVERY_MODE) -> dict:
    prompt = body.get("prompt")
    options = body.get("stream_options") or {}
    temperature = body.get("temperature")
    seed = body.get("seed")
    seeded = (
        mode == RESUME_KV_MODE
        and type(temperature) in (int, float)
        and 0 < temperature < float("inf")
        and type(seed) is int
        and -(1 << 63) <= seed < (1 << 63)
        and seed != -1
    )
    if (
        body.get("stream") is not True
        or not (seeded or (type(temperature) in (int, float) and temperature == 0))
        or body.get("n", 1) != 1
        or not (
            isinstance(prompt, str)
            or (
                isinstance(prompt, list)
                and prompt
                and all(type(token) is int for token in prompt)
            )
        )
        or body.get("echo")
        or body.get("use_beam_search")
        or body.get("logprobs") is not None
        or body.get("logprob_token_ids")
        or not isinstance(options, dict)
        or options.get("continuous_usage_stats")
        or body.get("kv_transfer_params")
    ):
        raise ValueError(
            "stream recovery requires one greedy completion or a KV resume "
            "request with an explicit sampling seed"
        )
    return {**body, "return_token_ids": True}


def match_prefix(values, prefix, offset: int) -> int:
    count = min(len(values), len(prefix) - offset)
    if values[:count] != prefix[offset : offset + count]:
        raise ReplayMismatch("replacement differs from the emitted prefix")
    return count


class CompletionReplay:
    def __init__(self, body: dict, mode: str = RECOVERY_MODE) -> None:
        self.request = validated_request(body, mode)
        self.seed = (
            self.request.get("seed") if self.request["temperature"] > 0 else None
        )
        if mode in RESUME_MODES and (
            body.get("prompt_logprobs") is not None
            or body.get("structured_outputs")
            or body.get("response_format")
            or body.get("repetition_detection")
            or body.get("thinking_token_budget", -1) not in (None, -1)
            or body.get("prompt_embeds")
            or body.get("vllm_xargs")
        ):
            raise ValueError("Plait resume requires a plain text completion")
        self.payload = json.dumps(self.request).encode()
        self.mode = mode
        self.model = body["model"]
        self.return_token_ids = bool(body.get("return_token_ids"))
        self.started = False
        self.failed_epoch = 0
        self.identity: dict = {}
        self.tokens: list[int] = []
        self.text: list[str] = []
        self.text_bytes = 0
        self.prompt_ids: list[int] | None = None
        self.prompt_sent = False

    def attempt_payload(self) -> bytes:
        if self.mode not in RESUME_MODES or (not self.tokens and self.seed is None):
            return self.payload
        return json.dumps(self.generation_request()).encode()

    def generation_request(self, allow_resume: bool = True) -> dict:
        request = dict(self.request)
        params = {}
        if self.seed is not None:
            params["plait_sampling_mode"] = SEEDED_KV_RESUME
        if allow_resume and self.tokens:
            if self.mode == RESUME_KV_MODE:
                params["plait_resume_token_ids"] = self.tokens.copy()
            elif self.mode == RESUME_PREFIX_MODE:
                request["vllm_xargs"] = {"plait_resume_token_ids": self.tokens.copy()}
        if params:
            request["kv_transfer_params"] = params
        return request

    def drain_attempt(
        self,
        response: HTTPResponse,
        emit: Callable[[bytes], object],
        should_handoff: Callable[[], bool] | None = None,
        *,
        allow_resume: bool = True,
    ) -> None:
        reference_ids = self.tokens.copy()
        reference_text = "".join(self.text)
        token_offset = text_offset = 0
        new_ids: list[int] = []
        new_text: list[str] = []
        new_size = 0
        first_choice = True
        terminal: dict | None = None
        usage: dict | None = None
        for data in events(response):
            if data == b"[DONE]":
                if (
                    terminal is None
                    or token_offset != len(reference_ids)
                    or text_offset != len(reference_text)
                ):
                    raise ReplayMismatch(
                        "replacement ended before matching the emitted prefix"
                    )
                emit(
                    sse(terminal)
                    + (sse(usage) if usage is not None else b"")
                    + b"data: [DONE]\n\n"
                )
                return
            frame = self.frame(data)
            choices = frame["choices"]
            if (
                not choices
                and terminal is not None
                and "usage" in frame
                and usage is None
            ):
                usage = frame
                continue
            if len(choices) != 1 or terminal is not None:
                raise ReplayMismatch("invalid completion stream order")
            choice = choices[0]
            if not isinstance(choice, dict):
                raise ReplayMismatch("invalid completion choice")
            ids, delta = choice.get("token_ids"), choice.get("text")
            if (
                choice.get("index") != 0
                or not isinstance(delta, str)
                or not isinstance(ids, list)
                or any(type(token) is not int for token in ids)
            ):
                raise ReplayMismatch(
                    "completion stream did not carry one token-addressable choice"
                )
            prompt = choice.pop("prompt_token_ids", None)
            resumed_tokens = choice.pop("plait_resumed_tokens", None)
            sampled_seed = choice.pop("plait_sampled_seed", None)
            if (
                first_choice
                and self.seed is not None
                and (type(sampled_seed) is not int or sampled_seed != self.seed)
            ):
                raise ReplayMismatch("engine did not confirm the sampling seed")
            if (
                first_choice
                and allow_resume
                and self.mode in RESUME_MODES
                and reference_ids
            ) and (
                type(resumed_tokens) is not int or resumed_tokens != len(reference_ids)
            ):
                raise ReplayMismatch(
                    "replacement did not restore the emitted token checkpoint"
                )
            if first_choice or prompt is not None:
                if not isinstance(prompt, list) or any(
                    type(token) is not int for token in prompt
                ):
                    raise ReplayMismatch(
                        "completion stream did not carry its prompt token IDs"
                    )
                if self.prompt_ids is not None and prompt != self.prompt_ids:
                    raise ReplayMismatch("replacement tokenized the prompt differently")
                self.prompt_ids = prompt
                first_choice = False

            matched_ids = match_prefix(ids, reference_ids, token_offset)
            matched_text = match_prefix(delta, reference_text, text_offset)
            token_offset += matched_ids
            text_offset += matched_text
            new_ids.extend(ids[matched_ids:])
            suffix = delta[matched_text:]
            new_text.append(suffix)
            new_size += len(suffix.encode())
            if (
                self.text_bytes + new_size + 8 * (len(self.tokens) + len(new_ids))
                > MAX_JSON_BYTES
            ):
                raise ReplayMismatch(
                    "completion prefix exceeds the recovery size limit"
                )
            if token_offset != len(reference_ids) or text_offset != len(reference_text):
                if choice.get("finish_reason") is not None:
                    raise ReplayMismatch(
                        "replacement ended before matching the emitted prefix"
                    )
                continue

            choice["text"] = new_delta = "".join(new_text)
            if self.return_token_ids:
                choice["token_ids"] = new_ids
                if not self.prompt_sent and self.prompt_ids is not None:
                    choice["prompt_token_ids"] = self.prompt_ids
            else:
                choice.pop("token_ids")

            if choice.get("finish_reason") is not None:
                terminal = frame
            elif new_ids or new_delta:
                emit(sse(frame))
                self.tokens.extend(new_ids)
                self.text.append(new_delta)
                self.text_bytes += new_size
                self.prompt_sent = self.prompt_sent or self.prompt_ids is not None
                if should_handoff is not None:
                    limit = self.request.get("max_tokens")
                    if (
                        type(limit) is not int or len(self.tokens) < limit
                    ) and should_handoff():
                        raise PlannedHandoff
            new_ids = []
            new_text = []
            new_size = 0

    def frame(self, data: bytes) -> dict:
        try:
            frame = json.loads(data)
        except ValueError as error:
            raise ReplayMismatch("invalid vLLM completion event") from error
        if isinstance(frame, dict) and isinstance(frame.get("error"), dict):
            raise UpstreamInterrupted
        if (
            not isinstance(frame, dict)
            or frame.get("model") != self.model
            or not isinstance(frame.get("choices"), list)
            or not isinstance(frame.get("id"), str)
        ):
            raise ReplayMismatch("invalid vLLM completion event")
        if not self.identity:
            self.identity = {
                key: frame[key] for key in ("id", "created", "model") if key in frame
            }
        frame.update(self.identity)
        return frame
