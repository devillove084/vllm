# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Prepare a private candidate using the serving benchmark's workload."""

import argparse
import asyncio
import time
from collections.abc import Awaitable, Callable, Iterator


async def run_waves(
    widths: list[int],
    serial_requests: int,
    submit: Callable[[Callable[[], None] | None], Awaitable[None]],
) -> None:
    for _ in range(serial_requests):
        await submit(None)

    async def inject(events: list[asyncio.Event]) -> None:
        await asyncio.gather(*(event.wait() for event in events))
        await submit(None)

    for width in widths:
        events = [asyncio.Event() for _ in range(width)]
        tasks = [asyncio.create_task(submit(event.set)) for event in events]
        tasks.append(asyncio.create_task(inject(events)))
        try:
            await asyncio.gather(*tasks)
        finally:
            # gather does not cancel its other children when a request fails.
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)


def run_warmup(args: argparse.Namespace) -> dict:
    import aiohttp

    from vllm.benchmarks.datasets import RandomDataset, SampleRequest
    from vllm.benchmarks.lib.endpoint_request_func import (
        AIOHTTP_TIMEOUT,
        RequestFuncInput,
        async_request_openai_completions,
    )
    from vllm.tokenizers import get_tokenizer

    widths = args.warmup_decode_widths
    total = args.warmups + sum(width + 1 for width in widths)
    timed_seeds = set(args.seeds)
    seed = next(s for s in range(len(timed_seeds) + 1) if s not in timed_seeds)
    tokenizer = get_tokenizer(args.model)
    samples: Iterator[SampleRequest] = iter(
        RandomDataset(random_seed=seed).sample(
            tokenizer,
            num_requests=total,
            input_len=args.input_tokens,
            output_len=args.output_tokens,
        )
    )
    api_url = args.url.rstrip("/") + "/v1/completions"

    async def prepare() -> None:
        async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:

            async def submit(notify: Callable[[], None] | None) -> None:
                sample = next(samples)
                token_received = False

                def on_token() -> None:
                    nonlocal token_received
                    token_received = True
                    if notify is not None:
                        notify()

                result = await async_request_openai_completions(
                    RequestFuncInput(
                        prompt=sample.prompt,
                        api_url=api_url,
                        prompt_len=sample.prompt_len,
                        output_len=sample.expected_output_len,
                        model=args.model,
                        model_name=args.served_model,
                        ignore_eos=True,
                        extra_body={"return_token_ids": True},
                    ),
                    session,
                    on_first_token=on_token,
                )
                if not result.success or not token_received:
                    raise RuntimeError("Plait warmup did not receive a complete stream")
                if result.output_tokens != sample.expected_output_len:
                    raise RuntimeError(
                        "Plait warmup returned an unexpected token count"
                    )

            await run_waves(widths, args.warmups, submit)

    started = time.monotonic()
    asyncio.run(prepare())
    return {"requests": total, "seed": seed, "elapsed_s": time.monotonic() - started}
