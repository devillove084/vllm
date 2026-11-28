#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Huanbing
"""Calibrate a conservative Plait capacity profile with vLLM's serving benchmark."""

from __future__ import annotations

import argparse
import gc
import json
import math
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

TRANSFER_BYTES = re.compile(
    r"^vllm:nixl_bytes_transferred_sum(?:\{[^\n]*\})?\s+(\S+)", re.M
)


def transferred_bytes(url: str | None) -> int | None:
    if url is None:
        return None
    with urllib.request.urlopen(url, timeout=10) as response:
        metrics = response.read().decode()
    return int(sum(float(item) for item in TRANSFER_BYTES.findall(metrics)))


def meets_attainment(trials: list[dict], attainment: float) -> bool:
    return all(
        trial["good_requests"] / trial["arrivals"] >= attainment for trial in trials
    )


def summarize(points: list[dict], attainment: float, valid_for_ms: int) -> dict:
    """Use the last contiguous rate that met the SLO in every repeated trial."""
    selected = None
    for point in points:
        point["passes"] = meets_attainment(point["trials"], attainment)
    for point in points:
        if not point["passes"]:
            break
        selected = point
    if selected is None:
        return {"slo_profile": None, "transfer_profile": None, "selected_rate": None}
    trials = selected["trials"]
    rps = min(
        selected["offered_rps"],
        *(trial["benchmark"]["request_goodput"] for trial in trials),
    )
    measured = min(trial["finished_at_ms"] for trial in trials)
    transfers = None
    if all(trial["kv_bytes"] for trial in trials):
        transfers = {
            # Include probe/warmup traffic in the numerator. Benchmark successes
            # alone in the denominator keep this estimate conservative.
            "bytes_per_request": max(
                math.ceil(trial["kv_bytes"] / trial["benchmark"]["completed"])
                for trial in trials
            ),
            # Calculate over the whole metrics interval, including idle setup.
            "observed_bytes_per_second": min(
                int(trial["kv_bytes"] / trial["metrics_interval_s"]) for trial in trials
            ),
        }
    return {
        "selected_rate": selected["offered_rps"],
        "slo_profile": {"rps": rps, "valid_until_ms": measured + valid_for_ms},
        "transfer_profile": transfers,
    }


def run_trial(
    args: argparse.Namespace, rate: float, seed: int, directory: Path
) -> dict:
    report = directory / "trial.json"
    options = [
        "--backend",
        "openai",
        "--base-url",
        args.url,
        "--endpoint",
        "/v1/completions",
        "--model",
        args.model,
        "--served-model-name",
        args.served_model,
        "--dataset-name",
        "random",
        "--random-input-len",
        str(args.input_tokens),
        "--random-output-len",
        str(args.output_tokens),
        "--random-range-ratio",
        "0",
        "--num-prompts",
        str(args.prompts),
        "--num-warmups",
        str(args.warmups),
        "--seed",
        str(seed),
        "--request-rate",
        str(rate),
        "--burstiness",
        "inf",
        "--ignore-eos",
        "--percentile-metrics",
        "ttft,tpot,itl,e2el",
        "--metric-percentiles",
        "50,99",
        "--goodput",
        f"ttft:{args.ttft_ms}",
        f"tpot:{args.tpot_ms}",
        "--disable-tqdm",
        "--save-result",
        "--save-detailed",
        "--result-dir",
        str(directory),
        "--result-filename",
        report.name,
    ]
    start = time.monotonic()
    kv_before = transferred_bytes(args.metrics_url)
    report.unlink(missing_ok=True)
    if args.in_process:
        run_in_process(options)
    else:
        subprocess.run([args.vllm or "vllm", "bench", "serve", *options], check=True)
    kv_after = transferred_bytes(args.metrics_url)
    interval = time.monotonic() - start
    finished = time.time_ns() // 1_000_000
    if kv_before is not None and kv_after < kv_before:
        raise ValueError("NIXL transfer counter reset during the benchmark")
    benchmark = json.loads(report.read_text(), parse_constant=lambda token: token)
    benchmark.pop("generated_texts", None)
    arrivals = benchmark["num_prompts"]
    if (
        arrivals != args.prompts
        or benchmark["completed"] + benchmark["failed"] != arrivals
    ):
        raise ValueError("benchmark did not account for every arrival")
    good = benchmark["request_goodput"] * benchmark["duration"]
    good_requests = round(good)
    if (
        not math.isclose(good, good_requests)
        or not 0 <= good_requests <= benchmark["completed"]
    ):
        raise ValueError("benchmark goodput is inconsistent with its completion counts")
    return {
        "seed": seed,
        "arrivals": arrivals,
        "good_requests": good_requests,
        "finished_at_ms": finished,
        "metrics_interval_s": interval,
        "kv_bytes": None if kv_before is None else kv_after - kv_before,
        "benchmark": benchmark,
    }


def run_in_process(options: list[str]) -> None:
    from vllm.benchmarks import serve
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = FlexibleArgumentParser()
    serve.add_cli_args(parser)
    try:
        serve.main(parser.parse_args(options))
    finally:
        # vLLM freezes the GC heap during a trial. Release it between trials.
        gc.unfreeze()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    runner = parser.add_mutually_exclusive_group()
    runner.add_argument("--vllm", help="external vLLM CLI executable (default: vllm)")
    runner.add_argument(
        "--in-process",
        action="store_true",
        help="run timed trials in this calibration process",
    )
    parser.add_argument("--url", required=True, help="OpenAI HTTP origin")
    parser.add_argument(
        "--model", required=True, help="model artifact used to load the tokenizer"
    )
    parser.add_argument("--served-model", required=True)
    parser.add_argument(
        "--placement", required=True, help="hardware, layout and transport label"
    )
    parser.add_argument("--input-tokens", type=int, required=True)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--rates", type=float, nargs="+", required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument(
        "--warmup-decode-widths",
        type=int,
        nargs="+",
        default=[],
        help="after the serial warmups, inject a prompt into each decoding wave",
    )
    parser.add_argument("--ttft-ms", type=float, required=True)
    parser.add_argument("--tpot-ms", type=float, required=True)
    parser.add_argument("--attainment", type=float, default=0.95)
    parser.add_argument(
        "--stop-at-first-miss",
        action="store_true",
        help="stop after the first rate whose trials fail the SLO attainment target",
    )
    parser.add_argument(
        "--valid-for", type=int, default=1800, help="profile lifetime in seconds"
    )
    parser.add_argument(
        "--metrics-url", help="dedicated decoder's vLLM Prometheus metrics URL"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        args.input_tokens < 1
        or args.output_tokens < 2
        or args.prompts < 1
        or args.warmups < 0
    ):
        parser.error("require prompts, input tokens, and at least two output tokens")
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.rates)) != len(
        args.rates
    ):
        parser.error("seeds and offered rates must be unique")
    if not all(math.isfinite(rate) and rate > 0 for rate in args.rates):
        parser.error("rates must be finite and positive")
    if not all(math.isfinite(slo) and slo > 0 for slo in (args.ttft_ms, args.tpot_ms)):
        parser.error("TTFT and TPOT targets must be finite and positive")
    if not 0 < args.attainment <= 1 or args.valid_for < 1:
        parser.error("attainment and profile lifetime must be positive")
    if any(width < 1 for width in args.warmup_decode_widths):
        parser.error("warmup decode widths must be positive")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="plait-bench-", dir=args.output.parent
    ) as work:
        directory = Path(work)
        warmup = None
        timed = args
        if args.warmup_decode_widths:
            from .warmup import run_warmup

            warmup = run_warmup(args)
            timed = argparse.Namespace(**(vars(args) | {"warmups": 0}))
        points = []
        for rate in sorted(args.rates):
            trials = [run_trial(timed, rate, seed, directory) for seed in args.seeds]
            points.append({"offered_rps": rate, "trials": trials})
            if args.stop_at_first_miss and not meets_attainment(
                trials, args.attainment
            ):
                break
        result = {
            "placement": args.placement,
            "workload": {
                "model": args.model,
                "served_model": args.served_model,
                "input_tokens": args.input_tokens,
                "output_tokens": args.output_tokens,
                "prompts": args.prompts,
                "warmups": args.warmups,
                "warmup_decode_widths": args.warmup_decode_widths,
                "ttft_ms": args.ttft_ms,
                "tpot_ms": args.tpot_ms,
                "attainment": args.attainment,
            },
            "points": points,
            **summarize(points, args.attainment, args.valid_for * 1000),
        }
        if warmup is not None:
            result["warmup"] = warmup
        pending = directory / "profile.json"
        pending.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        pending.replace(args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "placement",
                    "selected_rate",
                    "slo_profile",
                    "transfer_profile",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
