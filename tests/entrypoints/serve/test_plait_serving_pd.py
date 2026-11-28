# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright (c) 2026 Huanbing
"""Regression tests for versioned serving, drain, and route replacement."""

import asyncio
import contextlib
import http.client
import json
import os
import socket
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import product
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm.entrypoints.serve.plait_serving import (
    aggregate,
    benchmark,
    disaggregate,
    mode,
    warmup,
)
from vllm.entrypoints.serve.plait_serving import group as group_tools
from vllm.entrypoints.serve.plait_serving.calibration import (
    Calibrator,
    Run,
    Settings,
    Target,
)
from vllm.entrypoints.serve.plait_serving.group import (
    ADMISSION_HEADER,
    CHECKPOINT_ADMISSION_HEADER,
    HANDOFF_HEADER,
    KV_RESUME_HEADER,
    RELEASE_KV_PATH,
    VERSION_HEADER,
    ControlError,
    Endpoint,
    Gate,
    check_backend_port,
    group_proxy,
    record_host_stop,
    run,
    verify_backend_capabilities,
)
from vllm.entrypoints.serve.plait_serving.launch import (
    GREEDY_KV_RESUME,
    SEEDED_KV_RESUME,
    host_command,
    resident_spec,
)
from vllm.entrypoints.serve.plait_serving.load import Load
from vllm.entrypoints.serve.plait_serving.mode_state import ProfileStore, fingerprint
from vllm.entrypoints.serve.plait_serving.replay import (
    RECOVERY_HEADER,
    RECOVERY_MODE,
    RESUME_KV_MODE,
    RESUME_PREFIX_MODE,
    sse,
)

PREFILL, DECODE, SPARE = (f"{number:032x}@1" for number in range(1, 4))
TOKEN = "f" * 32
COMPLETIONS = "/v1/completions"


class LaunchTest(unittest.TestCase):
    def test_amd_kfd_release_uses_the_device_identity_captured_before_launch(self):
        group, host = "a" * 32, "b" * 32
        node = list(bytes.fromhex(host))
        version = {"group": list(bytes.fromhex(group)), "epoch": 7}
        grants = [
            {
                "version": version,
                "device": {"node": node, "ordinal": 1},
                "incarnation": 1,
                "fence": 2,
            }
        ]
        monitor = {"driver": "amd_kfd", "program": "/vendor/amd-smi"}
        args = SimpleNamespace(
            control_bin="plait-node",
            control_listen=None,
            control_peer=[],
            group_id=group,
            group_epoch=7,
            host_id=host,
            role="decode",
            model="model",
            tp=1,
            port_base=8000,
            backend_port=9000,
            proxy_host="127.0.0.1",
            kv_lease_seconds=30,
            startup_timeout=1,
            drain_timeout=1,
            verify_release=None,
            release_driver=monitor,
            replicated_egress_bytes=0,
        )
        spec = {
            "api_host": host,
            "master_addr": "192.0.2.2",
            "master_port": 9008,
            "hosts": {
                host: {
                    "address": "192.0.2.2",
                    "argv": ["vllm"],
                    "model_path": "/model",
                    "device_env": "HIP_VISIBLE_DEVICES",
                }
            },
        }
        digest = [2] * 32
        control = Mock()
        control.request.return_value = {
            "group": {
                "phase": "Preparing",
                "version": version,
                "role": "Decode",
                "model": "model",
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "launch_digest": digest,
                "grants": grants,
            }
        }
        control.execute.return_value = {"Ok": {"Group": {"phase": "Preparing"}}}
        identities = [
            {
                "hip": 1,
                "management": 7,
                "kfd": 30,
                "node": 3,
                "render": 129,
                "bdf": "0000:02:00.0",
            }
        ]
        proof_record = {
            "monitor": monitor,
            "host": node,
            "version": version,
            "grants": grants,
        }
        calls = []

        def verify(argv, *, input, env, **_kwargs):
            self.assertNotIn("HIP_VISIBLE_DEVICES", env)
            action = argv[-1]
            calls.append(action)
            if action == "capture-host-release":
                process.assert_not_called()
                self.assertEqual(json.loads(input), proof_record)
                return SimpleNamespace(stdout=json.dumps(identities))
            stopped.assert_called_once()
            self.assertEqual(
                json.loads(input), {**proof_record, "identities": identities}
            )
            return SimpleNamespace(returncode=0)

        with (
            patch.object(group_tools, "Control", return_value=control),
            patch.object(group_tools, "group_proxy"),
            patch.object(group_tools, "check_backend_port"),
            patch.object(group_tools.subprocess, "run", side_effect=verify),
            patch.object(group_tools.subprocess, "Popen") as process,
            patch.object(
                group_tools, "stop_process_group", return_value=True
            ) as stopped,
            patch.object(group_tools, "record_host_stop") as retire,
        ):
            process.return_value.poll.return_value = 0
            run(
                args,
                ["vllm", "serve", "model"],
                digest,
                {
                    "HIP_VISIBLE_DEVICES": "5",
                    "VLLM_PLAIT_PEER_REPLICATED_EGRESS_BYTES": "393216",
                },
                lambda _: True,
                lambda: False,
                host_launch_spec=spec,
            )
        self.assertEqual(calls, ["capture-host-release", "verify-host-release"])
        self.assertEqual(process.call_args.kwargs["env"]["HIP_VISIBLE_DEVICES"], "1")
        self.assertEqual(
            process.call_args.kwargs["env"]["VLLM_PLAIT_PEER_REPLICATED_EGRESS_BYTES"],
            "0",
        )
        retire.assert_called_once()

    def test_backend_probe_accepts_a_drained_port_and_rejects_an_active_listener(self):
        with socket.socket() as listener:
            listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            port = listener.getsockname()[1]
            with self.assertRaises(OSError):
                check_backend_port(port)
            with (
                socket.create_connection(listener.getsockname(), timeout=2) as client,
                listener.accept()[0] as server,
            ):
                server.settimeout(2)
                server.shutdown(socket.SHUT_WR)
                self.assertEqual(client.recv(1), b"")
                client.shutdown(socket.SHUT_WR)
                self.assertEqual(server.recv(1), b"")

        check_backend_port(port)

    def test_single_device_hosted_pd_keeps_vllm_default_executor(self):
        host = "b" * 32
        recipe = {
            "api_host": list(bytes.fromhex(host)),
            "master_port": 9010,
            "hosts": [{"host": list(bytes.fromhex(host)), "address": "192.0.2.1"}],
        }
        profile = {
            "argv": ["vllm"],
            "model_path": "/local/model",
            "device_env": "CUDA_VISIBLE_DEVICES",
            "args": [],
            "env": {},
        }
        spec = resident_spec(recipe, profile, pd=True)
        command = ["vllm", "serve", "model", "--tensor-parallel-size", "1"]
        self.assertEqual(
            host_command(command, spec, host, 0, [1]),
            ["vllm", "serve", "/local/model", "--tensor-parallel-size", "1"],
        )
        self.assertIn(
            "--distributed-executor-backend", host_command(command, spec, host, 0, [2])
        )
        aggregate = resident_spec(recipe, profile)
        self.assertEqual(
            host_command(command, aggregate, host, 0, [1]),
            ["vllm", "serve", "/local/model", "--tensor-parallel-size", "1"],
        )
        with self.assertRaisesRegex(
            ValueError, "supplies the KV connector and block size"
        ):
            resident_spec(recipe, {**profile, "args": ["--block_size=256"]}, pd=True)

    def test_child_uses_the_claimed_version_even_if_parent_has_another_epoch(self):
        group, host = "a" * 32, "b" * 32
        version = {"group": list(bytes.fromhex(group)), "epoch": 7}
        digest = [2] * 32
        args = SimpleNamespace(
            control_bin="plait-node",
            control_listen="127.0.0.1:0",
            control_peer=[],
            group_id=group,
            group_epoch=7,
            host_id=host,
            role="aggregated",
            model="model",
            tp=1,
            port_base=8000,
            backend_port=9000,
            proxy_host="127.0.0.1",
            kv_lease_seconds=30,
            startup_timeout=1,
            verify_release=None,
        )
        control = Mock()
        control.request.return_value = {
            "group": {
                "phase": "Preparing",
                "version": version,
                "role": "Aggregated",
                "model": "model",
                "tensor_parallel": 1,
                "pipeline_parallel": 1,
                "launch_digest": digest,
                "grants": [
                    {"device": {"node": list(bytes.fromhex(host)), "ordinal": 2}}
                ],
            }
        }
        control.execute.return_value = {"Ok": {"Group": {"phase": "Preparing"}}}
        env = {"PLAIT_TP_GROUP_VERSION": f"{group}@6"}
        command = ["vllm", "serve", "model"]
        with (
            patch.object(group_tools, "Control", return_value=control),
            patch.object(group_tools, "group_proxy"),
            patch.object(group_tools.socket, "socket"),
            patch.object(group_tools.subprocess, "Popen") as process,
            patch.object(group_tools, "stop_process_group", return_value=True),
        ):
            process.return_value.poll.return_value = 0
            run(args, command, digest, env, lambda _: True, lambda: False)

        claim = control.execute.call_args_list[0].args[0]["ClaimLaunch"]
        self.assertEqual(claim["version"], version)
        self.assertEqual(
            process.call_args.kwargs["env"]["PLAIT_TP_GROUP_VERSION"], f"{group}@7"
        )
        self.assertEqual(env["PLAIT_TP_GROUP_VERSION"], f"{group}@6")

    def test_follower_uses_local_grants_and_reports_readiness_from_its_child(self):
        group, host, head = "a" * 32, "b" * 32, "c" * 32
        version = {"group": list(bytes.fromhex(group)), "epoch": 3}
        digest = [2] * 32
        args = SimpleNamespace(
            control_bin="plait-node",
            control_listen=None,
            control_peer=[],
            group_id=group,
            group_epoch=3,
            host_id=host,
            role="aggregated",
            model="model",
            tp=2,
            backend_port=9000,
            startup_timeout=1,
            verify_release=None,
        )
        spec = {
            "api_host": head,
            "master_addr": "192.0.2.1",
            "master_port": 9008,
            "hosts": {
                head: {"address": "192.0.2.1"},
                host: {
                    "address": "192.0.2.2",
                    "argv": ["vllm"],
                    "model_path": "/local/model",
                    "device_env": "ASCEND_RT_VISIBLE_DEVICES",
                },
            },
        }
        control = Mock()
        control.request.return_value = {
            "group": {
                "phase": "Preparing",
                "version": version,
                "role": "Aggregated",
                "model": "model",
                "tensor_parallel": 2,
                "pipeline_parallel": 2,
                "launch_digest": digest,
                "grants": [
                    {"device": {"node": list(bytes.fromhex(node)), "ordinal": ordinal}}
                    for node, ordinal in ((host, 6), (head, 1), (host, 2), (host, 8))
                ],
            }
        }
        probe = Mock()
        stopped = False

        def execute(action, **_kwargs):
            nonlocal stopped
            if "ObserveHostLaunch" in action:
                stopped = True
            return {"Ok": {"Group": {"phase": "Preparing"}}}

        control.execute.side_effect = execute
        child = Mock()
        child.poll.return_value = None

        def start(_command, *, pass_fds, **_kwargs):
            os.write(pass_fds[0], signal)
            return child

        for signal in (b"x", b"1"):
            with (
                self.subTest(signal=signal),
                patch.object(group_tools, "Control", return_value=control) as factory,
                patch.object(group_tools, "group_proxy") as proxy,
                patch.object(
                    group_tools.subprocess, "Popen", side_effect=start
                ) as process,
                patch.object(group_tools, "stop_process_group", return_value=True),
                patch.object(group_tools.time, "sleep"),
            ):
                control.execute.reset_mock()

                def call():
                    return run(
                        args,
                        ["vllm", "serve", "model"],
                        digest,
                        {"CUDA_VISIBLE_DEVICES": "0"},
                        probe,
                        lambda: stopped,
                        host_launch_spec=spec,
                    )

                if signal == b"x":
                    with self.assertRaisesRegex(ControlError, "readiness pipe"):
                        call()
                else:
                    self.assertEqual(call(), 0)
                calls = [
                    next(iter(item.args[0])) for item in control.execute.call_args_list
                ]
                self.assertEqual(calls[0], "ClaimHostLaunch")
                self.assertEqual("HostLaunchReady" in calls, signal == b"1")
                factory.assert_called_once_with(
                    "plait-node", [], "192.0.2.2:0", seeds=()
                )
                proxy.assert_not_called()
                probe.assert_not_called()
                command = process.call_args.args[0]
                self.assertEqual(
                    command[command.index("--plait-tp-layout") + 1 :],
                    ["1", "3", "--headless"],
                )
                self.assertEqual(
                    command[command.index("--pipeline-parallel-size") + 1], "2"
                )
                child_env = process.call_args.kwargs["env"]
                self.assertEqual(child_env["ASCEND_RT_VISIBLE_DEVICES"], "2,6,8")
                self.assertNotIn("CUDA_VISIBLE_DEVICES", child_env)

    def test_host_release_uses_the_committed_order_and_full_worker_set(self):
        control = Mock()
        control.execute.side_effect = [
            {"Err": "WorkersStillRunning"},
            {"Ok": {"Group": {"phase": "Draining"}}},
            {"Err": "WorkersStillRunning"},
        ]
        identity = {"version": {"group": [1] * 16, "epoch": 1}, "host": [2] * 16}
        workers = [{"node": [ordinal] * 16, "ordinal": 0} for ordinal in (2, 3)]
        with patch.object(group_tools.time, "sleep"):
            record_host_stop(control, identity, workers, 0)
        actions = [call.args[0] for call in control.execute.call_args_list]
        self.assertEqual(actions[:2], [{"HostLaunchStopped": identity}] * 2)
        self.assertEqual(
            actions[2], {"Retire": {"version": identity["version"], "workers": workers}}
        )


class CalibrationTest(unittest.TestCase):
    def test_builtin_runner_parses_each_seed_with_the_vllm_benchmark(self):
        settings = Settings("/model", 64, 8, (2,), (7, 23), 2000, 500, prompts=2)
        endpoint = Endpoint("127.0.0.1", 8000, "01" * 16 + "@1", "aggregated")
        target = Target(1, frozenset(), ((endpoint,),), 1)

        def trial(args):
            Path(args.result_dir, args.result_filename).write_text(
                json.dumps(
                    {
                        "num_prompts": args.num_prompts,
                        "completed": 2,
                        "failed": 0,
                        "request_goodput": 2.0,
                        "duration": 1.0,
                    }
                )
            )

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capacity.json"
            command = settings.command("http://127.0.0.1:7000", "test", target, output)
            with (
                patch("sys.argv", command[2:]),
                patch("vllm.benchmarks.serve.main", side_effect=trial) as run,
                patch.object(benchmark.subprocess, "run") as spawn,
            ):
                benchmark.main()
            spawn.assert_not_called()
            self.assertEqual(
                [call.args[0].seed for call in run.call_args_list], [7, 23]
            )
            for call in run.call_args_list:
                args = call.args[0]
                self.assertEqual(args.base_url, "http://127.0.0.1:7000")
                self.assertEqual(args.served_model_name, "test")
                self.assertEqual(args.request_rate, 2)
                self.assertEqual(args.goodput, ["ttft:2000.0", "tpot:500.0"])
            self.assertEqual(json.loads(output.read_text())["slo_profile"]["rps"], 2)

    def test_calibration_keeps_an_explicit_benchmark_executable(self):
        settings = Settings("/model", 16, 8, (2,), (1,), 100, 20, vllm="/custom/vllm")
        target = Target(1, frozenset(), (), 1)
        command = settings.command("http://localhost:7000", "test", target, Path("out"))
        self.assertNotIn("--in-process", command)
        self.assertEqual(command[command.index("--vllm") + 1], "/custom/vllm")

    def test_a_late_writer_preserves_the_newer_measurement_of_the_same_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ProfileStore(Path(directory) / "capacity.json")
            identity = {
                "context": "settings",
                "versions": ["layout"],
                "binding": "routes",
                "manifest": "grants",
            }
            latest = {**identity, "slo_profile": {"rps": 2, "valid_until_ms": 4000}}
            delayed = {**identity, "slo_profile": {"rps": 1, "valid_until_ms": 2000}}
            store.save(latest, 1000)
            self.assertEqual(store.save(delayed, 1000), latest)
            self.assertEqual(store.read(), [latest])

    def test_successful_process_observed_after_deadline_cannot_publish(self):
        settings = Settings("/model", 16, 8, (2,), (1,), 100, 20)
        calibrator = Calibrator(Mock())
        cleanup = Mock()
        process = Mock()
        process.poll.return_value = 0
        target = Target(1, frozenset(), (), 1)
        calibrator.run = Run(target, process, cleanup, Path("unused"), 0.0)
        with self.assertRaisesRegex(ValueError, "timed out"):
            calibrator.poll(settings)
        self.assertIsNone(calibrator.run)
        cleanup.close.assert_called_once()

    def test_requires_repeatable_slo_and_does_not_skip_an_overloaded_rate(self):

        def trial(good, rps, end):
            return {
                "good_requests": good,
                "arrivals": 100,
                "finished_at_ms": end,
                "metrics_interval_s": 10,
                "kv_bytes": 2400,
                "benchmark": {"completed": 100, "request_goodput": rps},
            }

        points = [
            {"offered_rps": 8, "trials": [trial(98, 8.1, 100), trial(97, 8.2, 200)]},
            {"offered_rps": 16, "trials": [trial(98, 15.4, 400), trial(93, 14.1, 500)]},
            {"offered_rps": 24, "trials": [trial(97, 23.1, 600), trial(96, 22.9, 800)]},
        ]
        result = benchmark.summarize(points, 0.95, 1000)
        self.assertEqual(result["selected_rate"], 8)
        self.assertEqual(result["slo_profile"], {"rps": 8, "valid_until_ms": 1100})
        self.assertEqual(result["transfer_profile"]["bytes_per_request"], 24)
        self.assertEqual([point["passes"] for point in points], [True, False, True])

    def test_online_calibration_finishes_the_first_failing_rate_and_stops(self):
        settings = Settings(
            "/model", 16, 8, (4, 1, 2), (7, 23), 100, 20, attainment=0.9
        )
        endpoint = Endpoint("127.0.0.1", 8000, "01" * 16 + "@1", "aggregated")
        target = Target(1, frozenset(), ((endpoint,),), 3)

        def trial(args, rate, seed, directory):
            return {
                "seed": seed,
                "good_requests": 8 if (rate, seed) == (2, 7) else 10,
                "arrivals": 10,
                "finished_at_ms": 1000,
                "kv_bytes": None,
                "benchmark": {"request_goodput": rate},
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capacity.json"
            command = settings.command("http://127.0.0.1:7000", "test", target, output)
            with (
                patch("sys.argv", command[2:]),
                patch.object(benchmark, "run_trial", side_effect=trial) as run,
            ):
                benchmark.main()
            self.assertEqual(
                [call.args[1:3] for call in run.call_args_list],
                [(1, 7), (1, 23), (2, 7), (2, 23)],
            )
            report = json.loads(output.read_text())
            self.assertEqual(report["selected_rate"], 1)
            self.assertEqual(
                [point["passes"] for point in report["points"]], [True, False]
            )

    def test_wave_preparation_runs_once_outside_the_timed_trials(self):
        settings = Settings(
            "/model", 64, 8, (2, 4), (7, 23), 2000, 500,
            warmup_decode_widths=(2, 2, 3, 4),
        )
        endpoint = Endpoint("127.0.0.1", 8000, "01" * 16 + "@1", "aggregated")
        target = Target(1, frozenset(), ((endpoint,),), 3)
        phases = []
        prepared = {"requests": 18, "seed": 0, "elapsed_s": 20}

        def prepare(args):
            self.assertEqual(args.warmup_decode_widths, [2, 2, 3, 4])
            phases.append("warmup")
            return prepared

        def trial(args, rate, seed, directory):
            self.assertEqual(args.warmups, 0)
            phases.append((rate, seed))
            return {
                "arrivals": 16,
                "good_requests": 16,
                "finished_at_ms": 1000,
                "kv_bytes": None,
                "benchmark": {"request_goodput": rate},
            }

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "capacity.json"
            command = settings.command("http://127.0.0.1:7000", "test", target, output)
            with (
                patch("sys.argv", command[2:]),
                patch.object(warmup, "run_warmup", side_effect=prepare),
                patch.object(benchmark, "run_trial", side_effect=trial),
            ):
                benchmark.main()
            self.assertEqual(phases, ["warmup", (2, 7), (2, 23), (4, 7), (4, 23)])
            report = json.loads(output.read_text())
            self.assertEqual(report["warmup"], prepared)
            self.assertEqual(report["selected_rate"], 4)


class WarmupWaveTest(unittest.IsolatedAsyncioTestCase):
    async def test_inject_waits_for_both_first_tokens(self):
        streams = asyncio.Queue()
        complete = asyncio.Event()
        inserted = asyncio.Event()

        async def submit(callback):
            if callback is None:
                inserted.set()
            else:
                streams.put_nowait(callback)
                await complete.wait()

        run = asyncio.create_task(warmup.run_waves([2], 0, submit))
        try:
            first = await asyncio.wait_for(streams.get(), 1)
            second = await asyncio.wait_for(streams.get(), 1)
            first()
            await asyncio.sleep(0)
            self.assertFalse(inserted.is_set())
            second()
            await asyncio.wait_for(inserted.wait(), 1)
            complete.set()
            await asyncio.wait_for(run, 1)
        finally:
            run.cancel()
            await asyncio.gather(run, return_exceptions=True)

    async def test_failed_stream_cancels_the_rest_of_the_wave(self):
        started = asyncio.Event()
        cancelled = asyncio.Event()

        async def submit(callback):
            self.assertIsNotNone(callback)
            if started.is_set():
                raise RuntimeError("stream failed before its first token")
            started.set()
            try:
                await asyncio.Future()
            finally:
                cancelled.set()

        with self.assertRaisesRegex(RuntimeError, "stream failed"):
            await asyncio.wait_for(warmup.run_waves([2], 0, submit), 1)
        self.assertTrue(cancelled.is_set())


def native_deployment(group: str, host: str, port: int) -> dict:
    return {
        "group_id": group,
        "model_alias": "test",
        "seeds": ["127.0.0.1:7950"],
        "hosts": {host: "192.0.2.1"},
        "api_port_base": port,
        "rendezvous_port_base": port + 100,
        "epoch_ports": 2,
    }


class DirectoryTest(unittest.TestCase):
    def test_discovery_derives_new_origins_from_grants_and_admits_a_complete_path(self):
        groups = [(bytes([index] * 16)).hex() for index in range(1, 4)]
        keys = [list(bytes.fromhex(group)) for group in groups]
        active = [1, None, None]
        phases = ["Active", "Ready", "Preparing"]
        host = [7] * 16

        def request(message):
            if message["request"] == "apply":
                return {
                    "result": {
                        "Ok": {"ActiveSet": list(zip(keys, active, strict=True))}
                    }
                }
            if message["request"] == "placement":
                key = message["group"]
                version = {"group": key, "epoch": 1}
                return {"placement": {"latest": [version, phases[keys.index(key)]]}}
            version = message["version"]
            phase = phases[keys.index(version["group"])]
            return {
                "group": {
                    "version": version,
                    "phase": phase,
                    "grants": [
                        {"device": {"node": host, "ordinal": 0}},
                    ],
                }
            }

        control = Mock()
        control.request.side_effect = request
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "routes.json"
            sources = {}
            for index, group in enumerate(groups):
                path = config.with_name(f"group-{index}.json")
                path.write_text(
                    json.dumps(
                        native_deployment(group, bytes(host).hex(), 8100 + 4 * index)
                    )
                )
                sources[group] = path.name
            config.write_text(
                json.dumps(
                    {
                        "model": "test",
                        "revision": 1,
                        "paths": [],
                        "discover": {
                            "deployments": sources,
                            "aggregated": [groups[0]],
                            "paths": [
                                {
                                    "prefill": groups[1],
                                    "decode": groups[2],
                                    "kv_host": "192.0.2.2",
                                    "kv_port": 7952,
                                }
                            ],
                        },
                    }
                )
            )
            directory = disaggregate.NativeDirectory("test", control, config)
            directory.refresh()
            self.assertEqual(len(directory.mode_snapshot().routes), 1)
            phases[2] = "Ready"
            directory.refresh()
            view = directory.mode_snapshot()
            self.assertEqual(len(view.routes), 2)
            candidate_routes = tuple(path for path in view.routes if len(path) == 2)
            target = Target(view.revision, view.active, candidate_routes, 2)
            _, [candidate] = directory.calibration_routes(target)
            self.assertEqual(candidate.decode.port, 8108)
            [incumbent] = directory.routes()[1]
            self.assertEqual(incumbent.port, 8100)
            active[:] = [None, 1, 1]
            phases[:] = ["Draining", "Active", "Active"]
            directory.refresh()
            with self.assertRaisesRegex(ValueError, "expired"):
                directory.calibration_routes(target)
            [route] = directory.routes()[1]
            self.assertEqual(
                (route.prefill.port, route.decode.port, route.kv_port),
                (8104, 8108, 7952),
            )

    def test_native_pd_requires_both_ends_of_one_active_set(self):
        groups = [bytes([index] * 16).hex() for index in (1, 2, 3)]
        versions = [f"{group}@1" for group in groups]
        document = {
            "model": "test",
            "revision": 2,
            "aggregated": [{"url": "http://127.0.0.1:8100", "version": versions[0]}],
            "paths": [
                {
                    "prefill": {
                        "url": "http://127.0.0.1:8101",
                        "version": versions[1],
                        "kv_host": "127.0.0.1",
                        "kv_port": 7952,
                    },
                    "decode": {"url": "http://127.0.0.1:8102", "version": versions[2]},
                }
            ],
        }
        keys = [list(bytes.fromhex(group)) for group in groups]
        control = Mock()
        control.request.side_effect = [
            {"result": {"Ok": {"ActiveSet": list(zip(keys, epochs, strict=True))}}}
            for epochs in ((1, None, None), (None, 1, 1), (None, 1, None))
        ]
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "routes.json"
            config.write_text(json.dumps(document))
            directory = disaggregate.NativeDirectory("test", control, config)
            directory.refresh()
            self.assertEqual(
                [route.role for route in directory.routes()[1]], ["aggregated"]
            )
            directory.refresh()
            [route] = directory.routes()[1]
            self.assertEqual(route.prefill.version, versions[1])
            self.assertEqual(route.decode.version, versions[2])
            self.assertFalse(route.prefill_lease)
            directory.refresh()
            self.assertEqual(directory.routes(), ("test", []))
            request = control.request.call_args.args[0]
            self.assertEqual(
                request["operation"]["action"], {"ObserveActiveSet": {"groups": keys}}
            )
            config.write_text(json.dumps({**document, "revision": 1}))
            directory.refresh()
            self.assertEqual(control.request.call_count, 3)
            config.write_text(json.dumps({**document, "paths": []}))
            with self.assertRaisesRegex(ValueError, "new revision"):
                directory.refresh()

    def test_native_route_waits_for_the_active_versions_binding(self):
        group = bytes(16)

        def view(epoch):
            return {
                "result": {
                    "Ok": {
                        "version": {"group": list(group), "epoch": epoch},
                        "phase": "Active",
                    }
                }
            }

        control = Mock()
        control.request.side_effect = [view(1), view(2), view(1), view(2)]
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder) / "routes.json"
            config.write_text('{"1":"http://127.0.0.1:8001"}')
            directory = aggregate.NativeGroupDirectory("test", group, control, config)
            directory.refresh()
            self.assertEqual(directory.endpoint().epoch, 1)
            directory.refresh()
            self.assertIsNone(directory.endpoint())
            directory.refresh()
            self.assertIsNone(directory.endpoint())
            config.write_text('{"2":"http://127.0.0.1:8002"}')
            directory.refresh()
            self.assertEqual(directory.endpoint().epoch, 2)
            self.assertEqual(control.request.call_args.args[0]["request"], "observe")

    def test_aggregate_route_never_reopens_a_replaced_or_closed_epoch(self):
        group = bytes(16)

        def view(epoch, phase="Active"):
            return {
                "result": {
                    "Ok": {
                        "version": {"group": list(group), "epoch": epoch},
                        "phase": phase,
                        "role": "Aggregated",
                        "endpoint": {"http": "127.0.0.1:12345", "api_model": "test"},
                    }
                }
            }

        control = Mock()
        control.request.side_effect = [
            view(1),
            view(2),
            view(1),
            view(2, "Draining"),
            view(2),
            view(3),
            {"result": {"Err": {"NoActiveGroup": list(group)}}},
            view(3),
        ]
        directory = aggregate.ActiveGroupDirectory("test", group, control)
        for expected in (1, 2, 2, None, None, 3, None, None):
            directory.refresh()
            endpoint = directory.endpoint()
            self.assertEqual(
                endpoint.version if endpoint else None,
                (f"{group.hex()}@{expected}" if expected is not None else None),
            )

    def test_stale_replicas_cannot_restore_an_older_or_drained_path(self):
        def path(decode):
            return {
                "prefill": {
                    "url": "http://127.0.0.1:8100",
                    "version": PREFILL,
                    "kv_host": "127.0.0.1",
                    "kv_port": 9100,
                },
                "decode": {"url": "http://127.0.0.1:8101", "version": decode},
            }

        class Replica:
            replies = iter(
                [
                    (1, [path(DECODE)]),
                    (2, [path(SPARE)]),
                    (1, [path(DECODE)]),
                    (2, []),
                    (2, [path(SPARE)]),
                ]
            )

            def execute(self, _action, *, retry):
                return {"Ok": {"RoutesObserved": {"revision": 1}}}

            def request(self, _request):
                revision, paths = next(self.replies)
                return {
                    "directory": {"model": "test", "revision": revision, "paths": paths}
                }

        directory = disaggregate.ControlDirectory("test", Replica())
        for expected in ([DECODE], [SPARE], [SPARE], [], []):
            directory.refresh()
            self.assertEqual(
                [route.decode.version for route in directory.routes()[1]], expected
            )
        directory.until = 0
        with self.assertRaises(ValueError):
            directory.routes()


class ModePolicyTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.policy = Path(directory.name) / "policy.json"
        self.groups = tuple(bytes([index] * 16).hex() for index in (1, 2, 3, 4))
        self.versions = [f"{group}@1" for group in self.groups]
        aggregate, prefill, decode, standby = self.versions
        self.versions.append(f"{self.groups[0]}@2")
        self.routes = (
            (Endpoint("127.0.0.1", 8001, aggregate, "aggregated"),),
            (
                Endpoint("127.0.0.1", 8002, prefill, "prefill"),
                Endpoint("127.0.0.1", 8003, decode, "decode"),
            ),
            (Endpoint("127.0.0.1", 8004, standby, "aggregated"),),
            (Endpoint("127.0.0.1", 8005, self.versions[-1], "aggregated"),),
        )
        self.config = {
            "route_revision": 1,
            "window_seconds": 1,
            "stable_windows": 2,
            "horizon_seconds": 20,
            "cooldown_seconds": 2,
            "device_cost_requests_per_second": 1,
            "switch_loss_requests": 2,
            "minimum_gain_requests": 1,
            "modes": [
                self.profile(versions, devices, rps)
                for versions, devices, rps in (
                    ([aggregate], 1, 4),
                    ([prefill, decode], 2, 15),
                    ([self.versions[-1]], 1, 4),
                )
            ],
        }
        self.policy.write_text(json.dumps(self.config))
        self.clock = 0
        self.arrivals = 0
        self.active = frozenset([aggregate])
        self.ready = {prefill, decode}
        self.requests = []
        self.plans = []
        self.preparations = {}
        self.aborts = []
        self.tickets = {}
        self.device_counts = {}
        self.preview_reply = {"decision": "no_placement"}
        self.plan_reply = {"decision": "pending"}
        self.may_lose_reply = False
        self.control = Mock()
        self.control.next_operation_id.return_value = "operation-id"
        self.control.request.side_effect = self.request
        self.load = Mock()
        self.load.snapshot.side_effect = lambda: {
            "session": "ingress",
            "uptime_ms": self.clock * 1000,
            "arrivals": self.arrivals,
        }
        self.controller = mode.ModeController(self.policy, self.load, self.control)
        clock = SimpleNamespace(
            monotonic=lambda: self.clock, time_ns=lambda: self.clock * 10**9
        )
        self.enterContext(patch.object(mode, "time", clock))
        self.health = self.enterContext(
            patch.object(mode, "endpoint_ready", return_value=True)
        )

    def profile(self, versions, devices, rps, valid_until_ms=60_000):
        return {
            "versions": versions,
            "devices": devices,
            "slo_profile": {"rps": rps, "valid_until_ms": valid_until_ms},
        }

    def request(self, request):
        if request["request"] == "preparation":
            return {"preparation": self.preparations.get(bytes(request["group"]).hex())}
        if request["request"] == "plan_set":
            self.plans.append(request)
            if self.may_lose_reply:
                self.may_lose_reply = False
                raise mode.ControlError("reply lost")
            if "after_releasing" in request:
                return self.preview_reply
            return self.plan_reply
        if request["request"] == "view":
            version = request["version"]
            epoch = f"{bytes(version['group']).hex()}@{version['epoch']}"
            phase = (
                "Active"
                if epoch in self.active
                else ("Ready" if epoch in self.ready else "Preparing")
            )
            return {
                "group": {
                    "phase": phase,
                    "version": version,
                    "grants": [{}] * self.device_counts.get(epoch, 1),
                    "manifest": {"ticket": self.tickets.get(epoch, epoch)},
                }
            }
        action = request["operation"]["action"]
        if "AbortFailedSet" in action:
            self.aborts.append(request)
            if self.may_lose_reply:
                self.may_lose_reply = False
                raise mode.ControlError("reply lost")
            return {
                "result": {"Ok": {"AbortedSet": action["AbortFailedSet"]["versions"]}}
            }
        transition = action["SwitchActiveSet"]["transitions"]
        self.requests.append(request)
        if self.may_lose_reply:
            self.may_lose_reply = False
            raise mode.ControlError("reply lost")
        return {
            "result": {
                "Ok": {
                    "ActiveSet": [
                        [change["group"], change["target"]] for change in transition
                    ]
                }
            }
        }

    def snapshot(self):
        return mode.ModeView(1, self.routes, self.groups, self.active)

    def sample(self, arrivals, seconds=1):
        self.clock += seconds
        self.arrivals += arrivals
        self.controller.reconcile(self.snapshot(), self.snapshot)

    def changes(self):
        return self.requests[-1]["operation"]["action"]["SwitchActiveSet"][
            "transitions"
        ]

    def test_switches_up_then_returns_to_a_new_epoch_after_a_stable_low_load(self):
        self.sample(0, seconds=0)
        self.sample(10)
        self.assertEqual(self.requests, [])
        self.may_lose_reply = True
        self.sample(10)
        self.assertEqual(self.requests[0], self.requests[1])
        self.assertEqual(
            [(change["expected"], change["target"]) for change in self.changes()],
            [(1, None), (None, 1), (None, 1), (None, None)],
        )
        self.active = frozenset(self.versions[1:3])
        self.ready = {self.versions[-1]}
        self.sample(1)
        self.sample(12)
        self.sample(12)
        self.assertEqual(len(self.requests), 2)
        self.sample(1)
        self.assertEqual(len(self.requests), 2)
        self.sample(1)
        self.assertEqual(len(self.requests), 3)
        self.assertEqual(
            [(change["expected"], change["target"]) for change in self.changes()],
            [(None, 2), (1, None), (1, None), (None, None)],
        )

    def test_requires_fresh_capacity_complete_routes_and_ready_http(self):
        self.sample(0, seconds=0)
        self.health.side_effect = lambda endpoint, timeout: endpoint.role != "decode"
        self.sample(12)
        self.sample(12)
        self.assertEqual(self.requests, [])

        self.health.side_effect = None
        self.config["modes"].append(self.profile([self.versions[1]], 1, 50))
        self.config["modes"][1]["slo_profile"]["valid_until_ms"] = 2000
        self.policy.write_text(json.dumps(self.config))
        self.sample(12)
        self.sample(12)
        self.sample(12)
        self.assertEqual(self.requests, [])

    def test_rechecks_the_observation_before_submitting(self):
        self.sample(0, seconds=0)
        self.sample(12)
        self.arrivals += 12
        self.clock += 1
        old = self.snapshot()
        self.active = frozenset([self.versions[0], self.versions[3]])
        self.controller.reconcile(old, self.snapshot)
        self.assertEqual(self.requests, [])

    def configure_provision(self):
        groups = []
        for group in self.groups[1:3]:
            deployment = self.policy.with_name(f"{group}.json")
            deployment.write_text(json.dumps(native_deployment(group, group, 8100)))
            groups.append({"deployment": deployment.name, "workload": {"widths": [1]}})
        self.config["provision"] = [
            {
                "groups": groups,
                "devices": 2,
                "slo_profile": {"rps": 15, "valid_until_ms": 60_000},
            }
        ]
        self.incumbent = self.config["modes"][0]
        self.target = self.config["modes"][1]
        self.config["modes"] = [self.incumbent]
        self.policy.write_text(json.dumps(self.config))

    def test_provisions_as_a_set_and_switches_after_exact_profile_and_readiness(self):
        self.configure_provision()
        self.ready.clear()
        self.plan_reply = {
            "decision": "prepare_set",
            "result": {
                "Ok": {
                    "PreparedSet": [
                        {"group": list(bytes.fromhex(group)), "epoch": 1}
                        for group in self.groups[1:3]
                    ]
                }
            },
        }
        self.may_lose_reply = True
        self.sample(0, seconds=0)
        self.sample(12)
        self.assertEqual(self.plans, [])
        self.sample(12)
        self.assertEqual(len(self.plans), 2)
        self.assertEqual(self.plans[0], self.plans[1])
        plan = self.plans[0]
        self.assertEqual(
            plan["guards"],
            [
                [list(bytes.fromhex(group)), 1 if index == 0 else None]
                for index, group in enumerate(self.groups)
            ],
        )
        self.assertEqual(
            [
                (bytes(member["group"]).hex(), bytes(member["hosts"][0]).hex())
                for member in plan["groups"]
            ],
            [(group, group) for group in self.groups[1:3]],
        )

        self.ready.update(self.versions[1:3])
        self.sample(12)
        self.assertEqual(self.requests, [])
        self.config["modes"].append(self.target)
        self.policy.write_text(json.dumps(self.config))
        self.ready.remove(self.versions[2])
        self.sample(12)
        self.sample(12)
        self.sample(12)
        self.assertEqual(self.requests, [])
        self.ready.add(self.versions[2])
        self.sample(12)
        self.sample(12)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(
            [change["target"] for change in self.changes()], [None, 1, 1, None]
        )

    def test_expired_projection_preserves_incumbent_without_reserving(self):
        self.configure_provision()
        self.config["provision"][0]["slo_profile"]["valid_until_ms"] = 2000
        self.policy.write_text(json.dumps(self.config))
        self.sample(0, seconds=0)
        self.sample(12)
        self.sample(12)
        self.assertEqual(self.plans, [])
        self.assertEqual(self.requests, [])

    def configure_device_reuse(self):
        primary, bridge, target = (self.versions[i] for i in (0, 3, 4))
        target_group = self.groups[0]
        deployment = self.policy.with_name("primary.json")
        deployment.write_text(
            json.dumps(native_deployment(target_group, target_group, 8100))
        )
        self.routes = (self.routes[0], self.routes[2])
        self.ready = {bridge}
        self.config["modes"] = [
            self.profile([primary], 1, 4),
            self.profile([bridge], 1, 15),
            self.profile([target], 4, 12),
        ]
        self.config["provision"] = [
            {
                "groups": [
                    {"deployment": deployment.name, "workload": {"widths": [4]}}
                ],
                "devices": 4,
                "bridge": bridge,
                "slo_profile": {"rps": 12, "valid_until_ms": 60_000},
            }
        ]
        self.policy.write_text(json.dumps(self.config))
        self.plan_reply = {"decision": "no_placement"}
        self.preview_reply = {"decision": "after_release", "plans": [{"workers": [{}] * 4}]}
        return bridge, target

    def test_device_reuse_continues_from_the_active_bridge_after_restart(self):
        bridge, target = self.configure_device_reuse()
        self.sample(0, seconds=0)
        self.sample(10)
        self.sample(10)
        self.assertEqual(len(self.plans), 2)
        prepare, preview = self.plans
        self.assertIn("prepare", prepare)
        self.assertNotIn("prepare", preview)
        self.assertEqual(
            preview["after_releasing"],
            [{"group": list(bytes.fromhex(self.groups[0])), "epoch": 1}],
        )
        self.assertEqual(
            [(change["expected"], change["target"]) for change in self.changes()],
            [(1, None), (None, None), (None, None), (None, 1)],
        )

        self.active = frozenset([bridge])
        self.controller = mode.ModeController(self.policy, self.load, self.control)
        self.sample(0, seconds=0)
        self.sample(10)
        resumed = self.plans[-1]
        self.assertIn("prepare", resumed)
        self.assertNotIn("after_releasing", resumed)
        self.assertEqual(resumed["guards"][-1], [list(bytes.fromhex(self.groups[3])), 1])
        self.assertEqual(len(self.requests), 1)

        self.routes += ((Endpoint("127.0.0.1", 8005, target, "aggregated"),),)
        self.ready = {target}
        self.device_counts[target] = 4
        self.health.return_value = False
        self.sample(10)
        self.assertEqual(len(self.requests), 1)
        self.health.return_value = True
        self.sample(10)
        self.assertEqual(
            [(change["expected"], change["target"]) for change in self.changes()],
            [(None, 2), (None, None), (None, None), (1, None)],
        )

    def test_device_reuse_needs_a_feasible_preview_and_a_bridge_that_can_carry_load(self):
        self.configure_device_reuse()
        self.preview_reply = {"decision": "no_placement"}
        self.sample(0, seconds=0)
        self.sample(10)
        self.sample(10)
        self.assertEqual(len(self.plans), 2)
        self.assertEqual(self.requests, [])
        self.preview_reply = {"decision": "after_release", "plans": [{"workers": [{}] * 4}]}
        self.sample(16)
        self.sample(16)
        self.assertNotIn("after_releasing", self.plans[-1])
        self.assertEqual(self.requests, [])

    def test_calibrates_bridge_before_switch_and_calibrates_target_after_release(self):
        bridge, target = self.configure_device_reuse()
        self.config["modes"] = self.config["modes"][:1]
        self.config["calibration"] = {
            "model": "/model",
            "input_tokens": 64,
            "output_tokens": 8,
            "rates": [10, 16],
            "seeds": [1, 2],
            "ttft_ms": 100,
            "tpot_ms": 20,
        }
        self.policy.write_text(json.dumps(self.config))
        self.controller.model = "test"
        self.controller.profile_store = ProfileStore(self.policy.with_name("capacity.json"))
        calibrator = self.controller.calibrator = Mock(run=None)
        self.sample(0, seconds=0)
        self.sample(10)
        self.sample(10)
        bridge_measurement = calibrator.start.call_args.args[0]
        self.assertEqual(bridge_measurement.versions, {bridge})
        self.assertEqual(self.requests, [])

        calibrator.run = SimpleNamespace(target=bridge_measurement)
        calibrator.poll.return_value = {"rps": 14, "valid_until_ms": 60_000}
        self.sample(10)
        calibrator.run = None
        self.sample(10)
        self.assertEqual(self.changes()[3]["target"], 1)

        self.active = frozenset([bridge])
        self.ready = {target}
        self.device_counts[target] = 4
        self.routes = (self.routes[1], (Endpoint("127.0.0.1", 8005, target, "aggregated"),))
        self.sample(10, seconds=2)
        target_measurement = calibrator.start.call_args.args[0]
        self.assertEqual(target_measurement.versions, {target})
        self.assertEqual(target_measurement.active, {bridge})
        self.assertEqual(len(self.requests), 1)

        calibrator.run = SimpleNamespace(target=target_measurement)
        calibrator.poll.return_value = {"rps": 16, "valid_until_ms": 60_000}
        self.sample(10)
        self.assertEqual(self.changes()[0]["target"], 2)
        self.assertIsNone(self.changes()[3]["target"])
        self.assertEqual(
            {tuple(record["versions"]) for record in self.controller.profile_store.read()},
            {(bridge,), (target,)},
        )

    def configure_calibration(self):
        self.configure_provision()
        self.config["calibration"] = {
            "model": "/model",
            "input_tokens": 128,
            "output_tokens": 8,
            "rates": [15],
            "seeds": [1],
            "ttft_ms": 100,
            "tpot_ms": 20,
        }
        self.policy.write_text(json.dumps(self.config))
        calibrator = self.controller.calibrator = Mock()
        calibrator.run = None
        self.controller.model = "test"
        self.sample(0, seconds=0)
        return calibrator

    def test_restarts_with_a_durable_incumbent_profile_and_rechecks_its_manifest(self):
        calibrator = self.configure_calibration()
        store = ProfileStore(self.policy.with_name("capacity.json"))
        self.controller.profile_store = store
        self.sample(12)
        self.sample(12)
        target = calibrator.start.call_args.args[0]
        calibrator.run = SimpleNamespace(target=target)
        calibrator.poll.side_effect = lambda settings: self.target["slo_profile"]
        self.sample(12)
        [profile] = store.read()
        self.assertEqual(profile["manifest"], target.manifest)
        policy = mode.Policy.read(self.policy, "test")
        assert policy.calibration is not None
        old_settings = asdict(policy.calibration)
        del old_settings["warmup_decode_widths"]
        measured_settings = old_settings.copy()
        del measured_settings["warmups"]
        old_context = {"model": "test", "revision": policy.route_revision}
        self.assertEqual(
            profile["context"],
            fingerprint({**old_context, "settings": measured_settings}),
        )
        legacy_profile = {
            **profile,
            "context": fingerprint({**old_context, "settings": old_settings}),
        }
        store.path.write_text(json.dumps({"format": 1, "profiles": [legacy_profile]}))
        self.active = target.versions
        self.ready = {self.versions[-1]}
        self.config["modes"] = []
        self.policy.write_text(json.dumps(self.config))
        self.controller = mode.ModeController(
            self.policy, self.load, self.control, "test", profile_store=store
        )
        self.sample(0, seconds=0)
        self.assertIn(target.versions, self.controller.measured)
        self.assertIn(profile, store.read())

        pending = self.controller.calibrator = Mock(run=None)
        self.config["calibration"]["warmups"] = 8
        self.config["calibration"]["warmup_decode_widths"] = [2, 2, 3, 4]
        self.policy.write_text(json.dumps(self.config))
        self.sample(0, seconds=0)
        pending.cancel.assert_called_once()
        self.assertIn(target.versions, self.controller.measured)
        self.controller = mode.ModeController(
            self.policy, self.load, self.control, "test", profile_store=store
        )
        self.sample(0, seconds=0)
        self.assertIn(target.versions, self.controller.measured)

        self.config["calibration"]["ttft_ms"] += 1
        self.policy.write_text(json.dumps(self.config))
        self.sample(0, seconds=0)
        self.assertNotIn(target.versions, self.controller.measured)
        self.config["calibration"]["ttft_ms"] -= 1
        self.config["modes"].append(self.profile([self.versions[-1]], 1, 4))
        self.policy.write_text(json.dumps(self.config))
        self.sample(1)
        self.sample(1)
        self.sample(1)
        self.assertEqual(len(self.requests), 1)

        self.tickets[self.versions[1]] = "different group incarnation"
        self.controller = mode.ModeController(
            self.policy, self.load, self.control, "test", profile_store=store
        )
        self.sample(0, seconds=0)
        self.assertEqual(self.controller.measured, {})

    def test_reaps_only_the_committed_failed_set_with_the_same_request_on_retry(self):
        self.configure_provision()
        groups = self.groups[1:3]
        versions = [
            {"group": list(bytes.fromhex(group)), "epoch": 1} for group in groups
        ]
        preparation = {
            "prepare": {"origin": [1] * 16, "counter": 5},
            "members": [[versions[0], "Ready"], [versions[1], "Retired"]],
            "activated": True,
        }
        self.preparations = dict.fromkeys(groups, preparation)
        self.sample(0, seconds=0)
        self.assertEqual(self.aborts, [])
        preparation["activated"] = False
        self.may_lose_reply = True
        self.sample(0)
        self.assertEqual(len(self.aborts), 2)
        self.assertEqual(self.aborts[0], self.aborts[1])
        self.assertEqual(
            self.aborts[0]["operation"]["action"],
            {
                "AbortFailedSet": {
                    "prepare": preparation["prepare"],
                    "versions": versions,
                }
            },
        )
        self.assertEqual(self.requests, [])

    def test_uses_live_calibration_to_switch_and_as_the_next_incumbent(self):
        calibrator = self.configure_calibration()
        self.sample(12)
        self.assertEqual(calibrator.start.call_count, 0)
        self.sample(12)
        target = calibrator.start.call_args.args[0]
        self.assertEqual(target.versions, frozenset(self.versions[1:3]))
        self.assertEqual(target.devices, 2)
        self.assertEqual(self.requests, [])

        def poll(_settings):
            calibrator.run = None
            return self.target["slo_profile"]

        calibrator.run = SimpleNamespace(target=target)
        calibrator.poll.side_effect = poll
        self.sample(12)
        self.assertEqual(len(self.requests), 0)
        # The measured candidate has to win its own complete window sequence.
        self.sample(12)
        self.assertEqual(len(self.requests), 1)
        self.active = target.versions
        self.ready = {self.versions[-1]}
        self.config["modes"].append(self.profile([self.versions[-1]], 1, 4))
        self.policy.write_text(json.dumps(self.config))
        self.sample(1)
        self.sample(1)
        self.sample(1)
        self.assertEqual(len(self.requests), 2)

    def test_discards_calibration_if_candidate_binding_changes_before_publication(self):
        calibrator = self.configure_calibration()
        self.sample(12)
        self.sample(12)
        target = calibrator.start.call_args.args[0]
        calibrator.run = SimpleNamespace(target=target)
        calibrator.poll.return_value = self.target["slo_profile"]
        self.clock += 1
        previous = self.snapshot()
        self.routes = tuple(path for path in self.routes if path not in target.routes)
        self.controller.reconcile(previous, self.snapshot)
        self.assertEqual(self.controller.measured, {})
        self.assertEqual(self.requests, [])


class HandoffTest(unittest.TestCase):
    def setUp(self):
        self.runtime = contextlib.ExitStack()
        self.addCleanup(self.runtime.close)

    def start(self, server):
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.01}
        )
        thread.start()
        self.runtime.callback(server.server_close)
        self.runtime.callback(thread.join)
        self.runtime.callback(server.shutdown)
        return server.server_port

    def backend(self, respond):
        class Native(BaseHTTPRequestHandler):
            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                self.send_response(200)
                content_type, body = respond(body)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, _format, *_args):
                pass

        return self.start(ThreadingHTTPServer(("127.0.0.1", 0), Native))

    def group(self, role, version, respond, *, kv_resume=""):
        gate = Gate()
        gate.allow_until(time.monotonic() + 30)
        port = self.start(
            group_proxy(
                0,
                self.backend(respond),
                gate,
                version,
                role=role,
                kv_lease=30,
                kv_resume=kv_resume,
            )
        )
        return port, gate

    def test_launcher_requires_the_backend_to_confirm_seeded_sampling(self):
        respond = Mock()
        legacy, _ = self.group("decode", DECODE, respond, kv_resume=GREEDY_KV_RESUME)
        seeded, _ = self.group("decode", SPARE, respond, kv_resume=SEEDED_KV_RESUME)
        with self.assertRaises(ControlError):
            verify_backend_capabilities(legacy, {KV_RESUME_HEADER: SEEDED_KV_RESUME})
        verify_backend_capabilities(seeded, {KV_RESUME_HEADER: SEEDED_KV_RESUME})
        with self.assertRaises(ControlError):
            verify_backend_capabilities(seeded, {CHECKPOINT_ADMISSION_HEADER: "1"})
        respond.assert_not_called()

    def request(self, port, path, body=None, headers=None):
        with contextlib.closing(
            http.client.HTTPConnection("127.0.0.1", port, timeout=2)
        ) as conn:
            conn.request(
                "POST",
                path,
                None if body is None else json.dumps(body).encode(),
                headers or {},
            )
            with conn.getresponse() as response:
                return response.status, response.headers, response.read()

    def test_aggregate_ingress_retries_a_rejected_request_in_the_new_epoch(self):
        old_port, old_gate = self.group(
            "aggregated", DECODE, lambda _: self.fail("closed group ran")
        )
        new_port, _ = self.group(
            "aggregated", SPARE, lambda _: ("application/json", b'{"ok":true}')
        )
        old_gate.close()
        directory = SimpleNamespace(
            model="test",
            endpoint=Mock(
                side_effect=[
                    Endpoint("127.0.0.1", old_port, DECODE, "aggregated"),
                    Endpoint("127.0.0.1", new_port, SPARE, "aggregated"),
                ]
            ),
        )
        ingress = self.start(aggregate.server("127.0.0.1", 0, directory, 2))
        status, headers, body = self.request(ingress, COMPLETIONS, {"model": "test"})
        self.assertEqual(
            (status, headers[VERSION_HEADER], json.loads(body)),
            (200, SPARE, {"ok": True}),
        )

    def native_switch_directory(self, old_port, new_port):
        folder = self.runtime.enter_context(tempfile.TemporaryDirectory())
        path = Path(folder) / "routes.json"
        path.write_text(
            json.dumps(
                {
                    "model": "test",
                    "revision": 1,
                    "paths": [],
                    "aggregated": [
                        {"url": f"http://127.0.0.1:{port}", "version": version}
                        for port, version in ((old_port, DECODE), (new_port, SPARE))
                    ],
                }
            )
        )
        keys = [list(bytes.fromhex(v.split("@")[0])) for v in (DECODE, SPARE)]
        control = Mock()
        directory = disaggregate.NativeDirectory("test", control, path)

        def publish(epochs):
            control.request.return_value = {
                "result": {"Ok": {"ActiveSet": list(zip(keys, epochs, strict=True))}}
            }
            directory.refresh()

        publish([1, None])
        return directory, publish

    def test_native_ingress_waits_for_a_replacement_after_closed_admission(self):
        old_port, old_gate = self.group(
            "aggregated", DECODE, lambda _: self.fail("closed group ran")
        )
        respond = Mock(return_value=("application/json", b'{"ok":true}'))
        new_port, _ = self.group("aggregated", SPARE, respond)
        directory, publish = self.native_switch_directory(old_port, new_port)
        rejected = threading.Event()
        probe = disaggregate.endpoint_ready
        check_rejection = disaggregate.admission_rejected

        def ready(endpoint, timeout):
            status = probe(endpoint, timeout)
            if endpoint.version == DECODE:
                old_gate.close()
            return status

        def saw_rejection(response):
            status = check_rejection(response)
            if status:
                rejected.set()
            return status

        ingress = self.start(disaggregate.server("127.0.0.1", 0, directory, 2))
        with (
            patch.object(disaggregate, "endpoint_ready", side_effect=ready),
            patch.object(disaggregate, "admission_rejected", side_effect=saw_rejection),
            ThreadPoolExecutor(max_workers=1) as executor,
        ):
            request = executor.submit(
                self.request, ingress, COMPLETIONS, {"model": "test"}
            )
            self.assertTrue(rejected.wait(2))
            publish([None, 1])
            status, _, body = request.result(timeout=2)
        self.assertEqual((status, json.loads(body)), (200, {"ok": True}))
        respond.assert_called_once()

    def test_native_ingress_never_retries_a_post_with_unknown_outcome(self):
        respond = Mock()
        new_port, _ = self.group("aggregated", SPARE, respond)
        received = threading.Event()

        class Dropped(BaseHTTPRequestHandler):
            def do_GET(self):
                self.send_response(200)
                headers = Endpoint("127.0.0.1", 1, DECODE, "aggregated").headers()
                for key, value in headers.items():
                    self.send_header(key, value)
                self.end_headers()

            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                received.set()
                publish([None, 1])

        old_port = self.start(ThreadingHTTPServer(("127.0.0.1", 0), Dropped))
        directory, publish = self.native_switch_directory(old_port, new_port)
        ingress = self.start(disaggregate.server("127.0.0.1", 0, directory, 2))
        status, _, _ = self.request(ingress, COMPLETIONS, {"model": "test"})
        self.assertEqual(status, 502)
        self.assertTrue(received.is_set())
        respond.assert_not_called()

    def test_aggregate_ingress_does_not_replay_after_the_upstream_receives_bytes(self):
        received = threading.Event()
        new_port, _ = self.group(
            "aggregated", SPARE, lambda _: self.fail("request was replayed")
        )

        class Dropped(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                received.set()

        old_port = self.start(ThreadingHTTPServer(("127.0.0.1", 0), Dropped))
        directory = SimpleNamespace(
            model="test",
            endpoint=Mock(
                side_effect=[
                    Endpoint("127.0.0.1", old_port, DECODE, "aggregated"),
                    Endpoint("127.0.0.1", new_port, SPARE, "aggregated"),
                ]
            ),
        )
        ingress = self.start(aggregate.server("127.0.0.1", 0, directory, 2))
        status, _, _ = self.request(ingress, COMPLETIONS, {"model": "test"})
        self.assertEqual(status, 502)
        self.assertTrue(received.is_set())
        directory.endpoint.assert_called_once()

    def test_native_ingress_recovers_only_a_checked_greedy_prefix(self):
        for recovery in (RECOVERY_MODE, RESUME_PREFIX_MODE):
            for matches_prefix in (True, False):
                with self.subTest(recovery=recovery, matches_prefix=matches_prefix):
                    self._check_native_prefix_recovery(recovery, matches_prefix)

    def _check_native_prefix_recovery(self, recovery, matches_prefix):
        group = bytes.fromhex("0a" * 16)
        versions = [f"{group.hex()}@{epoch}" for epoch in (1, 2)]
        control = Mock()
        control.request.side_effect = [
            {
                "result": {
                    "Ok": {
                        "version": {"group": list(group), "epoch": epoch},
                        "phase": "Active",
                    }
                }
            }
            for epoch in (1, 2)
        ]
        with tempfile.TemporaryDirectory() as folder:
            binding = Path(folder) / "routes.json"
            directory = aggregate.NativeGroupDirectory("test", group, control, binding)

            def chunk(identity, text, tokens, prompt=None, *, resumed=None):
                choice = {
                    "index": 0,
                    "text": text,
                    "token_ids": tokens,
                    "finish_reason": None if tokens else "length",
                }
                if prompt is not None:
                    choice["prompt_token_ids"] = prompt
                if resumed is not None:
                    choice["plait_resumed_tokens"] = resumed
                return sse({"id": identity, "model": "test", "choices": [choice]})

            def old_engine(body):
                self.assertTrue(body["return_token_ids"])
                directory.refresh()
                return "text/event-stream", chunk("first", "a", [8], [4])

            def new_engine(body):
                self.assertNotIn("kv_transfer_params", body)
                if recovery == RESUME_PREFIX_MODE:
                    self.assertEqual(
                        body["vllm_xargs"], {"plait_resume_token_ids": [8]}
                    )
                text = "ab" if matches_prefix else "xb"
                resumed = 1 if recovery == RESUME_PREFIX_MODE else None
                stream = chunk("second", text, [8, 9], [4], resumed=resumed)
                stream += chunk("second", "", []) + b"data: [DONE]\n\n"
                return "text/event-stream", stream

            old_port, _ = self.group("aggregated", versions[0], old_engine)
            new_port, _ = self.group("aggregated", versions[1], new_engine)
            binding.write_text(
                json.dumps(
                    {
                        "1": f"http://127.0.0.1:{old_port}",
                        "2": f"http://127.0.0.1:{new_port}",
                    }
                )
            )
            directory.refresh()
            ingress = self.start(aggregate.server("127.0.0.1", 0, directory, 2))
            payload = {
                "model": "test",
                "prompt": [4],
                "max_tokens": 2,
                "stream": True,
                "temperature": 0,
                "return_token_ids": True,
            }
            self.assertEqual(
                self.request(
                    ingress,
                    COMPLETIONS,
                    payload,
                    {RECOVERY_HEADER: RESUME_KV_MODE},
                )[0],
                400,
            )
            status, headers, response = self.request(
                ingress,
                COMPLETIONS,
                payload,
                {RECOVERY_HEADER: recovery},
            )
            self.assertEqual((status, headers[VERSION_HEADER]), (200, versions[0]))
            items = [
                line[6:] for line in response.splitlines() if line.startswith(b"data: ")
            ]
            frames = [json.loads(item) for item in items if item != b"[DONE]"]
            choices = [
                choice for frame in frames for choice in frame.get("choices", [])
            ]
            self.assertEqual(
                {frame["id"] for frame in frames if "id" in frame}, {"first"}
            )
            self.assertEqual(
                [token for choice in choices for token in choice["token_ids"]],
                [8, 9] if matches_prefix else [8],
            )
            self.assertEqual(
                "".join(choice["text"] for choice in choices),
                "ab" if matches_prefix else "a",
            )
            self.assertEqual(items.count(b"[DONE]"), int(matches_prefix))
            self.assertEqual("error" in frames[-1], not matches_prefix)
            self.assertEqual(
                [call.args[0]["request"] for call in control.request.call_args_list],
                ["observe", "observe"],
            )

    def test_drain_keeps_export_alive_until_versioned_release(self):
        port, gate = self.group(
            "prefill", PREFILL, lambda _: ("application/json", b"{}")
        )
        headers = {VERSION_HEADER: PREFILL, HANDOFF_HEADER: TOKEN}
        status, fields, _ = self.request(port, COMPLETIONS, {}, headers)
        self.assertEqual((status, fields[VERSION_HEADER]), (200, PREFILL))
        gate.close()
        self.assertFalse(gate.drained())
        status, fields, _ = self.request(port, COMPLETIONS, {}, headers)
        self.assertEqual((status, fields[ADMISSION_HEADER]), (503, "rejected"))
        stale = {**headers, VERSION_HEADER: PREFILL.replace("@1", "@2")}
        self.assertEqual(self.request(port, RELEASE_KV_PATH, headers=stale)[0], 409)
        self.assertFalse(gate.drained())
        self.assertEqual(self.request(port, RELEASE_KV_PATH, headers=headers)[0], 204)
        self.assertTrue(gate.drained())

    def test_explicit_decode_rejection_can_use_another_route(self):
        unused_d, closing_gate = self.group(
            "decode", DECODE, lambda _: self.fail("old D ran")
        )
        stream = b'data: {"choices": []}\n\ndata: [DONE]\n\n'
        exported = []
        decoded = []

        def prefill(body):
            self.assertEqual(
                body["kv_transfer_params"],
                {
                    "do_remote_decode": True,
                    "do_remote_prefill": False,
                },
            )
            self.assertEqual(body["max_tokens"], 1)
            self.assertIs(body["stream"], False)
            closing_gate.close()
            descriptor = {
                "do_remote_prefill": True,
                "remote_request_id": f"request-{len(exported)}",
                "remote_engine_id": "engine",
                "remote_block_ids": [[1]],
                "remote_host": "127.0.0.1",
                "remote_port": 9100,
            }
            exported.append(descriptor)
            return "application/json", json.dumps(
                {"kv_transfer_params": descriptor}
            ).encode()

        def decode(body):
            decoded.append(body)
            return "text/event-stream", stream

        p_port, p_gate = self.group("prefill", PREFILL, prefill)
        d_port, _ = self.group("decode", SPARE, decode)
        directory = (
            Path(self.runtime.enter_context(tempfile.TemporaryDirectory()))
            / "routes.json"
        )
        endpoint = {
            "url": f"http://127.0.0.1:{p_port}",
            "version": PREFILL,
            "kv_host": "127.0.0.1",
            "kv_port": 9100,
        }
        directory.write_text(
            json.dumps(
                {
                    "model": "test",
                    "paths": [
                        {
                            "prefill": endpoint,
                            "decode": {
                                "url": f"http://127.0.0.1:{port}",
                                "version": version,
                            },
                        }
                        for port, version in ((unused_d, DECODE), (d_port, SPARE))
                    ],
                }
            )
        )
        router_port = self.start(
            disaggregate.server("127.0.0.1", 0, disaggregate.Directory(directory), 5)
        )
        status, _, body = self.request(
            router_port,
            COMPLETIONS,
            {
                "model": "test",
                "prompt": "test",
                "stream": True,
                "max_tokens": 8,
                "kv_transfer_params": {"remote_host": "untrusted"},
            },
        )
        self.assertEqual((status, body), (200, stream))
        self.assertEqual(len(exported), 2)
        self.assertEqual(len(decoded), 1)
        self.assertEqual(decoded[0]["kv_transfer_params"], exported[1])
        self.assertEqual(decoded[0]["max_tokens"], 8)
        # The final ACK runs before the router closes the response socket.
        self.assertTrue(p_gate.drained())

        # vLLM may also send the terminal marker after a streaming error.
        stream = b'data: {"error": {"message": "decode failed"}}\n\ndata: [DONE]\n\n'
        status, _, body = self.request(
            router_port,
            COMPLETIONS,
            {
                "model": "test",
                "prompt": "next request",
                "stream": True,
                "max_tokens": 8,
            },
        )
        self.assertEqual((status, body), (200, stream))
        self.assertFalse(p_gate.drained())

    def test_plait_handoff_rejects_an_unexpected_store_before_decode(self):
        store = "127.0.0.1:9101"
        decoded = []

        def prefill(body):
            handoff = body["kv_transfer_params"]["plait_handoff"]
            self.assertEqual(set(handoff), {"ticket"})
            receipt = {
                **handoff,
                "store": store,
                "prompt_sha256": "a" * 64,
                "cached_tokens": 128,
            }
            return "application/json", json.dumps(
                {
                    "kv_transfer_params": {"plait_handoff": receipt},
                }
            ).encode()

        def decode(body):
            decoded.append(body)
            return "application/json", b'{"choices": []}'

        p_port, p_gate = self.group("prefill", PREFILL, prefill)
        d_port, _ = self.group("decode", DECODE, decode)
        directory = (
            Path(self.runtime.enter_context(tempfile.TemporaryDirectory()))
            / "routes.json"
        )
        directory.write_text(
            json.dumps(
                {
                    "model": "test",
                    "paths": [
                        {
                            "prefill": {
                                "url": f"http://127.0.0.1:{p_port}",
                                "version": PREFILL,
                                "kv_host": "127.0.0.1",
                                "kv_port": 9100,
                            },
                            "decode": {
                                "url": f"http://127.0.0.1:{d_port}",
                                "version": DECODE,
                            },
                        }
                    ],
                }
            )
        )
        router_port = self.start(
            disaggregate.server(
                "127.0.0.1",
                0,
                disaggregate.Directory(directory),
                5,
                kv_transport="plait",
            )
        )
        payload = {"model": "test", "prompt": "test", "max_tokens": 8}
        self.assertEqual(self.request(router_port, COMPLETIONS, payload)[0], 502)
        self.assertEqual(decoded, [])
        self.assertTrue(p_gate.drained())

        store = "127.0.0.1:9100"
        self.assertEqual(self.request(router_port, COMPLETIONS, payload)[0], 200)
        self.assertEqual(
            decoded[0]["kv_transfer_params"]["plait_handoff"]["store"], store
        )

    def test_plait_pd_replay_matches_the_existing_stream_before_emitting_a_suffix(self):
        for mismatched, aggregated in product((False, True), repeat=2):
            with self.subTest(mismatched=mismatched, aggregated=aggregated):
                self._check_pd_prefix_recovery(mismatched, aggregated)

    def _check_pd_prefix_recovery(self, mismatched, aggregated):
        tickets = []

        def prefill(body):
            ticket = body["kv_transfer_params"]["plait_handoff"]["ticket"]
            tickets.append(ticket)
            receipt = {
                "ticket": ticket,
                "store": "127.0.0.1:9100",
                "prompt_sha256": "c" * 64,
                "cached_tokens": 128,
            }
            return "application/json", json.dumps(
                {
                    "kv_transfer_params": {"plait_handoff": receipt},
                }
            ).encode()

        def chunk(identity, text, ids, prompt=None, terminal=False):
            choice = {
                "index": 0,
                "text": text,
                "token_ids": ids,
                "finish_reason": "length" if terminal else None,
            }
            if prompt is not None:
                choice["prompt_token_ids"] = prompt
            return sse({"id": identity, "model": "test", "choices": [choice]})

        def old_decode(body):
            self.assertTrue(body["return_token_ids"])
            self.assertEqual(
                body["kv_transfer_params"]["plait_handoff"]["ticket"],
                tickets[-1],
            )
            return "text/event-stream", chunk("first", "a", [8], [4])

        def new_decode(body):
            if aggregated:
                self.assertNotIn("kv_transfer_params", body)
            else:
                self.assertEqual(
                    body["kv_transfer_params"]["plait_handoff"]["ticket"],
                    tickets[-1],
                )
            prefix = "x" if mismatched else "a"
            stream = chunk("second", prefix + "b", [8, 9], [4])
            stream += chunk("second", "", [], terminal=True) + b"data: [DONE]\n\n"
            return "text/event-stream", stream

        p_port, gate = self.group("prefill", PREFILL, prefill)
        old_port, _ = self.group("decode", DECODE, old_decode)
        role = "aggregated" if aggregated else "decode"
        new_port, _ = self.group(role, SPARE, new_decode)
        source = Endpoint("127.0.0.1", p_port, PREFILL, "prefill")
        old_path = disaggregate.Route(
            source,
            Endpoint("127.0.0.1", old_port, DECODE, "decode"),
            "127.0.0.1",
            9100,
        )
        replacement = Endpoint("127.0.0.1", new_port, SPARE, role)
        new_path = (
            replacement
            if aggregated
            else disaggregate.Route(source, replacement, "127.0.0.1", 9100)
        )
        directory = SimpleNamespace(
            routes=Mock(side_effect=[("test", [old_path]), ("test", [new_path])])
        )
        router = self.start(
            disaggregate.server("127.0.0.1", 0, directory, 2, kv_transport="plait")
        )
        status, headers, body = self.request(
            router,
            COMPLETIONS,
            {
                "model": "test",
                "prompt": [4],
                "max_tokens": 2,
                "stream": True,
                "temperature": 0,
                "return_token_ids": True,
            },
            {RECOVERY_HEADER: RECOVERY_MODE},
        )
        self.assertEqual(
            (status, headers[VERSION_HEADER], headers[RECOVERY_HEADER]),
            (200, DECODE, RECOVERY_MODE),
        )
        data = [line[6:] for line in body.splitlines() if line.startswith(b"data: ")]
        frames = [json.loads(line) for line in data if line != b"[DONE]"]
        choices = [choice for frame in frames for choice in frame.get("choices", [])]
        self.assertEqual({frame["id"] for frame in frames if "id" in frame}, {"first"})
        self.assertEqual(
            [token for choice in choices for token in choice["token_ids"]],
            [8] if mismatched else [8, 9],
        )
        self.assertEqual(
            "".join(choice["text"] for choice in choices),
            "a" if mismatched else "ab",
        )
        self.assertEqual(data.count(b"[DONE]"), 0 if mismatched else 1)
        self.assertEqual("error" in frames[-1], mismatched)
        self.assertEqual(len(set(tickets)), 1 if aggregated else 2)
        self.assertTrue(gate.drained())

    def test_aggregate_request_finishes_while_split_routes_accept_new_requests(self):
        admitted, finish = threading.Event(), threading.Event()
        stream = b'data: {"choices": []}\n\ndata: [DONE]\n\n'

        def aggregate(body):
            self.assertNotIn("kv_transfer_params", body)
            admitted.set()
            self.assertTrue(finish.wait(5))
            return "text/event-stream", stream

        def prefill(body):
            ticket = body["kv_transfer_params"]["plait_handoff"]["ticket"]
            receipt = {
                "ticket": ticket,
                "store": "127.0.0.1:9100",
                "prompt_sha256": "c" * 64,
                "cached_tokens": 128,
            }
            return "application/json", json.dumps(
                {"kv_transfer_params": {"plait_handoff": receipt}}
            ).encode()

        agg_port, gate = self.group("aggregated", SPARE, aggregate)
        p_port, _ = self.group("prefill", PREFILL, prefill)
        d_port, _ = self.group(
            "decode", DECODE, lambda _: ("application/json", b'{"split":true}')
        )
        aggregate_document = {
            "model": "test",
            "paths": [],
            "aggregated": [
                {"url": f"http://127.0.0.1:{agg_port}", "version": SPARE},
            ],
        }
        split_document = {
            "model": "test",
            "paths": [
                {
                    "prefill": {
                        "url": f"http://127.0.0.1:{p_port}",
                        "version": PREFILL,
                        "kv_host": "127.0.0.1",
                        "kv_port": 9100,
                    },
                    "decode": {"url": f"http://127.0.0.1:{d_port}", "version": DECODE},
                }
            ],
        }
        directory = SimpleNamespace(
            routes=Mock(
                side_effect=[
                    disaggregate.parse_directory(aggregate_document),
                    disaggregate.parse_directory(split_document),
                ]
            )
        )
        router = self.start(
            disaggregate.server("127.0.0.1", 0, directory, 5, kv_transport="plait")
        )
        body = {"model": "test", "prompt": [1], "max_tokens": 2}
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(
                self.request, router, COMPLETIONS, {**body, "stream": True}
            )
            try:
                self.assertTrue(admitted.wait(2))
                gate.close()
                self.assertFalse(gate.drained())
                status, _, data = self.request(router, COMPLETIONS, body)
                self.assertEqual((status, json.loads(data)), (200, {"split": True}))
            finally:
                finish.set()
            status, _, data = first.result(timeout=2)
            self.assertEqual((status, data), (200, stream))
        self.assertTrue(gate.drained())

    def test_planned_stream_handoff_uses_only_delivered_tokens_and_checks_capability(
        self,
    ):
        for source_capability, target_capability in (
            (GREEDY_KV_RESUME, GREEDY_KV_RESUME),
            (GREEDY_KV_RESUME, ""),
            (SEEDED_KV_RESUME, SEEDED_KV_RESUME),
            (SEEDED_KV_RESUME, GREEDY_KV_RESUME),
        ):
            with self.subTest(source=source_capability, target=target_capability):
                self._check_planned_handoff(source_capability, target_capability)

    def _check_planned_handoff(self, source_capability, target_capability):
        seeded = source_capability == SEEDED_KV_RESUME
        can_switch = not seeded or target_capability == SEEDED_KV_RESUME
        seed = 648 if seeded else None
        routes = []
        transfers = []

        def prefill(body):
            params = body["kv_transfer_params"]
            self.assertEqual(set(params), {"plait_handoff"})
            ticket = params["plait_handoff"]["ticket"]
            transfers.append(ticket)
            return "application/json", json.dumps(
                {
                    "kv_transfer_params": {
                        "plait_handoff": {
                            "ticket": ticket,
                            "store": "127.0.0.1:9100",
                            "prompt_sha256": "b" * 64,
                            "cached_tokens": 128,
                        }
                    }
                }
            ).encode()

        def chunk(identity, ids, text, *, first=False, resumed=None, finish=None):
            choice = {
                "index": 0,
                "text": text,
                "token_ids": ids,
                "finish_reason": finish,
            }
            if first:
                choice["prompt_token_ids"] = [4]
                if seeded:
                    choice["plait_sampled_seed"] = seed
            if resumed is not None:
                choice["plait_resumed_tokens"] = resumed
            return sse({"id": identity, "model": "test", "choices": [choice]})

        def source(body):
            kv = body["kv_transfer_params"]
            self.assertEqual(kv["plait_handoff"]["ticket"], transfers[0])
            self.assertIs(kv["plait_checkpoint"], True)
            self.assertEqual(
                kv.get("plait_sampling_mode"),
                SEEDED_KV_RESUME if seeded else None,
            )
            self.assertEqual(body.get("seed"), seed)
            routes[:] = [target]
            # A healthy source has already computed a different suffix.
            return "text/event-stream", (
                chunk("source", [8], "a", first=True)
                + chunk("source", [99], "x", finish="length")
                + b"data: [DONE]\n\n"
            )

        def replacement(body):
            self.assertTrue(can_switch)
            kv = body["kv_transfer_params"]
            self.assertEqual(kv["plait_handoff"]["ticket"], transfers[1])
            self.assertIs(kv["plait_checkpoint"], True)
            self.assertEqual(
                kv.get("plait_sampling_mode"),
                SEEDED_KV_RESUME if seeded else None,
            )
            self.assertEqual(body.get("seed"), seed)
            if target_capability:
                self.assertEqual(kv["plait_resume_token_ids"], [8])
            else:
                self.assertNotIn("plait_resume_token_ids", kv)
            return "text/event-stream", (
                chunk(
                    "replacement",
                    [8, 9],
                    "ab",
                    first=True,
                    resumed=1 if target_capability else None,
                )
                + chunk("replacement", [], "", finish="length")
                + b"data: [DONE]\n\n"
            )

        p_port, gate = self.group("prefill", PREFILL, prefill)
        old_port, _ = self.group("decode", DECODE, source, kv_resume=source_capability)
        new_port, _ = self.group(
            "decode", SPARE, replacement, kv_resume=target_capability
        )
        prefill_endpoint = Endpoint("127.0.0.1", p_port, PREFILL, "prefill")
        old, target = [
            disaggregate.Route(
                prefill_endpoint,
                Endpoint("127.0.0.1", port, version, "decode"),
                "127.0.0.1",
                9100,
            )
            for port, version in [(old_port, DECODE), (new_port, SPARE)]
        ]
        routes.append(old)
        directory = SimpleNamespace(routes=lambda: ("test", routes.copy()))
        router = self.start(
            disaggregate.server("127.0.0.1", 0, directory, 3, kv_transport="plait")
        )
        status, headers, body = self.request(
            router,
            COMPLETIONS,
            {
                "model": "test",
                "prompt": [4],
                "max_tokens": 2,
                "stream": True,
                "temperature": 0.8 if seeded else 0,
                "return_token_ids": True,
                **({"seed": seed} if seeded else {}),
            },
            {RECOVERY_HEADER: RESUME_KV_MODE},
        )
        self.assertEqual((status, headers[VERSION_HEADER]), (200, DECODE))
        frames = [line[6:] for line in body.splitlines() if line.startswith(b"data: ")]
        self.assertEqual(frames.count(b"[DONE]"), 1)
        frames = [json.loads(item) for item in frames if item != b"[DONE]"]
        self.assertEqual({item["id"] for item in frames}, {"source"})
        choices = [choice for item in frames for choice in item["choices"]]
        self.assertEqual(
            [token for choice in choices for token in choice["token_ids"]],
            [8, 9 if can_switch else 99],
        )
        self.assertTrue(all("plait_sampled_seed" not in choice for choice in choices))
        self.assertEqual(len(set(transfers)), 2 if can_switch else 1)
        self.assertTrue(gate.drained())

    def test_load_counts_valid_arrivals_when_routes_are_unavailable(self):
        directory = Mock()
        directory.routes.side_effect = ValueError("directory unavailable")
        load = Load("test")
        port = self.start(
            disaggregate.server(
                "127.0.0.1",
                0,
                directory,
                5,
                kv_transport="plait",
                load=load,
            )
        )
        request = {"model": "test", "prompt": "input", "max_tokens": 8}
        self.assertEqual(self.request(port, COMPLETIONS, request)[0], 503)
        self.assertEqual(
            self.request(port, COMPLETIONS, {**request, "model": "other"})[0], 400
        )
        snapshot = load.snapshot()
        self.assertEqual(
            (
                snapshot["arrivals"],
                snapshot["completed"],
                snapshot["failed"],
                snapshot["in_flight"],
            ),
            (1, 0, 1, 0),
        )
        self.assertEqual(snapshot["checkpoints"], {"arrivals": 0, "in_flight": 0})
        checkpoint = {**request, "stream": True, "temperature": 0}
        self.assertEqual(
            self.request(
                port, COMPLETIONS, checkpoint, {RECOVERY_HEADER: RESUME_KV_MODE}
            )[0],
            503,
        )
        self.assertEqual(
            load.snapshot()["checkpoints"], {"arrivals": 1, "in_flight": 0}
        )


if __name__ == "__main__":
    unittest.main()
