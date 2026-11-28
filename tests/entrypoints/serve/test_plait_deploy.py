# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import fcntl
import json
import signal
import tempfile
import unittest
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
from socket import AF_INET
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from vllm.distributed.plait_rendezvous import CommittedModelGroup
from vllm.entrypoints.serve.plait_deploy import (
    Control,
    ControlUnavailable,
    Deployment,
    Engine,
    EngineGrantReturn,
    RoutePublisher,
    WorkerProcess,
    reconcile_orphan_cgroup,
    reconcile_owner,
)
from vllm.entrypoints.serve.plait_version import (
    DRAIN_PATH,
    VERSION_HEADER,
    NativeGroupHeaders,
)


class TestPlaitDeployment(unittest.TestCase):
    def setUp(self):
        # Direct process-group behavior is still used on POSIX hosts without
        # the Linux owner. Test the Linux launcher explicitly below.
        platform = patch("vllm.entrypoints.serve.plait_deploy.sys.platform", "darwin")
        platform.start()
        self.addCleanup(platform.stop)
        self.first, self.second = "01" * 16, "02" * 16
        self.config = Deployment(
            group_id="03" * 16,
            seeds=["192.0.2.1:7950"],
            model_alias="model",
            hosts={self.first: "192.0.2.1", self.second: "192.0.2.2"},
            api_port_base=8100,
            rendezvous_port_base=8200,
            epoch_ports=16,
        )
        self.view = {
            "version": {"group": [3] * 16, "epoch": 2},
            "grants": [
                {"device": {"node": [2] * 16, "ordinal": 5}},
                {"device": {"node": [1] * 16, "ordinal": 2}},
                {"device": {"node": [2] * 16, "ordinal": 1}},
            ],
            "phase": "Preparing",
        }
        self.placement = self.config.place(self.view)
        self.control = Mock(spec=Control)
        self.control.observe.return_value = self.view

    def test_uneven_hosts_agree_on_engine_slot_order(self):
        group = self.placement.descriptor(self.second, self.config.seeds)["tp_groups"][
            0
        ]
        assert group["ranks"] == [0, 1, 2]
        assert group["devices"] == [[self.first, 2], [self.second, 1], [self.second, 5]]
        assert self.placement.origin == "http://192.0.2.1:8101"

    def test_engine_uses_granted_devices_and_local_gloo_interface(self):
        engine = Engine(["vllm"], "/models/example", "CUDA_VISIBLE_DEVICES", [], {})
        addresses = {
            "management": [SimpleNamespace(family=AF_INET, address="192.0.2.3")],
            "data": [SimpleNamespace(family=AF_INET, address="192.0.2.2")],
        }
        module = "vllm.entrypoints.serve.plait_deploy"
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {}, clear=True),
            patch(f"{module}.psutil.net_if_addrs", return_value=addresses),
            patch(f"{module}.subprocess.Popen") as spawn,
        ):
            worker = engine.start(
                self.config, self.placement, self.second, Path(directory), self.control
            )
            assert isinstance(worker, WorkerProcess)
            self.addCleanup(worker.lock.close)
            env = spawn.call_args.kwargs["env"]
            assert env["CUDA_VISIBLE_DEVICES"] == "1,5"
            assert env["GLOO_SOCKET_IFNAME"] == "data"
            command = spawn.call_args.args[0]
            assert "--headless" in command
            assert command[command.index("--max-num-batched-tokens") + 1] == "2048"
            assert command[command.index("--max-num-seqs") + 1] == "128"

    def test_pipeline_launch_binds_the_full_group_and_preserves_layer_order(self):
        recipe = "05" * 16
        self.config = replace(
            self.config,
            layout_recipes={
                recipe: {
                    "width": 2,
                    "pipeline_layers": [11, 17],
                    "kv_cache_memory_bytes_by_stage": [201326592, 335544320],
                }
            },
        )
        grants = [
            {"device": {"node": [2] * 16, "ordinal": 1}},
            {"device": {"node": [1] * 16, "ordinal": 2}},
        ]
        placement = self.config.place(
            {"version": self.placement.version, "grants": grants, "recipe": [5] * 16}
        )
        engine = Engine(["vllm"], "/models/example", "CUDA_VISIBLE_DEVICES", [], {})
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "vllm.entrypoints.serve.plait_deploy.psutil.net_if_addrs",
                return_value={},
            ),
            patch("vllm.entrypoints.serve.plait_deploy.subprocess.Popen") as spawn,
        ):
            worker = engine.start(
                self.config, placement, self.first, Path(directory), self.control
            )
            assert isinstance(worker, WorkerProcess)
            self.addCleanup(worker.lock.close)
            env = spawn.call_args.kwargs["env"]
            command = spawn.call_args.args[0]
            assert command[command.index("--tensor-parallel-size") + 1] == "1"
            assert command[command.index("--pipeline-parallel-size") + 1] == "2"
            budget_index = command.index("--kv-cache-memory-bytes-by-stage")
            assert command[budget_index + 1 : budget_index + 3] == [
                "201326592",
                "335544320",
            ]
            assert env["VLLM_PP_LAYER_PARTITION"] == "11,17"
            descriptor = json.loads(Path(env["VLLM_PLAIT_TP_GROUP_CONFIG"]).read_text())
            assert set(descriptor) == {"host_id", "seeds", "pp_groups"}
            assert descriptor["pp_groups"][0]["devices"] == [
                [self.first, 2],
                [self.second, 1],
            ]

    def test_mixed_recipe_starts_both_axes_with_one_device_binding(self):
        recipe = "05" * 16
        self.config = replace(
            self.config,
            layout_recipes={
                recipe: {
                    "width": 4,
                    "tensor_parallel": 2,
                    "pipeline_layers": [11, 17],
                }
            },
        )
        self.view["recipe"] = list(bytes.fromhex(recipe))
        self.view["grants"].append({"device": {"node": [1] * 16, "ordinal": 7}})
        placement = self.config.place(self.view)
        engine = Engine(["vllm"], "/models/example", "CUDA_VISIBLE_DEVICES", [], {})
        deploy = "vllm.entrypoints.serve.plait_deploy"
        join_path = (
            "vllm.distributed.device_communicators.pynccl.join_committed_model_group"
        )
        library, tp, pp = object(), object(), object()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(f"{deploy}.psutil.net_if_addrs", return_value={}),
            patch(f"{deploy}.subprocess.Popen") as spawn,
            patch(join_path, return_value=(library, tp, pp)) as join,
        ):
            worker = engine.start(
                self.config, placement, self.second, Path(directory), self.control
            )
            assert isinstance(worker, WorkerProcess)
            self.addCleanup(worker.lock.close)
            command = spawn.call_args.args[0]
            assert command[command.index("--tensor-parallel-size") + 1] == "2"
            assert command[command.index("--pipeline-parallel-size") + 1] == "2"
            env = spawn.call_args.kwargs["env"]
            assert env["VLLM_PP_LAYER_PARTITION"] == "11,17"
            path = env["VLLM_PLAIT_TP_GROUP_CONFIG"]
            model = CommittedModelGroup.load(path, 3, 2, 2)
            device = Mock()
            with self.assertRaisesRegex(ValueError, "PP slots"):
                model.take_axis("pp", [0, 3], device)
            assert model.take_axis("tp", [2, 3], device)[1] == (library, tp)
            assert model.take_axis("pp", [1, 3], device)[1] == (library, pp)
            join.assert_called_once_with(device, model.binding, 3, 2)

    def test_mixed_recipe_rejects_a_nonrectangular_width(self):
        with self.assertRaisesRegex(ValueError, "valid TP × PP width"):
            replace(
                self.config,
                layout_recipes={
                    "05" * 16: {
                        "width": 5,
                        "tensor_parallel": 2,
                        "pipeline_layers": [11, 17],
                    }
                },
            )

    def test_recipe_rejects_invalid_stage_budgets(self):
        for layers, budgets in [
            ([], [1, 2]),
            ([11, 17], [1]),
            ([11, 17], [0, 2]),
            ([11, 17], [True, 2]),
        ]:
            with (
                self.subTest(layers=layers, budgets=budgets),
                self.assertRaisesRegex(
                    ValueError, "one positive KV budget per PP stage"
                ),
            ):
                replace(
                    self.config,
                    layout_recipes={
                        "05" * 16: {
                            "width": 2,
                            "pipeline_layers": layers,
                            "kv_cache_memory_bytes_by_stage": budgets,
                        }
                    },
                )

    def test_committed_recipe_disambiguates_layouts_with_the_same_width(self):
        tensor = "04" * 16
        pipeline = "05" * 16
        self.config.layout_recipes = {
            tensor: {"width": 2},
            pipeline: {"width": 2, "pipeline_layers": [11, 17]},
        }
        view = {
            "version": self.placement.version,
            "grants": [
                {"device": {"node": [2] * 16, "ordinal": 1}},
                {"device": {"node": [1] * 16, "ordinal": 2}},
            ],
        }
        with self.assertRaisesRegex(ValueError, "recipe-aware"):
            self.config.place(view)
        tensor_placement = self.config.place({**view, "recipe": [4] * 16})
        pipeline_placement = self.config.place({**view, "recipe": [5] * 16})
        assert tensor_placement.devices == pipeline_placement.devices
        assert tensor_placement.pipeline_layers == ()
        assert pipeline_placement.pipeline_layers == (11, 17)
        with self.assertRaisesRegex(ValueError, "committed recipe"):
            self.config.place({**view, "recipe": [6] * 16})

    def test_live_recipe_addition_preserves_the_active_binding(self):
        original, new = "03" * 16, "04" * 16
        self.config.layout_recipes = {original: {"width": 3}}
        tuned = {"width": 3, "replicated_egress_bytes": 256 * 1024}
        view = {
            "version": self.placement.version,
            "grants": [
                {"device": {"node": [1] * 16, "ordinal": 0}},
                {"device": {"node": [1] * 16, "ordinal": 1}},
                {"device": {"node": [2] * 16, "ordinal": 1}},
            ],
        }
        legacy_view = {**view, "recipe": [3] * 16}
        planned_view = {
            **view,
            "version": {**view["version"], "epoch": view["version"]["epoch"] + 1},
            "recipe": list(bytes.fromhex("a04f7569b5d79140032b71bdf79dd49a")),
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "deployment.json"
            recipes = {**self.config.layout_recipes, new: tuned}
            path.write_text(
                json.dumps({**vars(self.config), "layout_recipes": recipes})
            )
            updated = self.config.reload_hosts(path)
            old = updated.place(legacy_view).descriptor(self.first, updated.seeds)
            newer = updated.place(planned_view).descriptor(self.first, updated.seeds)
            assert "replicated_egress_bytes" not in old["tp_groups"][0]
            assert newer["tp_groups"][0]["replicated_egress_bytes"] == 256 * 1024
            with self.assertRaisesRegex(ValueError, "committed recipe"):
                updated.place({**planned_view, "recipe": [4] * 16})
            recipes[new] = {**tuned, "replicated_egress_bytes": 0}
            path.write_text(
                json.dumps({**vars(self.config), "layout_recipes": recipes})
            )
            with self.assertRaisesRegex(ValueError, "only add hosts or recipes"):
                updated.reload_hosts(path)
            wrong = Deployment.read(path)
            with self.assertRaisesRegex(ValueError, "committed recipe"):
                wrong.place(planned_view)

    def test_activation_stages_binding_and_retries_the_same_operation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "routes.json"
            path.write_text('{"1":"http://192.0.2.2:8100"}')
            control = Mock(spec=Control)
            publisher = RoutePublisher(path)
            with patch(
                "vllm.entrypoints.serve.plait_deploy.healthy", return_value=False
            ):
                publisher.reconcile(control, self.placement, "Ready")
            control.request.assert_not_called()
            assert "2" not in json.loads(path.read_text())

            submitted = []

            def apply(request, *, operation):
                assert request == "apply"
                assert json.loads(path.read_text()) == {
                    "1": "http://192.0.2.2:8100",
                    "2": self.placement.origin,
                }
                submitted.append(operation)
                if len(submitted) == 1:
                    raise ControlUnavailable("reply lost")
                return {"result": {"Ok": {}}}

            control.request.side_effect = apply
            with patch(
                "vllm.entrypoints.serve.plait_deploy.healthy", return_value=True
            ):
                with self.assertRaises(ControlUnavailable):
                    publisher.reconcile(control, self.placement, "Ready")
                publisher.reconcile(control, self.placement, "Ready")
            assert submitted[0] == submitted[1]
            assert publisher.bound_epoch == 2

    def test_engine_rejects_wrong_version_before_dispatch(self):
        async def app(scope, receive, send):
            await Response(status_code=200)(scope, receive, send)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "group.json"
            descriptor = self.placement.descriptor(self.first, self.config.seeds)
            path.write_text(json.dumps(descriptor))
            with patch.dict(
                "os.environ",
                {"VLLM_USE_PLAIT": "1", "VLLM_PLAIT_TP_GROUP_CONFIG": str(path)},
            ):
                engine = TestClient(NativeGroupHeaders(app))
                response = engine.get("/health")
                assert response.status_code == 200
                assert response.headers[VERSION_HEADER] == self.placement.version_text
                response = engine.post(
                    "/v1/completions", headers={VERSION_HEADER: "03" * 16 + "@1"}
                )
                assert response.status_code == 409
                assert response.headers["X-Plait-Admission"] == "rejected"

    def test_engine_binds_the_api_role_to_the_group_version(self):
        async def app(scope, receive, send):
            await Response(status_code=200)(scope, receive, send)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "group.json"
            descriptor = self.placement.descriptor(self.first, self.config.seeds)
            path.write_text(json.dumps({**descriptor, "api_role": "prefill"}))
            with patch.dict(
                "os.environ",
                {"VLLM_USE_PLAIT": "1", "VLLM_PLAIT_TP_GROUP_CONFIG": str(path)},
            ):
                engine = TestClient(NativeGroupHeaders(app))
                headers = {VERSION_HEADER: self.placement.version_text}
                response = engine.post(
                    "/v1/completions",
                    headers={**headers, "X-Plait-Group-Role": "decode"},
                )
                assert response.status_code == 409
                assert response.headers["X-Plait-Admission"] == "rejected"
                response = engine.post(
                    "/v1/completions",
                    headers={**headers, "X-Plait-Group-Role": "prefill"},
                )
                assert response.status_code == 200
                assert response.headers["X-Plait-Group-Role"] == "prefill"

    def test_failed_launch_replays_report_and_release_across_restarts(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            engine = Engine(
                [str(state / "missing")],
                "/models/example",
                "CUDA_VISIBLE_DEVICES",
                [],
                {},
            )
            prefix = self.placement.version_text
            journal = state / f"{prefix}.failed.json"
            phase = "Preparing"
            grant = {"device": {"node": [1] * 16, "ordinal": 2}, "fence": 7}
            self.control.observe.side_effect = lambda version: {
                "version": version,
                "phase": phase,
                "grants": [grant],
            }
            operations = []

            def request(name, **fields):
                assert name == "apply"
                operations.append(fields["operation"])
                if len(operations) in (1, 3):
                    raise ControlUnavailable("reply lost")
                return {"result": {"Ok": {}}}

            self.control.request.side_effect = request
            launch = engine.start(
                self.config, self.placement, self.first, state, self.control
            )
            assert isinstance(launch, EngineGrantReturn)
            assert journal.is_file()
            self.addCleanup(launch.lock.close)
            with self.assertRaises(ControlUnavailable):
                launch.reconcile(self.control)
            with (state / f"{prefix}.lock").open("a") as probe:
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                launch.lock.close()
                recovered = EngineGrantReturn.recover(
                    journal, self.config.group_id, self.first
                )
                assert recovered is not None
                self.addCleanup(recovered.lock.close)
                assert not recovered.reconcile(self.control)
                assert operations[0] == operations[1]
                assert operations[0]["action"] == {
                    "ReportFailure": {"grant": grant, "fault": "Device"}
                }

                phase = "Revoked"
                with self.assertRaises(ControlUnavailable):
                    recovered.reconcile(self.control)
                with self.assertRaises(BlockingIOError):
                    fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)
                recovered.lock.close()

                phase = "Retired"
                resumed = EngineGrantReturn.recover(
                    journal, self.config.group_id, self.first
                )
                assert resumed is not None
                self.addCleanup(resumed.lock.close)
                assert resumed.reconcile(self.control)
                assert operations[2] == operations[3]
                assert operations[2]["action"] == {"Release": grant}
                assert not journal.exists()
                fcntl.flock(probe, fcntl.LOCK_EX | fcntl.LOCK_NB)

    def test_failed_launch_journal_blocks_spawn_and_rejects_changed_grants(self):
        with tempfile.TemporaryDirectory() as directory:
            state = Path(directory)
            engine = Engine(
                [str(state / "missing")],
                "/models/example",
                "CUDA_VISIBLE_DEVICES",
                [],
                {},
            )
            version = self.placement.version
            original = {"device": {"node": [1] * 16, "ordinal": 2}, "fence": 7}
            self.control.observe.return_value = {
                "version": version,
                "phase": "Preparing",
                "grants": [original],
            }
            launch = engine.start(
                self.config, self.placement, self.first, state, self.control
            )
            assert isinstance(launch, EngineGrantReturn)
            launch.lock.close()

            with patch("vllm.entrypoints.serve.plait_deploy.subprocess.Popen") as spawn:
                assert (
                    engine.start(
                        self.config, self.placement, self.first, state, self.control
                    )
                    is None
                )
                spawn.assert_not_called()

            resumed = EngineGrantReturn.recover(
                launch.path, self.config.group_id, self.first
            )
            assert resumed is not None
            with resumed.lock:
                self.control.observe.return_value = {
                    "version": version,
                    "phase": "Revoked",
                    "grants": [dict(original, fence=8)],
                }
                with self.assertRaisesRegex(ValueError, "grants changed"):
                    resumed.reconcile(self.control)
                assert launch.path.is_file()
                self.control.request.assert_not_called()

    def test_engine_rechecks_phase_after_acquiring_launch_lock(self):
        engine = Engine(["vllm"], "/models/example", "CUDA_VISIBLE_DEVICES", [], {})
        self.control.observe.return_value = {"phase": "Revoked"}
        with (
            tempfile.TemporaryDirectory() as directory,
            patch("vllm.entrypoints.serve.plait_deploy.subprocess.Popen") as spawn,
        ):
            assert (
                engine.start(
                    self.config,
                    self.placement,
                    self.second,
                    Path(directory),
                    self.control,
                )
                is None
            )
            spawn.assert_not_called()
            assert not list(Path(directory).glob("*.json"))

    def test_linux_engine_publishes_stop_for_the_recorded_instance(self):
        module = "vllm.entrypoints.serve.plait_deploy"
        engine = Engine(["vllm"], "/models/example", "CUDA_VISIBLE_DEVICES", [], {})
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(f"{module}.sys.platform", "linux"),
            patch(f"{module}.subprocess.Popen") as spawn,
        ):
            state = Path(directory)
            assert (
                engine.start(
                    self.config, self.placement, self.first, state, self.control
                )
                is spawn.return_value
            )
            args = spawn.call_args.args[0]
            assert Path(args[1]).name == "plait_engine_owner.py"
            descriptor = Path(args[2])
            assert not descriptor.exists()
            record = json.loads(args[4])
            owner = descriptor.with_suffix(".owner.json")
            owner.write_text(json.dumps(record))
            stop = descriptor.with_suffix(".stop.json")
            self.control.request.return_value = {"group": self.view}
            reconcile_owner(owner, self.config, self.control, self.first)
            assert not stop.exists()

            self.control.request.return_value = {
                "group": {**self.view, "phase": "Revoked"}
            }
            reconcile_owner(owner, self.config, self.control, self.first)
            assert not stop.exists()  # The ordered observation still says Preparing.
            self.control.observe.return_value = {
                **self.view,
                "phase": "Revoked",
                "grants": [dict(g, fence=2) for g in self.view["grants"]],
            }
            with self.assertRaisesRegex(ValueError, "grants changed"):
                reconcile_owner(owner, self.config, self.control, self.first)
            assert not stop.exists()
            stop.write_text(json.dumps({"origin": "ff" * 16}))
            self.control.observe.return_value = {**self.view, "phase": "Revoked"}
            reconcile_owner(owner, self.config, self.control, self.first)
            assert json.loads(stop.read_text()) == {"origin": record["origin"]}
            assert (
                engine.start(
                    self.config, self.placement, self.first, state, self.control
                )
                is None
            )
            spawn.assert_called_once()

    def test_orphan_cgroup_requires_exact_revoked_grants_and_empty_tree(self):
        module = "vllm.entrypoints.serve.plait_deploy"
        version = self.placement.version
        grant = self.view["grants"][1]
        record = {
            "version": version,
            "host": self.first,
            "grants": [grant],
            "origin": "ab" * 16,
            "cgroup": {},
        }
        with (
            tempfile.TemporaryDirectory() as directory,
            patch(f"{module}.verified_cgroup", return_value=nullcontext(7)),
            patch(f"{module}.write_control") as write,
            patch(f"{module}.populated", return_value=True) as populated,
        ):
            prefix = self.placement.version_text
            root = Path(directory)
            path = root / f"{prefix}.owner.json"
            journal = root / f"{prefix}.stopped.json"
            path.write_text(json.dumps(record))
            with (root / f"{prefix}.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                assert not reconcile_orphan_cgroup(
                    path, self.control, record, self.first
                )
                self.control.observe.assert_not_called()
            self.control.observe.return_value = {**self.view, "phase": "Active"}
            self.control.request.return_value = {"result": {"Ok": {}}}
            assert reconcile_orphan_cgroup(path, self.control, record, self.first)
            operation = self.control.request.call_args.kwargs["operation"]
            assert operation["action"] == {
                "ReportFailure": {"grant": grant, "fault": "Device"}
            }
            write.assert_not_called()
            assert not journal.exists()

            self.control.observe.return_value = {
                **self.view,
                "phase": "Revoked",
                "grants": [dict(grant, fence=13)],
            }
            with self.assertRaisesRegex(ValueError, "grants changed"):
                reconcile_orphan_cgroup(path, self.control, record, self.first)
            write.assert_not_called()
            self.control.observe.return_value = {**self.view, "phase": "Revoked"}
            assert reconcile_orphan_cgroup(path, self.control, record, self.first)
            write.assert_called_once_with(7, "cgroup.kill", "1")
            assert not journal.exists()
            populated.return_value = False
            assert reconcile_orphan_cgroup(path, self.control, record, self.first)
            assert json.loads(journal.read_text()) == {
                key: record[key] for key in ("host", "version", "grants", "origin")
            }

    def test_worker_releases_only_after_process_group_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "group.drain"
            process = Mock()
            process.poll.return_value = None
            grant = {"device": {"node": [1] * 16, "ordinal": 2}, "fence": 7}
            state = Path(directory)
            prefix = self.placement.version_text
            stop_record = state / f"{prefix}.stopped.json"
            launch_lock = (state / f"{prefix}.lock").open("a")
            self.addCleanup(launch_lock.close)
            fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            worker = WorkerProcess(
                process,
                self.placement,
                [grant],
                marker,
                self.first,
                stop_record,
                launch_lock,
            )
            control = Mock(spec=Control)
            phase = "Draining"
            submitted = []
            lose_release_reply = True

            def request(name, **fields):
                nonlocal lose_release_reply
                if name == "view":
                    return {"group": {"phase": phase}}
                assert name == "apply"
                submitted.append(fields["operation"])
                if "Release" in submitted[-1]["action"] and lose_release_reply:
                    lose_release_reply = False
                    raise ControlUnavailable("release reply lost")
                return {"result": {"Ok": {}}}

            control.request.side_effect = request
            control.observe.side_effect = lambda version: {
                "version": version,
                "phase": phase,
                "grants": [grant],
            }
            module = "vllm.entrypoints.serve.plait_deploy"
            with (
                patch(f"{module}.versioned_status") as status,
                patch.object(worker, "signal") as stop,
                patch.object(worker, "gone", return_value=False) as gone,
                patch(f"{module}.time.monotonic", return_value=10) as clock,
            ):
                status.return_value = False
                assert not worker.reconcile(control)
                assert marker.exists()
                assert not submitted
                stop.assert_not_called()
                status.return_value = True
                assert not worker.reconcile(control)
                assert submitted[-1]["action"] == {"Revoke": self.placement.version}
                stop.assert_not_called()
                phase = "Revoked"
                assert not worker.reconcile(control)
                stop.assert_called_once_with(signal.SIGTERM)
                assert len(submitted) == 1
                assert not stop_record.exists()
                assert worker.kill_after == 20
                clock.return_value = 19
                assert not worker.reconcile(control)
                stop.assert_called_once()
                clock.return_value = 20
                assert not worker.reconcile(control)
                stop.assert_called_with(signal.SIGKILL)
                assert stop.call_count == 2
                assert len(submitted) == 1
                assert not stop_record.exists()
                gone.return_value = True
                with self.assertRaises(ControlUnavailable):
                    worker.reconcile(control)
                assert stop_record.exists()
                assert (
                    EngineGrantReturn.recover(
                        stop_record, self.config.group_id, self.first
                    )
                    is None
                )
                worker.lock.close()
                recovered = EngineGrantReturn.recover(
                    stop_record, self.config.group_id, self.first
                )
                assert recovered is not None
                with recovered.lock:
                    assert recovered.reconcile(control)
                assert not stop_record.exists()
                assert submitted[-1]["action"] == {"Release": grant}
                assert submitted[-1] == submitted[-2]


class TestPlaitDrain(unittest.IsolatedAsyncioTestCase):
    async def test_existing_stream_and_device_work_drain_before_ack(self):
        started = asyncio.Event()
        emit_last = asyncio.Event()
        paused = asyncio.Event()
        finish_pause = asyncio.Event()

        async def stream(_request):
            async def body():
                yield "first\n"
                started.set()
                await emit_last.wait()
                yield "last\n"

            return StreamingResponse(body())

        async def pause(**kwargs):
            assert kwargs == {"mode": "wait", "clear_cache": False}
            paused.set()
            await finish_pause.wait()

        with tempfile.TemporaryDirectory() as directory:
            version = "03" * 16 + "@2"
            path = Path(directory) / "group.json"
            path.write_text(json.dumps({"tp_groups": [{"version": version}]}))
            with patch.dict(
                "os.environ",
                {"VLLM_USE_PLAIT": "1", "VLLM_PLAIT_TP_GROUP_CONFIG": str(path)},
            ):
                app = Starlette(routes=[Route("/generate", stream, methods=["POST"])])
                app.add_middleware(NativeGroupHeaders)
                app.state.engine_client = Mock(
                    pause_generation=AsyncMock(side_effect=pause)
                )
                async with AsyncClient(
                    transport=ASGITransport(app),
                    base_url="http://engine",
                    headers={VERSION_HEADER: version},
                ) as client:
                    old = asyncio.create_task(client.post("/generate"))
                    await asyncio.wait_for(started.wait(), 2)
                    path.with_suffix(".drain").touch()
                    rejected = await client.post("/generate")
                    assert rejected.status_code == 503
                    assert rejected.headers["X-Plait-Admission"] == "rejected"
                    assert (await client.get(DRAIN_PATH)).status_code == 202
                    app.state.engine_client.pause_generation.assert_not_called()
                    emit_last.set()
                    assert (await old).text == "first\nlast\n"
                    assert (await client.get(DRAIN_PATH)).status_code == 202
                    await asyncio.wait_for(paused.wait(), 2)
                    assert (await client.get(DRAIN_PATH)).status_code == 202
                    finish_pause.set()
                    await asyncio.sleep(0)
                    ack = await client.get(DRAIN_PATH)
                    assert ack.status_code == 204
                    assert ack.headers[VERSION_HEADER] == version
                    app.state.engine_client.pause_generation.assert_awaited_once()
