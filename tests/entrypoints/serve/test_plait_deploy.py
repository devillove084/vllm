# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from socket import AF_INET
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.responses import Response, StreamingResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from vllm.entrypoints.serve.plait_deploy import (
    Control,
    ControlUnavailable,
    Deployment,
    Engine,
    RoutePublisher,
    WorkerProcess,
)
from vllm.entrypoints.serve.plait_version import (
    DRAIN_PATH,
    VERSION_HEADER,
    NativeGroupHeaders,
)


class TestPlaitDeployment(unittest.TestCase):
    def setUp(self):
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
        self.placement = self.config.place(
            {
                "version": {"group": [3] * 16, "epoch": 2},
                "grants": [
                    {"device": {"node": [2] * 16, "ordinal": 5}},
                    {"device": {"node": [1] * 16, "ordinal": 2}},
                    {"device": {"node": [2] * 16, "ordinal": 1}},
                ],
            }
        )

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
            engine.start(self.config, self.placement, self.second, Path(directory))
            env = spawn.call_args.kwargs["env"]
            assert env["CUDA_VISIBLE_DEVICES"] == "1,5"
            assert env["GLOO_SOCKET_IFNAME"] == "data"
            command = spawn.call_args.args[0]
            assert "--headless" in command
            assert command[command.index("--max-num-batched-tokens") + 1] == "2048"
            assert command[command.index("--max-num-seqs") + 1] == "128"

    def test_pipeline_launch_binds_the_full_group_and_preserves_layer_order(self):
        self.config.pipeline_layers = {"2": [11, 17]}
        grants = [
            {"device": {"node": [2] * 16, "ordinal": 1}},
            {"device": {"node": [1] * 16, "ordinal": 2}},
        ]
        placement = self.config.place(
            {"version": self.placement.version, "grants": grants}
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
            engine.start(self.config, placement, self.first, Path(directory))
            env = spawn.call_args.kwargs["env"]
            command = spawn.call_args.args[0]
            assert command[command.index("--tensor-parallel-size") + 1] == "1"
            assert command[command.index("--pipeline-parallel-size") + 1] == "2"
            assert env["VLLM_PP_LAYER_PARTITION"] == "11,17"
            descriptor = json.loads(Path(env["VLLM_PLAIT_TP_GROUP_CONFIG"]).read_text())
            assert set(descriptor) == {"host_id", "seeds", "pp_groups"}
            assert descriptor["pp_groups"][0]["devices"] == [
                [self.first, 2],
                [self.second, 1],
            ]

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

    def test_worker_releases_only_after_process_group_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            marker = Path(directory) / "group.drain"
            process = Mock()
            process.poll.return_value = None
            grant = {"device": {"node": [1] * 16, "ordinal": 2}, "fence": 7}
            worker = WorkerProcess(process, self.placement, [grant], marker)
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
            control.observe.side_effect = lambda version: {"phase": phase}
            module = "vllm.entrypoints.serve.plait_deploy"
            with (
                patch(f"{module}.versioned_status") as status,
                patch.object(worker, "signal") as stop,
                patch.object(worker, "gone", return_value=False) as gone,
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
                stop.assert_called_once()
                assert len(submitted) == 1
                gone.return_value = True
                with self.assertRaises(ControlUnavailable):
                    worker.reconcile(control)
                assert worker.reconcile(control)
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
