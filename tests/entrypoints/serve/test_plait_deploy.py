# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import tempfile
import unittest
from pathlib import Path
from socket import AF_INET
from types import SimpleNamespace
from unittest.mock import Mock, patch

from starlette.responses import Response
from starlette.testclient import TestClient

from vllm.entrypoints.serve.plait_deploy import (
    Control,
    ControlUnavailable,
    Deployment,
    Engine,
    RoutePublisher,
)
from vllm.entrypoints.serve.plait_version import VERSION_HEADER, NativeGroupHeaders


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
