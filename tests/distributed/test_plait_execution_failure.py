# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import Mock, patch

from vllm.distributed.device_communicators.plait_communicator import PlaitCommunicator
from vllm.v1.executor import multiproc_executor
from vllm.v1.executor.multiproc_executor import WorkerProc


class TestPlaitExecutionFailure(unittest.TestCase):
    def setUp(self):
        self.peer = SimpleNamespace(peer_failed=False)
        self.peer.report_execution_failure = Mock(
            side_effect=lambda: setattr(self.peer, "peer_failed", True)
        )
        self.communicator = PlaitCommunicator.__new__(PlaitCommunicator)
        self.communicator.pynccl_comm = SimpleNamespace(comm=self.peer)
        self.proc = WorkerProc.__new__(WorkerProc)
        self.error = RuntimeError("failure in the device kernel")
        self.proc.worker = SimpleNamespace(
            execute_model=Mock(side_effect=self.error),
            sample_tokens=Mock(),
            reset_mm_cache=Mock(side_effect=self.error),
        )
        self.proc.handle_output = Mock()
        patches = ExitStack()
        self.addCleanup(patches.close)
        patches.enter_context(
            patch.object(multiproc_executor.envs, "VLLM_USE_PLAIT", True)
        )
        patches.enter_context(
            patch.object(
                multiproc_executor,
                "get_pp_group",
                return_value=SimpleNamespace(device_communicator=None),
            )
        )
        patches.enter_context(
            patch.object(
                multiproc_executor, "model_parallel_is_initialized", return_value=True
            )
        )
        patches.enter_context(
            patch.object(
                multiproc_executor,
                "get_tp_group",
                return_value=SimpleNamespace(device_communicator=self.communicator),
            )
        )

    def test_follower_failure_closes_comm_and_stops_further_model_calls(self):
        self.proc.rank = 2
        self.proc._execute_worker_rpc(("execute_model", (), {}, 0))
        self.assertTrue(self.communicator.failed)
        self.proc._execute_worker_rpc(("sample_tokens", (), {}, 0))
        self.proc.worker.execute_model.assert_called_once_with()
        self.proc.worker.sample_tokens.assert_not_called()
        self.peer.report_execution_failure.assert_called_once_with()
        self.proc.handle_output.assert_not_called()

    def test_primary_failure_reports_the_original_error(self):
        self.proc.rank = 0
        self.proc._execute_worker_rpc(("execute_model", (), {}, 0))
        self.peer.report_execution_failure.assert_called_once_with()
        self.proc.handle_output.assert_called_once_with(self.error)

    def test_pipeline_failure_closes_the_committed_group(self):
        self.proc.rank = 1
        with (
            patch.object(
                multiproc_executor.get_tp_group,
                "return_value",
                SimpleNamespace(device_communicator=None),
            ),
            patch.object(
                multiproc_executor.get_pp_group,
                "return_value",
                SimpleNamespace(device_communicator=self.communicator),
            ),
        ):
            self.proc._execute_worker_rpc(("execute_model", (), {}, 0))
            self.proc._execute_worker_rpc(("sample_tokens", (), {}, 0))
        self.peer.report_execution_failure.assert_called_once_with()
        self.proc.worker.sample_tokens.assert_not_called()

    def test_non_model_rpc_keeps_comm_open(self):
        self.proc.rank = 0
        self.proc._execute_worker_rpc(("reset_mm_cache", (), {}, 0))
        self.peer.report_execution_failure.assert_not_called()
        self.proc.handle_output.assert_called_once_with(self.error)

    def test_report_failure_exits_the_worker_rpc_loop(self):
        self.proc.rank = 2
        self.peer.report_execution_failure.side_effect = RuntimeError(
            "failed to close peer"
        )
        with self.assertRaisesRegex(RuntimeError, "failed to close peer"):
            self.proc._execute_worker_rpc(("execute_model", (), {}, 0))
        self.proc.handle_output.assert_not_called()
