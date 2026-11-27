# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm.distributed.plait_calibration import GemmCalibration
from vllm.distributed.plait_rendezvous import CommittedGroup


class TestPlaitCalibration(unittest.TestCase):
    def test_id_includes_shape_and_precision(self):
        base = GemmCalibration(64, 1024, 1024, "float16")
        other_shape = GemmCalibration(32, 1024, 1024, "float16")
        other_dtype = GemmCalibration(64, 1024, 1024, "bfloat16")
        self.assertEqual(base.id, "a0ffd44e6a9b8fcdb867751039f09647")
        self.assertEqual(len({base.id, other_shape.id, other_dtype.id}), 3)

    def test_requires_positive_integer_shape(self):
        for dim in (0, -1, 1.5, True):
            with (
                self.subTest(dim=dim),
                self.assertRaisesRegex(ValueError, "positive integer dimensions"),
            ):
                GemmCalibration(dim, 1024, 1024, "float16")

    def test_committed_group_calibration_is_optional(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "groups.json"
            host = "00" * 16
            group = {"ranks": [0], "devices": [[host, 0]], "version": "01" * 16 + "@1"}
            config = {
                "host_id": host,
                "seeds": ["127.0.0.1:7001"],
                "tp_groups": [group],
            }
            path.write_text(json.dumps(config))
            self.assertIsNone(
                CommittedGroup.load(str(path), [0], 0).compute_calibration
            )
            group["compute_calibration"] = {
                "m": 64,
                "n": 1024,
                "k": 1024,
                "dtype": "float16",
            }
            path.write_text(json.dumps(config))
            self.assertEqual(
                CommittedGroup.load(str(path), [0], 0).compute_calibration,
                GemmCalibration(64, 1024, 1024, "float16"),
            )

    def test_pipeline_binding_from_the_same_device_grants(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "groups.json"
            host = "00" * 16
            devices = [[host, 0], ["01" * 16, 1]]
            group = {"ranks": [0, 1], "devices": devices, "version": "02" * 16 + "@3"}
            path.write_text(
                json.dumps({"host_id": host, "seeds": [], "pp_groups": [group]})
            )
            self.assertIsNone(CommittedGroup.load(str(path), [0], 0, "tp"))
            pp = CommittedGroup.load(str(path), [0, 1], 0, "pp")
            self.assertIsNotNone(pp)
            self.assertEqual(pp.devices, [tuple(device) for device in devices])
