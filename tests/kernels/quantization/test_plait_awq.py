# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ctypes
import os
import unittest

import torch


def pack_awq(codes: torch.Tensor) -> torch.Tensor:
    packed_order = [0, 2, 4, 6, 1, 3, 5, 7]
    words = codes.reshape(codes.shape[0], -1, 8).to(torch.int64)
    shifts = torch.arange(8) * 4
    return (words[..., packed_order] << shifts).sum(dim=-1).to(torch.int32)


def reference(x, codes, scales, zero, bias, group_size, bits):
    limit = (1 << (bits - 1)) - 1
    result = torch.zeros(x.shape[0], codes.shape[1])
    for group, begin in enumerate(range(0, x.shape[1], group_size)):
        end = begin + group_size
        values = x[:, begin:end].float()
        scale = values.abs().amax(dim=1) / limit
        scale[scale == 0] = 1
        activation = (values / scale[:, None]).round().clamp(-limit, limit).int()
        weight = codes[begin:end].int() - zero[group].int()
        result += (activation @ weight) * (scale[:, None] * scales[group].float())
    return (result + bias.float()).half()


class PlaitAWQTest(unittest.TestCase):
    def test_awq_words_repack_as_k_neighbors_including_odd_tail(self):
        from vllm.model_executor.layers.quantization.plait_awq import canonical_awq

        codes = (torch.arange(3 * 16).reshape(3, 16) % 13).to(torch.uint8)
        zero = codes[1:]
        scales = torch.arange(32, dtype=torch.float16).reshape(2, 16)
        weight, scale, zeros = canonical_awq(pack_awq(codes), pack_awq(zero), scales)
        torch.testing.assert_close(weight[:, 0] & 15, codes[0])
        torch.testing.assert_close(weight[:, 0] >> 4, codes[1])
        torch.testing.assert_close(weight[:, 1], codes[2])
        torch.testing.assert_close(scale, scales.T)
        torch.testing.assert_close(zeros, zero.T)

    @unittest.skipUnless(
        os.environ.get("VLLM_PLAIT_W4_LIBRARY") and torch.cuda.is_available(),
        "requires a Plait integer kernel build for the accelerator",
    )
    def test_integer_projection_on_caller_stream(self):
        from vllm.model_executor.layers.quantization.plait_awq import (
            Shape,
            canonical_awq,
            kernel_library,
        )

        library_path = os.environ["VLLM_PLAIT_W4_LIBRARY"]
        for bits in (4, 8):
            for m, n, k, group_size in (
                (1, 64, 128, 64),
                (16, 128, 128, 64),
                (3, 16, 9, 3),
            ):
                with self.subTest(bits=bits, shape=(m, n, k), group_size=group_size):
                    generator = torch.Generator().manual_seed(19)
                    groups = (k + group_size - 1) // group_size
                    codes = torch.randint(
                        16, (k, n), dtype=torch.uint8, generator=generator
                    )
                    zeros = torch.randint(
                        16, (groups, n), dtype=torch.uint8, generator=generator
                    )
                    scales = (
                        torch.randint(1, 5, (groups, n), generator=generator).half()
                        / 16
                    )
                    bias = torch.arange(n, dtype=torch.float16) / 64
                    x = torch.randint(-29, 30, (m, k), generator=generator).half() / 16
                    expected = reference(
                        x, codes, scales, zeros, bias, group_size, bits
                    )
                    p_w, p_s, p_z = canonical_awq(
                        pack_awq(codes), pack_awq(zeros), scales
                    )
                    p_w, p_s, p_z, x_device, bias_device = (
                        tensor.cuda() for tensor in (p_w, p_s, p_z, x, bias)
                    )
                    stream = torch.cuda.Stream()
                    stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(stream):
                        result = torch.ops.vllm.plait_w4_projection(
                            x_device,
                            p_w,
                            p_s,
                            p_z,
                            bias_device,
                            group_size,
                            bits,
                            library_path,
                        )
                        rejected = torch.full_like(result, 42)
                        shape = Shape(m, n, k, group_size, 16)
                        library = kernel_library(library_path)
                        size = library.plait_w4_stream_workspace_v1(
                            ctypes.byref(shape), bits
                        )
                        scratch = torch.empty(size, dtype=torch.uint8, device="cuda")
                        status = library.plait_w4_stream_launch_v1(
                            ctypes.byref(shape),
                            bits,
                            x_device.data_ptr(),
                            p_w.data_ptr(),
                            p_s.data_ptr(),
                            p_z.data_ptr(),
                            bias_device.data_ptr(),
                            rejected.data_ptr(),
                            scratch.data_ptr(),
                            size - 1,
                            stream.cuda_stream,
                        )
                        self.assertNotEqual(status, 0)
                        x_device[0, 0] = torch.inf
                        invalid = torch.ops.vllm.plait_w4_projection(
                            x_device,
                            p_w,
                            p_s,
                            p_z,
                            bias_device,
                            group_size,
                            bits,
                            library_path,
                        )
                    stream.synchronize()
                    torch.testing.assert_close(
                        result.cpu(), expected, atol=0.002, rtol=0.002
                    )
                    self.assertTrue(torch.all(rejected == 42).item())
                    self.assertTrue(torch.isnan(invalid[0]).all().item())
                    torch.testing.assert_close(
                        invalid[1:].cpu(), expected[1:], atol=0.002, rtol=0.002
                    )
