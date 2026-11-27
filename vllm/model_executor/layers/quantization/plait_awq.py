# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dense AWQ with Plait integer activation kernels on the current device stream."""

import ctypes
from functools import cache

import torch
from torch.nn import Parameter

from vllm import envs
from vllm.logger import init_logger
from vllm.utils.torch_utils import direct_register_custom_op

from .auto_awq import _REVERSE_AWQ_PACK_ORDER, AutoAWQConfig, BaseAWQLinearMethod

logger = init_logger(__name__)

SUPPORTED_DEVICE_TYPES = ("cuda", "npu")


def stream_handle(stream: torch.Stream) -> int:
    if stream.device.type == "npu":
        # The accessor submits pending torch_npu tasks before returning the ACL stream.
        return stream.npu_stream
    return stream.cuda_stream


class Shape(ctypes.Structure):
    _fields_ = [
        (name, ctypes.c_uint32) for name in ("m", "n", "k", "group_size", "output_bits")
    ]


@cache
def kernel_library(path: str) -> ctypes.CDLL:
    library = ctypes.CDLL(path)
    library.plait_w4_stream_workspace_v1.argtypes = [
        ctypes.POINTER(Shape),
        ctypes.c_uint32,
    ]
    library.plait_w4_stream_workspace_v1.restype = ctypes.c_size_t
    library.plait_w4_stream_launch_v1.argtypes = [
        ctypes.POINTER(Shape),
        ctypes.c_uint32,
        *([ctypes.c_void_p] * 7),
        ctypes.c_size_t,
        ctypes.c_void_p,
    ]
    library.plait_w4_stream_launch_v1.restype = ctypes.c_int
    return library


def _unpack_awq_words(words: torch.Tensor, shifts: torch.Tensor) -> torch.Tensor:
    lanes = words.unsqueeze(-1).expand(-1, -1, shifts.numel())
    return ((lanes >> shifts) & 15).to(torch.uint8).flatten(1)


def canonical_awq(
    qweight: torch.Tensor,
    qzeros: torch.Tensor,
    scales: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shifts = (
        torch.tensor(_REVERSE_AWQ_PACK_ORDER, dtype=torch.int32, device=qweight.device)
        * 4
    )
    # AWQ words pack output channels. Plait packs adjacent K values.
    codes = _unpack_awq_words(qweight, shifts)
    if codes.size(0) % 2:
        codes = torch.nn.functional.pad(codes, (0, 0, 0, 1))
    weights = (codes[0::2] | (codes[1::2] << 4)).transpose(0, 1).contiguous()
    zeros = _unpack_awq_words(qzeros, shifts)
    return (
        weights,
        scales.transpose(0, 1).contiguous(),
        zeros.transpose(0, 1).contiguous(),
    )


def _projection_fake(
    x: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    zero: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    bits: int,
    path: str,
) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], weights.shape[0]))


def _projection(
    x: torch.Tensor,
    weights: torch.Tensor,
    scales: torch.Tensor,
    zero: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    bits: int,
    path: str,
) -> torch.Tensor:
    if x.ndim == 0 or x.shape[-1] == 0 or group_size < 1 or weights.ndim != 2:
        raise ValueError("invalid Plait AWQ projection dimensions")
    output = _projection_fake(x, weights, scales, zero, bias, group_size, bits, path)
    if not output.numel():
        return output
    if x.dtype != torch.half or x.device.type not in SUPPORTED_DEVICE_TYPES:
        raise ValueError("Plait AWQ needs FP16 activation on a CUDA or NPU device")
    k = x.shape[-1]
    n = weights.shape[0]
    rows = x.numel() // k
    groups = (k + group_size - 1) // group_size
    expected = (
        (weights, (n, (k + 1) // 2), torch.uint8),
        (scales, (n, groups), torch.half),
        (zero, (n, groups), torch.uint8),
    )
    if max(rows, n, k, group_size) > 2**32 - 1 or any(
        tensor.shape != dimensions
        or tensor.dtype != dtype
        or tensor.device != x.device
        or not tensor.is_contiguous()
        for tensor, dimensions, dtype in expected
    ):
        raise ValueError("Plait AWQ weights do not match the activation and grouping")
    if bias is not None and (
        bias.shape != (n,) or bias.dtype != x.dtype or bias.device != x.device
    ):
        raise ValueError("Plait AWQ bias must match the output channels and device")
    shape = Shape(rows, n, k, group_size, 16)
    library = kernel_library(path)
    required = library.plait_w4_stream_workspace_v1(ctypes.byref(shape), bits)
    if required == ctypes.c_size_t(-1).value:
        raise ValueError(f"unsupported Plait W4A{bits} projection shape")
    backend = torch.get_device_module(x.device)
    with backend.device(x.device):
        activation = x.contiguous()
        scratch = torch.empty(required, dtype=torch.uint8, device=x.device)
        bias = bias.contiguous() if bias is not None else None
        status = library.plait_w4_stream_launch_v1(
            ctypes.byref(shape),
            bits,
            activation.data_ptr(),
            weights.data_ptr(),
            scales.data_ptr(),
            zero.data_ptr(),
            None if bias is None else bias.data_ptr(),
            output.data_ptr(),
            scratch.data_ptr(),
            required,
            stream_handle(backend.current_stream(x.device)),
        )
    if status != 0:
        raise RuntimeError(f"Plait W4A{bits} projection launch failed ({status})")
    return output


direct_register_custom_op(
    "plait_w4_projection", _projection, fake_impl=_projection_fake
)


class PlaitAWQLinearMethod(BaseAWQLinearMethod):
    def __init__(self, quant_config: AutoAWQConfig):
        super().__init__(quant_config)
        self.path = envs.VLLM_PLAIT_W4_LIBRARY
        self.bits = envs.VLLM_PLAIT_W4_ACTIVATION_BITS
        if self.path is None or self.bits not in (4, 8):
            raise ValueError(
                "set the Plait W4 library and choose activation bits 4 or 8"
            )
        kernel_library(self.path)
        logger.info_once("Plait dense AWQ uses W4A%d", self.bits)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if (
            layer.scales.dtype != torch.half
            or layer.qweight.device.type not in SUPPORTED_DEVICE_TYPES
        ):
            raise ValueError("Plait AWQ needs FP16 scales on a CUDA or NPU device")
        self.group_size = self.quant_config.group_size
        if self.group_size == -1:
            self.group_size = layer.qweight.shape[0]
        weight, scales, zero = canonical_awq(layer.qweight, layer.qzeros, layer.scales)
        layer.qweight = Parameter(weight, requires_grad=False)
        layer.scales = Parameter(scales, requires_grad=False)
        layer.qzeros = Parameter(zero, requires_grad=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return torch.ops.vllm.plait_w4_projection(
            x,
            layer.qweight,
            layer.scales,
            layer.qzeros,
            bias,
            self.group_size,
            self.bits,
            self.path,
        )
