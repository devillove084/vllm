# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch.distributed import ProcessGroup

from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)
from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
from vllm.distributed.plait_rendezvous import CommittedGroup
from vllm.logger import init_logger
from vllm.utils.network_utils import get_ip

logger = init_logger(__name__)


class PlaitCommunicator(DeviceCommunicatorBase):
    """Use Plait peer transport for device tensors across platforms."""

    def __init__(
        self,
        cpu_group: ProcessGroup,
        device: torch.device,
        unique_name: str,
        committed_group: CommittedGroup | None = None,
    ):
        super().__init__(cpu_group, device, unique_name=unique_name)
        self.pynccl_comm = PyNcclCommunicator(
            cpu_group, device, use_peer_group=True, committed_group=committed_group
        )
        if self.pynccl_comm.disabled:
            raise RuntimeError(f"Plait communicator {unique_name} is disabled")
        if (
            committed_group is not None
            and committed_group.compute_calibration is not None
        ):
            self._publish_compute_calibration(committed_group)

    def _publish_compute_calibration(self, group: CommittedGroup) -> None:
        calibration = group.compute_calibration
        assert calibration is not None
        comm = self.pynccl_comm.comm
        address = get_ip()
        try:
            previous = comm.compute_rate_revision(address, group.seeds, calibration.id)
            speed = calibration.measure(self.device)
            revision = comm.publish_compute_rate(
                address, group.seeds, calibration.id, previous, speed
            )
        except (AttributeError, OSError, RuntimeError, ValueError) as exc:
            logger.warning(
                "Skipping Plait compute calibration %s: %s", calibration.id, exc
            )
            return
        logger.info(
            "Plait compute calibration %s at revision %d: %d operations/s",
            calibration.id,
            revision,
            speed,
        )

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        return self.pynccl_comm.all_reduce(input_.contiguous())

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        shards = input_.new_empty((self.world_size, *input_.shape))
        self.pynccl_comm.all_gather(shards, input_.contiguous())
        dim %= input_.dim()
        return shards.movedim(0, dim).flatten(dim, dim + 1)

    def gather(
        self, input_: torch.Tensor, dst: int = 0, dim: int = -1
    ) -> torch.Tensor | None:
        result = self.all_gather(input_, dim)
        return result if self.rank_in_group == dst else None

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        packed = input_.movedim(dim, 0).contiguous()
        if packed.shape[0] % self.world_size:
            raise ValueError("Plait reduce-scatter requires equal-sized shards")
        chunk = packed.shape[0] // self.world_size
        output = input_.new_empty((chunk, *packed.shape[1:]))
        self.pynccl_comm.reduce_scatter(output, packed)
        return output.movedim(0, dim).contiguous()

    def broadcast(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        packed = tensor.contiguous()
        self.pynccl_comm.broadcast(packed, src)
        if packed is not tensor:
            tensor.copy_(packed)
        return tensor

    def send(self, tensor: torch.Tensor, dst: int | None = None) -> None:
        if dst is None:
            dst = (self.rank_in_group + 1) % self.world_size
        self.pynccl_comm.send(tensor.contiguous(), dst)

    @property
    def failed(self) -> bool:
        return getattr(self.pynccl_comm.comm, "peer_failed", False) is True

    def report_execution_failure(self) -> None:
        comm = self.pynccl_comm.comm
        report = getattr(comm, "report_execution_failure", None)
        if report is None:
            self.pynccl_comm.nccl.ncclCommAbort(comm)
        else:
            report()

    def recv(
        self, size: torch.Size, dtype: torch.dtype, src: int | None = None
    ) -> torch.Tensor:
        if src is None:
            src = (self.rank_in_group - 1) % self.world_size
        result = torch.empty(size, dtype=dtype, device=self.device)
        self.recv_into(result, src)
        return result

    def recv_into(self, tensor: torch.Tensor, src: int) -> None:
        self.pynccl_comm.recv(tensor, src)

    def destroy(self) -> None:
        self.pynccl_comm.destroy()
