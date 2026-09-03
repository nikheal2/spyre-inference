# Copyright 2026 The Spyre-Inference Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeviceCommunicator override for IBM Spyre devices.

`all_reduce` uses `torch.ops._c10d_functional`, which torch-spyre lowers to
`spyre::all_reduce_async` inside a compiled graph and which falls through to
eager spyreccl outside one, so the reduction compiles into the model graph
without needing a separate eager path.

`all_gather` cannot — see the method. `reduce_scatter` raises; `broadcast`,
`send` and `recv` are inherited unchanged.
"""

from __future__ import annotations

import math

import torch
import torch.distributed as dist
import torch.nn.functional as F
from vllm.distributed.device_communicators.base_device_communicator import (
    DeviceCommunicatorBase,
)

from spyre_inference.custom_ops.utils import convert
from spyre_inference.v1.pool import select_rows

# spyre-comms cannot build a work schedule for a collective whose 128-byte stick
# count is above the 32-core width without dividing it: hidden 5120 is 80 sticks
# per token, and dxp_standalone exits 1 on the 1-token TP=2 reduction.
_COLLECTIVE_CORES = 32
_STICK_BYTES = 128


def collective_row_pad(input_: torch.Tensor) -> int:
    """Zero rows to append so the element count spreads evenly over the cores."""
    if input_.dim() < 2 or input_.shape[0] == 0:
        return 0
    rows = input_.shape[0]
    row_bytes = (input_.numel() // rows) * input_.element_size()
    if row_bytes % _STICK_BYTES:
        return 0
    row_sticks = row_bytes // _STICK_BYTES
    sticks = rows * row_sticks
    if sticks <= _COLLECTIVE_CORES or sticks % _COLLECTIVE_CORES == 0:
        return 0
    step = _COLLECTIVE_CORES // math.gcd(row_sticks, _COLLECTIVE_CORES)
    return -(-rows // step) * step - rows


class SpyreCommunicator(DeviceCommunicatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Resolved once: a ProcessGroup object in dynamo's path breaks the trace.
        self._group_name: str | None = None
        if self.device_group is not None:
            from torch.distributed._functional_collectives import _resolve_group_name

            self._group_name = _resolve_group_name(self.device_group)

    def all_reduce(self, input_: torch.Tensor) -> torch.Tensor:
        if self.world_size == 1:
            return input_
        if input_.device.type == "cpu" or self._group_name is None:
            return super().all_reduce(input_)

        pad_rows = collective_row_pad(input_)
        reduced = input_
        if pad_rows:
            # Rows, not the last dim: a row is a whole number of sticks, so the
            # layout is unchanged and zeros are the identity for a sum.
            reduced = F.pad(input_, (0, 0) * (input_.dim() - 1) + (0, pad_rows))

        # Out-of-place, unlike the base class's in-place `dist.all_reduce`: vLLM's
        # `torch.ops.vllm.all_reduce` wrapper declares no mutation, so under
        # torch.compile functionalization misses the overwrite and the graph
        # computes garbage. Inductor's reinplacing pass recovers the in-place op.
        out = torch.ops._c10d_functional.all_reduce(
            reduced,  # ty: ignore[invalid-argument-type]
            "sum",  # ty: ignore[invalid-argument-type]
            self._group_name,  # ty: ignore[invalid-argument-type]
        )
        out = torch.ops._c10d_functional.wait_tensor(out)
        if pad_rows:
            # index_select, not a slice: the result must not alias the collective's
            # output buffer, which the next same-shaped reduction overwrites.
            # int32 at creation, not int64: torch-spyre has no int64->int32 typecast,
            # so the conversion inside select_rows would lower to `spyre::to_dtype_cpu`
            # -- an op registered for spyre tensors only -- on this CPU index tensor.
            out = select_rows(out, torch.arange(input_.shape[0], dtype=torch.int32))
        return out

    # libspyre_comms allgather transfers each rank's buffer in 64-element chunks
    # along the gathered dim, so a shard whose size along `dim` is not a multiple
    # of 64 has its tail rounded off and every later rank's data lands shifted.
    # Worse, on comms build 121 such a gather faults the card and needs a device
    # recovery, so do not drop this padding without re-checking on hardware.
    _GATHER_ALIGN = 64

    def all_gather(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Two independent blockers keep this off the functional form `all_reduce`
        # uses: in eager, `_c10d_functional.all_gather_into_tensor` routes to
        # `allgather_into_tensor_coalesced`, which spyreccl rejects; compiled, it
        # lowers to `spyre::all_gather_async`, whose reassembly narrows the output
        # along dim 0 at a `rank * per_rank_numel` storage offset that
        # `copy_from_d2d` requires to be 64-aligned.
        if self.world_size == 1:
            return input_
        if input_.device.type == "cpu":
            return super().all_gather(input_, dim)

        dim = dim % input_.dim()
        orig_size = input_.shape[dim]
        pad = (-orig_size) % self._GATHER_ALIGN
        if not pad:
            output_list = [torch.empty_like(input_) for _ in range(self.world_size)]
            dist.all_gather(  # ty: ignore[possibly-missing-attribute]
                output_list, input_, group=self.device_group
            )
            return torch.cat(output_list, dim=dim)

        # Padded on CPU: on device the tail starts `orig_size % 64` elements into a
        # stick, which torch-spyre's layout pass cannot express ("no offset-free
        # alternative stick dim for mutation target"). Stripped and re-concatenated
        # on CPU too, since Spyre narrow corrupts memory (see spyre_attn.py).
        pad_spec = [0, 0] * (input_.dim() - dim - 1) + [0, pad]
        padded = convert(
            torch.nn.functional.pad(convert(input_, device="cpu"), pad_spec).contiguous(),
            device=input_.device,
        )
        output_list = [torch.empty_like(padded) for _ in range(self.world_size)]
        dist.all_gather(  # ty: ignore[possibly-missing-attribute]
            output_list, padded, group=self.device_group
        )
        stripped = [convert(o, device="cpu").narrow(dim, 0, orig_size) for o in output_list]
        return convert(torch.cat(stripped, dim=dim), device=input_.device)

    def reduce_scatter(self, input_: torch.Tensor, dim: int = -1) -> torch.Tensor:
        # Not on the standard TP path; raise loudly if anything tries it.
        if self.world_size == 1:
            return input_
        raise NotImplementedError(
            f"SpyreCommunicator: reduce_scatter has no Spyre implementation and no "
            f"fallback for world_size={self.world_size}. Either wait for the upstream "
            f"comms implementation to land + a comms RPM rebuild, or extend "
            f"SpyreCommunicator with a manual fallback."
        )
