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

"""Spyre OOT replacement for RMSNorm.

Spyre constraints:
    - Dtype-changing ops (fp16->fp32) are not registered as Spyre kernels
      (see torch_spyre/ops/eager.py), so the fp32 promotion is done on CPU and
      the result copied back to Spyre.

References:
    - Upstream RMSNorm: vllm/model_executor/layers/layernorm.py
"""

from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert

logger = init_logger(__name__)


def _rms_norm_fp32_op(
    x: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
    has_residual: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Opaque-op body: the fp32 CPU core of RMSNorm (residual add + variance +
    rsqrt normalize), returning ``(normed, new_residual)`` in the input dtype on
    the input device.

    torch-spyre does not register dtype-changing (fp16->fp32) ops on-device, so
    the promotion + reduction must run on CPU. Hidden behind
    ``torch.ops.vllm.spyre_rms_norm_fp32`` it becomes a single graph node, so the
    CPU ``mean`` never reaches Inductor's cpp backend (which cannot codegen a
    ``mean`` reduction) — it works in **both eager and torch.compile**. The weight
    multiply stays outside the op (a spyre-supported on-device ``mul``).

    When ``has_residual`` is False, ``residual`` is an ignored dummy and the
    returned ``new_residual`` is meaningless (the caller discards it).
    """
    orig_dtype = x.dtype
    device = x.device

    xf = convert(x, device="cpu", dtype=torch.float32)
    if has_residual:
        xf = xf + convert(residual, device="cpu", dtype=torch.float32)
    # Always a fresh tensor (convert copies), so no output aliases an input —
    # required for a functional custom op. Meaningless (and discarded) when
    # has_residual is False.
    new_residual = convert(xf, device=device, dtype=orig_dtype)

    variance = xf.pow(2).mean(dim=-1, keepdim=True)
    xf = xf * torch.rsqrt(variance + eps)
    normed = convert(xf, device=device, dtype=orig_dtype)
    return normed, new_residual


def _rms_norm_fp32_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    eps: float,
    has_residual: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(x)


@lru_cache(maxsize=1)
def register():
    """Register ``torch.ops.vllm.spyre_rms_norm_fp32`` once.

    CompositeExplicitAutograd so it dispatches on Spyre input tensors while the
    body runs the fp32 math on CPU (mirrors ``spyre_convert``)."""
    direct_register_custom_op(
        op_name="spyre_rms_norm_fp32",
        op_func=_rms_norm_fp32_op,
        fake_impl=_rms_norm_fp32_fake,
        dispatch_key="CompositeExplicitAutograd",
    )
    logger.debug_once("Registered custom op: spyre_rms_norm_fp32")


@RMSNorm.register_oot(name="RMSNorm")
class SpyreRMSNorm(RMSNorm):
    """Out-of-tree (OOT) RMSNorm implementation for IBM's Spyre."""

    def forward_oot(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """RMSNorm kernel for Spyre."""

        if self.variance_size_override is not None:
            raise NotImplementedError("TODO: variance_size_override not yet implemented")

        # fp32 promotion, matching upstream RMSNorm: promote to fp32 for the
        # residual add + variance + rsqrt to avoid fp16 overflow of x**2 /
        # precision loss in the residual add when the residual stream is large.
        # torch-spyre does not register dtype-changing ops as Spyre kernels
        # (see torch_spyre/ops/eager.py), so the fp32 math runs on CPU. It is
        # wrapped in the opaque `spyre_rms_norm_fp32` op so the CPU `mean` is not
        # traced into a torch.compile graph (Inductor's cpp backend cannot
        # codegen a `mean` reduction). The weight multiply stays on-device.
        has_residual = residual is not None
        x, new_residual = torch.ops.vllm.spyre_rms_norm_fp32(
            x,
            residual if has_residual else x,
            self.variance_epsilon,
            has_residual,
        )

        if self.has_weight:
            x = x * self.weight
        if not has_residual:
            return x
        else:
            return x, new_residual
