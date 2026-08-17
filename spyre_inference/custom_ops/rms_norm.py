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

import torch

from vllm.logger import init_logger
from vllm.model_executor.layers.layernorm import RMSNorm

from .utils import convert

logger = init_logger(__name__)


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
        # (see torch_spyre/ops/eager.py), so the fp32 math runs on CPU and the
        # result is copied back to Spyre as fp16. The residual is stored back in
        # the input dtype (fp16) on-device.
        orig_dtype = x.dtype
        device = x.device
        x = convert(x, device="cpu", dtype=torch.float32)

        if residual is not None:
            x = x + convert(residual, device="cpu", dtype=torch.float32)
            residual = convert(x, device=device, dtype=orig_dtype)

        variance = x.pow(2).mean(dim=-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        x = convert(x, device=device, dtype=orig_dtype)

        if self.has_weight:
            x = x * self.weight
        if residual is None:
            return x
        else:
            return x, residual
