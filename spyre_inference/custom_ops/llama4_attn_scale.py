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

"""Opaque op for Mistral/Llama-4 attention temperature scaling.

Models with ``llama_4_scaling`` (e.g. Ministral-3) run, per attention layer,
``MistralAttention._get_llama_4_attn_scale``::

    1 + beta * log(1 + floor(positions / original_max_position_embeddings))

The ``div``/``floor``/``log`` execute in fp32, and Spyre has no fp32 ``log``
(``log on DataFormats.IEEE_FP32``). Wrapping the whole thing in an opaque custom
op (registered via ``direct_register_custom_op`` with a ``fake_impl``) means:

- **eager**: the op body runs eagerly — the fp32 math is done on CPU and an fp16
  result is returned on the query device (exact precision, no fp16 rounding of
  large positions);
- **torch.compile**: Dynamo sees a single black-box node, so the CPU math is
  never traced into the Spyre graph — no CPU-resident intermediate for a
  Spyre-only op (e.g. ``spyre::to_dtype_cpu``) to choke on.

Mirrors the ``spyre_rope_rot`` opaque-op pattern in ``rotary_embedding.py``. The
method swap that routes ``_get_llama_4_attn_scale`` through this op is installed
in the model runner's ``load_model`` (``_patch_llama4_attn_scale_for_spyre``).
"""

from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert

logger = init_logger(__name__)


def _llama4_attn_scale_func(
    positions: torch.Tensor, beta: float, original_max_position_embeddings: int
) -> torch.Tensor:
    """Compute the attention temperature scale on CPU (Spyre lacks fp32 ``log``),
    returning fp16 on ``positions``' device so the downstream ``q * attn_scale``
    stays on-device fp16."""
    pos = convert(positions, device="cpu")
    scaling = 1.0 + beta * torch.log(
        1.0 + torch.floor(pos / original_max_position_embeddings)
    )
    scaling = scaling.unsqueeze(-1)
    return convert(scaling, device=positions.device, dtype=torch.float16)


def _llama4_attn_scale_fake(
    positions: torch.Tensor, beta: float, original_max_position_embeddings: int
) -> torch.Tensor:
    return torch.empty(
        (positions.shape[0], 1), dtype=torch.float16, device=positions.device
    )


@lru_cache(maxsize=1)
def register():
    """Register the ``spyre_llama4_attn_scale`` opaque op."""
    direct_register_custom_op(
        op_name="spyre_llama4_attn_scale",
        op_func=_llama4_attn_scale_func,
        fake_impl=_llama4_attn_scale_fake,
        dispatch_key=current_platform.dispatch_key,
    )
    logger.debug_once("Registered custom op: spyre_llama4_attn_scale")
