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

"""Op for Mistral/Llama-4 attention temperature scaling.

Models with ``llama_4_scaling`` (e.g. Ministral-3) run, per attention layer,
``MistralAttention._get_llama_4_attn_scale``::

    1 + beta * log(1 + floor(positions / original_max_position_embeddings))

which promotes the on-device ``positions`` to fp32. Spyre supports fp32 casting,
so the whole thing runs **on-device in fp32** and returns fp16 (keeping the
downstream ``q * attn_scale`` in fp16). The method swap that routes
``_get_llama_4_attn_scale`` through this op is installed in the model runner's
``load_model`` (``_patch_llama4_attn_scale_for_spyre``).

Registered via ``direct_register_custom_op`` with a ``fake_impl``. The opaque
wrapper is no longer strictly required now the body is fully on-device; it's kept
for the stable ``_get_llama_4_attn_scale`` seam and could be inlined later if
fusing the tiny op into the graph is worthwhile.
"""

from functools import lru_cache

import torch

from vllm.logger import init_logger
from vllm.platforms import current_platform
from vllm.utils.torch_utils import direct_register_custom_op

logger = init_logger(__name__)


def _llama4_attn_scale_func(
    positions: torch.Tensor, beta: float, original_max_position_embeddings: int
) -> torch.Tensor:
    """Compute the attention temperature scale **on-device in fp32** (Spyre
    supports fp32 casting), returning fp16 on ``positions``' device so the
    downstream ``q * attn_scale`` stays fp16.

    NOTE: if ``log`` on fp32 is rejected on this torch-spyre revision
    (``log on DataFormats.IEEE_FP32``), cast just the (small-integer) log
    argument to fp16 — ``(1 + floor(pos/orig_max)).to(fp16)`` — since fp16 log of
    values in ~[1, 17] is exact.
    """
    pos = positions.to(torch.float32)
    scaling = 1.0 + beta * torch.log(
        1.0 + torch.floor(pos / original_max_position_embeddings)
    )
    return scaling.unsqueeze(-1).to(torch.float16)


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
