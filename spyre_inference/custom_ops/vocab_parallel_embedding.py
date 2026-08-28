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

"""Spyre OOT replacement for VocabParallelEmbedding.

Spyre constraints:
    - torch-spyre's on-device embedding gather returns numerically wrong values
      (empirically: forcing the gather to CPU yields coherent output, running it
      on-device — even after DMA-ing the result to a fresh buffer — is garbage).
      So the gather runs on CPU from a cached CPU copy of the weight and only the
      small [tokens, hidden] result is copied to Spyre, mirroring the CPU detours
      in rms_norm.py and the pixtral rope patch.
"""

from functools import lru_cache

import torch
import torch.nn.functional as F

from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.logger import init_logger
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod,
    VocabParallelEmbedding,
    get_masked_input_and_mask,
)
from vllm.utils.torch_utils import direct_register_custom_op

from .utils import convert

logger = init_logger(__name__)

# torch-spyre's per-core addressing limit (``work_division.MAX_SPAN_BYTES``,
# 65535 * 4096 = 255.99 MB). A tensor whose per-core memory span exceeds this is
# silently mis-addressed: torch-spyre only logs CRITICAL and keeps going, so an
# on-device embedding gather returns the WRONG ROWS with no error. Watch for
# "per-core tensor span X MB ... exceeds hardware limit" in the logs.
#
# The embedding table is work-divided over 4 cores, so the per-core span is
# a quarter of the local weight. Measured with an isolated F.embedding
# spyre-vs-CPU sweep: every shape at or under the limit matched CPU, every shape
# over it returned whole wrong rows.
#   Ministral-3-14B: 131072/4 x 5120 x 2 = 320 MB -> over  -> CPU gather
#   granite-8B:       49152/4 x 4096 x 2 =  96 MB -> under -> on-card gather
_MAX_SPAN_BYTES = 65535 * 4096
_EMBED_CORE_SPLIT = 4


@VocabParallelEmbedding.register_oot(name="VocabParallelEmbedding")
class SpyreVocabParallelEmbedding(VocabParallelEmbedding):
    """Out-of-tree (OOT) VocabParallelEmbedding implementation for IBM's Spyre device."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if not isinstance(self.quant_method, UnquantizedEmbeddingMethod):
            raise NotImplementedError(
                f"SpyreVocabParallelEmbedding does not support quantized "
                f"embeddings (got {type(self.quant_method).__name__})."
            )
        # Lazily-cached CPU copy of the weight for the CPU gather (see forward).
        self._cpu_weight = None
        # Whether this weight's per-core span exceeds what Spyre can address (see
        # forward). `self.weight` is already the local TP shard, so TP is accounted
        # for on top of the work-division split.
        span = (self.weight.data.numel() // _EMBED_CORE_SPLIT) * self.weight.data.element_size()
        self._gather_on_cpu = span > _MAX_SPAN_BYTES
        logger.info(
            "SpyreVocabParallelEmbedding: per-core span %.1f MB (limit %.1f MB) -> gathering on %s",
            span / (1024 * 1024),
            _MAX_SPAN_BYTES / (1024 * 1024),
            "CPU" if self._gather_on_cpu else "device",
        )

    def forward(self, input_: torch.Tensor) -> torch.Tensor:
        if self.tp_size > 1:
            # The per-rank mask still runs on CPU: upstream get_masked_input_and_mask
            # does `input_ >= start` under torch.compile, which Spyre's inductor backend
            # rejects for int64 constants (see test_int64_compiled_compare_against_python_int).
            # The embedding gather itself runs on-device below.
            masked_input, keep = torch.ops.vllm.spyre_vocab_mask(
                convert(input_, device="cpu"),
                self.shard_indices.org_vocab_start_index,  # ty: ignore[invalid-argument-type]
                self.shard_indices.org_vocab_end_index,  # ty: ignore[invalid-argument-type]
                self.shard_indices.num_org_vocab_padding,  # ty: ignore[invalid-argument-type]
                self.shard_indices.added_vocab_start_index,  # ty: ignore[invalid-argument-type]
                self.shard_indices.added_vocab_end_index,  # ty: ignore[invalid-argument-type]
                self.weight.data.dtype,  # ty: ignore[invalid-argument-type]
            )
            masked_input = convert(masked_input, device=input_.device)
            keep = convert(keep, device=input_.device)
        else:
            masked_input = input_
            keep = None

        # An over-limit weight cannot be addressed on-card, and the failure is silent
        # (wrong rows, no error), so gather on CPU from a cached CPU copy; only the
        # small [tokens, hidden] result crosses back to Spyre. Weights that fit stay
        # on-card and avoid both the host copy and the round trip.
        if self._gather_on_cpu:
            if self._cpu_weight is None:
                self._cpu_weight = self.weight.data.detach().to("cpu")
            ids = convert(masked_input, device="cpu").long()
            output = convert(F.embedding(ids, self._cpu_weight), device=input_.device)
        else:
            output = F.embedding(masked_input.long(), self.weight.data)

        if keep is not None:
            output = output * keep
            output = tensor_model_parallel_all_reduce(output)
        return output


def _vocab_mask_op_func(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    device = input_.device
    masked_input, input_mask = get_masked_input_and_mask(
        input_,
        org_vocab_start_index,
        org_vocab_end_index,
        num_org_vocab_padding,
        added_vocab_start_index,
        added_vocab_end_index,
    )
    keep = (~input_mask).to(dtype=dtype).unsqueeze(-1)
    return masked_input.to(device), keep.to(device)


def _vocab_mask_op_fake(
    input_: torch.Tensor,
    org_vocab_start_index: int,
    org_vocab_end_index: int,
    num_org_vocab_padding: int,
    added_vocab_start_index: int,
    added_vocab_end_index: int,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_input = torch.empty(input_.shape, dtype=input_.dtype, device=input_.device)
    keep = torch.empty((*input_.shape, 1), dtype=dtype, device=input_.device)
    return masked_input, keep


@lru_cache(maxsize=1)
def register():
    """Register the spyre_vocab_mask custom op with vLLM."""
    direct_register_custom_op(
        op_name="spyre_vocab_mask",
        op_func=_vocab_mask_op_func,
        fake_impl=_vocab_mask_op_fake,
        mutates_args=[],
        dispatch_key="CPU",
    )
    logger.debug_once("Registered custom op: spyre_vocab_mask")
