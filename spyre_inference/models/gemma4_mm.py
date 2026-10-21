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

"""Gemma 4 with a compiled, bucketed vision encoder (tensor math from hf-adapters#495).

The tower is stock transformers code, so its attention and norm instances, not classes, switch
to Spyre subclasses. Video is rejected: upstream runs it eagerly through the same tower.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.gemma4 import modeling_gemma4
from vllm.model_executor.models.gemma4_mm import Gemma4ForConditionalGeneration
from vllm.model_executor.models.utils import scatter_output_slices

from spyre_inference.custom_ops.utils import convert
from spyre_inference.models._vision_encoder import (
    GatherTable,
    SpyreVisionEncoderMixin,
    VisionBuffer,
    offset_major,
)
from spyre_inference.models.pixtral import _linear_weight
from spyre_inference.multimodal.gemma4_vision import (
    SpyreGemma4RMSNorm,
    SpyreGemma4VisionAttention,
    _fp32_inv_freq,
    _host,
    _pad_mlp,
    _padded_head_dim,
    _prepare_attention,
    _swap_class,
)
from spyre_inference.multimodal.utils import STICK, align_up

if TYPE_CHECKING:
    from vllm.config import VllmConfig

# Half of fp16's most negative value: a full `finfo.min` saturates to -inf and NaNs SDPA.
_MASK_MIN = torch.finfo(torch.float16).min / 2


def gemma4_rope_tables(
    inv_freq: torch.Tensor, size: int, padded_head_dim: int, attention_scaling: float = 1.0
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-axis cos and signed-sin tables over `_pad_qk_linear`'s `[X, Y, 0 | X, Y, 0]` head:
    a patch at ``(x, y)`` rotates by ``cos[0, x] + cos[1, y]``, ``sin[0, x] + sin[1, y]``."""
    quarter = inv_freq.shape[0]
    half = padded_head_dim // 2
    angles = torch.arange(size, dtype=torch.float32)[:, None] * inv_freq.float()
    cos_axis = angles.cos() * attention_scaling
    sin_axis = angles.sin() * attention_scaling
    cos = torch.zeros(2, size, padded_head_dim)
    sin = torch.zeros(2, size, padded_head_dim)
    for axis in range(2):
        lanes = slice(axis * quarter, (axis + 1) * quarter)
        upper = slice(half + axis * quarter, half + (axis + 1) * quarter)
        cos[axis, :, lanes] = cos_axis
        cos[axis, :, upper] = cos_axis
        sin[axis, :, lanes] = -sin_axis
        sin[axis, :, upper] = sin_axis
    cos[0, :, 2 * quarter : half] = 1.0
    cos[0, :, half + 2 * quarter :] = 1.0
    return cos, sin


@dataclass
class _VisionTail:
    """Not a module, so the model's move to the card leaves it on the host."""

    scale: float
    std_bias: torch.Tensor | None
    std_scale: torch.Tensor | None
    eps: float
    weight: torch.Tensor  # embed_vision's projection, [out, in]
    dtype: torch.dtype


def _unbind(t: Any) -> list[torch.Tensor]:
    return list(t.unbind(0)) if isinstance(t, torch.Tensor) else list(t)


def _images(mm_kwargs: dict[str, Any]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    pixel_values = mm_kwargs.get("pixel_values")
    if pixel_values is None:
        return []
    positions = mm_kwargs["pixel_position_ids"]
    return list(zip(_unbind(pixel_values), _unbind(positions), strict=True))


def _valid_patches(
    pixel_values: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pixel_values = convert(pixel_values, device="cpu")
    positions = convert(positions, device="cpu")
    valid = (positions != -1).any(dim=-1)
    return pixel_values[valid], positions[valid]


class SpyreGemma4ForConditionalGeneration(SpyreVisionEncoderMixin, Gemma4ForConditionalGeneration):
    # The one-hot image matrix is 64 wide, and its last column belongs to the padding.
    vision_max_items_per_call = STICK - 1

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        # Transformers builds the tower at vision_config.dtype (the checkpoint's bf16), not the
        # model's fp16; Spyre stores both as one 16-bit format, but a logical bf16 operand
        # promotes fp16 math to fp32 inside a graph. Loading converts the checkpoint into it.
        model_config = vllm_config.model_config
        model_config.hf_config.vision_config.dtype = model_config.dtype
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._init_spyre_vision()

    def _init_spyre_vision(self) -> None:
        """Separate from ``__init__`` so tests can run it without a ``VllmConfig``."""
        vision_config = self.config.vision_config
        self._pool = vision_config.pooling_kernel_size
        self.patches_per_token = self._pool**2
        self.out_hidden_size = self.config.text_config.hidden_size
        self._patch_width = align_up(3 * vision_config.patch_size**2)
        self.vision_buffers = {
            "patches": VisionBuffer(self._patch_width),
            "segments": VisionBuffer(STICK, pad=F.one_hot(torch.tensor(STICK - 1), STICK).half()),
            "pos_x": VisionBuffer(None, pad=0, dtype=torch.int64),
            "pos_y": VisionBuffer(None, pad=0, dtype=torch.int64),
        }

        # With image and video limits of 0, upstream leaves a placeholder instead of the tower.
        if not isinstance(self.vision_tower, modeling_gemma4.Gemma4VisionModel):
            return
        if vision_config.num_key_value_heads != vision_config.num_attention_heads:
            raise NotImplementedError(
                "Gemma 4 vision GQA is not supported on Spyre; num_key_value_heads "
                "must equal num_attention_heads."
            )
        # In place, not rebuilt: the instances already hold vLLM's replaced linears.
        encoder = self.vision_tower.encoder
        _swap_class(encoder, modeling_gemma4.Gemma4RMSNorm, SpyreGemma4RMSNorm)
        _swap_class(encoder, modeling_gemma4.Gemma4VisionAttention, SpyreGemma4VisionAttention)

    def process_weights_after_loading(self) -> None:
        """Rebuild the vision weights in the graphs' layouts, so one copy of each reaches the
        card. Runs on the host, after the Spyre linears store their weights transposed."""
        tower = self.vision_tower
        if not isinstance(tower, modeling_gemma4.Gemma4VisionModel):
            return
        config = tower.config
        padded = _padded_head_dim(config.head_dim)
        for layer in tower.encoder.layers[: config.num_hidden_layers]:
            _prepare_attention(layer.self_attn, config.num_attention_heads, config.head_dim, padded)
            _pad_mlp(layer, config.intermediate_size, align_up(config.intermediate_size))

        embedder = tower.patch_embedder
        weight = _linear_weight(embedder.input_proj).float()
        # The model's dtype, which __init__ builds the tower in: a bf16 + fp16 add in a graph
        # goes through an fp32 convert whose layout Spyre cannot combine with the other operand.
        dtype = cast(torch.dtype, self.model_dtype)
        weight_t = F.pad(weight, (0, self._patch_width - weight.shape[1])).t().contiguous()
        # A plain holder, not the linear: nothing runs it, so no Spyre-layout copy goes to the card.
        embedder.input_proj = nn.Module()
        embedder.input_proj.weight = nn.Parameter(weight_t.to(dtype), requires_grad=False)

        # One row per coordinate and axis: [position embedding | cos | sin].
        table = _host(cast(torch.Tensor, embedder.position_embedding_table)).float()
        inv_freq, attention_scaling = _fp32_inv_freq(tower.encoder.rotary_emb, config)
        cos, sin = gemma4_rope_tables(inv_freq, table.shape[1], padded, attention_scaling)
        rows = torch.cat([table, cos, sin], dim=-1).to(dtype)
        self.spyre_vision_x = GatherTable(rows[0].contiguous())
        self.spyre_vision_y = GatherTable(rows[1].contiguous())
        del embedder.position_embedding_table

        standardize = getattr(config, "standardize", False)
        projection = self.embed_vision.embedding_projection
        self._spyre_vision_tail = _VisionTail(
            # vision_project sums the k*k patches; the pooler averages and scales them.
            scale=tower.pooler.root_hidden_size / self.patches_per_token,
            std_bias=_host(tower.std_bias).float() if standardize else None,
            std_scale=_host(tower.std_scale).float() if standardize else None,
            eps=self.embed_vision.eps,
            weight=_host(_linear_weight(projection)).float(),
            dtype=dtype,
        )
        # Only upstream's eager image and video paths ran it; the tail above replaces it.
        self.embed_vision = None

    def embed_multimodal(self, **kwargs: object) -> Any:
        if "pixel_values_videos" in kwargs:
            raise NotImplementedError(
                "Gemma 4 video input is not supported on Spyre; send the frames as images."
            )
        return super().embed_multimodal(**kwargs)

    def postprocess_encoder_output(
        self,
        outputs: dict[str, torch.Tensor],
        indices: list[int],
        per_item_out_tokens: list[int],
        dest: dict[int, torch.Tensor] | list[torch.Tensor | None],
        clone: bool = False,
        batch_mm_kwargs: dict[str, Any] | None = None,
    ) -> None:
        """The pool's scale, standardize and ``embed_vision``, in fp32 on the host as upstream
        runs them: the ``sqrt(hidden_size)`` scale can leave the fp16 range."""
        num_tokens = sum(per_item_out_tokens[i] for i in indices)
        tail = self._spyre_vision_tail
        x = outputs["default"][:num_tokens].float() * tail.scale
        if tail.std_bias is not None and tail.std_scale is not None:
            x = (x - tail.std_bias) * tail.std_scale
        # embed_vision: a weightless RMSNorm, then the projection.
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + tail.eps)
        tokens = (x @ tail.weight.t()).to(tail.dtype)
        scatter_output_slices(tokens, indices, per_item_out_tokens, dest, clone)

    def vision_item_tokens(self, mm_kwargs: dict[str, Any]) -> list[int]:
        return [
            int((pos != -1).any(dim=-1).sum()) // self.patches_per_token
            for _, pos in _images(mm_kwargs)
        ]

    def vision_inputs(self, mm_kwargs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """The processor makes both grid sides divide by the pooling kernel, so laid out
        offset-major, the pool is a sum over offset groups."""
        k = self._pool
        patches, segments, xs, ys = [], [], [], []
        for i, (pixel_values, positions) in enumerate(_images(mm_kwargs)):
            values, pos = _valid_patches(pixel_values, positions)
            wp, hp = (int(n) + 1 for n in pos.max(dim=0).values)
            if hp * wp != pos.shape[0] or hp % k or wp % k:
                raise ValueError(
                    f"Gemma 4 image {i}: {pos.shape[0]} valid patches do not fill a "
                    f"{hp}x{wp} grid divisible by the {k}x{k} pooling kernel"
                )
            grid = torch.empty(hp, wp, values.shape[-1], dtype=values.dtype)
            grid[pos[:, 1], pos[:, 0]] = values
            patches.append(offset_major(grid, k))
            xs.append(offset_major(torch.arange(wp).expand(hp, wp), k))
            ys.append(offset_major(torch.arange(hp)[:, None].expand(hp, wp), k))
            segments.append(torch.full((k * k, hp * wp // (k * k)), i))
        # Stock's patch embedder starts with 2 * (x - 0.5).
        flat = (torch.cat(patches, dim=1).flatten(0, 1).float() * 2 - 1).to(torch.float16)
        return {
            "patches": F.pad(flat, (0, self._patch_width - flat.shape[1])),
            "segments": F.one_hot(torch.cat(segments, dim=1).flatten(), STICK).to(torch.float16),
            "pos_x": torch.cat(xs, dim=1).flatten().to(torch.int64),
            "pos_y": torch.cat(ys, dim=1).flatten().to(torch.int64),
        }

    def vision_embed(self, values: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, Any]]:
        tower = self.vision_tower
        hidden = tower.config.hidden_size
        head_dim = _padded_head_dim(tower.config.head_dim)
        rows = self.spyre_vision_x.gather(values["pos_x"]) + self.spyre_vision_y.gather(
            values["pos_y"]
        )
        # [patch width, D] since process_weights_after_loading.
        patch_weight_t = cast(torch.Tensor, tower.patch_embedder.input_proj.weight)
        x = (torch.matmul(values["patches"], patch_weight_t) + rows[:, :hidden]).unsqueeze(0)
        cos = rows[:, hidden : hidden + head_dim][None, :, None]
        sin = rows[:, hidden + head_dim :][None, :, None]
        # Same-image mask as a matmul: Spyre cannot lay out a column-minus-row broadcast.
        segments = values["segments"].to(x.dtype)
        mask = ((1.0 - torch.matmul(segments, segments.t())) * _MASK_MIN)[None, None]
        return x, {"position_embeddings": (cos, sin), "attention_mask": mask}

    def vision_blocks(self) -> nn.ModuleList:
        return self.vision_tower.encoder.layers

    def vision_project(self, x: torch.Tensor, taps: list[torch.Tensor]) -> torch.Tensor:
        """The pool only: the tail runs in ``postprocess_encoder_output``."""
        x = x.squeeze(0)
        return x.reshape(self.patches_per_token, -1, x.shape[-1]).sum(dim=0)
