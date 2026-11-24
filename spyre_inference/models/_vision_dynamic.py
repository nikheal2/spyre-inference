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

"""A Pixtral vision block over a patch count declared dynamic on dim 0 (#1473).

torch-spyre compiles a dimension declared with ``.to("spyre", dynamic=...)`` once for its whole
range, but only inside ``for_each_tile``: a plain op over it is refused at compile. So every op
that iterates over the patch rows runs in one of three loops of ``tile`` rows:

    rows:       attention norm, q/k/v projections, RoPE
    attention:  per query tile, an online softmax over the key tiles (a nested loop)
    rows:       output projection, residual, FFN norm, FFN, residual

Weights enter each loop as untiled operands, so every trip re-streams them. The key-padding
mask stays: a real patch count rounded up to ``tile`` still has padding slots.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from spyre_inference.multimodal.utils import rope_rotate_matmul


class PixtralBlockWeights(NamedTuple):
    """One block's weights, input-major so each projection is ``x @ w``."""

    attention_norm: torch.Tensor  # [D]
    wq: torch.Tensor  # [D, H * head_dim]
    wk: torch.Tensor  # [D, H * head_dim]
    wv: torch.Tensor  # [D, H * head_dim]
    wo: torch.Tensor  # [H * head_dim, D]
    ffn_norm: torch.Tensor  # [D]
    w_gate: torch.Tensor  # [D, F]
    w_up: torch.Tensor  # [D, F]
    w_down: torch.Tensor  # [F, D]
    rope_perm: torch.Tensor  # [head_dim, head_dim], rope_perm_matrix("pair", ...)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, upcast: bool) -> torch.Tensor:
    """Upstream's ``RMSNorm.forward_native``; ``upcast=False`` stays in fp16 (torch-spyre#6453)."""
    h = x.float() if upcast else x
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return h.to(x.dtype) * weight


def pixtral_block_dynamic(
    x: torch.Tensor,
    key_pad: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w: PixtralBlockWeights,
    *,
    num_heads: int,
    tile: int,
    max_rows: int,
    eps: float = 1e-5,
    upcast: bool = True,
) -> torch.Tensor:
    """``SpyrePixtralTransformerBlock`` on ``[P, D]`` patches, ``P`` declared dynamic.

    Args:
        x: ``[P, D]`` block input.
        key_pad: ``[P]`` additive mask, ``MASK_MIN`` at padding slots.
        cos, sin: ``[P, head_dim]`` RoPE rows, as separate tensors: slicing one gathered
            ``[P, 2 * head_dim]`` table changes the view's geometry and drops the declaration.
        tile: rows per trip, the declaration's granularity; ``P`` is a multiple of it.
        max_rows: the declaration's maximum.
    """
    from torch_spyre._inductor.wsr import for_each_tile

    rows = x.shape[0]
    torch._check(rows >= tile)
    torch._check(rows <= max_rows)
    torch._check(rows % tile == 0)
    head_dim = cos.shape[-1]
    scale = head_dim**-0.5

    def qkv_body(_carry, tiles):
        x_t, cos_t, sin_t, norm_w, wq, wk, wv, perm = tiles
        h = rms_norm(x_t, norm_w, eps, upcast)
        cos_t, sin_t = cos_t.unsqueeze(1), sin_t.unsqueeze(1)

        def heads(weight):
            return torch.matmul(h, weight).reshape(tile, num_heads, head_dim)

        q = rope_rotate_matmul(heads(wq), cos_t, sin_t, perm) * scale
        k = rope_rotate_matmul(heads(wk), cos_t, sin_t, perm)
        return None, (q, k, heads(wv))

    _, (q, k, v) = for_each_tile(
        qkv_body,
        (x, cos, sin, w.attention_norm, w.wq, w.wk, w.wv, w.rope_perm),
        dims=(0, 0, 0, None, None, None, None, None),
        tile_size=tile,
        out_dim=0,
    )

    def kv_body(carry, tiles):
        k_t, v_t, pad_t, q_h = tiles
        scores = torch.matmul(q_h, k_t.permute(1, 2, 0)) + pad_t
        run_max, run_sum, run_out = carry
        tile_max = torch.amax(scores, dim=-1, keepdim=True)
        # Read run_max before the maximum that supersedes it, as page_attn_prefill does.
        rescale = torch.exp(-torch.relu(tile_max - run_max))
        new_max = torch.maximum(run_max, tile_max)
        probs = torch.exp(scores - new_max)
        new_sum = run_sum * rescale + probs.sum(dim=-1, keepdim=True)
        new_out = run_out * rescale + torch.matmul(probs, v_t.transpose(0, 1))
        return (new_max, new_sum, new_out), None

    def attn_body(_carry, tiles):
        q_t, k, v, key_pad = tiles
        q_h = q_t.transpose(0, 1)
        state = {"dtype": q_t.dtype, "device": q_t.device}
        init = (
            torch.full((num_heads, tile, 1), float("-inf"), **state),
            torch.zeros((num_heads, tile, 1), **state),
            torch.zeros((num_heads, tile, head_dim), **state),
        )
        (_, run_sum, run_out), _ = for_each_tile(
            kv_body, (k, v, key_pad, q_h), dims=(0, 0, 0, None), tile_size=tile, init=init
        )
        out = (run_out / run_sum).transpose(0, 1)
        return None, out.reshape(tile, num_heads * head_dim)

    _, attn = for_each_tile(
        attn_body, (q, k, v, key_pad), dims=(0, None, None, None), tile_size=tile, out_dim=0
    )

    def ffn_body(_carry, tiles):
        x_t, a_t, wo, norm_w, w_gate, w_up, w_down = tiles
        h = x_t + torch.matmul(a_t, wo)
        n = rms_norm(h, norm_w, eps, upcast)
        ffn = torch.matmul(F.silu(torch.matmul(n, w_gate)) * torch.matmul(n, w_up), w_down)
        return None, h + ffn

    _, out = for_each_tile(
        ffn_body,
        (x, attn, w.wo, w.ffn_norm, w.w_gate, w.w_up, w.w_down),
        dims=(0, 0, None, None, None, None, None),
        tile_size=tile,
        out_dim=0,
    )
    return out
