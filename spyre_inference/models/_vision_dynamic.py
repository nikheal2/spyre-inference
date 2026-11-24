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

"""Pixtral vision graphs over a patch count declared dynamic on dim 0 (#1473, #1787).

torch-spyre compiles a dimension declared with ``.to("spyre", dynamic=...)`` once for its whole
range, but only inside ``for_each_tile``: a plain op over it is refused at compile. So every op
that iterates over the patch rows runs in a loop of ``tile`` rows. A block is three loops:

    rows:       attention norm, q/k/v projections, RoPE
    attention:  per query tile, an online softmax over the key tiles (a nested loop)
    rows:       output projection, residual, FFN norm, FFN, residual

Weights enter each loop as untiled operands, so every trip re-streams them. The key-padding
mask stays: a real patch count rounded up to ``tile`` still has padding slots.

Patches are laid out per tile: each tile holds ``tile / M**2`` image tokens, offset-major
within the tile, so the patch merge stays inside one trip.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F

from spyre_inference.multimodal.utils import align_up, rope_rotate_matmul


class PixtralBlockWeights(NamedTuple):
    """One block's weights, input-major as the Spyre linears store them (``x @ w``)."""

    attention_norm: torch.Tensor  # [D]
    wqkv: torch.Tensor  # [D, 3 * H * head_dim]
    wo: torch.Tensor  # [H * head_dim, D]
    ffn_norm: torch.Tensor  # [D]
    w_gate_up: torch.Tensor  # [D, 2 * F]
    w_down: torch.Tensor  # [F, D]
    rope_perm: torch.Tensor  # [head_dim, head_dim], rope_perm_matrix("pair", ...)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, upcast: bool) -> torch.Tensor:
    """Upstream's ``RMSNorm.forward_native``; ``upcast=False`` stays in fp16 (torch-spyre#6453)."""
    h = x.float() if upcast else x
    h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)
    return h.to(x.dtype) * weight


def tile_offset_major(t: torch.Tensor, tile: int, pad: float) -> torch.Tensor:
    """``[G, tokens, ...]`` offset-major -> ``[rows, ...]`` with each tile offset-major inside.

    Tokens are padded with ``pad`` to whole tiles of ``tile // G`` tokens; the padding tokens
    sit at the tail of the last tile.
    """
    groups, tokens = t.shape[:2]
    per_tile = tile // groups
    padded = align_up(tokens, per_tile)
    out = t.new_full((groups, padded, *t.shape[2:]), pad)
    out[:, :tokens] = t
    out = out.reshape(groups, padded // per_tile, per_tile, *t.shape[2:]).transpose(0, 1)
    return out.flatten(0, 2)


def _check_rows(rows, tile: int, max_rows: int | None) -> None:
    torch._check(rows >= tile)
    torch._check(rows % tile == 0)
    if max_rows is not None:
        torch._check(rows <= max_rows)


def pixtral_embed_dynamic(
    patches: torch.Tensor,
    patch_weight_t: torch.Tensor,
    ln_pre: torch.Tensor,
    *,
    tile: int,
    max_rows: int | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """``ln_pre(patches @ patch_weight_t)`` on ``[P, patch width]`` patches, ``P`` declared."""
    from torch_spyre._inductor.wsr import for_each_tile

    _check_rows(patches.shape[0], tile, max_rows)

    def body(_carry, tiles):
        p_t, w, norm_w = tiles
        return None, rms_norm(torch.matmul(p_t, w), norm_w, eps, upcast=True)

    _, x = for_each_tile(
        body, (patches, patch_weight_t, ln_pre), dims=(0, None, None), tile_size=tile, out_dim=0
    )
    return x


def pixtral_block_dynamic(
    x: torch.Tensor,
    key_pad: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    w: PixtralBlockWeights,
    *,
    num_heads: int,
    tile: int,
    max_rows: int | None = None,
    eps: float = 1e-5,
    upcast: bool = True,
) -> torch.Tensor:
    """``SpyrePixtralTransformerBlock`` on ``[P, D]`` patches, ``P`` declared dynamic.

    Args:
        x: ``[P, D]`` block input.
        key_pad: ``[P, 1]`` additive mask, ``MASK_MIN`` at padding slots. Not ``[P]``: the
            declared dim cannot be the innermost (stick) one.
        cos, sin: ``[P, head_dim]`` RoPE rows, as separate tensors: slicing one gathered
            ``[P, 2 * head_dim]`` table changes the view's geometry and drops the declaration.
        tile: rows per trip, the declaration's granularity; ``P`` is a multiple of it.
        max_rows: the declaration's maximum.
    """
    from torch_spyre._inductor.wsr import for_each_tile

    _check_rows(x.shape[0], tile, max_rows)
    head_dim = cos.shape[-1]
    scale = head_dim**-0.5

    def qkv_body(_carry, tiles):
        x_t, cos_t, sin_t, norm_w, wqkv, perm = tiles
        h = rms_norm(x_t, norm_w, eps, upcast)
        cos_t, sin_t = cos_t.unsqueeze(1), sin_t.unsqueeze(1)
        q, k, v = (
            t.reshape(tile, num_heads, head_dim) for t in torch.matmul(h, wqkv).chunk(3, dim=-1)
        )
        q = rope_rotate_matmul(q, cos_t, sin_t, perm) * scale
        k = rope_rotate_matmul(k, cos_t, sin_t, perm)
        return None, (q, k, v)

    _, (q, k, v) = for_each_tile(
        qkv_body,
        (x, cos, sin, w.attention_norm, w.wqkv, w.rope_perm),
        dims=(0, 0, 0, None, None, None),
        tile_size=tile,
        out_dim=0,
    )

    def kv_body(carry, tiles):
        k_t, v_t, pad_t, q_h = tiles
        scores = torch.matmul(q_h, k_t.permute(1, 2, 0)) + pad_t.transpose(0, 1)
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
        x_t, a_t, wo, norm_w, w_gate_up, w_down = tiles
        h = x_t + torch.matmul(a_t, wo)
        n = rms_norm(h, norm_w, eps, upcast)
        gate, up = torch.matmul(n, w_gate_up).chunk(2, dim=-1)
        return None, h + torch.matmul(F.silu(gate) * up, w_down)

    _, out = for_each_tile(
        ffn_body,
        (x, attn, w.wo, w.ffn_norm, w.w_gate_up, w.w_down),
        dims=(0, 0, None, None, None, None),
        tile_size=tile,
        out_dim=0,
    )
    return out


def pixtral_project_dynamic(
    x: torch.Tensor,
    norm: torch.Tensor | None,
    merge_weight: torch.Tensor | None,
    w_in: torch.Tensor,
    b_in: torch.Tensor | None,
    w_out: torch.Tensor,
    b_out: torch.Tensor | None,
    *,
    tile: int,
    max_rows: int | None = None,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Pre-projector norm, patch merge and adapter on ``[P, D]`` -> ``[P / M**2, D_out]``.

    ``merge_weight`` is ``[M**2, D, D_out]`` (``offset_major_merge_weight``): each tile's rows
    are ``M**2`` offset groups, so the merge is a batched matmul and a sum within the trip.
    """
    from torch_spyre._inductor.wsr import for_each_tile

    _check_rows(x.shape[0], tile, max_rows)
    named = {"norm": norm, "merge": merge_weight, "b_in": b_in, "b_out": b_out}
    present = [k for k, v in named.items() if v is not None]

    def body(_carry, tiles):
        x_t, w_in, w_out, *rest = tiles
        extra = dict(zip(present, rest, strict=True))
        if "norm" in extra:
            x_t = rms_norm(x_t, extra["norm"], eps, upcast=True)
        if "merge" in extra:
            merge = extra["merge"]
            x_t = torch.matmul(x_t.reshape(merge.shape[0], -1, x_t.shape[-1]), merge).sum(dim=0)
        h = torch.matmul(x_t, w_in)
        if "b_in" in extra:
            h = h + extra["b_in"]
        h = torch.matmul(F.gelu(h), w_out)
        if "b_out" in extra:
            h = h + extra["b_out"]
        return None, h

    operands = (x, w_in, w_out, *(named[k] for k in present))
    _, out = for_each_tile(
        body,
        operands,
        dims=(0,) + (None,) * (len(operands) - 1),
        tile_size=tile,
        out_dim=0,
    )
    return out
