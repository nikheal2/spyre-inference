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

"""Helpers shared by the vision-tower workarounds in this package."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from spyre_inference.custom_ops.utils import convert

# Spyre stick width in 2-byte elements (128-byte stick). Matmul reduction dims and
# the sequence axis must land on it.
STICK = 64


def align_up(n: int, align: int = STICK) -> int:
    return (n + align - 1) // align * align


# Attribute under which a source mask caches its padded counterpart `(key, padded)`.
# `padded` is None when the source attends everywhere; see `_padded_attn_mask`.
_MASK_ATTR = "_spyre_padded_mask"

# Largest `n_pad / seq_pad` at which the mask may be dropped for the fused kernel.
# Dropping it leaves the zero-padded keys attended: each scores exactly 0 and adds
# nothing through its zero `v` row, but it still takes softmax mass off the real keys,
# shrinking the output by at most that ratio. 0.5% at Pixtral's 3120→3136 — below fp16
# tower noise — but 22% at CLIP's 50→64, which is why this is a ratio and not a flag.
_MASKLESS_PAD_BOUND = 0.01


def _mask_attends_everywhere(mc: torch.Tensor) -> bool:
    """True when a CPU source mask permits every (query, key) pair."""
    if mc.dtype == torch.bool:
        return bool(mc.all())
    # Additive form. A nonzero constant would be softmax-invariant too, but no caller
    # produces one, so only the all-zero case is claimed.
    return bool((mc == 0).all())


def _padded_attn_mask(
    mask: torch.Tensor,
    b: int,
    seq: int,
    seq_pad: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """Additive `[b, 1, seq_pad, seq_pad]` mask on `device`, or None to attend everywhere.

    None is returned when the source mask permits every pair *and* the key padding is
    within `_MASKLESS_PAD_BOUND`. The first is the single-image case — a block-diagonal
    mask over one block is all-ones and carries no information — and the second is what
    keeps the dropped `-inf` columns harmless. `padded_sdpa` then calls SDPA with no
    `attn_mask`, the only form that reaches the fused attention kernel; see its
    docstring for what that is worth.

    O(L²) and shared by every layer, so both outcomes are cached on the source mask:
    one upload per image, released with its source.
    """
    key = (b, seq, seq_pad, dtype, str(device))
    cached = getattr(mask, _MASK_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]

    mc = convert(mask, "cpu")
    if (seq_pad - seq) / seq_pad <= _MASKLESS_PAD_BOUND and _mask_attends_everywhere(mc):
        setattr(mask, _MASK_ATTR, (key, None))
        return None

    # Assembled on CPU: strided slice-assign is not stick-safe on Spyre.
    neg_inf = torch.finfo(dtype).min
    m = torch.zeros(b, 1, seq_pad, seq_pad, dtype=dtype)
    m[:, :, :, seq:] = neg_inf  # padded keys never attended
    if mc.dtype == torch.bool:
        m[:, :, :seq, :seq] = torch.zeros(seq, seq, dtype=dtype).masked_fill(
            ~mc.reshape(seq, seq), neg_inf
        )
    else:
        m[:, :, :seq, :seq] = mc.to(dtype).reshape(seq, seq)

    m = convert(m, device)
    setattr(mask, _MASK_ATTR, (key, m))
    return m


def padded_sdpa(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor,
    scale: float | None = None,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """SDPA over `[B, H, L, D]` with L and D padded to the stick, then cropped.

    At a sequence length coprime with the stick, stock SDPA either fails to
    restickify a batch-matmul operand or returns silently wrong values, so the
    padding is a correctness requirement rather than a tuning choice. Padded queries
    are cropped off the output.

    A mask that attends everywhere is dropped rather than materialized, because
    `attn_mask=None` is the only form that reaches the fused attention kernel. With a
    mask, Spyre softmaxes a *materialized* score matrix in three passes over memory —
    at Pixtral's 3136 patches that is 314 MB live per layer plus a 39 MB mask read per
    pass, and it measured 172 ms/layer against 12 ms for the fused kernel.

    The cost is the padded keys, which then have no `-inf` column: the output shrinks
    by at most `n_pad / seq_pad` (see `_MASKLESS_PAD_BOUND`, which is why a long,
    barely-padded sequence takes this path and a short one does not). Head-dim padding
    is exact either way — its `q`/`k` lanes are zero, so they add nothing to the dot
    product, and its `v` lanes are cropped off.

    `scale` defaults to the head dim seen here, which assumes `q`/`k`/`v` arrive unpadded
    so the padding cannot change it. Pass it explicitly when the head dim is already
    padded, or when the model carries its own scale.
    """
    b, _, seq, d = q.shape
    if scale is None:
        scale = d**-0.5
    seq_pad = align_up(seq)
    d_pad = align_up(d)
    device = q.device
    padded = (seq_pad, d_pad) != (seq, d)

    if padded:
        # F.pad's tuple runs from the last dim backwards: (D left, D right, L left, L right).
        pad = (0, d_pad - d, 0, seq_pad - seq)
        q = F.pad(q, pad)
        k = F.pad(k, pad)
        v = F.pad(v, pad)
    else:
        # Offset operands read as offset 0 (torch-spyre#3770), so SDPA is silently
        # wrong here; the padded branch escapes it only because F.pad materializes.
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()

    out = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=_padded_attn_mask(mask, b, seq, seq_pad, q.dtype, device),
        is_causal=False,
        scale=scale,
        enable_gqa=enable_gqa,
    )

    if padded:
        # Offset-0 prefix slice, so torch-spyre#3770 cannot bite. Left as a view: the
        # caller's transpose+reshape materializes it anyway.
        out = out[:, :, :seq, :d]
    return out
