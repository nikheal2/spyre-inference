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

from spyre_inference import envs
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


# --------------------------------------------------------------------------------
# Tiled vision attention
#
# SDPA on Spyre lowers to `spyre__sdpa_overrideable` in torch-spyre's inductor
# decompositions. That decomposition *is* an online softmax -- running max,
# rescaling correction, accumulated output -- but it applies the update to a single
# full-width `scores = query @ keys_T`, asking the compiler to tile it via
# `spyre_hint(tiles=...)`. At Pixtral's 3136 patches the hint does not take: the
# measured lowering streams a materialized [1, 16, 3136, 3136] score matrix (314 MB
# in fp16) three times per layer, at 172 ms/layer -- roughly 230 GFLOP/s against the
# ~500 GFLOP/s the decoder's `page_attn_kernel` reaches on the same hardware.
#
# The kernel below is the decoder's structure applied here: the block loop is written
# into the graph rather than requested from the tiler, so a full-width score matrix is
# never expressed and cannot be materialized. Live tile is [B, H, L, block] -- 51 MB at
# block 512 against 314 MB.
#
# Compiling it is not optional. torch-spyre compiles SDPA internally on every dispatch
# even under enforce_eager, so replacing it with plain eager ops would trade one
# compiled region for ~10 eager ops per block, each paying a host round trip on a
# multi-MB tensor (measured at 0.45-0.88 GB/s against a 28 GB/s wire). `dynamic=False`
# also gives the unrolled loop Dynamo needs to specialize the block count.
# --------------------------------------------------------------------------------

# Below this padded length the whole score matrix is small enough that SDPA's
# materialization costs little, and the unrolled loop is not worth its compile time.
_TILE_MIN_SEQ = 1024

# Ceiling on unrolled block iterations. Exceeding it doubles the block width instead,
# trading a larger live tile for a smaller graph.
_TILE_MAX_BLOCKS = 16

# `(seq, seq_pad, block, dtype, device) -> (block_indices, tail_mask)`. The plan is
# identical for every layer of a tower and every call at one image size, and its
# index tensors live on device, so rebuilding it per layer would be pure H2D traffic.
_TILE_PLAN_CACHE: dict[tuple, tuple[list[torch.Tensor], torch.Tensor | None]] = {}

# Plans are keyed by image shape; a long-running server sees few distinct ones. The
# cap only stops an unbounded leak if some caller sweeps sizes.
_TILE_PLAN_CACHE_MAX = 64


def _kv_block_size(seq_pad: int) -> int | None:
    """Key-block width for `_tiled_attention`, or None to stay on SDPA.

    Returns a multiple of `STICK` so that every block -- including the ragged last
    one, whose width is `seq_pad - i * block` -- keeps the `p @ v` reduction axis
    stick-aligned.
    """
    if seq_pad < _TILE_MIN_SEQ:
        return None
    block = align_up(max(envs.SPYRE_VISION_TILED_ATTN_BLOCK, STICK))
    while (seq_pad + block - 1) // block > _TILE_MAX_BLOCKS:
        block *= 2
    # A single block is the materialized path again, with extra machinery.
    return block if block < seq_pad else None


def _tile_plan(
    seq: int,
    seq_pad: int,
    block: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[list[torch.Tensor], torch.Tensor | None]:
    """Device gather indices per key block, plus an additive mask for the last one.

    Only blocks holding at least one real key are visited, so the padded tail beyond
    `ceil(seq / block) * block` is never gathered. The one block that straddles `seq`
    gets a `[1, 1, 1, width]` additive mask -- O(block), not O(L²) -- which makes the
    result exact rather than accepting `_MASKLESS_PAD_BOUND`'s softmax-mass error.
    """
    key = (seq, seq_pad, block, dtype, str(device))
    cached = _TILE_PLAN_CACHE.get(key)
    if cached is not None:
        return cached

    num_blocks = (seq + block - 1) // block
    indices: list[torch.Tensor] = []
    tail: torch.Tensor | None = None
    for i in range(num_blocks):
        start = i * block
        # Never runs past seq_pad: start < seq <= seq_pad, and both are STICK multiples.
        width = min(block, seq_pad - start)
        # index_select, not a slice: a compiled region reads an offset view as offset 0
        # (torch-spyre#3770), so the gather has to be explicit.
        indices.append(convert(torch.arange(start, start + width, dtype=torch.int32), device))
        if start + width > seq and tail is None:
            valid = seq - start  # >= 1: start < seq by construction of num_blocks
            m = torch.zeros(1, 1, 1, width, dtype=dtype)
            m[:, :, :, valid:] = torch.finfo(dtype).min
            tail = convert(m, device)

    if len(_TILE_PLAN_CACHE) >= _TILE_PLAN_CACHE_MAX:
        _TILE_PLAN_CACHE.clear()
    _TILE_PLAN_CACHE[key] = (indices, tail)
    return indices, tail


def _tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    block_indices: list[torch.Tensor],
    tail_mask: torch.Tensor | None,
    scale: float,
) -> torch.Tensor:
    """Online-softmax attention over key blocks, `[B, H, L, D]` in and out.

    Mirrors `v1/attention/ops/page_attn.py::page_attn_kernel`: per block, score
    against that block only, rescale the running accumulators by
    `exp(old_max - new_max)`, and accumulate. `block_indices` is a Python list, so
    Dynamo specializes on its length and unrolls -- the blocking is structural in the
    compiled graph, which is the whole point of not calling SDPA.

    The first block initializes rather than folding into `-inf` accumulators: seeding
    `m` at `-inf` makes `exp(m - m_new)` produce a NaN when the first block is itself
    fully masked.
    """
    num_blocks = len(block_indices)
    last = num_blocks - 1

    m_i: torch.Tensor | None = None
    l_i: torch.Tensor | None = None
    o_i: torch.Tensor | None = None

    for i in range(num_blocks):
        idx = block_indices[i]
        k_blk = k.index_select(2, idx)
        v_blk = v.index_select(2, idx)

        scores = torch.matmul(q, k_blk.transpose(-1, -2)) * scale
        if tail_mask is not None and i == last:
            # Broadcast over B, H and every query row: the padded keys are invalid
            # for all of them.
            scores = scores + tail_mask

        block_max = torch.amax(scores, dim=-1, keepdim=True)
        if i == 0:
            m_i = block_max
            probs = torch.exp(scores - m_i)
            o_i = torch.matmul(probs, v_blk)
            l_i = probs.sum(dim=-1, keepdim=True)
        else:
            assert m_i is not None and l_i is not None and o_i is not None
            m_new = torch.maximum(m_i, block_max)
            rescale = torch.exp(m_i - m_new)
            probs = torch.exp(scores - m_new)
            o_i = o_i * rescale + torch.matmul(probs, v_blk)
            l_i = l_i * rescale + probs.sum(dim=-1, keepdim=True)
            m_i = m_new

    assert o_i is not None and l_i is not None
    return o_i / l_i


# dynamic=False for the unroll, and because the Spyre backend rejects SymInt shapes.
_tiled_attention_compiled = torch.compile(_tiled_attention, dynamic=False)


def _use_tiled_attention(
    mask_tensor: torch.Tensor | None,
    seq_pad: int,
    enable_gqa: bool,
    q: torch.Tensor,
    k: torch.Tensor,
) -> int | None:
    """Block width to tile with, or None to fall through to SDPA.

    Restricted to the case that actually motivated the kernel -- one long, unmasked,
    non-GQA sequence -- so Gemma 4's real masks and CLIP's 50-patch towers keep the
    lowering they are already validated against.
    """
    if not envs.SPYRE_VISION_TILED_ATTN:
        return None
    if mask_tensor is not None:
        # A real mask would have to be tiled alongside the scores. Supportable, but it
        # also admits fully-masked blocks, which the accumulator seeding above does not
        # handle. Left to SDPA until something needs it.
        return None
    if enable_gqa and q.shape[1] != k.shape[1]:
        return None
    return _kv_block_size(seq_pad)


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

    A mask that attends everywhere is dropped rather than materialized. Spyre softmaxes
    a *materialized* score matrix — at Pixtral's 3136 patches that is 314 MB live per
    layer, streamed three times, plus a 39 MB mask read per pass, measured at
    172 ms/layer. Dropping the mask removes the per-pass mask read; it does not by
    itself stop the score matrix being materialized, which is what `_tiled_attention`
    above is for.

    The cost of dropping it is the padded keys, which then have no `-inf` column: the
    output shrinks by at most `n_pad / seq_pad` (see `_MASKLESS_PAD_BOUND`, which is why
    a long, barely-padded sequence takes this path and a short one does not). Head-dim
    padding is exact either way — its `q`/`k` lanes are zero, so they add nothing to the
    dot product, and its `v` lanes are cropped off. On the tiled path the key padding is
    exact too: `_tile_plan` masks the one straddling block instead.

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

    mask_tensor = _padded_attn_mask(mask, b, seq, seq_pad, q.dtype, device)
    block = _use_tiled_attention(mask_tensor, seq_pad, enable_gqa, q, k)

    if block is not None:
        block_indices, tail_mask = _tile_plan(seq, seq_pad, block, q.dtype, device)
        out = _tiled_attention_compiled(q, k, v, block_indices, tail_mask, scale)
    else:
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask_tensor,
            is_causal=False,
            scale=scale,
            enable_gqa=enable_gqa,
        )

    if padded:
        # Offset-0 prefix slice, so torch-spyre#3770 cannot bite. Left as a view: the
        # caller's transpose+reshape materializes it anyway.
        out = out[:, :, :seq, :d]
    return out
