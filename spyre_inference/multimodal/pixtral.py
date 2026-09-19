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

"""Pixtral/Ministral vision-tower workarounds for Spyre.

The tower is plain `nn.Module` code outside vLLM's layer registries, so nothing here
can go through `CustomOp.register_oot`; every fix is a guarded, idempotent
monkeypatch and `apply()` is the only entry point.
"""

from __future__ import annotations

from functools import cache
from typing import NamedTuple

import torch
import torch.nn.functional as F
from vllm.config import CompilationMode, get_cached_compilation_config
from vllm.logger import init_logger

from spyre_inference import envs
from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)

# Matmul reduction dims must land on the Spyre stick: 64 fp16 elements.
SEQ_ALIGNMENT = 64


def _align_up(n: int, align: int = SEQ_ALIGNMENT) -> int:
    return (n + align - 1) // align * align


@cache
def rope_perm_matrix(kind: str, head_dim: int, device: torch.device) -> torch.Tensor:
    """Constant `[head_dim, head_dim]` permutation `M` so `x @ M` is a rope shuffle.

    Rotating by a full-width matmul avoids slicing the head into `d/2`-wide halves:
    at head_dim=64 that half is 32, which torch-spyre cannot lay out ("Unexpected
    stick expression ... Mod(var, 32)"). kind="pair" swaps each `(2k, 2k+1)` pair.
    """
    if kind != "pair":
        raise ValueError(f"unknown rope permutation kind {kind!r}")
    m = torch.zeros(head_dim, head_dim, dtype=torch.float16)
    even = torch.arange(0, head_dim, 2)
    m[even, even + 1] = 1.0
    m[even + 1, even] = 1.0
    return convert(m, device=device, dtype=torch.float16)


def rope_rotate_matmul(x, cos, sin, m: torch.Tensor):
    """`x*cos + (x @ m)*sin` — the rope rotation as a stick-aligned matmul."""
    return x * cos + torch.matmul(x, m) * sin


# Attribute under which a source mask carries its padded counterpart `(key, padded)`.
_MASK_ATTR = "_spyre_padded_mask"


def _padded_attn_mask(
    mask: torch.Tensor,
    b: int,
    seq: int,
    seq_pad: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Additive `[b, 1, seq_pad, seq_pad]` mask on `device`.

    The tensor is O(L²) and the tower hands the same mask to every layer, so it is
    cached on the mask itself: one upload per image, released with its source.
    """
    key = (b, seq, seq_pad, dtype, str(device))
    cached = getattr(mask, _MASK_ATTR, None)
    if cached is not None and cached[0] == key:
        return cached[1]

    # Assembled on CPU: strided slice-assign is not stick-safe on Spyre.
    neg_inf = torch.finfo(dtype).min
    m = torch.zeros(b, 1, seq_pad, seq_pad, dtype=dtype)
    m[:, :, :, seq:] = neg_inf  # padded keys never attended
    mc = convert(mask, "cpu")
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
) -> torch.Tensor:
    """SDPA over `[B, H, L, D]` with L and D padded to the 64 stick, then cropped.

    Padded keys are masked to `-inf` and padded queries cropped off. `scale` comes
    from the unpadded head dim, so the padding cannot change it.
    """
    b, _, seq, d = q.shape
    scale = d**-0.5
    seq_pad = _align_up(seq)
    d_pad = _align_up(d)
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
        scale=scale,
    )

    if padded:
        # Offset-0 prefix slice, so torch-spyre#3770 cannot bite. Left as a view: the
        # caller's transpose+reshape materializes it anyway.
        out = out[:, :, :seq, :d]
    return out


@torch.compiler.disable
def _bucketed_sdpa(q, k, v, bias, scale):
    # Outside the block graph: traced with the rope/reshape producers of q/k/v, the
    # SDPA decomposition's named-dim seeding fails ("reshape split a named dim").
    return F.scaled_dot_product_attention(q, k, v, attn_mask=bias, scale=scale)


def patch_vision_attention() -> None:
    """Replace Pixtral's vision `Attention.forward` with the padded on-card SDPA.

    At a patch count coprime with the 64 stick, stock SDPA either fails to restickify
    a batch-matmul operand or returns silently wrong values, so the padding is a
    correctness requirement. The body is upstream's non-xformers branch with only the
    SDPA call swapped; `patch_vision_rope_vit` must run first because
    `apply_rotary_emb_vit` is resolved by name at call time.
    """
    try:
        from vllm.model_executor.models import pixtral
    except ImportError:
        return

    attn_cls = getattr(pixtral, "Attention", None)
    if attn_cls is None or getattr(attn_cls.forward, "_spyre_patched", False):
        return

    def _forward(self, x, mask, freqs_cis):
        batch, patches, _ = x.shape
        qkv, _ = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q = q.reshape(batch, patches, self.n_heads, self.head_dim)
        k = k.reshape(batch, patches, self.n_heads, self.head_dim)
        v = v.reshape(batch, patches, self.n_heads, self.head_dim)
        q, k = pixtral.apply_rotary_emb_vit(q, k, freqs_cis=freqs_cis)
        # [B, H, L, D] for SDPA.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if isinstance(mask, KeyPaddingBias):
            # The tower already padded L to a stick-aligned bucket (`patch_vision_tower`).
            # Materialized inside the block graph, so the eager SDPA gets fresh
            # offset-0 buffers instead of paying three device copies of its own.
            out = _bucketed_sdpa(
                q.contiguous(), k.contiguous(), v.contiguous(), mask.bias, self.head_dim**-0.5
            )
        else:
            out = padded_sdpa(q, k, v, mask)
        out = out.transpose(1, 2).reshape(batch, patches, self.n_heads * self.head_dim)
        out, _ = self.o_proj(out)
        return out

    _forward._spyre_patched = True
    attn_cls.forward = _forward  # ty: ignore[invalid-assignment]
    logger.info(
        "Spyre: patched Pixtral vision Attention to stick-aligned padded "
        "on-card SDPA (pad L/D to 64, mask, crop)."
    )


class RopeCosSin(NamedTuple):
    """Broadcastable `[1, patches, 1, head_dim]` rope factors, one upload each."""

    cos: torch.Tensor
    sin: torch.Tensor


class TowerRope(NamedTuple):
    """`RopeCosSin` plus the pair-swap matrix, as the patched tower hands it to blocks."""

    cos: torch.Tensor
    sin: torch.Tensor
    perm: torch.Tensor


class KeyPaddingBias(NamedTuple):
    """Additive `[1, 1, 1, L_b]` bias masking the tower's padded keys."""

    bias: torch.Tensor


def patch_vision_rope_vit() -> None:
    """Run the Pixtral `VisionTransformer` 2D-RoPE on-card.

    Upstream's rope is complex and gathers per-token freqs by advanced indexing;
    Spyre has neither `complex64` nor `aten::index.Tensor_out`. So `freqs_cis`
    becomes a real packed cos/sin table and `apply_rotary_emb_vit` becomes
    `x·cos + (x @ P)·sin` over the full stick width.

    The table stays on the host and the per-image gather runs there, so a forward
    uploads two contiguous `[1, patches, 1, head_dim]` tensors and nothing else. The
    on-card alternative (upload the flat index, `index_select`, then slice cos and sin
    out of the `[patches, 2, head_dim]` result) hands every layer two strided views,
    one of them at a nonzero storage offset — which torch-spyre's eager dispatch
    materializes with a fresh device copy on each of the 48 uses (2 per layer).
    """
    try:
        from vllm.model_executor.models import pixtral
    except ImportError:
        return

    orig = getattr(pixtral, "apply_rotary_emb_vit", None)
    vt = getattr(pixtral, "VisionTransformer", None)
    if orig is None or vt is None or getattr(orig, "_spyre_patched", False):
        return

    class _CpuFreqsTable:
        """Host-side freqs table; `__getitem__` returns device-ready cos/sin."""

        def __init__(self, table: torch.Tensor, width: int, device: torch.device):
            self._table = table  # (H*W, 2, head_dim) on CPU
            self._width = width
            self._device = device

        def __getitem__(self, idx) -> RopeCosSin:
            # Upstream indexes with `positions[:, 0], positions[:, 1]`.
            row, col = idx
            return self.factors(row, col)

        def factors(self, row, col, padded_len: int | None = None) -> RopeCosSin:
            # Fold (row, col) into one flat index and gather on the host, where
            # advanced indexing exists.
            flat = (row.to("cpu") * self._width + col.to("cpu")).to(torch.int64)
            gathered = self._table[flat]  # (seq, 2, head_dim)
            if padded_len is not None and padded_len > gathered.shape[0]:
                # Identity rotation (cos=1, sin=0) on the tower's pad rows.
                pad = torch.zeros(padded_len - gathered.shape[0], *gathered.shape[1:])
                pad[:, 0, :] = 1.0
                gathered = torch.cat([gathered, pad.to(gathered.dtype)])
            # Contiguous before the upload: every layer broadcasts these over heads,
            # and a strided or offset operand is copied again on the device.
            cos = gathered[:, 0, :][None, :, None, :].contiguous()
            sin = gathered[:, 1, :][None, :, None, :].contiguous()
            return RopeCosSin(
                convert(cos, device=self._device),
                convert(sin, device=self._device),
            )

    def _freqs_cis_cpu(self):
        # Packed real table (H*W, 2, head_dim): [..., 0, :]=cos, [..., 1, :]=sin.
        if self._freqs_cis is None:
            fc = pixtral.precompute_freqs_cis_2d(
                dim=self.args.hidden_size // self.args.num_attention_heads,
                height=self.max_patches_per_side,
                width=self.max_patches_per_side,
                theta=self.args.rope_theta,
            )  # (H, W, head_dim//2) complex64 on CPU
            cos = fc.real
            sin = fc.imag
            cos_full = cos.repeat_interleave(2, dim=-1)
            sin_signed = torch.stack([-sin, sin], dim=-1).reshape(*sin.shape[:-1], -1)
            packed = torch.stack([cos_full, sin_signed], dim=-2)  # (H, W, 2, head_dim)
            self._freqs_cis = packed.reshape(-1, packed.shape[-2], packed.shape[-1]).to(
                torch.float16
            )  # (H*W, 2, head_dim) on CPU
        return _CpuFreqsTable(self._freqs_cis, self.max_patches_per_side, self.device)

    def _apply_rotary_emb_vit(xq, xk, freqs_cis):
        # xq, xk: [batch, patches, n_heads, head_dim].
        if not isinstance(freqs_cis, (RopeCosSin, TowerRope)):
            raise TypeError(
                "Spyre vision rope expects the RopeCosSin pair produced by the patched "
                f"VisionTransformer.freqs_cis, got {type(freqs_cis).__name__}"
            )
        if isinstance(freqs_cis, TowerRope):
            # Hoisted out of the compiled block: the cached constant is not traceable.
            cos, sin, p = freqs_cis
        else:
            p = rope_perm_matrix("pair", xq.shape[-1], xq.device)
            cos, sin = freqs_cis

        return (
            rope_rotate_matmul(xq, cos, sin, p).type_as(xq),
            rope_rotate_matmul(xk, cos, sin, p).type_as(xk),
        )

    _apply_rotary_emb_vit._spyre_patched = True
    pixtral.apply_rotary_emb_vit = _apply_rotary_emb_vit  # ty: ignore[invalid-assignment]
    vt.freqs_cis = property(_freqs_cis_cpu)  # ty: ignore[invalid-assignment]
    logger.info(
        "Spyre: patched Pixtral VisionTransformer 2D-RoPE to real rotation with a "
        "host-side freqs gather (two cos/sin uploads per image, pair-swap matmul)."
    )


def patch_block_attention_mask() -> None:
    """Build Pixtral's block-diagonal vision mask on CPU.

    Upstream zeroes one `[start:end, start:end]` sub-block per image on
    `patch_embeds.device`; with N images those are strided sub-block writes, which are
    not stick-safe. `_padded_attn_mask` pulls the mask to CPU anyway.
    """
    try:
        from transformers.models.pixtral import modeling_pixtral
    except ImportError:
        return

    orig = getattr(modeling_pixtral, "generate_block_attention_mask", None)
    if orig is None or getattr(orig, "_spyre_patched", False):
        return

    def _cpu_mask(patch_embeds_list, tensor):
        if tensor.device.type != "spyre":
            return orig(patch_embeds_list, tensor)
        # Only `dtype` and the two leading dims are read off `tensor`, so a CPU stand-in
        # gives an identical mask without a D2H of patch_embeds.
        stand_in = torch.empty((tensor.shape[0], tensor.shape[1]), dtype=tensor.dtype)
        return orig(patch_embeds_list, stand_in)

    _cpu_mask._spyre_patched = True
    # vLLM imports this symbol inside the function body, so patching the module
    # attribute is picked up at call time.
    modeling_pixtral.generate_block_attention_mask = _cpu_mask  # ty: ignore[invalid-assignment]
    logger.info("Spyre: Pixtral block attention mask built on CPU (N-image sub-block writes).")


def patch_patch_merger() -> None:
    """Run Pixtral `PatchMerger.permute` (spatial s×s regroup) on CPU.

    It uses `F.unfold` (`aten::im2col`), unsupported on Spyre, and a reshape/permute
    rewrite does not lower either — the regroup is a geometry-dependent multi-counter
    stick scatter. The `merging_layer` GEMM stays on-card.
    """
    try:
        from vllm.model_executor.models import pixtral
    except ImportError:
        return

    pm_cls = getattr(pixtral, "PatchMerger", None)
    if pm_cls is None or getattr(pm_cls.forward, "_spyre_patched", False):
        return

    def _forward(self, x, image_sizes):
        dev = x.device
        x_perm = self.permute(x.to("cpu"), image_sizes)  # unfold on CPU
        return self.merging_layer(convert(x_perm, device=dev))  # GEMM on-card

    _forward._spyre_patched = True
    pm_cls.forward = _forward  # ty: ignore[invalid-assignment]
    logger.info(
        "Spyre: patched Pixtral PatchMerger permute to CPU (merging_layer GEMM stays on-card)."
    )


def vision_seq_buckets() -> list[int]:
    """Stick-aligned sequence buckets the tower pads each image to.

    Every bucket is one compiled graph per block shape, so they are coarse:
    multiples of 256 up to 2048, then of 1024 (the largest image, 110² patches,
    lands in 12288). `SPYRE_VISION_SEQ_BUCKETS` overrides them.
    """
    override = envs.SPYRE_VISION_SEQ_BUCKETS
    if override:
        buckets = sorted({_align_up(int(b)) for b in override.split(",") if b.strip()})
    else:
        buckets = list(range(256, 2048 + 1, 256)) + list(range(3072, 12288 + 1, 1024))
    return buckets


def vision_seq_bucket(seq: int) -> int:
    for bucket in vision_seq_buckets():
        if bucket >= seq:
            return bucket
    return _align_up(seq)


@cache
def _key_padding_bias(seq: int, seq_pad: int, device: torch.device) -> torch.Tensor:
    bias = torch.zeros(1, 1, 1, seq_pad, dtype=torch.float16)
    bias[..., seq:] = torch.finfo(torch.float16).min
    return convert(bias, device=device)


def _patch_weight(tower) -> torch.Tensor:
    """Patch-conv weight as a `[C·k·k padded to 64, D]` GEMM operand, built once."""
    w = getattr(tower, "_spyre_patch_w", None)
    if w is None or w.device != tower.patch_conv.weight.device:
        conv_w = tower.patch_conv.weight.detach().to("cpu")
        w = conv_w.flatten(1)
        w = F.pad(w, (0, _align_up(w.shape[1]) - w.shape[1])).t().contiguous()
        w = convert(w, device=tower.patch_conv.weight.device, dtype=tower.dtype)
        tower._spyre_patch_w = w
    return w


def _patch_embed(tower, img: torch.Tensor, hp: int, wp: int, seq_pad: int) -> torch.Tensor:
    """The stride == kernel patch conv as host patchify + one on-card GEMM.

    The on-card conv lowers a 14-wide kernel through a CPU `unfold` fallback and a
    per-call weight round trip. Rows come out in the conv's `flatten(2).permute`
    order, already zero-padded to the tower's bucket (a zero row stays zero through
    the GEMM and `ln_pre`), so the image is uploaded once and nothing is padded
    on the card.
    """
    ps = tower.args.patch_size
    w = _patch_weight(tower)
    c = img.shape[0]
    patches = (
        img.to("cpu")[:, : hp * ps, : wp * ps]
        .reshape(c, hp, ps, wp, ps)
        .permute(1, 3, 0, 2, 4)
        .reshape(hp * wp, c * ps * ps)
    )
    x = torch.zeros(1, seq_pad, w.shape[0], dtype=tower.dtype)
    x[0, : hp * wp, : patches.shape[1]] = patches
    return torch.matmul(convert(x, device=w.device), w)


def patch_vision_tower() -> None:
    """Run the Pixtral `VisionTransformer` one image at a time at a bucketed length.

    Upstream concatenates every image into one sequence behind an O(L²) block mask
    and leaves each op to see the raw, per-image patch count. Here each image is
    padded once, after `ln_pre`, to a stick-aligned bucket and cropped once at the
    end: pad rows get identity rope and are masked as keys by a `[1, 1, 1, L_b]` bias,
    and norm/linear/MLP are row-independent, so they cannot reach real tokens. Every
    block then sees one static shape per bucket, which is what lets
    `compile_vision_blocks` turn ~30 eager launches per block into one graph.
    """
    try:
        from vllm.model_executor.models import pixtral
    except ImportError:
        return

    vt_cls = getattr(pixtral, "VisionTransformer", None)
    if vt_cls is None or getattr(vt_cls.forward, "_spyre_patched", False):
        return

    def _forward(self, images):
        outs = []
        for img in images:
            ps = self.args.patch_size
            hp, wp = img.shape[-2] // ps, img.shape[-1] // ps
            seq = hp * wp
            seq_pad = vision_seq_bucket(seq)
            x = self.ln_pre(_patch_embed(self, img, hp, wp, seq_pad))
            rows = torch.arange(hp).repeat_interleave(wp)
            cols = torch.arange(wp).repeat(hp)
            cos, sin = self.freqs_cis.factors(rows, cols, seq_pad)
            rope = TowerRope(cos, sin, rope_perm_matrix("pair", cos.shape[-1], x.device))
            mask = KeyPaddingBias(_key_padding_bias(seq, seq_pad, x.device))
            x = self.transformer(x, mask=mask, freqs_cis=rope)
            # Offset-0 prefix: safe to leave as a view.
            outs.append(x[0, :seq])
        return tuple(outs)

    _forward._spyre_patched = True
    vt_cls.forward = _forward  # ty: ignore[invalid-assignment]
    # `_SpyreModelWrapper.embed_multimodal` leaves the pixels on the host for this.
    vt_cls._spyre_host_images = True  # ty: ignore[unresolved-attribute]
    logger.info(
        "Spyre: patched Pixtral VisionTransformer to run per image at a bucketed, "
        "stick-aligned length (buckets %s).",
        vision_seq_buckets(),
    )


def compile_vision_blocks(model: torch.nn.Module) -> int:
    """Compile each vision `TransformerBlock` in place as one static graph."""
    from vllm.model_executor.models import pixtral

    tower = getattr(model, "vision_encoder", None)
    if not isinstance(tower, pixtral.VisionTransformer):
        return 0
    num = 0
    for block in tower.transformer.layers:
        # In place, like `_compile_blocks`: rebinding would rename the parameters.
        # fullgraph=False: SDPA is deliberately left between the two graphs.
        block.compile(backend="inductor", fullgraph=False, dynamic=False)
        num += 1
    return num


def apply(model: torch.nn.Module, device: torch.device) -> None:
    """Install every Pixtral vision-tower workaround, in dependency order.

    The patch-embedding conv is absent on purpose: `SpyreConv2d` in
    `custom_ops/conv.py` handles it through OOT dispatch. The patches rewrite
    upstream module attributes; `model` is only touched to compile the tower's blocks.
    """
    try:
        from vllm.model_executor.models import pixtral
    except ImportError:
        return

    # True whenever xformers merely imports: upstream only disables it on CUDA B200.
    if getattr(pixtral, "USE_XFORMERS_OPS", False):
        raise RuntimeError(
            "xformers is installed; Pixtral on Spyre needs the non-xformers mask path. "
            "Uninstall xformers in this environment."
        )

    # Must precede the attention patch, which resolves apply_rotary_emb_vit by name.
    patch_vision_rope_vit()
    patch_vision_attention()
    patch_vision_tower()
    patch_block_attention_mask()
    patch_patch_merger()

    compile_mode = get_cached_compilation_config().mode
    if envs.SPYRE_VISION_COMPILE and compile_mode is not CompilationMode.NONE:
        num = compile_vision_blocks(model)
        if num:
            logger.info("Wrapped %d Pixtral vision blocks for per-block compile.", num)
