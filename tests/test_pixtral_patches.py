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

"""Tests for the Pixtral vision-tower monkeypatches in `spyre_model_runner.py`.

Four module-level patches make the Pixtral vision tower runnable on Spyre:

  * `_patch_pixtral_vision_rope`      -> HF-format 2D rope as a matmul
  * `_patch_pixtral_vision_rope_vit`  -> mistral-format 2D rope: real (not
    complex) freqs table + `index_select` gather + pair-swap matmul
  * `_patch_pixtral_vision_attention` -> SDPA with L/D padded to the 64 stick
  * `_patch_pixtral_patch_merger`     -> `F.unfold` regroup back on CPU

Each is written defensively: `try: import pixtral / except ImportError: return`
plus `getattr(..., None)` guards. That is right for non-Pixtral models, but it
means a vLLM upgrade that renames any of the target symbols turns the patch into
a **silent no-op** — the model then fails deep inside torch-spyre, or produces
garbage, with nothing naming the cause.

So this file does two things:

  1. **Staleness tripwires** — assert the target symbols still exist and that
     calling each patch actually installs something. These are what break on a
     vLLM bump, pointing straight at the patch that needs updating.
  2. **Semantic equivalence on CPU** — the rewritten rope/SDPA/regroup must
     compute what upstream computes. A sign or interleave error here produces
     plausible-but-wrong attention that an end-to-end test cannot localise.

All CPU; no Spyre hardware required.
"""

import sys

import pytest
import torch

pixtral = pytest.importorskip("vllm.model_executor.models.pixtral")

HIDDEN_SIZE = 256
NUM_HEADS = 4
HEAD_DIM = HIDDEN_SIZE // NUM_HEADS  # 64 — the case that motivated the rewrite
# 16x16 = 256 grid positions, so a 67-patch (stick-coprime) image still fits.
MAX_PATCHES_PER_SIDE = 16
ROPE_THETA = 10000.0


@pytest.fixture(autouse=True)
def restore_pixtral(monkeypatch):
    """The patches mutate the shared `pixtral` module and its classes, so undo
    them after every test — otherwise one test leaks a patched vision tower into
    the rest of the session."""
    monkeypatch.setattr(
        pixtral, "apply_rotary_pos_emb", pixtral.apply_rotary_pos_emb, raising=False
    )
    monkeypatch.setattr(pixtral, "apply_rotary_emb_vit", pixtral.apply_rotary_emb_vit)
    monkeypatch.setattr(
        pixtral.VisionTransformer, "freqs_cis", pixtral.VisionTransformer.__dict__["freqs_cis"]
    )
    monkeypatch.setattr(pixtral.Attention, "forward", pixtral.Attention.forward)
    monkeypatch.setattr(pixtral.PatchMerger, "forward", pixtral.PatchMerger.forward)
    yield


def _vision_args(spatial_merge_size: int = 1):
    return pixtral.VisionEncoderArgs(
        hidden_size=HIDDEN_SIZE,
        num_channels=3,
        image_size=128,
        patch_size=16,
        intermediate_size=512,
        num_hidden_layers=1,
        num_attention_heads=NUM_HEADS,
        rope_theta=ROPE_THETA,
        image_token_id=10,
        spatial_merge_size=spatial_merge_size,
    )


# ---------------------------------------------------------------------------
# 1. Staleness tripwires
# ---------------------------------------------------------------------------


@pytest.mark.pixtral
@pytest.mark.parametrize(
    "symbol",
    [
        "apply_rotary_pos_emb",  # HF-format rope (re-exported from transformers)
        "apply_rotary_emb_vit",  # mistral-format rope
        "precompute_freqs_cis_2d",
        "VisionTransformer",
        "Attention",
        "PatchMerger",
    ],
)
def test_patch_target_symbols_still_exist(symbol):
    """Every symbol the Spyre patches reach for must still be there. The patches
    themselves `getattr(..., None)` and return silently, so this is the only
    place a rename gets caught."""
    assert getattr(pixtral, symbol, None) is not None, (
        f"vllm.model_executor.models.pixtral.{symbol} is gone — the corresponding "
        "Spyre patch in spyre_model_runner.py is now a silent no-op and must be updated"
    )


@pytest.mark.pixtral
def test_vision_rope_patch_is_applied_and_idempotent():
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    assert not getattr(pixtral.apply_rotary_pos_emb, "_spyre_patched", False)

    TorchSpyreModelRunner._patch_pixtral_vision_rope()
    patched = pixtral.apply_rotary_pos_emb
    assert getattr(patched, "_spyre_patched", False) is True

    TorchSpyreModelRunner._patch_pixtral_vision_rope()
    assert pixtral.apply_rotary_pos_emb is patched, "second call must be a no-op"


@pytest.mark.pixtral
def test_vision_rope_vit_patch_is_applied_and_idempotent():
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    original_prop = pixtral.VisionTransformer.__dict__["freqs_cis"]

    TorchSpyreModelRunner._patch_pixtral_vision_rope_vit()
    patched = pixtral.apply_rotary_emb_vit
    assert getattr(patched, "_spyre_patched", False) is True
    assert pixtral.VisionTransformer.__dict__["freqs_cis"] is not original_prop

    TorchSpyreModelRunner._patch_pixtral_vision_rope_vit()
    assert pixtral.apply_rotary_emb_vit is patched, "second call must be a no-op"


@pytest.mark.pixtral
def test_vision_attention_patch_is_applied_and_idempotent():
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    TorchSpyreModelRunner._patch_pixtral_vision_attention()
    patched = pixtral.Attention.forward
    assert getattr(patched, "_spyre_patched", False) is True

    TorchSpyreModelRunner._patch_pixtral_vision_attention()
    assert pixtral.Attention.forward is patched, "second call must be a no-op"


@pytest.mark.pixtral
def test_patch_merger_patch_is_applied_and_idempotent():
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    TorchSpyreModelRunner._patch_pixtral_patch_merger()
    patched = pixtral.PatchMerger.forward
    assert getattr(patched, "_spyre_patched", False) is True

    TorchSpyreModelRunner._patch_pixtral_patch_merger()
    assert pixtral.PatchMerger.forward is patched, "second call must be a no-op"


# ---------------------------------------------------------------------------
# 2a. mistral-format 2D rope: real rewrite == upstream complex rope
# ---------------------------------------------------------------------------


class _FreqsStub:
    """Stands in for a `VisionTransformer` for the patched `freqs_cis` property,
    which only reads `args`, `max_patches_per_side`, `_freqs_cis` and `device`."""

    def __init__(self):
        self.args = _vision_args()
        self.max_patches_per_side = MAX_PATCHES_PER_SIDE
        self._freqs_cis = None
        self.device = torch.device("cpu")


def _positions(num_patches: int) -> torch.Tensor:
    """`[L, 2]` (row, col) patch positions inside the max-patch grid."""
    torch.manual_seed(7)
    rows = torch.randint(0, MAX_PATCHES_PER_SIDE, (num_patches,), dtype=torch.int64)
    cols = torch.randint(0, MAX_PATCHES_PER_SIDE, (num_patches,), dtype=torch.int64)
    return torch.stack([rows, cols], dim=-1)


@pytest.mark.pixtral
@pytest.mark.parametrize("num_patches", [1, 17, 67])
def test_real_rope_matches_upstream_complex_rope(num_patches):
    """The packed real (cos, sin_signed) table + pair-swap matmul reproduces
    upstream's `view_as_complex` rotation. This is where a sign flip or an
    even/odd interleave error would hide."""
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    original_apply = pixtral.apply_rotary_emb_vit

    torch.manual_seed(11)
    xq = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
    xk = torch.randn(1, num_patches, NUM_HEADS, HEAD_DIM, dtype=torch.float16)
    positions = _positions(num_patches)

    # Upstream reference: complex table, advanced-index gather, complex multiply.
    complex_table = pixtral.precompute_freqs_cis_2d(
        dim=HEAD_DIM,
        height=MAX_PATCHES_PER_SIDE,
        width=MAX_PATCHES_PER_SIDE,
        theta=ROPE_THETA,
    )
    complex_gathered = complex_table[positions[:, 0], positions[:, 1]]
    expected_q, expected_k = original_apply(xq, xk, complex_gathered)

    # Spyre rewrite: real table, flat index_select gather, pair-swap matmul.
    TorchSpyreModelRunner._patch_pixtral_vision_rope_vit()
    table = pixtral.VisionTransformer.__dict__["freqs_cis"].fget(_FreqsStub())
    real_gathered = table[(positions[:, 0], positions[:, 1])]
    assert real_gathered.shape == (num_patches, 2, HEAD_DIM)
    assert not real_gathered.is_complex(), "the table must be real — Spyre has no complex dtype"

    actual_q, actual_k = pixtral.apply_rotary_emb_vit(xq, xk, real_gathered)

    torch.testing.assert_close(actual_q.float(), expected_q.float(), atol=2e-2, rtol=2e-2)
    torch.testing.assert_close(actual_k.float(), expected_k.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.pixtral
def test_flat_index_gather_matches_2d_index():
    """The `_OnCardFreqsTable` wrapper folds `(row, col)` into `row*W + col` and
    uses `index_select` (Spyre has no `aten::index`). That flattening must agree
    with a plain 2-D advanced index."""
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    TorchSpyreModelRunner._patch_pixtral_vision_rope_vit()

    stub = _FreqsStub()
    table = pixtral.VisionTransformer.__dict__["freqs_cis"].fget(stub)
    flat = stub._freqs_cis.reshape(MAX_PATCHES_PER_SIDE, MAX_PATCHES_PER_SIDE, 2, HEAD_DIM)

    positions = _positions(23)
    gathered = table[(positions[:, 0], positions[:, 1])]

    torch.testing.assert_close(gathered, flat[positions[:, 0], positions[:, 1]])


# ---------------------------------------------------------------------------
# 2b. Padded on-card SDPA == stock SDPA
# ---------------------------------------------------------------------------


@pytest.mark.pixtral
@pytest.mark.parametrize(
    "num_patches",
    [
        64,  # stick-aligned
        67,  # coprime with the 64 stick — the case the stock lowering rejects
    ],
)
@pytest.mark.parametrize("mask_kind", ["none", "bool", "additive"])
def test_padded_vision_attention_matches_stock(tp_group, num_patches, mask_kind):
    """The pad-to-64 + `-inf` mask + crop SDPA must equal upstream's
    `Attention.forward`. Padded keys must contribute nothing, and padded queries
    must be cropped off — a leak in either shows up as a numeric mismatch."""
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    args = _vision_args()
    layer = pixtral.Attention(args, disable_tp=True).to(torch.float16)
    torch.manual_seed(13)
    for param in layer.parameters():
        param.data.normal_(std=0.02)

    torch.manual_seed(17)
    x = torch.randn(1, num_patches, HIDDEN_SIZE, dtype=torch.float16)
    freqs_cis = pixtral.precompute_freqs_cis_2d(
        dim=HEAD_DIM,
        height=MAX_PATCHES_PER_SIDE,
        width=MAX_PATCHES_PER_SIDE,
        theta=ROPE_THETA,
    ).reshape(-1, HEAD_DIM // 2)[:num_patches]

    if mask_kind == "none":
        mask = None
    elif mask_kind == "bool":
        mask = torch.ones(num_patches, num_patches, dtype=torch.bool).tril()
    else:
        mask = torch.zeros(num_patches, num_patches, dtype=torch.float16)
        mask[:, num_patches // 2 :] = torch.finfo(torch.float16).min

    expected = layer.forward(x, mask, freqs_cis)

    TorchSpyreModelRunner._patch_pixtral_vision_attention()
    actual = pixtral.Attention.forward(layer, x, mask, freqs_cis)

    assert actual.shape == expected.shape == (1, num_patches, HIDDEN_SIZE)
    torch.testing.assert_close(actual.float(), expected.float(), atol=2e-2, rtol=2e-2)


# ---------------------------------------------------------------------------
# 2c. PatchMerger regroup on CPU == stock regroup
# ---------------------------------------------------------------------------


@pytest.mark.pixtral
@pytest.mark.parametrize("image_size", [(4, 4), (6, 8)])
def test_patch_merger_cpu_regroup_matches_stock(tp_group, image_size):
    """Moving `permute` (which uses the unsupported `aten::im2col`) to CPU must
    not change the result; the `merging_layer` GEMM stays untouched."""
    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    spatial_merge_size = 2
    merger = pixtral.PatchMerger(
        vision_encoder_dim=HIDDEN_SIZE,
        spatial_merge_size=spatial_merge_size,
    ).to(torch.float16)
    torch.manual_seed(19)
    merger.merging_layer.weight.data.normal_(std=0.02)

    h, w = image_size
    torch.manual_seed(23)
    x = torch.randn(h * w, HIDDEN_SIZE, dtype=torch.float16)

    expected = merger.forward(x, [image_size])

    TorchSpyreModelRunner._patch_pixtral_patch_merger()
    actual = pixtral.PatchMerger.forward(merger, x, [image_size])

    torch.testing.assert_close(actual, expected)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
