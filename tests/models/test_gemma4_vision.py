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

"""``SpyreGemma4ForConditionalGeneration``'s vision pipeline against the stock tower.

Each rewrite can be wrong while still giving plausible numbers, so the load-bearing tests
compare against stock transformers on CPU; the on-card twins skip without a device.
"""

import inspect
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from spyre_testing_plugin.pytest_plugin import spyre_available

modeling_gemma4 = pytest.importorskip("transformers.models.gemma4.modeling_gemma4")
configuration_gemma4 = pytest.importorskip("transformers.models.gemma4.configuration_gemma4")
gemma4_mm = pytest.importorskip("vllm.model_executor.models.gemma4_mm")

pytestmark = [pytest.mark.gemma4_vision]

# 26B-A4B's real vision head_dim; 72 is neither a 64-multiple nor evenly halvable
# onto the stick, which is the whole reason the padding exists.
ORIG_HEAD_DIM = 72
PADDED_HEAD_DIM = 128
NUM_HEADS = 4
NUM_PATCHES = 50
# Stick-wide, so the position tables gather rows-outermost on the card.
HIDDEN = 256
TEXT_HIDDEN = 64
PATCH = 4  # 3*4*4 = 48 values per patch: exercises the patch-width padding to 64
POOL = 3


def _vision_config(head_dim: int = ORIG_HEAD_DIM, num_heads: int = NUM_HEADS):
    return configuration_gemma4.Gemma4VisionConfig(
        hidden_size=num_heads * head_dim,
        num_attention_heads=num_heads,
        num_key_value_heads=num_heads,
        head_dim=head_dim,
    )


def _tower_config(num_layers: int = 2, standardize: bool = False):
    return configuration_gemma4.Gemma4VisionConfig(
        hidden_size=HIDDEN,
        intermediate_size=200,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=ORIG_HEAD_DIM,
        num_hidden_layers=num_layers,
        patch_size=PATCH,
        pooling_kernel_size=POOL,
        position_embedding_size=32,
        standardize=standardize,
    )


def _position_ids(num_patches: int = NUM_PATCHES) -> torch.Tensor:
    """`[1, num_patches, 2]` (x, y) patch coordinates, as the encoder feeds them."""
    side = 8
    xs = torch.arange(side).repeat(side)[:num_patches]
    ys = torch.arange(side).repeat_interleave(side)[:num_patches]
    return torch.stack([xs, ys], dim=-1).unsqueeze(0)


def _pad_activation_quarters(x: torch.Tensor, orig: int, padded: int) -> torch.Tensor:
    """The `_pad_qk_linear` channel remap, applied to activations instead of weights.

    Kept in the test rather than imported: it mirrors what the padded q_proj weight
    does to its output, so writing it out independently is what makes the rope
    equivalence check below meaningful.
    """
    quarter = orig // 4
    half = padded // 2
    out = torch.zeros((*x.shape[:-1], padded), dtype=x.dtype)
    out[..., :quarter] = x[..., :quarter]
    out[..., quarter : 2 * quarter] = x[..., 2 * quarter : 3 * quarter]
    out[..., half : half + quarter] = x[..., quarter : 2 * quarter]
    out[..., half + quarter : half + 2 * quarter] = x[..., 3 * quarter :]
    return out


def _unpad_activation_quarters(padded: torch.Tensor, orig: int, padded_dim: int) -> torch.Tensor:
    quarter = orig // 4
    half = padded_dim // 2
    out = torch.zeros((*padded.shape[:-1], orig), dtype=padded.dtype)
    out[..., :quarter] = padded[..., :quarter]
    out[..., 2 * quarter : 3 * quarter] = padded[..., quarter : 2 * quarter]
    out[..., quarter : 2 * quarter] = padded[..., half : half + quarter]
    out[..., 3 * quarter :] = padded[..., half + quarter : half + 2 * quarter]
    return out


# ---------------------------------------------------------------------------
# The pipeline against the stock tower
# ---------------------------------------------------------------------------


def _stock_modules(config) -> tuple[nn.Module, nn.Module]:
    # Seeded: two calls give identical weights.
    torch.manual_seed(0)
    tower = modeling_gemma4.Gemma4VisionModel(config).eval()
    # Without a named implementation `create_bidirectional_mask` early-exits to `None`,
    # and the stock forward would silently attend over the padding patches.
    tower.config._attn_implementation = "sdpa"
    embedder = modeling_gemma4.Gemma4MultimodalEmbedder(
        config, SimpleNamespace(hidden_size=TEXT_HIDDEN)
    ).eval()
    for module in (tower, embedder):
        for p in module.parameters():
            p.data.normal_(std=0.05)
    for name, p in tower.named_parameters():
        if name.endswith("norm.weight"):
            p.data.add_(1.0)
    if config.standardize:
        tower.std_bias.normal_()
        tower.std_scale.uniform_(0.5, 1.5)
    return tower, embedder


def _image(hp: int, wp: int, pad: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """As the processor emits it: row-major patches, then ``pad`` patches at ``(-1, -1)``."""
    generator = torch.Generator().manual_seed(seed)
    n = hp * wp
    pixel_values = torch.rand(n + pad, 3 * PATCH * PATCH, generator=generator)
    pixel_values[n:] = 0
    ys, xs = torch.meshgrid(torch.arange(hp), torch.arange(wp), indexing="ij")
    positions = torch.stack([xs.flatten(), ys.flatten()], dim=-1)
    positions = torch.cat([positions, torch.full((pad, 2), -1)])
    return pixel_values, positions


def _images() -> list[tuple[torch.Tensor, torch.Tensor]]:
    # Non-square, 54 + 18 = 72 valid patches: off the stick, so padding slots are live.
    return [_image(6, 9, pad=9, seed=1), _image(3, 6, pad=9, seed=2)]


def _mm_kwargs(images) -> dict:
    return {
        "pixel_values": [pv for pv, _ in images],
        "pixel_position_ids": [pos for _, pos in images],
    }


def _reference(config, images) -> list[torch.Tensor]:
    tower, embedder = _stock_modules(config)
    with torch.inference_mode():
        return [
            embedder(tower(pixel_values=pv[None], pixel_position_ids=pos[None]).last_hidden_state)
            for pv, pos in images
        ]


def _spyre_model(config, dtype=torch.float16):
    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration

    model = SpyreGemma4ForConditionalGeneration.__new__(SpyreGemma4ForConditionalGeneration)
    nn.Module.__init__(model)
    model.config = SimpleNamespace(
        vision_config=config, text_config=SimpleNamespace(hidden_size=TEXT_HIDDEN)
    )
    model.vision_tower, model.embed_vision = _stock_modules(config)
    model._init_spyre_vision()
    model.process_weights_after_loading()
    return model.to(dtype)


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.float().flatten(), b.float().flatten(), dim=0
    ).item()


@pytest.mark.parametrize("standardize", [False, True], ids=["plain", "standardize"])
@pytest.mark.parametrize("num_images", [1, 2])
def test_pipeline_matches_stock(standardize, num_images):
    config = _tower_config(standardize=standardize)
    images = _images()[:num_images]

    with torch.inference_mode():
        expected = _reference(config, images)
        actual = _spyre_model(config).embed_multimodal(**_mm_kwargs(images))

    assert [t.shape for t in actual] == [t.shape for t in expected]
    for got, want in zip(actual, expected, strict=True):
        assert got.device.type == "cpu"
        assert _cosine(got, want) > 0.995


def test_padding_does_not_leak_into_images():
    """Budget padding must not change any image's tokens: padding slots have their own
    segment, and each offset group is padded separately."""
    config = _tower_config(num_layers=1)
    model = _spyre_model(config)
    mm_kwargs = _mm_kwargs(_images())
    values = model.prepare_encoder_cudagraph_replay_buffers(mm_kwargs, 0, 0).values
    padding = model.get_encoder_cudagraph_config().padding_logics

    def run(num_patches):
        bufs = model._vision_empty(num_patches)
        for key, src in values.items():
            padding[key](bufs[key], src)
        return model.encoder_cudagraph_forward(bufs)

    with torch.inference_mode():
        tight, loose = run(576), run(1152)
    n = sum(model.vision_item_tokens(mm_kwargs))
    torch.testing.assert_close(tight[:n], loose[:n], atol=1e-3, rtol=1e-3)


def test_offset_major_sum_matches_the_stock_pooler():
    """Summing the offset groups of a k-divisible grid, scaled, is stock's position-driven
    average with its ``sqrt(hidden_size)`` scale."""
    from spyre_inference.models._vision_encoder import offset_major

    config = _tower_config()
    _, positions = _image(6, 9, pad=9, seed=0)
    torch.manual_seed(3)
    hidden = torch.randn(1, positions.shape[0], HIDDEN)
    padding = (positions == -1).all(dim=-1)[None]
    want, mask = modeling_gemma4.Gemma4VisionPooler(config)(
        hidden_states=hidden,
        pixel_position_ids=positions[None],
        padding_positions=padding,
        output_length=positions.shape[0] // POOL**2,
    )

    valid = positions[~padding[0]]
    grid = torch.empty(6, 9, HIDDEN)
    grid[valid[:, 1], valid[:, 0]] = hidden[0, ~padding[0]]
    got = offset_major(grid, POOL).sum(0) * (HIDDEN**0.5 / POOL**2)
    torch.testing.assert_close(got, want[mask], rtol=1e-5, atol=1e-5)


def test_item_tokens_count_valid_patches_only():
    model = _spyre_model(_tower_config(num_layers=1))
    assert model.vision_item_tokens(_mm_kwargs(_images())) == [6, 2]


def test_a_grid_the_kernel_does_not_divide_is_rejected():
    model = _spyre_model(_tower_config(num_layers=1))
    with pytest.raises(ValueError, match="pooling kernel"):
        model.vision_inputs(_mm_kwargs([_image(4, 6, pad=0, seed=0)]))


def test_video_is_rejected():
    model = _spyre_model(_tower_config(num_layers=1))
    with pytest.raises(NotImplementedError, match="video"):
        model.embed_multimodal(pixel_values_videos=torch.zeros(1, 9, 48))


def test_gqa_tower_is_rejected():
    config = _tower_config(num_layers=1)
    config.num_key_value_heads = 2
    with pytest.raises(NotImplementedError, match="GQA"):
        _spyre_model(config)


def test_tower_classes_stay_stock_and_weight_names_are_kept():
    """Only instances switch: the transformers classes are shared process-wide."""
    from spyre_inference.models import gemma4_mm as spyre_gemma4

    stock_norm = modeling_gemma4.Gemma4RMSNorm.forward
    stock_attn = modeling_gemma4.Gemma4VisionAttention.forward
    config = _tower_config(num_layers=1)
    names = [name for name, _ in _stock_modules(config)[0].named_parameters()]

    model = _spyre_model(config)

    assert modeling_gemma4.Gemma4RMSNorm.forward is stock_norm
    assert modeling_gemma4.Gemma4VisionAttention.forward is stock_attn
    layer = model.vision_tower.encoder.layers[0]
    assert type(layer.self_attn) is spyre_gemma4.SpyreGemma4VisionAttention
    assert type(layer.input_layernorm) is spyre_gemma4.SpyreGemma4RMSNorm
    assert [name for name, _ in model.vision_tower.named_parameters()] == names


def test_upstream_tower_contract_is_unchanged():
    """The Spyre attention and the block call rely on these; diff them on a bump."""
    layer_src = inspect.getsource(modeling_gemma4.Gemma4VisionEncoderLayer.forward)
    assert "position_embeddings=position_embeddings" in layer_src
    assert "attention_mask=attention_mask" in layer_src
    tower_src = inspect.getsource(modeling_gemma4.Gemma4VisionModel.__init__)
    for name in ("patch_embedder", "encoder", "pooler"):
        assert f"self.{name} =" in tower_src, f"Gemma4VisionModel no longer builds {name}"
    init_src = inspect.getsource(gemma4_mm.Gemma4ForConditionalGeneration.__init__)
    for name in ("vision_tower", "embed_vision"):
        assert f"self.{name} =" in init_src, f"upstream Gemma 4 no longer builds {name}"


def test_registry_resolves_to_spyre_class_and_keeps_processor():
    from spyre_inference.models import spyre_models
    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration

    assert (
        spyre_models()["Gemma4ForConditionalGeneration"]
        == "spyre_inference.models.gemma4_mm:SpyreGemma4ForConditionalGeneration"
    )
    upstream = gemma4_mm.Gemma4ForConditionalGeneration
    assert SpyreGemma4ForConditionalGeneration._processor_factory is upstream._processor_factory


def test_construction_reaches_upstream_init_and_protocol():
    """``super().__init__`` must land on upstream's: a ``Protocol`` base ahead of it in the
    MRO would substitute its placeholder ``__init__`` and skip building the model."""
    from vllm.model_executor.models.interfaces import supports_encoder_cudagraph

    from spyre_inference.models.gemma4_mm import SpyreGemma4ForConditionalGeneration

    mro = SpyreGemma4ForConditionalGeneration.__mro__[1:]
    first_init = next(cls for cls in mro if "__init__" in cls.__dict__)
    assert first_init is gemma4_mm.Gemma4ForConditionalGeneration
    assert supports_encoder_cudagraph(SpyreGemma4ForConditionalGeneration)


def test_multimodal_patches_leave_a_gemma4_tower_alone():
    """The registered subclass adapts the tower; the class-name dispatch must not patch it."""
    from spyre_inference import multimodal

    stock = {
        cls: dict(vars(cls))
        for cls in (
            modeling_gemma4.Gemma4RMSNorm,
            modeling_gemma4.Gemma4VisionEncoder,
            modeling_gemma4.Gemma4VisionPooler,
            modeling_gemma4.Gemma4VisionPatchEmbedder,
        )
    }
    model = torch.nn.Module()
    model.vision_tower = modeling_gemma4.Gemma4VisionModel(_tower_config(num_layers=1))
    multimodal.apply_multimodal_patches(model, torch.device("cpu"))
    assert {cls: dict(vars(cls)) for cls in stock} == stock


# ---------------------------------------------------------------------------
# RoPE: per-axis tables over the padded head == the transformers reference
# ---------------------------------------------------------------------------


@pytest.mark.rotary
def test_rope_tables_match_transformers_reference():
    """Gathering both axes' rows and rotating over the padded layout must reproduce stock
    `apply_multidimensional_rope` on the real channels. Catches an off-by-a-quarter
    interleave, a swapped axis, or a sin sign error -- each still plausible in magnitude.
    """
    from spyre_inference.models.gemma4_mm import _rope_swap_matrix, gemma4_rope_tables

    torch.manual_seed(0)
    config = _vision_config()
    position_ids = _position_ids()
    x = torch.randn(1, NUM_PATCHES, NUM_HEADS, ORIG_HEAD_DIM, dtype=torch.float32)

    rope = modeling_gemma4.Gemma4VisionRotaryEmbedding(config)
    cos, sin = rope(x, position_ids)
    want = modeling_gemma4.apply_multidimensional_rope(x, cos, sin, position_ids, unsqueeze_dim=2)

    cos_t, sin_t = gemma4_rope_tables(rope.inv_freq, 8, PADDED_HEAD_DIM)
    px, py = position_ids[0, :, 0], position_ids[0, :, 1]
    cos_rows = (cos_t[0, px] + cos_t[1, py])[None, :, None]
    sin_rows = (sin_t[0, px] + sin_t[1, py])[None, :, None]
    x_padded = _pad_activation_quarters(x, ORIG_HEAD_DIM, PADDED_HEAD_DIM)
    got_padded = x_padded * cos_rows + (x_padded @ _rope_swap_matrix(PADDED_HEAD_DIM)) * sin_rows
    got = _unpad_activation_quarters(got_padded, ORIG_HEAD_DIM, PADDED_HEAD_DIM)

    torch.testing.assert_close(got, want, rtol=1e-4, atol=1e-4)


@pytest.mark.rotary
def test_rope_tables_padding_lanes_are_the_identity_rotation():
    """Padding lanes must rotate by nothing (cos=1, sin=0) once both axes are summed, so a
    padded channel that is not exactly zero still cannot leak into a real one."""
    from spyre_inference.models.gemma4_mm import gemma4_rope_tables

    rope = modeling_gemma4.Gemma4VisionRotaryEmbedding(_vision_config())
    cos_t, sin_t = gemma4_rope_tables(rope.inv_freq, 8, PADDED_HEAD_DIM)
    cos, sin = cos_t[0, 3] + cos_t[1, 5], sin_t[0, 3] + sin_t[1, 5]

    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    for lanes in (slice(2 * quarter, half), slice(half + 2 * quarter, PADDED_HEAD_DIM)):
        assert torch.all(cos[lanes] == 1.0)
        assert torch.all(sin[lanes] == 0.0)


def test_rope_tables_apply_attention_scaling_to_the_real_lanes_only():
    from spyre_inference.models.gemma4_mm import gemma4_rope_tables

    rope = modeling_gemma4.Gemma4VisionRotaryEmbedding(_vision_config())
    cos, sin = gemma4_rope_tables(rope.inv_freq, 8, PADDED_HEAD_DIM)
    cos_scaled, sin_scaled = gemma4_rope_tables(rope.inv_freq, 8, PADDED_HEAD_DIM, 2.0)

    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    real = [slice(0, 2 * quarter), slice(half, half + 2 * quarter)]
    pad = [slice(2 * quarter, half), slice(half + 2 * quarter, PADDED_HEAD_DIM)]
    for lanes in real:
        torch.testing.assert_close(cos_scaled[..., lanes], 2.0 * cos[..., lanes])
        torch.testing.assert_close(sin_scaled[..., lanes], 2.0 * sin[..., lanes])
    for lanes in pad:
        assert torch.all(cos_scaled[0, :, lanes] == 1.0)
        assert torch.all(sin_scaled[..., lanes] == 0.0)


def test_rope_swap_matrix_swaps_halves():
    from spyre_inference.models.gemma4_mm import _rope_swap_matrix

    head_dim = 8
    x = torch.arange(head_dim, dtype=torch.float32)
    want = torch.cat([x[head_dim // 2 :], x[: head_dim // 2]])
    torch.testing.assert_close(x @ _rope_swap_matrix(head_dim), want)


def test_fp32_inv_freq_is_recomputed_not_read_off_a_downcast_buffer():
    """`model.to(...)` downcasts `inv_freq` to the model dtype, and these frequencies
    span a range where 16 bits cost real precision. Recomputing must recover full fp32."""
    from spyre_inference.models.gemma4_mm import _fp32_inv_freq

    config = _vision_config()
    rope = modeling_gemma4.Gemma4VisionRotaryEmbedding(config)
    exact = rope.inv_freq.float().clone()

    # Simulate the fp16 cast the real model applies to the whole tower.
    rope.inv_freq = rope.inv_freq.to(torch.float16)
    assert not torch.allclose(rope.inv_freq.float(), exact), "premise: fp16 loses bits"

    recovered, attention_scaling = _fp32_inv_freq(rope, config)
    assert recovered.dtype == torch.float32
    assert attention_scaling == 1.0
    torch.testing.assert_close(recovered, exact, rtol=1e-6, atol=1e-9)


def test_fp32_inv_freq_resolves_the_rope_type_off_the_module_not_the_registry():
    """transformers renames this tower's rope type from `default` to `axial` after the
    pinned 5.16.1, and its axial initializer is a class-local staticmethod that
    `ROPE_INIT_FUNCTIONS` does not carry -- resolving through the registry raises KeyError.
    """
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    from spyre_inference.models.gemma4_mm import _fp32_inv_freq

    assert "axial" not in ROPE_INIT_FUNCTIONS, "premise: the registry has no axial entry"

    config = _vision_config()
    config.rope_parameters = dict(config.rope_parameters, rope_type="axial")

    class AxialRotary(torch.nn.Module):
        @staticmethod
        def compute_axial_rope_parameters(cfg, device=None, **kwargs):
            spatial_dim = cfg.head_dim // 2
            base = cfg.rope_parameters["rope_theta"]
            inv_freq = 1.0 / (
                base ** (torch.arange(0, spatial_dim, 2, dtype=torch.float) / spatial_dim)
            )
            return inv_freq, 2.0

    inv_freq, attention_scaling = _fp32_inv_freq(AxialRotary(), config)
    assert inv_freq.shape == (ORIG_HEAD_DIM // 4,)
    assert attention_scaling == 2.0


# ---------------------------------------------------------------------------
# Head-dim and MLP padding
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("orig", "expected"),
    [(64, 128), (72, 128), (128, 128), (129, 256), (256, 256)],
)
def test_padded_head_dim_rounds_to_a_double_stick(orig, expected):
    """Rope needs each half on a whole stick, so head_dim pads to 2*64, not 64."""
    from spyre_inference.models.gemma4_mm import _padded_head_dim

    assert _padded_head_dim(orig) == expected


def test_pad_qk_linear_preserves_every_original_channel():
    """The quarter-interleave must be a permutation of the real channels into the
    padded layout -- no value dropped, no value duplicated."""
    from spyre_inference.models.gemma4_mm import _pad_qk_linear

    torch.manual_seed(0)
    in_features = 32
    linear = torch.nn.Linear(in_features, NUM_HEADS * ORIG_HEAD_DIM, bias=False)
    linear.weight.data.normal_()
    proj = type("_Proj", (), {"linear": linear})()

    padded = _pad_qk_linear(proj, NUM_HEADS, ORIG_HEAD_DIM, PADDED_HEAD_DIM)

    assert padded.weight.shape == (NUM_HEADS * PADDED_HEAD_DIM, in_features)
    orig_w = linear.weight.detach().view(NUM_HEADS, ORIG_HEAD_DIM, in_features)
    new_w = padded.weight.detach().view(NUM_HEADS, PADDED_HEAD_DIM, in_features)
    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    # X-axis quarters land in the first slot of each half, Y-axis in the second.
    torch.testing.assert_close(new_w[:, :quarter], orig_w[:, :quarter])
    torch.testing.assert_close(
        new_w[:, quarter : 2 * quarter], orig_w[:, 2 * quarter : 3 * quarter]
    )
    torch.testing.assert_close(new_w[:, half : half + quarter], orig_w[:, quarter : 2 * quarter])
    torch.testing.assert_close(
        new_w[:, half + quarter : half + 2 * quarter], orig_w[:, 3 * quarter :]
    )
    # Everything else is the zero padding.
    assert torch.all(new_w[:, 2 * quarter : half] == 0)
    assert torch.all(new_w[:, half + 2 * quarter :] == 0)


def test_pad_norm_weight_fills_padding_lanes_with_ones():
    """A norm weight is multiplicative, so padding lanes must be 1.0 (not 0.0) --
    a zero there would be harmless today but wrong if the lane ever carried data."""
    from spyre_inference.models.gemma4_mm import _pad_norm_weight

    norm = modeling_gemma4.Gemma4RMSNorm(dim=ORIG_HEAD_DIM, eps=1e-6, with_scale=True)
    norm.weight.data.normal_()

    padded = _pad_norm_weight(norm, ORIG_HEAD_DIM, PADDED_HEAD_DIM)

    assert padded.shape == (PADDED_HEAD_DIM,)
    quarter = ORIG_HEAD_DIM // 4
    half = PADDED_HEAD_DIM // 2
    assert torch.all(padded[2 * quarter : half] == 1.0)
    assert torch.all(padded[half + 2 * quarter :] == 1.0)


def test_pad_mlp_is_numerically_transparent():
    """26B-A4B's vision MLP is 4304 wide (not a 64-multiple), so it is zero-extended
    onto the stick. The zero lanes must contribute nothing: `gelu(0) * 0 == 0`, and
    the matching zero down-projection columns ignore whatever lands there."""
    from spyre_inference.models.gemma4_mm import _pad_mlp

    torch.manual_seed(0)
    orig, padded = 200, 256
    config = configuration_gemma4.Gemma4VisionConfig(
        hidden_size=128,
        intermediate_size=orig,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        num_hidden_layers=1,
        use_clipped_linears=False,
    )
    layer = modeling_gemma4.Gemma4VisionEncoderLayer(config=config, layer_idx=0).to(torch.float32)
    x = torch.randn(1, 5, 128)

    want = layer.mlp(x)
    _pad_mlp(layer, orig, padded)
    got = layer.mlp(x)

    assert layer.mlp.gate_proj.linear.out_features == padded
    assert layer.mlp.up_proj.linear.out_features == padded
    assert layer.mlp.down_proj.linear.in_features == padded
    torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)

    # Idempotent: a repeated weight-loading hook must not double the width.
    _pad_mlp(layer, orig, padded)
    assert layer.mlp.gate_proj.linear.out_features == padded


def test_pad_mlp_is_a_noop_when_already_stick_aligned():
    """E2B's 3072-wide MLP needs no padding; don't rebuild its linears."""
    from spyre_inference.models.gemma4_mm import _pad_mlp

    config = configuration_gemma4.Gemma4VisionConfig(
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=64,
        num_hidden_layers=1,
        use_clipped_linears=False,
    )
    layer = modeling_gemma4.Gemma4VisionEncoderLayer(config=config, layer_idx=0)
    before = layer.mlp.gate_proj.linear

    _pad_mlp(layer, 256, 256)

    assert layer.mlp.gate_proj.linear is before


def test_process_weights_after_loading_pads_every_layer():
    from spyre_inference.multimodal.utils import align_up

    model = _spyre_model(_tower_config(num_layers=3))
    for layer in model.vision_tower.encoder.layers:
        assert layer.self_attn.q_proj.linear.out_features == NUM_HEADS * PADDED_HEAD_DIM
        assert layer.mlp.gate_proj.linear.out_features == align_up(200)


# ---------------------------------------------------------------------------
# RMSNorm: no fp32 promotion (torch-spyre gap), padding-corrected denominator
# ---------------------------------------------------------------------------


def test_padded_rms_norm_does_not_promote_via_an_fp32_weight():
    """These weights are fp32, so an unguarded `x * weight` promotes the activation --
    which torch-spyre cannot lower. Stock hid it behind a trailing `.type_as`."""
    from spyre_inference.models.gemma4_mm import _padded_rms_norm

    x = torch.randn(1, 4, NUM_HEADS, PADDED_HEAD_DIM, dtype=torch.float16)
    fp32_weight = torch.randn(PADDED_HEAD_DIM, dtype=torch.float32)
    out = _padded_rms_norm(x, fp32_weight, 1e-6, ORIG_HEAD_DIM)
    assert out.dtype == torch.float16


def test_padded_rms_norm_ignores_zero_padding_lanes():
    """Normalizing the padded activation must match normalizing the real channels
    alone -- that is what the `padded/orig` variance rescale buys."""
    from spyre_inference.models.gemma4_mm import _padded_rms_norm

    torch.manual_seed(0)
    real = torch.randn(1, 4, NUM_HEADS, ORIG_HEAD_DIM, dtype=torch.float32)
    padded = _pad_activation_quarters(real, ORIG_HEAD_DIM, PADDED_HEAD_DIM)

    got = _padded_rms_norm(padded, None, 1e-6, ORIG_HEAD_DIM)
    got_real_channels = _unpad_activation_quarters(got, ORIG_HEAD_DIM, PADDED_HEAD_DIM)

    variance = (real * real).mean(-1, keepdim=True)
    want = real * torch.rsqrt(variance + 1e-6)

    torch.testing.assert_close(got_real_channels, want, rtol=1e-5, atol=1e-5)


def test_spyre_rms_norm_matches_stock_and_stays_in_input_dtype():
    """The Spyre norm drops stock's fp32 round trip; in fp32 (where the promotion is a
    no-op) it must agree with stock, and an fp16 input must not be promoted by the
    fp32 weight transformers builds."""
    from spyre_inference.models.gemma4_mm import SpyreGemma4RMSNorm

    torch.manual_seed(0)
    norm = modeling_gemma4.Gemma4RMSNorm(dim=16, eps=1e-6, with_scale=True)
    norm.weight.data.normal_()
    x = torch.randn(2, 3, 16, dtype=torch.float32)
    want = norm(x)

    norm.__class__ = SpyreGemma4RMSNorm
    torch.testing.assert_close(norm(x), want, rtol=1e-5, atol=1e-5)
    assert norm(x.half()).dtype == torch.float16


# ---------------------------------------------------------------------------
# Guards on assumptions the padding makes about the projections it rewrites
# ---------------------------------------------------------------------------


def _clipped_proj(output_min: float, output_max: float):
    """A `Gemma4ClippableLinear` with real (finite) output clipping bounds."""
    config = configuration_gemma4.Gemma4VisionConfig(
        hidden_size=NUM_HEADS * ORIG_HEAD_DIM,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=ORIG_HEAD_DIM,
        use_clipped_linears=True,
    )
    proj = modeling_gemma4.Gemma4ClippableLinear(config, 32, NUM_HEADS * ORIG_HEAD_DIM)
    proj.output_min = torch.nn.Buffer(torch.tensor(output_min))
    proj.output_max = torch.nn.Buffer(torch.tensor(output_max))
    return proj


@pytest.mark.parametrize(
    ("lo", "hi"),
    [(-6.0, 6.0), (0.0, 6.0), (-6.0, 0.0), (-float("inf"), float("inf"))],
)
def test_clipping_that_includes_zero_is_accepted(lo, hi):
    """A range containing zero keeps the padding lanes at zero; a zero bound is fine,
    since clamp(0) is still 0."""
    from spyre_inference.models.gemma4_mm import _assert_clipping_preserves_zero

    _assert_clipping_preserves_zero(_clipped_proj(lo, hi), "test proj")


@pytest.mark.parametrize(("lo", "hi"), [(0.5, 6.0), (-6.0, -0.5)])
def test_clipping_that_excludes_zero_is_rejected(lo, hi):
    """A range excluding zero maps the padding lanes to a nonzero bound, invalidating
    `_padded_rms_norm`'s variance correction."""
    from spyre_inference.models.gemma4_mm import _assert_clipping_preserves_zero

    with pytest.raises(NotImplementedError, match="must include zero"):
        _assert_clipping_preserves_zero(_clipped_proj(lo, hi), "test proj")


def test_unclipped_projection_skips_the_clipping_check():
    """Unclipped is the common case, so the check must be inert and must not touch
    bounds buffers that do not exist."""
    from spyre_inference.models.gemma4_mm import _assert_clipping_preserves_zero

    config = _vision_config()
    assert not config.use_clipped_linears, "premise: this config is unclipped"
    proj = modeling_gemma4.Gemma4ClippableLinear(config, 32, NUM_HEADS * ORIG_HEAD_DIM)
    assert not hasattr(proj, "output_min")

    _assert_clipping_preserves_zero(proj, "test proj")  # must not raise


def test_quantized_projection_is_rejected_rather_than_dequantized():
    """Rebuilding as a plain nn.Linear drops the quantization method, so a quantized
    projection must fail rather than be silently dequantized."""
    from spyre_inference.models.gemma4_mm import _as_plain_linear

    class _FakeQuantMethod:
        pass

    class _FakeQuantLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(32, 8))
            self.bias = None
            self.quant_method = _FakeQuantMethod()

    with pytest.raises(NotImplementedError, match="quantization method"):
        _as_plain_linear(_FakeQuantLinear())


def test_unquantized_vllm_linear_is_bounced_to_a_plain_linear():
    """A vLLM linear stores `Wᵀ`, so the bounce must transpose back to `[out, in]`."""
    from vllm.model_executor.layers.linear import UnquantizedLinearMethod

    from spyre_inference.models.gemma4_mm import _as_plain_linear

    class _FakeReplicatedLinear(torch.nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.weight = torch.nn.Parameter(torch.randn(8, 32))  # [in, out]
            self.bias = None
            self.quant_method = UnquantizedLinearMethod()

    src = _FakeReplicatedLinear()
    plain = _as_plain_linear(src)

    assert isinstance(plain, torch.nn.Linear)
    assert plain.weight.shape == (32, 8)  # [out, in]
    torch.testing.assert_close(plain.weight, src.weight.t().contiguous())


# ---------------------------------------------------------------------------
# On card
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not spyre_available(), reason="Spyre device not available")
def test_pipeline_matches_stock_on_spyre():
    """Same comparison with the encoder on the card, uncompiled (the eager path)."""
    config = _tower_config()
    images = _images()

    with torch.inference_mode():
        expected = _reference(config, images)
        actual = _spyre_model(config).to("spyre").embed_multimodal(**_mm_kwargs(images))

    for got, want in zip(actual, expected, strict=True):
        assert _cosine(got, want) > 0.99


@pytest.mark.skipif(not spyre_available(), reason="Spyre device not available")
def test_compiled_graphs_match_eager_on_the_same_inputs_on_spyre():
    """The same padded device buffers through the uncompiled and the compiled pipeline."""
    from spyre_inference.custom_ops.utils import convert

    config = _tower_config()
    model = _spyre_model(config).to("spyre")
    mm_kwargs = _mm_kwargs(_images())
    values = model.prepare_encoder_cudagraph_replay_buffers(mm_kwargs, 0, 0).values
    padding = model.get_encoder_cudagraph_config().padding_logics
    bufs = model.prepare_encoder_cudagraph_capture_inputs(
        64, 1, 1, torch.device("cpu"), torch.float16
    ).values
    for key, src in values.items():
        padding[key](bufs[key], src)
    inputs = {k: convert(v, device="spyre") for k, v in bufs.items()}
    tokens = sum(model.vision_item_tokens(mm_kwargs))

    with torch.inference_mode():
        eager = model.encoder_cudagraph_forward(inputs)[:tokens]
        model.compile_vision_graphs()
        compiled = model.encoder_cudagraph_forward(inputs)[:tokens]

    assert _cosine(compiled, eager) > 0.9999
    torch.testing.assert_close(compiled.float(), eager.float(), atol=2e-2, rtol=2e-2)


@pytest.mark.skipif(not spyre_available(), reason="Spyre device not available")
def test_padding_on_device_weights_matches_padding_on_host():
    """Padding a layer already on Spyre must give the same weights as padding it on the
    host and moving afterwards: composed on device-resident weights, the helpers'
    strided slice-assignments lower silently wrong, which is what `_host` exists for.
    """
    import copy

    from spyre_inference.models import gemma4_mm as gm

    # Real 26B-A4B vision proportions: head_dim 72 -> 128, intermediate 4304 -> 4352.
    config = configuration_gemma4.Gemma4VisionConfig(
        hidden_size=NUM_HEADS * 72,
        intermediate_size=4304,
        num_attention_heads=NUM_HEADS,
        num_key_value_heads=NUM_HEADS,
        head_dim=72,
        num_hidden_layers=1,
        use_clipped_linears=False,
    )
    orig_hd = config.head_dim
    padded_hd = gm._padded_head_dim(orig_hd)
    padded_inter = 4352

    torch.manual_seed(0)
    layer = modeling_gemma4.Gemma4VisionEncoderLayer(config=config, layer_idx=0)
    layer = layer.to(torch.float16).eval()

    on_host = copy.deepcopy(layer)
    gm._prepare_attention(on_host.self_attn, NUM_HEADS, orig_hd, padded_hd)
    gm._pad_mlp(on_host, config.intermediate_size, padded_inter)
    on_host = on_host.to(torch.device("spyre"))

    on_device = copy.deepcopy(layer).to(torch.device("spyre"))
    gm._prepare_attention(on_device.self_attn, NUM_HEADS, orig_hd, padded_hd)
    gm._pad_mlp(on_device, config.intermediate_size, padded_inter)

    # Every host<->device convert halves an fp16 subnormal (normals round-trip exactly),
    # and padding on device round-trips through `_host`, so its subnormals come back one
    # halving further down -- an error bounded by half the largest subnormal. A wrongly
    # composed slice-assign moves whole weights, so `smallest_normal` still catches it.
    checked = 0
    for name, want in on_host.named_parameters():
        got = dict(on_device.named_parameters())[name]
        torch.testing.assert_close(
            got.detach().to("cpu").float(),
            want.detach().to("cpu").float(),
            rtol=0,
            atol=torch.finfo(torch.float16).smallest_normal,
            msg=lambda m, name=name: f"{name} differs when padded on device:\n{m}",
        )
        checked += 1
    assert checked, "no parameters compared -- the layer walk found nothing"
