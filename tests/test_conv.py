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

"""Tests for `SpyreConv2d` (custom_ops/conv.py), the Pixtral patch-embed conv.

`SpyreConv2d` is registered OOT for **every** `Conv2dLayer` in vLLM, but its
tiled `SpyreTensorLayout`s only make sense for a patch embed (single image,
in-channels inside one 64-wide stick, out-channels tiling into whole sticks).
`_layouts_supported` is the gate that keeps every other conv on vLLM's stock
path, so it is the highest-value thing in this file — and it needs no hardware.

The layout-derivation and numeric tests need torch-spyre / a Spyre card and skip
without them.
"""

import sys

import pytest
import torch
import torch.nn.functional as F

from spyre_testing_plugin.pytest_plugin import spyre_available

# Pixtral patch embed: 1x3xHxW image, 16x16 patches, 1024 out-channels.
PATCH = 16
OUT_CHANNELS = 1024


def _layer(in_ch=3, out_ch=OUT_CHANNELS, kernel=PATCH, stride=PATCH, bias=False):
    """A `Conv2dLayer` with deterministic weights (its weight is `torch.empty`)."""
    from vllm.model_executor.layers.conv import Conv2dLayer

    layer = Conv2dLayer(
        in_ch,
        out_ch,
        kernel,
        stride=stride,
        bias=bias,
        params_dtype=torch.float16,
    )
    torch.manual_seed(0)
    layer.weight.data.normal_(std=0.02)
    if bias:
        layer.bias.data.normal_(std=0.02)
    return layer


# ---------------------------------------------------------------------------
# OOT dispatch
# ---------------------------------------------------------------------------


@pytest.mark.conv
def test_conv2d_oot_dispatch():
    """`Conv2dLayer(...)` instantiates `SpyreConv2d` and selects `forward_oot`."""
    from spyre_inference.custom_ops.conv import SpyreConv2d

    layer = _layer()
    assert isinstance(layer, SpyreConv2d)
    assert layer._forward_method == layer.forward_oot


# ---------------------------------------------------------------------------
# _layouts_supported — the fallback gate
# ---------------------------------------------------------------------------


@pytest.mark.conv
def test_layouts_supported_accepts_a_patch_embed():
    from spyre_inference.custom_ops.conv import _layouts_supported

    x = torch.randn(1, 3, 64, 64, dtype=torch.float16)
    weight = torch.randn(OUT_CHANNELS, 3, PATCH, PATCH, dtype=torch.float16)
    assert _layouts_supported(x, weight) is True


@pytest.mark.conv
@pytest.mark.parametrize(
    "x_shape,w_shape,reason",
    [
        ((2, 3, 64, 64), (OUT_CHANNELS, 3, PATCH, PATCH), "batch > 1"),
        ((1, 65, 64, 64), (OUT_CHANNELS, 65, PATCH, PATCH), "in_channels > 64"),
        ((1, 3, 64, 64), (100, 3, PATCH, PATCH), "out_channels not a multiple of 64"),
        ((1, 3, 64), (OUT_CHANNELS, 3, PATCH, PATCH), "input not 4-D"),
        ((1, 3, 64, 64), (OUT_CHANNELS, 3, PATCH), "weight not 4-D"),
    ],
)
def test_layouts_supported_rejects_non_patch_shapes(x_shape, w_shape, reason):
    """Anything outside the tiled-layout assumptions must fall back to vLLM's
    stock path instead of building a layout that would silently mis-tile."""
    from spyre_inference.custom_ops.conv import _layouts_supported

    x = torch.randn(*x_shape, dtype=torch.float16)
    weight = torch.randn(*w_shape, dtype=torch.float16)
    assert _layouts_supported(x, weight) is False, reason


@pytest.mark.conv
def test_unsupported_shape_falls_back_to_forward_native():
    """A non-patch-shaped conv routes through `forward_native` and matches it
    exactly — `SpyreConv2d` must be transparent for every other `Conv2dLayer`."""
    from spyre_inference.custom_ops.conv import SpyreConv2d

    # groups=1, kernel != stride -> enable_linear False, and out_channels 100
    # is not a multiple of 64, so the tiled layouts do not apply.
    layer = _layer(in_ch=3, out_ch=100, kernel=3, stride=1)
    assert isinstance(layer, SpyreConv2d)

    x = torch.randn(1, 3, 32, 32, dtype=torch.float16)
    torch.testing.assert_close(layer.forward_oot(x), layer.forward_native(x))


# ---------------------------------------------------------------------------
# Layout derivation (needs torch-spyre, not necessarily a card)
# ---------------------------------------------------------------------------


@pytest.mark.conv
def test_weight_layout_rejects_unaligned_out_channels():
    """The layout tuples are derived from shape, and the 64-stick assumption is
    asserted rather than silently mis-tiled."""
    pytest.importorskip("torch_spyre")
    from spyre_inference.custom_ops.conv import _weight_layout

    with pytest.raises(AssertionError, match="multiple of the 64-wide stick"):
        _weight_layout(torch.randn(100, 3, PATCH, PATCH, dtype=torch.float16))


@pytest.mark.conv
@pytest.mark.parametrize(
    "shape,match",
    [
        ((2, 3, 64, 64), "batch"),
        ((1, 65, 64, 64), "in_channels"),
    ],
)
def test_input_layout_rejects_unsupported_shapes(shape, match):
    pytest.importorskip("torch_spyre")
    from spyre_inference.custom_ops.conv import _input_layout

    with pytest.raises(AssertionError, match=match):
        _input_layout(torch.randn(*shape, dtype=torch.float16))


@pytest.mark.conv
@pytest.mark.parametrize("out_ch", [64, 128, OUT_CHANNELS])
@pytest.mark.parametrize("hw", [(64, 64), (48, 80)])
def test_layouts_build_for_valid_shapes(out_ch, hw):
    """Layouts are derived from tensor shape, not hardcoded: any 64-aligned
    out-channel count and any image size must build without raising."""
    pytest.importorskip("torch_spyre")
    from spyre_inference.custom_ops.conv import _input_layout, _weight_layout

    assert _weight_layout(torch.randn(out_ch, 3, PATCH, PATCH, dtype=torch.float16)) is not None
    assert _input_layout(torch.randn(1, 3, *hw, dtype=torch.float16)) is not None


# ---------------------------------------------------------------------------
# Numeric correctness on-card
# ---------------------------------------------------------------------------


@pytest.mark.conv
@pytest.mark.parametrize(
    "height,width",
    [
        (64, 64),  # stick-aligned patch grid (4x4 patches)
        (272, 272),  # 17x17 patches — coprime with the 64 stick; the case the
        # stock im2col lowering cannot restickify
    ],
)
@pytest.mark.parametrize("use_bias", [False, True])
def test_patch_conv_matches_cpu_reference(height, width, use_bias):
    """On-card `F.conv2d` with tiled layouts matches a plain CPU `F.conv2d`."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    layer = _layer(bias=use_bias)

    torch.manual_seed(3)
    x = torch.randn(1, 3, height, width, dtype=torch.float16)
    expected = F.conv2d(
        x,
        layer.weight.data,
        layer.bias.data if use_bias else None,
        stride=PATCH,
    )

    layer = layer.to("spyre")
    actual = layer.forward_oot(x.to("spyre"))

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
