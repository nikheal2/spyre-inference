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

`SpyreConv2d` repacks a patch conv (space-to-depth) into a 2- or 3-tap conv that
torch-spyre's direct conv2d lowering accepts. The repack must be exact, which the
CPU tests prove against a plain `F.conv2d`; `_repack_plan` keeps every other
`Conv2dLayer` on the stock path. The on-card tests need a card.
"""

import sys
import warnings

import pytest
import torch
import torch.nn.functional as F
from spyre_testing_plugin.pytest_plugin import spyre_available

# Pixtral patch embed: 1x3xHxW image, 16x16 patches, 1024 out-channels.
PATCH = 16
OUT_CHANNELS = 1024


def _layer(in_ch=3, out_ch=OUT_CHANNELS, kernel=PATCH, stride=PATCH, bias=False, **kwargs):
    """A `Conv2dLayer` with deterministic weights (its weight is `torch.empty`)."""
    from vllm.model_executor.layers.conv import Conv2dLayer

    layer = Conv2dLayer(
        in_ch,
        out_ch,
        kernel,
        stride=stride,
        bias=bias,
        params_dtype=torch.float16,
        **kwargs,
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
# _repack_plan — which convs take the direct path
# ---------------------------------------------------------------------------


@pytest.mark.conv
@pytest.mark.parametrize(
    "kernel,plan,channels",
    [
        (14, ((2, 2), (7, 7)), 192),  # Ministral-3: 3*7*7 = 147 -> 192
        (16, ((2, 2), (8, 8)), 192),  # Pixtral-12B: 3*8*8 = 192, already whole sticks
        (9, ((3, 3), (3, 3)), 64),  # no factor of 2: 3-tap kernel, 27 -> 64
        (2, ((2, 2), (1, 1)), 64),  # already small: no space-to-depth, 3 -> 64
        ((14, 9), ((2, 3), (7, 3)), 64),  # chosen per axis: 3*7*3 = 63 -> 64
    ],
)
def test_repack_plan_for_patch_convs(kernel, plan, channels):
    from spyre_inference.custom_ops.conv import _packed_channels, _repack_plan

    layer = _layer(kernel=kernel, stride=kernel)
    assert _repack_plan(layer) == plan
    assert _packed_channels(3, plan) == channels


@pytest.mark.conv
@pytest.mark.parametrize(
    "kwargs,reason",
    [
        (dict(kernel=3, stride=1), "kernel != stride"),
        (dict(kernel=4, stride=4, padding=1), "padding"),
        (dict(kernel=4, stride=4, dilation=2), "dilation"),
        (dict(in_ch=4, out_ch=64, kernel=4, stride=4, groups=2), "groups"),
        (dict(kernel=1, stride=1), "1x1 kernel"),
        (dict(kernel=7, stride=7), "kernel with no 2 or 3 factor"),
    ],
)
def test_repack_plan_rejects_other_convs(kwargs, reason):
    """Anything that is not a repackable patch conv keeps the stock path."""
    from spyre_inference.custom_ops.conv import _repack_plan

    assert _repack_plan(_layer(**kwargs)) is None, reason


# ---------------------------------------------------------------------------
# The repack is exact (CPU, no card)
# ---------------------------------------------------------------------------


@pytest.mark.conv
@pytest.mark.parametrize(
    "patch,height,width",
    [
        (16, 64, 64),
        (14, 336, 308),  # Ministral-3 patch size, NON-SQUARE so an H/W swap shows
        (14, 345, 311),  # not a multiple of the patch: the border must be cropped
        (9, 45, 36),  # 3-tap repack
        (2, 8, 6),  # no space-to-depth, channels padded only
    ],
)
@pytest.mark.parametrize("use_bias", [False, True])
def test_repacked_conv_matches_patch_conv(patch, height, width, use_bias):
    """Space-to-depth input + repacked weight + small strided conv == the original
    conv, and the returned NCHW view flattens the way Pixtral consumes it."""
    from spyre_inference.custom_ops.conv import (
        _pack_input,
        _pack_weight,
        _packed_channels,
        _repack_plan,
    )

    layer = _layer(kernel=patch, stride=patch, bias=use_bias)
    plan = _repack_plan(layer)
    channels = _packed_channels(3, plan)
    weight = layer.weight.data.float()
    bias = layer.bias.data.float() if use_bias else None

    torch.manual_seed(3)
    x = torch.randn(1, 3, height, width)
    expected = F.conv2d(x, weight, bias, stride=patch)

    # Same ops as `_conv_direct`, on the CPU.
    x_packed = _pack_input(x, plan, channels)
    w_packed = _pack_weight(weight, plan, channels)
    assert x_packed.shape[-1] == w_packed.shape[0] == channels
    assert x_packed.is_contiguous() and w_packed.is_contiguous()
    out = F.conv2d(x_packed.permute(0, 3, 1, 2), w_packed.permute(3, 0, 1, 2), bias, stride=plan[0])
    actual = out.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)

    assert actual.shape == expected.shape
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert actual.flatten(2).permute(0, 2, 1).is_contiguous()


# ---------------------------------------------------------------------------
# Fallbacks (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.conv
def test_unsupported_shape_falls_back_to_forward_native():
    """A non-patch conv routes through `forward_native` and matches it exactly —
    `SpyreConv2d` must be transparent for every other `Conv2dLayer`."""
    from spyre_inference.custom_ops.conv import SpyreConv2d

    layer = _layer(in_ch=3, out_ch=100, kernel=3, stride=1)
    assert isinstance(layer, SpyreConv2d)
    assert layer._plan is None

    x = torch.randn(1, 3, 32, 32, dtype=torch.float16)
    torch.testing.assert_close(layer.forward_oot(x), layer.forward_native(x))


@pytest.mark.conv
def test_cpu_patch_input_uses_conv_not_mulmat():
    """`forward_native` would route a patch conv into `_forward_mulmat`. Both paths
    are numerically equal, so this pins the route by making the wrong one raise."""
    layer = _layer(kernel=16, stride=16)
    assert layer.enable_linear is True

    def _boom(_x):
        raise AssertionError("CPU input must use F.conv2d, not the im2col/GEMM path")

    layer._forward_mulmat = _boom
    x = torch.randn(1, 3, 32, 32, dtype=torch.float16)
    torch.testing.assert_close(layer.forward_oot(x), layer._forward_conv(x))


@pytest.mark.conv
def test_weights_not_prepared_off_card():
    """`process_weights_after_loading` only repacks a weight that lives on Spyre."""
    layer = _layer()
    layer.process_weights_after_loading()
    assert layer._w_dev is None


# ---------------------------------------------------------------------------
# On-card: numerics and no CPU fallback
# ---------------------------------------------------------------------------


@pytest.mark.conv
@pytest.mark.parametrize(
    "patch,height,width",
    [
        (16, 64, 64),  # stick-aligned patch grid (4x4 patches)
        (16, 272, 272),  # 17x17 patches, coprime with the stick
        (14, 336, 308),  # 24x22 at Ministral-3's patch size, non-square
        (14, 345, 311),  # border cropped
        (14, 1540, 1540),  # Ministral-3's largest image: 110x110 patches
    ],
)
@pytest.mark.parametrize("use_bias", [False, True])
def test_patch_conv_on_card_matches_cpu_reference(patch, height, width, use_bias):
    """The direct conv on-card matches a CPU `F.conv2d`, element by element (so a
    single bad row cannot hide), without any torch-spyre CPU fallback."""
    if not spyre_available():
        pytest.skip("Spyre device not available")
    from torch_spyre.ops.fallbacks import FallbackWarning

    layer = _layer(kernel=patch, stride=patch, bias=use_bias)

    torch.manual_seed(3)
    x = torch.randn(1, 3, height, width, dtype=torch.float16)
    expected = F.conv2d(
        x.float(),
        layer.weight.data.float(),
        layer.bias.data.float() if use_bias else None,
        stride=patch,
    )

    layer = layer.to("spyre")
    layer.process_weights_after_loading()
    x_dev = x.to("spyre")
    with warnings.catch_warnings():
        warnings.simplefilter("error", FallbackWarning)
        actual = layer.forward_oot(x_dev)

    assert actual.shape == expected.shape
    # Pixtral's view of the result: [1, patches, out_channels].
    rows = actual.flatten(2).permute(0, 2, 1).cpu().float()[0]
    expected_rows = expected.flatten(2).permute(0, 2, 1)[0]
    torch.testing.assert_close(rows, expected_rows, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
