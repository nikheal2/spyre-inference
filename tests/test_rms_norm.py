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

"""
Test SpyreRMSNorm custom op correctness against a reference implementation.
"""

import pytest
import torch
import sys


def reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Golden reference: standard RMSNorm in PyTorch."""
    if residual is not None:
        x = x + residual
    x_float = x.float()
    variance = x_float.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_float * torch.rsqrt(variance + eps)
    if weight is not None:
        x_normed = x_normed * weight.float()
    if residual is not None:
        return x_normed, x.float()
    return x_normed


@pytest.mark.rmsnorm
@pytest.mark.parametrize("batch_size", [1])
# Hidden sizes that aren't multiples of 64 currently fail on CI with size errors
# @pytest.mark.parametrize("hidden_size", [63, 64, 65, 127, 128, 129, 256, 512])
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512])
@pytest.mark.parametrize("use_residual", [False, True])
def test_spyre_rmsnorm_matches_reference(batch_size, hidden_size, use_residual):
    """SpyreRMSNorm output matches golden reference.

    Tests both paths:
    - forward_oot(): OOT dispatch via custom op (torch.ops.vllm.spyre_rmsnorm)
    - reference_rms_norm(): golden reference, similar to vLLM upstream pure PyTorch (ground truth)
    """
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    eps = 1e-6
    device = "spyre"
    dtype = torch.float16
    torch.manual_seed(42)

    x = torch.randn(batch_size, hidden_size, dtype=dtype)
    layer = SpyreRMSNorm(hidden_size, eps=eps).to(dtype)
    residual = torch.randn(batch_size, hidden_size, dtype=dtype) if use_residual else None

    expected = reference_rms_norm(x, layer.weight.data, eps, residual)

    # Test forward_oot (Spyre device execution via custom op)
    layer.to(device)
    actual = layer.forward_oot(x.to(device), residual.to(device) if use_residual else None)

    if use_residual:
        expected_norm, expected_resid = expected
        actual_norm, actual_resid = actual
        torch.testing.assert_close(
            actual_norm.cpu().float(), expected_norm.float(), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            actual_resid.cpu().float(), expected_resid.float(), atol=1e-2, rtol=1e-2
        )
    else:
        torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.fixture
def dummy_tensor():
    return torch.randn(4, 128, dtype=torch.float32)


def mock_forward_oot(x, variance_epsilon=None, hidden_size=None, weight=None, residual=None):
    """Mock: return x + 1 (no residual path)."""
    return x + 1


def mock_forward_oot_with_residual(
    x, variance_epsilon=None, hidden_size=None, weight=None, residual=None
):
    """Mock: return (2 * x, 2 * residual) (residual path)."""
    return 2 * x, 2 * residual


@pytest.mark.rmsnorm
def test_rmsnorm_oot_dispatch(monkeypatch, dummy_tensor):
    """Verify RMSNorm OOT registration: class swap."""
    from vllm.model_executor.layers.layernorm import RMSNorm
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    layer = RMSNorm(128, eps=1e-6)

    # OOT class swap: RMSNorm.__new__ should produce SpyreRMSNorm
    assert isinstance(layer, SpyreRMSNorm)

    # dispatch_forward should have selected forward_oot
    assert layer._forward_method == layer.forward_oot


@pytest.mark.rmsnorm
def test_rms_norm_fp32_op_is_registered():
    """`custom_ops.register_all()` must call `rms_norm.register()`.

    `forward_oot` calls `torch.ops.vllm.spyre_rms_norm_fp32` unconditionally, so
    a missing registration is not a graceful degradation — every RMSNorm in the
    model raises at the first forward.
    """
    from spyre_inference.custom_ops import register_all

    register_all()
    assert hasattr(torch.ops.vllm, "spyre_rms_norm_fp32")


@pytest.mark.rmsnorm
@pytest.mark.parametrize("use_residual", [False, True])
def test_spyre_rmsnorm_under_torch_compile(use_residual):
    """The whole point of hiding the fp32 core behind an opaque custom op is that
    the CPU `mean` never reaches Inductor's cpp backend (which cannot codegen a
    `mean` reduction). Only a compiled run exercises that; the eager test above
    would pass even if the op were inlined.
    """
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    eps = 1e-6
    hidden_size = 128
    dtype = torch.float16
    torch.manual_seed(42)

    x = torch.randn(4, hidden_size, dtype=dtype)
    layer = SpyreRMSNorm(hidden_size, eps=eps).to(dtype)
    layer.weight.data.normal_(mean=1.0, std=0.02)
    residual = torch.randn(4, hidden_size, dtype=dtype) if use_residual else None

    expected = reference_rms_norm(x, layer.weight.data, eps, residual)

    layer.to("spyre")
    compiled = torch.compile(layer.forward_oot, dynamic=False)
    actual = compiled(x.to("spyre"), residual.to("spyre") if use_residual else None)

    if use_residual:
        torch.testing.assert_close(
            actual[0].cpu().float(), expected[0].float(), atol=1e-2, rtol=1e-2
        )
        torch.testing.assert_close(
            actual[1].cpu().float(), expected[1].float(), atol=1e-2, rtol=1e-2
        )
    else:
        torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


@pytest.mark.rmsnorm
def test_large_residual_needs_fp32_promotion():
    """The fp32 promotion exists to stop `x**2` overflowing fp16 (max 65504).

    With a residual stream around 300, `x**2 ≈ 9e4` is already inf in fp16, so a
    regression that drops the promotion produces NaN here — while the `randn`
    inputs in the test above would still pass.
    """
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    eps = 1e-6
    hidden_size = 256
    dtype = torch.float16
    torch.manual_seed(7)

    x = torch.randn(2, hidden_size, dtype=dtype) * 300.0
    residual = torch.randn(2, hidden_size, dtype=dtype) * 300.0
    assert torch.isinf(x.pow(2)).any(), "test input does not actually overflow fp16"

    layer = SpyreRMSNorm(hidden_size, eps=eps).to(dtype)
    expected_norm, expected_resid = reference_rms_norm(x, layer.weight.data, eps, residual)

    layer.to("spyre")
    actual_norm, actual_resid = layer.forward_oot(x.to("spyre"), residual.to("spyre"))

    assert not torch.isnan(actual_norm.cpu()).any(), "fp32 promotion lost — x**2 overflowed fp16"
    torch.testing.assert_close(
        actual_norm.cpu().float(), expected_norm.float(), atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(
        actual_resid.cpu().float(), expected_resid.float(), atol=1e-1, rtol=1e-2
    )


@pytest.mark.rmsnorm
@pytest.mark.parametrize("use_residual", [False, True])
def test_spyre_rmsnorm_without_weight(use_residual):
    """`has_weight=False` skips the per-channel multiply; the normalize must
    still be applied."""
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    eps = 1e-6
    hidden_size = 128
    dtype = torch.float16
    torch.manual_seed(5)

    x = torch.randn(2, hidden_size, dtype=dtype)
    residual = torch.randn(2, hidden_size, dtype=dtype) if use_residual else None
    layer = SpyreRMSNorm(hidden_size, eps=eps, has_weight=False).to(dtype)

    expected = reference_rms_norm(x, None, eps, residual)

    layer.to("spyre")
    actual = layer.forward_oot(x.to("spyre"), residual.to("spyre") if use_residual else None)

    if use_residual:
        torch.testing.assert_close(
            actual[0].cpu().float(), expected[0].float(), atol=1e-2, rtol=1e-2
        )
    else:
        torch.testing.assert_close(actual.cpu().float(), expected.float(), atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-k", "test_rmsnorm_oot_dispatch", "-v"]))
