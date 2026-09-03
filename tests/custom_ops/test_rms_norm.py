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

import sys

import pytest
import torch
from spyre_testing_plugin.pytest_plugin import spyre_available


def reference_rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    eps: float,
    residual: torch.Tensor | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """fp16 RMSNorm reference (no fp32 upcast): an oracle for the device lowering,
    not for fp16-vs-fp32 precision the op does not promise."""
    if residual is not None:
        x = x + residual
        residual = x
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x * torch.rsqrt(variance + eps)
    if weight is not None:
        x_normed = x_normed * weight
    if residual is not None:
        return x_normed, residual
    return x_normed


@pytest.mark.rmsnorm
@pytest.mark.parametrize("batch_size", [1])
# Hidden sizes must be a multiple of 64 (Spyre 128-byte stick / 2 bytes fp16).
@pytest.mark.parametrize("hidden_size", [64, 128, 256, 512])
@pytest.mark.parametrize("use_residual", [False, True])
def test_spyre_rmsnorm_matches_reference(default_vllm_config, batch_size, hidden_size, use_residual):
    """SpyreRMSNorm.forward_oot on device matches the eager fp16 reference."""
    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    eps = 1e-6
    device = "spyre"
    dtype = torch.float16
    torch.manual_seed(42)

    x = torch.randn(batch_size, hidden_size, dtype=dtype)
    layer = SpyreRMSNorm(hidden_size, eps=eps).to(dtype)
    residual = torch.randn(batch_size, hidden_size, dtype=dtype) if use_residual else None

    expected = reference_rms_norm(x, layer.weight.data, eps, residual)

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


def reference_rms_norm_fp64(x: torch.Tensor, weight: torch.Tensor | None, eps: float):
    """Precision oracle: the same maths with no intermediate saturation."""
    xf = x.double()
    out = xf * torch.rsqrt(xf.pow(2).mean(dim=-1, keepdim=True) + eps)
    if weight is not None:
        out = out * weight.double()
    return out


@pytest.mark.rmsnorm
def test_fp16_variance_saturates_above_256():
    """Pins the failure the promotion exists to fix, with no device needed.

    255**2 fits in fp16 and 256**2 does not, so one large activation takes the whole
    row's variance to inf, rsqrt to 0, and the output to zeros.
    """
    assert torch.tensor([255.0], dtype=torch.float16).pow(2).isfinite().all()
    assert torch.tensor([256.0], dtype=torch.float16).pow(2).isinf().all()

    x = torch.full((1, 512), 8.0, dtype=torch.float16)
    x[0, 0] = 400.0
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    assert variance.isinf().all()
    assert (x * torch.rsqrt(variance + 1e-6) == 0).all()


@pytest.mark.rmsnorm
@pytest.mark.parametrize("hidden_size", [512, 5120])
def test_spyre_rmsnorm_survives_large_activations(default_vllm_config, hidden_size):
    """fp32 promotion, forced on regardless of architecture, handles |x| > 256 rows
    where the fp16-only path returns zeros. Needs hardware."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    eps = 1e-6
    torch.manual_seed(42)
    x = (torch.randn(2, hidden_size) * 128.0).half()
    assert x.abs().max() > 256.0, "input must exceed the fp16 square limit"

    layer = SpyreRMSNorm(hidden_size, eps=eps).to(torch.float16)
    layer.spyre_promote_fp32 = True
    expected = reference_rms_norm_fp64(x, layer.weight.data, eps)

    layer.to("spyre")
    actual = layer.forward_oot(x.to("spyre")).cpu().double()

    assert torch.isfinite(actual).all(), "output saturated"
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@pytest.mark.rmsnorm
@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (["Mistral3ForConditionalGeneration"], True),
        (["Mistral3ForConditionalGeneration", "SomeOtherArch"], True),
        (["LlamaForCausalLM"], False),
        ([], False),
    ],
)
def test_promotes_fp32_predicate(architectures, expected):
    from spyre_inference.custom_ops.rms_norm import _promotes_fp32

    assert _promotes_fp32(architectures) is expected


@pytest.mark.rmsnorm
@pytest.mark.parametrize(
    ("architectures", "expected"),
    [
        (["Mistral3ForConditionalGeneration"], True),
        (["LlamaForCausalLM"], False),
    ],
)
def test_spyre_rmsnorm_gates_fp32_promotion_by_architecture(
    monkeypatch, default_vllm_config, architectures, expected
):
    """Only Ministral pays the CPU round-trip; other architectures keep the plain
    fp16 path this op had before fp32 promotion was added."""
    from types import SimpleNamespace

    from spyre_inference.custom_ops import rms_norm

    default_vllm_config.model_config = SimpleNamespace(
        hf_config=SimpleNamespace(architectures=architectures)
    )
    monkeypatch.setattr(rms_norm, "get_current_vllm_config", lambda: default_vllm_config)

    layer = rms_norm.SpyreRMSNorm(128, eps=1e-6)
    assert layer.spyre_promote_fp32 is expected


@pytest.mark.rmsnorm
def test_rmsnorm_oot_dispatch(default_vllm_config):
    """Verify RMSNorm OOT registration: class swap."""
    from vllm.model_executor.layers.layernorm import RMSNorm

    from spyre_inference.custom_ops.rms_norm import SpyreRMSNorm

    layer = RMSNorm(128, eps=1e-6)

    # OOT class swap: RMSNorm.__new__ should produce SpyreRMSNorm
    assert isinstance(layer, SpyreRMSNorm)

    # dispatch_forward should have selected forward_oot
    assert layer._forward_method == layer.forward_oot


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
