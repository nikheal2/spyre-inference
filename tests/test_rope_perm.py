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

"""Tests for the RoPE permutation matrices in `v1/worker/spyre_model_runner.py`.

`_rope_perm_matrix` exists because the vision head_dim is 64, so the `d/2 = 32`
half-slice a normal `rotate_half` needs is *half a Spyre stick* and torch-spyre
cannot lay it out. The shuffle is expressed as a full-width `[d, d]` matmul
instead. That rewrite is pure linear algebra, so it is verified here on CPU
against the reference formulas — no Spyre hardware required.

A sign or interleave error in these matrices produces plausible-looking but
subtly wrong attention, which is exactly the failure mode an end-to-end test
struggles to localise.
"""

import sys

import pytest
import torch

HEAD_DIMS = [64, 128]


def reference_rotate_half(x: torch.Tensor) -> torch.Tensor:
    """transformers' `rotate_half`: `cat((-x[..., d//2:], x[..., :d//2]))`."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def reference_pair_swap(x: torch.Tensor) -> torch.Tensor:
    """Swap each `(2k, 2k+1)` element pair along the last dim."""
    out = x.clone()
    out[..., 0::2] = x[..., 1::2]
    out[..., 1::2] = x[..., 0::2]
    return out


@pytest.mark.rotary
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_half_matrix_equals_rotate_half(head_dim):
    """`x @ M_half` reproduces transformers' `rotate_half` exactly."""
    from spyre_inference.v1.worker.spyre_model_runner import _rope_perm_matrix

    torch.manual_seed(0)
    x = torch.randn(3, 5, head_dim, dtype=torch.float16)

    m = _rope_perm_matrix("half", head_dim, torch.device("cpu"))
    assert m.shape == (head_dim, head_dim)

    torch.testing.assert_close(torch.matmul(x, m), reference_rotate_half(x))


@pytest.mark.rotary
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_pair_matrix_equals_pair_swap(head_dim):
    """`x @ M_pair` swaps each `(2k, 2k+1)` pair (no negation — the sign lives
    in the `sin_signed` half of the packed freqs table)."""
    from spyre_inference.v1.worker.spyre_model_runner import _rope_perm_matrix

    torch.manual_seed(1)
    x = torch.randn(2, 7, head_dim, dtype=torch.float16)

    m = _rope_perm_matrix("pair", head_dim, torch.device("cpu"))
    assert m.shape == (head_dim, head_dim)

    torch.testing.assert_close(torch.matmul(x, m), reference_pair_swap(x))


@pytest.mark.rotary
@pytest.mark.parametrize("kind", ["half", "pair"])
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_perm_matrix_is_a_permutation(kind, head_dim):
    """Each matrix is a signed permutation: exactly one non-zero (+-1) per
    row and per column. Catches an off-by-one in the index arithmetic that a
    single random-input comparison could still slip past."""
    from spyre_inference.v1.worker.spyre_model_runner import _rope_perm_matrix

    m = _rope_perm_matrix(kind, head_dim, torch.device("cpu")).float()

    assert torch.equal((m != 0).sum(dim=0), torch.ones(head_dim, dtype=torch.int64))
    assert torch.equal((m != 0).sum(dim=1), torch.ones(head_dim, dtype=torch.int64))
    assert torch.equal(m.abs().sum(), torch.tensor(float(head_dim)))


@pytest.mark.rotary
@pytest.mark.parametrize("head_dim", HEAD_DIMS)
def test_rope_rotate_matmul_matches_neox_formula(head_dim):
    """`_rope_rotate_matmul` is the neox rotation `x*cos + rotate_half(x)*sin`."""
    from spyre_inference.v1.worker.spyre_model_runner import (
        _rope_perm_matrix,
        _rope_rotate_matmul,
    )

    torch.manual_seed(2)
    x = torch.randn(1, 4, 11, head_dim, dtype=torch.float16)
    angles = torch.randn(11, head_dim, dtype=torch.float16)
    cos = angles.cos()[None, None, :, :]
    sin = angles.sin()[None, None, :, :]

    m = _rope_perm_matrix("half", head_dim, torch.device("cpu"))
    expected = x * cos + reference_rotate_half(x) * sin

    torch.testing.assert_close(_rope_rotate_matmul(x, cos, sin, m), expected)


@pytest.mark.rotary
def test_rope_perm_matrix_rejects_unknown_kind():
    """An unknown kind raises rather than silently returning zeros (which would
    turn the rotation into a plain `x*cos` and degrade quality quietly)."""
    from spyre_inference.v1.worker.spyre_model_runner import _rope_perm_matrix

    with pytest.raises(ValueError, match="unknown rope permutation kind"):
        _rope_perm_matrix("bogus", 64, torch.device("cpu"))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
