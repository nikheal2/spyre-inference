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

"""Shape arithmetic for the TP all_reduce row padding; the hardware half is in
``tests/probes/`` (``all_reduce_hidden5120_*``)."""

from __future__ import annotations

import pytest
import torch

from spyre_inference.distributed.spyre_communicator import (
    _COLLECTIVE_CORES,
    _STICK_BYTES,
    collective_row_pad,
)

FP16_STICK = _STICK_BYTES // 2  # 64 elements


def _pad(rows: int, hidden: int, dtype: torch.dtype = torch.float16) -> int:
    return collective_row_pad(torch.empty((rows, hidden), dtype=dtype))


@pytest.mark.parametrize(
    ("rows", "pad"),
    [
        (1, 1),  # 80 sticks -> 160: the Ministral TP=2 decode that fails to build
        (2, 0),
        (3, 1),
        (4, 0),
        (5, 1),
        (512, 0),  # the largest default prefill bucket
    ],
)
def test_hidden_5120_pads_to_a_multiple_of_the_core_width(rows: int, pad: int) -> None:
    assert _pad(rows, 5120) == pad
    padded_sticks = (rows + pad) * 5120 // FP16_STICK
    assert padded_sticks % _COLLECTIVE_CORES == 0


@pytest.mark.parametrize("rows", [1, 2, 3, 7, 512])
def test_hidden_4096_never_pads(rows: int) -> None:
    """64 sticks per row divides the core width, so micro-g3.3 is untouched."""
    assert _pad(rows, 4096) == 0


@pytest.mark.parametrize("numel", [64, 128, 256, 1024])
def test_small_counts_are_left_alone(numel: int) -> None:
    """At most one stick per core: these build today."""
    assert _pad(1, numel) == 0


def test_one_dim_input_is_left_alone() -> None:
    assert collective_row_pad(torch.empty((1024,), dtype=torch.float16)) == 0


def test_row_that_is_not_a_whole_number_of_sticks_is_left_alone() -> None:
    """Padding rows cannot fix sub-stick geometry."""
    assert _pad(1, 100) == 0


def test_empty_input_is_left_alone() -> None:
    assert _pad(0, 5120) == 0


def test_three_dim_input_uses_the_whole_row() -> None:
    """A [T, H, D] row is H*D elements."""
    assert collective_row_pad(torch.empty((1, 40, 128), dtype=torch.float16)) == 1


def test_fp32_stick_holds_half_as_many_elements() -> None:
    """The rule is bytes per row, not elements: 5120 fp32 is already 160 sticks."""
    assert _pad(1, 5120, dtype=torch.float32) == 0
    assert _pad(1, 2560, dtype=torch.float32) == 1
