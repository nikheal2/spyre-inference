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

"""Tests for the Inductor tail-row padding workaround in `_SpyreModelWrapper`.

Inductor miscompiles the final row of the flat `[num_tokens, hidden]` activation,
so the last sequence of every batch decodes to garbage under
`CompilationMode.STOCK_TORCH_COMPILE`. The workaround appends dummy row(s) before
the forward and crops them off the result.

The padding logic is pure tensor bookkeeping, so it runs on CPU with a stub
model — no Spyre hardware required. What is worth guarding here:

  * eager mode must not pay the cost (the bug is compile-only),
  * only the token-dimension inputs may be padded,
  * the crop must undo the pad for every output shape the model can return
    (a bare tensor and a tuple), which is where a `tree_map` mismatch would
    silently truncate real rows.
"""

import sys

import pytest
import torch
import torch.nn as nn


class _EchoModel(nn.Module):
    """Returns its `input_ids` (as float) so the caller can see exactly which
    rows survived the pad/crop round-trip."""

    def __init__(self, extra_outputs: int = 0):
        super().__init__()
        self._extra_outputs = extra_outputs
        # A parameter so `.modules()`/device introspection behaves like a real model.
        self.dummy = nn.Parameter(torch.zeros(1))

    def forward(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
        out = input_ids if input_ids is not None else inputs_embeds
        out = out.float()
        if self._extra_outputs:
            return (out,) * (1 + self._extra_outputs)
        return out


def _wrapper(model, compiled: bool):
    from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

    return _SpyreModelWrapper(model, torch.device("cpu"), [], compiled)


# ---------------------------------------------------------------------------
# _pad_rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("n_pad", [1, 3])
def test_pad_rows_repeats_last_row(n_pad):
    from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

    x = torch.tensor([[1, 2], [3, 4], [5, 6]])
    padded = _SpyreModelWrapper._pad_rows(x, n_pad)

    assert padded.shape == (3 + n_pad, 2)
    torch.testing.assert_close(padded[:3], x)
    for i in range(n_pad):
        torch.testing.assert_close(padded[3 + i], x[-1])


def test_pad_rows_handles_1d_and_empty():
    """1-D `positions` must pad like `input_ids`; a 0-row tensor is left alone
    (there is no last row to repeat)."""
    from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

    positions = torch.tensor([0, 1, 2])
    assert _SpyreModelWrapper._pad_rows(positions, 2).tolist() == [0, 1, 2, 2, 2]

    empty = torch.zeros(0, 4)
    assert _SpyreModelWrapper._pad_rows(empty, 1).shape == (0, 4)


def test_pad_rows_passes_through_non_tensors():
    from spyre_inference.v1.worker.spyre_model_runner import _SpyreModelWrapper

    assert _SpyreModelWrapper._pad_rows(None, 1) is None
    assert _SpyreModelWrapper._pad_rows("not-a-tensor", 1) == "not-a-tensor"


# ---------------------------------------------------------------------------
# _batch_pad_rows (the gate)
# ---------------------------------------------------------------------------


def test_no_pad_in_eager_mode():
    """Eager is unaffected by the Inductor bug and must not pay the extra row."""
    w = _wrapper(_EchoModel(), compiled=False)
    assert w._batch_pad_rows({"input_ids": torch.arange(4)}) == 0


def test_pad_applied_in_compile_mode():
    from spyre_inference.v1.worker.spyre_model_runner import _PAD_BATCH_ROWS

    w = _wrapper(_EchoModel(), compiled=True)
    assert w._batch_pad_rows({"input_ids": torch.arange(4)}) == _PAD_BATCH_ROWS


def test_inputs_embeds_is_the_anchor_when_input_ids_absent():
    """The multimodal path passes `inputs_embeds` instead of `input_ids`; the
    pad must still engage or the last image prompt decodes to garbage."""
    from spyre_inference.v1.worker.spyre_model_runner import _PAD_BATCH_ROWS

    w = _wrapper(_EchoModel(), compiled=True)
    assert w._batch_pad_rows({"inputs_embeds": torch.randn(4, 8)}) == _PAD_BATCH_ROWS


def test_no_pad_without_a_token_anchor():
    w = _wrapper(_EchoModel(), compiled=True)
    assert w._batch_pad_rows({"positions": torch.arange(4)}) == 0


def test_pad_can_be_disabled(monkeypatch):
    """`_PAD_BATCH_ROWS = 0` (via `SPYRE_PAD_BATCH_ROWS=0`) turns the workaround
    off, which is how the underlying torch-spyre bug gets re-tested. Patch the
    resolved module global rather than reloading the module — a reload would
    rebind the classes other tests in the session hold references to."""
    from spyre_inference.v1.worker import spyre_model_runner as runner

    monkeypatch.setattr(runner, "_PAD_BATCH_ROWS", 0)
    w = _wrapper(_EchoModel(), compiled=True)
    assert w._batch_pad_rows({"input_ids": torch.arange(4)}) == 0


# ---------------------------------------------------------------------------
# End-to-end pad -> forward -> crop
# ---------------------------------------------------------------------------


def test_round_trip_returns_original_rows():
    """The wrapper's output must have exactly the caller's token count, with the
    dummy rows removed — not shifted, not truncated into the real rows."""
    from spyre_inference.v1.worker.spyre_model_runner import _PAD_BATCH_ROWS

    if _PAD_BATCH_ROWS == 0:
        pytest.skip("padding workaround disabled in this environment")

    w = _wrapper(_EchoModel(), compiled=True)
    input_ids = torch.arange(5)
    out = w(input_ids=input_ids, positions=torch.arange(5))

    assert out.shape[0] == 5
    torch.testing.assert_close(out, input_ids.float())


def test_round_trip_crops_every_tuple_output():
    """Models returning a tuple (e.g. hidden states + residual) must have every
    element cropped, not just the first."""
    from spyre_inference.v1.worker.spyre_model_runner import _PAD_BATCH_ROWS

    if _PAD_BATCH_ROWS == 0:
        pytest.skip("padding workaround disabled in this environment")

    w = _wrapper(_EchoModel(extra_outputs=1), compiled=True)
    input_ids = torch.arange(5)
    out = w(input_ids=input_ids, positions=torch.arange(5))

    assert isinstance(out, tuple) and len(out) == 2
    for element in out:
        assert element.shape[0] == 5
        torch.testing.assert_close(element, input_ids.float())


def test_only_paddable_inputs_are_extended():
    """Anything outside `_PADDABLE_INPUTS` (kv caches, intermediate tensors)
    must reach the model untouched — padding them would corrupt the cache."""
    from spyre_inference.v1.worker.spyre_model_runner import _PAD_BATCH_ROWS

    if _PAD_BATCH_ROWS == 0:
        pytest.skip("padding workaround disabled in this environment")

    seen = {}

    class _Recorder(_EchoModel):
        def forward(self, input_ids=None, positions=None, inputs_embeds=None, **kwargs):
            seen["input_ids"] = input_ids.shape[0]
            seen["positions"] = positions.shape[0]
            seen["other"] = kwargs["intermediate_tensors"].shape[0]
            return super().forward(input_ids=input_ids)

    w = _wrapper(_Recorder(), compiled=True)
    w(
        input_ids=torch.arange(5),
        positions=torch.arange(5),
        intermediate_tensors=torch.zeros(5, 3),
    )

    assert seen["input_ids"] == 5 + _PAD_BATCH_ROWS
    assert seen["positions"] == 5 + _PAD_BATCH_ROWS
    assert seen["other"] == 5, "non-token-dim kwargs must not be padded"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
