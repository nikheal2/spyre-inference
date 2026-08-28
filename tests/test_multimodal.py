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

"""End-to-end multimodal (Pixtral vision + Ministral-3 decoder) tests.

No upstream vLLM VLM test is runnable on Spyre today (see the
`tests/models/multimodal/generation/test_common.py` entry in
`upstream_tests.yaml` for why), so the end-to-end guarantee for the vision path
lives here.

Rather than pin an exact reference string — brittle, and meaningless against a
synthetic image — these tests assert **self-consistency**, which is what the
bugs this branch fixed actually broke:

  * `test_batch_size_does_not_change_output` targets the Inductor tail-tile bug
    directly: the last sequence of every batch decoded to garbage, so the same
    prompt produced different text at batch size 1 vs 2. See the
    `_PAD_BATCH_ROWS` workaround in `spyre_model_runner.py`.
  * `test_eager_and_compiled_agree` targets the same bug from the other side —
    eager was always correct, compile was not.

Both are slow (model load + vision tower) and need a Spyre card, so they carry
the `multimodal` marker and skip without hardware.
"""

import io
import sys

import pytest

from spyre_testing_plugin.pytest_plugin import spyre_available

# The model this branch was brought up against (Pixtral vision encoder +
# multimodal projector + Ministral-3 text decoder).
MODEL = "mistralai/Ministral-3-14B-Instruct-2512-BF16"

MAX_MODEL_LEN = 4096
MAX_TOKENS = 16


def _synthetic_image_data_uri() -> str:
    """A deterministic image built in-process — no network, no binary asset.

    Content does not matter: these tests compare runs against each other, not
    against a reference answer. Size does: the patch grid must exercise the
    coprime-with-64 case the vision SDPA/conv patches exist for, so 176x176
    (11x11 = 121 patches at patch_size 16) is deliberate.
    """
    import base64

    from PIL import Image

    image = Image.new("RGB", (176, 176))
    pixels = image.load()
    for y in range(176):
        for x in range(176):
            pixels[x, y] = ((x * 7) % 256, (y * 5) % 256, ((x + y) * 3) % 256)

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("utf-8")


def _conversation(uri: str):
    return [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": uri}},
                {"type": "text", "text": "Describe this image in one short sentence."},
            ],
        }
    ]


def _generate(conversations, enforce_eager: bool):
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=MODEL,
        max_model_len=MAX_MODEL_LEN,
        max_num_seqs=len(conversations),
        dtype="float16",
        enforce_eager=enforce_eager,
        limit_mm_per_prompt={"image": 1},
    )
    outputs = llm.chat(
        conversations,
        SamplingParams(max_tokens=MAX_TOKENS, temperature=0.0),
    )
    return [o.outputs[0].text for o in outputs]


@pytest.mark.multimodal
def test_single_image_prompt_produces_output():
    """Smoke: the whole vision path (conv patch embed -> vision rope -> padded
    SDPA -> patch merger -> projector norm -> decoder) runs and decodes text."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    uri = _synthetic_image_data_uri()
    (text,) = _generate([_conversation(uri)], enforce_eager=True)

    assert text.strip(), "empty generation from the multimodal path"


@pytest.mark.multimodal
def test_batch_size_does_not_change_output():
    """Greedy decoding of the same prompt must not depend on batch size.

    This is the end-to-end guard for the Inductor tail-tile workaround: without
    the appended dummy row, the *last* sequence of the batch decoded to garbage,
    so the batch-of-2 run disagreed with the batch-of-1 run.
    """
    if not spyre_available():
        pytest.skip("Spyre device not available")

    uri = _synthetic_image_data_uri()
    conversation = _conversation(uri)

    (single,) = _generate([conversation], enforce_eager=False)
    batched = _generate([conversation, conversation], enforce_eager=False)

    assert batched[0] == single, "first sequence differs between batch sizes"
    assert batched[1] == single, (
        "last sequence of the batch differs — the tail-row corruption is back; "
        "check _PAD_BATCH_ROWS / _batch_pad_rows in spyre_model_runner.py"
    )


@pytest.mark.multimodal
def test_eager_and_compiled_agree():
    """Eager is the known-correct reference; torch.compile must match it."""
    if not spyre_available():
        pytest.skip("Spyre device not available")

    uri = _synthetic_image_data_uri()
    conversation = _conversation(uri)

    (eager,) = _generate([conversation], enforce_eager=True)
    (compiled,) = _generate([conversation], enforce_eager=False)

    assert compiled == eager, (
        "compiled output diverges from eager — the compile path regressed "
        "(tail-row padding, RMSNorm opaque op, or a Pixtral patch)"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
