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

"""Layer-by-layer Spyre-vs-CPU debug tool (standalone, edits no existing code).

Localizes the first decoder-layer / sub-block whose Spyre output diverges from a
HuggingFace-on-CPU reference (fp16). The HF-CPU path is the same reference
`torch_spyre_inference.py --compare-with-cpu` uses, except that one only compares
the final *text*; here we capture and diff *per-layer* activations.

Two runs, one prompt, eager mode:
  1. Spyre / current code — build the real vLLM LLM (reusing torch_spyre_inference's
     LLM kwargs), runtime-monkeypatch the model runner to install forward hooks on
     each decoder layer + its Spyre-op sub-blocks, run one prefill, capture outputs.
  2. HF transformers on CPU — same model, same submodule hooks, forward the *exact*
     input_ids Spyre saw, capture outputs.
Then diff each (layer, boundary) and flag the first that crosses fp16 noise.

Run (single Spyre accelerator -> serial only):
    uv run --no-sync python examples/offline_inference/debug_layer_compare.py \
      --model /home/senuser/Ministral-3-14B-Instruct-2512-BF16
"""

import argparse
import os

# Must be set BEFORE importing vllm. Run the engine core in THIS process so the
# runtime monkeypatch + hooks below are visible to the worker's model. Force it
# (not setdefault) — if the environment already exports "1", the worker runs
# out-of-process and no hooks fire.
os.environ["VLLM_PLUGINS"] = "spyre_inference"
if os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING") not in (None, "0"):
    print("[debug] overriding VLLM_ENABLE_V1_MULTIPROCESSING -> 0 (need in-process hooks)")
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

import torch
import torch.nn as nn

# Import the real offline script "underneath" so its plugin/env setup runs in the
# same order; the LLM(...) kwargs below mirror its main() exactly.
import torch_spyre_inference  # noqa: F401

# Sub-block boundaries that correspond 1:1 to Spyre custom ops, per decoder layer.
# (These module names exist on both vLLM and HF Mistral/Llama decoder layers.)
SUBMODULES = ("input_layernorm", "self_attn", "post_attention_layernorm", "mlp")

# max_abs above this (per boundary) is treated as a real divergence, not fp16 noise.
NOISE_THRESHOLD = 0.1

# vLLM runs warmup/_dummy_run forwards during LLM(...) construction, BEFORE the real
# prompt. Hooks stay inert until we arm them right before llm.generate(), so the first
# captured forward is the real prefill (not a dummy warmup run).
ARMED = {"on": False}


def _tensor_of(out):
    """Module outputs may be a tuple (e.g. vLLM RMSNorm returns (normed, residual),
    attention/mlp may wrap in tuples). Take the first tensor."""
    if isinstance(out, tuple):
        out = out[0]
    return out


def _find_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Return the decoder-layer ModuleList across vLLM and HF module trees."""
    for path in (
        ("model", "layers"),
        ("language_model", "model", "layers"),  # HF Mistral3 wrapper
        ("model", "language_model", "model", "layers"),  # vLLM Mistral3 wrapper
        ("model", "language_model", "layers"),
    ):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj
    # Fallback: the largest ModuleList in the tree is the decoder stack.
    best = None
    for _, m in model.named_modules():
        if isinstance(m, nn.ModuleList) and (best is None or len(m) > len(best)):
            best = m
    if best is None or len(best) == 0:
        raise RuntimeError("Could not locate decoder layers in the model tree.")
    return best


def _install_hooks(top_model: nn.Module, store: dict) -> None:
    """Register capture hooks. `store` gets keys: (layer_idx, name) -> tensor.

    Hooks are inert until ARMED['on'] is True (set right before the real generate),
    so warmup/_dummy_run forwards are skipped. The first armed forward (the prefill)
    is captured; `store['done']` then latches so later decode steps aren't captured.
    """
    store["done"] = False
    layers = _find_decoder_layers(top_model)

    def make_capture(key):
        def hook(module, inp, out):
            if not ARMED["on"] or store["done"]:
                return
            store[key] = _tensor_of(out).detach().to("cpu").float()

        return hook

    for i, layer in enumerate(layers):
        layer.register_forward_hook(make_capture((i, "layer")))
        for name in SUBMODULES:
            sub = getattr(layer, name, None)
            if isinstance(sub, nn.Module):
                sub.register_forward_hook(make_capture((i, name)))

    def done_hook(module, args, out):
        # Fires after the full model forward; latch so decode steps aren't captured.
        if ARMED["on"] and not store["done"] and any(isinstance(k, tuple) for k in store):
            store["done"] = True

    top_model.register_forward_hook(done_hook)


# --------------------------------------------------------------------------- #
# Run 2: Spyre / current code
# --------------------------------------------------------------------------- #
def run_spyre(args) -> dict:
    from vllm import LLM, SamplingParams

    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    store: dict = {}

    orig_load_model = TorchSpyreModelRunner.load_model

    def patched_load_model(self, *a, **k):
        orig_load_model(self, *a, **k)
        _install_hooks(self.get_model(), store)
        print("[debug] installed Spyre-side hooks on the decoder layers")

    TorchSpyreModelRunner.load_model = patched_load_model
    try:
        llm = LLM(
            model=args.model,
            tokenizer=args.model,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            tensor_parallel_size=1,
            max_num_batched_tokens=args.max_num_batched_tokens,
            dtype="float16",
            enforce_eager=args.enforce_eager,
            num_gpu_blocks_override=args.num_gpu_blocks_override,
        )
        # Arm hooks only now — LLM(...) above already ran all warmup/_dummy_run
        # forwards, which we want to skip. The first armed forward is the real prefill.
        ARMED["on"] = True
        # Only the prefill is needed to compare the forward pass.
        outputs = llm.generate([args.prompt], SamplingParams(max_tokens=1, temperature=0.0))
    finally:
        ARMED["on"] = False
        TorchSpyreModelRunner.load_model = orig_load_model

    # The exact prompt tokens vLLM used — robust, independent of the forward hooks.
    store["input_ids"] = torch.tensor(outputs[0].prompt_token_ids, dtype=torch.long)

    n_boundaries = sum(isinstance(k, tuple) for k in store)
    if n_boundaries == 0:
        raise RuntimeError(
            "No layer activations captured — the model ran out-of-process, so the "
            "monkeypatch/hooks never fired. VLLM_ENABLE_V1_MULTIPROCESSING is forced to "
            "0 at import; if you still see this, the Spyre worker is spawning a separate "
            "process anyway. Check for a WorkerProc/executor that ignores the flag."
        )
    return store


# --------------------------------------------------------------------------- #
# Run 1: HuggingFace transformers on CPU (reference)
# --------------------------------------------------------------------------- #
def run_hf_cpu(args, input_ids: torch.Tensor) -> dict:
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText

    # Match the Spyre run's fp16 weights so greedy paths are comparable.
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    except ValueError:
        # Mistral3ForConditionalGeneration isn't accepted by AutoModelForCausalLM.
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float16)
    model.eval()

    store: dict = {}
    _install_hooks(model, store)

    ids = input_ids.to("cpu")
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    ARMED["on"] = True  # HF does no warmup; single forward = the reference prefill.
    try:
        with torch.no_grad():
            try:
                model(input_ids=ids)
            except (TypeError, ValueError):
                # Multimodal wrapper may require reaching the text decoder directly.
                lm = getattr(getattr(model, "model", model), "language_model", None)
                lm = lm or getattr(model, "language_model", None)
                if lm is None:
                    raise
                lm(input_ids=ids)
    finally:
        ARMED["on"] = False
    return store


# --------------------------------------------------------------------------- #
def _aligned(a: torch.Tensor, b: torch.Tensor):
    """Squeeze HF batch dim, flatten to [tokens, hidden], trim to common length."""
    if a.dim() == 3:
        a = a[0]
    if b.dim() == 3:  # HF is [1, seq, H]; Spyre prefill is flattened [seq, H]
        b = b[0]
    n = min(a.shape[0], b.shape[0])
    return a[:n].reshape(n, -1), b[:n].reshape(n, -1)


def compare(spyre: dict, hf: dict) -> None:
    keys = [k for k in spyre if isinstance(k, tuple) and k in hf]
    order = {"input_layernorm": 0, "self_attn": 1, "post_attention_layernorm": 2,
             "mlp": 3, "layer": 4}

    print("\n================ per-layer Spyre vs HF-CPU diff ================")
    print(f"{'layer':>5} {'boundary':>26} {'max_abs':>12} {'mean_abs':>12} {'rel':>10}")
    layer_mean = {}  # layer_idx -> mean_abs of the residual-stream ('layer') row
    for key in sorted(keys, key=lambda k: (k[0], order.get(k[1], 9))):
        a, b = _aligned(spyre[key], hf[key])
        diff = (a - b).abs()
        max_abs = diff.max().item()
        mean_abs = diff.mean().item()
        denom = b.abs().max().item() or 1.0
        rel = max_abs / denom
        if key[1] == "layer":
            layer_mean[key[0]] = mean_abs
        print(f"{key[0]:>5} {key[1]:>26} {max_abs:>12.4f} {mean_abs:>12.5f} {rel:>10.4f}")
    print("===============================================================")

    # --- Culprit = first layer whose residual-stream mean_abs jumps (>3x prev) ---
    # The residual-stream ('layer') rows are the reliable cross-impl boundary; the
    # *_layernorm rows are inflated by vLLM's residual-fused RMSNorm. Ignore max_abs
    # (dominated by a single persistent 'massive activation' coordinate) and use the
    # systematic mean_abs trend instead.
    JUMP, FLOOR = 3.0, 0.05
    culprit = None
    for i in sorted(layer_mean):
        prev = layer_mean.get(i - 1, 0.0)
        if layer_mean[i] > FLOOR and layer_mean[i] > JUMP * max(prev, 1e-6):
            culprit = i
            break

    # --- Locate the worst coordinate of each layer's residual diff (massive act?) ---
    print("\n--- residual-stream diff: worst (token, dim) per layer ---")
    print(f"{'layer':>5} {'token':>6} {'dim':>6} {'spyre':>12} {'hf':>12} {'abs_diff':>12}")
    for i in sorted(layer_mean):
        a, b = _aligned(spyre[(i, "layer")], hf[(i, "layer")])
        diff = (a - b).abs()
        flat = int(diff.argmax())
        tok, dim = divmod(flat, diff.shape[1])
        sp, hfv = a[tok, dim].item(), b[tok, dim].item()
        print(f"{i:>5} {tok:>6} {dim:>6} {sp:>12.3f} {hfv:>12.3f} {diff[tok, dim].item():>12.3f}")

    print("===============================================================")
    if culprit is None:
        print("No clear residual-stream jump — divergence may be gradual, below the "
              "floor, or downstream (final norm / lm_head / sampling).")
    else:
        print(f"Suspected culprit layer (residual-stream mean jump): layer {culprit}.")
        print("Inspect that layer's sub-block rows (mlp vs self_attn) above to see which "
              "injected it, and the worst-(token,dim) table: if HF has a large value there "
              "and Spyre ~0, Spyre is dropping a 'massive activation' (early-MLP outlier).")
        print("Next: op-level repro of that sub-block at the captured real input.")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, required=True)
    p.add_argument("--max-model-len", type=int, default=2048, dest="max_model_len")
    p.add_argument(
        "--max-num-batched-tokens", type=int, default=2, dest="max_num_batched_tokens"
    )
    p.add_argument(
        "--num-gpu-blocks-override", type=int, default=None, dest="num_gpu_blocks_override"
    )
    p.add_argument(
        "--enforce-eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        dest="enforce_eager",
        help="Skip torch.compile and run eager (default on; --no-enforce-eager to compile).",
    )
    p.add_argument(
        "--prompt", type=str, default="What are IBMs main businesses?",
        help="Single prompt to run through both models (mirrors torch_spyre_inference's "
        "simple_prompt).",
    )
    args = p.parse_args()

    print(f"[debug] prompt: {args.prompt!r}")
    print("[debug] === RUN 2: Spyre (current code) ===")
    spyre = run_spyre(args)
    print(f"[debug] captured {sum(isinstance(k, tuple) for k in spyre)} Spyre boundaries")

    print("[debug] === RUN 1: HuggingFace on CPU (reference) ===")
    hf = run_hf_cpu(args, spyre["input_ids"])
    print(f"[debug] captured {sum(isinstance(k, tuple) for k in hf)} HF boundaries")

    compare(spyre, hf)


if __name__ == "__main__":
    main()
