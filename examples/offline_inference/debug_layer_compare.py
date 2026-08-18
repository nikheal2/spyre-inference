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

# MLP-internal children to hook on the --target-layer, for op-level attribution.
# vLLM MLP has {gate_up_proj (merged), act_fn (SiluAndMul), down_proj}; HF MLP has
# {gate_proj, up_proj, act_fn (SiLU), down_proj}. Hook whichever exist by name.
MLP_CHILDREN = ("gate_up_proj", "gate_proj", "up_proj", "act_fn", "down_proj")

# Which decoder layer to break down at the MLP-op level (set from --target-layer).
TARGET_LAYER = {"i": 2}

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

    def _snap(t):
        # .clone() so a later reuse of Spyre's shared static output buffer can't
        # corrupt an already-captured activation after the fact.
        return t.detach().to("cpu").float().clone()

    def make_capture(key, full_stream=False):
        def hook(module, inp, out):
            if not ARMED["on"] or store["done"]:
                return
            # vLLM fused-residual decoder layer returns (hidden, residual), both the
            # same shape → the true residual stream is their sum. HF returns
            # (hidden_states, present_kv/attn_weights) or a bare tensor → out[1] is
            # not a same-shape tensor, so we keep out[0] (already the full stream).
            if (
                full_stream
                and isinstance(out, tuple)
                and len(out) >= 2
                and isinstance(out[1], torch.Tensor)
                and out[1].shape == out[0].shape
            ):
                store[key] = _snap(out[0]) + _snap(out[1])
            else:
                store[key] = _snap(_tensor_of(out))

        return hook

    def make_input_capture(key):
        def pre_hook(module, args):
            if not ARMED["on"] or store["done"]:
                return
            if args and isinstance(args[0], torch.Tensor):
                store[key] = _snap(args[0])

        return pre_hook

    def make_norm_input_capture(prefix):
        # RMSNorm.forward(x, residual=None) — capture both so we can recompute the
        # norm from Spyre's OWN input. vLLM passes (hidden, residual); HF passes (hidden,).
        def pre_hook(module, args):
            if not ARMED["on"] or store["done"]:
                return
            if args and isinstance(args[0], torch.Tensor):
                store[(*prefix, "h")] = _snap(args[0])
            if len(args) > 1 and isinstance(args[1], torch.Tensor):
                store[(*prefix, "r")] = _snap(args[1])

        return pre_hook

    for i, layer in enumerate(layers):
        layer.register_forward_hook(make_capture((i, "layer"), full_stream=True))
        for name in SUBMODULES:
            sub = getattr(layer, name, None)
            if isinstance(sub, nn.Module):
                sub.register_forward_hook(make_capture((i, name)))
        # Capture the MLP input (post_attention_layernorm output) so the op-repro
        # can feed the exact real large-magnitude input into the Spyre MLP.
        mlp = getattr(layer, "mlp", None)
        if isinstance(mlp, nn.Module):
            mlp.register_forward_pre_hook(make_input_capture((i, "mlp_in")))
        # On the target layer, break the MLP down to its op-level children and capture
        # the RMSNorm inputs, so op-locality can isolate gate_up/silu/down AND the norms.
        if i == TARGET_LAYER["i"]:
            if isinstance(mlp, nn.Module):
                for name in MLP_CHILDREN:
                    child = getattr(mlp, name, None)
                    if isinstance(child, nn.Module):
                        child.register_forward_hook(make_capture(("mlp_int", name)))
            for name in ("input_layernorm", "post_attention_layernorm"):
                norm = getattr(layer, name, None)
                if isinstance(norm, nn.Module):
                    norm.register_forward_pre_hook(make_norm_input_capture(("norm_in", name)))

    def done_hook(module, args, out):
        # Fires after the full model forward; latch so decode steps aren't captured.
        if ARMED["on"] and not store["done"] and any(isinstance(k, tuple) for k in store):
            store["done"] = True

    top_model.register_forward_hook(done_hook)


# --------------------------------------------------------------------------- #
# Cumulative-prefix CPU bisection: force embed_tokens + decoder layers 0..N onto
# CPU (rest stay on Spyre), run the FULL generation, and see when the text turns
# coherent. The layer that flips it from garbage -> coherent holds the bug.
#
# All wiring is a runtime monkeypatch of the loaded model's module forwards; no
# spyre_inference/ code is edited. The CPU layers keep their own per-layer KV
# (a simple append list) since they can't use Spyre's on-device paged KV cache.
# --------------------------------------------------------------------------- #

# Per-forced-layer CPU KV: layer_idx -> {"k":[Tk,Hkv,D], "v":[Tk,Hkv,D], "pos":[Tk]}.
_CPU_KV: dict = {}


def _find_embed_tokens(model: nn.Module) -> nn.Module:
    """Locate the token-embedding module across vLLM and HF module trees."""
    for path in (
        ("model", "embed_tokens"),
        ("language_model", "model", "embed_tokens"),
        ("model", "language_model", "model", "embed_tokens"),
        ("model", "language_model", "embed_tokens"),
    ):
        obj = model
        for attr in path:
            obj = getattr(obj, attr, None)
            if obj is None:
                break
        if isinstance(obj, nn.Module) and hasattr(obj, "weight"):
            return obj
    for name, m in model.named_modules():
        if name.endswith("embed_tokens") and hasattr(m, "weight"):
            return m
    raise RuntimeError("Could not locate embed_tokens in the model tree.")


def _force_embedding_cpu(model: nn.Module, spyre_device) -> None:
    """Replace embed_tokens.forward with a CPU gather (return to the Spyre device)."""
    from spyre_inference.custom_ops.utils import convert

    emb = _find_embed_tokens(model)
    W = emb.weight.detach().to("cpu")  # fp16 [vocab, hidden]
    hidden = W.shape[-1]

    def emb_forward(input_):
        ids = input_.detach().to("cpu").long()
        out = W.index_select(0, ids.flatten()).view(*ids.shape, hidden)
        return convert(out, device=spyre_device)

    emb.forward = emb_forward
    print("[debug][cpu-layers] forced embed_tokens onto CPU")


def _lin(mod: nn.Module):
    """Return (weight, bias) of a linear-like module as CPU fp32 tensors."""
    W = mod.weight.detach().to("cpu").float()
    b = getattr(mod, "bias", None)
    b = b.detach().to("cpu").float() if isinstance(b, torch.Tensor) else None
    return W, b


def _extract_cpu_layer(layer: nn.Module) -> dict:
    """Snapshot a decoder layer's weights + attention shapes into CPU fp32 tensors."""
    attn = layer.self_attn
    rot = attn.rotary_emb
    d = {
        "ln1_w": layer.input_layernorm.weight.detach().to("cpu").float(),
        "ln2_w": layer.post_attention_layernorm.weight.detach().to("cpu").float(),
        "eps": float(getattr(layer.input_layernorm, "variance_epsilon", 1e-5)),
        "num_heads": attn.num_heads,
        "num_kv_heads": attn.num_kv_heads,
        "head_dim": attn.head_dim,
        "q_size": attn.q_size,
        "kv_size": attn.kv_size,
        "scaling": attn.scaling,
        "rot_cache": rot.cos_sin_cache.detach().to("cpu").float(),
        "head_size": rot.head_size,
        "rotary_dim": rot.rotary_dim,
        "is_neox": rot.is_neox_style,
    }
    d["Wqkv"], d["bqkv"] = _lin(attn.qkv_proj)
    d["Wo"], d["bo"] = _lin(attn.o_proj)
    d["Wgu"], d["bgu"] = _lin(layer.mlp.gate_up_proj)
    d["Wd"], d["bd"] = _lin(layer.mlp.down_proj)
    return d


def _make_cpu_layer_forward(d: dict, layer_idx: int, spyre_device):
    """Build a self-contained CPU forward for one fused-residual decoder layer.

    Mirrors vLLM's LlamaDecoderLayer exactly (fused-residual RMSNorm returning
    (normed, new_residual); qkv -> rotary -> causal GQA SDPA -> o_proj; then MLP),
    but on CPU with a per-layer append-list KV cache. Inputs/outputs cross the
    Spyre<->CPU boundary at the layer edges; internal math is fp32.
    """
    import torch.nn.functional as F

    from spyre_inference.custom_ops.utils import convert
    from vllm.model_executor.layers.rotary_embedding.base import RotaryEmbedding

    Hh, Hkv, D = d["num_heads"], d["num_kv_heads"], d["head_dim"]
    rep = Hh // Hkv

    def _rmsnorm(hidden, residual, weight):
        residual = hidden if residual is None else hidden + residual
        x = residual.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        normed = x * torch.rsqrt(var + d["eps"]) * weight
        return normed, residual

    def fwd(positions, hidden_states, residual):
        pos = positions.detach().to("cpu").long().flatten()
        h = hidden_states.detach().to("cpu").float()
        r = None if residual is None else residual.detach().to("cpu").float()

        # A new sequence's prefill starts at absolute position 0 -> reset this
        # layer's KV (wipes any warmup/_dummy_run junk). Chunked prefill and decode
        # (pos[0] != 0) append to the accumulation.
        if pos.numel() and int(pos[0]) == 0:
            _CPU_KV[layer_idx] = {"k": None, "v": None, "pos": None}
        kv = _CPU_KV.setdefault(layer_idx, {"k": None, "v": None, "pos": None})

        normed, r = _rmsnorm(h, r, d["ln1_w"])

        qkv = normed @ d["Wqkv"].t()
        if d["bqkv"] is not None:
            qkv = qkv + d["bqkv"]
        q, k, v = qkv.split([d["q_size"], d["kv_size"], d["kv_size"]], dim=-1)
        q, k = RotaryEmbedding.forward_static(
            pos, q, k, d["head_size"], d["rotary_dim"], d["rot_cache"], d["is_neox"]
        )

        T = q.shape[0]
        q = q.view(T, Hh, D)
        k = k.view(T, Hkv, D)
        v = v.view(T, Hkv, D)

        if kv["k"] is None:
            kv["k"], kv["v"], kv["pos"] = k, v, pos
        else:
            kv["k"] = torch.cat([kv["k"], k], dim=0)
            kv["v"] = torch.cat([kv["v"], v], dim=0)
            kv["pos"] = torch.cat([kv["pos"], pos], dim=0)
        Ke = kv["k"].repeat_interleave(rep, dim=1)  # [Tk, Hh, D]
        Ve = kv["v"].repeat_interleave(rep, dim=1)

        scores = torch.einsum("ihd,jhd->hij", q, Ke) * d["scaling"]  # [Hh, T, Tk]
        mask = kv["pos"].view(1, -1) > pos.view(T, 1)  # key after query -> block
        scores = scores.masked_fill(mask.unsqueeze(0), float("-inf"))
        attn = torch.softmax(scores.float(), dim=-1)
        ctx = torch.einsum("hij,jhd->ihd", attn, Ve).reshape(T, Hh * D)

        attn_out = ctx @ d["Wo"].t()
        if d["bo"] is not None:
            attn_out = attn_out + d["bo"]

        normed2, r2 = _rmsnorm(attn_out, r, d["ln2_w"])

        gu = normed2 @ d["Wgu"].t()
        if d["bgu"] is not None:
            gu = gu + d["bgu"]
        inter = gu.shape[-1] // 2
        act = F.silu(gu[..., :inter]) * gu[..., inter:]
        mlp_out = act @ d["Wd"].t()
        if d["bd"] is not None:
            mlp_out = mlp_out + d["bd"]

        # Return (hidden, residual) on the Spyre device as fp16 — same contract the
        # unmodified layer has, so the next (Spyre) layer continues transparently.
        return (
            convert(mlp_out.to(torch.float16), device=spyre_device),
            convert(r2.to(torch.float16), device=spyre_device),
        )

    return fwd


def _install_cpu_prefix(model: nn.Module, n: int, spyre_device) -> None:
    """Force embed_tokens (+ decoder layers 0..n when n>=0) onto CPU."""
    _CPU_KV.clear()
    _force_embedding_cpu(model, spyre_device)
    if n >= 0:
        layers = _find_decoder_layers(model)
        last = min(n, len(layers) - 1)
        for i in range(last + 1):
            d = _extract_cpu_layer(layers[i])
            layers[i].forward = _make_cpu_layer_forward(d, i, spyre_device)
        print(f"[debug][cpu-layers] forced decoder layers 0..{last} onto CPU")


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

    # Extract the target layer's MLP + RMSNorm weights (values match Spyre's checkpoint)
    # and the rms eps, so op-locality can recompute each op from Spyre's OWN input.
    weights = {}
    try:
        layers = _find_decoder_layers(model)
        layer = layers[TARGET_LAYER["i"]]
        for name in ("gate_proj", "up_proj", "down_proj"):
            w = getattr(getattr(layer.mlp, name, None), "weight", None)
            if isinstance(w, torch.Tensor):
                weights[name] = w.detach().to("cpu").float()
        for name in ("input_layernorm", "post_attention_layernorm"):
            w = getattr(getattr(layer, name, None), "weight", None)
            if isinstance(w, torch.Tensor):
                weights[name] = w.detach().to("cpu").float()
        cfg = getattr(model, "config", None)
        tcfg = getattr(cfg, "text_config", cfg)
        weights["rms_eps"] = float(getattr(tcfg, "rms_norm_eps", 1e-5))
    except Exception as e:  # noqa: BLE001 - best-effort; op-locality test is optional
        print(f"[debug] could not extract layer-{TARGET_LAYER['i']} weights: {e}")
    return store, weights


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
    # Per-layer boundary keys are (int, name). 'mlp_in' (repro input) and
    # ('mlp_int', *) (op-level breakdown) are handled elsewhere — exclude here.
    keys = [
        k for k in spyre
        if isinstance(k, tuple) and k in hf and isinstance(k[0], int) and k[1] != "mlp_in"
    ]
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
              "injected it, and the worst-(token,dim) table.")


def _report_mlp(name: str, a: torch.Tensor, b: torch.Tensor) -> None:
    """Print max/mean/rel + the worst (token, dim) and both values for an MLP sub-op."""
    a, b = _aligned(a, b)
    diff = (a - b).abs()
    flat = int(diff.argmax())
    tok, dim = divmod(flat, diff.shape[1])
    denom = b.abs().max().item() or 1.0
    print(f"{name:>14} {diff.max().item():>12.4f} {diff.mean().item():>12.5f} "
          f"{diff.max().item() / denom:>8.4f}   worst=(t{tok},d{dim}) "
          f"spyre={a[tok, dim].item():.3f} hf={b[tok, dim].item():.3f}")


def compare_mlp_internal(spyre: dict, hf: dict) -> None:
    """Attribute the layer-TARGET MLP divergence to gate_up / swiglu / down.

    vLLM MLP: gate_up_proj (merged [gate,up]) → act_fn (SiluAndMul) → down_proj.
    HF MLP:   gate_proj, up_proj → silu(gate)*up → down_proj.
    """
    import torch.nn.functional as F

    def sg(store, name):
        return store.get(("mlp_int", name))

    s_gate_up, s_act, s_down = sg(spyre, "gate_up_proj"), sg(spyre, "act_fn"), sg(spyre, "down_proj")
    h_gate, h_up = sg(hf, "gate_proj"), sg(hf, "up_proj")
    h_act, h_down = sg(hf, "act_fn"), sg(hf, "down_proj")

    if all(t is None for t in (s_gate_up, s_act, s_down)):
        print(f"\n[mlp-internal] no Spyre MLP children captured on layer {TARGET_LAYER['i']} "
              "(names differ?) — skipping op-level breakdown.")
        return

    print(f"\n--- MLP op-level breakdown, layer {TARGET_LAYER['i']} "
          "(Spyre vs HF, reconciled) ---")
    print(f"{'op':>14} {'max_abs':>12} {'mean_abs':>12} {'rel':>8}   worst-coord + values")

    # gate_up: Spyre merged [gate,up] vs HF cat(gate, up).
    if s_gate_up is not None and h_gate is not None and h_up is not None:
        h_gate_up = torch.cat([h_gate.float(), h_up.float()], dim=-1)
        _report_mlp("gate_up", s_gate_up, h_gate_up)
    # swiglu (down_proj input): Spyre act_fn out vs HF silu(gate)*up.
    if s_act is not None and h_gate is not None and h_up is not None:
        h_swiglu = F.silu(h_gate.float()) * h_up.float()
        _report_mlp("swiglu(act)", s_act, h_swiglu)
    # down_proj output: the massive activation lives here (dim 4000 of hidden).
    if s_down is not None and h_down is not None:
        _report_mlp("down", s_down, h_down)

    print("===============================================================")
    print("Read top-down: the FIRST op whose diff is large is where the overshoot enters "
          "(down large but swiglu small → down_proj; swiglu already large → gate_up/silu).")
    print("Caveat: live MLP inputs differ slightly (Spyre vs HF), so this is a strong hint; "
          "the op-locality test below is the decisive isolation.")


def test_mlp_op_locality(spyre: dict, weights: dict) -> None:
    """DECISIVE op isolation: recompute each MLP op's CPU reference from Spyre's OWN
    captured input (fp32 accumulation, as torch/HF does) using the real weights, and
    compare to Spyre's captured output. A mismatch means THAT op is locally unfaithful
    (isolates it); a match means the op is fine and only propagating a wrong input.
    """
    import torch.nn.functional as F

    def sg(name):
        return spyre.get(("mlp_int", name))

    mlp_in = spyre.get((TARGET_LAYER["i"], "mlp_in"))
    s_gate_up, s_act, s_down = sg("gate_up_proj"), sg("act_fn"), sg("down_proj")
    Wg, Wu, Wd = weights.get("gate_proj"), weights.get("up_proj"), weights.get("down_proj")

    if mlp_in is None or Wg is None or Wu is None or Wd is None:
        print(f"\n[op-locality] missing Spyre mlp_in or HF weights for layer "
              f"{TARGET_LAYER['i']} — skipping op-locality test.")
        return

    print(f"\n--- MLP op-locality, layer {TARGET_LAYER['i']} "
          "(Spyre op output vs CPU-ref from Spyre's OWN input; fp32 accumulate) ---")
    print(f"{'op':>14} {'max_abs':>12} {'mean_abs':>12} {'rel':>8}   worst-coord + values")

    x = mlp_in.float()  # [T, hidden], Spyre's own MLP input
    # gate_up: ref = x @ cat(Wg, Wu).T  ->  [T, 2*inter], order [gate, up].
    if s_gate_up is not None:
        ref_gate_up = x @ torch.cat([Wg, Wu], dim=0).t()
        _report_mlp("gate_up", s_gate_up, ref_gate_up)
    # swiglu: ref = silu(gate)*up computed from Spyre's OWN gate_up (isolates SiluAndMul).
    if s_act is not None and s_gate_up is not None:
        gu = s_gate_up.float()
        inter = gu.shape[-1] // 2
        ref_act = F.silu(gu[..., :inter]) * gu[..., inter:]
        _report_mlp("swiglu(act)", s_act, ref_act)
    # down: ref = (Spyre's OWN swiglu) @ Wd.T  (isolates down_proj) -> inspect dim 4000.
    if s_down is not None and s_act is not None:
        ref_down = s_act.float() @ Wd.t()
        _report_mlp("down", s_down, ref_down)

    print("===============================================================")
    print("A large diff here = that op is LOCALLY wrong on Spyre (bug isolated). Small diff "
          "everywhere = ops are faithful; the divergence is inherited from upstream (RMSNorm "
          "/ residual) — see the RMSNorm op-locality below.")


def test_rmsnorm_op_locality(spyre: dict, weights: dict) -> None:
    """Op-locality for the two RMSNorms at the target layer: recompute the norm from
    Spyre's OWN captured input (added = hidden + residual, fp32) using the real weight
    + eps, and compare to Spyre's captured normed output. Isolates SpyreRMSNorm from
    the value it was fed. (vLLM fuses residual-add into RMSNorm; we replicate it.)
    """
    eps = weights.get("rms_eps", 1e-5)
    i = TARGET_LAYER["i"]
    printed_header = False
    for name in ("input_layernorm", "post_attention_layernorm"):
        h = spyre.get(("norm_in", name, "h"))
        r = spyre.get(("norm_in", name, "r"))
        w = weights.get(name)
        out = spyre.get((i, name))  # Spyre's captured normed output (out[0])
        if h is None or w is None or out is None:
            continue
        added = h.float() if r is None else h.float() + r.float()
        var = added.pow(2).mean(dim=-1, keepdim=True)
        ref = added * torch.rsqrt(var + eps) * w.float()
        if not printed_header:
            print(f"\n--- RMSNorm op-locality, layer {i} "
                  "(Spyre normed vs CPU-ref from Spyre's OWN input; fp32) ---")
            print(f"{'op':>26} {'max_abs':>12} {'mean_abs':>12} {'rel':>8}   worst-coord + values")
            printed_header = True
        _report_mlp(name, out, ref)
    if printed_header:
        print("===============================================================")
        print("Large diff = SpyreRMSNorm is locally wrong (the op we recently changed to "
              "fp32-on-CPU). Small diff = RMSNorm faithful → the wrong value comes from its "
              "input (attention output / incoming residual) — walk back to attention / layer 0.")


def _hf_generate(args) -> None:
    """Coherent reference: greedy HF-CPU generation on the same prompt (fp16)."""
    from transformers import (
        AutoModelForCausalLM,
        AutoModelForImageTextToText,
        AutoTokenizer,
    )

    tok = AutoTokenizer.from_pretrained(args.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float16)
    except ValueError:
        model = AutoModelForImageTextToText.from_pretrained(args.model, dtype=torch.float16)
    model.eval()
    ids = tok(args.prompt, return_tensors="pt").input_ids
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=args.gen_tokens, do_sample=False)
    text = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    print("\n---------------- HF-CPU reference generation ----------------")
    print(f"output: {text!r}")


def run_generate(args) -> None:
    """Full multi-token generation with the CPU prefix forced in, printing the text.

    --cpu-layers -1 forces only embed_tokens to CPU; N>=0 forces embed_tokens +
    decoder layers 0..N. Increment N across manual runs until the text is coherent.
    """
    from vllm import LLM, SamplingParams

    from spyre_inference.v1.worker.spyre_model_runner import TorchSpyreModelRunner

    orig_load_model = TorchSpyreModelRunner.load_model

    def patched_load_model(self, *a, **k):
        orig_load_model(self, *a, **k)
        if args.cpu_layers is not None:
            model = self.get_model()
            spyre_device = next(model.parameters()).device
            _install_cpu_prefix(model, args.cpu_layers, spyre_device)

    TorchSpyreModelRunner.load_model = patched_load_model
    # Disable chunked prefill so the prompt is a single forward (the per-layer CPU
    # KV reset keys off absolute position 0 either way, but this keeps it simple).
    batched = max(args.max_num_batched_tokens, args.max_model_len)
    try:
        llm = LLM(
            model=args.model,
            tokenizer=args.model,
            max_model_len=args.max_model_len,
            max_num_seqs=1,
            tensor_parallel_size=1,
            max_num_batched_tokens=batched,
            dtype="float16",
            enforce_eager=args.enforce_eager,
            num_gpu_blocks_override=args.num_gpu_blocks_override,
        )
        _CPU_KV.clear()  # drop any warmup-pass junk before the real generation
        outputs = llm.generate(
            [args.prompt], SamplingParams(max_tokens=args.gen_tokens, temperature=0.0)
        )
    finally:
        TorchSpyreModelRunner.load_model = orig_load_model

    text = outputs[0].outputs[0].text
    tag = "off" if args.cpu_layers is None else str(args.cpu_layers)
    print(f"\n================ Spyre generation (cpu-layers={tag}) ================")
    print(f"prompt: {args.prompt!r}")
    print(f"output: {text!r}")
    print("=====================================================================")
    if args.compare_with_cpu:
        _hf_generate(args)


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
    p.add_argument(
        "--target-layer", type=int, default=2, dest="target_layer",
        help="Decoder layer to break down at the MLP-op level (default 2, where the "
        "massive-activation overshoot is introduced).",
    )
    p.add_argument(
        "--dump", type=str, default=None,
        help="Optional path to torch.save the captured Spyre+HF tensors (incl. each layer's "
        "'mlp_in') so the op-repro can load the exact real input.",
    )
    p.add_argument(
        "--cpu-layers", type=int, default=None, dest="cpu_layers",
        help="Cumulative-prefix CPU bisection (requires --generate): force embed_tokens + "
        "decoder layers 0..N onto CPU, rest on Spyre. -1 = embedding only. Increment N per "
        "run until the generated text is coherent — that layer holds the bug.",
    )
    p.add_argument(
        "--generate", action="store_true",
        help="Run a real multi-token generation and print the text (coherence check) instead "
        "of the 1-token per-layer capture/diff. Use with --cpu-layers.",
    )
    p.add_argument(
        "--gen-tokens", type=int, default=64, dest="gen_tokens",
        help="Number of tokens to generate in --generate mode (default 64).",
    )
    p.add_argument(
        "--compare-with-cpu", action="store_true", dest="compare_with_cpu",
        help="In --generate mode, also print a greedy HF-CPU generation as a coherent "
        "reference to eyeball against.",
    )
    args = p.parse_args()
    TARGET_LAYER["i"] = args.target_layer

    if args.generate:
        run_generate(args)
        return
    if args.cpu_layers is not None:
        print("[debug] --cpu-layers is only applied in --generate mode; ignoring for capture.")

    print(f"[debug] prompt: {args.prompt!r}")
    print("[debug] === RUN 2: Spyre (current code) ===")
    spyre = run_spyre(args)
    print(f"[debug] captured {sum(isinstance(k, tuple) for k in spyre)} Spyre boundaries")

    print("[debug] === RUN 1: HuggingFace on CPU (reference) ===")
    hf, mlp_weights = run_hf_cpu(args, spyre["input_ids"])
    print(f"[debug] captured {sum(isinstance(k, tuple) for k in hf)} HF boundaries")

    compare(spyre, hf)
    compare_mlp_internal(spyre, hf)
    test_mlp_op_locality(spyre, mlp_weights)
    test_rmsnorm_op_locality(spyre, mlp_weights)

    if args.dump:
        torch.save(
            {"spyre": spyre, "hf": hf, "mlp_weights": mlp_weights,
             "input_ids": spyre.get("input_ids")},
            args.dump,
        )
        print(f"\n[debug] saved captured tensors to {args.dump}")


if __name__ == "__main__":
    main()
