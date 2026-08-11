# Copyright 2026 The Spyre-Inference Authors. Apache-2.0.
"""Standalone repro: does our Spyre 2x2 RoPE realization match the reference
neox rotation for Ministral-3's YaRN config?

Isolates the DEVICE realization (`_rotate_neox_2x2`, the code we wrote) from the
question of whether vLLM's YaRN `cos_sin_cache` is itself correct — both sides
use the *same* `rope.cos_sin_cache`, so a large diff here means our on-device
rotation is wrong (the bug), while ~0 (fp16 noise) means rope is faithful and we
look elsewhere (attention).

Run on the Spyre host:
    uv run --no-sync python \
      .claude/skills/debug-spyre/logs/yarn-rope-gibberish/repro_rope.py
"""

import torch
import torch_spyre  # noqa: F401  (registers the "spyre" device)

from vllm.config import VllmConfig, set_current_vllm_config

from spyre_inference.custom_ops import register_all
from spyre_inference.custom_ops.rotary_embedding import (
    SpyreYaRNScalingRotaryEmbedding,
    _rotate_neox_2x2,
)

# Register the plugin's custom ops (spyre_convert used by convert(), the rope
# opaque ops, etc.). Safe to call repeatedly (lru_cached).
register_all()

torch.manual_seed(0)

HEAD = 128           # head_dim
NH = 32              # q heads
T = 8               # tokens (short prompt)
ORIG_MAX = 16384    # original_max_position_embeddings
BASE = 1.0e9        # rope_theta
FACTOR = 16.0       # yarn scaling factor
dev = torch.device("spyre")

# Build the exact OOT rope the model uses (mixin + YaRN). Positional args match
# YaRNScalingRotaryEmbedding.__init__(head_size, rotary_dim,
#   max_position_embeddings, base, is_neox_style, scaling_factor, dtype, ...).
# RotaryEmbeddingBase is a vLLM CustomOp, so __init__ needs an active vLLM config.
with set_current_vllm_config(VllmConfig()):
    rope = SpyreYaRNScalingRotaryEmbedding(
        HEAD, HEAD, ORIG_MAX, BASE, True, FACTOR, torch.float16,
        beta_fast=32, beta_slow=1,
    )

positions = torch.arange(T)
q = torch.randn(T, NH * HEAD, dtype=torch.float16)

# ---- Reference: neox rotation on CPU using the SAME cos_sin_cache ----
cos_sin = rope.cos_sin_cache[positions]            # [T, HEAD]
cos, sin = cos_sin.chunk(2, dim=-1)                # [T, HEAD/2] each
cos_f = torch.cat([cos, cos], dim=-1).unsqueeze(1).float()  # [T,1,HEAD]
sin_f = torch.cat([sin, sin], dim=-1).unsqueeze(1).float()
qh = q.view(T, NH, HEAD).float()


def rotate_half(x):
    x1, x2 = x[..., : HEAD // 2], x[..., HEAD // 2:]
    return torch.cat([-x2, x1], dim=-1)


q_ref = (qh * cos_f + rotate_half(qh) * sin_f).reshape(T, -1)

# ---- Our Spyre 2x2 realization on device ----
rot = rope.gather_rotation(positions.to(dev), dev)  # [T,2,2,padded] on spyre
q_dev = _rotate_neox_2x2(q.to(dev), rot, HEAD).to("cpu").float()

diff = (q_ref - q_dev).abs()
print("=== Spyre 2x2 RoPE vs reference neox (same cos_sin_cache) ===")
print(f"max  abs diff: {diff.max().item():.6f}")
print(f"mean abs diff: {diff.mean().item():.6f}")
print(f"ref[0,:6]: {[round(v, 4) for v in q_ref[0, :6].tolist()]}")
print(f"dev[0,:6]: {[round(v, 4) for v in q_dev[0, :6].tolist()]}")
print()
print("Interpretation: max diff >> fp16 noise (say > 0.05) => our on-device")
print("rotation is the bug. max diff ~0 => rope is faithful; look at attention.")
