#!/usr/bin/env python3
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

"""Spike for #1473: one Pixtral vision block compiled once over a declared patch-count range.

Runs ``pixtral_block_dynamic`` on one set of device buffers declared at the maximum, resized
to each image's padded patch count, and checks for every image:

- that one graph serves every size (no recompile),
- the real patches against an fp32 CPU reference of today's block,
- first-call and steady-state time, and with ``--static-baseline`` the same for today's
  static block compiled at exactly that size (a lower bound on a budget's cost).

Images are patch grids ``HPxWP`` with even sides, laid out offset-major with each of the
four merge groups padded at its own tail, as ``SpyrePixtralForConditionalGeneration`` does.
Defaults are Ministral-3's vision tower.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import warnings
from functools import partial

os.environ.setdefault("VLLM_PLUGINS", "spyre_inference")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

MERGE_GROUPS = 4  # spatial_merge_size 2


def _weights(hidden, heads, inter, head_dim, device, dtype, seed):
    from spyre_inference.models._vision_dynamic import PixtralBlockWeights
    from spyre_inference.multimodal.utils import rope_perm_matrix

    gen = torch.Generator().manual_seed(seed)

    def proj(rows, cols):
        return torch.randn(rows, cols, generator=gen) * 0.02

    def norm():
        return 1.0 + torch.randn(hidden, generator=gen) * 0.05

    width = heads * head_dim
    host = PixtralBlockWeights(
        attention_norm=norm(),
        wq=proj(hidden, width),
        wk=proj(hidden, width),
        wv=proj(hidden, width),
        wo=proj(width, hidden),
        ffn_norm=norm(),
        w_gate=proj(hidden, inter),
        w_up=proj(hidden, inter),
        w_down=proj(inter, hidden),
        rope_perm=rope_perm_matrix("pair", head_dim, torch.device("cpu"), torch.float32),
    )
    return PixtralBlockWeights(*(t.to(device=device, dtype=dtype) for t in host))


def reference_block(x, key_pad, cos, sin, w, *, num_heads, eps=1e-5):
    """Today's ``SpyrePixtralTransformerBlock`` as plain ops on ``[P, D]``."""
    from spyre_inference.models._vision_dynamic import rms_norm
    from spyre_inference.multimodal.utils import padded_sdpa, rope_rotate_matmul

    rows = x.shape[0]
    head_dim = cos.shape[-1]

    def heads(t):
        return t.reshape(1, rows, num_heads, head_dim).transpose(1, 2)

    h = rms_norm(x, w.attention_norm, eps, upcast=True)
    q = rope_rotate_matmul(heads(h @ w.wq), cos, sin, w.rope_perm)
    k = rope_rotate_matmul(heads(h @ w.wk), cos, sin, w.rope_perm)
    attn = padded_sdpa(q, k, heads(h @ w.wv), key_pad.to(q.dtype)[None, None, None, :])
    h = x + attn.transpose(1, 2).reshape(rows, -1) @ w.wo
    n = rms_norm(h, w.ffn_norm, eps, upcast=True)
    return h + (F.silu(n @ w.w_gate) * (n @ w.w_up)) @ w.w_down


def _host_inputs(hp, wp, rows, max_rows, hidden, tables, seed):
    """``[max_rows, ...]`` host buffers for one image, padded to ``rows`` slots, then to the max."""
    from spyre_inference.models._vision_encoder import MASK_MIN, offset_major

    cos_table, sin_table, max_side = tables
    real = hp * wp // MERGE_GROUPS
    group = rows // MERGE_GROUPS
    grid = torch.arange(hp)[:, None] * max_side + torch.arange(wp)[None, :]
    positions = torch.zeros(MERGE_GROUPS, group, dtype=torch.int64)
    positions[:, :real] = offset_major(grid, 2)
    key_pad = torch.zeros(MERGE_GROUPS, group)
    key_pad[:, real:] = MASK_MIN

    def to_max(t):
        return F.pad(t, (0, 0) * (t.dim() - 1) + (0, max_rows - t.shape[0]))

    positions = positions.flatten()
    gen = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(max_rows, hidden, generator=gen),
        "key_pad": to_max(key_pad.flatten()[:, None]),
        "cos": to_max(cos_table[positions]),
        "sin": to_max(sin_table[positions]),
    }


def _real_rows(out, rows, real_patches):
    return out.reshape(MERGE_GROUPS, rows // MERGE_GROUPS, -1)[
        :, : real_patches // MERGE_GROUPS
    ].flatten(0, 1)


def _timed(fn, iters):
    torch.spyre.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.spyre.synchronize()
    first = time.perf_counter() - start
    samples = []
    for _ in range(iters):
        start = time.perf_counter()
        fn()
        torch.spyre.synchronize()
        samples.append((time.perf_counter() - start) * 1e3)
    return out, first, statistics.median(samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--images", default="16x16,32x48,64x64,110x110")
    parser.add_argument("--max-tokens", type=int, default=3040, help="largest budget today")
    parser.add_argument("--tile", type=int, default=256, help="rows per trip, the granularity")
    parser.add_argument("--hidden", type=int, default=1024)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--intermediate", type=int, default=4096)
    parser.add_argument("--max-side", type=int, default=110, help="image_size // patch_size")
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument(
        "--no-upcast", action="store_true", help="fp16 RMSNorm, to sidestep torch-spyre#6453"
    )
    parser.add_argument("--static-baseline", action="store_true")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--json", help="write the rows here as well")
    args = parser.parse_args()

    from torch._dynamo.utils import counters
    from torch_spyre.ops.fallbacks import FallbackWarning

    from spyre_inference.models._vision_dynamic import pixtral_block_dynamic
    from spyre_inference.models.pixtral import pixtral_rope_tables
    from spyre_inference.multimodal.utils import align_up

    warnings.simplefilter("error", FallbackWarning)
    device = torch.device("spyre")
    head_dim = args.hidden // args.heads
    tile = args.tile
    if tile % 64 or tile % MERGE_GROUPS:
        parser.error("--tile must be a multiple of the 64-element stick")
    max_rows = align_up(args.max_tokens * MERGE_GROUPS, tile)
    if max_rows // tile > 256:
        parser.error(f"{max_rows // tile} tiles exceed torch-spyre's MAX_BUCKETS of 256")
    spec = {"min": tile, "max": max_rows, "granularity": tile}

    cos_table, sin_table = pixtral_rope_tables(head_dim, args.max_side, args.rope_theta)
    tables = (cos_table, sin_table, args.max_side)
    w_cpu = _weights(args.hidden, args.heads, args.intermediate, head_dim, "cpu", torch.float32, 0)
    w_dev = _weights(args.hidden, args.heads, args.intermediate, head_dim, device, torch.float16, 0)

    block = partial(
        pixtral_block_dynamic,
        num_heads=args.heads,
        tile=tile,
        max_rows=max_rows,
        upcast=not args.no_upcast,
    )
    compiled = torch.compile(block, fullgraph=True)
    static = torch.compile(partial(reference_block, num_heads=args.heads), fullgraph=True)

    results = []
    for i, image in enumerate(args.images.split(",")):
        hp, wp = (int(s) for s in image.split("x"))
        if hp % 2 or wp % 2 or hp > args.max_side or wp > args.max_side:
            parser.error(f"{image}: sides must be even and at most {args.max_side}")
        patches = hp * wp
        rows = align_up(patches, tile)
        host = _host_inputs(hp, wp, rows, max_rows, args.hidden, tables, seed=i + 1)

        # Pad on the host, copy whole, then resize: a partial copy into a declared buffer is
        # slow (torch-spyre#6026).
        dev = {}
        for key, value in host.items():
            dev[key] = value.half().to(device, dynamic={0: spec})
            dev[key].resize_((rows, *value.shape[1:]))

        graphs = counters["stats"]["unique_graphs"]
        out, first, step = _timed(
            partial(compiled, dev["x"], dev["key_pad"], dev["cos"], dev["sin"], w_dev), args.iters
        )
        row = {
            "image": image,
            "patches": patches,
            "rows": rows,
            "trips": rows // tile,
            "new_graphs": counters["stats"]["unique_graphs"] - graphs,
            "first_call_s": round(first, 2),
            "step_ms": round(step, 2),
        }

        x, key_pad, cos, sin = (host[k][:rows] for k in ("x", "key_pad", "cos", "sin"))
        key_pad = key_pad[:, 0]
        want = reference_block(x, key_pad, cos, sin, w_cpu, num_heads=args.heads)
        got = _real_rows(out.cpu().float(), rows, patches)
        want = _real_rows(want, rows, patches)
        row["cosine"] = round(F.cosine_similarity(got.flatten(), want.flatten(), dim=0).item(), 6)
        row["max_abs"] = round((got - want).abs().max().item(), 4)

        if args.static_baseline:
            static_in = [t.half().to(device) for t in (x, key_pad, cos, sin)]
            _, s_first, s_step = _timed(partial(static, *static_in, w=w_dev), args.iters)
            row["static_first_call_s"] = round(s_first, 2)
            row["static_step_ms"] = round(s_step, 2)

        print(json.dumps(row), flush=True)
        results.append(row)

    dynamic_graphs = sum(r["new_graphs"] for r in results)
    print(f"dynamic block: {dynamic_graphs} graph(s) for {len(results)} sizes (want 1)")
    if args.json:
        with open(args.json, "w") as f:
            json.dump({"args": vars(args), "rows": results}, f, indent=2)


if __name__ == "__main__":
    main()
