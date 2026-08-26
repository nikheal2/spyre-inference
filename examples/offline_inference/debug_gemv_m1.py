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

"""DIAGNOSTIC (temporary): which compiled op is wrong on Spyre when the token dim is 1?

This whole file is scaffolding for the compile-mode corruption investigation and
should be deleted once that bug is fixed. See DIAGNOSTICS.md.


Motivation: in compile mode the Ministral decoder is coherent on every step whose
token count is >= 2 and garbage on every step whose token count is exactly 1.
Running with `--max-num-seqs 1 --max-num-batched-tokens 1` makes *every* step 1
token, so only the m=1 graph is ever compiled -- and it is wrong end to end.
`-n 2` compiles only m=2 and is correct. Eager is correct at every size.

A bare `x @ W` at m=1 already tested clean, so the fault is another op in the
graph. This script compiles each candidate op chain on its own, with
`dynamic=False` (matching _compile_for_spyre), and compares Spyre-compiled vs
Spyre-eager vs a CPU fp16 reference across token counts.

The case that shows a large error only at m=1, only in the `compiled` column, is
the culprit -- and is a self-contained reproducer to file against torch-spyre.

Run on the Spyre box:
    python examples/offline_inference/debug_gemv_m1.py
    python examples/offline_inference/debug_gemv_m1.py --case mlp --sizes 1,2

Delete this file once the underlying issue is fixed.
"""

import argparse

import torch
import torch.nn.functional as F

# fp16 matmuls over `hidden` terms at std 0.02 land well under this; an order of
# magnitude above it is a structural failure (wrong data/layout), not rounding.
TOL = 1e-2


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hidden", type=int, default=5120, help="hidden size (Ministral-14B)")
    parser.add_argument("--inter", type=int, default=8192, help="MLP intermediate size")
    parser.add_argument("--heads", type=int, default=40, help="attention heads")
    parser.add_argument("--head-dim", type=int, default=128, dest="head_dim")
    parser.add_argument(
        "--sizes", type=str, default="1,2,4", help="comma-separated token counts to test"
    )
    parser.add_argument("--case", type=str, default="all", help="single case name, or 'all'")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def build_cases(args, dtype):
    """Return {name: (fn, [cpu tensors])}; fn(*tensors) is compiled per token count.

    Each case isolates one candidate. `x` is always the leading [m, hidden]
    activation, so every case is exercised at the same token counts.
    """
    hidden, inter = args.hidden, args.inter
    heads, head_dim = args.heads, args.head_dim

    def randn(*shape):
        return torch.empty(*shape, dtype=dtype).normal_(std=0.02)

    w_gate_up = randn(hidden, 2 * inter)
    w_down = randn(inter, hidden)
    w_qkv = randn(hidden, heads * head_dim)
    w_sq = randn(hidden, hidden)

    def mm(x, w):
        return torch.matmul(x, w)

    def residual(x, w):
        # Residual stream: add feeding the next op, the pattern most likely to be
        # broken by a bad broadcast when the leading dim is 1.
        return x + torch.matmul(x, w)

    def silu_mul(x, w):
        # SiluAndMul: chunk on the last dim then multiply -- a strided view pair.
        y = torch.matmul(x, w)
        gate, up = y.chunk(2, dim=-1)
        return F.silu(gate) * up

    def mlp(x, w_up, w_dn):
        y = torch.matmul(x, w_up)
        gate, up = y.chunk(2, dim=-1)
        return torch.matmul(F.silu(gate) * up, w_dn)

    def head_reshape(x, w):
        # QKV reshape/transpose into head layout: with m == 1 the [1, heads, dim]
        # view is degenerate and a squeeze-like lowering would go unnoticed.
        y = torch.matmul(x, w)
        y = y.view(-1, heads, head_dim).transpose(0, 1).contiguous()
        return y.transpose(0, 1).reshape(-1, heads * head_dim)

    def chained(x, w):
        # Two dependent matmuls with an add between: catches a stale/reordered
        # intermediate buffer that a single matmul cannot expose.
        a = torch.matmul(x, w)
        b = torch.matmul(a + x, w)
        return a + b

    cases = {
        "mm": (mm, [w_sq]),
        "residual": (residual, [w_sq]),
        "silu_mul": (silu_mul, [w_gate_up]),
        "mlp": (mlp, [w_gate_up, w_down]),
        "head_reshape": (head_reshape, [w_qkv]),
        "chained": (chained, [w_sq]),
    }
    if args.case != "all":
        cases = {args.case: cases[args.case]}
    return cases


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Import for the side effect of registering the "spyre" device.
    import torch_spyre  # noqa: F401

    dtype = torch.float16
    sizes = [int(s) for s in args.sizes.split(",")]
    cases = build_cases(args, dtype)

    print(f"hidden={args.hidden} inter={args.inter} heads={args.heads} dtype={dtype}")
    print(f"{'case':>14} {'m':>4} {'eager absmax':>14} {'compiled absmax':>17}  verdict")
    print("-" * 72)

    for name, (fn, weights_cpu) in cases.items():
        weights_dev = [w.to("spyre") for w in weights_cpu]
        for m in sizes:
            x_cpu = torch.empty(m, args.hidden, dtype=dtype).normal_(std=0.02)
            x_dev = x_cpu.to("spyre")

            # dynamic=False compiles a separate graph per m, exactly as the
            # decoder is compiled; build it fresh so no shape is reused.
            compiled = torch.compile(fn, dynamic=False)

            ref = fn(x_cpu, *weights_cpu).float()
            eager_err = (fn(x_dev, *weights_dev).to("cpu").float() - ref).abs().amax().item()
            comp_err = (compiled(x_dev, *weights_dev).to("cpu").float() - ref).abs().amax().item()

            verdict = "ok" if comp_err < TOL else "COMPILED MISMATCH"
            if eager_err >= TOL:
                verdict += " (eager also wrong)"
            print(f"{name:>14} {m:>4} {eager_err:>14.6f} {comp_err:>17.6f}  {verdict}")


if __name__ == "__main__":
    main()
