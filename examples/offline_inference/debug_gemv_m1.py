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

"""Minimal repro: is a compiled Spyre matmul wrong when the token dim is 1?

Motivation: in compile mode the Ministral decoder produces coherent text on every
step where the running batch is >= 2 tokens and garbage on every step where it is
exactly 1 token (single-sequence decode). Prefill (chunked at 2 tokens) is fine,
batch-2 decode is fine, eager is fine at every size. That isolates the token
dimension being 1 under torch.compile.

This script strips vLLM away entirely and compares, for m in {1, 2, 4, 32}:

    cpu     : x @ W on CPU (fp16 reference)
    eager   : x @ W on Spyre, eager
    compiled: x @ W on Spyre, under torch.compile(dynamic=False)

`x @ W` is the exact shape the decoder linears take: `[m, hidden] @ [hidden, out]`
with the weight already physically transposed (see custom_ops/linear.py). A large
error at m == 1 for `compiled` while `eager` is clean confirms the miscompile and
gives torch-spyre a self-contained reproducer.

Run on the Spyre box:
    python examples/offline_inference/debug_gemv_m1.py
    python examples/offline_inference/debug_gemv_m1.py --hidden 5120 --out 5120

Delete this file once the underlying issue is fixed.
"""

import argparse

import torch


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--hidden", type=int, default=5120, help="input features (Ministral-14B)")
    parser.add_argument("--out", type=int, default=5120, help="output features")
    parser.add_argument(
        "--sizes",
        type=str,
        default="1,2,4,32",
        help="comma-separated token counts (leading matmul dim) to test",
    )
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def _mm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.matmul(x, w)


def main():
    args = parse_args()
    torch.manual_seed(args.seed)

    # Import for the side effect of registering the "spyre" device.
    import torch_spyre  # noqa: F401

    dtype = torch.float16
    # Small std keeps fp16 accumulation well inside range, so any large error is
    # structural (wrong data / wrong layout), not accumulation noise.
    w_cpu = torch.empty(args.hidden, args.out, dtype=dtype).normal_(std=0.02)
    w_dev = w_cpu.to("spyre")

    compiled = torch.compile(_mm, dynamic=False)

    print(f"hidden={args.hidden} out={args.out} dtype={dtype}")
    print(f"{'m':>6} {'eager absmax':>14} {'compiled absmax':>17}  verdict")
    print("-" * 60)

    for m in (int(s) for s in args.sizes.split(",")):
        x_cpu = torch.empty(m, args.hidden, dtype=dtype).normal_(std=0.02)
        x_dev = x_cpu.to("spyre")

        ref = _mm(x_cpu, w_cpu)
        eager = _mm(x_dev, w_dev).to("cpu")
        # dynamic=False means each m compiles its own graph, which is exactly how
        # the decoder is compiled (see _compile_for_spyre).
        comp = compiled(x_dev, w_dev).to("cpu")

        eager_err = (eager.float() - ref.float()).abs().amax().item()
        comp_err = (comp.float() - ref.float()).abs().amax().item()
        # fp16 matmul over `hidden` terms at std 0.02 lands well under 1e-2; an
        # order of magnitude above that is a structural failure, not rounding.
        tol = 1e-2
        verdict = "ok" if comp_err < tol else "COMPILED MISMATCH"
        if eager_err >= tol:
            verdict += " (eager also wrong)"
        print(f"{m:>6} {eager_err:>14.6f} {comp_err:>17.6f}  {verdict}")


if __name__ == "__main__":
    main()
