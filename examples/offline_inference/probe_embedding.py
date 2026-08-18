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

"""Isolated probe: is torch-spyre's on-device embedding gather correct for a shape?

Runs ONLY `F.embedding` (aten::embedding — the exact op SpyreVocabParallelEmbedding
uses on-device) on the Spyre device vs CPU, for a set of (vocab, hidden) shapes, and
reports the max difference. No model load, no vLLM.

Hypothesis under test: the gather is wrong when `hidden_size` is not a multiple of
2048 (torch-spyre's 64*32 work-division granularity). Ministral-3-14B has hidden=5120
(= 2.5 * 2048, misaligned) and produced garbage; granite-8B has hidden=4096 (aligned)
and worked. The shapes below separate the hidden-alignment effect from the vocab size.

Run (single Spyre accelerator -> serial only):
    uv run --no-sync python examples/offline_inference/probe_embedding.py
"""

import argparse

import torch
import torch.nn.functional as F

# Register the "spyre" device. Importing the backend is enough; fall back to the
# plugin's custom ops if the backend package name differs.
try:
    import torch_spyre  # noqa: F401
except Exception:  # noqa: BLE001
    try:
        import spyre_inference.custom_ops.rms_norm  # noqa: F401
    except Exception as e:  # noqa: BLE001
        raise SystemExit(
            "Could not register the 'spyre' device (import torch_spyre failed). "
            f"Run this under the same env as torch_spyre_inference.py. Original: {e}"
        )

# (vocab, hidden, expected) — expected is a human note, not enforced.
DEFAULT_SHAPES = [
    (49152, 4096, "granite-like  (hidden aligned)   -> expect MATCH"),
    (131072, 5120, "ministral     (hidden 2.5x2048)  -> expect MISMATCH"),
    (131072, 4096, "big vocab, aligned hidden        -> isolates vocab"),
    (131072, 6144, "aligned hidden (3x2048)          -> expect MATCH"),
    (131072, 8192, "aligned hidden (4x2048)          -> expect MATCH"),
    (131072, 2560, "misaligned hidden (1.25x2048)    -> expect MISMATCH"),
    (131072, 3072, "misaligned hidden (1.5x2048)     -> expect MISMATCH"),
]


def probe_one(vocab: int, hidden: int, num_tokens: int, tol: float) -> dict:
    """Gather `num_tokens` random rows on Spyre vs CPU; return the diff + verdict."""
    weight = torch.randn(vocab, hidden, dtype=torch.float16)
    ids = torch.randint(0, vocab, (num_tokens,), dtype=torch.long)

    ref = F.embedding(ids, weight).float()  # CPU golden

    w_s = weight.to("spyre")
    ids_s = ids.to("spyre")
    out_s = F.embedding(ids_s, w_s).to("cpu").float()  # on-device gather -> D2H

    diff = (out_s - ref).abs()
    max_abs = diff.max().item()
    # Where does the error live along the row? (helps confirm a "trailing chunk" bug)
    per_col = diff.max(dim=0).values  # [hidden]
    first_bad = int((per_col > tol).nonzero()[0]) if (per_col > tol).any() else -1

    del weight, ids, w_s, ids_s, out_s, ref, diff  # free the big weight before next shape
    return {
        "vocab": vocab,
        "hidden": hidden,
        "aligned": hidden % 2048 == 0,
        "max_abs": max_abs,
        "match": max_abs <= tol,
        "first_bad_col": first_bad,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-tokens", type=int, default=8, dest="num_tokens")
    p.add_argument("--tol", type=float, default=1e-2)
    p.add_argument(
        "--shape", action="append", default=None,
        help="Override the default shape set. Repeatable: --shape 131072,5120",
    )
    args = p.parse_args()

    if args.shape:
        shapes = [(*map(int, s.split(",")), "custom") for s in args.shape]
    else:
        shapes = DEFAULT_SHAPES

    torch.manual_seed(0)
    print(f"[probe] on-device F.embedding vs CPU  (num_tokens={args.num_tokens}, tol={args.tol})\n")
    header = f"{'vocab':>8} {'hidden':>7} {'h%2048==0':>9} {'max_abs':>12} {'first_bad_col':>13} {'verdict':>9}"
    print(header)
    print("-" * len(header))

    results = []
    for vocab, hidden, note in shapes:
        r = probe_one(vocab, hidden, args.num_tokens, args.tol)
        results.append(r)
        verdict = "MATCH" if r["match"] else "MISMATCH"
        print(
            f"{r['vocab']:>8} {r['hidden']:>7} {str(r['aligned']):>9} "
            f"{r['max_abs']:>12.5f} {r['first_bad_col']:>13} {verdict:>9}   # {note}"
        )

    print("\n[probe] verdict correlation:")
    by_align = {True: [], False: []}
    for r in results:
        by_align[r["aligned"]].append(r["match"])
    aligned_all_match = all(by_align[True]) if by_align[True] else None
    misaligned_all_fail = (not any(by_align[False])) if by_align[False] else None
    print(f"  hidden multiple-of-2048 shapes: {'all MATCH' if aligned_all_match else 'NOT all match'}")
    print(f"  hidden NOT multiple-of-2048 shapes: "
          f"{'all MISMATCH' if misaligned_all_fail else 'NOT all mismatch'}")
    if aligned_all_match and misaligned_all_fail:
        print("  => CONFIRMED: the on-device gather breaks exactly when hidden % 2048 != 0.")
    else:
        print("  => hidden-alignment does NOT fully explain it; inspect the table above "
              "(vocab or another factor may be involved).")


if __name__ == "__main__":
    main()
