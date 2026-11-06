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

"""Child process that runs the Mistral-format Pixtral vision tower on NNPA or the CPU.

Run as a script by ``spyre_inference.multimodal.nnpa``, never imported: it must not import
``spyre_inference``, vLLM or torch-spyre. torch-spyre and torch_nnpa both rename torch's one
PrivateUse1 backend, so they cannot share a process. Its only dependencies are torch and, for
``nnpa``, torch_nnpa, so it can also run under a separate interpreter (``SPYRE_VISION_PYTHON``).

Protocol over the inherited socket (``argv[1]`` is its fd), each message a ``torch.save`` blob:
  parent -> ``{"config": ..., "weights": {checkpoint name: tensor}}``, reply ``{"ok", ...}``;
  parent -> ``{"images": [[C, H, W], ...]}``, reply ``{"ok", "embeds": [[tokens, dim], ...]}``;
  parent -> ``None`` (or EOF): exit.

The math follows upstream's ``PixtralForConditionalGeneration._process_image_input``, one image
at a time, so attention needs no mask. Patchify, RoPE positions and the merge are reshapes and
permutes done so the device sees only matmuls, norms, SDPA and elementwise ops.
"""

import os
import sys

# Running a file puts its directory first on sys.path, where sibling modules (utils.py, clip.py)
# would shadow top-level imports.
if sys.path and sys.path[0] == os.path.dirname(os.path.abspath(__file__)):
    del sys.path[0]

import io  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from multiprocessing.connection import Connection  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.nn.attention import SDPBackend, sdpa_kernel  # noqa: E402

_EPS = 1e-5


def _log(msg: str) -> None:
    print(f"[spyre-vision-worker pid={os.getpid()}] {msg}", file=sys.stderr, flush=True)


def _send(conn: Connection, obj) -> None:
    buf = io.BytesIO()
    torch.save(obj, buf)
    conn.send_bytes(buf.getbuffer())


def _recv(conn: Connection):
    return torch.load(io.BytesIO(conn.recv_bytes()), weights_only=True)


def _init_nnpa() -> str:
    """Import torch_nnpa and finish its registration; returns its version.

    The parent runs this process with ``TORCH_DEVICE_BACKEND_AUTOLOAD=0`` so ``import torch``
    cannot autoload torch-spyre, which leaves torch_nnpa's PrivateUse1 hooks unregistered after
    a bare import; run its ``torch.backends`` entry point as autoload would.
    """
    from importlib.metadata import entry_points

    import torch_nnpa  # ty: ignore[unresolved-import]

    for ep in entry_points(group="torch.backends"):
        if "nnpa" in (ep.name or "").lower() or (ep.value or "").startswith("torch_nnpa"):
            ep.load()()
    torch.zeros(1, device="nnpa")
    return getattr(torch_nnpa, "__version__", "unknown")


def _rope_tables(head_dim: int, max_side: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    """fp32 ``[max_side**2, head_dim // 2]`` cos and sin, indexed by ``row * max_side + col``.

    Upstream's ``precompute_freqs_cis_2d`` as real tables.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
    side = torch.arange(max_side).float()
    freqs_h = torch.outer(side, freqs[::2])
    freqs_w = torch.outer(side, freqs[1::2])
    angles = torch.cat(
        [
            freqs_h[:, None, :].expand(max_side, max_side, -1),
            freqs_w[None, :, :].expand(max_side, max_side, -1),
        ],
        dim=-1,
    ).reshape(max_side * max_side, -1)
    return angles.cos(), angles.sin()


class PixtralVisionTower:
    def __init__(self, config: dict, weights: dict[str, torch.Tensor]) -> None:
        self.device = torch.device(config["device"])
        self.dtype = getattr(torch, config["dtype"])
        self.out_dtype = getattr(torch, config["out_dtype"])
        self.patch = config["patch_size"]
        self.merge = config["spatial_merge_size"]
        self.max_side = config["image_size"] // self.patch
        self.n_heads = config["num_attention_heads"]
        self.head_dim = config["hidden_size"] // self.n_heads
        self.cos, self.sin = _rope_tables(self.head_dim, self.max_side, config["rope_theta"])

        missing: list[str] = []

        def take(name: str) -> torch.Tensor:
            t = weights.get(name)
            if t is None:
                missing.append(name)
                return torch.empty(0)
            return t.to(device=self.device, dtype=self.dtype)

        enc = "vision_encoder."
        # [D, C, p, p] -> [D, C*p*p], matching the (c, py, px) patch flattening in encode.
        self.patch_w = take(enc + "patch_conv.weight").flatten(1)
        self.ln_pre = take(enc + "ln_pre.weight")
        self.layers = []
        for i in range(config["num_hidden_layers"]):
            p = f"{enc}transformer.layers.{i}."
            self.layers.append(
                {
                    k: take(p + name)
                    for k, name in (
                        ("attention_norm", "attention_norm.weight"),
                        ("wq", "attention.wq.weight"),
                        ("wk", "attention.wk.weight"),
                        ("wv", "attention.wv.weight"),
                        ("wo", "attention.wo.weight"),
                        ("ffn_norm", "ffn_norm.weight"),
                        ("w1", "feed_forward.w1.weight"),
                        ("w2", "feed_forward.w2.weight"),
                        ("w3", "feed_forward.w3.weight"),
                    )
                }
            )
        self.pre_norm = (
            take("pre_mm_projector_norm.weight") if config["pre_mm_projector_norm"] else None
        )
        self.merger = take("patch_merger.merging_layer.weight") if self.merge > 1 else None
        bias = config["adapter_bias"]
        self.w_in = take("vision_language_adapter.w_in.weight")
        self.b_in = take("vision_language_adapter.w_in.bias") if bias else None
        self.w_out = take("vision_language_adapter.w_out.weight")
        self.b_out = take("vision_language_adapter.w_out.bias") if bias else None
        if missing:
            raise KeyError(
                f"Vision weights missing from the Mistral-format checkpoint: {missing[:8]}"
                + (f" (+{len(missing) - 8} more)" if len(missing) > 8 else "")
            )

    def _rms(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + _EPS)
        return xf.to(x.dtype) * weight

    def _rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """``x`` is ``[L, H, head_dim]``; rotates each (even, odd) pair, in fp32."""
        xf = x.float().reshape(*x.shape[:-1], -1, 2)
        re, im = xf[..., 0], xf[..., 1]
        out = torch.stack((re * cos - im * sin, re * sin + im * cos), dim=-1)
        return out.flatten(-2).to(x.dtype)

    def _block(self, x: torch.Tensor, w: dict, cos: torch.Tensor, sin: torch.Tensor):
        length = x.shape[0]
        h = self._rms(x, w["attention_norm"])
        q = F.linear(h, w["wq"]).reshape(length, self.n_heads, self.head_dim)
        k = F.linear(h, w["wk"]).reshape(length, self.n_heads, self.head_dim)
        v = F.linear(h, w["wv"]).reshape(length, self.n_heads, self.head_dim)
        q, k = self._rope(q, cos, sin), self._rope(k, cos, sin)
        # torch_nnpa supports only the math backend.
        with sdpa_kernel(SDPBackend.MATH):
            attn = F.scaled_dot_product_attention(
                q.transpose(0, 1)[None], k.transpose(0, 1)[None], v.transpose(0, 1)[None]
            )
        attn = attn[0].transpose(0, 1).reshape(length, -1)
        x = x + F.linear(attn, w["wo"])
        h = self._rms(x, w["ffn_norm"])
        return x + F.linear(F.silu(F.linear(h, w["w1"])) * F.linear(h, w["w3"]), w["w2"])

    def encode(self, image: torch.Tensor) -> torch.Tensor:
        """One ``[C, H, W]`` image -> ``[tokens, text hidden]`` on the host."""
        c, height, width = image.shape
        p, m = self.patch, self.merge
        hp, wp = height // p, width // p
        assert hp % m == 0 and wp % m == 0, f"{height}x{width} is not a multiple of {p * m}"
        # Patchify and gather the RoPE rows on the host: no conv or index op on the device.
        patches = image.float().reshape(c, hp, p, wp, p).permute(1, 3, 0, 2, 4)
        patches = patches.reshape(hp * wp, c * p * p)
        pos = (torch.arange(hp)[:, None] * self.max_side + torch.arange(wp)[None, :]).flatten()
        cos = self.cos[pos][:, None, :].to(self.device)
        sin = self.sin[pos][:, None, :].to(self.device)

        x = F.linear(patches.to(device=self.device, dtype=self.dtype), self.patch_w)
        x = self._rms(x, self.ln_pre)
        for w in self.layers:
            x = self._block(x, w, cos, sin)
        if self.pre_norm is not None:
            x = self._rms(x, self.pre_norm)
        if self.merger is not None:
            # upstream's unfold: one row per m x m block, row-major, columns in (c, ki, kj) order.
            d = x.shape[-1]
            x = x.reshape(hp // m, m, wp // m, m, d).permute(0, 2, 4, 1, 3)
            x = F.linear(x.reshape(-1, d * m * m), self.merger)
        x = F.linear(F.gelu(F.linear(x, self.w_in, self.b_in)), self.w_out, self.b_out)
        return x.cpu().to(self.out_dtype)


def main() -> None:
    conn = Connection(int(sys.argv[1]))
    try:
        init = _recv(conn)
        config = init["config"]
        threads = config.get("threads") or 0
        if threads > 0:
            torch.set_num_threads(threads)
        # Plain str: the parent loads replies with weights_only, which rejects TorchVersion.
        info: dict[str, object] = {
            "torch": str(torch.__version__),
            "threads": torch.get_num_threads(),
        }
        if config["device"] == "nnpa":
            info["torch_nnpa"] = _init_nnpa()
        t0 = time.time()
        tower = PixtralVisionTower(config, init.pop("weights"))
        del init
        # Fail at startup, not on the first request, if an op does not run on the device.
        side = config["patch_size"] * config["spatial_merge_size"] * 2
        with torch.inference_mode():
            smoke = tower.encode(torch.zeros(config["num_channels"], side, side))
        if not torch.isfinite(smoke).all():
            raise RuntimeError("smoke encode produced non-finite values")
        info["load_s"] = round(time.time() - t0, 2)
        _send(conn, {"ok": True, "info": info})
    except BaseException:
        _send(conn, {"ok": False, "error": traceback.format_exc()})
        return

    while True:
        try:
            msg = _recv(conn)
        except EOFError:
            return
        if msg is None:
            return
        try:
            with torch.inference_mode():
                embeds = [tower.encode(img) for img in msg["images"]]
            _send(conn, {"ok": True, "embeds": embeds})
        except Exception:
            _send(conn, {"ok": False, "error": traceback.format_exc()})


if __name__ == "__main__":
    main()
