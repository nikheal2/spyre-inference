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

"""Run the Mistral-format Pixtral vision tower (Ministral-3) off Spyre: on NNPA or the CPU.

Selected by ``SPYRE_VISION_DEVICE``. torch_nnpa and torch-spyre each rename torch's single
PrivateUse1 backend, so the tower runs in a child process (``nnpa_worker.py``) that never
imports torch-spyre. TP rank 0 owns the child; it reads the full, unsharded vision weights from
the checkpoint (the in-process copy is TP-sharded) and broadcasts each encoder result to the
other ranks. Every rank drops its own tower, so nothing of it reaches the card.
"""

from __future__ import annotations

import atexit
import io
import multiprocessing
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from vllm.config import ModelConfig
from vllm.distributed.parallel_state import get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.model_loader.default_loader import DefaultModelLoader

import spyre_inference.envs as envs
from spyre_inference.models._vision_encoder import unbind_items
from spyre_inference.threading_config import THREADING_ENVS

logger = init_logger(__name__)

_VISION_DEVICES = ("auto", "spyre", "nnpa", "cpu")
_VISION_PREFIXES = (
    "vision_encoder.",
    "patch_merger.",
    "pre_mm_projector_norm.",
    "vision_language_adapter.",
)
_WORKER = Path(__file__).with_name("nnpa_worker.py")


def vision_offload_device() -> str | None:
    """``SPYRE_VISION_DEVICE`` when it moves the tower off Spyre, else ``None``."""
    device = envs.SPYRE_VISION_DEVICE.lower()
    if device not in _VISION_DEVICES:
        raise ValueError(f"SPYRE_VISION_DEVICE must be one of {_VISION_DEVICES}, got {device!r}")
    if device == "auto":
        device = "nnpa" if platform.machine() == "s390x" else "spyre"
    if device == "nnpa" and platform.machine() != "s390x":
        raise ValueError(f"SPYRE_VISION_DEVICE=nnpa needs an s390x host, not {platform.machine()}")
    return None if device == "spyre" else device


def _send(conn, obj) -> None:
    buf = io.BytesIO()
    torch.save(obj, buf)
    conn.send_bytes(buf.getbuffer())


def _read_vision_weights(model_loader: Any, model_config: ModelConfig) -> dict[str, torch.Tensor]:
    """The vision tensors of the Mistral-format checkpoint, as stored (unsharded)."""
    from safetensors import safe_open

    if not isinstance(model_loader, DefaultModelLoader):
        raise ValueError(
            f"SPYRE_VISION_DEVICE needs a safetensors checkpoint; {type(model_loader).__name__} "
            "loads none (dummy weights are not supported)."
        )
    files = model_loader._prepare_weights(
        model_config.model,
        None,
        model_config.revision,
        fall_back_to_pt=False,
        allow_patterns_overrides=["consolidated*.safetensors"],
    )[1]
    weights: dict[str, torch.Tensor] = {}
    for path in sorted(files):
        with safe_open(path, framework="pt") as f:
            for name in f.keys():  # noqa: SIM118 - safe_open has no __iter__
                if name.startswith(_VISION_PREFIXES) and name not in weights:
                    weights[name] = f.get_tensor(name)
    if not weights:
        raise ValueError(f"No vision weights found in {sorted(files)}")
    return weights


class VisionOffload:
    """``embed_multimodal`` for an offloaded tower: one tuple of ``[tokens, H]`` host tensors."""

    def __init__(
        self,
        config: dict[str, Any],
        weights: dict[str, torch.Tensor] | None,
        out_dtype: torch.dtype,
    ) -> None:
        self._out_dtype = out_dtype
        self._tp = get_tp_group()
        self._proc: subprocess.Popen | None = None
        self._conn = None
        if self._tp.rank_in_group == 0:
            assert weights is not None
            self._start(config, weights)

    def _start(self, config: dict[str, Any], weights: dict[str, torch.Tensor]) -> None:
        parent_conn, child_conn = multiprocessing.Pipe()
        env = dict(os.environ)
        # `import torch` would otherwise autoload torch-spyre in the child as well.
        env["TORCH_DEVICE_BACKEND_AUTOLOAD"] = "0"
        if config["threads"] > 0:
            env.update({name: str(config["threads"]) for name in THREADING_ENVS})
        python = envs.SPYRE_VISION_PYTHON or sys.executable
        t0 = time.time()
        self._proc = subprocess.Popen(
            [python, str(_WORKER), str(child_conn.fileno())],
            pass_fds=(child_conn.fileno(),),
            env=env,
        )
        child_conn.close()
        self._conn = parent_conn
        atexit.register(self.close)
        _send(self._conn, {"config": config, "weights": weights})
        reply = self._recv()
        if not reply["ok"]:
            self.close()
            raise RuntimeError(
                f"Vision worker failed to start on {config['device']} "
                f"(SPYRE_VISION_DEVICE=spyre keeps the tower on the card):\n{reply['error']}"
            )
        logger.info(
            "Vision tower running on %s in pid %d (%s), ready in %.1fs.",
            config["device"],
            self._proc.pid,
            reply["info"],
            time.time() - t0,
        )

    def _recv(self) -> dict[str, Any]:
        assert self._conn is not None and self._proc is not None
        while not self._conn.poll(1.0):
            if self._proc.poll() is not None:
                raise RuntimeError(f"Vision worker exited with code {self._proc.returncode}")
        return torch.load(io.BytesIO(self._conn.recv_bytes()), weights_only=True)

    def __call__(self, mm_kwargs: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        reply = None
        if self._conn is not None:
            images = [img.detach().cpu() for img in unbind_items(mm_kwargs.get("images"))]
            if images:
                _send(self._conn, {"images": images})
                reply = self._recv()
            else:
                reply = {"ok": True, "embeds": []}
        # Every rank runs the same encoder step, so each one waits here for rank 0's result.
        reply = self._tp.broadcast_object(reply, src=0)
        if not reply["ok"]:
            raise RuntimeError(f"Vision worker failed:\n{reply['error']}")
        return tuple(t.to(self._out_dtype) for t in reply["embeds"])

    def close(self) -> None:
        if self._proc is None or self._proc.poll() is not None:
            return
        try:
            _send(self._conn, None)
            self._proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 - shutting down
            self._proc.kill()


def offload_vision_tower(
    model: nn.Module,
    model_loader: Any,
    model_config: ModelConfig,
    device: str,
    out_dtype: torch.dtype,
) -> VisionOffload | None:
    """Start the vision worker for ``model`` and drop its tower; call before the move to Spyre.

    ``None`` when ``SPYRE_VISION_DEVICE=auto`` and the model has no tower this can run: other
    vision models keep theirs on the card. An explicit device raises instead.
    """
    from vllm.model_executor.models import pixtral

    from spyre_inference.models.pixtral import SpyrePixtralForConditionalGeneration

    explicit = envs.SPYRE_VISION_DEVICE.lower() != "auto"
    if not isinstance(model, SpyrePixtralForConditionalGeneration):
        if not explicit:
            return None
        raise NotImplementedError(
            f"SPYRE_VISION_DEVICE={device} supports the Mistral-format Pixtral tower "
            f"(Ministral-3) only, not {type(model).__name__}."
        )
    if not isinstance(model.vision_encoder, pixtral.VisionTransformer):
        if not explicit:
            return None
        raise ValueError(f"SPYRE_VISION_DEVICE={device} needs the vision tower; image limit is 0.")
    args = model.vision_args
    config = {
        "device": device,
        "dtype": envs.SPYRE_VISION_DTYPE,
        "out_dtype": str(out_dtype).removeprefix("torch."),
        "threads": envs.SPYRE_VISION_THREADS,
        "hidden_size": args.hidden_size,
        "num_attention_heads": args.num_attention_heads,
        "num_hidden_layers": args.num_hidden_layers,
        "num_channels": args.num_channels,
        "patch_size": args.patch_size,
        "image_size": args.image_size,
        "rope_theta": args.rope_theta,
        "spatial_merge_size": args.spatial_merge_size if model.patch_merger is not None else 1,
        "pre_mm_projector_norm": model.pre_mm_projector_norm is not None,
        "adapter_bias": args.adapter_bias,
    }
    is_owner = get_tp_group().rank_in_group == 0
    weights = _read_vision_weights(model_loader, model_config) if is_owner else None
    offload = VisionOffload(config, weights, out_dtype)
    del weights

    model.vision_encoder = None  # ty: ignore[invalid-assignment]
    model.patch_merger = None
    model.pre_mm_projector_norm = None
    model.vision_language_adapter = None  # ty: ignore[invalid-assignment]
    return offload
