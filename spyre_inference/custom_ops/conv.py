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

"""Spyre-specific Conv2d implementation (Pixtral/Ministral vision patch embed).

torch-spyre's direct conv2d lowering runs a conv natively on the card, but only
for a 2- or 3-tap kernel over a whole number of 64-wide channel sticks. A patch
embed (3 channels, 14x14 kernel) misses both, so it decomposes to
`spyre::unfold`, which round-trips the image through the CPU.

A patch conv (kernel == stride, no padding) is rewritten into one the direct path
accepts: space-to-depth the image by ``b = k / 2`` (or ``k / 3``), so every
``b x b`` pixel block becomes one pixel with ``C*b*b`` channels, zero-padded to a
whole stick. The ``k``-tap conv over the image is then exactly a 2- (or 3-) tap,
stride-2 (or 3) conv over the blocks, with the weight repacked the same way once
at load. Tensors go channel-last so the channels land on the stick.
"""

import torch
import torch.nn.functional as F
from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv2dLayer

from .lazy_compile import CompileOutermost, maybe_compile
from .utils import convert

logger = init_logger(__name__)

_STICK = 64  # fp16 elements per 128-byte stick
_DIRECT_KERNELS = (2, 3)  # kernel taps torch-spyre's direct conv2d lowering accepts

Plan = tuple[tuple[int, int], tuple[int, int]]  # (repacked kernel, space-to-depth block)


def _repack_plan(layer: Conv2dLayer) -> Plan | None:
    """The repacked kernel and block size per axis, or None if `layer` cannot be
    rewritten (not a patch conv, or a kernel with no 2 or 3 factor).

    This class is registered OOT for *every* `Conv2dLayer`, so anything else keeps
    the stock path.
    """
    if not layer.enable_linear or any(d != 1 for d in layer.dilation):
        return None
    kernel = []
    for k in layer.kernel_size:
        tap = next((t for t in _DIRECT_KERNELS if k % t == 0), None)
        if tap is None:
            return None
        kernel.append(tap)
    kh, kw = kernel
    return (kh, kw), (layer.kernel_size[0] // kh, layer.kernel_size[1] // kw)


def _packed_channels(in_channels: int, plan: Plan) -> int:
    (_, _), (bh, bw) = plan
    return -(-in_channels * bh * bw // _STICK) * _STICK


def _pack_input(x: torch.Tensor, plan: Plan, channels: int) -> torch.Tensor:
    """[N, C, H, W] -> channel-last [N, H', W', channels] of ``b x b`` pixel blocks.

    Crops the border a stride-k conv never reads, so the repacked conv's windows
    cover the width exactly (the direct path rejects a ragged width).
    """
    n, c, h, w = x.shape
    (kh, kw), (bh, bw) = plan
    rows, cols = h // (kh * bh) * kh, w // (kw * bw) * kw
    x = x[:, :, : rows * bh, : cols * bw].reshape(n, c, rows, bh, cols, bw)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(n, rows, cols, c * bh * bw)
    return F.pad(x, (0, channels - c * bh * bw)).contiguous()


def _pack_weight(weight: torch.Tensor, plan: Plan, channels: int) -> torch.Tensor:
    """[O, C, K1, K2] -> [channels, kh, kw, O], matching `_pack_input`'s channel
    order (c, row-in-block, col-in-block) and sticked on out-channels."""
    o, c = weight.shape[:2]
    (kh, kw), (bh, bw) = plan
    w = weight.reshape(o, c, kh, bh, kw, bw).permute(0, 1, 3, 5, 2, 4)
    w = F.pad(w.reshape(o, c * bh * bw, kh, kw), (0, 0, 0, 0, 0, channels - c * bh * bw))
    return w.permute(1, 2, 3, 0).contiguous()


@Conv2dLayer.register_oot(name="Conv2dLayer")
class SpyreConv2d(CompileOutermost, Conv2dLayer):
    """Out-of-tree Conv2d for Spyre: a patch conv repacked onto torch-spyre's direct
    conv2d lowering.

    Spyre needs static shapes, so the kernel recompiles per distinct (H, W). Past
    ``torch._dynamo.config.cache_size_limit`` (default 8) dynamo falls back to
    eager, which has no direct conv — bucket or resize images if a workload uses
    many resolutions.
    """

    # Per-(H, W) recompiles are this layer's contract, so the compile guard must not
    # report them; warmup cannot enumerate every image resolution.
    allow_inference_recompiles = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._plan = _repack_plan(self)
        self._channels = _packed_channels(self.in_channels, self._plan) if self._plan else 0
        self._w_dev: torch.Tensor | None = None

    # force: the direct conv exists only as a compiled lowering, even under enforce_eager.
    @maybe_compile(force=True)
    def _conv_direct(self, x: torch.Tensor, w: torch.Tensor, bias) -> torch.Tensor:
        assert self._plan is not None
        out = F.conv2d(x.permute(0, 3, 1, 2), w.permute(3, 0, 1, 2), bias, stride=self._plan[0])
        return out.permute(0, 2, 3, 1)

    def process_weights_after_loading(self) -> None:
        """Repack the patch-conv weight for the direct conv once after model load."""
        if self._w_dev is not None or self._plan is None or self.weight.device.type != "spyre":
            return
        w_cpu = convert(self.weight.detach(), device="cpu")
        self._w_dev = convert(
            _pack_weight(w_cpu, self._plan, self._channels), device=self.weight.device
        )

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        # `_forward_conv`, not `forward_native`: a patch embed sets `enable_linear`, so
        # `forward_native` picks an unfold/reshape path Spyre cannot lower.
        if x.device.type != "spyre":
            return self._forward_conv(x)
        if self._plan is None or x.dtype != torch.float16:
            logger.warning_once(
                "Spyre conv2d: %s (kernel %s, stride %s, dtype %s) is not a patch conv "
                "the direct conv2d lowering can take; falling back to F.conv2d.",
                tuple(x.shape),
                self.kernel_size,
                self.stride,
                x.dtype,
            )
            return self._forward_conv(x)
        logger.info_once("Spyre conv2d: space-to-depth onto the direct conv2d lowering")
        assert self._w_dev is not None, "Conv weights must be prepared after model loading."
        from torch_spyre._inductor import config as spyre_config

        x_packed = _pack_input(convert(x, device="cpu"), self._plan, self._channels)
        # Scoped so no other conv in the process changes path.
        with spyre_config.patch(conv2d_direct_lowering=True):
            out = self._conv_direct(convert(x_packed, device=x.device), self._w_dev, self.bias)
        # NCHW view of the channel-last result; Pixtral's flatten(2).permute(0, 2, 1)
        # turns it back into a contiguous [N, H*W, O] without a copy.
        return out.permute(0, 3, 1, 2)
