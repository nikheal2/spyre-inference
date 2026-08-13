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

vLLM lowers a patch conv (kernel == stride, no padding) to an im2col + GEMM in
`Conv2dLayer._forward_mulmat`: the on-device `permute(0,2,3,1,4,5).reshape(...)`
becomes a `copy_from_d2d` whose source-stick expression, for patch grids whose
spatial size is coprime with the 64-wide stick, is a sub-stick `Mod(k*var, 32)`
that torch-spyre's restickify pass cannot lay out.

Instead we run the real `F.conv2d` on-card, but place the weight and each input
image into the tiled layouts the Spyre conv kernel expects — the weight sticked
on the out-channel dim (the matmul `n`), the input sticked on the in-channel dim
(the reduction `k`) — via explicit `SpyreTensorLayout`s, then compile. The layout
tuples are derived from tensor shapes (not hardcoded), so any out-channel count
and variable image size are handled.
"""

import torch
import torch.nn.functional as F

from torch._inductor.exc import InductorError

from vllm.logger import init_logger
from vllm.model_executor.layers.conv import Conv2dLayer


logger = init_logger(__name__)


def _weight_layout(weight: torch.Tensor):
    """SpyreTensorLayout for a conv weight (O, C, K1, K2), sticked on out-channels.

    device_size = [K2, K1, O//64, C, 64]; the 64-wide stick walks the out-channel
    dim (host stride C*K1*K2), so out-channels tile into O//64 sticks.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    o, c, k1, k2 = weight.shape
    assert o % 64 == 0, f"conv out_channels {o} must be a multiple of the 64-wide stick"
    return SpyreTensorLayout(
        [k2, k1, o // 64, c, 64],
        [1, k2, c * k1 * k2 * 64, k1 * k2, c * k1 * k2],
        get_device_dtype(weight.dtype),
    )


def _input_layout(x: torch.Tensor):
    """SpyreTensorLayout for a conv input (1, C, H, W), sticked on in-channels.

    device_size = [W, H, 1, 1, 64]; the 64-wide stick walks the channel dim (host
    stride H*W), padding C up to a full stick. Spatial W/H are the outer loops.
    """
    from torch_spyre._C import SpyreTensorLayout, get_device_dtype

    b, c, h, w = x.shape
    assert b == 1, f"conv input batch {b} != 1 (Pixtral feeds one image at a time)"
    assert c <= 64, f"conv in_channels {c} must fit in one 64-wide stick"
    return SpyreTensorLayout(
        [w, h, 1, 1, 64],
        [1, w, -1, c * h * w, h * w],
        get_device_dtype(x.dtype),
    )


@Conv2dLayer.register_oot(name="Conv2dLayer")
class SpyreConv2d(Conv2dLayer):
    """Out-of-tree (OOT) Conv2d for IBM's Spyre device.

    Runs `F.conv2d` on-card with explicit tiled layouts instead of vLLM's
    im2col + GEMM lowering (whose on-device reshape is not stick-layout-able).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._w_dev: torch.Tensor | None = None
        self._path_logged = False
        # Mirror SpyreSiluAndMul: compile the on-card conv unless the outer
        # graph is already being traced (then it captures the eager call).
        if not torch.compiler.is_dynamo_compiling():
            self._conv = torch.compile(self._conv_native, dynamic=False)
        else:
            self._conv = self._conv_native

    def _conv_native(self, x: torch.Tensor, w: torch.Tensor, bias) -> torch.Tensor:
        return F.conv2d(
            x,
            w,
            bias,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )

    def _weight_on_device(self) -> torch.Tensor:
        """Place the conv weight into its tiled layout once, then cache."""
        if self._w_dev is None:
            w_cpu = self.weight.detach().to("cpu")
            self._w_dev = w_cpu.to("spyre", device_layout=_weight_layout(w_cpu))
        return self._w_dev

    def _log_path(self, msg: str, warn: bool = False) -> None:
        if self._path_logged:
            return
        self._path_logged = True
        (logger.warning if warn else logger.info)("Spyre conv2d: %s", msg)

    def _forward_host_im2col(self, x: torch.Tensor) -> torch.Tensor:
        """Layout-safe fallback: im2col + output reshape on host, GEMM on-card.

        The layout-hostile unfold/permute/reshape and the output transpose run on
        CPU (no stick constraints there); only the conv-as-matmul `F.linear` runs
        on Spyre, and tensors cross the H2D entry path (which stickifies arbitrary
        shapes) rather than an on-device restickify. `input_size = C*K1*K2` is a
        multiple of the 64-wide stick, so the GEMM input lands stick-clean.
        """
        dev = x.device
        b, _, h, w = x.shape
        k1, k2 = self.kernel_size
        ho, wo = h // k1, w // k2
        xc = x.to("cpu")
        xc = xc.unfold(2, k1, k1).unfold(3, k2, k2)
        xc = xc.permute(0, 2, 3, 1, 4, 5).reshape(-1, self.input_size)
        gemm = F.linear(
            xc.to(dev),
            self.weight.view(self.out_channels, self.input_size),
            self.bias,
        )
        out = gemm.to("cpu").view(b, ho, wo, self.out_channels).permute(0, 3, 1, 2)
        return out.contiguous().to(dev)

    def forward_oot(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 4
        try:
            # Place the input in its tiled layout via CPU (CPU->spyre is the tested
            # entry path; a device restickify would hit the same bad layout).
            x_cpu = x.to("cpu")
            x_dev = x_cpu.to("spyre", device_layout=_input_layout(x_cpu))
            out = self._conv(x_dev, self._weight_on_device(), self.bias)
            self._log_path("native F.conv2d via SpyreTensorLayout")
            return out
        except InductorError as e:
            # Native conv can't lay out this image (e.g. patch grid coprime with
            # the stick). Fall back to the layout-safe host im2col + on-card GEMM.
            self._log_path(f"host im2col fallback (native conv unsupported: {e})", warn=True)
            return self._forward_host_im2col(x)
