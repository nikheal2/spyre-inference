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

"""Spyre-specific model runner for vLLM v1.

Inherits from GPUModelRunner to preserve the CpuGpuBuffer
dual-buffer pattern where .cpu = CPU staging and .gpu = Spyre device tensors.

Data flow in the current WIP version:
- self.device = CPU. Buffers and scatter ops stay on CPU.
- _SpyreModelWrapper converts input_ids/positions to Spyre int64 at the
  model call boundary.
- _SpyreModelWrapper converts final hidden_states to CPU for downstream
  operations (logits indexing, lm_head, sampling).
- Embedding: Spyre int64 input → Spyre compute → float16 output on Spyre.
- Hidden states flow on Spyre between decoder layers.
- There are few exceptions where a CPU fallback is currently needed:
  - Attention block: Spyre input → CPU (and partial Spyre) compute → Spyre output.
  - Layers that are not yet wrapped for torch-spyre,
    for example RotaryEmbedding

As the TorchSpyreModelRunner is evolving, more layers will natively support inputs
arriving as a Spyre tensor and perform their operations on Spyre.
Thus, in the final state of the runner minimal D2H and H2D transfers will be necessary,
the CPU fallbacks will be obsolete and most operations will be performed on Spyre.
"""

from __future__ import annotations

import os
import time
import types
from contextlib import contextmanager
from functools import lru_cache

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map

import numpy as np

from vllm.config import VllmConfig, CompilationMode
from vllm.forward_context import get_forward_context, is_forward_context_available
from vllm.logger import init_logger
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.layers.attention.attention import Attention
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.cpu_model_runner import _torch_cuda_wrapper
from vllm.v1.worker.gpu_model_runner import GPUModelRunner

from spyre_inference.custom_ops.rotary_embedding import _SpyreRotaryMixin
from spyre_inference.custom_ops.linear import transpose_linear_weights_for_spyre
from spyre_inference.custom_ops.unfuse import analyze_and_unfuse
from spyre_inference.custom_ops.utils import convert

logger = init_logger(__name__)

# DIAGNOSTIC: SPYRE_EMBED_ROUNDTRIP=1 routes the text-only inputs_embeds through
# D2H->H2D so it reaches the compiled graph with the same "transferred" provenance
# as the multimodal merge. Used to test whether transferred graph inputs are what
# corrupts compile-mode multimodal output. Default 0 (device-native, unchanged).
_EMBED_ROUNDTRIP = os.environ.get("SPYRE_EMBED_ROUNDTRIP", "0") == "1"

# DIAGNOSTIC (SPYRE_COMPILE_SCOPE): narrow what torch.compile covers, to bisect
# which region of the decoder is miscompiled when the token count is 1 (compiled
# output is garbage at 1 token, correct at >= 2; eager is correct at every size).
#   "model"    - default, unchanged: one fullgraph over the whole model
#   "none"     - compile nothing (equivalent to eager, but keeps the compile path)
#   "mlp"      - compile only each decoder layer's `mlp`
#   "attn"     - compile only each decoder layer's `self_attn`
#   "attn:X"   - compile only child X of self_attn (qkv_proj, o_proj, rotary_emb,
#                attn); an unknown name raises listing the available children
#   "layers:N" - compile only the first N decoder layers, whole-layer
# Scoped modes use fullgraph=False, since a submodule boundary legitimately
# graph-breaks (e.g. the opaque attention op); they answer "does compiling this
# region corrupt output", not "does this region compile as one graph".
_COMPILE_SCOPE = os.environ.get("SPYRE_COMPILE_SCOPE", "model")

# DIAGNOSTIC (SPYRE_COMPILE_BACKEND): backend for scoped compiles, to tell which
# compiler stage corrupts. "eager" = dynamo trace only (guards/specialization),
# "aot_eager" = + AOTAutograd functionalization, "inductor" = + codegen. The
# first backend that produces garbage is the stage at fault.
_COMPILE_BACKEND = os.environ.get("SPYRE_COMPILE_BACKEND", "inductor")


def _llama4_attn_scale_op(
    positions: torch.Tensor, beta: float, original_max_position_embeddings: int
) -> torch.Tensor:
    """Opaque-op body: run **upstream's** Mistral/Llama-4 attention temperature
    scaling on CPU and return fp16 on the query device.

    ``positions`` is int64 on Spyre and torch-spyre can't convert int64->float32
    on-device (nor run fp32 ``log``), so the scale is computed on CPU. We reuse
    upstream ``MistralAttention._get_llama_4_attn_scale`` verbatim (no formula
    duplication) — it reads only ``self.llama_4_scaling_beta`` and
    ``self.llama_4_scaling_original_max_position_embeddings``, which we supply via
    a stub since a custom op can't take the module itself. Being an opaque op it
    works in BOTH eager and ``torch.compile`` (the CPU math never enters the
    Spyre graph).
    """
    from types import SimpleNamespace

    from vllm.model_executor.models.mistral import MistralAttention

    stub = SimpleNamespace(
        llama_4_scaling_beta=beta,
        llama_4_scaling_original_max_position_embeddings=original_max_position_embeddings,
    )
    scaling = MistralAttention._get_llama_4_attn_scale(stub, positions.to("cpu"))
    return convert(scaling, device=positions.device, dtype=torch.float16)


def _llama4_attn_scale_fake(
    positions: torch.Tensor, beta: float, original_max_position_embeddings: int
) -> torch.Tensor:
    return torch.empty((positions.shape[0], 1), dtype=torch.float16, device=positions.device)


@lru_cache(maxsize=1)
def _register_llama4_attn_scale_op() -> None:
    """Register ``torch.ops.vllm.spyre_llama4_attn_scale`` once. Called from
    ``load_model`` (not import) so ``current_platform`` is resolved."""
    from vllm.platforms import current_platform
    from vllm.utils.torch_utils import direct_register_custom_op

    direct_register_custom_op(
        op_name="spyre_llama4_attn_scale",
        op_func=_llama4_attn_scale_op,
        fake_impl=_llama4_attn_scale_fake,
        dispatch_key=current_platform.dispatch_key,
    )


# Observed Spyre DMA failure threshold for encoder-only dummy batches with
# multiple sequences.  Pooling warmup stays below this limit.
SPYRE_ENCODER_DMA_TOKEN_LIMIT = 30
# Token count for pooling warmup (single sequence), kept under the DMA limit.
SPYRE_ENCODER_WARMUP_MAX_TOKENS = 16

# Pure-PyTorch replacement for torch.ops._C.compute_slot_mapping_kernel_impl
# (unavailable with VLLM_TARGET_DEVICE=empty).

_PAD_SLOT_ID = -1


def _compute_slot_mapping_impl(
    num_tokens: int,
    max_num_tokens: int,
    query_start_loc: torch.Tensor,
    positions: torch.Tensor,
    block_table: torch.Tensor,
    block_table_stride: int,
    block_size: int,
    slot_mapping: torch.Tensor,
    KV_CACHE_BLOCK_SIZE: int | None = None,
    BLOCKS_PER_KV_BLOCK: int = 1,
    TOTAL_CP_WORLD_SIZE: int = 1,
    TOTAL_CP_RANK: int = 0,
    CP_KV_CACHE_INTERLEAVE_SIZE: int = 1,
    PAD_ID: int = _PAD_SLOT_ID,
    # Triton tile width; unused here, kept for call compatibility.
    BLOCK_SIZE: int = 1024,
) -> None:
    """Map each token position to its flat index in the paged KV cache.

    The upstream vLLM implementation is a Triton kernel (requires a GPU) and
    the CPU backend delegates to a C++ op in _C.so. Neither is available with
    VLLM_TARGET_DEVICE=empty, so we reimplement the logic in pure PyTorch.

    Correctness is validated indirectly by the upstream attention backend test
    (test_causal_backend_correctness) and end-to-end model generation tests.

    ``block_size`` is the kernel's block size, ``KV_CACHE_BLOCK_SIZE`` the KV
    manager's, and ``BLOCKS_PER_KV_BLOCK`` the ratio between them (1 on Spyre).
    """
    assert TOTAL_CP_WORLD_SIZE == 1, "Context Parallelism is not supported on Spyre."
    kv_block_size = block_size if KV_CACHE_BLOCK_SIZE is None else KV_CACHE_BLOCK_SIZE

    # KV manager block, then the kernel block within it.
    token_positions = positions[:num_tokens]
    virtual_block_indices = (token_positions // kv_block_size).to(torch.int64)
    local_block_offsets = (token_positions % kv_block_size).to(torch.int64)
    block_indices = virtual_block_indices * BLOCKS_PER_KV_BLOCK + local_block_offsets // block_size

    num_reqs = query_start_loc.shape[0] - 1
    req_indices = torch.empty(num_tokens, dtype=torch.int64, device=positions.device)
    for i in range(num_reqs):
        start = query_start_loc[i].item()
        end = query_start_loc[i + 1].item()
        req_indices[start:end] = i

    flat_indices = req_indices * block_table_stride + block_indices
    block_numbers = block_table.flatten()[flat_indices].to(torch.int64)
    slot_mapping[:num_tokens] = block_numbers * block_size + local_block_offsets % block_size
    if max_num_tokens > num_tokens:
        slot_mapping[num_tokens:max_num_tokens] = PAD_ID


class _FuncWrapper:
    """Mimics Triton's grid-launch syntax: kernel[(grid,)](...) → kernel(...)."""

    def __init__(self, func):
        self.func = func

    def __getitem__(self, grid):
        return self.func


_compute_slot_mapping_kernel = _FuncWrapper(_compute_slot_mapping_impl)


class SpyreCpuGpuBuffer(CpuGpuBuffer):
    """Spyre-specific CpuGpuBuffer with Spyre-safe copies and split dtypes.
    This buffer is closely related to the CpuGpuBuffer in vllm/v1/utils.py.

    For float dtypes: .cpu on CPU, .gpu on Spyre (float16).
    For int/bool dtypes: .gpu aliased to .cpu (CPUModelRunner pattern).
    Float H2D uses ``non_blocking=True``; callers must sync via
    ``TorchSpyreModelRunner._sync_device`` (``torch.spyre.synchronize``)
    before consuming the Spyre tensors.

    Inherits from `CpuGpuBuffer` (without invoking its `__init__`) so that
    `_make_buffer` overrides remain Liskov-compatible with `GPUModelRunner`.
    """

    def __init__(
        self,
        *size: int | torch.SymInt,
        cpu_dtype: torch.dtype,
        gpu_dtype: torch.dtype,
        device: torch.device,
        pin_memory: bool,
        with_numpy: bool = True,
    ) -> None:
        self.cpu = torch.zeros(*size, dtype=cpu_dtype, device="cpu", pin_memory=pin_memory)
        if device.type == "spyre":
            self.gpu = torch.zeros(*size, dtype=gpu_dtype, device=device)
        else:
            # int/bool: alias gpu = cpu (CPUModelRunner pattern)
            self.gpu = self.cpu
        self.np: np.ndarray
        if with_numpy:
            if cpu_dtype == torch.bfloat16:
                raise ValueError(
                    "Bfloat16 torch tensors cannot be directly cast to a "
                    "numpy array, so call SpyreCpuGpuBuffer with "
                    "with_numpy=False"
                )
            self.np = self.cpu.numpy()

    def copy_to_gpu(self, n: int | None = None) -> torch.Tensor:
        if self.gpu is self.cpu:
            # Aliased (int/bool) — no copy needed
            return self.gpu if n is None else self.gpu[:n]
        src = self.cpu if n is None else self.cpu[:n]
        dst = self.gpu if n is None else self.gpu[:n]
        # Async H2D via torch-spyre's aten::_copy_from / copyAsync path.
        # GPUModelRunner calls _sync_device before the tensors are consumed.
        dst.copy_(src, non_blocking=True)
        return dst

    def copy_to_cpu(self, n: int | None = None) -> torch.Tensor:
        # Currently only the copy_to_gpu function is invoked.
        # If the copy_to_cpu also becomes required, override it here with
        # spyre-specific aspects.
        raise NotImplementedError("SpyreCpuGpuBuffer.copy_to_cpu is not implemented")


class _SpyreModelWrapper:
    """Transparent wrapper that converts model inputs/outputs at the boundary.

    Input conversion (CPU → Spyre):
        For example, input_ids and positions arrive as CPU tensors (int32/int64) because
        self.device=CPU in the runner and buffer scatter ops run on CPU.
        Convert them to int64 and provide them to the model.

    Output conversion (Spyre → CPU):
        The model's final hidden_states come out on Spyre. Downstream
        operations (indexing via logits_indices, sampling) run on CPU.
        The lm_head matmul runs on Spyre via SpyreParallelLMHead,
        which handles H2D/D2H for the sample_hidden_states subset.

    RoPE priming (per forward pass):
        Gather each RoPE module's per-token rotation slice on the host (no D2H)
        and stash it in the forward context; forward_oot reads it back, shared
        across all attention layers.

    Wrapping at the model level ensures ALL call sites get the right
    device — both execute_model (via _model_forward) and _dummy_run
    (which calls self.model(...) directly).
    """

    def __init__(
        self,
        model: nn.Module,
        spyre_device: torch.device,
        rope_modules: list[_SpyreRotaryMixin] | None = None,
    ):
        # Use object.__setattr__ to avoid triggering __setattr__ override
        object.__setattr__(self, "_model", model)
        object.__setattr__(self, "_spyre_device", spyre_device)
        object.__setattr__(self, "_rope_modules", rope_modules or [])

    def __call__(self, *args, **kwargs):
        # Prime RoPE while positions are still on the host (no D2H).
        self._prime_rope_rotation(kwargs.get("positions"))

        # Convert integer tensor inputs to Spyre int64
        def _convert_int(t):
            if (
                t is not None
                and isinstance(t, torch.Tensor)
                and t.dtype in (torch.int32, torch.int64)
            ):
                return convert(t, dtype=torch.int64, device=self._spyre_device)
            return t

        args_converted = []
        for arg in args:
            args_converted.append(_convert_int(arg))

        kwargs_converted = {}
        for key in kwargs:
            val = kwargs.get(key)
            kwargs_converted[key] = _convert_int(val)

        t0 = time.time()
        result = self._model(*args_converted, **kwargs_converted)

        def _to_cpu(x):
            return convert(x, device="cpu")

        result = tree_map(_to_cpu, result)

        input_ids = kwargs_converted.get("input_ids")
        num_tokens = input_ids.shape[0] if input_ids is not None else -1
        logger.debug("t_token: %.2fms [num tokens %d]", (time.time() - t0) * 1000, num_tokens)

        return result

    def _prime_rope_rotation(self, positions: torch.Tensor | None) -> None:
        """Pre-gather each RoPE module's per-token rotation slice into the forward
        context. Modules with no Spyre path return None from gather_rotation."""
        if positions is None or not self._rope_modules or not is_forward_context_available():
            return
        # vLLM's positions buffer is int64; downcast on the host (free, and positions are
        # always < max_model_len) so the on-device gather uses int32 indices directly and
        # skips torch-spyre's internal int64 downcast.
        positions = positions.to(torch.int32)
        rope_rot = {}
        for rope in self._rope_modules:
            rot = rope.gather_rotation(positions, self._spyre_device)
            if rot is not None:
                rope_rot[rope._rope_key] = rot
        if rope_rot:
            get_forward_context().additional_kwargs["spyre_rope_rot"] = rope_rot

    def embed_multimodal(self, **kwargs):
        """Move float multimodal inputs (e.g. ``pixel_values``) onto Spyre.

        vLLM's runner calls ``embed_multimodal`` directly (via ``__getattr__``,
        bypassing ``__call__``'s input conversion), so the pixel tensors would
        otherwise reach the vision tower + projector — whose weights live on
        Spyre after ``model.to`` in ``load_model`` — while still on CPU, hitting
        a cpu-activations × spyre-weights mismatch. Mirror ``compute_logits``'
        H2D for the float inputs; ``convert`` is a no-op for tensors already on
        the target device, and non-float entries pass through untouched.
        """

        def _to_spyre_float(t):
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                return convert(t, dtype=torch.float16, device=self._spyre_device)
            return t

        kwargs = tree_map(_to_spyre_float, kwargs)
        out = self._model.embed_multimodal(**kwargs)
        return out

    def embed_input_ids(
        self,
        input_ids,
        multimodal_embeddings=None,
        *,
        is_multimodal=None,
    ):
        """Text-token embedding + multimodal merge, Spyre-aware.

        The runner calls this directly (``gpu_model_runner`` :3526/:3573) via
        ``__getattr__``, bypassing ``__call__``'s conversion, so ``input_ids``
        arrive on CPU while ``embed_tokens`` lives on Spyre.

        - **No image in the batch:** move ``input_ids`` to Spyre and do the
          embedding lookup on-card (``embedding`` is now a Spyre op).
        - **Image present:** the upstream merge
          (``_merge_multimodal_embeddings``) scatters image rows via
          ``inputs_embeds[is_multimodal] = ...`` — a **dim-0 boolean-mask
          scatter Spyre cannot do**. So we keep the on-card text lookup, then
          D2H and run the (unmodified) merge on CPU, then H2D the result for the
          decoder. ``image_token_index`` is in-vocab for Mistral3 (10 <
          vocab_size), so the placeholder tokens embed without an OOV masked
          fill; they are overwritten by the merge regardless.
        """
        input_ids = convert(input_ids, dtype=torch.int64, device=self._spyre_device)
        inputs_embeds = self._model.embed_input_ids(input_ids)

        if multimodal_embeddings is None or len(multimodal_embeddings) == 0:
            if _EMBED_ROUNDTRIP:
                # DIAGNOSTIC (SPYRE_EMBED_ROUNDTRIP=1): give the text path the same
                # D2H->H2D provenance the multimodal merge has, so its inputs_embeds
                # enters the compiled graph as a *transferred* tensor rather than a
                # device-native one. If text output then turns to gibberish under
                # compile, transferred graph inputs (missing device_tensor_layout)
                # are the multimodal corruptor.
                inputs_embeds = convert(
                    convert(inputs_embeds, device="cpu"), device=self._spyre_device
                )
            return inputs_embeds

        from vllm.model_executor.models.utils import _merge_multimodal_embeddings

        inputs_embeds = convert(inputs_embeds, device="cpu")
        mm_embeds_cpu = tree_map(
            lambda t: convert(t, device="cpu") if isinstance(t, torch.Tensor) else t,
            multimodal_embeddings,
        )
        merged = _merge_multimodal_embeddings(
            inputs_embeds=inputs_embeds,
            multimodal_embeddings=mm_embeds_cpu,
            is_multimodal=is_multimodal.to("cpu"),
        )
        return convert(merged, device=self._spyre_device)

    def compute_logits(self, hidden_states, *args, **kwargs):
        """Move hidden_states onto Spyre for the lm_head custom op.

        gpu_model_runner.execute_model slices `hidden_states[logits_indices]`
        on CPU (Spyre cannot slice), so the tensor handed to compute_logits
        is on CPU; move it onto Spyre for the lm_head matmul. The logits are
        returned on CPU: SpyreParallelLMHead.forward_oot keeps them on Spyre
        for the TP all_gather, and SpyreLogitsProcessor._gather_logits
        converts back to CPU right after the gather (before the vocab slice
        and scale), so downstream sampling gets CPU logits.
        """
        hidden_states = convert(hidden_states, device=self._spyre_device)
        return self._model.compute_logits(hidden_states, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)

    def __setattr__(self, name, value):
        setattr(self._model, name, value)


class TorchSpyreModelRunner(GPUModelRunner):
    """Model runner for Spyre.

    Treats Spyre as the 'GPU' device in vLLM's CpuGpuBuffer pattern:
    - .cpu tensors on CPU (numpy staging for scheduler)
    - .gpu tensors on Spyre for floats, aliased to CPU for int/bool

    Inherits from GPUModelRunner to preserve
    the dual-buffer device placement pattern.
    """

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # Store the real Spyre device before super().__init__ so that
        # _make_buffer can place .gpu tensors on Spyre directly.
        self._spyre_device = device

        # Phase 1: Init with device="cpu" to avoid dtype/device errors.
        # Many components create tensors on self.device during init, and
        # Spyre doesn't support all dtypes (int32, bool) natively.
        # _make_buffer (overridden below) already places .gpu on Spyre
        # via self._spyre_device regardless of self.device.
        with _torch_cuda_wrapper():
            super().__init__(vllm_config, torch.device("cpu"))

        # Keep self.device as CPU so buffer management (scatter, copy) stays
        # on CPU. _SpyreModelWrapper converts input_ids/positions to Spyre
        # int64 at the model boundary.
        # _make_buffer (overridden below) places float .gpu tensors on Spyre
        # regardless of self.device.

        # Disable GPU-specific features (same as CPUModelRunner)
        self.use_cuda_graph = False
        self.cascade_attn_enabled = False

        # Replace Triton kernel with a pure-PyTorch implementation.
        # GPUModelRunner uses @triton.jit which is mocked on non-GPU platforms.
        # The upstream CPU backend uses a C++ kernel (torch.ops._C) as its
        # fallback, but we don't have _C.abi3.so with VLLM_TARGET_DEVICE=empty.
        from vllm.v1.worker import block_table

        # Deliberately swap the Triton JITFunction for the grid-launch-compatible
        # _FuncWrapper; the type mismatch is the point of the patch.
        block_table._compute_slot_mapping_kernel = _compute_slot_mapping_kernel  # ty: ignore[invalid-assignment]

    @staticmethod
    def _patch_encoder_ops_for_spyre(model_config) -> None:
        """Stub ``bert._decode_token_type_ids`` with zeros; Spyre cannot lower
        the bitwise unpack. Pooling BERT-family only (segment 0). Process-global.
        """
        from vllm.model_executor.models import bert

        if not hasattr(bert, "_decode_token_type_ids"):
            raise RuntimeError(
                "vllm.model_executor.models.bert._decode_token_type_ids "
                "not found; Spyre encoder patch needs updating for this "
                "vLLM version."
            )

        if model_config.runner_type != "pooling":
            return

        logger.warning(
            "Spyre: patching bert._decode_token_type_ids to zeros (segment 0 only). Model: %s",
            model_config.model,
        )

        def _decode_token_type_ids(input_ids: torch.Tensor) -> torch.Tensor:
            return torch.zeros_like(input_ids)

        bert._decode_token_type_ids = _decode_token_type_ids  # ty: ignore[invalid-assignment]

    def _patch_llama4_attn_scale(self) -> None:
        """Route Mistral/Llama-4 attention temperature scaling through the
        ``spyre_llama4_attn_scale`` opaque op (which reuses upstream's formula on
        CPU; see ``_llama4_attn_scale_op``).

        ``MistralAttention._get_llama_4_attn_scale`` computes
        ``1 + beta * log(1 + floor(positions / orig_max))``; ``positions`` is int64
        on Spyre and torch-spyre can't convert int64->float32 on-device (nor run
        fp32 ``log``). The opaque op runs it on CPU and returns fp16, so it works in
        **both eager and torch.compile**.

        Patched **per-instance via module traversal**, which requires running AFTER
        the model is loaded (the real ``MistralAttention`` modules must exist) and
        BEFORE ``torch.compile`` wraps it — an ``OptimizedModule`` doesn't traverse
        to the underlying submodules.
        """
        from vllm.model_executor.models.mistral import MistralAttention

        _register_llama4_attn_scale_op()

        def _make_scale(module: MistralAttention):
            beta = float(module.llama_4_scaling_beta)
            orig_max = int(module.llama_4_scaling_original_max_position_embeddings)

            def _get_llama_4_attn_scale(positions: torch.Tensor) -> torch.Tensor:
                return torch.ops.vllm.spyre_llama4_attn_scale(positions, beta, orig_max)

            return _get_llama_4_attn_scale

        mistral_attns = [m for m in self.model.modules() if isinstance(m, MistralAttention)]

        # Guard against a stale patch: if MistralAttention exists but none carries
        # the `do_llama_4_scaling` attribute, the upstream API changed and the gate
        # below would silently skip every layer (llama-4 scaling never applied).
        # A legit non-scaled model still has the attribute (value False), so this
        # only fires on a real rename/removal.
        if mistral_attns and not any(hasattr(m, "do_llama_4_scaling") for m in mistral_attns):
            raise RuntimeError(
                "MistralAttention has no 'do_llama_4_scaling' attribute — the Spyre "
                "llama-4 attention-scale patch is stale for this vLLM version; update "
                "_patch_llama4_attn_scale."
            )

        n = 0
        for module in mistral_attns:
            if getattr(module, "do_llama_4_scaling", False):
                module._get_llama_4_attn_scale = _make_scale(module)
                n += 1
        if n:
            logger.info("Spyre: patched %d Llama-4 attention-scale module(s) to CPU.", n)

    @staticmethod
    def _patch_pixtral_vision_rope() -> None:
        """Run Pixtral's 2D vision RoPE ON-CARD, expressing `rotate_half` as a matmul.

        transformers' `apply_rotary_pos_emb`/`rotate_half` half-slice the head dim
        (`x[..., :d/2]` / `x[..., d/2:]`), which corrupts on Spyre — and it's a
        plain module-level function, not a vLLM CustomOp, so `register_oot` can't
        catch it. The text path's 2x2 helper (`_rotate_neox_2x2`) isn't usable here:
        vision head_dim=64 → the `d/2 = 32`-wide half is *half a Spyre stick* (64),
        which torch-spyre can't lay out ("Unexpected stick expression ... Mod(var,
        32)"). Instead we keep the neox formula `q*cos + rotate_half(q)*sin` but
        realize `rotate_half` as a constant `[head_dim, head_dim]` matmul (`x @ R`),
        which is stick-aligned over the full 64-wide dim — so the whole rotation of
        the large q/k ViT activations stays ON SPYRE with no sub-stick slice.

        Monkeypatches the module-level name `PixtralHFAttention.forward` resolves at
        call time; guarded/no-op if the symbol is absent (non-Pixtral / different
        vLLM layout). Eager path; wrap in an opaque op if the vision tower must
        compile. If an op still raises on hardware, fall *that* call back to CPU —
        do not revert the whole rope.
        """
        try:
            from vllm.model_executor.models import pixtral
        except ImportError:
            return

        orig = getattr(pixtral, "apply_rotary_pos_emb", None)
        if orig is None or getattr(orig, "_spyre_patched", False):
            return

        @lru_cache(maxsize=None)
        def _rotate_half_matrix(head_dim: int, device: torch.device) -> torch.Tensor:
            """Constant `[head_dim, head_dim]` matrix `R` with `x @ R == rotate_half(x)`.

            neox `rotate_half(x) = cat([-x[d/2:], x[:d/2]])`. Expressing it as a
            full-width matmul avoids slicing the head into two `d/2`-wide halves —
            for head_dim=64 that half is 32 = half a Spyre stick, which torch-spyre
            can't lay out ("Unexpected stick expression ... Mod(var, 32)"). A matmul
            over the full 64-wide dim is stick-aligned, so the rotation stays on-card.
            Built once on CPU per (head_dim, device), then moved to the device.
            """
            half = head_dim // 2
            r = torch.zeros(head_dim, head_dim, dtype=torch.float16)
            idx = torch.arange(half)
            r[idx + half, idx] = -1.0  # out[:half] = -x[half:]
            r[idx, idx + half] = 1.0  # out[half:] =  x[:half]
            return convert(r, device=device, dtype=torch.float16)

        def _apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
            # q, k: [batch, n_heads, patches, head_dim]; cos/sin: [patches, head_dim].
            dev = q.device
            head_dim = q.shape[-1]
            rot_r = _rotate_half_matrix(head_dim, dev)
            # cos/sin come from PixtralRotaryEmbedding (arange/sin/cos → may be CPU);
            # move to device and add [batch, n_heads] broadcast dims.
            cos = convert(cos, device=dev, dtype=torch.float16).unsqueeze(0).unsqueeze(0)
            sin = convert(sin, device=dev, dtype=torch.float16).unsqueeze(0).unsqueeze(0)

            def rotate(x):
                # q*cos + rotate_half(q)*sin, with rotate_half as a stick-aligned matmul.
                return x * cos + torch.matmul(x, rot_r) * sin

            return rotate(q), rotate(k)

        _apply_rotary_pos_emb._spyre_patched = True
        pixtral.apply_rotary_pos_emb = _apply_rotary_pos_emb  # ty: ignore[invalid-assignment]
        logger.info(
            "Spyre: patched Pixtral vision apply_rotary_pos_emb to on-card "
            "rotate-half-matrix rotation."
        )

    @staticmethod
    def _patch_pixtral_vision_rope_vit() -> None:
        """Run the mistral-native Pixtral `VisionTransformer` 2D-RoPE ON-CARD.

        This `VisionTransformer` (mistral checkpoint format) has a *complex* rope
        whose `forward` also gathers per-token freqs with advanced indexing. Three
        Spyre-hostile steps, hit in this order as each was cleared:

        1. `freqs_cis.to("spyre")` — `freqs_cis` is `complex64`; Spyre has no complex
           dtype → `RuntimeError: Spyre backend does not support dtype ComplexFloat`.
        2. `freqs_cis[positions[:, 0], positions[:, 1]]` (pixtral.py:949) — advanced
           index on a Spyre tensor → `NotImplementedError: Could not run
           'aten::index.Tensor_out' ... from the 'spyre' backend`.
        3. `view_as_complex(q) * freqs_cis` in `apply_rotary_emb_vit` — complex math,
           unsupported on Spyre.

        All three run on-card by (a) representing the rope as a *real* rotation and
        (b) doing the 2-D gather as a 1-D `aten::index_select` (which Spyre supports,
        unlike `aten::index`) on a flattened table:

        - `VisionTransformer.freqs_cis` builds a real `(H, W, 2, head_dim)` table on
          CPU (`[...,0,:]=cos_full`, `[...,1,:]=sin_signed`), flattens it to
          `(H*W, 2, head_dim)` and moves the *real* tensor to Spyre (fixes 1). It
          returns a wrapper whose `__getitem__` turns `(row, col)` into a flat index
          `row*W + col` and gathers with `index_select` on-card (fixes 2).
        - `apply_rotary_emb_vit` is the real rotation `x·cos + (x @ P)·sin`, with `P`
          a constant `[head_dim, head_dim]` pair-swap matrix — a full-stick-width
          matmul, no sub-stick slice, no complex (fixes 3). Same trade as
          `_rotate_half_matrix`.

        CPU-rotary fallback (documented, not enabled): if `index_select` / the flat
        index / the pair-swap matmul ever fail to lay out, keep the freqs table
        *complex on CPU*, have the wrapper move the Spyre indices to CPU and gather
        there, and replace the rotation with a version that D2H's q/k, runs the
        original `view_as_complex → *freqs_cis → view_as_real` on CPU, and H2D's the
        result — correctness-safe but a D2H/H2D per vision attention layer.
        """
        try:
            from vllm.model_executor.models import pixtral
        except ImportError:
            return

        orig = getattr(pixtral, "apply_rotary_emb_vit", None)
        vt = getattr(pixtral, "VisionTransformer", None)
        if orig is None or vt is None or getattr(orig, "_spyre_patched", False):
            return

        @lru_cache(maxsize=None)
        def _pair_swap_matrix(head_dim: int, device: torch.device) -> torch.Tensor:
            """Constant `[head_dim, head_dim]` `P`: `(x @ P)` swaps each `(2k, 2k+1)` pair."""
            p = torch.zeros(head_dim, head_dim, dtype=torch.float16)
            even = torch.arange(0, head_dim, 2)
            p[even, even + 1] = 1.0
            p[even + 1, even] = 1.0
            return convert(p, device=device, dtype=torch.float16)

        class _OnCardFreqsTable:
            """Flattened real freqs table on Spyre; gathers per-token rows on-card
            via `index_select` (a 1-D flat index), avoiding `aten::index`."""

            def __init__(self, table: torch.Tensor, width: int):
                self._table = table  # (H*W, 2, head_dim) on Spyre
                self._width = width

            def __getitem__(self, idx):
                # idx = (positions[:, 0], positions[:, 1]): column *views* of the
                # Spyre positions tensor. positions[:, 1] has storage_offset=1, and
                # materializing that sliced view on-device needs a stick-aligned
                # offset (1 is not). Fold the columns into a flat index on CPU (tiny,
                # offset-free), then H2D just the contiguous index and gather on-card.
                row, col = idx
                flat = (row.to("cpu") * self._width + col.to("cpu")).to(torch.int64)
                flat = convert(flat, device=self._table.device, dtype=torch.int64)
                return self._table.index_select(0, flat)  # (seq, 2, head_dim)

        def _freqs_cis_ondev(self):
            # Real packed table [..., 0, :]=cos_full, [..., 1, :]=sin_signed, flattened
            # to (H*W, 2, head_dim). Complex precompute on CPU; store real so neither
            # the device move nor the on-card rope touches a complex tensor.
            if self._freqs_cis is None:
                fc = pixtral.precompute_freqs_cis_2d(
                    dim=self.args.hidden_size // self.args.num_attention_heads,
                    height=self.max_patches_per_side,
                    width=self.max_patches_per_side,
                    theta=self.args.rope_theta,
                )  # (H, W, head_dim//2) complex64 on CPU
                cos = fc.real
                sin = fc.imag
                cos_full = cos.repeat_interleave(2, dim=-1)
                sin_signed = torch.stack([-sin, sin], dim=-1).reshape(*sin.shape[:-1], -1)
                packed = torch.stack([cos_full, sin_signed], dim=-2)  # (H, W, 2, head_dim)
                self._freqs_cis = packed.reshape(
                    -1, packed.shape[-2], packed.shape[-1]
                ).to(torch.float16)  # (H*W, 2, head_dim) on CPU
            if self._freqs_cis.device != self.device:
                self._freqs_cis = convert(
                    self._freqs_cis, device=self.device, dtype=torch.float16
                )
            return _OnCardFreqsTable(self._freqs_cis, self.max_patches_per_side)

        def _apply_rotary_emb_vit(xq, xk, freqs_cis):
            # xq, xk: [batch, patches, n_heads, head_dim] on Spyre.
            # freqs_cis: real [patches, 2, head_dim] on Spyre (gathered per token).
            p = _pair_swap_matrix(xq.shape[-1], xq.device)
            cos = freqs_cis[:, 0, :][None, :, None, :]  # [1, patches, 1, head_dim]
            sin = freqs_cis[:, 1, :][None, :, None, :]

            def rot(x):
                return x * cos + torch.matmul(x, p) * sin

            return rot(xq).type_as(xq), rot(xk).type_as(xk)

        _apply_rotary_emb_vit._spyre_patched = True
        pixtral.apply_rotary_emb_vit = _apply_rotary_emb_vit  # ty: ignore[invalid-assignment]
        vt.freqs_cis = property(_freqs_cis_ondev)  # ty: ignore[invalid-assignment]
        logger.info(
            "Spyre: patched Pixtral VisionTransformer 2D-RoPE to on-card real "
            "rotation (index_select freqs gather + pair-swap matmul)."
        )

    @staticmethod
    def _patch_pixtral_vision_attention() -> None:
        """Run the Pixtral vision `Attention` SDPA ON-CARD with stick-aligned padding.

        Stock `Attention.forward` calls `nn.functional.scaled_dot_product_attention`
        (pixtral.py:769), which torch-spyre lowers to two batch-matmuls. For a raw
        patch count `L` that is coprime with the 64-wide stick, one operand can't be
        restickified onto its matmul dim → `Unsupported: batchmatmul: cannot
        restickify any input layout of y to carry y_var=...`.

        Same fix as `SpyreEncoderAttentionImpl`: pad the sequence length (and head
        dim) up to the 64 stick so both matmul reduction dims are stick-aligned,
        assemble the padded dense batch + additive mask on CPU (strided
        pad/slice/transpose are CPU-only on Spyre), run one SDPA on-device, then crop
        back. Padded keys get `-inf` mask so they never contribute; padded queries
        are cropped. Replaces `Attention.forward` (no xformers on Spyre); guarded and
        idempotent.
        """
        try:
            from vllm.model_executor.models import pixtral
        except ImportError:
            return

        attn_cls = getattr(pixtral, "Attention", None)
        if attn_cls is None or getattr(attn_cls.forward, "_spyre_patched", False):
            return

        stick = 64

        def _spyre_vision_sdpa(q, k, v, mask):
            # q, k, v: [B, H, L, D] on device. Pad L,D to the stick, SDPA on-card, crop.
            b, h, seq, d = q.shape
            scale = d**-0.5
            seq_pad = ((seq + stick - 1) // stick) * stick
            d_pad = ((d + stick - 1) // stick) * stick

            qc = convert(q, "cpu")
            kc = convert(k, "cpu")
            vc = convert(v, "cpu")
            dtype = qc.dtype

            qb = torch.zeros(b, h, seq_pad, d_pad, dtype=dtype)
            kb = torch.zeros(b, h, seq_pad, d_pad, dtype=dtype)
            vb = torch.zeros(b, h, seq_pad, d_pad, dtype=dtype)
            qb[:, :, :seq, :d] = qc
            kb[:, :, :seq, :d] = kc
            vb[:, :, :seq, :d] = vc

            neg_inf = torch.finfo(dtype).min
            m = torch.zeros(b, 1, seq_pad, seq_pad, dtype=dtype)
            m[:, :, :, seq:] = neg_inf  # padded keys never attended
            if mask is not None:
                mc = convert(mask, "cpu")
                if mc.dtype == torch.bool:
                    add = torch.zeros(seq, seq, dtype=dtype).masked_fill(
                        ~mc.reshape(seq, seq), neg_inf
                    )
                else:
                    add = mc.to(dtype).reshape(seq, seq)
                m[:, :, :seq, :seq] = m[:, :, :seq, :seq] + add

            dev = q.device.type
            out = torch.nn.functional.scaled_dot_product_attention(
                convert(qb, dev),
                convert(kb, dev),
                convert(vb, dev),
                attn_mask=convert(m, dev),
                scale=scale,
            )
            out = convert(out, "cpu")[:, :, :seq, :d].contiguous()
            return convert(out, q.device.type)

        def _forward(self, x, mask, freqs_cis):
            batch, patches, _ = x.shape
            qkv, _ = self.qkv_proj(x)
            q, k, v = qkv.chunk(3, dim=-1)
            q = q.reshape(batch, patches, self.n_heads, self.head_dim)
            k = k.reshape(batch, patches, self.n_heads, self.head_dim)
            v = v.reshape(batch, patches, self.n_heads, self.head_dim)
            q, k = pixtral.apply_rotary_emb_vit(q, k, freqs_cis=freqs_cis)
            # [B, H, L, D] for SDPA.
            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
            out = _spyre_vision_sdpa(q, k, v, mask)
            out = out.transpose(1, 2).reshape(batch, patches, self.n_heads * self.head_dim)
            out, _ = self.o_proj(out)
            return out

        _forward._spyre_patched = True
        attn_cls.forward = _forward  # ty: ignore[invalid-assignment]
        logger.info(
            "Spyre: patched Pixtral vision Attention to stick-aligned padded "
            "on-card SDPA (pad L/D to 64, mask, crop)."
        )

    def _offload_pixtral_projector_norm_cpu(self) -> None:
        """Run Pixtral's `pre_mm_projector_norm` (RMSNorm) on CPU.

        The projector norm reduces over the last dim of the (N_patches, D) vision
        features; on-device that reduction + restickify stalls torch-spyre's compile
        for these shapes. It runs once per image and is tiny, so offload just this
        instance: move its weight to CPU, D2H the input, H2D the output. The on-card
        `SpyreRMSNorm` used by the text decoder is unaffected. No-op if absent.
        """
        dev = self._spyre_device
        for module_name, module in self.model.named_modules():
            if module_name.rsplit(".", 1)[-1] != "pre_mm_projector_norm":
                continue
            module.to("cpu")

            def _to_cpu(mod, args):
                return tuple(
                    a.to("cpu") if isinstance(a, torch.Tensor) else a for a in args
                )

            def _to_dev(mod, args, output, _dev=dev):
                if isinstance(output, tuple):
                    return tuple(
                        convert(o, device=_dev) if isinstance(o, torch.Tensor) else o
                        for o in output
                    )
                return convert(output, device=_dev)

            module.register_forward_pre_hook(_to_cpu)
            module.register_forward_hook(_to_dev)
            logger.info(
                "Spyre: %s offloaded to CPU (D2H in / H2D out)", module_name
            )

    @staticmethod
    def _patch_pixtral_patch_merger() -> None:
        """Run Pixtral `PatchMerger.permute` (spatial s×s regroup) ON-CARD, CPU fallback.

        Stock `PatchMerger.permute` -> `get_sub_grids` uses `F.unfold`
        (`aten::im2col`), which Spyre has no kernel for. But the s×s regroup is a
        pure space-to-depth reshuffle: it can be written with `reshape`/`permute`/
        `cat` (no im2col) that produces the *identical* result — feature order
        `(d, kh, kw)` and patch order row-major `(oh, ow)`, matching `F.unfold`. We
        try that on-card first; if the on-device relayout can't lower (or any op is
        unsupported), fall back to the original CPU unfold path. Either way the
        `merging_layer` GEMM stays on-card. Guarded/no-op if absent.
        """
        try:
            from vllm.model_executor.models import pixtral
        except ImportError:
            return

        pm_cls = getattr(pixtral, "PatchMerger", None)
        if pm_cls is None or getattr(pm_cls.forward, "_spyre_patched", False):
            return

        def _permute_on_device(x, image_sizes, s):
            # (N, d) patch tokens -> (N/s², d·s²), s×s blocks flattened as
            # (d, kh, kw). No im2col: (h·w, d) reshapes directly to
            # (h/s, s, w/s, s, d) because the flat patch index i·w+j decomposes as
            # oh·s·w + kh·w + ow·s + kw — exactly this 5-D split.
            d = x.shape[-1]
            tokens_per_image = [h * w for h, w in image_sizes]
            outs = []
            for img, (h, w) in zip(x.split(tokens_per_image), image_sizes):
                g = img.reshape(h // s, s, w // s, s, d)  # (oh, kh, ow, kw, d)
                g = g.permute(0, 2, 4, 1, 3)              # (oh, ow, d, kh, kw)
                g = g.reshape((h // s) * (w // s), d * s * s)
                outs.append(g)
            return outs[0] if len(outs) == 1 else torch.cat(outs, dim=0)

        _logged: set[str] = set()

        def _forward(self, x, image_sizes):
            dev = x.device
            try:
                x_perm = _permute_on_device(x, image_sizes, self.spatial_merge_size)
                if "on-device" not in _logged:
                    logger.info(
                        "Spyre PatchMerger: on-device regroup (reshape/permute, no "
                        "im2col)."
                    )
                    _logged.add("on-device")
            except Exception as e:  # noqa: BLE001 - fall back to CPU unfold path
                if "cpu" not in _logged:
                    logger.warning(
                        "Spyre PatchMerger: on-device regroup failed (%s: %s); "
                        "falling back to CPU unfold.",
                        type(e).__name__,
                        e,
                    )
                    _logged.add("cpu")
                x_perm = convert(self.permute(x.to("cpu"), image_sizes), device=dev)
            return self.merging_layer(x_perm)  # GEMM on-card

        _forward._spyre_patched = True
        pm_cls.forward = _forward  # ty: ignore[invalid-assignment]
        logger.info(
            "Spyre: patched Pixtral PatchMerger to on-card regroup (CPU unfold "
            "fallback; merging_layer GEMM stays on-card)."
        )

    def _patch_pixtral_patch_conv(self) -> None:
        """Run the Pixtral patch-embedding as a real on-card `F.conv2d`.

        Upstream `Conv2dLayer._forward_mulmat` lowers the patch conv (kernel==
        stride, no padding -> `enable_linear`) to a hand-rolled im2col
        (`unfold -> permute(0,2,3,1,4,5) -> reshape -> F.linear`). That on-device
        `permute/reshape` restickify has no feasible layout when a patch-grid dim
        is coprime with the 64-wide stick ("Unexpected stick expression ...
        Mod(var, 32)"), and it emits `aten.linear`, so torch-spyre's conv2d
        support is never reached.

        Route it through `F.conv2d` (a real `aten.convolution`, which torch-spyre
        lowers via `conv2d_via_bmm_decomp`) instead. Plain `.to("spyre")` tensors
        still raise in `propagate_spyre_tensor_layouts`, so the input and weight
        are pre-placed in the channel-on-stick `SpyreTensorLayout` the conv layout
        pass requires: activation sticks on `C_in`, weight/output on `C_out` (see
        torch-spyre `_conv_layouts`). The layout is built from each tensor's own
        shape, so it generalizes across Pixtral's variable image resolutions, and
        is applied on H2D (route via CPU) rather than a device->device restickify.
        Weight is relayouted once. Eager path. Guarded/no-op for non-Pixtral
        models and idempotent.
        """
        try:
            from vllm.model_executor.layers.conv import Conv2dLayer
            from torch_spyre._C import SpyreTensorLayout
        except ImportError:
            return

        convs = [
            m
            for m in self.model.modules()
            if isinstance(m, Conv2dLayer)
            and getattr(m, "enable_linear", False)
            and not getattr(m, "_spyre_conv2d", False)
        ]
        if not convs:
            return

        def _stick_layout(t: torch.Tensor, dim_order: list[int]):
            # 4-arg overload (host_size, host_strides, torch.dtype, dim_order); the
            # stick (innermost) device dim is dim_order[-1]. Mirrors _conv_layouts.
            return SpyreTensorLayout(list(t.shape), list(t.stride()), torch.float16, dim_order)

        _logged: set[str] = set()

        def _relayout(t: torch.Tensor, dim_order: list[int], tag: str) -> torch.Tensor:
            # Realize the channel-on-stick layout via H2D (CPU -> Spyre): that path
            # returns a real tensor. The device->device `.to(device_layout=)` does
            # NOT — torch-spyre's spyre_to D2D branch returns `copy_from_d2d`'s
            # result, which is None (the op mutates `dst` in place;
            # mutates_args=("dst",)). See ministral-multimodal-bringup-issues.md #6a.
            out = t.to("cpu").to(device_layout=_stick_layout(t, dim_order))
            if tag not in _logged:
                logger.info("Spyre patch-conv %s: H2D (CPU->Spyre) restickify.", tag)
                _logged.add(tag)
            return out

        def _conv2d_native(self, x: torch.Tensor) -> torch.Tensor:
            assert x.dim() == 4
            # activation: C_in (logical dim 1) on the 64-stick.
            x_dev = _relayout(x, [0, 2, 3, 1], "activation")
            return torch.nn.functional.conv2d(
                x_dev,
                self._spyre_conv2d_weight,
                self.bias,
                stride=self.stride,
                padding=self.padding,
            )

        for c in convs:
            # weight (C_out, C_in, kH, kW): C_out on the stick. Relayout once.
            c._spyre_conv2d_weight = _relayout(c.weight.detach(), [1, 2, 3, 0], "weight")
            bound = types.MethodType(_conv2d_native, c)
            c.forward_native = bound
            c._forward_method = bound
            c._spyre_conv2d = True
        logger.info(
            "Spyre: patched %d Pixtral patch-conv(s) to on-card F.conv2d "
            "(channel-on-stick layout via H2D restickify).",
            len(convs),
        )

    def load_model(self, load_dummy_weights: bool = False) -> None:
        """Load weights on CPU, move Spyre layers to device, compile, and wrap."""
        logger.info("Loading model %s...", self.model_config.model)
        t0 = time.time()

        if load_dummy_weights:
            self.load_config.load_format = "dummy"
        model_loader = get_model_loader(self.load_config)

        self._patch_encoder_ops_for_spyre(self.model_config)

        # Load model on CPU
        self.model = model_loader.load_model(
            vllm_config=self.vllm_config, model_config=self.model_config
        )
        self.model_memory_usage = 0  # No GPU memory profiling for Spyre

        # Cases appearing in GPUModelRunner.
        # When needed, they can be implemented for Spyre.
        if self.lora_config:
            raise NotImplementedError("LoRA adapters are not yet implemented and tested for Spyre.")

        if hasattr(self, "drafter"):
            raise NotImplementedError(
                "Models with a drafter model are not yet implemented and tested for Spyre."
            )

        # Un-fuse QKV projections.
        analyze_and_unfuse(self.model)

        # Keep Attention module buffers (_k_scale, _v_scale, etc.) on CPU.
        # Note: This _apply cannot reside in SpyreAttentionImpl, as it is not
        # an nn.Module, but just the attention implementation.
        Attention._apply = lambda self, fn, recurse=True: self  # ty: ignore[invalid-assignment]

        # Store 2-D linear weights transposed (Wᵀ) and matmul directly, so the
        # forward GEMM is the fast `x @ A` instead of `F.linear`'s slow `x @ Aᵀ`
        # (torch-spyre #3512). vLLM linears aren't nn.Linear, so torch-spyre's
        # [1,0]-layout `.to("spyre")` patch skips them; we do the equivalent here
        # in pure PyTorch. Runs after un-fusing (QKV carries its own transpose).
        transpose_linear_weights_for_spyre(self.model)

        # Move layer weights to Spyre device.
        self.model.to(device=self._spyre_device)

        # Pooler / classify heads run on CPU after _SpyreModelWrapper D2H's
        # hidden_states. Cross-encoder rerankers (e.g. bge-reranker) call
        # ClassifierPoolerHead → RobertaClassificationHead (nn.Linear) on those
        # CPU tensors — weights must stay on CPU or F.linear hits a device
        # mismatch (cpu activations × spyre weights). Same pattern as embed
        # poolers (CLS/MEAN/LAST + normalize), which already assume CPU.
        if self.model_config.runner_type == "pooling":
            pinned = []
            if hasattr(self.model, "pooler"):
                self.model.pooler.to("cpu")
                pinned.append("pooler")
            if hasattr(self.model, "classifier"):
                self.model.classifier.to("cpu")
                pinned.append("classifier")
            if pinned:
                logger.info(
                    "Pooling: kept %s on CPU (match D2H hidden_states for classify/score)",
                    ", ".join(pinned),
                )

        logger.info("Spyre-native layer weights moved to %s", self._spyre_device)
        logger.info("Model loaded for Spyre in %.3fs.", time.time() - t0)

        # Patch Llama-4 attention scaling per-instance — must be after load (real
        # modules exist) and before compile (OptimizedModule breaks traversal).
        self._patch_llama4_attn_scale()

        # Pixtral vision 2D-RoPE on CPU (module-level monkeypatch; no-op for
        # non-Pixtral models). Before compile like the llama-4 patch.
        self._patch_pixtral_vision_rope()

        # Mistral-native Pixtral VisionTransformer uses a complex 2D-RoPE (plus an
        # aten::index freqs gather) that Spyre can't run; patch it to an on-card real
        # rotation with an index_select gather. No-op for non-Pixtral / HF-Pixtral
        # models. Before compile like the other patches.
        self._patch_pixtral_vision_rope_vit()

        # Pixtral patch-embedding conv: upstream lowers it to a hand-rolled im2col
        # whose on-device reshape can't restickify for coprime patch grids; run it
        # as a real on-card F.conv2d with channel-on-stick layout. No-op for
        # non-Pixtral models. Before compile like the other patches.
        self._patch_pixtral_patch_conv()

        # Pixtral vision attention: stock SDPA's batch-matmul can't restickify for
        # coprime patch counts; run it on-card with sequence/head padded to the 64
        # stick. No-op for non-Pixtral models. Before compile like the other patches.
        self._patch_pixtral_vision_attention()

        # The projector RMSNorm's on-device reduction stalls compile for the vision
        # feature shape; run it on CPU (tiny, once per image).
        self._offload_pixtral_projector_norm_cpu()

        # PatchMerger's spatial regroup uses aten::im2col (unsupported on Spyre);
        # run the reshuffle on CPU, keep the merging GEMM on-card.
        self._patch_pixtral_patch_merger()

        # Collect RoPE modules for _SpyreModelWrapper to prime (modules() dedupes
        # a shared instance by identity).
        rope_modules = [m for m in self.model.modules() if isinstance(m, _SpyreRotaryMixin)]

        # Compile for Spyre (no-op if enforce_eager=True)
        self._compile_for_spyre()

        # Wrap model so ALL forward() calls to the entire model,
        # for example in execute_model, _dummy_run, etc.,
        # automatically convert Spyre outputs to CPU. This ensures downstream
        # indexing (logits_indices), lm_head (CPU weights), and sampling all
        # receive CPU tensors without needing per-call-site overrides.
        self.model = _SpyreModelWrapper(self.model, self._spyre_device, rope_modules)

    def _compile_for_spyre(self) -> None:
        """Apply torch.compile for Spyre with static shapes.

        Spyre requires static shapes — dynamic shapes (SymInt) are not yet supported.
        We therefore pass `dynamic=False` to torch.compile(...).

        Supported modes:

        - CompilationMode.NONE: eager execution
        - CompilationMode.STOCK_TORCH_COMPILE: whole-model torch.compile
        """
        mode = self.compilation_config.mode
        # vLLM appends "none" to custom_ops whenever backend=="inductor" and
        # mode!=NONE, which routes every CustomOp to forward_native and silently
        # bypasses the Spyre forward_oot implementations. platform.py appends
        # "all" first to prevent that; log the resolved value so a regression in
        # config ordering is visible rather than silent.
        logger.info(
            "Spyre compile config: mode=%s backend=%s custom_ops=%s",
            mode,
            self.compilation_config.backend,
            self.compilation_config.custom_ops,
        )
        if mode not in (CompilationMode.NONE, CompilationMode.STOCK_TORCH_COMPILE):
            raise ValueError(
                f"Unsupported compilation mode {mode} for Spyre. Only "
                f"CompilationMode.NONE and CompilationMode.STOCK_TORCH_COMPILE "
                f"are supported."
            )

        if self.vllm_config.model_config.enforce_eager or mode is CompilationMode.NONE:
            logger.info("Compilation disabled (enforce_eager=True)")
            return

        if _COMPILE_SCOPE != "model":
            self._compile_scoped(_COMPILE_SCOPE)
            return

        # Trigger whole-model compile:
        # a single fullgraph over the entire model using dynamic=False.
        t0 = time.time()
        self.model = torch.compile(
            self.model,
            backend="inductor",
            fullgraph=True,
            dynamic=False,
        )
        logger.info(
            "Compiled model %s as a single graph for Spyre in %.3fs.",
            type(self.get_model()).__name__,
            time.time() - t0,
        )

    def _compile_scoped(self, scope: str) -> None:
        """DIAGNOSTIC: compile only part of the decoder. See _COMPILE_SCOPE."""
        if scope == "none":
            logger.info("SPYRE_COMPILE_SCOPE=none: compiling nothing.")
            return

        def _wrap(m):
            return torch.compile(m, backend=_COMPILE_BACKEND, fullgraph=False, dynamic=False)

        root = self.get_model()
        layer_lists = [
            m
            for m in root.modules()
            if isinstance(m, torch.nn.ModuleList)
            and len(m) > 0
            and type(m[0]).__name__.endswith("DecoderLayer")
        ]
        if not layer_lists:
            raise ValueError(f"SPYRE_COMPILE_SCOPE={scope}: found no decoder layer list")
        # The language model's stack is the longest; a vision tower's is shorter.
        layers = max(layer_lists, key=len)

        n = 0
        if scope.startswith("layers:"):
            for i in range(min(int(scope.split(":", 1)[1]), len(layers))):
                layers[i] = _wrap(layers[i])
                n += 1
        elif scope.startswith("attn:"):
            # Compile one child of self_attn, leaving the connective tissue
            # (qkv split, head reshapes) uncompiled. If no single child breaks,
            # the fault is in that connective tissue rather than in any child.
            attr = scope.split(":", 1)[1]
            seen: set[str] = set()
            for layer in layers:
                parent = getattr(layer, "self_attn", None)
                if parent is None:
                    continue
                seen.update(name for name, _ in parent.named_children())
                sub = getattr(parent, attr, None)
                if sub is not None:
                    setattr(parent, attr, _wrap(sub))
                    n += 1
            if n == 0:
                raise ValueError(
                    f"SPYRE_COMPILE_SCOPE={scope}: self_attn has no child {attr!r}; "
                    f"available: {sorted(seen)}"
                )
        elif scope in ("mlp", "attn"):
            attr = "mlp" if scope == "mlp" else "self_attn"
            for layer in layers:
                sub = getattr(layer, attr, None)
                if sub is not None:
                    setattr(layer, attr, _wrap(sub))
                    n += 1
        else:
            raise ValueError(f"Unknown SPYRE_COMPILE_SCOPE={scope!r}")

        logger.info(
            "SPYRE_COMPILE_SCOPE=%s (backend=%s): compiled %d module(s) of %d decoder layers.",
            scope,
            _COMPILE_BACKEND,
            n,
            len(layers),
        )

    def warming_up_model(self) -> None:
        """Run a dummy forward pass to warm up kernels and optional compile.

        In eager mode, pooling models cap token count
        (``SPYRE_ENCODER_WARMUP_MAX_TOKENS``) and force ``max_num_seqs=1`` to
        stay under the Spyre DMA limit for encoder dummy batches. Compiled
        mode uses the normal warmup size so shapes match torch.compile.
        """
        logger.info("Warming up model...")
        t0 = time.time()
        num_tokens = min(
            max(16, self.max_num_reqs),
            self.scheduler_config.max_num_batched_tokens,
        )
        with _set_spyre_compilation_settings(self.vllm_config):
            use_eager_pooling_warmup = (
                self.model_config.runner_type == "pooling"
                and self.vllm_config.model_config.enforce_eager
            )
            if use_eager_pooling_warmup:
                # Match single-sequence embed metadata; cap tokens for DMA.
                num_tokens = min(num_tokens, SPYRE_ENCODER_WARMUP_MAX_TOKENS)
                saved_max_num_seqs = self.scheduler_config.max_num_seqs
                try:
                    self.scheduler_config.max_num_seqs = 1
                    logger.info(
                        "Pooling warmup (eager): %d tokens, max_num_seqs=1 (was %d)",
                        num_tokens,
                        saved_max_num_seqs,
                    )
                    self._dummy_run(num_tokens)
                finally:
                    self.scheduler_config.max_num_seqs = saved_max_num_seqs
            else:
                self._dummy_run(num_tokens)
        logger.info("Warmup done in %.3fs.", time.time() - t0)

    # --- KV cache allocation ---

    def initialize_kv_cache_tensors(self, kv_cache_config, kernel_block_sizes):
        """Allocate KV cache as lists of individual page tensors on Spyre.

        Each layer gets its own SpyrePagedKVCache(k_pages, v_pages) where each
        is a list of tensors of shape [num_kv_heads, block_size, head_size] on
        the Spyre device. This matches upstream vLLM's paged model but uses
        list indices instead of tensor indices — enabling direct per-page bmm
        without advanced indexing.
        """
        from vllm.v1.worker.utils import bind_kv_cache
        from spyre_inference.v1.attention.backends.spyre_attn import SpyrePagedKVCache

        # Iterate kv_cache_tensors (one entry per physical buffer)
        spec_by_layer = {
            ln: g.kv_cache_spec for g in kv_cache_config.kv_cache_groups for ln in g.layer_names
        }

        # vLLM's `bind_kv_cache` types this dict as `dict[str, torch.Tensor]`,
        # but the matching `SpyreAttentionImpl.forward` consumes the
        # SpyrePagedKVCache — see the suppression on `bind_kv_cache(...)` below.
        kv_caches: dict[str, SpyrePagedKVCache] = {}

        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            # All layers in `shared_by` use the same spec by construction.
            spec = spec_by_layer[kv_cache_tensor.shared_by[0]]
            num_blocks = kv_cache_tensor.size // spec.page_size_bytes

            # Default stickification splits head_size into 64-element sticks.
            # Alternative: stickify block_size or num_kv_heads for different
            # access patterns (would require explicit SpyreTensorLayout).
            k_pages: list[torch.Tensor] = [
                torch.zeros(
                    spec.num_kv_heads,
                    spec.block_size,
                    spec.head_size,
                    dtype=torch.float16,
                    device=self._spyre_device,
                )
                for _ in range(num_blocks)
            ]
            v_pages: list[torch.Tensor] = [
                torch.zeros(
                    spec.num_kv_heads,
                    spec.block_size,
                    spec.head_size,
                    dtype=torch.float16,
                    device=self._spyre_device,
                )
                for _ in range(num_blocks)
            ]

            page_cache = SpyrePagedKVCache(k_pages=k_pages, v_pages=v_pages)
            for layer_name in kv_cache_tensor.shared_by:
                kv_caches[layer_name] = page_cache

        for layer_name, target in self.shared_kv_cache_layers.items():
            kv_caches[layer_name] = kv_caches[target]

        bind_kv_cache(
            kv_caches,  # ty: ignore[invalid-argument-type]
            self.compilation_config.static_forward_context,
            self.kv_caches,
        )
        return kv_caches

    # --- Stubs copied from CPUModelRunner ---
    # These are trivial overrides that GPUModelRunner expects.

    def _init_device_properties(self) -> None:
        # No CUDA/GPU device properties to query for Spyre
        pass

    def _sync_device(self) -> None:
        # Wait for outstanding async H2D from SpyreCpuGpuBuffer.copy_to_gpu
        # (and any other non_blocking copies) before the runner consumes
        # Spyre tensors. torch.spyre is registered by torch-spyre autoload.
        torch.spyre.synchronize(self._spyre_device)

    def get_dp_padding(self, num_tokens: int) -> tuple[int, torch.Tensor | None]:
        return 0, None

    def get_model(self) -> nn.Module:
        # Return the unwrapped model for isinstance checks
        # (e.g. is_text_generation_model in get_supported_tasks).
        model = self.model
        if isinstance(model, _SpyreModelWrapper):
            model = model._model
        # Unwrap torch.compile's OptimizedModule (has _orig_mod attribute)
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        assert isinstance(model, nn.Module)
        return model

    # --- Buffer management ---

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> SpyreCpuGpuBuffer:
        """Create a SpyreCpuGpuBuffer with float tensors on Spyre.

        - Float dtypes: .cpu on CPU, .gpu on Spyre as float16
        - Int/bool dtypes: .gpu aliased to .cpu (stays on CPU)
        """
        if dtype.is_floating_point:
            return SpyreCpuGpuBuffer(
                *size,
                cpu_dtype=dtype,
                gpu_dtype=torch.float16,
                device=self._spyre_device,
                pin_memory=False,
                with_numpy=numpy,
            )
        # Int/bool → CPU-only (aliased)
        return SpyreCpuGpuBuffer(
            *size,
            cpu_dtype=dtype,
            gpu_dtype=dtype,
            device=torch.device("cpu"),
            pin_memory=False,
            with_numpy=numpy,
        )


@contextmanager
def _set_spyre_compilation_settings(config: VllmConfig):
    """Context manager for Spyre-specific compilation settings during warmup.

    Similar to _set_global_compilation_settings in cpu_model_runner.py but
    adapted for Spyre's compilation requirements.
    """
    import torch._inductor.config as torch_inductor_config

    inductor_config = config.compilation_config.inductor_compile_config
    freezing_value = torch_inductor_config.freezing
    try:
        if inductor_config.get("max_autotune", False):
            torch_inductor_config.freezing = True
        yield
    finally:
        torch_inductor_config.freezing = freezing_value
