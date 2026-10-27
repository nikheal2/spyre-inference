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

"""Attention warmup, shared by both model runners."""

import time
from typing import Any

from vllm.logger import init_logger

from spyre_inference.v1.attention.backends.spyre_attn import (
    SpyreAttentionImpl,
    SpyreAttentionMetadataBuilder,
    SpyrePagedKVCache,
)

logger = init_logger(__name__)


def record_attention_graphs(
    static_ctx: dict[str, Any],
    kv_caches: dict[str, SpyrePagedKVCache],
    builders: dict[str, SpyreAttentionMetadataBuilder],
) -> int:
    """Pre-compile every Spyre attention layer's kernels; returns the graph count."""
    # Every layer keeps its own kernel cache, so each is recorded separately;
    # layers sharing a head configuration trace to the same graph and only
    # the first pays a full Inductor compile.
    t0 = time.time()
    total = 0
    for layer_name, kv_cache in kv_caches.items():
        layer = static_ctx.get(layer_name)
        impl = getattr(layer, "impl", None)
        if not isinstance(impl, SpyreAttentionImpl):
            continue
        builder = builders.get(layer_name)
        # A KV-cache layer on this impl is always in an attention group whose
        # backend builds SpyreAttentionMetadataBuilder, so a miss is a wiring or
        # ordering bug (recording before initialize_attn_backend), not a config.
        assert builder is not None, (
            f"Layer {layer_name} has a Spyre attention impl and a KV cache but no "
            "Spyre metadata builder; initialize_attn_backend() must run first."
        )
        logger.info("Recording attention graphs for layer %s...", layer_name)
        total += impl.record_graphs(layer, kv_cache, builder)
    logger.info(
        "Attention graph recording complete: %d graphs in %.3fs.",
        total,
        time.time() - t0,
    )
    return total
