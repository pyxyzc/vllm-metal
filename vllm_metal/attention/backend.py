# SPDX-License-Identifier: Apache-2.0
"""Metal attention backend for vLLM V1 architecture.

This backend uses Rust/Metal kernels via unified memory, avoiding the MPS backend.
PyTorch tensors are kept on CPU, but the underlying memory is directly accessible
by Metal GPU kernels on Apple Silicon's unified memory architecture.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
import torch.nn.functional as nnf  # noqa: N812
from vllm.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionType,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.attention.backends.utils import (
    AttentionCGSupport,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.kv_cache_interface import AttentionSpec

logger = init_logger(__name__)

# Try to import Rust Metal extensions
# NOTE: We don't initialize Metal at import time to avoid issues with multiprocessing.
# Metal contexts cannot survive a fork, so we defer initialization to first use.
try:
    import vllm_metal_rust

    METAL_RUST_IMPORTABLE = True
except ImportError:
    METAL_RUST_IMPORTABLE = False
    logger.warning("Rust Metal extensions not available, using PyTorch SDPA fallback")

# Lazy initialization flag - will be set on first use in worker process
_metal_initialized = False
_metal_available = False


def _ensure_metal_initialized():
    """Initialize Metal context lazily in the worker process."""
    global _metal_initialized, _metal_available
    if _metal_initialized:
        return _metal_available

    _metal_initialized = True
    if not METAL_RUST_IMPORTABLE:
        _metal_available = False
        return False

    try:
        _metal_available = vllm_metal_rust.is_metal_available()
        if _metal_available:
            device_name, max_threads, max_mem = vllm_metal_rust.metal_device_info()
            logger.info(f"Metal Rust backend initialized: {device_name}")
    except Exception as e:
        logger.warning(f"Failed to initialize Metal: {e}")
        _metal_available = False

    return _metal_available


@dataclass
class MetalAttentionMetadata:
    """V1 attention metadata for Metal backend.

    Follows the same pattern as TritonAttentionMetadata.
    """

    num_actual_tokens: int
    max_query_len: int
    query_start_loc: torch.Tensor
    max_seq_len: int
    seq_lens: torch.Tensor
    block_table: torch.Tensor
    slot_mapping: torch.Tensor

    # For cascade attention (prefix sharing)
    use_cascade: bool = False
    common_prefix_len: int = 0


class MetalAttentionMetadataBuilder(AttentionMetadataBuilder[MetalAttentionMetadata]):
    """V1 builder for Metal attention metadata."""

    # Metal does not support CUDA graphs
    _cudagraph_support: ClassVar[AttentionCGSupport] = AttentionCGSupport.NEVER

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

        self.block_size = kv_cache_spec.block_size
        model_config = vllm_config.model_config
        self.num_heads_q = model_config.get_num_attention_heads(
            vllm_config.parallel_config
        )
        self.num_heads_kv = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.head_dim = model_config.get_head_size()

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> MetalAttentionMetadata:
        """Build attention metadata from common metadata."""
        return MetalAttentionMetadata(
            num_actual_tokens=common_attn_metadata.num_actual_tokens,
            max_query_len=common_attn_metadata.max_query_len,
            query_start_loc=common_attn_metadata.query_start_loc,
            max_seq_len=common_attn_metadata.max_seq_len,
            seq_lens=common_attn_metadata.seq_lens,
            block_table=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping,
            use_cascade=common_prefix_len > 0,
            common_prefix_len=common_prefix_len,
        )


class MetalAttentionImpl(AttentionImpl):
    """Metal attention implementation using PyTorch SDPA.

    Uses PyTorch's scaled_dot_product_attention which is optimized
    for Apple Silicon via MPS backend.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        blocksparse_params: dict | None = None,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.sliding_window = sliding_window
        self.alibi_slopes = alibi_slopes
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap
        self.attn_type = attn_type
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        if blocksparse_params is not None:
            logger.warning("Block sparse attention not supported on Metal, ignoring")

        if alibi_slopes is not None:
            self.alibi_slopes_tensor = torch.tensor(alibi_slopes, dtype=torch.float32)
        else:
            self.alibi_slopes_tensor = None

    def forward(
        self,
        layer: "torch.nn.Module",
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
        output: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Forward pass for Metal attention.

        Args:
            layer: The Attention layer (required by vLLM interface)
            query: Query tensor [num_tokens, num_heads * head_size]
            key: Key tensor [num_tokens, num_kv_heads * head_size]
            value: Value tensor [num_tokens, num_kv_heads * head_size]
            kv_cache: KV cache tensor
            attn_metadata: Attention metadata
            output: Optional output tensor

        Returns:
            Output tensor [num_tokens, num_heads * head_size]
        """
        num_tokens = query.shape[0]
        original_device = query.device

        # Reshape Q, K, V: [num_tokens, num_heads, head_size]
        query = query.view(num_tokens, self.num_heads, self.head_size)
        key = key.view(num_tokens, self.num_kv_heads, self.head_size)
        value = value.view(num_tokens, self.num_kv_heads, self.head_size)

        # Store new KV to cache
        if kv_cache is not None and attn_metadata.slot_mapping is not None:
            self._store_kv_cache(key, value, kv_cache, attn_metadata.slot_mapping)

        # Determine if this is prefill or decode based on query length
        is_prefill = attn_metadata.max_query_len > 1

        if is_prefill:
            out = self._prefill_attention(query, key, value, attn_metadata)
        else:
            out = self._decode_attention(query, key, value, kv_cache, attn_metadata)

        # Reshape output: [num_tokens, num_heads * head_size]
        # Use reshape instead of view to handle non-contiguous tensors from SDPA
        out = out.reshape(num_tokens, self.num_heads * self.head_size)

        # Ensure output is on the same device as input
        if out.device != original_device:
            out = out.to(original_device)

        if output is not None:
            output.copy_(out)
            return output
        return out

    def _prefill_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
    ) -> torch.Tensor:
        """Compute attention for prefill phase.

        For prefill, use the incoming K, V directly with causal masking.
        Uses Rust/Metal kernel when available, falls back to PyTorch SDPA.
        """
        # Input: [num_tokens, num_heads, head_size]
        # Metal SDPA expects: [num_queries, num_heads, head_dim] for Q
        #                     [num_queries, seq_len, num_kv_heads, head_dim] for K, V

        # Expand KV for GQA if needed
        if self.num_queries_per_kv > 1:
            key = key.repeat_interleave(self.num_queries_per_kv, dim=1)
            value = value.repeat_interleave(self.num_queries_per_kv, dim=1)

        # Metal kernels support head_dim of 64, 128, 256
        # Fall back to PyTorch SDPA for unsupported sizes
        head_dim = query.shape[-1]
        metal_supported_head_dims = (64, 128, 256)

        # NOTE: Metal SDPA kernel has issues - using PyTorch SDPA fallback for now
        # TODO: Fix Metal SDPA kernel dispatch (threadgroup vs thread indexing)
        # if _ensure_metal_initialized() and head_dim in metal_supported_head_dims:
        #     # Use Rust/Metal kernel with zero-copy from CPU tensors
        #     ...

        # Use PyTorch SDPA (well-optimized for MPS)
        # SDPA expects: [batch, num_heads, seq_len, head_size]
        query = query.transpose(0, 1).unsqueeze(0)  # [1, num_heads, seq, head]
        key = key.transpose(0, 1).unsqueeze(0)
        value = value.transpose(0, 1).unsqueeze(0)

        out = nnf.scaled_dot_product_attention(
            query,
            key,
            value,
            is_causal=True,
            scale=self.scale,
        )

        # Reshape: [1, num_heads, seq, head] -> [seq, num_heads, head]
        out = out.squeeze(0).transpose(0, 1)
        return out

    def _decode_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: MetalAttentionMetadata,
    ) -> torch.Tensor:
        """Compute attention for decode phase.

        For decode, read K, V from cache using block tables.
        Uses Rust/Metal paged attention kernel when available.
        """
        batch_size = query.shape[0]

        # Metal kernels support head_dim of 64, 128, 256
        # Fall back to PyTorch SDPA for unsupported sizes
        head_dim = query.shape[-1]
        metal_supported_head_dims = (64, 128, 256)

        # NOTE: Metal paged attention kernel has issues - using PyTorch fallback for now
        # TODO: Fix Metal paged attention kernel dispatch (threadgroup vs thread indexing)
        # if (
        #     _ensure_metal_initialized()
        #     and kv_cache is not None
        #     and head_dim in metal_supported_head_dims
        # ):
        #     # Use Rust/Metal paged attention kernel
        #     ...

        # Use PyTorch loop-based decode
        outputs = []

        for i in range(batch_size):
            # Get block table for this sequence
            block_table = attn_metadata.block_table[i]
            seq_len = int(attn_metadata.seq_lens[i].item())

            # Gather KV from cache
            k_cache, v_cache = self._gather_from_cache(kv_cache, block_table, seq_len)

            # Single query attention: [1, num_heads, head_size]
            q = query[i : i + 1]

            # Expand KV for GQA
            if self.num_queries_per_kv > 1:
                k_cache = k_cache.repeat_interleave(self.num_queries_per_kv, dim=1)
                v_cache = v_cache.repeat_interleave(self.num_queries_per_kv, dim=1)

            # SDPA: [1, num_heads, 1, head] vs [1, num_heads, seq, head]
            q = q.transpose(0, 1).unsqueeze(0)  # [1, heads, 1, head]
            k = k_cache.transpose(0, 1).unsqueeze(0)  # [1, heads, seq, head]
            v = v_cache.transpose(0, 1).unsqueeze(0)

            out = nnf.scaled_dot_product_attention(
                q,
                k,
                v,
                is_causal=False,  # No causal for decode
                scale=self.scale,
            )

            # [1, heads, 1, head] -> [1, heads, head]
            out = out.squeeze(0).transpose(0, 1)
            outputs.append(out)

        return torch.cat(outputs, dim=0)

    def _store_kv_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        """Store key and value into KV cache using slot mapping.

        KV cache layout: [num_blocks, 2, block_size, num_kv_heads, head_size]
        Uses vectorized operations for better performance.
        """
        block_size = kv_cache.shape[2]

        # Filter out padding slots (slot < 0)
        valid_mask = slot_mapping >= 0
        if not valid_mask.any():
            return

        valid_slots = slot_mapping[valid_mask]
        valid_keys = key[valid_mask]
        valid_values = value[valid_mask]

        # Compute block indices and offsets vectorized
        block_indices = valid_slots // block_size
        block_offsets = valid_slots % block_size

        # Use advanced indexing to scatter into cache
        kv_cache[block_indices, 0, block_offsets] = valid_keys
        kv_cache[block_indices, 1, block_offsets] = valid_values

    def _gather_from_cache(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather keys and values from paged KV cache.

        Uses vectorized gather for better performance.

        Returns:
            (keys, values) both with shape [context_len, num_kv_heads, head_size]
        """
        block_size = kv_cache.shape[2]
        num_kv_heads = kv_cache.shape[3]
        head_size = kv_cache.shape[4]
        num_blocks_needed = (context_len + block_size - 1) // block_size

        # Vectorized approach: compute all slot positions and gather at once
        # Create flat indices for all tokens we need
        slot_positions = torch.arange(context_len, device=kv_cache.device)
        block_indices = slot_positions // block_size
        block_offsets = slot_positions % block_size

        # Get physical block indices from block table
        physical_blocks = block_table[block_indices]

        # Gather all keys and values at once using advanced indexing
        keys = kv_cache[physical_blocks, 0, block_offsets]
        values = kv_cache[physical_blocks, 1, block_offsets]

        return keys, values

    def _gather_from_cache_loop(
        self,
        kv_cache: torch.Tensor,
        block_table: torch.Tensor,
        context_len: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Legacy loop-based gather (kept for reference).

        Returns:
            (keys, values) both with shape [context_len, num_kv_heads, head_size]
        """
        block_size = kv_cache.shape[2]
        num_blocks_needed = (context_len + block_size - 1) // block_size

        keys = []
        values = []
        tokens_gathered = 0

        for block_idx in range(num_blocks_needed):
            if block_idx >= len(block_table):
                break

            physical_block = int(block_table[block_idx].item())
            tokens_in_block = min(block_size, context_len - tokens_gathered)

            k_block = kv_cache[physical_block, 0, :tokens_in_block]
            v_block = kv_cache[physical_block, 1, :tokens_in_block]

            keys.append(k_block)
            values.append(v_block)
            tokens_gathered += tokens_in_block

        return torch.cat(keys, dim=0), torch.cat(values, dim=0)


class MetalAttentionBackend(AttentionBackend):
    """V1 attention backend for Apple Metal."""

    accept_output_buffer: bool = False

    @staticmethod
    def get_name() -> str:
        return "METAL"

    @staticmethod
    def get_impl_cls() -> type[AttentionImpl]:
        return MetalAttentionImpl

    @staticmethod
    def get_metadata_cls() -> type[MetalAttentionMetadata]:
        return MetalAttentionMetadata

    @staticmethod
    def get_builder_cls() -> type[MetalAttentionMetadataBuilder]:
        return MetalAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> tuple[int, ...]:
        """KV cache shape: [num_blocks, 2, block_size, num_kv_heads, head_size]"""
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_indices = src_to_dst[:, 0]
        dst_indices = src_to_dst[:, 1]
        dst_kv_cache[dst_indices] = src_kv_cache[src_indices]

    @staticmethod
    def copy_blocks(
        kv_caches: list[torch.Tensor],
        src_to_dsts: torch.Tensor,
    ) -> None:
        for kv_cache in kv_caches:
            src_indices = src_to_dsts[:, 0]
            dst_indices = src_to_dsts[:, 1]
            kv_cache[dst_indices] = kv_cache[src_indices]

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        return [64, 80, 96, 112, 128, 192, 256]
