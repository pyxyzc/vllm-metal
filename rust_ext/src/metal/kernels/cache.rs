//! KV cache operations kernel dispatch.

use crate::metal::buffer::MetalBuffer;
use crate::metal::dispatch::ComputeCommand;

/// Dispatch reshape_and_cache kernel.
///
/// Stores new K/V tokens into paged cache at specified slots.
///
/// # Arguments
/// * `key` - Key tensor [num_tokens, num_kv_heads, head_dim]
/// * `value` - Value tensor [num_tokens, num_kv_heads, head_dim]
/// * `key_cache` - Key cache [num_blocks, block_size, num_kv_heads, head_dim]
/// * `value_cache` - Value cache [num_blocks, block_size, num_kv_heads, head_dim]
/// * `slot_mapping` - Slot indices for each token [num_tokens]
/// * `num_tokens` - Number of tokens to cache
/// * `num_kv_heads` - Number of KV heads
/// * `head_dim` - Head dimension
/// * `block_size` - Cache block size
pub fn dispatch_reshape_and_cache(
    key: &MetalBuffer,
    value: &MetalBuffer,
    key_cache: &MetalBuffer,
    value_cache: &MetalBuffer,
    slot_mapping: &MetalBuffer,
    num_tokens: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
) -> Result<(), String> {
    let kernel_name = match key.dtype() {
        crate::metal::buffer::DType::Float16 | crate::metal::buffer::DType::BFloat16 => "reshape_and_cache_f16",
        crate::metal::buffer::DType::Float32 => "reshape_and_cache_f32",
        _ => return Err(format!("Unsupported dtype for reshape_and_cache: {:?}", key.dtype())),
    };

    ComputeCommand::new(kernel_name)?
        .buffer(0, key)
        .buffer(1, value)
        .buffer(2, key_cache)
        .buffer(3, value_cache)
        .buffer(4, slot_mapping)
        .constant(5, &num_tokens)
        .constant(6, &num_kv_heads)
        .constant(7, &head_dim)
        .constant(8, &block_size)
        .grid(num_tokens as usize, num_kv_heads as usize, 1)
        .threadgroup(1, 1, 1)
        .execute()
}

/// Dispatch copy_blocks kernel.
///
/// Copies cache blocks between sequences (used for beam search).
///
/// # Arguments
/// * `key_cache` - Key cache [num_blocks, block_size, num_kv_heads, head_dim]
/// * `value_cache` - Value cache [num_blocks, block_size, num_kv_heads, head_dim]
/// * `block_mapping` - (src_block, dst_block) pairs
/// * `num_pairs` - Number of block pairs to copy
/// * `block_size` - Cache block size
/// * `num_kv_heads` - Number of KV heads
/// * `head_dim` - Head dimension
pub fn dispatch_copy_blocks(
    key_cache: &MetalBuffer,
    value_cache: &MetalBuffer,
    block_mapping: &MetalBuffer,
    num_pairs: i32,
    block_size: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> Result<(), String> {
    ComputeCommand::new("copy_blocks_f16")?
        .buffer(0, key_cache)
        .buffer(1, value_cache)
        .buffer(2, block_mapping)
        .constant(3, &num_pairs)
        .constant(4, &block_size)
        .constant(5, &num_kv_heads)
        .constant(6, &head_dim)
        .grid(num_pairs as usize, 1, 1)
        .threadgroup(1, 1, 1)
        .execute()
}

/// Dispatch gather_cached kernel.
///
/// Gathers K/V from cache for specific positions.
///
/// # Arguments
/// * `cache` - Cache tensor [num_blocks, block_size, num_kv_heads, head_dim]
/// * `output` - Output tensor [num_tokens, num_kv_heads, head_dim]
/// * `slot_mapping` - Slot indices for each token [num_tokens]
/// * `num_tokens` - Number of tokens to gather
/// * `num_kv_heads` - Number of KV heads
/// * `head_dim` - Head dimension
/// * `block_size` - Cache block size
pub fn dispatch_gather_cached(
    cache: &MetalBuffer,
    output: &MetalBuffer,
    slot_mapping: &MetalBuffer,
    num_tokens: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
) -> Result<(), String> {
    ComputeCommand::new("gather_cached_f16")?
        .buffer(0, cache)
        .buffer(1, output)
        .buffer(2, slot_mapping)
        .constant(3, &num_tokens)
        .constant(4, &num_kv_heads)
        .constant(5, &head_dim)
        .constant(6, &block_size)
        .grid(num_tokens as usize, num_kv_heads as usize, 1)
        .threadgroup(1, 1, 1)
        .execute()
}

/// Dispatch init_cache_block kernel.
///
/// Zero-initializes a cache block.
pub fn dispatch_init_cache_block(
    key_cache: &MetalBuffer,
    value_cache: &MetalBuffer,
    block_idx: i32,
    block_size: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> Result<(), String> {
    let block_elements = (block_size * num_kv_heads * head_dim) as usize;

    ComputeCommand::new("init_cache_block_f16")?
        .buffer(0, key_cache)
        .buffer(1, value_cache)
        .constant(2, &block_idx)
        .constant(3, &block_size)
        .constant(4, &num_kv_heads)
        .constant(5, &head_dim)
        .grid(block_elements, 1, 1)
        .threadgroup(256.min(block_elements), 1, 1)
        .execute()
}
