//! RoPE kernel dispatch for rotary position embeddings.

use crate::metal::buffer::MetalBuffer;
use crate::metal::dispatch::ComputeCommand;

/// Dispatch RoPE forward kernel.
///
/// Applies rotary position embeddings to input tensor.
///
/// # Arguments
/// * `input` - Input tensor [batch_size, seq_len, num_heads, head_dim]
/// * `cos` - Cosine table [max_seq_len, head_dim/2]
/// * `sin` - Sine table [max_seq_len, head_dim/2]
/// * `output` - Output tensor [batch_size, seq_len, num_heads, head_dim]
/// * `batch_size` - Batch size
/// * `seq_len` - Sequence length
/// * `num_heads` - Number of attention heads
/// * `head_dim` - Head dimension
/// * `offset` - Position offset (for KV cache)
pub fn dispatch_rope_forward(
    input: &MetalBuffer,
    cos: &MetalBuffer,
    sin: &MetalBuffer,
    output: &MetalBuffer,
    batch_size: i32,
    seq_len: i32,
    num_heads: i32,
    head_dim: i32,
    offset: i32,
) -> Result<(), String> {
    let kernel_name = match input.dtype() {
        crate::metal::buffer::DType::Float16 | crate::metal::buffer::DType::BFloat16 => "rope_forward_f16",
        crate::metal::buffer::DType::Float32 => "rope_forward_f32",
        _ => return Err(format!("Unsupported dtype for RoPE: {:?}", input.dtype())),
    };

    ComputeCommand::new(kernel_name)?
        .buffer(0, input)
        .buffer(1, cos)
        .buffer(2, sin)
        .buffer(3, output)
        .constant(4, &batch_size)
        .constant(5, &seq_len)
        .constant(6, &num_heads)
        .constant(7, &head_dim)
        .constant(8, &offset)
        .grid(batch_size as usize, seq_len as usize, num_heads as usize)
        .threadgroup(1, 1, 1)
        .execute()
}

/// Dispatch RoPE inplace kernel.
///
/// Applies rotary position embeddings in-place.
pub fn dispatch_rope_inplace(
    data: &MetalBuffer,
    cos: &MetalBuffer,
    sin: &MetalBuffer,
    batch_size: i32,
    seq_len: i32,
    num_heads: i32,
    head_dim: i32,
    offset: i32,
) -> Result<(), String> {
    ComputeCommand::new("rope_inplace_f16")?
        .buffer(0, data)
        .buffer(1, cos)
        .buffer(2, sin)
        .constant(3, &batch_size)
        .constant(4, &seq_len)
        .constant(5, &num_heads)
        .constant(6, &head_dim)
        .constant(7, &offset)
        .grid(batch_size as usize, seq_len as usize, num_heads as usize)
        .threadgroup(1, 1, 1)
        .execute()
}

/// Dispatch RoPE decode kernel (optimized for single position).
///
/// Applies RoPE to both Q and K for decode phase.
///
/// # Arguments
/// * `q` - Query tensor [batch_size, num_heads, head_dim]
/// * `k` - Key tensor [batch_size, num_kv_heads, head_dim]
/// * `cos` - Cosine table [max_seq_len, head_dim/2]
/// * `sin` - Sine table [max_seq_len, head_dim/2]
/// * `positions` - Position indices [batch_size]
/// * `batch_size` - Batch size
/// * `num_heads` - Number of query heads
/// * `num_kv_heads` - Number of KV heads
/// * `head_dim` - Head dimension
pub fn dispatch_rope_decode(
    q: &MetalBuffer,
    k: &MetalBuffer,
    cos: &MetalBuffer,
    sin: &MetalBuffer,
    positions: &MetalBuffer,
    batch_size: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
) -> Result<(), String> {
    // Use max of num_heads and num_kv_heads for grid height
    let max_heads = num_heads.max(num_kv_heads) as usize;

    ComputeCommand::new("rope_decode_f16")?
        .buffer(0, q)
        .buffer(1, k)
        .buffer(2, cos)
        .buffer(3, sin)
        .buffer(4, positions)
        .constant(5, &batch_size)
        .constant(6, &num_heads)
        .constant(7, &num_kv_heads)
        .constant(8, &head_dim)
        .grid(batch_size as usize, max_heads, 1)
        .threadgroup(1, 1, 1)
        .execute()
}

/// Dispatch precompute_freqs kernel.
///
/// Precomputes cos/sin tables for RoPE.
pub fn dispatch_precompute_freqs(
    cos_out: &MetalBuffer,
    sin_out: &MetalBuffer,
    max_seq_len: i32,
    head_dim: i32,
    base: f32,
) -> Result<(), String> {
    let half_dim = head_dim / 2;

    ComputeCommand::new("precompute_freqs")?
        .buffer(0, cos_out)
        .buffer(1, sin_out)
        .constant(2, &max_seq_len)
        .constant(3, &head_dim)
        .constant(4, &base)
        .grid(max_seq_len as usize, half_dim as usize, 1)
        .threadgroup(1, 1, 1)
        .execute()
}
