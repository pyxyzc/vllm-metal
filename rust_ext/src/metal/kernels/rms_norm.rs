//! RMS normalization kernel dispatch.

use crate::metal::buffer::MetalBuffer;
use crate::metal::dispatch::ComputeCommand;

/// Dispatch RMS normalization kernel.
///
/// Computes: output = (input / sqrt(mean(input^2) + eps)) * weight
///
/// # Arguments
/// * `input` - Input tensor [batch_size, hidden_size]
/// * `weight` - Weight tensor [hidden_size]
/// * `output` - Output tensor [batch_size, hidden_size]
/// * `batch_size` - Number of rows
/// * `hidden_size` - Hidden dimension
/// * `eps` - Epsilon for numerical stability
pub fn dispatch_rms_norm(
    input: &MetalBuffer,
    weight: &MetalBuffer,
    output: &MetalBuffer,
    batch_size: i32,
    hidden_size: i32,
    eps: f32,
) -> Result<(), String> {
    // Choose kernel based on dtype
    let kernel_name = match input.dtype() {
        crate::metal::buffer::DType::Float16 | crate::metal::buffer::DType::BFloat16 => "rms_norm_f16",
        crate::metal::buffer::DType::Float32 => "rms_norm_f32",
        _ => return Err(format!("Unsupported dtype for RMS norm: {:?}", input.dtype())),
    };

    // Threads per threadgroup - use 256 for hidden sizes > 256, else hidden_size
    let threads_per_group = (hidden_size as usize).min(256);

    ComputeCommand::new(kernel_name)?
        .buffer(0, input)
        .buffer(1, weight)
        .buffer(2, output)
        .constant(3, &batch_size)
        .constant(4, &hidden_size)
        .constant(5, &eps)
        .grid(1, batch_size as usize, 1)  // One threadgroup per row
        .threadgroup(threads_per_group, 1, 1)
        .execute()
}

/// Dispatch RMS normalization inplace kernel.
///
/// Modifies data in-place.
///
/// # Arguments
/// * `data` - Input/output tensor [batch_size, hidden_size]
/// * `weight` - Weight tensor [hidden_size]
/// * `batch_size` - Number of rows
/// * `hidden_size` - Hidden dimension
/// * `eps` - Epsilon for numerical stability
pub fn dispatch_rms_norm_inplace(
    data: &MetalBuffer,
    weight: &MetalBuffer,
    batch_size: i32,
    hidden_size: i32,
    eps: f32,
) -> Result<(), String> {
    let threads_per_group = (hidden_size as usize).min(256);

    ComputeCommand::new("rms_norm_inplace_f16")?
        .buffer(0, data)
        .buffer(1, weight)
        .constant(2, &batch_size)
        .constant(3, &hidden_size)
        .constant(4, &eps)
        .grid(1, batch_size as usize, 1)
        .threadgroup(threads_per_group, 1, 1)
        .execute()
}

/// Dispatch fused add + RMS normalization kernel.
///
/// Computes:
///   residual = input + residual
///   output = rms_norm(residual)
///
/// # Arguments
/// * `input` - Input tensor [batch_size, hidden_size]
/// * `residual` - Residual tensor [batch_size, hidden_size] (modified in-place)
/// * `weight` - Weight tensor [hidden_size]
/// * `output` - Output tensor [batch_size, hidden_size]
/// * `batch_size` - Number of rows
/// * `hidden_size` - Hidden dimension
/// * `eps` - Epsilon for numerical stability
pub fn dispatch_fused_add_rms_norm(
    input: &MetalBuffer,
    residual: &MetalBuffer,
    weight: &MetalBuffer,
    output: &MetalBuffer,
    batch_size: i32,
    hidden_size: i32,
    eps: f32,
) -> Result<(), String> {
    let threads_per_group = (hidden_size as usize).min(256);

    ComputeCommand::new("fused_add_rms_norm_f16")?
        .buffer(0, input)
        .buffer(1, residual)
        .buffer(2, weight)
        .buffer(3, output)
        .constant(4, &batch_size)
        .constant(5, &hidden_size)
        .constant(6, &eps)
        .grid(1, batch_size as usize, 1)
        .threadgroup(threads_per_group, 1, 1)
        .execute()
}
