//! Python-exposed kernel operations using Metal.
//!
//! These functions provide the interface between Python/PyTorch and
//! the Metal kernels for RMS norm, RoPE, and cache operations.

use super::buffer::{DType, MetalBuffer};
use super::attention_ops::ensure_library_loaded;
use super::kernels::{
    dispatch_rms_norm, dispatch_fused_add_rms_norm,
    dispatch_rope_forward, dispatch_rope_inplace, dispatch_rope_decode,
    dispatch_reshape_and_cache, dispatch_copy_blocks,
};
use pyo3::prelude::*;
use std::ffi::c_void;

/// Helper to create a MetalBuffer from tensor attributes.
fn tensor_to_buffer(
    data_ptr: usize,
    shape: Vec<usize>,
    dtype_str: &str,
) -> PyResult<MetalBuffer> {
    let dtype = match dtype_str {
        "torch.float16" | "float16" | "half" => DType::Float16,
        "torch.bfloat16" | "bfloat16" => DType::BFloat16,
        "torch.float32" | "float32" | "float" => DType::Float32,
        "torch.int32" | "int32" | "int" => DType::Int32,
        "torch.int64" | "int64" | "long" => DType::Int64,
        _ => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Unsupported dtype: {}",
                dtype_str
            )))
        }
    };

    let numel: usize = shape.iter().product();
    let size_bytes = numel * dtype.size_bytes();

    unsafe { MetalBuffer::from_ptr(data_ptr as *mut c_void, size_bytes, shape, dtype) }
        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

// ============================================================================
// RMS Normalization
// ============================================================================

/// RMS normalization using Metal kernel.
///
/// Computes: output = (input / sqrt(mean(input^2) + eps)) * weight
///
/// Args:
///     output: Output tensor [batch, hidden_size] (modified in-place)
///     input: Input tensor [batch, hidden_size]
///     weight: Weight tensor [hidden_size]
///     epsilon: Numerical stability constant
#[pyfunction]
#[pyo3(signature = (output, input, weight, epsilon=1e-6))]
pub fn metal_rms_norm(
    py: Python<'_>,
    output: &Bound<'_, PyAny>,
    input: &Bound<'_, PyAny>,
    weight: &Bound<'_, PyAny>,
    epsilon: f32,
) -> PyResult<()> {
    ensure_library_loaded()?;

    // Extract tensor info
    let out_ptr: usize = output.call_method0("data_ptr")?.extract()?;
    let in_ptr: usize = input.call_method0("data_ptr")?.extract()?;
    let w_ptr: usize = weight.call_method0("data_ptr")?.extract()?;

    let in_shape: Vec<usize> = input
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let w_shape: Vec<usize> = weight
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;

    let dtype_str = input.getattr("dtype")?.str()?.to_string();

    // Create Metal buffers
    let in_buf = tensor_to_buffer(in_ptr, in_shape.clone(), &dtype_str)?;
    let w_buf = tensor_to_buffer(w_ptr, w_shape.clone(), &dtype_str)?;
    let out_buf = tensor_to_buffer(out_ptr, in_shape.clone(), &dtype_str)?;

    // Extract dimensions
    let batch_size = if in_shape.len() >= 2 { in_shape[0] } else { 1 };
    let hidden_size = in_shape[in_shape.len() - 1];

    dispatch_rms_norm(
        &in_buf,
        &w_buf,
        &out_buf,
        batch_size as i32,
        hidden_size as i32,
        epsilon,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Fused add + RMS normalization using Metal kernel.
///
/// Computes:
///   residual = input + residual  (in-place)
///   output = rms_norm(residual)
///
/// Args:
///     input: Input tensor [batch, hidden_size]
///     residual: Residual tensor [batch, hidden_size] (modified in-place)
///     weight: Weight tensor [hidden_size]
///     output: Output tensor [batch, hidden_size]
///     epsilon: Numerical stability constant
#[pyfunction]
#[pyo3(signature = (input, residual, weight, output, epsilon=1e-6))]
pub fn metal_fused_add_rms_norm(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    residual: &Bound<'_, PyAny>,
    weight: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    epsilon: f32,
) -> PyResult<()> {
    ensure_library_loaded()?;

    let in_ptr: usize = input.call_method0("data_ptr")?.extract()?;
    let res_ptr: usize = residual.call_method0("data_ptr")?.extract()?;
    let w_ptr: usize = weight.call_method0("data_ptr")?.extract()?;
    let out_ptr: usize = output.call_method0("data_ptr")?.extract()?;

    let in_shape: Vec<usize> = input
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let w_shape: Vec<usize> = weight
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;

    let dtype_str = input.getattr("dtype")?.str()?.to_string();

    let in_buf = tensor_to_buffer(in_ptr, in_shape.clone(), &dtype_str)?;
    let res_buf = tensor_to_buffer(res_ptr, in_shape.clone(), &dtype_str)?;
    let w_buf = tensor_to_buffer(w_ptr, w_shape.clone(), &dtype_str)?;
    let out_buf = tensor_to_buffer(out_ptr, in_shape.clone(), &dtype_str)?;

    let batch_size = if in_shape.len() >= 2 { in_shape[0] } else { 1 };
    let hidden_size = in_shape[in_shape.len() - 1];

    dispatch_fused_add_rms_norm(
        &in_buf,
        &res_buf,
        &w_buf,
        &out_buf,
        batch_size as i32,
        hidden_size as i32,
        epsilon,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

// ============================================================================
// Rotary Position Embeddings (RoPE)
// ============================================================================

/// Apply RoPE using Metal kernel.
///
/// Args:
///     input: Input tensor [batch, seq_len, num_heads, head_dim]
///     cos: Cosine table [max_seq_len, head_dim/2]
///     sin: Sine table [max_seq_len, head_dim/2]
///     output: Output tensor [batch, seq_len, num_heads, head_dim]
///     offset: Position offset for KV cache
#[pyfunction]
#[pyo3(signature = (input, cos, sin, output, offset=0))]
pub fn metal_rope_forward(
    py: Python<'_>,
    input: &Bound<'_, PyAny>,
    cos: &Bound<'_, PyAny>,
    sin: &Bound<'_, PyAny>,
    output: &Bound<'_, PyAny>,
    offset: i32,
) -> PyResult<()> {
    ensure_library_loaded()?;

    let in_ptr: usize = input.call_method0("data_ptr")?.extract()?;
    let cos_ptr: usize = cos.call_method0("data_ptr")?.extract()?;
    let sin_ptr: usize = sin.call_method0("data_ptr")?.extract()?;
    let out_ptr: usize = output.call_method0("data_ptr")?.extract()?;

    let in_shape: Vec<usize> = input
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let cos_shape: Vec<usize> = cos
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;

    let dtype_str = input.getattr("dtype")?.str()?.to_string();

    let in_buf = tensor_to_buffer(in_ptr, in_shape.clone(), &dtype_str)?;
    let cos_buf = tensor_to_buffer(cos_ptr, cos_shape.clone(), "float32")?;
    let sin_buf = tensor_to_buffer(sin_ptr, cos_shape.clone(), "float32")?;
    let out_buf = tensor_to_buffer(out_ptr, in_shape.clone(), &dtype_str)?;

    let batch_size = in_shape[0] as i32;
    let seq_len = in_shape[1] as i32;
    let num_heads = in_shape[2] as i32;
    let head_dim = in_shape[3] as i32;

    dispatch_rope_forward(
        &in_buf, &cos_buf, &sin_buf, &out_buf,
        batch_size, seq_len, num_heads, head_dim, offset,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Apply RoPE in-place using Metal kernel.
#[pyfunction]
#[pyo3(signature = (data, cos, sin, offset=0))]
pub fn metal_rope_inplace(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    cos: &Bound<'_, PyAny>,
    sin: &Bound<'_, PyAny>,
    offset: i32,
) -> PyResult<()> {
    ensure_library_loaded()?;

    let data_ptr: usize = data.call_method0("data_ptr")?.extract()?;
    let cos_ptr: usize = cos.call_method0("data_ptr")?.extract()?;
    let sin_ptr: usize = sin.call_method0("data_ptr")?.extract()?;

    let data_shape: Vec<usize> = data
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let cos_shape: Vec<usize> = cos
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;

    let dtype_str = data.getattr("dtype")?.str()?.to_string();

    let data_buf = tensor_to_buffer(data_ptr, data_shape.clone(), &dtype_str)?;
    let cos_buf = tensor_to_buffer(cos_ptr, cos_shape.clone(), "float32")?;
    let sin_buf = tensor_to_buffer(sin_ptr, cos_shape.clone(), "float32")?;

    let batch_size = data_shape[0] as i32;
    let seq_len = data_shape[1] as i32;
    let num_heads = data_shape[2] as i32;
    let head_dim = data_shape[3] as i32;

    dispatch_rope_inplace(
        &data_buf, &cos_buf, &sin_buf,
        batch_size, seq_len, num_heads, head_dim, offset,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Apply RoPE for decode (single position per sequence).
#[pyfunction]
pub fn metal_rope_decode(
    py: Python<'_>,
    q: &Bound<'_, PyAny>,
    k: &Bound<'_, PyAny>,
    cos: &Bound<'_, PyAny>,
    sin: &Bound<'_, PyAny>,
    positions: &Bound<'_, PyAny>,
) -> PyResult<()> {
    ensure_library_loaded()?;

    let q_ptr: usize = q.call_method0("data_ptr")?.extract()?;
    let k_ptr: usize = k.call_method0("data_ptr")?.extract()?;
    let cos_ptr: usize = cos.call_method0("data_ptr")?.extract()?;
    let sin_ptr: usize = sin.call_method0("data_ptr")?.extract()?;
    let pos_ptr: usize = positions.call_method0("data_ptr")?.extract()?;

    let q_shape: Vec<usize> = q
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let k_shape: Vec<usize> = k
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let cos_shape: Vec<usize> = cos
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let pos_shape: Vec<usize> = positions
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;

    let dtype_str = q.getattr("dtype")?.str()?.to_string();

    let q_buf = tensor_to_buffer(q_ptr, q_shape.clone(), &dtype_str)?;
    let k_buf = tensor_to_buffer(k_ptr, k_shape.clone(), &dtype_str)?;
    let cos_buf = tensor_to_buffer(cos_ptr, cos_shape.clone(), "float32")?;
    let sin_buf = tensor_to_buffer(sin_ptr, cos_shape.clone(), "float32")?;
    let pos_buf = tensor_to_buffer(pos_ptr, pos_shape.clone(), "int32")?;

    let batch_size = q_shape[0] as i32;
    let num_heads = q_shape[1] as i32;
    let num_kv_heads = k_shape[1] as i32;
    let head_dim = q_shape[2] as i32;

    dispatch_rope_decode(
        &q_buf, &k_buf, &cos_buf, &sin_buf, &pos_buf,
        batch_size, num_heads, num_kv_heads, head_dim,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

// ============================================================================
// KV Cache Operations
// ============================================================================

/// Store K/V into paged cache using Metal kernel.
///
/// Args:
///     key: Key tensor [num_tokens, num_kv_heads, head_dim]
///     value: Value tensor [num_tokens, num_kv_heads, head_dim]
///     key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim]
///     value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim]
///     slot_mapping: Slot indices [num_tokens]
#[pyfunction]
pub fn metal_reshape_and_cache(
    py: Python<'_>,
    key: &Bound<'_, PyAny>,
    value: &Bound<'_, PyAny>,
    key_cache: &Bound<'_, PyAny>,
    value_cache: &Bound<'_, PyAny>,
    slot_mapping: &Bound<'_, PyAny>,
) -> PyResult<()> {
    ensure_library_loaded()?;

    let k_ptr: usize = key.call_method0("data_ptr")?.extract()?;
    let v_ptr: usize = value.call_method0("data_ptr")?.extract()?;
    let kc_ptr: usize = key_cache.call_method0("data_ptr")?.extract()?;
    let vc_ptr: usize = value_cache.call_method0("data_ptr")?.extract()?;
    let sm_ptr: usize = slot_mapping.call_method0("data_ptr")?.extract()?;

    let k_shape: Vec<usize> = key
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let kc_shape: Vec<usize> = key_cache
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let sm_shape: Vec<usize> = slot_mapping
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;

    let dtype_str = key.getattr("dtype")?.str()?.to_string();

    let k_buf = tensor_to_buffer(k_ptr, k_shape.clone(), &dtype_str)?;
    let v_buf = tensor_to_buffer(v_ptr, k_shape.clone(), &dtype_str)?;
    let kc_buf = tensor_to_buffer(kc_ptr, kc_shape.clone(), &dtype_str)?;
    let vc_buf = tensor_to_buffer(vc_ptr, kc_shape.clone(), &dtype_str)?;
    let sm_buf = tensor_to_buffer(sm_ptr, sm_shape.clone(), "int32")?;

    let num_tokens = k_shape[0] as i32;
    let num_kv_heads = k_shape[1] as i32;
    let head_dim = k_shape[2] as i32;
    let block_size = kc_shape[1] as i32;

    dispatch_reshape_and_cache(
        &k_buf, &v_buf, &kc_buf, &vc_buf, &sm_buf,
        num_tokens, num_kv_heads, head_dim, block_size,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// Copy cache blocks using Metal kernel.
#[pyfunction]
pub fn metal_copy_blocks(
    py: Python<'_>,
    key_cache: &Bound<'_, PyAny>,
    value_cache: &Bound<'_, PyAny>,
    block_mapping: &Bound<'_, PyAny>,
) -> PyResult<()> {
    ensure_library_loaded()?;

    let kc_ptr: usize = key_cache.call_method0("data_ptr")?.extract()?;
    let vc_ptr: usize = value_cache.call_method0("data_ptr")?.extract()?;
    let bm_ptr: usize = block_mapping.call_method0("data_ptr")?.extract()?;

    let kc_shape: Vec<usize> = key_cache
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;
    let bm_shape: Vec<usize> = block_mapping
        .getattr("shape")?
        .iter()?
        .map(|x| x.and_then(|v| v.extract()))
        .collect::<PyResult<_>>()?;

    let dtype_str = key_cache.getattr("dtype")?.str()?.to_string();

    let kc_buf = tensor_to_buffer(kc_ptr, kc_shape.clone(), &dtype_str)?;
    let vc_buf = tensor_to_buffer(vc_ptr, kc_shape.clone(), &dtype_str)?;
    let bm_buf = tensor_to_buffer(bm_ptr, bm_shape.clone(), "int32")?;

    let num_pairs = bm_shape[0] as i32;
    let block_size = kc_shape[1] as i32;
    let num_kv_heads = kc_shape[2] as i32;
    let head_dim = kc_shape[3] as i32;

    dispatch_copy_blocks(
        &kc_buf, &vc_buf, &bm_buf,
        num_pairs, block_size, num_kv_heads, head_dim,
    )
    .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}
