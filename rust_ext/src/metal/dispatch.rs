//! Kernel dispatch utilities for encoding and running compute shaders.

use super::device::{
    MetalContext, MTLCommandBufferTrait, MTLCommandQueueTrait,
    MTLComputeCommandEncoderTrait, MTLCommandEncoderTrait,
};
use super::buffer::MetalBuffer;
use super::pipeline::{global_pipeline_cache, ComputePipeline};
use objc2_metal::MTLSize;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Arc;

/// Command encoder wrapper for building compute commands.
pub struct ComputeCommand<'a> {
    ctx: Arc<MetalContext>,
    pipeline: Arc<ComputePipeline>,
    buffers: Vec<&'a MetalBuffer>,
    constants: Vec<(usize, Vec<u8>)>, // (index, bytes)
    grid_size: MTLSize,
    threadgroup_size: MTLSize,
}

impl<'a> ComputeCommand<'a> {
    /// Create a new compute command for the given kernel.
    pub fn new(kernel_name: &str) -> Result<Self, String> {
        let ctx = MetalContext::get();
        let pipeline = global_pipeline_cache().get(kernel_name)?;

        Ok(Self {
            ctx,
            pipeline,
            buffers: Vec::new(),
            constants: Vec::new(),
            grid_size: MTLSize { width: 1, height: 1, depth: 1 },
            threadgroup_size: MTLSize { width: 1, height: 1, depth: 1 },
        })
    }

    /// Add a buffer argument at the specified index.
    pub fn buffer(mut self, index: usize, buffer: &'a MetalBuffer) -> Self {
        // Ensure buffers vec is large enough
        while self.buffers.len() <= index {
            // This is a placeholder - will be filled properly
            self.buffers.push(buffer);
        }
        self.buffers[index] = buffer;
        self
    }

    /// Add a constant value at the specified index.
    pub fn constant<T: Copy>(mut self, index: usize, value: &T) -> Self {
        let bytes = unsafe {
            std::slice::from_raw_parts(
                value as *const T as *const u8,
                std::mem::size_of::<T>(),
            )
        };
        self.constants.push((index, bytes.to_vec()));
        self
    }

    /// Set the grid size (total number of threads).
    pub fn grid(mut self, width: usize, height: usize, depth: usize) -> Self {
        self.grid_size = MTLSize { width, height, depth };
        self
    }

    /// Set the threadgroup size.
    pub fn threadgroup(mut self, width: usize, height: usize, depth: usize) -> Self {
        self.threadgroup_size = MTLSize { width, height, depth };
        self
    }

    /// Execute the command synchronously.
    pub fn execute(self) -> Result<(), String> {
        // Create command buffer
        let cmd_buffer = self.ctx.command_queue()
            .commandBuffer()
            .ok_or("Failed to create command buffer")?;

        // Create compute encoder
        let encoder = cmd_buffer
            .computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        // Set pipeline state
        encoder.setComputePipelineState(self.pipeline.state());

        // Set buffer arguments
        for (index, buffer) in self.buffers.iter().enumerate() {
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(buffer.raw_buffer()),
                    0,
                    index,
                );
            }
        }

        // Set constant arguments
        for (index, bytes) in &self.constants {
            if let Some(ptr) = NonNull::new(bytes.as_ptr() as *mut c_void) {
                unsafe {
                    encoder.setBytes_length_atIndex(
                        ptr,
                        bytes.len(),
                        *index,
                    );
                }
            }
        }

        // Dispatch threadgroups (grid_size is number of threadgroups, not total threads)
        // Our kernels use threadgroup_position_in_grid, so we need threadgroup dispatch
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            self.grid_size,
            self.threadgroup_size,
        );

        // End encoding and commit
        encoder.endEncoding();
        cmd_buffer.commit();
        unsafe { cmd_buffer.waitUntilCompleted(); }

        Ok(())
    }

    /// Execute the command asynchronously (returns immediately).
    pub fn execute_async(self) -> Result<(), String> {
        // Create command buffer
        let cmd_buffer = self.ctx.command_queue()
            .commandBuffer()
            .ok_or("Failed to create command buffer")?;

        // Create compute encoder
        let encoder = cmd_buffer
            .computeCommandEncoder()
            .ok_or("Failed to create compute encoder")?;

        // Set pipeline state
        encoder.setComputePipelineState(self.pipeline.state());

        // Set buffer arguments
        for (index, buffer) in self.buffers.iter().enumerate() {
            unsafe {
                encoder.setBuffer_offset_atIndex(
                    Some(buffer.raw_buffer()),
                    0,
                    index,
                );
            }
        }

        // Set constant arguments
        for (index, bytes) in &self.constants {
            if let Some(ptr) = NonNull::new(bytes.as_ptr() as *mut c_void) {
                unsafe {
                    encoder.setBytes_length_atIndex(
                        ptr,
                        bytes.len(),
                        *index,
                    );
                }
            }
        }

        // Dispatch threadgroups
        encoder.dispatchThreadgroups_threadsPerThreadgroup(
            self.grid_size,
            self.threadgroup_size,
        );

        // End encoding and commit (but don't wait)
        encoder.endEncoding();
        cmd_buffer.commit();

        Ok(())
    }
}

/// Dispatch helper for SDPA kernel.
pub fn dispatch_sdpa(
    queries: &MetalBuffer,
    keys: &MetalBuffer,
    values: &MetalBuffer,
    output: &MetalBuffer,
    num_queries: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    seq_len: i32,
    scale: f32,
) -> Result<(), String> {
    // Choose kernel based on head_dim
    let kernel_name = match head_dim {
        64 => "sdpa_vector_f16_64",
        96 => "sdpa_vector_f16_96",
        128 => "sdpa_vector_f16_128",
        _ => return Err(format!("Unsupported head_dim: {}", head_dim)),
    };

    // Attention params struct (must match Metal shader)
    #[repr(C)]
    #[derive(Copy, Clone)]
    struct AttentionParams {
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        seq_len: i32,
        num_queries: i32,
        scale: f32,
        gqa_ratio: i32,
    }

    let gqa_ratio = num_heads / num_kv_heads;
    let params = AttentionParams {
        num_heads,
        num_kv_heads,
        head_dim,
        seq_len,
        num_queries,
        scale,
        gqa_ratio,
    };

    // Grid: one threadgroup per (query, head)
    // Threadgroup: 32 threads (one simdgroup)
    let grid_width = num_queries as usize;
    let grid_height = num_heads as usize;

    ComputeCommand::new(kernel_name)?
        .buffer(0, queries)
        .buffer(1, keys)
        .buffer(2, values)
        .buffer(3, output)
        .constant(4, &params)
        .grid(grid_width, grid_height, 1)
        .threadgroup(32, 1, 1)
        .execute()
}

/// Dispatch helper for paged attention kernel.
pub fn dispatch_paged_attention(
    queries: &MetalBuffer,
    key_cache: &MetalBuffer,
    value_cache: &MetalBuffer,
    block_tables: &MetalBuffer,
    seq_lens: &MetalBuffer,
    output: &MetalBuffer,
    batch_size: i32,
    num_heads: i32,
    num_kv_heads: i32,
    head_dim: i32,
    block_size: i32,
    num_blocks: i32,
    scale: f32,
    max_blocks_per_seq: i32,
) -> Result<(), String> {
    // Choose kernel based on head_dim and block_size
    let kernel_name = match (head_dim, block_size) {
        (64, 16) => "paged_attention_f16_64_16",
        (128, 16) => "paged_attention_f16_128_16",
        _ => return Err(format!("Unsupported head_dim={}, block_size={}", head_dim, block_size)),
    };

    #[repr(C)]
    #[derive(Copy, Clone)]
    struct PagedAttentionParams {
        num_heads: i32,
        num_kv_heads: i32,
        head_dim: i32,
        block_size: i32,
        num_blocks: i32,
        scale: f32,
        gqa_ratio: i32,
    }

    let gqa_ratio = num_heads / num_kv_heads;
    let params = PagedAttentionParams {
        num_heads,
        num_kv_heads,
        head_dim,
        block_size,
        num_blocks,
        scale,
        gqa_ratio,
    };

    // Grid: one threadgroup per (batch, head)
    let grid_width = batch_size as usize;
    let grid_height = num_heads as usize;

    ComputeCommand::new(kernel_name)?
        .buffer(0, queries)
        .buffer(1, key_cache)
        .buffer(2, value_cache)
        .buffer(3, block_tables)
        .buffer(4, seq_lens)
        .buffer(5, output)
        .constant(6, &params)
        .constant(7, &max_blocks_per_seq)
        .grid(grid_width, grid_height, 1)
        .threadgroup(32, 1, 1)
        .execute()
}
