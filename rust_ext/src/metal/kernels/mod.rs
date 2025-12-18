//! Metal kernel wrappers for high-performance inference.
//!
//! Each kernel follows Apple Metal best practices:
//! - Simdgroup-based threading (32 threads/simdgroup)
//! - Online softmax with max tracking
//! - Compile-time specialization for head dimensions
//! - Minimal memory bandwidth through careful data layout

pub mod attention;
pub mod gemv;
pub mod rope;
pub mod rms_norm;
pub mod cache;

// Re-export dispatch functions for easy access
pub use rms_norm::{dispatch_rms_norm, dispatch_rms_norm_inplace, dispatch_fused_add_rms_norm};
pub use rope::{dispatch_rope_forward, dispatch_rope_inplace, dispatch_rope_decode, dispatch_precompute_freqs};
pub use cache::{dispatch_reshape_and_cache, dispatch_copy_blocks, dispatch_gather_cached, dispatch_init_cache_block};
