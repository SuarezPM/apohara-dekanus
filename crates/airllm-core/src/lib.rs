//! Layer-streaming inference engine core.
//!
//! Phase 1b: Qwen3 forward pass via candle-transformers + std::fs layer streaming.
//! Phase 2: extend with Qwen3 MoE + sparse expert routing.
//! Phase 3: extend with custom Qwen3-Next hybrid (linear attention + MoE).

#![deny(unsafe_code)]

pub mod config;
pub mod dispatch;
pub mod layer_stream;
pub mod layer_stream_v2;
pub mod pinned_buffer;
pub mod qwen3_moe_streaming;
pub mod qwen3_next_streaming;
pub mod qwen3_runner;
pub mod qwen3_streaming;
pub mod rms_norm_cuda;
pub mod rope_qknorm;

pub use config::EngineConfig;
pub use dispatch::{narrow, reshape, stack};
pub use layer_stream::{LayerStream, LayerShard};
pub use layer_stream_v2::LayerStreamedBuilder;
pub use pinned_buffer::PinnedHostBuffer;
pub use qwen3_moe_streaming::{MoEKVCache, Qwen3MoeConfig, Qwen3MoeStreamingModel};
pub use qwen3_next_streaming::{Qwen3NextConfig, Qwen3NextStreamingModel};
pub use qwen3_runner::{Qwen3Runner, Qwen3Variant, RunConfig, RunOutput};
pub use qwen3_streaming::Qwen3StreamingModel;
pub use rms_norm_cuda::rms_norm_cuda;
pub use rope_qknorm::{qk_norm, RoPETables};