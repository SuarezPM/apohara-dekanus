//! Layer-streaming inference engine core (skeleton).
//!
//! Pipeline: NVMe (safetensors shard) → glommio executor → pinned-host
//! double-buffer → cudarc H2D stream → CUDA forward → GPU KV cache →
//! release layer to next prefetch.
//!
//! Phase 1: skeleton modules. Phase 2: candle-nn Qwen3 forward wired.

#![forbid(unsafe_code)]

pub mod config;
pub mod layer_stream;
pub mod pinned_buffer;

pub use config::EngineConfig;
pub use layer_stream::{LayerStream, LayerShard};
pub use pinned_buffer::PinnedHostBuffer;