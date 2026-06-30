//! Engine configuration loaded from TOML or CLI args.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineConfig {
    pub model_path: String,
    pub context_length: usize,
    pub max_new_tokens: usize,
    pub num_pinned_buffers: usize,
    pub layer_batch_size: usize,
    pub gpu_device_id: usize,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            model_path: String::new(),
            context_length: 4096,
            max_new_tokens: 128,
            num_pinned_buffers: 2,
            layer_batch_size: 1,
            gpu_device_id: 0,
        }
    }
}