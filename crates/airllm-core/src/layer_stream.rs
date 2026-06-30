//! Layer-stream executor (skeleton).
//!
//! Coordinates async prefetch of next layer while current layer is forward-computing.
//! Real impl uses glommio + safetensors mmap + cudarc H2D stream.

use crate::config::EngineConfig;

pub struct LayerStream {
    config: EngineConfig,
}

pub struct LayerShard {
    pub layer_idx: usize,
    pub bytes: Vec<u8>,
}

impl LayerStream {
    pub fn new(config: EngineConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &EngineConfig {
        &self.config
    }

    /// Skeleton: returns dummy shard. Phase 1 real impl uses safetensors mmap.
    pub async fn fetch_layer(&self, layer_idx: usize) -> LayerShard {
        LayerShard {
            layer_idx,
            bytes: Vec::new(),
        }
    }
}