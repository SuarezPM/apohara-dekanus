//! Selective per-token evaluation policy — the missing primitive.
//!
//! Defines `SelectivePolicy` trait with hooks for sparse activation:
//! - `active_layers` — which decoder layers to forward this token
//! - `active_experts` — which MoE experts to route this token
//! - `speculate` — produce draft tokens (speculative decoding)
//! - `early_exit` — terminate early at this layer if confident
//!
//! Phase 3: no-op impl. Phase 4: ASET (ACL Findings 2026) real impl.

#![forbid(unsafe_code)]

pub mod policy;
pub mod state;

pub use policy::{NoOpPolicy, SelectivePolicy};
pub use state::{LayerState, TokenState};

/// Bit-vector of active layer IDs (140 layers max for 397B-A17B)
#[derive(Debug, Clone)]
pub struct LayerSet(pub Vec<bool>);

impl LayerSet {
    /// Empty layer set (no layers active)
    pub fn none(n_layers: usize) -> Self {
        Self(vec![false; n_layers])
    }

    /// All layers active
    pub fn all(n_layers: usize) -> Self {
        Self(vec![true; n_layers])
    }

    /// Number of active layers
    pub fn count_active(&self) -> usize {
        self.0.iter().filter(|x| **x).count()
    }

    /// Sparsity ratio = active / total
    pub fn sparsity(&self) -> f32 {
        self.count_active() as f32 / self.0.len() as f32
    }
}