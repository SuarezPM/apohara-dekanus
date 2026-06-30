//! SelectivePolicy trait + no-op reference implementation.
//!
//! Phase 3: only NoOpPolicy (all layers active, top-k experts = k).
//! Phase 4: ASET (ACL Findings 2026) real impl.

use super::{LayerSet, LayerState, TokenState};

/// Trait implemented by sparse activation policies.
/// All methods must be `Send + Sync` (called from glommio executor).
pub trait SelectivePolicy: Send + Sync {
    /// Determine which decoder layers are active for this token.
    fn active_layers(&self, token: &TokenState, layer: &LayerState) -> LayerSet;

    /// Determine which MoE experts are active (returns indices into expert array).
    /// Returns `Vec<u16>` of expert IDs (typically top-2 for Mixtral, top-10 for Qwen3-MoE).
    fn active_experts(&self, token: &TokenState) -> Vec<u16>;

    /// Produce up to `k` speculative draft tokens given the context.
    /// Returns empty Vec if speculation disabled.
    fn speculate(&self, _ctx_tokens: &[u32], _k: usize) -> Vec<u32> {
        Vec::with_capacity(0) // default: no speculation
    }

    /// Early-exit at this layer if confident. Returns Some(layer_idx+1) if exit, None to continue.
    fn early_exit(&self, _layer_idx: usize, _logits_max_prob: f32) -> Option<usize> {
        None // default: never early-exit
    }
}

/// Reference implementation: all layers active, all experts active (dense fallback).
/// Used as the default for v0.1 / 8B smoke where sparsity gain is unnecessary.
pub struct NoOpPolicy;

impl SelectivePolicy for NoOpPolicy {
    fn active_layers(&self, _token: &TokenState, layer: &LayerState) -> LayerSet {
        LayerSet::all(layer.total_layers)
    }

    fn active_experts(&self, _token: &TokenState) -> Vec<u16> {
        // Dense fallback: caller decides actual top-k based on model config
        Vec::new()
    }
}