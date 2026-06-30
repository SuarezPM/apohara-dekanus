//! Token and layer state structs shared by SelectivePolicy implementations.

/// Per-token runtime state.
#[derive(Debug, Clone, Default)]
pub struct TokenState {
    /// Token ID
    pub id: u32,
    /// Embedding vector (last hidden state from previous forward)
    pub embedding: Vec<f32>,
    /// Top-1 predicted token at last layer (for early-exit confidence)
    pub top1_prob: f32,
    /// Top-2 predicted token probability (entropy proxy)
    pub top2_prob: f32,
    /// Sequence position in current forward pass
    pub position: usize,
}

/// Per-layer runtime state (set up before each forward iteration).
#[derive(Debug, Clone, Default)]
pub struct LayerState {
    /// Layer index (0-indexed)
    pub idx: usize,
    /// Total number of layers in the model
    pub total_layers: usize,
    /// Layer type (attention, linear-attention, MoE, dense)
    pub kind: LayerKind,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub enum LayerKind {
    /// Standard full attention (vanilla O(n²))
    Attention,
    /// Linear attention (GatedDeltaNet, RWKV, Mamba)
    LinearAttention,
    /// MoE layer with shared+top-k experts
    MoE,
    /// Dense MLP layer
    #[default]
    Dense,
}