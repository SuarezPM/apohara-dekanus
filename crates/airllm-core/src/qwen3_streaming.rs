//! Minimal Qwen3 forward pass using LayerStreamedBuilder for per-step weight loading.
//!
//! Phase 2b honest PoC: implements ONLY the inference primitives that prove the
//! load → use → release cycle with REAL Qwen3-8B weights via LayerStreamedBuilder.
//!
//! Implemented:
//! - Embedding lookup (load embed_tokens, gather row)
//! - Simplified decoder layer (RMSNorm + 2 linear + residual — NO attention/MLP yet)
//! - LM head (load lm_head.weight, matmul with hidden_states)
//!
//! NOT implemented (deferred to Phase 2b full):
//! - Q/K/V projections (requires custom RoPE + QK-norm + GQA)
//! - Scaled dot-product attention
//! - Full MLP (gate_proj + up_proj + down_proj with SiLU)
//! - 36-layer orchestration
//! - Auto-regressive decode loop with KV cache
//!
//! This PoC proves the I/O pattern works end-to-end. Full Qwen3 attention + MLP
//! is documented as Phase 2b-full work in AUDIT D0008.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::ops::rms_norm;

use crate::layer_stream_v2::LayerStreamedBuilder;

/// Minimal streaming Qwen3 model: embed + N simplified layers + lm_head.
pub struct Qwen3StreamingModel {
    builder: LayerStreamedBuilder,
    hidden_size: usize,
    n_layers: usize,
    vocab_size: usize,
}

impl Qwen3StreamingModel {
    /// Load config from config.json and open LayerStreamedBuilder.
    pub fn open(model_dir: &std::path::Path, device: Device, dtype: DType) -> Result<Self> {
        // Parse config.json (needed for hidden_size, n_layers, vocab_size)
        let config_str = std::fs::read_to_string(model_dir.join("config.json"))
            .with_context(|| "reading config.json")?;
        let config: serde_json::Value = serde_json::from_str(&config_str)?;
        let hidden_size = config["hidden_size"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("hidden_size missing in config.json"))? as usize;
        let n_layers = config["num_hidden_layers"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("num_hidden_layers missing"))? as usize;
        let vocab_size = config["vocab_size"]
            .as_u64()
            .ok_or_else(|| anyhow::anyhow!("vocab_size missing"))? as usize;

        let builder = LayerStreamedBuilder::open(model_dir, device, dtype)
            .with_context(|| "opening LayerStreamedBuilder")?;

        Ok(Self {
            builder,
            hidden_size,
            n_layers,
            vocab_size,
        })
    }

    pub fn n_layers(&self) -> usize {
        self.n_layers
    }
    pub fn hidden_size(&self) -> usize {
        self.hidden_size
    }
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Embedding lookup: load embed_tokens.weight, gather row for token_id.
    /// This is the FIRST inference primitive: load big tensor, use one row, drop.
    pub fn embed(&self, token_id: u32) -> Result<Tensor> {
        let embed = self
            .builder
            .get_tensor("model.embed_tokens.weight")
            .with_context(|| "loading embed_tokens.weight")?;
        // embed shape: [vocab_size, hidden_size]; gather row for token_id
        let row = embed
            .narrow(0, token_id as usize, 1)
            .with_context(|| format!("narrowing embed to row {}", token_id))?;
        // Drop embed to release ~600MB after use
        drop(embed);
        Ok(row)
    }

    /// Simplified decoder layer: RMSNorm + linear + residual.
    /// NOT Qwen3's actual attention+MLP — this is a PoC that proves the load/use/drop
    /// cycle for layer weights works. Full attention+MLP is Phase 2b-full work.
    /// We use self_attn.q_proj.weight [hidden_size, hidden_size] as the linear
    /// (one of the attention weights) — semantically a stand-in for the full
    /// attention+MLP block. Honest PoC, not real Qwen3 attention.
    pub fn forward_simplified_layer(&self, layer_idx: usize, hidden_states: &Tensor) -> Result<Tensor> {
        let input_ln_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
        let q_proj_name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);

        // Load input_layernorm.weight [hidden_size] and apply RMSNorm
        let ln_weight = self
            .builder
            .get_tensor(&input_ln_name)
            .with_context(|| format!("loading {}", input_ln_name))?;
        let normed = rms_norm(hidden_states, &ln_weight, 1e-6)
            .with_context(|| "RMSNorm forward")?;
        drop(ln_weight);

        // Load self_attn.q_proj.weight [hidden_size, hidden_size] and apply linear
        let q_weight = self
            .builder
            .get_tensor(&q_proj_name)
            .with_context(|| format!("loading {}", q_proj_name))?;
        let projected = normed
            .matmul(&q_weight.t()?)
            .with_context(|| "matmul with q_proj.weight")?;
        drop(q_weight);

        // Residual: hidden + projected (simplified; full layer would have attention+MLP)
        let output = (hidden_states + projected)
            .with_context(|| "residual add")?;
        Ok(output)
    }

    /// LM head: load lm_head.weight [vocab_size, hidden_size], matmul with hidden.
    pub fn lm_head(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let lm_weight = self
            .builder
            .get_tensor("lm_head.weight")
            .with_context(|| "loading lm_head.weight")?;
        let logits = hidden_states
            .matmul(&lm_weight.t()?)
            .with_context(|| "matmul with lm_head.weight")?;
        drop(lm_weight);
        Ok(logits)
    }

    /// Forward a single token through embed → N simplified layers → lm_head.
    /// Returns logits [vocab_size].
    pub fn forward_one_token(&self, token_id: u32) -> Result<Tensor> {
        let mut hidden = self.embed(token_id)?;
        // hidden shape: [1, hidden_size]
        for layer_idx in 0..self.n_layers {
            hidden = self.forward_simplified_layer(layer_idx, &hidden)?;
        }
        // Final norm + lm_head (Qwen3 has model.norm.weight at end before lm_head)
        let final_norm = self
            .builder
            .get_tensor("model.norm.weight")
            .with_context(|| "loading model.norm.weight")?;
        let normed = rms_norm(&hidden, &final_norm, 1e-6)?;
        drop(final_norm);

        self.lm_head(&normed)
    }
}