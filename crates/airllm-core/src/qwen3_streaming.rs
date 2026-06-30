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

    /// Real Qwen3 MLP block: gate_proj + up_proj + SiLU(gate)*up + down_proj.
/// This is the FULL MLP block (not a stand-in). Returns output [1, hidden_size].
    pub fn mlp_block(&self, layer_idx: usize, post_normed: &Tensor) -> Result<Tensor> {
        let gate_name = format!("model.layers.{}.mlp.gate_proj.weight", layer_idx);
        let up_name = format!("model.layers.{}.mlp.up_proj.weight", layer_idx);
        let down_name = format!("model.layers.{}.mlp.down_proj.weight", layer_idx);

        let gate_w = self
            .builder
            .get_tensor(&gate_name)
            .with_context(|| format!("loading {}", gate_name))?;
        let gate = post_normed.matmul(&gate_w.t()?).with_context(|| "mlp gate_proj")?;
        drop(gate_w);

        let up_w = self
            .builder
            .get_tensor(&up_name)
            .with_context(|| format!("loading {}", up_name))?;
        let up = post_normed.matmul(&up_w.t()?).with_context(|| "mlp up_proj")?;
        drop(up_w);

        // SiLU(gate) * up
        let silu_gate = candle_nn::ops::silu(&gate).with_context(|| "silu")?;
        let act = (silu_gate * up).with_context(|| "act = silu(gate) * up")?;

        let down_w = self
            .builder
            .get_tensor(&down_name)
            .with_context(|| format!("loading {}", down_name))?;
        let out = act.matmul(&down_w.t()?).with_context(|| "mlp down_proj")?;
        drop(down_w);

        Ok(out)
    }

    /// Decoder layer: 2 RMSNorms + simplified attention stand-in (RMSNorm + linear) + real MLP.
    /// Honest: attention is simplified (no RoPE, no QK-norm, no SDPA), MLP is real Qwen3.
    /// Phase 2b-full replaces simplified attention with real Q/K/V/RoPE/QK-norm/SDPA/O.
    pub fn forward_layer(&self, layer_idx: usize, hidden_states: &Tensor) -> Result<Tensor> {
        let input_ln_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
        let post_ln_name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
        let q_proj_name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);

        // Pre-attention: input_layernorm + simplified attention stand-in (RMSNorm + q_proj linear)
        let ln_weight = self
            .builder
            .get_tensor(&input_ln_name)
            .with_context(|| format!("loading {}", input_ln_name))?;
        let pre_normed = rms_norm(hidden_states, &ln_weight, 1e-6)
            .with_context(|| "pre_attention RMSNorm")?;
        drop(ln_weight);

        let q_weight = self
            .builder
            .get_tensor(&q_proj_name)
            .with_context(|| format!("loading {}", q_proj_name))?;
        let attn_out = pre_normed.matmul(&q_weight.t()?).with_context(|| "attn stand-in")?;
        drop(q_weight);

        let hidden_after_attn = (hidden_states + attn_out).with_context(|| "attn residual")?;

        // Pre-MLP: post_attention_layernorm + real MLP (gate + up + SiLU + down)
        let post_ln = self
            .builder
            .get_tensor(&post_ln_name)
            .with_context(|| format!("loading {}", post_ln_name))?;
        let post_normed = rms_norm(&hidden_after_attn, &post_ln, 1e-6)
            .with_context(|| "post_attention RMSNorm")?;
        drop(post_ln);

        let mlp_out = self.mlp_block(layer_idx, &post_normed)?;
        let output = (hidden_after_attn + mlp_out).with_context(|| "mlp residual")?;
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

    /// Forward a single token through embed → N decoder layers → lm_head.
    /// Uses real Qwen3 MLP block + simplified attention stand-in.
    /// Returns logits [vocab_size].
    pub fn forward_one_token(&self, token_id: u32) -> Result<Tensor> {
        let mut hidden = self.embed(token_id)?;
        // hidden shape: [1, hidden_size]
        for layer_idx in 0..self.n_layers {
            hidden = self.forward_layer(layer_idx, &hidden)?;
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