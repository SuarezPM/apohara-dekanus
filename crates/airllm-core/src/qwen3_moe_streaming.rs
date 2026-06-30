//! Qwen3-30B-A3B sparse MoE forward pass with layer-streamed expert loading.
//!
//! Phase 3 architecture:
//! - 48 layers (vs 36 in 8B)
//! - hidden_size = 2048
//! - num_attention_heads = 32, num_key_value_heads = 4 (GQA 8:1)
//! - num_experts = 128, num_experts_per_tok = 8
//! - moe_intermediate_size = 768 (per-expert FFN hidden)
//! - shared_expert: always active (additional 768 hidden)
//! - vocab_size = 151936
//!
//! Layer-streaming benefit for MoE:
//! - Without streaming: load all 128 experts × 3 weights = 384 weights per layer
//!   48 layers × 384 = 18,432 weights to load, ~36GB total
//! - With sparse MoE + streaming: load only 8 active experts per token
//!   48 layers × 8 experts × 3 = 1,152 weights active, ~1.1GB peak
//!
//! This module uses LayerStreamedBuilder to lazy-load per-tensor,
//! activating only the experts selected by the router.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

use crate::layer_stream_v2::LayerStreamedBuilder;
use crate::rms_norm_cuda::rms_norm_cuda;
use crate::rope_qknorm::{qk_norm, RoPETables};

/// Qwen3-30B-A3B model config (loaded from config.json).
#[derive(Debug, Clone)]
pub struct Qwen3MoeConfig {
    pub hidden_size: usize,
    pub n_layers: usize,
    pub vocab_size: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert: bool,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
}

impl Qwen3MoeConfig {
    pub fn from_config_json(path: &std::path::Path) -> Result<Self> {
        let s = std::fs::read_to_string(path)?;
        let v: serde_json::Value = serde_json::from_str(&s)?;
        Ok(Self {
            hidden_size: v["hidden_size"].as_u64().unwrap_or(2048) as usize,
            n_layers: v["num_hidden_layers"].as_u64().unwrap_or(48) as usize,
            vocab_size: v["vocab_size"].as_u64().unwrap_or(151936) as usize,
            num_attention_heads: v["num_attention_heads"].as_u64().unwrap_or(32) as usize,
            num_kv_heads: v["num_key_value_heads"].as_u64().unwrap_or(4) as usize,
            head_dim: v["head_dim"].as_u64().unwrap_or(128) as usize,
            num_experts: v["num_experts"].as_u64().unwrap_or(128) as usize,
            num_experts_per_tok: v["num_experts_per_tok"].as_u64().unwrap_or(8) as usize,
            moe_intermediate_size: v["moe_intermediate_size"].as_u64().unwrap_or(768) as usize,
            shared_expert: v["shared_expert"].as_str().map(|s| s == "true").unwrap_or(false),
            rope_theta: v["rope_theta"].as_f64().unwrap_or(1_000_000.0),
            max_position_embeddings: v["max_position_embeddings"].as_u64().unwrap_or(40960) as usize,
        })
    }
}

/// KV cache for MoE model (same as dense Qwen3StreamingModel).
pub struct MoEKVCache {
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
}

impl MoEKVCache {
    pub fn new(n_layers: usize, device: &Device) -> Result<Self> {
        let mut keys = Vec::with_capacity(n_layers);
        let mut values = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            keys.push(Tensor::zeros((0, 4, 128), DType::F32, device)?);
            values.push(Tensor::zeros((0, 4, 128), DType::F32, device)?);
        }
        Ok(Self { keys, values })
    }
}

/// Qwen3-30B-A3B streaming model with sparse MoE expert routing.
pub struct Qwen3MoeStreamingModel {
    pub builder: LayerStreamedBuilder,
    pub config: Qwen3MoeConfig,
    pub rope_tables: Option<RoPETables>,
}

impl Qwen3MoeStreamingModel {
    pub fn open(model_dir: &std::path::Path, device: Device, dtype: DType) -> Result<Self> {
        let config_path = model_dir.join("config.json");
        let config = Qwen3MoeConfig::from_config_json(&config_path)
            .with_context(|| "loading MoE config")?;
        let rope_tables = RoPETables::new(256, config.head_dim, config.rope_theta, &device)
            .with_context(|| "building RoPE tables")?;
        let builder = LayerStreamedBuilder::open(model_dir, device, dtype)
            .with_context(|| "opening LayerStreamedBuilder")?;
        Ok(Self {
            builder,
            config,
            rope_tables: Some(rope_tables),
        })
    }

    /// Embedding lookup.
    pub fn embed(&self, token_id: u32) -> Result<Tensor> {
        let embed = self.builder.get_tensor("model.embed_tokens.weight")?;
        let row = embed.narrow(0, token_id as usize, 1)?;
        drop(embed);
        Ok(row)
    }

    /// Single decoder layer forward with sparse MoE + layer-streamed experts.
    /// Active experts per layer: only 8 (per num_experts_per_tok) out of 128.
    /// This is the KILLER FEATURE of layer-streaming: we never load the 120 inactive experts.
    pub fn forward_layer_moe(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        position: usize,
        kv_cache: &mut MoEKVCache,
    ) -> Result<Tensor> {
        let input_ln_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
        let post_ln_name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
        let q_name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);
        let k_name = format!("model.layers.{}.self_attn.k_proj.weight", layer_idx);
        let v_name = format!("model.layers.{}.self_attn.v_proj.weight", layer_idx);
        let o_name = format!("model.layers.{}.self_attn.o_proj.weight", layer_idx);

        // 1. Pre-attention RMSNorm
        let ln_w = self.builder.get_tensor(&input_ln_name)?;
        let pre_normed = rms_norm_cuda(hidden_states, &ln_w, 1e-6)?;
        drop(ln_w);

        // 2. Q/K/V projections
        let q_w = self.builder.get_tensor(&q_name)?;
        let q = pre_normed.matmul(&q_w.t()?)?.reshape((1, self.config.num_attention_heads, self.config.head_dim))?;
        drop(q_w);

        let k_w = self.builder.get_tensor(&k_name)?;
        let k_new = pre_normed.matmul(&k_w.t()?)?.reshape((1, self.config.num_kv_heads, self.config.head_dim))?;
        drop(k_w);

        let v_w = self.builder.get_tensor(&v_name)?;
        let v_new = pre_normed.matmul(&v_w.t()?)?.reshape((1, self.config.num_kv_heads, self.config.head_dim))?;
        drop(v_w);

        // 3. RoPE (Qwen3 partial=0.25)
        let rope = self.rope_tables.as_ref()
            .ok_or_else(|| anyhow::anyhow!("rope_tables not initialized"))?;
        let q = rope.apply(&q, position)?;
        let k_new = rope.apply(&k_new, position)?;

        // 4. QK-norm (per-head RMSNorm on Q/K)
        let q_norm_name = format!("model.layers.{}.self_attn.q_norm.weight", layer_idx);
        let k_norm_name = format!("model.layers.{}.self_attn.k_norm.weight", layer_idx);
        let q_norm_w = self.builder.get_tensor(&q_norm_name)?;
        let k_norm_w = self.builder.get_tensor(&k_norm_name)?;
        let q = qk_norm(&q, &q_norm_w, 1e-6)?;
        let k_new = qk_norm(&k_new, &k_norm_w, 1e-6)?;
        drop(q_norm_w);
        drop(k_norm_w);

        // 5. Append to KV cache
        if kv_cache.keys[layer_idx].dim(0)? == 0 {
            kv_cache.keys[layer_idx] = k_new.clone();
            kv_cache.values[layer_idx] = v_new.clone();
        } else {
            kv_cache.keys[layer_idx] = Tensor::cat(&[&kv_cache.keys[layer_idx], &k_new], 0)?;
            kv_cache.values[layer_idx] = Tensor::cat(&[&kv_cache.values[layer_idx], &v_new], 0)?;
        }

        // 6. SDPA (per-head loop with GQA expansion) — simplified for MoE
        // For Phase 3 honest PoC, we use simple GQA expand (no RoPE/QK-norm
        // already applied above). Per-head SDPA at sequence length up to 256.
        let k_cache = &kv_cache.keys[layer_idx];
        let v_cache = &kv_cache.values[layer_idx];
        let scale = 1.0 / (self.config.head_dim as f64).sqrt();
        let mut attn_outs = Vec::with_capacity(self.config.num_attention_heads);
        let ratio = self.config.num_attention_heads / self.config.num_kv_heads;
        for h in 0..self.config.num_attention_heads {
            let kv_h = h / ratio;
            let q_h = q.narrow(1, h, 1)?.squeeze(1)?;
            let k_h = k_cache.narrow(1, kv_h, 1)?.squeeze(1)?;
            let v_h = v_cache.narrow(1, kv_h, 1)?.squeeze(1)?;
            let scores = q_h.matmul(&k_h.t()?)?.affine(scale, 0.0)?;
            let weights = candle_nn::ops::softmax_last_dim(&scores)?;
            let ctx = weights.matmul(&v_h)?;
            attn_outs.push(ctx);
        }
        let attn_concat = Tensor::cat(&attn_outs.iter().collect::<Vec<_>>(), 1)?;

        let o_w = self.builder.get_tensor(&o_name)?;
        let attn_out = attn_concat.matmul(&o_w.t()?)?;
        drop(o_w);

        let hidden_after_attn = (hidden_states + attn_out)?;

        // 7. Post-attention RMSNorm
        let post_ln = self.builder.get_tensor(&post_ln_name)?;
        let post_normed = rms_norm_cuda(&hidden_after_attn, &post_ln, 1e-6)?;
        drop(post_ln);

        // 8. Sparse MoE MLP: only load 8 active experts per token
        let moe_out = self.moe_mlp_block(layer_idx, &post_normed)?;
        Ok((hidden_after_attn + moe_out)?)
    }

    /// Sparse MoE MLP block: router selects top-k experts, only those are loaded.
    /// Memory benefit: 128 experts × 3MB = 384MB per layer → 8 active × 3MB = 24MB per layer.
    /// Plus shared expert (always loaded, ~6MB per layer).
    /// Total active: ~30MB per layer × 48 layers = 1.4GB peak.
    pub fn moe_mlp_block(&self, layer_idx: usize, hidden_states: &Tensor) -> Result<Tensor> {
        // 1. Router: load gate weight [hidden, num_experts]
        let gate_name = format!("model.layers.{}.mlp.gate.weight", layer_idx);
        let gate_w = self.builder.get_tensor(&gate_name)?;
        let scores = hidden_states.matmul(&gate_w.t()?)?; // [1, num_experts=128]
        drop(gate_w);

        // 2. Top-k selection (softmax BEFORE topk to get valid router weights over
        // the full expert distribution, not just renormalized over the k winners).
        let all_probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let (top_weights, top_indices) = topk_last_dim(&all_probs, self.config.num_experts_per_tok as usize)?;
        // top_weights now contains real router probabilities (sum to < 1.0 across 8 experts).

        // 3. For each selected expert, compute and accumulate
        let mut output = Tensor::zeros(
            (1, self.config.hidden_size),
            hidden_states.dtype(),
            hidden_states.device(),
        )?;
        for i in 0..self.config.num_experts_per_tok as usize {
            let expert_id = top_indices.get(0)?.get(i)?.to_scalar::<u32>()? as usize;
            let weight = top_weights.get(0)?.get(i)?.to_scalar::<f32>()?;
            let expert_out = self.expert_forward(layer_idx, expert_id, hidden_states)?;
            output = (output + (expert_out * weight as f64)?)?;
        }

        // 4. Shared expert (always active)
        if self.config.shared_expert {
            let shared_out = self.shared_expert_forward(layer_idx, hidden_states)?;
            output = (output + shared_out)?;
        }

        Ok(output)
    }

    /// Single expert forward: load gate/up/down weights for this expert, compute SiLU(gate) * up @ down.
    pub fn expert_forward(
        &self,
        layer_idx: usize,
        expert_id: usize,
        hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let gate_name = format!("model.layers.{}.mlp.experts.{}.gate_proj.weight", layer_idx, expert_id);
        let up_name = format!("model.layers.{}.mlp.experts.{}.up_proj.weight", layer_idx, expert_id);
        let down_name = format!("model.layers.{}.mlp.experts.{}.down_proj.weight", layer_idx, expert_id);

        let gate_w = self.builder.get_tensor(&gate_name)?;
        let gate = hidden_states.matmul(&gate_w.t()?)?; // [1, moe_intermediate=768]
        drop(gate_w);

        let up_w = self.builder.get_tensor(&up_name)?;
        let up = hidden_states.matmul(&up_w.t()?)?; // [1, 768]
        drop(up_w);

        let silu_gate = candle_nn::ops::silu(&gate)?;
        let act = (silu_gate * up)?;

        let down_w = self.builder.get_tensor(&down_name)?;
        let out = act.matmul(&down_w.t()?)?; // [1, hidden]

        Ok(out)
    }

    /// Shared expert forward: always loaded, no router.
    pub fn shared_expert_forward(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
    ) -> Result<Tensor> {
        let gate_name = format!("model.layers.{}.mlp.shared_expert.gate_proj.weight", layer_idx);
        let up_name = format!("model.layers.{}.mlp.shared_expert.up_proj.weight", layer_idx);
        let down_name = format!("model.layers.{}.mlp.shared_expert.down_proj.weight", layer_idx);

        let gate_w = self.builder.get_tensor(&gate_name)?;
        let gate = hidden_states.matmul(&gate_w.t()?)?;
        drop(gate_w);

        let up_w = self.builder.get_tensor(&up_name)?;
        let up = hidden_states.matmul(&up_w.t()?)?;
        drop(up_w);

        let silu_gate = candle_nn::ops::silu(&gate)?;
        let act = (silu_gate * up)?;

        let down_w = self.builder.get_tensor(&down_name)?;
        let out = act.matmul(&down_w.t()?)?;

        Ok(out)
    }

    /// Forward one token through all 48 MoE layers + lm_head.
    pub fn forward_one_token(
        &self,
        token_id: u32,
        position: usize,
        kv_cache: &mut MoEKVCache,
    ) -> Result<Tensor> {
        let mut hidden = self.embed(token_id)?;

        for layer_idx in 0..self.config.n_layers {
            hidden = self.forward_layer_moe(layer_idx, &hidden, position, kv_cache)?;
        }

        let final_norm = self.builder.get_tensor("model.norm.weight")?;
        let normed = rms_norm_cuda(&hidden, &final_norm, 1e-6)?;
        drop(final_norm);

        // LM head
        let lm_w = self.builder.get_tensor("lm_head.weight")?;
        let logits = normed.matmul(&lm_w.t()?)?; // [1, vocab=151936]
        drop(lm_w);
        Ok(logits)
    }

    /// Auto-regressive decode loop.
    pub fn decode(&self, initial_token: u32, max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut kv_cache = MoEKVCache::new(self.config.n_layers, self.builder.device())?;
        let mut generated = vec![initial_token];

        for _ in 0..max_new_tokens {
            let position = generated.len() - 1;
            let last = *generated.last().unwrap();
            let logits = self.forward_one_token(last, position, &mut kv_cache)?;
            // Argmax
            let logits_vec: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
            let (idx, _) = logits_vec
                .iter()
                .enumerate()
                .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .map(|(i, v)| (i, *v))
                .unwrap_or((0, 0.0));
            generated.push(idx as u32);
        }
        Ok(generated)
    }
}

/// Top-k along last dim (candle doesn't have built-in for arbitrary k).
fn topk_last_dim(x: &Tensor, k: usize) -> Result<(Tensor, Tensor)> {
    use candle_core::D;
    let last_dim = x.dim(D::Minus1)?;
    if k > last_dim {
        anyhow::bail!("k={} larger than last_dim={}", k, last_dim);
    }
    // Get values + indices
    let values_vec: Vec<f32> = x.flatten_all()?.to_vec1()?;
    let shape: Vec<usize> = x.dims().to_vec();
    let last_size = *shape.last().unwrap();
    let n_rows: usize = shape[..shape.len() - 1].iter().product();

    // Build (index, value) pairs per row, sort, take top-k
    let mut top_values = Vec::with_capacity(n_rows * k);
    let mut top_indices = Vec::with_capacity(n_rows * k);
    for r in 0..n_rows {
        let row = &values_vec[r * last_size..(r + 1) * last_size];
        let mut indexed: Vec<(usize, f32)> = row.iter().enumerate().map(|(i, &v)| (i, v)).collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        for i in 0..k {
            top_indices.push((indexed[i].0 + r * last_size) as u32);
            top_values.push(indexed[i].1);
        }
    }

    let mut out_shape = shape.clone();
    *out_shape.last_mut().unwrap() = k;
    let values_t = Tensor::from_vec(top_values, out_shape.clone(), x.device())?;
    let indices_t = Tensor::from_vec(top_indices, out_shape, x.device())?;
    Ok((values_t, indices_t))
}