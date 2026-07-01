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
use candle_core::{D, DType, Device, Tensor};
use candle_nn::ops::rms_norm;

use crate::layer_stream_v2::LayerStreamedBuilder;
use crate::rms_norm_cuda::rms_norm_cuda;
use crate::rope_qknorm::{qk_norm, RoPETables};

/// Minimal streaming Qwen3 model: embed + N simplified layers + lm_head.
pub struct Qwen3StreamingModel {
    builder: LayerStreamedBuilder,
    hidden_size: usize,
    n_layers: usize,
    vocab_size: usize,
    rope_tables: Option<RoPETables>,
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

        // Precompute RoPE tables (Qwen3: rope_theta=1_000_000, head_dim=128, partial=0.25)
        // Max seq 256 covers most use cases; longer sequences need rebuild.
        // Must come BEFORE LayerStreamedBuilder::open which consumes `device`.
        let rope_tables = RoPETables::new(256, 128, 1_000_000.0, &device)
            .with_context(|| "building RoPE tables")?;

        let builder = LayerStreamedBuilder::open(model_dir, device, dtype)
            .with_context(|| "opening LayerStreamedBuilder")?;

        Ok(Self {
            builder,
            hidden_size,
            n_layers,
            vocab_size,
            rope_tables: Some(rope_tables),
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

    /// Real Qwen3 attention block: Q/K/V projections + GQA single-token attention + O projection.
/// Loaded weights per layer: q_proj, k_proj, v_proj, o_proj.
/// Deferred to Phase 2b-full: RoPE (positional encoding) + QK-norm (per-head RMSNorm on q/k).
/// For single-token decode, attention reduces to: attn_out[q_head] = v[kv_head_for_q_head]
/// (since softmax over a single element = 1.0).
pub fn attention_block(&self, layer_idx: usize, pre_normed: &Tensor) -> Result<Tensor> {
    use candle_core::DType;

    let q_name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);
    let k_name = format!("model.layers.{}.self_attn.k_proj.weight", layer_idx);
    let v_name = format!("model.layers.{}.self_attn.v_proj.weight", layer_idx);
    let o_name = format!("model.layers.{}.self_attn.o_proj.weight", layer_idx);

    // Qwen3-8B config: hidden=4096, num_heads=32, num_kv_heads=8, head_dim=128
    let num_heads = 32usize;
    let num_kv_heads = 8usize;
    let head_dim = 128usize;
    let hidden_size = 4096usize;

    let q_w = self
        .builder
        .get_tensor(&q_name)
        .with_context(|| format!("loading {}", q_name))?;
    let q = pre_normed.matmul(&q_w.t()?)?.reshape((1, num_heads, head_dim))?;
    drop(q_w);

    let k_w = self
        .builder
        .get_tensor(&k_name)
        .with_context(|| format!("loading {}", k_name))?;
    let _k = pre_normed.matmul(&k_w.t()?)?.reshape((1, num_kv_heads, head_dim))?;
    drop(k_w);

    let v_w = self
        .builder
        .get_tensor(&v_name)
        .with_context(|| format!("loading {}", v_name))?;
    let v = pre_normed.matmul(&v_w.t()?)?.reshape((1, num_kv_heads, head_dim))?;
    drop(v_w);

    // Single-token decode: GQA attention reduces to attn_out[q_h] = v[kv_h] where kv_h = q_h // 4
    // (no softmax needed since only 1 key per query head, softmax(1) = 1)
    // RoPE + QK-norm deferred (would modify q and k before this step).
    // Expand v from [1, 8, 128] to [1, 32, 128] via GQA repeat (each q_head uses its kv_head)
    let v_expanded = v
        .reshape((1, num_kv_heads, 1, head_dim))?
        .broadcast_as((1, num_kv_heads, num_heads / num_kv_heads, head_dim))?
        .reshape((1, num_heads, head_dim))?;

    // Concatenate back to [1, hidden_size]
    let attn_concat = v_expanded.reshape((1, hidden_size))?;
    // Cast to F32 if needed (matmul result might be BF16 if we ran on GPU)
    let attn_concat = if attn_concat.dtype() != DType::F32 {
        attn_concat.to_dtype(DType::F32)?
    } else {
        attn_concat
    };

    let o_w = self
        .builder
        .get_tensor(&o_name)
        .with_context(|| format!("loading {}", o_name))?;
    let attn_out = attn_concat.matmul(&o_w.t()?)?;
    drop(o_w);

    // Apply q identity: Q contribution (without RoPE it's not strictly meaningful,
    // but for honest PoC we want some signal from the attention path)
    // attn_out += q @ something? Actually Q without RoPE is just random directions.
    // Honest PoC: skip Q-K interaction, just pass v through o_proj (captures ~15% of attn compute).
    // The q tensor is loaded (verifies pipeline) but not used in the dot product.

    Ok(attn_out)
}

/// Decoder layer: 2 RMSNorms + real attention + real MLP.
pub fn forward_layer(&self, layer_idx: usize, hidden_states: &Tensor) -> Result<Tensor> {
    let input_ln_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
    let post_ln_name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);

    let ln_weight = self
        .builder
        .get_tensor(&input_ln_name)
        .with_context(|| format!("loading {}", input_ln_name))?;
    let pre_normed = rms_norm_cuda(hidden_states, &ln_weight, 1e-6)
        .with_context(|| "pre_attention RMSNorm")?;
    drop(ln_weight);

    let attn_out = self.attention_block(layer_idx, &pre_normed)?;
    let hidden_after_attn = (hidden_states + attn_out).with_context(|| "attn residual")?;

    let post_ln = self
        .builder
        .get_tensor(&post_ln_name)
        .with_context(|| format!("loading {}", post_ln_name))?;
    let post_normed = rms_norm_cuda(&hidden_after_attn, &post_ln, 1e-6)
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
        let normed = rms_norm_cuda(&hidden, &final_norm, 1e-6)?;
        drop(final_norm);

        self.lm_head(&normed)
    }

    /// Forward multiple tokens (independent, no KV cache). Each token is processed
    /// in isolation through all 36 layers. This is NOT autoregressive generation
    /// (each step is independent, no history). For real generation, KV cache +
    /// RoPE + QK-norm + decode loop needed (Phase 2b-full multi-token, deferred).
    /// Returns Vec<logits> of length tokens.len().
    pub fn forward_multi_token(&self, token_ids: &[u32]) -> Result<Vec<Tensor>> {
        let mut all_logits = Vec::with_capacity(token_ids.len());
        for &tid in token_ids {
            let logits = self.forward_one_token(tid)?;
            all_logits.push(logits);
        }
        Ok(all_logits)
    }
}

/// Per-layer KV cache: stores K and V tensors for all positions processed so far.
/// Sequence length grows by 1 per token. Total size at seq_len=128:
/// 36 layers × 2 (K+V) × [128, 8, 128] BF16 ≈ 72 MB on host (fits in RAM).
pub struct KVCache {
    /// Per-layer: (K, V) tensors of shape [seq_len, num_kv_heads, head_dim]
    pub keys: Vec<Tensor>,
    pub values: Vec<Tensor>,
}

impl KVCache {
    /// Allocate empty cache for n_layers.
    pub fn new(n_layers: usize, device: &Device) -> Self {
        Self {
            keys: (0..n_layers).map(|_| Tensor::zeros((0, 8, 128), DType::F32, device).unwrap()).collect(),
            values: (0..n_layers).map(|_| Tensor::zeros((0, 8, 128), DType::F32, device).unwrap()).collect(),
        }
    }
}

impl Qwen3StreamingModel {
    /// Forward a single token at a given position with KV cache (multi-token aware).
    /// Appends new K/V to the cache; returns logits + updated cache position.
    /// RoPE + QK-norm still deferred (positional info + per-head RMSNorm).
    /// For Phase 2b-full multi-token honest PoC.
    pub fn forward_with_kv_cache(
        &self,
        token_id: u32,
        position: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let mut hidden = self.embed(token_id)?;

        for layer_idx in 0..self.n_layers {
            hidden = self.forward_layer_with_kv(layer_idx, &hidden, position, kv_cache)?;
        }

        let final_norm = self
            .builder
            .get_tensor("model.norm.weight")
            .with_context(|| "loading model.norm.weight")?;
        let normed = rms_norm_cuda(&hidden, &final_norm, 1e-6)?;
        drop(final_norm);

        self.lm_head(&normed)
    }

    /// Single decoder layer forward with KV cache (real multi-position attention).
    /// Real SDPA over cached K/V with RoPE (Qwen3 partial=0.25). QK-norm deferred.
    pub fn forward_layer_with_kv(
        &self,
        layer_idx: usize,
        hidden_states: &Tensor,
        position: usize,
        kv_cache: &mut KVCache,
    ) -> Result<Tensor> {
        let input_ln_name = format!("model.layers.{}.input_layernorm.weight", layer_idx);
        let post_ln_name = format!("model.layers.{}.post_attention_layernorm.weight", layer_idx);
        let q_name = format!("model.layers.{}.self_attn.q_proj.weight", layer_idx);
        let k_name = format!("model.layers.{}.self_attn.k_proj.weight", layer_idx);
        let v_name = format!("model.layers.{}.self_attn.v_proj.weight", layer_idx);
        let o_name = format!("model.layers.{}.self_attn.o_proj.weight", layer_idx);

        let num_heads = 32usize;
        let num_kv_heads = 8usize;
        let head_dim = 128usize;

        // Pre-attention RMSNorm
        let ln_w = self.builder.get_tensor(&input_ln_name)?;
        let pre_normed = rms_norm_cuda(hidden_states, &ln_w, 1e-6)?;
        drop(ln_w);

        // Q/K/V projections
        let q_w = self.builder.get_tensor(&q_name)?;
        let q = pre_normed.matmul(&q_w.t()?)?.reshape((1, num_heads, head_dim))?;
        drop(q_w);

        let k_w = self.builder.get_tensor(&k_name)?;
        let k_new = pre_normed.matmul(&k_w.t()?)?.reshape((1, num_kv_heads, head_dim))?;
        drop(k_w);

        let v_w = self.builder.get_tensor(&v_name)?;
        let v_new = pre_normed.matmul(&v_w.t()?)?.reshape((1, num_kv_heads, head_dim))?;
        drop(v_w);

        // Apply RoPE to q and k_new at this position (Qwen3 partial_rotary_factor=0.25)
        let rope = self.rope_tables.as_ref()
            .ok_or_else(|| anyhow::anyhow!("rope_tables not initialized"))?;
        let q = rope.apply(&q, position)?;
        let k_new = rope.apply(&k_new, position)?;

        // Apply QK-norm (Qwen3-specific per-head RMSNorm on q and k)
        let q_norm_name = format!("model.layers.{}.self_attn.q_norm.weight", layer_idx);
        let k_norm_name = format!("model.layers.{}.self_attn.k_norm.weight", layer_idx);
        let q_norm_w = self.builder.get_tensor(&q_norm_name)?;
        let k_norm_w = self.builder.get_tensor(&k_norm_name)?;
        let q = qk_norm(&q, &q_norm_w, 1e-6)?;
        let k_new = qk_norm(&k_new, &k_norm_w, 1e-6)?;
        drop(q_norm_w);
        drop(k_norm_w);

        // Append to KV cache (concat along seq dim 0; k_new shape [1, kv_heads, head_dim] = [seq=1, ...])
        if kv_cache.keys[layer_idx].dim(0)? == 0 {
            kv_cache.keys[layer_idx] = k_new.clone();
            kv_cache.values[layer_idx] = v_new.clone();
        } else {
            kv_cache.keys[layer_idx] = Tensor::cat(&[&kv_cache.keys[layer_idx], &k_new], 0)?;
            kv_cache.values[layer_idx] = Tensor::cat(&[&kv_cache.values[layer_idx], &v_new], 0)?;
        }
        let k_cache = &kv_cache.keys[layer_idx]; // [seq_len, num_kv_heads, head_dim]
        let v_cache = &kv_cache.values[layer_idx];

        // Real SDPA: q [1, num_heads, head_dim] x k_cache^T [num_kv_heads, head_dim, seq_len]
        // Per GQA: q_head i uses k_cache[kv_head=i//4]
        // attn_score[q_h, k_pos] = (q[0, q_h] · k_cache[k_pos, q_h//4]) / sqrt(head_dim)
        let scale = 1.0 / (head_dim as f64).sqrt();
        // Per-head loop (honest PoC, clearer than broadcast gymnastics)
        let mut attn_outs = Vec::with_capacity(num_heads);
        for h in 0..num_heads {
            let kv_h = h / (num_heads / num_kv_heads);
            let q_h = crate::dispatch::narrow(&q, D::Minus2, h, 1)?.squeeze(1)?; // [1, head_dim]
            let k_h = crate::dispatch::narrow(k_cache, D::Minus2, kv_h, 1)?.squeeze(1)?; // [seq_len, head_dim]
            let v_h = crate::dispatch::narrow(v_cache, D::Minus2, kv_h, 1)?.squeeze(1)?; // [seq_len, head_dim]
            // scores [1, seq_len]
            let scores = q_h.matmul(&k_h.t()?)?.affine(scale, 0.0)?;
            // softmax over seq_len dim (manual: subtract max for numerical
            // stability, exp, normalize — all ops have CUDA impls in candle 0.11).
            let max_scores = scores.max_keepdim(candle_core::D::Minus1)?;
            let shifted = scores.broadcast_sub(&max_scores)?;
            let exp_scores = shifted.exp()?;
            let sum_exp = exp_scores.sum_keepdim(candle_core::D::Minus1)?;
            let weights = exp_scores.broadcast_div(&sum_exp)?;
            // context [1, head_dim]
            let ctx = weights.matmul(&v_h)?;
            attn_outs.push(ctx);
        }
        // Concat all heads back to [1, num_heads * head_dim]
        let attn_concat = Tensor::cat(
            &attn_outs.iter().collect::<Vec<_>>(),
            1,
        )?;

        // O projection
        let o_w = self.builder.get_tensor(&o_name)?;
        let attn_out = attn_concat.matmul(&o_w.t()?)?;
        drop(o_w);

        // Residual
        let hidden_after_attn = (hidden_states + attn_out)?;

        // Post-attention RMSNorm + MLP (unchanged from single-token version)
        let post_ln = self.builder.get_tensor(&post_ln_name)?;
        let post_normed = rms_norm_cuda(&hidden_after_attn, &post_ln, 1e-6)?;
        drop(post_ln);

        let mlp_out = self.mlp_block(layer_idx, &post_normed)?;
        Ok((hidden_after_attn + mlp_out)?)
    }

    /// Auto-regressive decode loop: feed token, get argmax, repeat N times.
    /// Each iteration appends to KV cache (real autoregressive behavior).
    /// RoPE + QK-norm deferred (output quality not coherent Qwen3 but pipeline works).
    pub fn decode(&self, initial_token: u32, max_new_tokens: usize) -> Result<Vec<u32>> {
        let mut kv_cache = KVCache::new(self.n_layers, self.builder.device());
        let mut generated = vec![initial_token];

        for _ in 0..max_new_tokens {
            let position = generated.len() - 1;
            let last_token = *generated.last().unwrap();
            let logits = self.forward_with_kv_cache(last_token, position, &mut kv_cache)?;
            let next = self.argmax_token(&logits)?;
            generated.push(next);
        }
        Ok(generated)
    }

    /// Argmax over vocab logits.
pub fn argmax_token(&self, logits: &Tensor) -> Result<u32> {
        let logits_vec: Vec<f32> = logits.squeeze(0)?.to_vec1()?;
        let (idx, _) = logits_vec
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, v)| (i, *v))
            .unwrap_or((0, 0.0));
        Ok(idx as u32)
    }
}