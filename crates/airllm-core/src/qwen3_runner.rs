//! Qwen3 forward pass runner.
//!
//! Phase 1b: Qwen3-8B dense via candle-transformers::models::qwen3
//! Phase 2: Qwen3-30B-A3B MoE via candle-transformers::models::qwen3_moe
//! Phase 3: Qwen3-Coder-Next hybrid via custom impl (NOT YET, see engram 1014)

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::qwen3 as qwen3_dense;
use candle_transformers::models::qwen3_moe as qwen3_moe;
use serde::{Deserialize, Serialize};
use std::path::Path;
use tokenizers::Tokenizer;

/// Which Qwen3 variant to load.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Qwen3Variant {
    /// Dense Qwen3 (e.g. Qwen3-8B)
    Dense,
    /// MoE Qwen3 (e.g. Qwen3-30B-A3B)
    Moe,
}

impl Qwen3Variant {
    pub fn from_model_type(model_type: &str) -> Option<Self> {
        match model_type {
            "qwen3" => Some(Self::Dense),
            "qwen3_moe" => Some(Self::Moe),
            _ => None,
        }
    }
}

/// Runtime config for a forward pass.
#[derive(Debug, Clone)]
pub struct RunConfig {
    pub model_path: std::path::PathBuf,
    pub variant: Qwen3Variant,
    pub prompt: String,
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub seed: u64,
}

impl Default for RunConfig {
    fn default() -> Self {
        Self {
            model_path: std::path::PathBuf::new(),
            variant: Qwen3Variant::Dense,
            prompt: String::new(),
            max_new_tokens: 128,
            temperature: 0.0,
            seed: 42,
        }
    }
}

/// Output of a generation run.
#[derive(Debug, Clone)]
pub struct RunOutput {
    pub prompt_tokens: usize,
    pub generated_tokens: usize,
    pub elapsed_secs: f64,
    pub tok_per_sec: f64,
    pub generated_text: String,
}

/// Qwen3 forward pass + greedy sampling runner.
///
/// Uses candle-transformers for model impl; std::fs for safetensors loading.
/// Phase 2 will add glommio + pinned host buffer for layer-streaming.
pub struct Qwen3Runner {
    device: Device,
    dtype: DType,
}

impl Qwen3Runner {
    pub fn new(device: Device, dtype: DType) -> Self {
        Self { device, dtype }
    }

    pub fn cpu() -> Self {
        Self::new(Device::Cpu, DType::F32)
    }

    /// CUDA device (sm_75 supported via vendor-patched candle-kernels).
    /// Returns Self on CUDA device 0 with BF16 dtype (best for Qwen3 8B).
    #[cfg(feature = "cuda")]
    pub fn cuda() -> Result<Self> {
        let device = Device::new_cuda(0).with_context(|| "creating CUDA device 0")?;
        Ok(Self::new(device, DType::BF16))
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn dtype(&self) -> DType {
        self.dtype
    }

    /// Build VarBuilder from safetensors in a directory (mmap).
    /// Requires unsafe due to memmap2 mmap semantics (mmap can SIGBUS on file truncation).
    #[allow(unsafe_code)]
    pub fn load_varbuilder(&self, model_dir: &Path) -> Result<VarBuilder<'_>> {
        let mut paths = Vec::new();
        for entry in std::fs::read_dir(model_dir)? {
            let entry = entry?;
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) == Some("safetensors") {
                paths.push(path);
            }
        }
        paths.sort();
        if paths.is_empty() {
            anyhow::bail!("no safetensors files found in {}", model_dir.display());
        }
        // SAFETY: mmap safety is the caller's responsibility. We hold open the file
        // descriptors via Vec<PathBuf> passed to from_mmaped_safetensors which keeps
        // the mmap'd regions valid for the lifetime of the returned VarBuilder.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&paths, self.dtype, &self.device)
                .with_context(|| "loading safetensors via mmap")?
        };
        Ok(vb)
    }

    /// Load Qwen3 dense model from a directory of safetensors.
    pub fn load_dense(&self, model_dir: &Path) -> Result<qwen3_dense::ModelForCausalLM> {
        let config_path = model_dir.join("config.json");
        let config: qwen3_dense::Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("loading {}", config_path.display()))?,
        )
        .with_context(|| "parsing config.json")?;
        let vb = self.load_varbuilder(model_dir)?;
        Ok(qwen3_dense::ModelForCausalLM::new(&config, vb)?)
    }

    /// Load Qwen3 MoE model from a directory of safetensors.
    pub fn load_moe(&self, model_dir: &Path) -> Result<qwen3_moe::ModelForCausalLM> {
        let config_path = model_dir.join("config.json");
        let config: qwen3_moe::Config = serde_json::from_str(
            &std::fs::read_to_string(&config_path)
                .with_context(|| format!("loading {}", config_path.display()))?,
        )
        .with_context(|| "parsing config.json")?;
        let vb = self.load_varbuilder(model_dir)?;
        Ok(qwen3_moe::ModelForCausalLM::new(&config, vb)?)
    }

    /// Load tokenizer from a directory.
    pub fn load_tokenizer(model_dir: &Path) -> Result<Tokenizer> {
        let tokenizer_path = model_dir.join("tokenizer.json");
        Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading {}: {}", tokenizer_path.display(), e))
    }

    /// Tokenize a prompt and return input_ids tensor.
    pub fn tokenize_prompt(&self, tokenizer: &Tokenizer, prompt: &str) -> Result<Tensor> {
        let encoding = tokenizer
            .encode(prompt, true)
            .map_err(|e| anyhow::anyhow!("encoding prompt: {}", e))?;
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        let input = Tensor::new(ids.as_slice(), &self.device)
            .with_context(|| "creating input tensor")?
            .unsqueeze(0)
            .with_context(|| "unsqueeze batch dim")?;
        Ok(input)
    }

    /// Greedy sampling: return argmax token over last logits row.
    /// Temperature 0.0 = pure greedy; temperature > 0 = scaled sampling via argmax of softmax(logits/T).
    pub fn sample_next(&self, logits: &Tensor, temperature: f32) -> Result<u32> {
        let logits = logits.squeeze(0)?;
        let logits = if temperature > 0.0 {
            (logits / temperature as f64)?
        } else {
            logits
        };
        let next = logits.argmax(candle_core::D::Minus1)?;
        let next = next.to_vec1::<u32>()?;
        Ok(next[0])
    }

    /// Decode token IDs to text.
    pub fn decode(&self, tokenizer: &Tokenizer, tokens: &[u32]) -> Result<String> {
        tokenizer
            .decode(tokens, true)
            .map_err(|e| anyhow::anyhow!("decoding tokens: {}", e))
    }

    /// Generate text from a prompt (greedy decode loop).
    /// Returns RunOutput with timing + generated text.
    pub fn generate_dense(&self, model: &mut qwen3_dense::ModelForCausalLM, cfg: &RunConfig) -> Result<RunOutput> {
        let tokenizer = Self::load_tokenizer(&cfg.model_path)?;
        let input = self.tokenize_prompt(&tokenizer, &cfg.prompt)?;
        let prompt_tokens = input.dim(1)?;

        let start = std::time::Instant::now();
        let mut generated = Vec::with_capacity(cfg.max_new_tokens);
        let mut next_token: u32;

        // Prefill: forward the full prompt
        let mut logits = model.forward(&input, 0)?;
        next_token = self.sample_next(&logits, cfg.temperature)?;
        generated.push(next_token);

        // Decode loop: append token, forward, sample
        let mut position = prompt_tokens;
        for _ in 1..cfg.max_new_tokens {
            let input_next = Tensor::new(&[next_token], &self.device)?
                .unsqueeze(0)?;
            logits = model.forward(&input_next, position)?;
            next_token = self.sample_next(&logits, cfg.temperature)?;
            generated.push(next_token);
            position += 1;

            // Stop on EOS (Qwen3 eos_token_id = 151645 from config.json)
            if next_token == 151645 {
                break;
            }
        }

        let elapsed = start.elapsed().as_secs_f64();
        let gen_count = generated.len();
        let tok_per_sec = if elapsed > 0.0 { gen_count as f64 / elapsed } else { 0.0 };
        let text = self.decode(&tokenizer, &generated)?;

        Ok(RunOutput {
            prompt_tokens,
            generated_tokens: gen_count,
            elapsed_secs: elapsed,
            tok_per_sec,
            generated_text: text,
        })
    }
}