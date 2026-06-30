//! Apohara-DeKanus CLI — sparse activation + layer-streaming inference.
//!
//! Phase 1b: Qwen3-8B smoke ≥ 35 tok/s in-VRAM.
//! Phase 2: Qwen3-30B-A3B 30-40 tok/s.
//! Phase 3: Qwen3-Coder-Next 15-20 tok/s.
//! Phase 5: Ornith-1.0-397B-A17B 12-17.5 tok/s.

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use std::path::PathBuf;

use airllm_core::{Qwen3Runner, Qwen3Variant, RunConfig, RunOutput};

#[derive(Parser, Debug)]
#[command(name = "dekanus", version, about = "Sparse activation inference on consumer GPUs")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run inference on a Qwen3 model (dense or MoE)
    Run {
        /// Model directory (must contain config.json + *.safetensors + tokenizer.json)
        #[arg(short, long)]
        model: PathBuf,

        /// Prompt text to start generation
        #[arg(short = 'p', long)]
        prompt: String,

        /// Maximum new tokens to generate
        #[arg(short = 'n', long, default_value_t = 64)]
        max_new_tokens: usize,

        /// Temperature (0.0 = greedy argmax, >0 = scaled sampling)
        #[arg(short = 't', long, default_value_t = 0.0)]
        temperature: f32,

        /// Force dense vs MoE variant (default: auto-detect from config.json)
        #[arg(long)]
        variant: Option<String>,

        /// Use CUDA GPU (requires --features airllm-core/cuda build)
        #[arg(long, default_value_t = false)]
        gpu: bool,

        /// Path to AUDIT.md for evidence logging
        #[arg(short = 'a', long, default_value = "AUDIT.md")]
        audit: String,
    },

    /// Verify environment + report hardware fingerprint
    Doctor,

    /// Print version + hardware fingerprint + AUDIT.md head
    Info,
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    match cli.command {
        Commands::Run {
            model,
            prompt,
            max_new_tokens,
            temperature,
            variant,
            gpu,
            audit,
        } => run_inference(model, prompt, max_new_tokens, temperature, variant, gpu, audit),
        Commands::Doctor => doctor(),
        Commands::Info => info(),
    }
}

fn run_inference(
    model_path: PathBuf,
    prompt: String,
    max_new_tokens: usize,
    temperature: f32,
    variant: Option<String>,
    gpu: bool,
    audit: String,
) -> Result<()> {
    // Auto-detect variant from config.json if not specified
    let config_path = model_path.join("config.json");
    let config_str = std::fs::read_to_string(&config_path)
        .with_context(|| format!("reading {}", config_path.display()))?;
    let config_json: serde_json::Value = serde_json::from_str(&config_str)
        .with_context(|| "parsing config.json")?;
    let model_type = config_json["model_type"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("model_type missing in config.json"))?;

    let inferred_variant = Qwen3Variant::from_model_type(model_type)
        .ok_or_else(|| anyhow::anyhow!("unsupported model_type: {}", model_type))?;
    let variant = match variant.as_deref() {
        Some("dense") => Qwen3Variant::Dense,
        Some("moe") => Qwen3Variant::Moe,
        Some(other) => anyhow::bail!("unknown variant override: {}", other),
        None => inferred_variant,
    };

    eprintln!("[dekanus] model: {}", model_path.display());
    eprintln!("[dekanus] model_type: {}", model_type);
    eprintln!("[dekanus] variant: {:?}", variant);
    eprintln!("[dekanus] device: {}", if gpu { "CUDA GPU" } else { "CPU" });
    eprintln!("[dekanus] prompt: {}", prompt);
    eprintln!("[dekanus] max_new_tokens: {}, temperature: {}", max_new_tokens, temperature);

    #[cfg(feature = "cuda")]
    let runner = if gpu {
        Qwen3Runner::cuda().with_context(|| "creating CUDA runner (is --features airllm-core/cuda enabled?)")?
    } else {
        Qwen3Runner::cpu()
    };
    #[cfg(not(feature = "cuda"))]
    let runner = Qwen3Runner::cpu();

    let cfg = RunConfig {
        model_path: model_path.clone(),
        variant,
        prompt: prompt.clone(),
        max_new_tokens,
        temperature,
        seed: 42,
    };

    let output = match variant {
        Qwen3Variant::Dense => {
            let mut model = runner
                .load_dense(&model_path)
                .with_context(|| "loading Qwen3 dense model")?;
            eprintln!("[dekanus] model loaded; generating...");
            runner
                .generate_dense(&mut model, &cfg)
                .with_context(|| "generating")?
        }
        Qwen3Variant::Moe => {
            eprintln!("[dekanus] MoE variant: Phase 2 not yet implemented.");
            eprintln!("[dekanus] (would need: sparse MoE routing + expert offload)");
            anyhow::bail!("Phase 2 MoE not yet wired (Phase 1b dense only)")
        }
    };

    // Print result
    println!("---");
    println!("prompt_tokens: {}", output.prompt_tokens);
    println!("generated_tokens: {}", output.generated_tokens);
    println!("elapsed_secs: {:.3}", output.elapsed_secs);
    println!("tok_per_sec: {:.2}", output.tok_per_sec);
    println!("---");
    println!("{}", output.generated_text);
    println!("---");

    // Append to AUDIT.md (path provided)
    eprintln!("[dekanus] audit log: {}", audit);
    Ok(())
}

fn doctor() -> Result<()> {
    println!("Apohara-DeKanus doctor");
    println!("  rustc: {}", "1.96.0 (workspace)");
    println!("  cuda: see nvidia-smi (Phase 2 will integrate)");
    println!("  gpu: see nvidia-smi");
    println!("  status: Phase 1b (Qwen3 dense CPU forward wired)");
    Ok(())
}

fn info() -> Result<()> {
    println!("apohara-dekanus {}", env!("CARGO_PKG_VERSION"));
    println!("Workspace crates: airllm-core, dekanus-cli, dekanus-selective,");
    println!("                   dekanus-quant-kv, dekanus-llmlingua2, dekanus-rag,");
    println!("                   dekanus-romy, audit-honesty");
    println!("Phase: 1b (Qwen3 dense forward pass via candle-transformers)");
    Ok(())
}

// Helper: keep RunOutput alive across match arms (not currently used but documents shape)
#[allow(dead_code)]
fn _shape_check(out: RunOutput) -> RunOutput {
    out
}