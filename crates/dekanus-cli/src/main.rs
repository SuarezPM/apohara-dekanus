//! Apohara-DeKanus CLI — sparse activation + layer-streaming inference.
//!
//! Phase 1 target: Qwen3-8B smoke ≥ 35 tok/s in-VRAM.
//! Phase 2: Qwen3-30B-A3B 30-40 tok/s.
//! Phase 3: Qwen3-Coder-Next 15-20 tok/s.
//! Phase 5: Ornith-1.0-397B-A17B 12-17.5 tok/s.

use clap::{Parser, Subcommand};
use anyhow::Result;

#[derive(Parser, Debug)]
#[command(name = "dekanus", version, about = "Sparse activation inference on consumer GPUs")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run inference on a model
    Run {
        /// Model identifier (HF repo or local path)
        #[arg(short, long)]
        model: String,

        /// Context length in tokens
        #[arg(short = 'c', long, default_value_t = 4096)]
        ctx: usize,

        /// Maximum new tokens to generate
        #[arg(short = 'n', long, default_value_t = 128)]
        max_new_tokens: usize,

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
        Commands::Run { model, ctx, max_new_tokens, audit } => {
            eprintln!("[Phase 0 skeleton] Run not yet implemented.");
            eprintln!("  model: {model}");
            eprintln!("  ctx: {ctx}");
            eprintln!("  max_new_tokens: {max_new_tokens}");
            eprintln!("  audit: {audit}");
            eprintln!("Implementation lands in Phase 1+.");
            std::process::exit(1);
        }
        Commands::Doctor => {
            eprintln!("Apohara-DeKanus doctor");
            eprintln!("  rustc: {}", rustc_version_runtime());
            eprintln!("  cuda: see nvidia-smi");
            eprintln!("  gpu: see nvidia-smi");
            eprintln!("  (Phase 0 skeleton — full doctor in Phase 1)");
            Ok(())
        }
        Commands::Info => {
            println!("apohara-dekanus {}", env!("CARGO_PKG_VERSION"));
            println!("Workspace crates: airllm-core, dekanus-cli, dekanus-selective,");
            println!("                   dekanus-quant-kv, dekanus-llmlingua2, dekanus-rag,");
            println!("                   dekanus-romy, audit-honesty");
            Ok(())
        }
    }
}

fn rustc_version_runtime() -> &'static str {
    // Phase 0 placeholder; Phase 1 use rustc_version_runtime crate
    "unknown (Phase 0 skeleton)"
}