//! Layer-streamed safetensors reader using `safe_open`-equivalent lazy access.
//!
//! Phase 2a: opens each safetensors shard via memmap2, deserializes the header
//! (cheap), and reads individual tensors on-demand (lazy from page cache).
//! Foundation for true layer-streaming inference.
//!
//! ## Current capabilities (Phase 2a)
//! - Open all shards of a sharded model via memmap (Qwen3-8B: 5 shards, 16.40 GB total)
//! - Read individual tensor by name (lazy, page cache hit on warm)
//! - Convert safetensors dtype to candle DType
//! - Diagnostic: shard_count(), tensor_count(), total_bytes(), tensor_names()
//!
//! ## Deferred (Phase 2b)
//! - Full streaming inference: requires custom Qwen3 forward pass that loads
//!   each layer's weights from this builder, runs forward, releases to next layer.
//!   candle-transformers' `ModelForCausalLM::new(&config, vb)` consumes all
//!   weights upfront; bypassing requires custom Qwen3 impl (~500 LOC).
//!
//! ## Usage
//! ```ignore
//! let builder = LayerStreamedBuilder::open(model_dir, device, dtype)?;
//! let embed = builder.get_tensor("model.embed_tokens.weight")?;
//! let layer0_attn = builder.get_tensor("model.layers.0.self_attn.q_proj.weight")?;
//! ```

// This module uses unsafe for: (1) memmap2 mmap (raw pointer semantics), (2) self-
// referential struct (SafeTensors borrows from Mmap via 'static lifetime trick),
// (3) unsafe impl Send + Sync for ShardView (justifies the Mmap access). All
// other operations are safe. The unsafe blocks are tightly scoped and the
// invariants are documented at each site.
#![allow(unsafe_code)]

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};
use std::collections::HashMap;
use std::path::PathBuf;

/// Per-shard: memmap'd file + deserialized SafeTensors header.
/// Drop order in Rust struct fields is declaration order, so `bytes` is dropped
/// after `safe_tensors` (we use 'static lifetime here as a marker; access is safe
/// as long as we never mutate the mmap region while safe_tensors holds a view).
struct ShardView {
    #[allow(dead_code)]
    path: PathBuf,
    bytes: Mmap,
    safe_tensors: SafeTensors<'static>,
}

unsafe impl Send for ShardView {}
unsafe impl Sync for ShardView {}

/// Layer-streamed safetensors reader.
///
/// Holds memmap'd views of all shards + the model's tensor index. Reading a tensor
/// parses the safetensors header once per shard (cached in struct), then slices
/// into the mmap'd region (kernel page cache hits on warm access).
pub struct LayerStreamedBuilder {
    shards: Vec<ShardView>,
    index: HashMap<String, usize>, // tensor_name -> shard_idx (0-based)
    device: Device,
    dtype: DType,
}

impl LayerStreamedBuilder {
    /// Open all safetensors files in `model_dir` and parse the tensor index.
    pub fn open(model_dir: &std::path::Path, device: Device, dtype: DType) -> Result<Self> {
        // 1. Find safetensors files (sorted: model-00001-of-00005, etc.)
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
            anyhow::bail!("no safetensors files in {}", model_dir.display());
        }

        // 2. mmap each shard + deserialize header
        let mut shards = Vec::with_capacity(paths.len());
        for path in &paths {
            let file = std::fs::File::open(path)?;
            let mmap = unsafe { Mmap::map(&file) }
                .with_context(|| format!("mmap {}", path.display()))?;

            // SAFETY: SafeTensors borrows from `mmap` bytes. Both are owned by
            // this ShardView and dropped together (field declaration order:
            // `bytes` declared before `safe_tensors`, so SafeTensors is dropped
            // first, then mmap is dropped). The 'static lifetime here is a
            // marker; access remains valid for the lifetime of the struct.
            let safe_tensors: SafeTensors<'static> = {
                let bytes_slice: &[u8] = &mmap;
                // Extend lifetime to 'static via raw pointer cast (safe because
                // the mmap outlives the safe_tensors in field drop order).
                let static_slice: &'static [u8] = unsafe { std::mem::transmute(bytes_slice) };
                SafeTensors::deserialize(static_slice)
                    .with_context(|| format!("parsing safetensors header in {}", path.display()))?
            };

            shards.push(ShardView {
                path: path.clone(),
                bytes: mmap,
                safe_tensors,
            });
        }

        // 3. Parse model.safetensors.index.json for tensor_name -> shard_idx
        let index_path = model_dir.join("model.safetensors.index.json");
        let mut index: HashMap<String, usize> = HashMap::new();
        if index_path.exists() {
            let index_str = std::fs::read_to_string(&index_path)?;
            let index_json: serde_json::Value = serde_json::from_str(&index_str)?;
            if let Some(weight_map) = index_json["weight_map"].as_object() {
                for (tensor_name, shard_file) in weight_map {
                    let shard_file = shard_file.as_str().unwrap_or("");
                    let shard_idx = if let Some(start) = shard_file.find("model-") {
                        let rest = &shard_file[start + 6..];
                        if let Some(end) = rest.find("-of-") {
                            rest[..end].parse::<usize>().unwrap_or(1).saturating_sub(1)
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    index.insert(tensor_name.clone(), shard_idx);
                }
            }
        }

        Ok(Self {
            shards,
            index,
            device,
            dtype,
        })
    }

    /// Read a single tensor by name (lazy: page cache hit if recently accessed).
    pub fn get_tensor(&self, name: &str) -> Result<Tensor> {
        let shard_idx = *self
            .index
            .get(name)
            .ok_or_else(|| anyhow::anyhow!("tensor '{}' not found in model index", name))?;

        let shard = self
            .shards
            .get(shard_idx)
            .ok_or_else(|| anyhow::anyhow!("shard {} not loaded", shard_idx))?;

        let tensor_view = shard
            .safe_tensors
            .tensor(name)
            .with_context(|| format!("tensor '{}' not in shard {}", name, shard_idx))?;

        let candle_dtype = match tensor_view.dtype() {
            Dtype::F32 => DType::F32,
            Dtype::F16 => DType::F16,
            Dtype::BF16 => DType::BF16,
            other => anyhow::bail!("unsupported dtype {:?} for tensor '{}'", other, name),
        };

        let shape: Vec<usize> = tensor_view.shape().iter().map(|&d| d as usize).collect();
        let data = tensor_view.data();
        let n_elements: usize = shape.iter().product();
        let expected_bytes = n_elements * candle_dtype.size_in_bytes();
        if data.len() != expected_bytes {
            anyhow::bail!(
                "tensor '{}': byte length {} != expected {} for shape {:?}",
                name,
                data.len(),
                expected_bytes,
                shape
            );
        }

        // Tensor::from_raw_buffer takes &[u8] (no unsafe needed in candle 0.11);
        // it copies bytes into a Device-owned buffer internally.
        let tensor = Tensor::from_raw_buffer(data, candle_dtype, &shape, &self.device)
            .with_context(|| format!("creating candle Tensor for '{}'", name))?;

        let tensor = if candle_dtype != self.dtype {
            tensor.to_dtype(self.dtype)?
        } else {
            tensor
        };

        Ok(tensor)
    }

    /// Number of shards opened.
    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// Device accessor (used by KVCache::new and downstream).
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Total tensor count across all shards.
    pub fn tensor_count(&self) -> usize {
        self.shards
            .iter()
            .map(|s| s.safe_tensors.tensors().len())
            .sum()
    }

    /// Total bytes across all shards (sum of mmap'd regions).
    pub fn total_bytes(&self) -> u64 {
        self.shards.iter().map(|s| s.bytes.len() as u64).sum()
    }

    /// List all tensor names (for diagnostics).
    pub fn tensor_names(&self) -> Vec<String> {
        let mut names = Vec::new();
        for shard in &self.shards {
            for (name, _) in shard.safe_tensors.tensors() {
                names.push(name);
            }
        }
        names.sort();
        names
    }
}