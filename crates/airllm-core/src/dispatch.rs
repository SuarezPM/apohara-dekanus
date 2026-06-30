//! Centralized dispatch shim for ops that lack CUDA kernels in candle-core 0.11.
//!
//! Phase 2b GPU unblock plan (Wave 5):
//! - T7: replace .narrow(...) + .squeeze(...) and Tensor::stack(...) in rope_qknorm::apply
//! - T8: replace .narrow(...) + .squeeze(...) in forward_layer_with_kv lines 404-406
//!
//! Behavior:
//! - CPU device: passthrough to candle's Tensor::narrow / Tensor::stack (fast, no overhead)
//! - CUDA device: error with actionable message (custom kernel plumbing deferred to
//!   T6.5 fix — see AUDIT D0019 for blocker analysis)
//!
//! This is a SAFE non-breaking change: CPU F32 path (0.04 tok/s baseline) preserved
//! exactly, GPU path now gives clean error instead of CUDA_ERROR_NOT_FOUND.

use anyhow::{anyhow, Result};
use candle_core::{D, Tensor};

/// Narrow a tensor along `dim` over [start, start+length).
/// CPU: passthrough to Tensor::narrow.
/// CUDA: error (kernel not yet integrated).
/// `dim` is an axis index: positive = 0-based axis, negative = from-end axis
/// (e.g. -1 = last axis, -2 = second-to-last, matching candle's D enum convention).
/// Only supports rank 1-2 tensors (candle's D enum has only Minus1 and Minus2).
pub fn narrow(t: &Tensor, dim: i64, start: usize, length: usize) -> Result<Tensor> {
    if t.device().is_cuda() {
        Err(anyhow!(
            "narrow CUDA dispatch not yet wired (T6.5 fix pending). \
             CPU passthrough works; for GPU, see AUDIT D0019."
        ))
    } else {
        let rank = t.dims().len() as i64;
        let normalized = if dim < 0 { (dim + rank) as usize } else { dim as usize };
        let d = match normalized {
            1 => D::Minus1,
            2 => D::Minus2,
            _ => return Err(anyhow!("narrow: dim index {} not supported (rank={})", dim, rank)),
        };
        Ok(t.narrow(d, start, length)?)
    }
}

/// Stack N tensors along a new dim of size 1 at `dim`.
/// CPU: passthrough to Tensor::stack.
/// CUDA: error.
pub fn stack(tensors: &[&Tensor], dim: D) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(anyhow!("stack: empty input"));
    }
    if tensors[0].device().is_cuda() {
        Err(anyhow!(
            "stack CUDA dispatch not yet wired (T6.5 fix pending). \
             CPU passthrough works; for GPU, see AUDIT D0019."
        ))
    } else {
        Ok(Tensor::stack(tensors, dim)?)
    }
}

/// Reshape a tensor to a new shape.
/// CPU: passthrough to Tensor::reshape.
/// CUDA: error.
pub fn reshape(t: &Tensor, shape: &[usize]) -> Result<Tensor> {
    if t.device().is_cuda() {
        Err(anyhow!(
            "reshape CUDA dispatch not yet wired (T6.5 fix pending). \
             CPU passthrough works; for GPU, see AUDIT D0019."
        ))
    } else {
        Ok(t.reshape(shape)?)
    }
}