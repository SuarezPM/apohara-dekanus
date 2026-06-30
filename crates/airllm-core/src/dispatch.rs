//! Centralized dispatch shim for ops that lack CUDA kernels in candle-core 0.11.
//!
//! Phase 2b GPU unblock plan (Wave 5+): when custom CUDA kernels exist in
//! `airllm-kernels`, this shim will route to them on CUDA device.
//! Until then (T6.5 fix pending), this is a CPU-only passthrough that
//! preserves original candle API behavior exactly.
//!
//! API matches candle's Tensor::narrow / Tensor::stack exactly (D enum parameter),
//! so call sites can be wired by adding `crate::dispatch::` prefix without
//! any other change. This is the key difference from the previous attempt
//! (which translated to integer and introduced a semantic bug).

use anyhow::{anyhow, Result};
use candle_core::{D, Tensor};

/// Narrow a tensor along `dim` over [start, start+length).
/// CPU: passthrough to Tensor::narrow (1:1 byte-identical, no transformation).
/// CUDA: error with actionable message (custom kernel pending T6.5).
pub fn narrow(t: &Tensor, dim: D, start: usize, length: usize) -> Result<Tensor> {
    if t.device().is_cuda() {
        Err(anyhow!(
            "narrow CUDA dispatch not yet wired (T6.5 deferred). \
             CPU passthrough works; for GPU, see AUDIT D0019."
        ))
    } else {
        Ok(t.narrow(dim, start, length)?)
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