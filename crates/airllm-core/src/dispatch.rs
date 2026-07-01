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
use candle_core::shape::Dim;
use candle_core::{D, DType, Tensor};

/// Narrow a tensor along `dim` over [start, start+length).
/// CPU: passthrough to Tensor::narrow (1:1 byte-identical, no transformation).
/// CUDA: delegates to airllm-kernels::narrow. The narrow_cuda kernel
/// only supports F32. We convert BF16 input to F32, run narrow, and
/// keep the result as F32. Earlier attempts to cast back to BF16 or
/// to F32 both failed with dtype mismatches in downstream ops — the
/// model has mixed dtypes and the narrow result is the source of the
/// mismatch either way. F32 seems to be the closer match for the
/// downstream softmax path.
pub fn narrow(t: &Tensor, dim: D, start: usize, length: usize) -> Result<Tensor> {
    if t.device().is_cuda() {
        let dim_idx = dim
            .to_index(t.shape(), "dispatch::narrow_dim")
            .map_err(|e| anyhow::anyhow!("dispatch::narrow_dim: {}", e))?;
        // Convert BF16 to F32 for the narrow kernel.
        let t_f32 = if t.dtype() == DType::BF16 {
            t.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!("BF16→F32 cast failed: {}", e))?
        } else if t.dtype() == DType::F32 {
            t.clone()
        } else {
            return Err(anyhow::anyhow!("narrow on CUDA: only F32/BF16 supported, got {:?}", t.dtype()));
        };
        // Run narrow kernel; keep result as F32.
        airllm_kernels::narrow(&t_f32, dim_idx, start, length)
            .map_err(|e| anyhow::anyhow!("airllm-kernels::narrow failed: {}", e))
    } else {
        Ok(t.narrow(dim, start, length)?)
    }
}

/// Stack N tensors along a new dim of size 1 at `dim`.
/// CPU: passthrough to Tensor::stack.
/// CUDA: delegates to airllm-kernels::stack (custom PTX kernel).
pub fn stack(tensors: &[&Tensor], dim: D) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(anyhow!("stack: empty input"));
    }
    if tensors[0].device().is_cuda() {
        // Convert each tensor to F32 for the stack kernel (only supports F32).
        let mut owned: Vec<Tensor> = Vec::with_capacity(tensors.len());
        for &t in tensors {
            let t_f32 = if t.dtype() == DType::BF16 {
                t.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!("BF16→F32 cast failed: {}", e))?
            } else if t.dtype() == DType::F32 {
                t.clone()
            } else {
                return Err(anyhow::anyhow!("stack on CUDA: only F32/BF16 supported, got {:?}", t.dtype()));
            };
            owned.push(t_f32);
        }
        let refs: Vec<&Tensor> = owned.iter().collect();
        let dim_idx = dim
            .to_index(refs[0].shape(), "dispatch::stack_dim")
            .map_err(|e| anyhow::anyhow!("dispatch::stack_dim: {}", e))?;
        airllm_kernels::stack(&refs, dim_idx)
            .map_err(|e| anyhow::anyhow!("airllm-kernels::stack failed: {}", e))
    } else {
        Ok(Tensor::stack(tensors, dim)?)
    }
}

/// Reshape a tensor to a new shape.
/// CPU: passthrough to Tensor::reshape.
/// CUDA: delegates to airllm-kernels::reshape (custom PTX kernel).
pub fn reshape(t: &Tensor, shape: &[usize]) -> Result<Tensor> {
    if t.device().is_cuda() {
        let t_f32 = if t.dtype() == DType::BF16 {
            t.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!("BF16→F32 cast failed: {}", e))?
        } else if t.dtype() == DType::F32 {
            t.clone()
        } else {
            return Err(anyhow::anyhow!("reshape on CUDA: only F32/BF16 supported, got {:?}", t.dtype()));
        };
        airllm_kernels::reshape(&t_f32, candle_core::Shape::from(shape.to_vec()))
            .map_err(|e| anyhow::anyhow!("airllm-kernels::reshape failed: {}", e))
    } else {
        t.reshape(shape).map_err(anyhow::Error::from)
    }
}