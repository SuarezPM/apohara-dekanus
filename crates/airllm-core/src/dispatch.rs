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
/// CUDA: dispatches to the dtype-matching narrow kernel in airllm-kernels.
/// For BF16 input, the BF16 narrow kernel is used (added in D0026); output
/// is BF16. For F32 input, the F32 narrow kernel is used; output is F32.
/// Preserving input dtype avoids the dtype-mismatch issue documented in D0025.
pub fn narrow(t: &Tensor, dim: D, start: usize, length: usize) -> Result<Tensor> {
    if t.device().is_cuda() {
        let dim_idx = dim
            .to_index(t.shape(), "dispatch::narrow_dim")
            .map_err(|e| anyhow::anyhow!("dispatch::narrow_dim: {}", e))?;
        match t.dtype() {
            DType::BF16 => airllm_kernels::narrow(t, dim_idx, start, length)
                .map_err(|e| anyhow::anyhow!("airllm-kernels::narrow (BF16) failed: {}", e)),
            DType::F32 => airllm_kernels::narrow(t, dim_idx, start, length)
                .map_err(|e| anyhow::anyhow!("airllm-kernels::narrow (F32) failed: {}", e)),
            other => Err(anyhow::anyhow!(
                "narrow on CUDA: only F32/BF16 supported, got {:?}",
                other
            )),
        }
    } else {
        Ok(t.narrow(dim, start, length)?)
    }
}

/// Stack N tensors along a new dim of size 1 at `dim`.
/// CPU: passthrough to Tensor::stack.
/// CUDA: delegates to airllm-kernels::stack (F32 kernel). For BF16 inputs,
/// the kernel runs on F32 and the result is cast back to BF16 to preserve
/// dtype (avoiding the matmul/cat dtype-mismatch issue documented in D0025).
pub fn stack(tensors: &[&Tensor], dim: D) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(anyhow!("stack: empty input"));
    }
    if tensors[0].device().is_cuda() {
        let original_dtype = tensors[0].dtype();
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
        let out_f32 = airllm_kernels::stack(&refs, dim_idx)
            .map_err(|e| anyhow::anyhow!("airllm-kernels::stack failed: {}", e))?;
        if original_dtype == DType::BF16 {
            out_f32.to_dtype(DType::BF16).map_err(|e| anyhow::anyhow!("F32→BF16 cast failed: {}", e))
        } else {
            Ok(out_f32)
        }
    } else {
        Ok(Tensor::stack(tensors, dim)?)
    }
}

/// Concatenate N tensors along `dim`.
/// CPU: passthrough to Tensor::cat.
/// CUDA: Tensor::cat lacks a BF16 CUDA kernel in candle 0.11. We convert
/// all inputs to F32, run cat (F32 has a kernel), then cast the result
/// back to the original dtype. This preserves dtype and unblocks the
/// rope_qknorm cat. Same pattern as reshape/stack.
pub fn cat<Dm: candle_core::shape::Dim>(tensors: &[&Tensor], dim: Dm) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(anyhow!("cat: empty input"));
    }
    if tensors[0].device().is_cuda() {
        let original_dtype = tensors[0].dtype();
        let mut owned: Vec<Tensor> = Vec::with_capacity(tensors.len());
        for &t in tensors {
            let t_f32 = if t.dtype() == DType::BF16 {
                t.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!("BF16→F32 cast failed: {}", e))?
            } else if t.dtype() == DType::F32 {
                t.clone()
            } else {
                return Err(anyhow::anyhow!("cat on CUDA: only F32/BF16 supported, got {:?}", t.dtype()));
            };
            owned.push(t_f32);
        }
        let refs: Vec<&Tensor> = owned.iter().collect();
        let dim_idx = dim
            .to_index(refs[0].shape(), "dispatch::cat_dim")
            .map_err(|e| anyhow::anyhow!("dispatch::cat_dim: {}", e))?;
        let out_f32 = Tensor::cat(&refs, dim_idx).map_err(|e| anyhow::anyhow!("cat failed: {}", e))?;
        if original_dtype == DType::BF16 {
            out_f32.to_dtype(DType::BF16).map_err(|e| anyhow::anyhow!("F32→BF16 cast failed: {}", e))
        } else {
            Ok(out_f32)
        }
    } else {
        Ok(Tensor::cat(tensors, dim)?)
    }
}

/// Reshape a tensor to a new shape.
/// CPU: passthrough to Tensor::reshape.
/// CUDA: delegates to airllm-kernels::reshape (F32 kernel). For BF16 input,
/// the kernel runs on F32 and the result is cast back to BF16 to preserve
/// dtype (avoiding the matmul dtype-mismatch issue documented in D0025).
pub fn reshape(t: &Tensor, shape: &[usize]) -> Result<Tensor> {
    if t.device().is_cuda() {
        let original_dtype = t.dtype();
        let t_f32 = if original_dtype == DType::BF16 {
            t.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!("BF16→F32 cast failed: {}", e))?
        } else if original_dtype == DType::F32 {
            t.clone()
        } else {
            return Err(anyhow::anyhow!("reshape on CUDA: only F32/BF16 supported, got {:?}", original_dtype));
        };
        let out_f32 = airllm_kernels::reshape(&t_f32, candle_core::Shape::from(shape.to_vec()))
            .map_err(|e| anyhow::anyhow!("airllm-kernels::reshape failed: {}", e))?;
        if original_dtype == DType::BF16 {
            out_f32.to_dtype(DType::BF16).map_err(|e| anyhow::anyhow!("F32→BF16 cast failed: {}", e))
        } else {
            Ok(out_f32)
        }
    } else {
        t.reshape(shape).map_err(anyhow::Error::from)
    }
}

/// Element-wise add. CPU: passthrough to `+`. CUDA: dispatches to the
/// dtype-matching kernel in airllm-kernels. BF16 path uses the custom
/// `add_bf16` kernel (D0029, item 1 of perf roadmap) — no F32 dance.
/// The start_offset fix (D0029 debug) makes the kernel work in the model
/// path by slicing the device buffer at layout.start_offset() before
/// passing to the kernel.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.device().is_cuda() {
        airllm_kernels::add(a, b).map_err(|e| anyhow::anyhow!("airllm-kernels::add: {}", e))
    } else {
        Ok((a + b)?)
    }
}

/// Element-wise mul with F32 dance on CUDA. Same rationale as `add`.
pub fn mul(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.device().is_cuda() {
        let target_dtype = a.dtype();
        let a_f32 = a.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!("a→F32: {}", e))?;
        let b_f32 = b.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!("b→F32: {}", e))?;
        let out_f32 = (&a_f32 * &b_f32).map_err(|e| anyhow::anyhow!("mul (F32): {}", e))?;
        if target_dtype == DType::BF16 {
            out_f32.to_dtype(DType::BF16).map_err(|e| anyhow::anyhow!("F32→BF16: {}", e))
        } else {
            Ok(out_f32)
        }
    } else {
        Ok((a * b)?)
    }
}

/// SiLU activation with F32 dance on CUDA. candle_nn::ops::silu is
/// not CUDA-aware in 0.11; we manually compute x * sigmoid(x) in F32.
pub fn silu(t: &Tensor) -> Result<Tensor> {
    if t.device().is_cuda() {
        let target_dtype = t.dtype();
        let t_f32 = t.to_dtype(DType::F32).map_err(|e| anyhow::anyhow!("t→F32: {}", e))?;
        // sigmoid(x) = 1 / (1 + exp(-x))
        let neg = t_f32.affine(-1.0, 0.0).map_err(|e| anyhow::anyhow!("neg: {}", e))?;
        let exp_neg = neg.exp().map_err(|e| anyhow::anyhow!("exp: {}", e))?;
        // 1 + exp(-x) via affine (1*exp_neg + 1)
        let one_plus = exp_neg.affine(1.0, 1.0).map_err(|e| anyhow::anyhow!("1+exp: {}", e))?;
        let sigmoid = one_plus.recip().map_err(|e| anyhow::anyhow!("recip: {}", e))?;
        let silu_out = (&t_f32 * &sigmoid).map_err(|e| anyhow::anyhow!("silu: {}", e))?;
        if target_dtype == DType::BF16 {
            silu_out.to_dtype(DType::BF16).map_err(|e| anyhow::anyhow!("F32→BF16: {}", e))
        } else {
            Ok(silu_out)
        }
    } else {
        Ok(candle_nn::ops::silu(t)?)
    }
}