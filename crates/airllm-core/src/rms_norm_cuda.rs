//! CUDA-compatible RMSNorm via element-wise + reduction primitives.
//!
//! candle_nn::ops::rms_norm only has CPU implementation (no CUDA kernel).
//! This manual impl uses mean/sqr/sqrt/div/mul which all have CUDA kernels.
//! Mathematically identical: rms_norm(x) = x / sqrt(mean(x²) + eps) * weight

use anyhow::Result;
use candle_core::{DType, Tensor};

/// RMSNorm with last-dim reduction. CUDA-compatible.
pub fn rms_norm_cuda(x: &Tensor, weight: &Tensor, eps: f64) -> Result<Tensor> {
    use candle_core::D;
    // Cast to F32 for stable variance computation if needed
    let orig_dtype = x.dtype();
    let x_f32 = if orig_dtype == DType::F32 || orig_dtype == DType::F16 || orig_dtype == DType::BF16 {
        x.to_dtype(DType::F32)?
    } else {
        x.clone()
    };
    // Compute mean(x²) along last dim, keep dim for broadcasting
    let x_sq = x_f32.sqr()?;
    let mean = x_sq.mean_keepdim(D::Minus1)?;
    // rms = sqrt(mean + eps)
    let rms = mean.affine(1.0, eps)?.sqrt()?;
    // normalized = x / rms
    let normalized = x_f32.broadcast_div(&rms)?;
    // scale by weight
    let scaled = normalized.broadcast_mul(&weight.to_dtype(DType::F32)?)?;
    // Cast back to original dtype
    Ok(scaled.to_dtype(orig_dtype)?)
}