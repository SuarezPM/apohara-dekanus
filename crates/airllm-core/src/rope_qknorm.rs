//! Qwen3 RoPE (Rotary Position Embedding) with partial_rotary_factor + QK-norm.

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};

/// Precomputed RoPE tables for a range of positions.
pub struct RoPETables {
    pub cos: Vec<Tensor>,
    pub sin: Vec<Tensor>,
    pub rotary_dim: usize,
}

impl RoPETables {
    /// Precompute cos/sin for positions 0..max_seq_len.
    pub fn new(max_seq_len: usize, head_dim: usize, rope_theta: f64, device: &Device) -> Result<Self> {
        let rotary_dim = (head_dim as f64 * 0.25) as usize;
        let half = rotary_dim / 2;
        let mut cos = Vec::with_capacity(max_seq_len);
        let mut sin = Vec::with_capacity(max_seq_len);
        for pos in 0..max_seq_len {
            let mut cos_data = Vec::with_capacity(half);
            let mut sin_data = Vec::with_capacity(half);
            for i in 0..half {
                let theta = rope_theta.powf(-2.0 * (i as f64) / (rotary_dim as f64));
                let angle = (pos as f64) * theta;
                cos_data.push(angle.cos() as f32);
                sin_data.push(angle.sin() as f32);
            }
            cos.push(Tensor::from_vec(cos_data, (half,), device)?);
            sin.push(Tensor::from_vec(sin_data, (half,), device)?);
        }
        Ok(Self { cos, sin, rotary_dim })
    }

    /// Apply partial RoPE to x with cos/sin for given position.
    pub fn apply(&self, x: &Tensor, position: usize) -> Result<Tensor> {
        use candle_core::D;
        // cos/sin are precomputed F32 on CPU. On GPU, x may be BF16 (mixed-dtype
        // model forward). Cast cos/sin to match x's dtype so broadcast_mul works.
        let x_dtype = x.dtype();
        let cos = self.cos[position].to_dtype(x_dtype)?;
        let sin = self.sin[position].to_dtype(x_dtype)?;
        let half = self.rotary_dim / 2;
        let head_dim = x.dim(D::Minus1)?;
        let pass_dim = head_dim - self.rotary_dim;

        let x_rot = crate::dispatch::narrow(x, D::Minus1, 0, self.rotary_dim)?;
        let x_pass = crate::dispatch::narrow(x, D::Minus1, self.rotary_dim, pass_dim)?;

        // Reshape last dim from [rotary_dim] to [half, 2], preserving all leading dims.
// x_rot shape: [..., rotary_dim] -> x_pairs shape: [..., half, 2]
        let x_rot_dims: Vec<usize> = x_rot.dims().to_vec();
        let mut target_shape: Vec<usize> = x_rot_dims[..x_rot_dims.len() - 1].to_vec();
        target_shape.push(half);
        target_shape.push(2);
        let x_pairs = crate::dispatch::reshape(&x_rot, &target_shape)
            .with_context(|| "reshape to pairs")?;
        let x_pairs_shape: Vec<usize> = x_pairs.dims().to_vec();
        let x_real = crate::dispatch::narrow(&x_pairs, D::Minus1, 0, 1)?.squeeze(D::Minus1)?;
        let x_imag = crate::dispatch::narrow(&x_pairs, D::Minus1, 1, 1)?.squeeze(D::Minus1)?;

        let mut cos_shape: Vec<usize> = vec![1; x_pairs_shape.len() - 1];
        if let Some(last) = cos_shape.last_mut() {
            *last = half;
        }
        let cos_b = crate::dispatch::reshape(&cos, &cos_shape)?;
        let sin_b = crate::dispatch::reshape(&sin, &cos_shape)?;

        // Tensor::mul lacks a CUDA kernel for BF16 broadcast (D0015's "possibly
        // others" category). Use the F32 dance: cast both operands to F32, do
        // the mul, cast back. This is the cost of running on candle 0.11's
        // incomplete CUDA coverage; optimized BF16 broadcast_mul kernel is a
        // post-T11 task.
        let x_real_f32 = if x_real.dtype() == DType::F32 { x_real } else { x_real.to_dtype(DType::F32)? };
        let x_imag_f32 = if x_imag.dtype() == DType::F32 { x_imag } else { x_imag.to_dtype(DType::F32)? };
        let cos_b_f32 = if cos_b.dtype() == DType::F32 { cos_b } else { cos_b.to_dtype(DType::F32)? };
        let sin_b_f32 = if sin_b.dtype() == DType::F32 { sin_b } else { sin_b.to_dtype(DType::F32)? };
        let new_real_f32 = (x_real_f32.broadcast_mul(&cos_b_f32)? - x_imag_f32.broadcast_mul(&sin_b_f32)?)?;
        let new_imag_f32 = (x_real_f32.broadcast_mul(&sin_b_f32)? + x_imag_f32.broadcast_mul(&cos_b_f32)?)?;
        let new_real = if new_real_f32.dtype() == DType::F32 && x_dtype == DType::BF16 { new_real_f32.to_dtype(DType::BF16)? } else { new_real_f32 };
        let new_imag = if new_imag_f32.dtype() == DType::F32 && x_dtype == DType::BF16 { new_imag_f32.to_dtype(DType::BF16)? } else { new_imag_f32 };

        let new_pairs = crate::dispatch::stack(&[&new_real, &new_imag], D::Minus1)?;
        // Collapse last two dims (half, 2) back into rotary_dim
        let x_rot_out = crate::dispatch::reshape(&new_pairs, &x_rot_dims)?;

        Ok(crate::dispatch::cat(&[&x_rot_out, &x_pass], D::Minus1)?)
    }
}

/// QK-norm: per-head RMSNorm on Q or K (Qwen3-specific).
/// Uses rms_norm_cuda (CUDA-compatible via mean/sqr/sqrt/div/mul primitives).
pub fn qk_norm(x: &Tensor, weight: &Tensor, eps: f32) -> Result<Tensor> {
    use crate::rms_norm_cuda::rms_norm_cuda;
    Ok(rms_norm_cuda(x, weight, eps as f64)?)
}