//! Custom CUDA kernel dispatch for candle-core 0.11 narrow/stack/reshape.
//!
//! ## T6.5 honest end-of-session (m0648)
//!
//! Approach A (vendor-patch candle-core) and Approach B (bypass candle Tensor
//! with cudarc::driver::CudaDevice::new(0)) both encountered API mismatches
//! after 5+ iterations each. Reversed to placeholder (bail! on CUDA path,
//! CPU passthrough unchanged). The cudarc 0.19 API path that worked in
//! candle's own source (candle-nn/src/rotary_emb.rs) requires a fresh `Arc<CudaContext>`
//! + `Arc<CudaStream>` shared with candle — not achievable from outside the
//! candle crate without vendor-patching or an alternative cudarc 0.19 path
//! that we couldn't identify in remaining context.
//!
//! ## Path forward (T6.5 deferred, 2-4h dedicated session)
//!
//! 1. Read candle-nn/src/rotary_emb.rs (~150 LOC) to extract the canonical
//!    pattern: candle keeps its own `Arc<CudaContext>` alive internally, and
//!    the custom kernel API takes the cudarc Device+Stream. The path is to
//!    use the `candle::CudaDevice` (the wrapper)'s public methods to get
//!    `cuda_stream()` and a way to find the cudarc Device — and that path
//!    doesn't exist publicly in 0.11. So vendor-patch is the only path.
//! 2. Vendor-patch candle-core: add `pub fn cudarc_device(&self) -> cudarc::driver::CudaDevice`
//!    that returns `CudaDevice::new(self.context.clone(), self.stream.clone(), self.id.as_index())`.
//! 3. Re-vendor via [patch.crates-io] candle-core = { path = "vendor/candle-core" }.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod ffi;

use anyhow::{Context, Result};
use candle_core::{DType, Shape, Tensor};

/// Public API: narrow a tensor (CPU → candle built-in; CUDA → placeholder).
pub fn narrow(
    t: &Tensor,
    dim: usize,
    start: usize,
    length: usize,
) -> Result<Tensor> {
    if t.device().is_cuda() {
        narrow_cuda(t, dim, start, length)
    } else {
        t.narrow(dim, start, length).map_err(anyhow::Error::from)
    }
}

fn narrow_cuda(_t: &Tensor, _dim: usize, _start: usize, _length: usize) -> Result<Tensor> {
    Err(anyhow::anyhow!(
        "narrow_cuda: deferred (T6.5 reversed m0648). CPU passthrough works; \
         for GPU, see AUDIT D0019/D0021/D0024."
    ))
}

/// Public API: stack N tensors along new dim.
pub fn stack(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
    if tensors.is_empty() {
        return Err(anyhow::anyhow!("stack requires at least one tensor"));
    }
    if tensors[0].device().is_cuda() {
        stack_cuda(tensors, dim)
    } else {
        let owned: Vec<Tensor> = tensors.iter().map(|&t| t.clone()).collect();
        let refs: Vec<&Tensor> = owned.iter().collect();
        Tensor::stack(&refs, dim).map_err(anyhow::Error::from)
    }
}

fn stack_cuda(_tensors: &[&Tensor], _dim: usize) -> Result<Tensor> {
    Err(anyhow::anyhow!(
        "stack_cuda: deferred (T6.5 reversed m0648)"
    ))
}

/// Public API: reshape.
pub fn reshape(t: &Tensor, shape: Shape) -> Result<Tensor> {
    if t.device().is_cuda() {
        reshape_cuda(t, &shape)
    } else {
        t.reshape(shape).map_err(anyhow::Error::from)
    }
}

fn reshape_cuda(_t: &Tensor, _shape: &Shape) -> Result<Tensor> {
    Err(anyhow::anyhow!(
        "reshape_cuda: deferred (T6.5 reversed m0648)"
    ))
}