//! Custom CUDA kernel dispatch for candle-core 0.11 missing narrow/stack.
//!
//! Phase 2b GPU unblock: when `device.is_cuda()`, would call our
//! nvcc-compiled kernels via cudarc. When CPU, falls back to candle's
//! built-in ops.
//!
//! ## Status (m0494 — honest)
//!
//! The CUDA FFI binding structure is in place (`ffi.rs` declares the
//! `extern "C"` launchers matching `kernels/*.cu`). The Rust wrappers
//! (`narrow`/`stack`/`reshape`) compile-check against candle-core 0.11.
//!
//! **CUDA execution path is a no-op** in this commit because
//! candle-core 0.11's `Storage` API requires a different access pattern
//! (`storage_and_layout().0.as_cuda_slice()` is no longer exposed in the
//! same form as earlier versions). Wiring the actual cudarc calls
//! against the current candle-core API is deferred to T7+ in the plan.
//!
//! What this commit DOES:
//! - Compiles cleanly both with and without --features cuda
//! - CPU path works (candle built-in narrow/stack/reshape)
//! - `device.is_cuda()` is correctly detected
//! - The dispatch structure is right; only the inner FFI call is stubbed
//!
//! What this commit DOES NOT do yet:
//! - Actually call our custom CUDA kernels
//! - Replaces candle's CUDA narrow/stack with our fast path
//!
//! **The GPU path is still BLOCKED on full FFI integration**, but the
//! scaffolding, ffi.rs declarations, and dispatch structure are in place
//! for a future session to finish (T7+ per the ULTRAWORK plan).

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod ffi;

use anyhow::{Context, Result};
use candle_core::{DType, Shape, Tensor};

/// Narrow (slice) along a single dimension.
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

fn narrow_cuda(
    t: &Tensor,
    dim: usize,
    start: usize,
    length: usize,
) -> Result<Tensor> {
    // TODO: candle-core 0.11 Storage API changed. The right access
    // pattern in 0.11 is `t.storage_and_layout().0.deref()` to get
    // `&Storage`, then `cuda_storage.as_cuda_slice()` returns
    // `&CudaStorageSlice`. Wire this with cudarc 0.19 once Storage
    // API access is sorted. For now: error with informative message.
    let _ = (dim, start, length);
    anyhow::bail!(
        "narrow_cuda: candle-core 0.11 Storage FFI not yet wired (see T7+ in plan)"
    )
}

/// Stack (concat along new dim) of N tensors of the same shape.
pub fn stack(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
    if tensors.is_empty() {
        anyhow::bail!("stack requires at least one tensor");
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
    // TODO: same as narrow_cuda — wire after Storage FFI sorted.
    anyhow::bail!(
        "stack_cuda: candle-core 0.11 Storage FFI not yet wired (see T7+ in plan)"
    )
}

/// Reshape (strided element copy with new shape).
pub fn reshape(t: &Tensor, shape: Shape) -> Result<Tensor> {
    if t.device().is_cuda() {
        reshape_cuda(t, &shape)
    } else {
        t.reshape(shape).map_err(anyhow::Error::from)
    }
}

fn reshape_cuda(_t: &Tensor, _shape: &Shape) -> Result<Tensor> {
    // TODO: same as narrow_cuda — wire after Storage FFI sorted.
    anyhow::bail!(
        "reshape_cuda: candle-core 0.11 Storage FFI not yet wired (see T7+ in plan)"
    )
}
