//! Custom CUDA kernel dispatch for candle-core 0.11 missing narrow/stack/reshape.
//!
//! ## Status (m0582 — honest end-of-session)
//!
//! The FFI binding structure is in place (`ffi.rs` declares the `extern "C"`
//! launchers matching `kernels/*.cu`). The Rust wrappers compile cleanly.
//!
//! **CUDA execution path is a no-op** because extracting the cudarc `Device`
//! handle from candle's `CudaDevice` wrapper requires non-public candle
//! internals (`inner()` method does not exist on `candle_core::CudaDevice`).
//! This is a candle-core 0.11 API design choice — the wrapper is not
//! intended to be unwrapped from outside.
//!
//! ## Path forward (T6.5 deferred)
//!
//! 1. **Approach A**: vendor-patch candle-core to expose `inner()` on CudaDevice
//!    (or implement the cudarc kernel launch via the existing `candle_kernels`
//!    crate pattern that the candle team itself uses).
//! 2. **Approach B**: bypass candle's Tensor wrapper for the CUDA path —
//!    load PTX via cudarc directly, manage device pointers manually. Loses
//!    candle's safe-Tensor handle but enables full custom kernel control.
//! 3. **Approach C**: wait for candle 0.12+ to add the missing CUDA kernels
//!    upstream (unlikely in near term; candle's own issue tracker has
//!    `narrow CUDA kernel` open since 0.9).
//!
//! Both Approaches A and B are multi-hour work (1-3h each) with high
//! debugging risk because they require understanding candle-core's internal
//! storage abstractions deeply. The current PoC correctly establishes
//! ALL the surrounding scaffolding (FFI signatures, dispatch shim, .cu
//! source code with real math, PTX compiled and embedded) — only the
//! final cudarc Device handle extraction is missing.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod ffi;

use anyhow::{Context, Result};
use candle_core::{DType, Shape, Tensor};

/// Narrow (slice along a dim) — public API. CPU passthrough to candle;
/// CUDA returns clean error explaining the missing T6.5 step.
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
    // T6.5 deferred: extract cudarc Device from candle's CudaDevice wrapper.
    // The wrapper doesn't expose `.inner()`, so launching our PTX kernel
    // requires either vendoring candle-core or bypassing Tensor entirely
    // (raw cudarc + manual device management).
    Err(anyhow::anyhow!(
        "narrow_cuda: cudarc Device handle extraction from candle-core 0.11 \
         CudaDevice wrapper is not exposed (T6.5 deferred). See \
         crates/airllm-kernels/src/lib.rs comment for path forward."
    ))
}

/// Stack (concat along new dim).
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
        "stack_cuda: T6.5 deferred (see narrow_cuda comment)"
    ))
}

/// Reshape.
pub fn reshape(t: &Tensor, shape: Shape) -> Result<Tensor> {
    if t.device().is_cuda() {
        reshape_cuda(t, &shape)
    } else {
        t.reshape(shape).map_err(anyhow::Error::from)
    }
}

fn reshape_cuda(_t: &Tensor, _shape: &Shape) -> Result<Tensor> {
    Err(anyhow::anyhow!(
        "reshape_cuda: T6.5 deferred (see narrow_cuda comment)"
    ))
}