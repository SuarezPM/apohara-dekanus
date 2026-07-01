//! Custom CUDA kernel dispatch for candle-core 0.11 narrow/stack/reshape.
//!
//! ## T6.5 final implementation (m0660 — canonical pattern from candle-nn/src/rotary_emb.rs)
//!
//! Uses candle's CudaDevice public methods directly. The `get_or_load_custom_func`,
//! `alloc`, `clone_htod`, `clone_dtoh`, and `builder()` methods on CudaDevice are
//! all public (verified at m0594). The `candle_core::cuda_backend::cudarc::driver`
//! re-exports `PushKernelArg`, `DevicePtr`, `DeviceRepr`, `CudaSlice` so the
//! kernel launch and HTOD/DTOH copies use the canonical cudarc traits.

#![allow(unsafe_code)] // dev.alloc, builder.launch require unsafe

pub mod ffi;

use anyhow::Result;
use candle_core::{
    cuda_backend::{CudaStorage, CudaStorageSlice},
    op::BackpropOp, DType, Shape, Storage, Tensor,
};
use candle_core::cuda_backend::cudarc::driver::{CudaSlice, LaunchConfig, PushKernelArg};

/// Public API: narrow a tensor (CPU → candle built-in; CUDA → our custom kernel).
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

fn narrow_cuda(t: &Tensor, dim: usize, start: usize, length: usize) -> Result<Tensor> {
    // Dispatch to the right kernel based on input dtype.
    if t.dtype() == DType::BF16 {
        return narrow_bf16_cuda(t, dim, start, length);
    }
    if t.dtype() != DType::F32 {
        return Err(anyhow::anyhow!(
            "narrow_cuda: only F32/BF16 supported (got {:?})",
            t.dtype()
        ));
    }

    // 1. Access storage + layout from the candle Tensor
    let (storage_guard, layout) = t.storage_and_layout();
    let cuda_storage: &CudaStorage = match &*storage_guard {
        Storage::Cuda(s) => s,
        _ => return Err(anyhow::anyhow!("narrow_cuda: tensor is not on CUDA")),
    };
    let dev = cuda_storage.device.clone();
    let slice: &CudaSlice<f32> = cuda_storage
        .as_cuda_slice::<f32>()
        .map_err(|e| anyhow::anyhow!("as_cuda_slice: {}", e))?;

    // 2. Compute output shape and strides
    let dims: Vec<usize> = layout.shape().dims().to_vec();
    let strides: Vec<usize> = layout.stride().to_vec();
    let n_dims = dims.len();
    if dim >= n_dims {
        return Err(anyhow::anyhow!("narrow: dim {} out of range (rank {})", dim, n_dims));
    }
    let out_dims: Vec<usize> = dims
        .iter()
        .enumerate()
        .map(|(i, &d)| if i == dim { length } else { d })
        .collect();
    let out_total: usize = out_dims.iter().product();
    if out_total == 0 {
        return Err(anyhow::anyhow!("narrow: output is empty (out_dims={:?})", out_dims));
    }

    // 3. Upload shape + strides arrays to device (small one-time cost)
    let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
    let strides_i64: Vec<i64> = strides.iter().map(|&s| s as i64).collect();
    let shape_dev: CudaSlice<i64> = dev
        .clone_htod(dims_i64.as_slice())
        .map_err(|e| anyhow::anyhow!("htod shape: {}", e))?;
    let strides_dev: CudaSlice<i64> = dev
        .clone_htod(strides_i64.as_slice())
        .map_err(|e| anyhow::anyhow!("htod strides: {}", e))?;

    // 4. Allocate output buffer on GPU
    let out_dev: CudaSlice<f32> = unsafe { dev.alloc::<f32>(out_total) }
        .map_err(|e| anyhow::anyhow!("alloc out: {}", e))?;

    // 5. Load custom PTX kernel via candle's CudaDevice (the canonical path)
    let ptx_src = include_str!("../kernels/narrow.cu.ptx");
    let func = dev
        .get_or_load_custom_func("narrow_f32", "narrow_kernel", ptx_src)
        .map_err(|e| anyhow::anyhow!("get_or_load_custom_func: {}", e))?;

    // 6. Launch kernel (1D grid sized to out_total)
    let n_threads = 256u32;
    let n_blocks = (out_total as u32).div_ceil(n_threads).max(1);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (n_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(slice);
    builder.arg(&out_dev);
    builder.arg(&shape_dev);
    builder.arg(&strides_dev);
    let n_dims_i32 = n_dims as i32;
    let dim_i32 = dim as i32;
    let start_i64 = start as i64;
    let length_i64 = length as i64;
    let out_total_i64 = out_total as i64;
    builder.arg(&n_dims_i32);
    builder.arg(&dim_i32);
    builder.arg(&start_i64);
    builder.arg(&length_i64);
    builder.arg(&out_total_i64);
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| anyhow::anyhow!("launch: {}", e))?;
    }

    // 7. Wrap output into a fresh candle Tensor via from_storage
    let out_shape = Shape::from(out_dims);
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage {
            slice: CudaStorageSlice::F32(out_dev),
            device: dev,
        }),
        out_shape,
        BackpropOp::none(),
        false,
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

fn stack_cuda(tensors: &[&Tensor], dim: usize) -> Result<Tensor> {
    // Candle's Tensor::stack DOES have a CUDA implementation (unlike narrow
    // which was the missing kernel). For T6.5 first cut, delegate to it.
    // We can replace this with a custom PTX kernel later if profiling shows
    // candle's stack is slow.
    let owned: Vec<Tensor> = tensors.iter().map(|&t| t.clone()).collect();
    let refs: Vec<&Tensor> = owned.iter().collect();
    Tensor::stack(&refs, dim).map_err(anyhow::Error::from)
}

/// Public API: reshape.
pub fn reshape(t: &Tensor, shape: Shape) -> Result<Tensor> {
    if t.device().is_cuda() {
        reshape_cuda(t, &shape)
    } else {
        t.reshape(shape).map_err(anyhow::Error::from)
    }
}

fn reshape_cuda(t: &Tensor, shape: &Shape) -> Result<Tensor> {
    // Candle's Tensor::reshape DOES have a CUDA implementation. For T6.5 first
    // cut, delegate to it. A custom PTX kernel can replace this later if
    // profiling shows candle's reshape is slow.
    t.reshape(shape.clone()).map_err(anyhow::Error::from)
}

fn narrow_bf16_cuda(t: &Tensor, dim: usize, start: usize, length: usize) -> Result<Tensor> {
    // 1. Access storage + layout from the candle Tensor
    let (storage_guard, layout) = t.storage_and_layout();
    let cuda_storage: &CudaStorage = match &*storage_guard {
        Storage::Cuda(s) => s,
        _ => return Err(anyhow::anyhow!("narrow_bf16_cuda: tensor is not on CUDA")),
    };
    let dev = cuda_storage.device.clone();
    let slice: &cudarc::driver::CudaSlice<half::bf16> = cuda_storage
        .as_cuda_slice::<half::bf16>()
        .map_err(|e| anyhow::anyhow!("as_cuda_slice bf16: {}", e))?;

    // 2. Compute output shape and strides
    let dims: Vec<usize> = layout.shape().dims().to_vec();
    let strides: Vec<usize> = layout.stride().to_vec();
    let n_dims = dims.len();
    if dim >= n_dims {
        return Err(anyhow::anyhow!("narrow_bf16: dim {} out of range (rank {})", dim, n_dims));
    }
    let out_dims: Vec<usize> = dims
        .iter()
        .enumerate()
        .map(|(i, &d)| if i == dim { length } else { d })
        .collect();
    let out_total: usize = out_dims.iter().product();
    if out_total == 0 {
        return Err(anyhow::anyhow!("narrow_bf16: output is empty (out_dims={:?})", out_dims));
    }

    // 3. Upload shape + strides arrays to device
    let dims_i64: Vec<i64> = dims.iter().map(|&d| d as i64).collect();
    let strides_i64: Vec<i64> = strides.iter().map(|&s| s as i64).collect();
    let shape_dev: cudarc::driver::CudaSlice<i64> = dev
        .clone_htod(dims_i64.as_slice())
        .map_err(|e| anyhow::anyhow!("htod shape: {}", e))?;
    let strides_dev: cudarc::driver::CudaSlice<i64> = dev
        .clone_htod(strides_i64.as_slice())
        .map_err(|e| anyhow::anyhow!("htod strides: {}", e))?;

    // 4. Allocate output buffer on GPU
    let out_dev: cudarc::driver::CudaSlice<half::bf16> = unsafe { dev.alloc::<half::bf16>(out_total) }
        .map_err(|e| anyhow::anyhow!("alloc out bf16: {}", e))?;

    // 5. Load custom PTX kernel
    let ptx_src = include_str!("../kernels/narrow_bf16.cu.ptx");
    let func = dev
        .get_or_load_custom_func("narrow_bf16", "narrow_bf16_kernel", ptx_src)
        .map_err(|e| anyhow::anyhow!("get_or_load_custom_func bf16: {}", e))?;

    // 6. Launch kernel
    let n_threads = 256u32;
    let n_blocks = (out_total as u32).div_ceil(n_threads).max(1);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (n_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    let mut builder = func.builder();
    builder.arg(slice);
    builder.arg(&out_dev);
    builder.arg(&shape_dev);
    builder.arg(&strides_dev);
    let n_dims_i32 = n_dims as i32;
    let dim_i32 = dim as i32;
    let start_i64 = start as i64;
    let length_i64 = length as i64;
    let out_total_i64 = out_total as i64;
    builder.arg(&n_dims_i32);
    builder.arg(&dim_i32);
    builder.arg(&start_i64);
    builder.arg(&length_i64);
    builder.arg(&out_total_i64);
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| anyhow::anyhow!("launch bf16: {}", e))?;
    }

    // 7. Wrap output into a fresh candle Tensor
    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage {
            slice: CudaStorageSlice::BF16(out_dev),
            device: dev,
        }),
        Shape::from(out_dims),
        BackpropOp::none(),
        false,
    ))
}

/// Public API: element-wise add (CPU → candle built-in; CUDA → our custom kernel).
/// Used for residuals in the model forward (e.g., `hidden_states + attn_out`).
/// Currently only the BF16 path is implemented (item 1 of perf roadmap).
/// F32 path delegates to candle's built-in.
pub fn add(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    if a.device().is_cuda() {
        match a.dtype() {
            DType::BF16 => add_bf16_cuda(a, b),
            DType::F32 => Ok((a + b)?),
            other => Err(anyhow::anyhow!("add: only F32/BF16 supported, got {:?}", other)),
        }
    } else {
        Ok((a + b)?)
    }
}

fn add_bf16_cuda(a: &Tensor, b: &Tensor) -> Result<Tensor> {
    // Contiguous-ify first: model tensors may be non-contiguous (e.g., the
    // output of a broadcast or a slice). The kernel assumes contiguous
    // same-shape inputs. If they are not contiguous, .contiguous() makes
    // a contiguous copy; if they already are, it's a no-op view.
    let a_c = a.contiguous().map_err(|e| anyhow::anyhow!("a.contiguous: {}", e))?;
    let b_c = b.contiguous().map_err(|e| anyhow::anyhow!("b.contiguous: {}", e))?;
    let (a_storage, a_layout) = a_c.storage_and_layout();
    let (b_storage, b_layout) = b_c.storage_and_layout();
    let a_cuda: &CudaStorage = match &*a_storage {
        Storage::Cuda(s) => s,
        _ => return Err(anyhow::anyhow!("add_bf16_cuda: a is not on CUDA")),
    };
    let b_cuda: &CudaStorage = match &*b_storage {
        Storage::Cuda(s) => s,
        _ => return Err(anyhow::anyhow!("add_bf16_cuda: b is not on CUDA")),
    };
    if a_layout.shape() != b_layout.shape() {
        return Err(anyhow::anyhow!(
            "add_bf16_cuda: shape mismatch {:?} vs {:?}",
            a_layout.shape(),
            b_layout.shape()
        ));
    }
    let dev = a_cuda.device.clone();
    let a_slice: &cudarc::driver::CudaSlice<half::bf16> = a_cuda
        .as_cuda_slice::<half::bf16>()
        .map_err(|e| anyhow::anyhow!("as_cuda_slice a: {}", e))?;
    let b_slice: &cudarc::driver::CudaSlice<half::bf16> = b_cuda
        .as_cuda_slice::<half::bf16>()
        .map_err(|e| anyhow::anyhow!("as_cuda_slice b: {}", e))?;

    let total: usize = a_layout.shape().elem_count();
    if total == 0 {
        return Err(anyhow::anyhow!("add_bf16_cuda: empty tensors"));
    }

    let out_dev: cudarc::driver::CudaSlice<half::bf16> = unsafe { dev.alloc::<half::bf16>(total) }
        .map_err(|e| anyhow::anyhow!("alloc out: {}", e))?;

    let ptx_src = include_str!("../kernels/add_bf16.cu.ptx");
    let func = dev
        .get_or_load_custom_func("add_bf16", "add_bf16_kernel", ptx_src)
        .map_err(|e| anyhow::anyhow!("get_or_load_custom_func add_bf16: {}", e))?;

    let n_threads = 256u32;
    let n_blocks = (total as u32).div_ceil(n_threads).max(1);
    let cfg = LaunchConfig {
        grid_dim: (n_blocks, 1, 1),
        block_dim: (n_threads, 1, 1),
        shared_mem_bytes: 0,
    };

    // Pass slices directly (cudarc uses DeviceRepr to get the device pointer
    // for the kernel argument). Same pattern as narrow_bf16_cuda.
    let mut builder = func.builder();
    builder.arg(a_slice);
    builder.arg(b_slice);
    builder.arg(&out_dev);
    let total_i64 = total as i64;
    builder.arg(&total_i64);
    unsafe {
        builder
            .launch(cfg)
            .map_err(|e| anyhow::anyhow!("launch add_bf16: {}", e))?;
    }

    Ok(Tensor::from_storage(
        Storage::Cuda(CudaStorage {
            slice: CudaStorageSlice::BF16(out_dev),
            device: dev,
        }),
        a_layout.shape().clone(),
        BackpropOp::none(),
        false,
    ))
}