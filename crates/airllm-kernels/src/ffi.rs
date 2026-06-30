//! Raw FFI declarations matching the `extern "C"` launchers in
//! `kernels/narrow.cu`, `kernels/stack.cu`, `kernels/reshape.cu`.
//!
//! These are hand-written (cudaforge-style generated bindings are
//! a future option). Each launcher is `extern "C"` and has C ABI
//! compatible with `cudarc::driver::LaunchKernel::launch` argument
//! conventions: raw device pointers + scalar arguments; complex types
//! (shape, strides) are passed as device-pointer-to-array.

#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(dead_code)]

/// Narrow (slice) launcher.
///
/// `in_strides` and `in_shape` are host-side arrays of length `n_dims`
/// (row-major contiguous strides). The kernel does not need GPU
/// device-side memory allocation; the host caller passes already-
/// allocated device pointers.
#[cfg(feature = "cuda")]
extern "C" {
    pub fn narrow_f32(
        in_ptr: *const f32,
        out_ptr: *mut f32,
        in_shape_ptr: *const i64,
        in_strides_ptr: *const i64,
        n_dims: i32,
        dim: i32,
        start: i64,
        length: i64,
        out_total: i64,
    );
}

/// Stack (concat along new dim) launcher.
///
/// `in_ptrs` is host-side array of N device pointers, one per input
/// tensor. `out_strides` is the precomputed row-major strides of the
/// output tensor (which has size 1 inserted at `dim`).
#[cfg(feature = "cuda")]
extern "C" {
    pub fn stack_f32(
        in_ptrs: *const *const f32,
        out_ptr: *mut f32,
        in_shape_ptr: *const i64,
        out_strides_ptr: *const i64,
        n_dims: i32,
        dim: i32,
        n_inputs: i32,
        out_total: i64,
    );
}

/// Reshape (strided element copy) launcher.
#[cfg(feature = "cuda")]
extern "C" {
    pub fn reshape_f32(
        in_ptr: *const f32,
        out_ptr: *mut f32,
        in_strides_ptr: *const i64,
        out_strides_ptr: *const i64,
        n_dims: i32,
        out_total: i64,
    );
}
