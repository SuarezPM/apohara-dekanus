// CUDA kernel: narrow (slice along a dim, then strided copy).
//
// Phase 2b GPU unblock: candle-core 0.11 missing CUDA kernels for
// Tensor::narrow. This kernel is the custom replacement.
//
// Generic narrow: handles arbitrary dim, start, length, and input strides
// (handles non-contiguous input). For contiguous input (common case for
// per-head attention narrow), this collapses to a simple memcpy loop.

#include <stdint.h>
#include <stddef.h>

extern "C" __global__ void narrow_f32(
    const float* __restrict__ in,
    float* __restrict__ out,
    const long* __restrict__ in_shape,
    const long* __restrict__ in_strides,
    int n_dims,
    int dim,
    long start,
    long length,
    long out_total
) {
    long out_idx = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    if (out_idx >= out_total) return;

    // Compute multi-dim coords from out_idx using OUT shape (same as in_shape
    // except in_shape[dim] is replaced by `length`).
    long in_pos = 0;
    long temp = out_idx;
    for (int d = 0; d < n_dims; d++) {
        long dim_size = in_shape[d];
        long coord = temp % dim_size;
        temp /= dim_size;
        if (d == dim) coord += start;
        in_pos += coord * in_strides[d];
    }
    out[out_idx] = in[in_pos];
}