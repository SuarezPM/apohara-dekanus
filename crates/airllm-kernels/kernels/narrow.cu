// CUDA kernel: narrow (slice along a dim, then copy).
//
// Phase 2b GPU unblock: candle-core 0.11 missing CUDA kernels for
// Tensor::narrow. This kernel is the custom replacement.
//
// extern "C" launcher: each output element is copied from the
// corresponding input position (offset by `start` along `dim`).
//
// Mathematically trivial — pure memory copy. Optimized for the case
// where the input is contiguous and the output is contiguous after
// slicing (the common case for per-head attention narrow).

#include <stdint.h>
#include <stddef.h>

// Narrow launcher: in is contiguous, out is contiguous (size = product
// of in_shape[0..n_dims] with in_shape[dim] replaced by `length`).
//   in_shape[i]   : shape of input, dim is the slicing dim
//   in_strides[i] : row-major strides of input
//   start         : offset along `dim` where the slice begins
//   length        : size of the slice along `dim`
//
// For each output element at multi-index out_idx, the corresponding
// input element is at multi-index in_idx = out_idx with in_idx[dim] += start.
extern "C" __global__ void narrow_launch_f32(
        const float* __restrict__ in,
        float* __restrict__ out,
        const int64_t* __restrict__ in_shape,
        const int64_t* __restrict__ in_strides,
        int n_dims,
        int dim,
        int64_t start,
        int64_t length,
        int64_t out_total) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= out_total) return;

    // Decode tid into multi-index over out_shape (= in_shape with dim → length)
    int64_t in_off = start * in_strides[dim];
    int64_t rem = tid;
    for (int d = n_dims - 1; d >= 0; --d) {
        if (d == dim) {
            // out_shape[d] = length, so decode that many at this slot
            int64_t slot = d == n_dims - 1 ? 1 : 1; // size of d in out
            // We need out_shape[d] = length. Compute by tracking remaining.
            // Simpler: walk in order using running products of out_shape.
        }
    }

    // Simpler approach: compute in_off directly by walking dims and
    // using out position. out dims = in dims with dim replaced by length.
    int64_t out_off = 0;
    int64_t stride = 1;
    // We need to know out_shape[d]. Build it on the fly: out_shape[d] =
    // (d == dim) ? length : in_shape[d].
    // Walk dims from innermost to outermost.
    for (int d = n_dims - 1; d >= 0; --d) {
        int64_t out_d = (d == dim) ? length : in_shape[d];
        int64_t idx_d = rem % out_d;
        rem /= out_d;
        out_off += idx_d * in_strides[d];
    }
    in_off += out_off;
    out[tid] = in[in_off];
}

// Workgroup 32 (sm_75 warp = 32 threads). One block per 32 elements.
extern "C" void narrow_f32(
        const float* in, float* out,
        const int64_t* in_shape, const int64_t* in_strides,
        int n_dims, int dim, int64_t start, int64_t length,
        int64_t out_total) {
    if (out_total == 0) return;
    int threads = 32;
    int blocks = (int)((out_total + threads - 1) / threads);
    narrow_launch_f32<<<blocks, threads>>>(
        in, out, in_shape, in_strides, n_dims, dim, start, length, out_total);
}
