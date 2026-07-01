// CUDA kernel: narrow (slice along a dim, then strided copy) for BF16.
// Phase 2b GPU unblock: T6.5 follow-up — the F32 kernel is incompatible
// with the Qwen3-8B forward pass (which has mixed dtypes and expects
// narrow output to match input dtype). This is the BF16 specialization.
//
// IMPORTANT: must use `__nv_bfloat16` (BF16 — Google Brain float, 1+8+7
// bit layout) from `cuda_bf16.h`, NOT `__half` (FP16 — IEEE 754 binary16,
// 1+5+10) from `cuda_fp16.h`. The two types share a 16-bit memory layout
// but interpret bits differently. Using `__half` against BF16 data reads
// the right number of elements but the wrong VALUES (BF16 bits decoded
// as FP16 give garbage like 0.0 and out-of-range numbers). This kernel
// was previously mis-typed as `__half*`; the D0026 test caught it.

#include <stdint.h>
#include <stddef.h>
#include <cuda_bf16.h>

extern "C" __global__ void narrow_bf16(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
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

    long in_pos = 0;
    long temp = out_idx;
    for (int d = 0; d < n_dims; d++) {
        // dim_size for coord computation is the OUTPUT size: in_shape[d]
        // except at d == dim, where it is `length`. The previous version
        // used in_shape[d] unconditionally, which over-counted coords
        // and produced out-of-bounds reads (caught by D0026 GPU test
        // with output [4.0, 0.0, 5.0] instead of [4.0, 5.0, 6.0]).
        long dim_size = (d == dim) ? length : in_shape[d];
        long coord = temp % dim_size;
        temp /= dim_size;
        if (d == dim) coord += start;
        in_pos += coord * in_strides[d];
    }
    out[out_idx] = in[in_pos];
}