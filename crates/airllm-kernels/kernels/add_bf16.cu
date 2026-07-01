// CUDA kernel: element-wise add (BF16 + BF16 -> BF16).
// Phase 2b GPU unblock, item 1 of perf roadmap: real BF16 kernels to
// eliminate the F32 dance documented in D0027.
//
// Same pattern as narrow_bf16.cu: uses __nv_bfloat16 (from cuda_bf16.h),
// NOT __half (from cuda_fp16.h). The two types share a 16-bit memory
// layout but interpret bits differently.
//
// Assumes contiguous input and output of the same shape (the dispatch
// shim routes through this kernel for residuals in the model forward,
// which are always same-shape element-wise adds).

#include <stdint.h>
#include <stddef.h>
#include <cuda_bf16.h>

extern "C" __global__ void add_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    long total
) {
    long idx = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    if (idx >= total) return;
    out[idx] = __hadd(a[idx], b[idx]);
}
