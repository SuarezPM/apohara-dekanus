// CUDA kernel: element-wise multiply (BF16 * BF16 -> BF16).
// Phase 1 of v2.2 plan: real BF16 kernels to eliminate the F32 dance
// (D0027 baseline). This is the 2nd of 6 missing kernels after
// add_bf16 (D0029). Same D0029 pattern: __nv_bfloat16 + start_offset
// slicing in the dispatch shim.

#include <stdint.h>
#include <stddef.h>
#include <cuda_bf16.h>

extern "C" __global__ void mul_bf16(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    long total
) {
    long idx = (long)blockIdx.x * (long)blockDim.x + (long)threadIdx.x;
    if (idx >= total) return;
    out[idx] = __hmul(a[idx], b[idx]);
}
