// The input side of a trellis linear: out[g][r] = round(H128(x[r] * suh[g])) for each of the s input scale vectors.
#ifndef EUHEDRAL_TRELLIS_PREPASS_CUH
#define EUHEDRAL_TRELLIS_PREPASS_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "trellis/codebook.cuh"

// x bf16 [rows][k], suh f16 [s][k], out [s][rows][k] (FP16, or BF16 when bf16_out != 0). Rows are independent, so a
// row's bits do not depend on how many rows run with it.
//   grid (k / 128, rows, s), block 128.
extern "C" __global__ __launch_bounds__(128) void euhedral_trellis_prepass(
        const unsigned short* __restrict__ x, const unsigned short* __restrict__ suh, unsigned short* __restrict__ out,
        int rows, int k, int s, int bf16_out) {
    __shared__ float v[128];
    const int t = threadIdx.x, r = blockIdx.y, g = blockIdx.z;
    const size_t column = (size_t)blockIdx.x * 128 + t;
    const float a = __uint_as_float((unsigned int)x[(size_t)r * k + column] << 16);
    const float b = __half2float(__ushort_as_half(suh[(size_t)g * k + column]));
    v[t] = a * b;
    trellis_hadamard128(v, t);
    const size_t at = ((size_t)g * rows + r) * k + column;
    out[at] = bf16_out ? __bfloat16_as_ushort(__float2bfloat16_rn(v[t])) : __half_as_ushort(__float2half_rn(v[t]));
}

#endif
