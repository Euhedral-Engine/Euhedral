// The input side of a trellis linear: out[g][r] = round(H128(x[r] * suh[g])) for each of the s input scale vectors.
#ifndef EUHEDRAL_TRELLIS_PREPASS_CUH
#define EUHEDRAL_TRELLIS_PREPASS_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "trellis/codebook.cuh"

// x bf16 [rows][k], suh f16 [s][k], out [s][rows][k]: FP16 in the tensor-core GEMM's fragment order, or BF16 in
// column order when bf16_out != 0. Rows are independent, so a row's bits do not depend on how many rows run with it.
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
    const size_t row = ((size_t)g * rows + r) * k;
    if (bf16_out) {
        out[row + column] = __bfloat16_as_ushort(__float2bfloat16_rn(v[t]));
    } else {
        // Fragment order: in each k16 group, the elements 2q, 2q + 1, 2q + 8, 2q + 9 an m16n8k16 A lane reads are
        // contiguous at 4q.
        const int e = t & 15;
        out[row + (column & ~(size_t)15) + 4 * ((e & 7) >> 1) + 2 * (e >> 3) + (e & 1)] = __half_as_ushort(__float2half_rn(v[t]));
    }
}

#endif
