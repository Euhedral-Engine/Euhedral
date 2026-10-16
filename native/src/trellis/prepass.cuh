// The input side of a trellis linear: out[g][r] = round(H128(x[r] * suh[g])) for each of the s input scale vectors.
#ifndef EUHEDRAL_TRELLIS_PREPASS_CUH
#define EUHEDRAL_TRELLIS_PREPASS_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "trellis/codebook.cuh"

// x bf16 [rows][k], suh f16 [s][k], out [s][rows][k]: FP16 in the tensor-core GEMM's fragment order, or BF16 in
// column order when bf16_out != 0. Each warp transforms one 128-column block of one row in registers
// (trellis_hadamard128_warp), four per block. Rows are independent, so a row's bits do not depend on how many rows run
// with it.
//   grid (ceil(k / 512), rows, s), block 128.
extern "C" __global__ __launch_bounds__(128) void euhedral_trellis_prepass(
        const unsigned short* __restrict__ x, const unsigned short* __restrict__ suh, unsigned short* __restrict__ out,
        int rows, int k, int s, int bf16_out) {
    const int lane = threadIdx.x & 31, r = blockIdx.y, g = blockIdx.z;
    const int block = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (block * 128 >= k) return;
    const size_t column = (size_t)block * 128 + lane * 4;
    const uint2 xs = *reinterpret_cast<const uint2*>(x + (size_t)r * k + column);
    const uint2 scales = *reinterpret_cast<const uint2*>(suh + (size_t)g * k + column);
    const unsigned int xw[2] = {xs.x, xs.y}, sw[2] = {scales.x, scales.y};
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned int xb = (xw[j >> 1] >> (16 * (j & 1))) & 0xFFFFu, sb = (sw[j >> 1] >> (16 * (j & 1))) & 0xFFFFu;
        v[j] = __uint_as_float(xb << 16) * __half2float(__ushort_as_half((unsigned short)sb));
    }
    trellis_hadamard128_warp(v, lane);
    unsigned short* row = out + ((size_t)g * rows + r) * k;
    if (bf16_out) {
        unsigned short bits[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) bits[j] = __bfloat16_as_ushort(__float2bfloat16_rn(v[j]));
        *reinterpret_cast<uint2*>(row + column) =
                make_uint2(bits[0] | (unsigned int)bits[1] << 16, bits[2] | (unsigned int)bits[3] << 16);
    } else {
        // Fragment order: in each k16 group, the elements 2q, 2q + 1, 2q + 8, 2q + 9 an m16n8k16 A lane reads are
        // contiguous at 4q. This lane's pairs (4 lane, + 1) and (4 lane + 2, + 3) stay pairs.
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const int e = (lane * 4 + pair * 2) & 15;
            const size_t at = (column & ~(size_t)15) + 4 * ((e & 7) >> 1) + 2 * (e >> 3);
            const unsigned int bits = __half_as_ushort(__float2half_rn(v[2 * pair]))
                    | (unsigned int)__half_as_ushort(__float2half_rn(v[2 * pair + 1])) << 16;
            *reinterpret_cast<unsigned int*>(row + at) = bits;
        }
    }
}

#endif
