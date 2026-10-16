// The output side of a trellis linear: out = bf16(svh * H128(sum of the K-split partials)), and the same for rotated
// BF16 products of the prefill route.
#ifndef EUHEDRAL_TRELLIS_EPILOGUE_CUH
#define EUHEDRAL_TRELLIS_EPILOGUE_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "trellis/codebook.cuh"

// partials f32 [S][rows][n]; the splits are summed in order 0..S-1. Each warp finishes one 128-column block of one
// row in registers (trellis_hadamard128_warp), four per block.
//   grid (ceil(n / 512), rows), block 128.
extern "C" __global__ __launch_bounds__(128) void euhedral_trellis_epilogue(
        const float* __restrict__ partials, int splits, const unsigned short* __restrict__ svh,
        unsigned short* __restrict__ out, int rows, int n) {
    const int lane = threadIdx.x & 31, r = blockIdx.y;
    const int block = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (block * 128 >= n) return;
    const size_t column = (size_t)block * 128 + lane * 4;
    float4 sum = *reinterpret_cast<const float4*>(partials + (size_t)r * n + column);
#pragma unroll 4
    for (int i = 1; i < splits; ++i) {
        const float4 part = *reinterpret_cast<const float4*>(partials + ((size_t)i * rows + r) * n + column);
        sum.x += part.x;
        sum.y += part.y;
        sum.z += part.z;
        sum.w += part.w;
    }
    float v[4] = {sum.x, sum.y, sum.z, sum.w};
    trellis_hadamard128_warp(v, lane);
    const uint2 scales = *reinterpret_cast<const uint2*>(svh + column);
    const unsigned int sw[2] = {scales.x, scales.y};
    unsigned short bits[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float scale = __half2float(__ushort_as_half((unsigned short)(sw[j >> 1] >> (16 * (j & 1)))));
        bits[j] = __bfloat16_as_ushort(__float2bfloat16_rn(scale * v[j]));
    }
    *reinterpret_cast<uint2*>(out + (size_t)r * n + column) =
            make_uint2(bits[0] | (unsigned int)bits[1] << 16, bits[2] | (unsigned int)bits[3] << 16);
}

// rotated bf16: the product of segment g starts at element rows * start_g with row stride n_g (start_g, n_g from the
// exclusive row ends).
//   grid (n / 128, rows), block 128.
extern "C" __global__ __launch_bounds__(128) void euhedral_trellis_epilogue_bf16(
        const unsigned short* __restrict__ rotated, const int* __restrict__ ends, int s,
        const unsigned short* __restrict__ svh, unsigned short* __restrict__ out, int rows, int n) {
    __shared__ float v[128];
    const int t = threadIdx.x, r = blockIdx.y;
    const int block_start = blockIdx.x * 128;
    int g = 0;
    while (g + 1 < s && block_start >= ends[g]) ++g;
    const int start = g == 0 ? 0 : ends[g - 1];
    const int width = ends[g] - start;
    const size_t column = (size_t)block_start + t;
    v[t] = __uint_as_float((unsigned int)rotated[(size_t)rows * start + (size_t)r * width + (column - start)] << 16);
    trellis_hadamard128(v, t);
    out[(size_t)r * n + column] =
            __bfloat16_as_ushort(__float2bfloat16_rn(__half2float(__ushort_as_half(svh[column])) * v[t]));
}

#endif
