// The output side of a trellis linear: out = bf16(svh * H128(sum of the K-split partials)), and the same for rotated
// BF16 products of the prefill route.
#ifndef EUHEDRAL_TRELLIS_EPILOGUE_CUH
#define EUHEDRAL_TRELLIS_EPILOGUE_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "trellis/codebook.cuh"

// partials f32 [S][rows][n]; the splits are summed in order 0..S-1.
//   grid (n / 128, rows), block 128.
extern "C" __global__ __launch_bounds__(128) void euhedral_trellis_epilogue(
        const float* __restrict__ partials, int splits, const unsigned short* __restrict__ svh,
        unsigned short* __restrict__ out, int rows, int n) {
    __shared__ float v[128];
    const int t = threadIdx.x, r = blockIdx.y;
    const size_t column = (size_t)blockIdx.x * 128 + t;
    float sum = partials[(size_t)r * n + column];
    for (int i = 1; i < splits; ++i) sum += partials[((size_t)i * rows + r) * n + column];
    v[t] = sum;
    trellis_hadamard128(v, t);
    out[(size_t)r * n + column] =
            __bfloat16_as_ushort(__float2bfloat16_rn(__half2float(__ushort_as_half(svh[column])) * v[t]));
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
