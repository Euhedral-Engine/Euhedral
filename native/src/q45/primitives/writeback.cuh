#pragma once
#include "../numeric.cuh"

namespace q45 {
// Combine a lane's four K-stripe FP32 sums, then reduce across the warp, in the
// original decode order ((s0 + s2) + (s1 + s3)) and shuffle tree.
// SCOPE: warp-collective. PRODUCES: the full sum in lane 0.
static __device__ __forceinline__ float reduce_stripes(const float (&s)[4]) {
    float sum = (s[0] + s[2]) + (s[1] + s[3]);
    #pragma unroll
    for (int distance = 16; distance; distance >>= 1) sum += __shfl_down_sync(0xffffffffu, sum, distance);
    return sum;
}

// Write one FP32 result as BF16 (round-to-nearest-even). Bounds are the caller's.
static __device__ __forceinline__ void write_bf16(
        unsigned short* output, unsigned int row, unsigned int col, unsigned int out_features, float value) {
    output[(unsigned long long)row * out_features + col] = float_to_bf16(value);
}

// Write a ROWS x COLS FP32 shared tile to global BF16, skipping out-of-matrix
// elements. SCOPE: CTA-cooperative; the strategy barriers before calling.
template<int ROWS, int COLS, int THREADS>
static __device__ __forceinline__ void write_output_tile(
        unsigned short* output, const float* result, unsigned int rows, unsigned int out_features,
        unsigned int row_start, unsigned int out_start, unsigned int thread) {
    for (unsigned int i = thread; i < ROWS * COLS; i += THREADS) {
        unsigned int r = row_start + i / COLS, n = out_start + i % COLS;
        if (r < rows && n < out_features) write_bf16(output, r, n, out_features, result[i]);
    }
}

}  // namespace q45
