#pragma once
#include "../primitives/packed_load.cuh"
#include "../primitives/writeback.cuh"

namespace q45 {
// Load a lane's K-stripe activations for R rows: x[r][s] holds column
// k_base + lane + 32 * s, zero for rows outside the matrix.
// SCOPE: thread-local reads; OWNS `x` until the next K step.
template<int R>
static __device__ __forceinline__ void load_activation_stripes(
        float (&x)[R][4], const unsigned short* input, unsigned int rows, unsigned int in_features,
        unsigned int first_row, unsigned int k_base, unsigned int lane) {
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int s = 0; s < 4; s++) {
            unsigned long long offset = (unsigned long long)(first_row + r) * in_features + k_base + lane + s * 32;
            x[r][s] = first_row + r < rows ? bf16_to_float(input[offset]) : 0.0f;
        }
    }
}

// Cooperative decode for R <= 4 token rows. Each of 4 warps owns 2 output
// columns; a CTA covers 8 columns x R rows. Per K128 step each column's code,
// fifth-bit and scale words arrive in one warp-wide round of 32-bit loads, and
// both columns' words are loaded before either is consumed. A lane holds four
// K-stripe activations per row and reuses each across both columns; per-stripe
// FP32 sums keep the original single-row order, so every R is bitwise equal.
template<int BITS, int R>
static __device__ __forceinline__ void cooperative_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features) {
    constexpr int kWarpCols = 2, kStripes = 4;
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    // Quotient-and-remainder form: out_features + 7 could wrap for valid ABI sizes.
    const unsigned int output_tiles = out_features / 8u + (out_features % 8u != 0u);
    const unsigned int first_out = (blockIdx.x % output_tiles) * 8u + warp * kWarpCols;
    const unsigned int first_row = (blockIdx.x / output_tiles) * R;
    const Layout<BITS> w(weights, in_features, out_features);
    float sums[R][kWarpCols][kStripes] = {};
    for (unsigned int k = 0; k < in_features; k += 2 * kGroup) {
        unsigned int word[kWarpCols];
        #pragma unroll
        for (int n = 0; n < kWarpCols; n++)
            word[n] = first_out + n < out_features
                    ? load_k128_word<BITS>(w, w.group(first_out + n, k / kGroup), lane) : 0u;
        float x[R][kStripes];
        load_activation_stripes<R>(x, input, rows, in_features, first_row, k, lane);
        #pragma unroll
        for (int n = 0; n < kWarpCols; n++) {
            if (first_out + n >= out_features) continue;
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                float scale = k128_scale<BITS>(word[n], half);
                #pragma unroll
                for (int p = 0; p < 2; p++) {
                    int code = k128_code<BITS>(word[n], half, p, lane);
                    #pragma unroll
                    for (int r = 0; r < R; r++)
                        sums[r][n][half * 2 + p] += scaled_product(x[r][half * 2 + p], code, scale);
                }
            }
        }
    }
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int n = 0; n < kWarpCols; n++) {
            float sum = reduce_stripes(sums[r][n]);
            if (lane == 0 && first_row + r < rows && first_out + n < out_features)
                write_bf16(output, first_row + r, first_out + n, out_features, sum);
        }
    }
}

}  // namespace q45
