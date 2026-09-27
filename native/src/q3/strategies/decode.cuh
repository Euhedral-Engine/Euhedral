#pragma once
#include "../primitives/activation.cuh"
#include "../primitives/packed_load.cuh"
#include "../primitives/decode.cuh"
#include "../primitives/accumulation.cuh"
#include "../primitives/writeback.cuh"

namespace q3 {
// Warp-cooperative load of one K128 step: both adjacent G64 groups starting at
// the even group index g and their two FP16 scales, in a single round of loads.
// SCOPE: lane-local. BORROWS: 48 contiguous code bytes and 4 scale bytes.
// PRODUCES (OWNS): lanes 0..11 hold code words 0..11 (group g, then g + 1);
// lane 12 holds scales[g] | scales[g + 1] << 16; other lanes hold 0.
// The scale pair is 4-byte aligned: g is even and the scale table is 256-byte
// aligned within a buffer whose group words are already read as 32-bit values.
static __device__ __forceinline__ unsigned int load_k128_word(
        const Layout& w, unsigned long long g, unsigned int lane) {
    if (lane < 12) return ((const unsigned int*)w.group_bytes(g))[lane];
    return lane == 12 ? *(const unsigned int*)(w.scales + g) : 0u;
}

// Hand one group of a K128 word set to load ownership (lane owns K = 2*lane,
// 2*lane + 1 of group `half`) and broadcast that group's scale.
// SCOPE: warp-collective. Produces the same values as load_packed_pair and
// load_group_scale for group g + half.
static __device__ __forceinline__ unsigned int k128_pairs(unsigned int word, int half, unsigned int lane) {
    unsigned int bit = lane * 6u;
    unsigned int lo = __shfl_sync(0xffffffffu, word, half * 6 + (bit >> 5));
    unsigned int hi = __shfl_sync(0xffffffffu, word, half * 6 + (bit >> 5) + 1);
    return (unsigned int)((((unsigned long long)hi << 32) | lo) >> (bit & 31u)) & 63u;
}
static __device__ __forceinline__ float k128_scale(unsigned int word, int half) {
    return fp16_to_float((unsigned short)(__shfl_sync(0xffffffffu, word, 12) >> (16 * half)));
}

// Cooperative decode for R <= 4 token rows. Each of 4 warps owns 2 output
// columns; a CTA covers 8 columns x R rows. Per 128-wide K step a lane holds four
// K-stripe activations per row and accumulates four FP32 stripes per (row, col),
// reusing each loaded activation across both columns. Both columns' K128 weight
// words are loaded before any is consumed.
template<int R>
static __device__ __forceinline__ void cooperative_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned long long scale_offset) {
    constexpr int kWarpCols = 2, kStripes = 4;
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const unsigned int output_tiles = (out_features + 7u) / 8u;
    const unsigned int first_out = (blockIdx.x % output_tiles) * 8 + warp * kWarpCols;
    const unsigned int first_row = (blockIdx.x / output_tiles) * R;
    const Layout w(weights, in_features, scale_offset);
    float sums[R][kWarpCols][kStripes] = {};
    for (unsigned int k = 0; k < in_features; k += 2 * kGroup) {
        unsigned int word[kWarpCols];
        #pragma unroll
        for (int n = 0; n < kWarpCols; n++)
            word[n] = first_out + n < out_features
                    ? load_k128_word(w, w.group(first_out + n, k / kGroup), lane) : 0u;
        float x[R][kStripes];
        load_activation_stripes<R, kStripes>(x, input, rows, in_features, first_row, k, lane);
        #pragma unroll
        for (int n = 0; n < kWarpCols; n++) {
            if (first_out + n < out_features) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    unsigned int pairs = k128_pairs(word[n], half, lane);
                    float scale = k128_scale(word[n], half);
                    #pragma unroll
                    for (int p = 0; p < 2; p++) {
                        int code = decode_code(stripe_pair(pairs, lane, p), lane & 1);
                        #pragma unroll
                        for (int r = 0; r < R; r++)
                            sums[r][n][half * 2 + p] += scaled_product(x[r][half * 2 + p], code, scale);
                    }
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


}  // namespace q3
