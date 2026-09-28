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
    const unsigned int* code = (const unsigned int*)w.group_bytes(g);
    const unsigned int* scale = (const unsigned int*)(w.scales + g);
    const unsigned int* selected = lane < 12 ? code + lane : scale;
    return lane < 13 ? *selected : 0u;
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
static __device__ __forceinline__ void generic_decode(
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


// R=1 full tiles only. Other row counts and all tails retain the generic schedule.
static __host__ __device__ constexpr bool use_full_tile_decode(
        unsigned int rows, unsigned int in_features, unsigned int out_features) {
    return rows == 1 && in_features != 0 && in_features % 128u == 0
            && out_features != 0 && out_features % 8u == 0;
}

// Preserve the generic stripe order; the full-tile branch hoists addresses.
// Decode-local half conversion can canonicalize NaNs: the following multiply
// and accumulation already do so. Keep the shared numeric helper unchanged.
// Constructing 2^23 + (raw ^ 4), then subtracting 2^23 + 4, yields the exact
// signed Q3 code in FP32 without an integer-to-float conversion.
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
    if (R != 1 || !use_full_tile_decode(rows, in_features, out_features)) {
        generic_decode<R>(input, weights, output, rows, in_features, out_features, scale_offset);
        return;
    }
    const unsigned long long g0 = w.group(first_out, 0);
    const unsigned long long g1 = w.group(first_out + 1, 0);
    const unsigned char* p0 = lane < 12 ? w.group_bytes(g0) + lane * 4 : (const unsigned char*)(w.scales + g0);
    const unsigned char* p1 = lane < 12 ? w.group_bytes(g1) + lane * 4 : (const unsigned char*)(w.scales + g1);
    const unsigned int step = lane < 12 ? 48 : 4;
    float sums[R][kWarpCols][kStripes] = {};
    for (unsigned int k = 0; k < in_features; k += 2 * kGroup) {
        unsigned int word[kWarpCols];
        word[0] = lane < 13 ? *(const unsigned int*)p0 : 0;
        word[1] = lane < 13 ? *(const unsigned int*)p1 : 0;
        p0 += step; p1 += step;
        float x[R][kStripes];
        #pragma unroll
        for(int s=0;s<4;++s) x[0][s] = bf16_to_float(input[k+lane+s*32]);
        #pragma unroll
        for (int n = 0; n < kWarpCols; n++) {
            if (true) {
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    unsigned int pairs = k128_pairs(word[n], half, lane);
                    float scale = __half2float(__ushort_as_half((unsigned short)(__shfl_sync(0xffffffffu, word[n], 12) >> (half * 16))));
                    #pragma unroll
                    for (int p = 0; p < 2; p++) {
                        unsigned int raw = (stripe_pair(pairs, lane, p) >> ((lane & 1u) * 3u)) & 7u;
                        float code = __uint_as_float(raw ^ 0x4b000004u) - 8388612.0f;
                        #pragma unroll
                        for (int r = 0; r < R; r++)
                            sums[r][n][half * 2 + p] += (x[r][half * 2 + p] * code * scale);
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
