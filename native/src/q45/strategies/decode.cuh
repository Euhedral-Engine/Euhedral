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
static __device__ __forceinline__ void generic_decode(
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

// Restrict the address-hoisted schedule to one complete R=1 tile family.
static __host__ __device__ constexpr bool use_full_tile_decode(
        unsigned int rows, unsigned int in_features, unsigned int out_features) {
    return rows == 1 && in_features != 0 && in_features % 128u == 0
            && out_features != 0 && out_features % 8u == 0;
}

// Keep each lane's selected plane pointer valid, including inactive lanes.
template<int BITS>
static __device__ __forceinline__ const unsigned char* decode_pointer(
        const Layout<BITS>& w, unsigned long long group, unsigned int lane) {
    if (lane < 16u) return (const unsigned char*)(w.code_words(group) + lane);
    if (BITS == 5 && lane < 20u) return (const unsigned char*)(w.high_words(group) + lane - 16u);
    return (const unsigned char*)(w.scales + group);
}

// Exact signed small-integer conversion through an FP32 mantissa. The XOR
// toggles the sign-code bit, so subtracting 2^23 + sign yields [-sign, sign-1].
// FP32 has unit spacing here; every code is represented exactly.
template<int BITS>
static __device__ __forceinline__ float decode_float_code(
        unsigned int word, int half, int stripe, unsigned int lane) {
    unsigned int code = __shfl_sync(0xffffffffu, word, half * 8 + stripe * 4 + (lane >> 3));
    unsigned int value = (code >> ((lane & 7u) * 4u)) & 15u;
    if (BITS == 5) {
        unsigned int high = __shfl_sync(0xffffffffu, word, 16 + half * 2 + stripe);
        value |= ((high >> lane) & 1u) << 4;
    }
    constexpr unsigned int sign = 1u << (BITS - 1);
    return __uint_as_float(value ^ (0x4b000000u | sign)) - (8388608.0f + (float)sign);
}

template<int BITS>
static __device__ __forceinline__ void full_tile_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int first_out = blockIdx.x * 8u + warp * 2u;
    const Layout<BITS> w(weights, in_features, out_features);
    const unsigned char* p0 = decode_pointer<BITS>(w, w.group(first_out, 0), lane);
    const unsigned char* p1 = decode_pointer<BITS>(w, w.group(first_out + 1, 0), lane);
    const unsigned int step = lane < 16u ? 64u : (BITS == 5 && lane < 20u ? 16u : 4u);
    float sums[2][4] = {};
    for (unsigned int k = 0; k < in_features; k += 128u) {
        unsigned int words[2];
        words[0] = lane <= K128Words<BITS>::kScaleLane ? *(const unsigned int*)p0 : 0u;
        words[1] = lane <= K128Words<BITS>::kScaleLane ? *(const unsigned int*)p1 : 0u;
        p0 += step;
        p1 += step;
        float x[4];
        #pragma unroll
        for (int s = 0; s < 4; s++) x[s] = bf16_to_float(input[k + lane + s * 32u]);
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            const unsigned int scales = __shfl_sync(0xffffffffu, words[n], K128Words<BITS>::kScaleLane);
            #pragma unroll
            for (int half = 0; half < 2; half++) {
                float scale = scale_to_float((unsigned short)(scales >> (half * 16)));
                #pragma unroll
                for (int s = 0; s < 2; s++) {
                    float code = decode_float_code<BITS>(words[n], half, s, lane);
                    sums[n][half * 2 + s] += x[half * 2 + s] * code * scale;
                }
            }
        }
    }
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        float sum = reduce_stripes(sums[n]);
        if (lane == 0) write_bf16(output, 0, first_out + n, out_features, sum);
    }
}

template<int BITS, int R>
static __device__ __forceinline__ void cooperative_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features) {
    if (R == 1 && use_full_tile_decode(rows, in_features, out_features)) {
        full_tile_decode<BITS>(input, weights, output, rows, in_features, out_features);
        return;
    }
    generic_decode<BITS, R>(input, weights, output, rows, in_features, out_features);
}

}  // namespace q45
