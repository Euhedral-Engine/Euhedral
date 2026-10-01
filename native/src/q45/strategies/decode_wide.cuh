#pragma once
#include "decode.cuh"
#include "common/pdl.cuh"

namespace q45 {
// Single-row Q4/Q5 decode with warp-local wide streaming (see q3/strategies/decode_wide.cuh).
//
// The CTA owns 8 output columns; each of its 4 warps owns 2 and streams both code rows itself:
// every lane loads 16 contiguous bytes, so one warp load fetches a 512-byte chunk, exactly eight
// K128 blocks of 64 code bytes. One chunk per row is held ahead in registers and two sit in a
// warp-private shared double slot. The row's fifth-bit words (Q5) and scale pairs are loaded into
// shared once. Each lane then reads the one code word it needs per (group, stripe) and the block's
// fifth-bit words with broadcast shared loads: no lane has a load role and no shuffles move words.
//
// The arithmetic is full_tile_decode: lane L owns the K offsets L + 32 * stripe of every K128
// block, accumulates x * code * scale per stripe in K order, and reduces through reduce_stripes,
// so results are bitwise identical to it.
//
// Requirements (checked by host dispatch): one row, in_features a multiple of 512 and at most
// 8192 (whole 16-byte scale vectors; fifth-bit and scale rows fit their shared slots),
// out_features a multiple of 8, and a 16-byte aligned weight base (every plane is 256-aligned).
template<int BITS>
struct DecodeWideShared {
    uint4 slots[4][2][2][32];
    uint4 high[4][2][BITS == 5 ? 64 : 1];
    unsigned int scales[4][2][64];
};

template<int BITS>
static __device__ __forceinline__ void wide_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int in_features, unsigned int out_features, DecodeWideShared<BITS>& shared) {
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int first_out = blockIdx.x * 8u + warp * 2u;
    const Layout<BITS> w(weights, in_features, out_features);
    const unsigned int steps = in_features / 128u;
    const unsigned int chunks = (steps + 7u) / 8u;
    const unsigned int row_vec = steps * 4u;
    uint4 (&slots)[2][2][32] = shared.slots[warp];
    const uint4* rows[2] = {reinterpret_cast<const uint4*>(w.code_words(w.group(first_out, 0))),
                            reinterpret_cast<const uint4*>(w.code_words(w.group(first_out + 1, 0)))};
    // Weights are immutable, so they are fetched before waiting on the producing kernel.
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        const uint4* scales = reinterpret_cast<const uint4*>(w.scales + w.group(first_out + n, 0));
        for (unsigned int v = lane; v < steps / 4u; v += 32) reinterpret_cast<uint4*>(shared.scales[warp][n])[v] = scales[v];
        if (BITS == 5) {
            const uint4* high = reinterpret_cast<const uint4*>(w.high_words(w.group(first_out + n, 0)));
            for (unsigned int v = lane; v < steps; v += 32) shared.high[warp][n][v] = high[v];
        }
    }
    uint4 ahead[2];
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        slots[n][0][lane] = lane < row_vec ? rows[n][lane] : make_uint4(0, 0, 0, 0);
        ahead[n] = 32u + lane < row_vec ? rows[n][32u + lane] : make_uint4(0, 0, 0, 0);
    }
    euhedral_pdl_begin();
    constexpr unsigned int sign = 1u << (BITS - 1);
    float sums[2][4] = {};
    for (unsigned int chunk = 0; chunk < chunks; chunk++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            if (chunk + 1u < chunks) slots[n][(chunk + 1u) & 1u][lane] = ahead[n];
            const unsigned int v = (chunk + 2u) * 32u + lane;
            if (chunk + 2u < chunks) ahead[n] = v < row_vec ? rows[n][v] : make_uint4(0, 0, 0, 0);
        }
        __syncwarp();
        const unsigned int end_step = min(steps, chunk * 8u + 8u);
        #pragma unroll 4
        for (unsigned int step = chunk * 8u; step < end_step; step++) {
            float x[4];
            #pragma unroll
            for (int stripe = 0; stripe < 4; stripe++) x[stripe] = bf16_to_float(input[step * 128u + lane + stripe * 32u]);
            #pragma unroll
            for (int n = 0; n < 2; n++) {
                const unsigned int* block = reinterpret_cast<const unsigned int*>(&slots[n][chunk & 1u][(step & 7u) * 4u]);
                const uint4 high = BITS == 5 ? shared.high[warp][n][step] : make_uint4(0, 0, 0, 0);
                const unsigned int highs[4] = {high.x, high.y, high.z, high.w};
                const unsigned int scales = shared.scales[warp][n][step];
                #pragma unroll
                for (int half = 0; half < 2; half++) {
                    float scale = scale_to_float((unsigned short)(scales >> (half * 16)));
                    #pragma unroll
                    for (int s = 0; s < 2; s++) {
                        unsigned int value = (block[half * 8 + s * 4 + (lane >> 3)] >> ((lane & 7u) * 4u)) & 15u;
                        if (BITS == 5) value |= ((highs[half * 2 + s] >> lane) & 1u) << 4;
                        float code = __uint_as_float(value ^ (0x4b000000u | sign)) - (8388608.0f + (float)sign);
                        sums[n][half * 2 + s] += x[half * 2 + s] * code * scale;
                    }
                }
            }
        }
        __syncwarp();
    }
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        float sum = reduce_stripes(sums[n]);
        if (lane == 0) write_bf16(output, 0, first_out + n, out_features, sum);
    }
}

}  // namespace q45
