#pragma once
#include "decode.cuh"
#include "pdl.cuh"

namespace q3 {
// Single-row decode with warp-local wide streaming.
//
// The CTA owns 8 output columns; each of its 4 warps owns 2 and streams both weight rows itself.
// Every lane loads 16 contiguous bytes, so one warp load fetches a 512-byte chunk of a row (about
// ten and a half K128 blocks). One chunk per row is held ahead in registers and two sit in a
// warp-private shared double slot; each K128 block (48 code bytes) is then read with three
// broadcast 16-byte shared loads, and the row's scale words are loaded into shared once. No lane
// has a load role, no word is redistributed by shuffles, and only __syncwarp orders a warp's own
// slots, so warps never wait for each other.
//
// The arithmetic is the full-tile branch of cooperative_decode<1>: lane L owns the K offsets
// L + 32 * stripe of every K128 block, accumulates (x * code) * scale per stripe in K order, and
// reduces through reduce_stripes, so results are bitwise identical to it.
//
// Requirements (checked by host dispatch): one row, in_features a multiple of 512 and at most
// 32768 (whole 16-byte scale vectors, a 256-word scale row), out_features a multiple of 8, and a
// 16-byte aligned weight base (code rows are in_features * 3 / 8 bytes, scale rows in_features / 32).
struct DecodeWideShared {
    uint4 slots[4][2][2][32];
    unsigned int scales[4][2][256];
};

namespace wide {
// This lane's word base + J of a 12-word K128 block; base = (3 * lane) >> 5 is lane-constant.
template<int J>
static __device__ __forceinline__ unsigned int pick(const unsigned int (&words)[12], unsigned int base) {
    constexpr int a = J < 12 ? J : 11, b = J + 1 < 12 ? J + 1 : 11, c = J + 2 < 12 ? J + 2 : 11;
    return base == 0u ? words[a] : (base == 1u ? words[b] : words[c]);
}

// One column, one K128 block. Lane L's code for (half, p) is code L + 32p of group `half`, at bit
// 3L + 96p + 192 half of the block: word base + 3p + 6 half, bit offset (3L) & 31.
static __device__ __forceinline__ void consume(float (&sums)[4], const unsigned int (&words)[12],
        unsigned int scales, const float (&x)[4], unsigned int base, unsigned int shift) {
    #pragma unroll
    for (int half = 0; half < 2; half++) {
        float scale = __half2float(__ushort_as_half((unsigned short)(scales >> (half * 16))));
        #pragma unroll
        for (int p = 0; p < 2; p++) {
            unsigned int lo = half == 0 ? (p == 0 ? pick<0>(words, base) : pick<3>(words, base))
                                        : (p == 0 ? pick<6>(words, base) : pick<9>(words, base));
            unsigned int hi = half == 0 ? (p == 0 ? pick<1>(words, base) : pick<4>(words, base))
                                        : (p == 0 ? pick<7>(words, base) : pick<10>(words, base));
            unsigned int raw = __funnelshift_r(lo, hi, shift) & 7u;
            float code = __uint_as_float(raw ^ 0x4b000004u) - 8388612.0f;
            sums[half * 2 + p] += (x[half * 2 + p] * code * scale);
        }
    }
}
}  // namespace wide

static __device__ __forceinline__ void wide_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int in_features, unsigned int out_features, unsigned long long scale_offset,
        DecodeWideShared& shared) {
    const unsigned int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const Layout w(weights, in_features, scale_offset);
    const unsigned int first_out = blockIdx.x * 8u + warp * 2u;
    const unsigned int steps = in_features / 128u;
    const unsigned int row_vec = (in_features / 64u) * 24u / 16u;
    const unsigned int chunks = (row_vec + 31u) / 32u;
    uint4 (&slots)[2][2][32] = shared.slots[warp];
    unsigned int (&scales)[2][256] = shared.scales[warp];
    const uint4* rows[2] = {reinterpret_cast<const uint4*>(w.group_bytes(w.group(first_out, 0))),
                            reinterpret_cast<const uint4*>(w.group_bytes(w.group(first_out + 1, 0)))};
    // Weights are immutable, so they are fetched before waiting on the producing kernel.
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        const uint4* source = reinterpret_cast<const uint4*>(w.scales + w.group(first_out + n, 0));
        for (unsigned int v = lane; v < steps / 4u; v += 32) reinterpret_cast<uint4*>(scales[n])[v] = source[v];
    }
    uint4 ahead[2];
    #pragma unroll
    for (int n = 0; n < 2; n++) {
        slots[n][0][lane] = lane < row_vec ? rows[n][lane] : make_uint4(0, 0, 0, 0);
        ahead[n] = 32u + lane < row_vec ? rows[n][32u + lane] : make_uint4(0, 0, 0, 0);
    }
    euhedral_pdl_begin();
    const unsigned int base = (3u * lane) >> 5, shift = (3u * lane) & 31u;
    float sums[2][4] = {};
    for (unsigned int chunk = 0; chunk < chunks; chunk++) {
        #pragma unroll
        for (int n = 0; n < 2; n++) {
            if (chunk + 1u < chunks) slots[n][(chunk + 1u) & 1u][lane] = ahead[n];
            const unsigned int v = (chunk + 2u) * 32u + lane;
            if (chunk + 2u < chunks) ahead[n] = v < row_vec ? rows[n][v] : make_uint4(0, 0, 0, 0);
        }
        __syncwarp();
        // Blocks that start in this chunk; the last one may extend into the next slot.
        const unsigned int first_step = (512u * chunk + 47u) / 48u;
        const unsigned int end_step = min(steps, (512u * (chunk + 1u) + 47u) / 48u);
        #pragma unroll 4
        for (unsigned int step = first_step; step < end_step; step++) {
            float x[4];
            #pragma unroll
            for (int stripe = 0; stripe < 4; ++stripe) x[stripe] = bf16_to_float(input[step * 128u + lane + stripe * 32]);
            #pragma unroll
            for (int n = 0; n < 2; n++) {
                unsigned int words[12];
                #pragma unroll
                for (int i = 0; i < 3; i++) {
                    const unsigned int offset = 48u * step + 16u * i;
                    const uint4 v = slots[n][(offset >> 9) & 1u][(offset >> 4) & 31u];
                    words[i * 4] = v.x; words[i * 4 + 1] = v.y; words[i * 4 + 2] = v.z; words[i * 4 + 3] = v.w;
                }
                wide::consume(sums[n], words, scales[n][step], x, base, shift);
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

}  // namespace q3
