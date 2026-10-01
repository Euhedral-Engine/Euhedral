#pragma once
#include "decode.cuh"
#include "pdl.cuh"

namespace q3 {
// Single-row decode with contiguous lane ownership.
//
// Three 32-bit words of a G64 group hold exactly 32 whole 3-bit codes (half a group), so a lane
// owns one half group: 32 contiguous K values whose bit positions are compile-time constants.
// A warp's 32 lanes cover 384 contiguous bytes (1024 K) of a row per slice, read with three
// coalesced word loads per lane, and each warp owns kRows rows, reusing its 32 activations for
// every row. Each lane forms an FP32 dot product of its 32 codes, scales it once by the group
// scale, and accumulates across slices; a warp shuffle reduces each row.
//
// Numerical contract: FP32 accumulation in a different order from cooperative_decode<1>, the exact
// oracle (euhedral_q3_decode_1 and euhedral_q3_decode_wide), so results agree within FP32 rounding
// before the BF16 output rounding, not bitwise. Products x * code are exact in FP32. The contract
// covers finite FP16 scales, which is what the artifact stores; a non-finite scale multiplies the
// half group's dot product as IEEE arithmetic, which the exact kernels do not reproduce.
//
// Requirements (checked by host dispatch): one row, in_features a multiple of 1024, out_features a
// multiple of 4 * kRows, a 16-byte aligned input and a 4-byte aligned weight base.
namespace contiguous {
static constexpr int kRows = 4;

template<int J>
static __device__ __forceinline__ unsigned int field(unsigned int w0, unsigned int w1, unsigned int w2) {
    constexpr int bit = 3 * J;
    if (bit + 3 <= 32) return (w0 >> bit) & 7u;
    if (bit < 32) return __funnelshift_r(w0, w1, bit) & 7u;
    if (bit + 3 <= 64) return (w1 >> (bit - 32)) & 7u;
    if (bit < 64) return __funnelshift_r(w1, w2, bit - 32) & 7u;
    return (w2 >> (bit - 64)) & 7u;
}

// Code J of a half group in FP32: 2^23 + (field ^ 4) - (2^23 + 4) is the signed code, exactly.
template<int J>
struct Dot {
    static __device__ __forceinline__ float run(const float (&x)[32], unsigned int w0, unsigned int w1,
            unsigned int w2, float sum) {
        const float code = __uint_as_float(field<J>(w0, w1, w2) ^ 0x4b000004u) - 8388612.0f;
        return Dot<J + 1>::run(x, w0, w1, w2, fmaf(x[J], code, sum));
    }
};
template<>
struct Dot<32> {
    static __device__ __forceinline__ float run(const float (&)[32], unsigned int, unsigned int, unsigned int,
            float sum) {
        return sum;
    }
};
}  // namespace contiguous

static __device__ __forceinline__ void contiguous_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) {
    constexpr int kRows = contiguous::kRows;
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int first_row = (blockIdx.x * (blockDim.x >> 5) + warp) * kRows;
    const Layout w(weights, in_features, scale_offset);
    const unsigned int slices = in_features / 1024u, groups = in_features / 64u;
    const unsigned long long row_words = (unsigned long long)groups * 6u;
    const unsigned int* codes = reinterpret_cast<const unsigned int*>(w.codes);
    float sums[kRows] = {};
    euhedral_pdl_begin();
    for (unsigned int slice = 0; slice < slices; slice++) {
        unsigned int words[kRows][3];
        float scales[kRows];
        #pragma unroll
        for (int r = 0; r < kRows; r++) {
            const unsigned int* p = codes + (first_row + r) * row_words + slice * 96u + 3u * lane;
            words[r][0] = p[0];
            words[r][1] = p[1];
            words[r][2] = p[2];
            scales[r] = __half2float(__ushort_as_half(w.scales[(first_row + r) * (unsigned long long)groups
                    + slice * 16u + (lane >> 1)]));
        }
        float x[32];
        const uint4* activation = reinterpret_cast<const uint4*>(input + slice * 1024u + 32u * lane);
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint4 v = activation[i];
            const unsigned int pairs[4] = {v.x, v.y, v.z, v.w};
            #pragma unroll
            for (int j = 0; j < 4; j++) {
                x[i * 8 + j * 2] = __uint_as_float(pairs[j] << 16);
                x[i * 8 + j * 2 + 1] = __uint_as_float(pairs[j] & 0xffff0000u);
            }
        }
        #pragma unroll
        for (int r = 0; r < kRows; r++) {
            const float dot = contiguous::Dot<0>::run(x, words[r][0], words[r][1], words[r][2], 0.0f);
            sums[r] = fmaf(dot, scales[r], sums[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < kRows; r++) {
        #pragma unroll
        for (int distance = 16; distance; distance >>= 1) sums[r] += __shfl_xor_sync(0xffffffffu, sums[r], distance);
    }
    float mine = sums[0];
    #pragma unroll
    for (int r = 1; r < kRows; r++) mine = lane == (unsigned int)r ? sums[r] : mine;
    if (lane < (unsigned int)kRows) write_bf16(output, 0, first_row + lane, out_features, mine);
}

}  // namespace q3
