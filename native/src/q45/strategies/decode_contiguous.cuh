#pragma once
#include "decode.cuh"
#include "pdl.cuh"

namespace q45 {
// Single-row Q4/Q5 decode with contiguous lane ownership.
//
// A G64 group's 32 code bytes hold its 64 four-bit codes in K order, and its 8 fifth-bit bytes (Q5)
// hold one bit per code, so a lane owns one half group: 16 code bytes read with one 16-byte load,
// plus one 32-bit word of fifth bits, covering 32 contiguous K values at compile-time positions. A
// warp's lanes cover 512 contiguous code bytes (1024 K) of a row per slice; each warp owns two rows
// and reuses its 32 activations for both. Each lane forms an FP32 dot product of its codes, scales
// it once by the group scale and accumulates across slices; a warp shuffle reduces each row.
//
// Numerical contract: FP32 accumulation in a different order from cooperative/wide decode (the exact
// kernels, selected by exact numerics); within one BF16 ulp of them.
//
// Requirements (checked by host dispatch): one row, in_features a multiple of 1024, out_features a
// multiple of 8, a 16-byte aligned input and weight base.
template<int BITS>
static __device__ __forceinline__ void contiguous_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int in_features, unsigned int out_features) {
    constexpr int kRows = 2;
    constexpr unsigned int sign = 1u << (BITS - 1);
    const unsigned int lane = threadIdx.x & 31u, warp = threadIdx.x >> 5;
    const unsigned int first_row = (blockIdx.x * (blockDim.x >> 5) + warp) * kRows;
    const Layout<BITS> w(weights, in_features, out_features);
    const unsigned int slices = in_features / 1024u, groups = in_features / 64u;
    float sums[kRows] = {};
    euhedral_pdl_begin();
    for (unsigned int slice = 0; slice < slices; slice++) {
        uint4 codes[kRows];
        unsigned int high[kRows];
        float scales[kRows];
        #pragma unroll
        for (int r = 0; r < kRows; r++) {
            const unsigned long long first_group = (unsigned long long)(first_row + r) * groups;
            codes[r] = reinterpret_cast<const uint4*>(w.code_words(first_group))[slice * 32u + lane];
            high[r] = BITS == 5 ? w.high_words(first_group)[slice * 32u + lane] : 0u;
            scales[r] = scale_to_float(w.scales[first_group + slice * 16u + (lane >> 1)]);
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
            const unsigned int words[4] = {codes[r].x, codes[r].y, codes[r].z, codes[r].w};
            float dot = 0.0f;
            #pragma unroll
            for (int j = 0; j < 32; j++) {
                unsigned int value = (words[j >> 3] >> ((j & 7) * 4)) & 15u;
                if (BITS == 5) value |= ((high[r] >> j) & 1u) << 4;
                // 2^23 + (value ^ sign) - (2^23 + sign) is the signed code, exactly.
                const float code = __uint_as_float(value ^ (0x4b000000u | sign)) - (8388608.0f + (float)sign);
                dot = fmaf(x[j], code, dot);
            }
            sums[r] = fmaf(dot, scales[r], sums[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < kRows; r++) {
        #pragma unroll
        for (int distance = 16; distance; distance >>= 1) sums[r] += __shfl_xor_sync(0xffffffffu, sums[r], distance);
    }
    if (lane < (unsigned int)kRows) write_bf16(output, 0, first_row + lane, out_features, lane == 0 ? sums[0] : sums[1]);
}

}  // namespace q45
