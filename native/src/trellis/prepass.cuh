// The input side of a trellis linear: out[g][r] = round(H128(x[r] * suh[g])) for each of the s input scale vectors.
#ifndef EUHEDRAL_TRELLIS_PREPASS_CUH
#define EUHEDRAL_TRELLIS_PREPASS_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "trellis/codebook.cuh"

// x bf16 [rows][k], suh f16 [s][k], out by `layout`: 0 FP16 [s][rows][k] in the tensor-core GEMM's fragment order,
// 1 BF16 [s][rows][k] in column order, 2 the FP8 layout of trellis/fp8.cuh: E4M3 bytes [s][rows][k] with one UE8M0
// scale per 32 columns, the bytes [s][rows][k / 32] after them. A block's scale 2^e is the least with every |v| / 2^e
// at most 448, the largest E4M3; its bytes are E4M3(v / 2^e), rounded to nearest even, in fragment order: in each
// k16 tile's 16 bytes, the elements 2q, 2q + 1, 2q + 8, 2q + 9 an m16n8k32 A lane q reads are contiguous at 4q. Each warp transforms one 128-column block of one row in registers
// (trellis_hadamard128_warp), four per block. Rows are independent, so a row's bits do not depend on how many rows run
// with it.
//   grid (ceil(k / 512), rows, s), block 128.
extern "C" __global__ __launch_bounds__(128) void euhedral_trellis_prepass(
        const unsigned short* __restrict__ x, const unsigned short* __restrict__ suh, unsigned short* __restrict__ out,
        int rows, int k, int s, int layout) {
    const int lane = threadIdx.x & 31, r = blockIdx.y, g = blockIdx.z;
    const int block = blockIdx.x * 4 + (threadIdx.x >> 5);
    if (block * 128 >= k) return;
    const size_t column = (size_t)block * 128 + lane * 4;
    const uint2 xs = *reinterpret_cast<const uint2*>(x + (size_t)r * k + column);
    const uint2 scales = *reinterpret_cast<const uint2*>(suh + (size_t)g * k + column);
    const unsigned int xw[2] = {xs.x, xs.y}, sw[2] = {scales.x, scales.y};
    float v[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const unsigned int xb = (xw[j >> 1] >> (16 * (j & 1))) & 0xFFFFu, sb = (sw[j >> 1] >> (16 * (j & 1))) & 0xFFFFu;
        v[j] = __uint_as_float(xb << 16) * __half2float(__ushort_as_half((unsigned short)sb));
    }
    trellis_hadamard128_warp(v, lane);
    unsigned short* row = out + ((size_t)g * rows + r) * k;
    if (layout == 2) {
        // The block of 32 columns is eight lanes' values.
        float amax = fmaxf(fmaxf(fabsf(v[0]), fabsf(v[1])), fmaxf(fabsf(v[2]), fabsf(v[3])));
#pragma unroll
        for (int offset = 1; offset < 8; offset <<= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, offset));
        int e = -127;
        if (amax > 0.0f) {
            int exponent;
            frexpf(amax, &exponent);  // amax < 2^exponent
            e = max(-127, min(127, exponent - 9));  // 448 = 1.75 * 2^8
            if (e > -127 && ldexpf(amax, -(e - 1)) <= 448.0f) --e;
            if (e < 127 && ldexpf(amax, -e) > 448.0f) ++e;
        }
        unsigned char* bytes = reinterpret_cast<unsigned char*>(out) + ((size_t)g * rows + r) * k;
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const size_t c = column + 2 * pair;
            const int kt = (int)(c & 15);
            const size_t at = (c & ~(size_t)15) + 4 * ((kt & 7) >> 1) + 2 * (kt >> 3);
            unsigned short e4m3;
            asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n"
                : "=h"(e4m3)
                : "f"(ldexpf(v[2 * pair + 1], -e)), "f"(ldexpf(v[2 * pair], -e)));
            *reinterpret_cast<unsigned short*>(bytes + at) = e4m3;
        }
        if ((lane & 7) == 0)
            reinterpret_cast<unsigned char*>(out)[(size_t)s * rows * k + ((size_t)g * rows + r) * (k >> 5)
                    + (column >> 5)] = (unsigned char)(e + 127);
    } else if (layout != 0) {
        unsigned short bits[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) bits[j] = __bfloat16_as_ushort(__float2bfloat16_rn(v[j]));
        *reinterpret_cast<uint2*>(row + column) =
                make_uint2(bits[0] | (unsigned int)bits[1] << 16, bits[2] | (unsigned int)bits[3] << 16);
    } else {
        // Fragment order: in each k16 group, the elements 2q, 2q + 1, 2q + 8, 2q + 9 an m16n8k16 A lane reads are
        // contiguous at 4q. This lane's pairs (4 lane, + 1) and (4 lane + 2, + 3) stay pairs.
#pragma unroll
        for (int pair = 0; pair < 2; ++pair) {
            const int e = (lane * 4 + pair * 2) & 15;
            const size_t at = (column & ~(size_t)15) + 4 * ((e & 7) >> 1) + 2 * (e >> 3);
            const unsigned int bits = __half_as_ushort(__float2half_rn(v[2 * pair]))
                    | (unsigned int)__half_as_ushort(__float2half_rn(v[2 * pair + 1])) << 16;
            *reinterpret_cast<unsigned int*>(row + at) = bits;
        }
    }
}

#endif
