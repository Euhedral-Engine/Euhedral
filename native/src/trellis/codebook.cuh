// The trellis (EXL3) decode: the one source of the mul1 codebook, the fractional ring window, the tensor-core element
// order and the 128-point Hadamard butterfly. The engine's kernels (trellis/kernels.cu) and the quantization module's
// (quantization/src/main/resources/cuda/trellis.cu, which compiles this header from its resources) both include it, so
// the encoder's reconstruction and the engine's decode cannot drift apart.
//
// Portions ported from exllamav3 1.6.1 (MIT License, Copyright (c) 2025 Turboderp): codebook.cuh (mul1 decode) and
// exl3_dq.cuh (the circular bit window of the decoder).
//
// A 256-weight tile is a tail-biting ring of w uint16 words (16 * K bits, or 16 * K + 8 for the half step layout).
// Position p of the tile decodes the 16-bit window ending at ring bit S(p); the windows are read MSB first from the
// ring's uint32 words (little-endian pairs of the stored int16 words). The decode is generic in w.
#ifndef EUHEDRAL_TRELLIS_CODEBOOK_CUH
#define EUHEDRAL_TRELLIS_CODEBOOK_CUH

#include <cuda_fp16.h>

/// Two mul1 codebook entries from the precomputed products x0 = state0 * 0x83DCD12D, x1 = state1 * 0x83DCD12D.
__device__ __forceinline__ half2 decode_mul1_product_2(unsigned int x0, unsigned int x1) {
    const unsigned int acc = 0x6400u;
    unsigned int sum0 = __dp4a(x0, 0x01010101u, acc);
    unsigned int sum1 = __dp4a(x1, 0x01010101u, acc);
    half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
    half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
    half h0 = __ushort_as_half((unsigned short)sum0);
    half h1 = __ushort_as_half((unsigned short)sum1);
    return __hfma2(__halves2half2(h0, h1), k_inv_h2, k_bias_h2);
}

__device__ __forceinline__ half2 decode_mul1_2(unsigned int x0, unsigned int x1) {
    return decode_mul1_product_2(x0 * 0x83DCD12Du, x1 * 0x83DCD12Du);
}

__device__ __forceinline__ half decode_mul1(unsigned int x) {
    x *= 0x83DCD12Du;
    unsigned int sum = __dp4a(x, 0x01010101u, 0x6400u);
    return __hfma(__ushort_as_half((unsigned short)sum), __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}

/// Bits of position i: ka, plus one where the half step mask has bit (i & 15) set.
__device__ __forceinline__ int frac_d(int i, int ka, unsigned int mask) { return ka + ((mask >> (i & 15)) & 1); }

/// Ring position where window i ends; `bpb` is the uint16 words per tile (w). Closed form of
/// (i >> 4) * w + sum_{j <= (i & 15)} d(j).
__device__ __forceinline__ int frac_s(int i, int ka, unsigned int mask, int bpb) {
    const int m = i & 15;
    return (i >> 4) * bpb + ka * (m + 1) + __popc(mask & ((2u << m) - 1u));
}

/// 16-bit window starting at ring bit `start` (any integer; it wraps), from MSB-first 32-bit words.
__device__ __forceinline__ unsigned int frac_window(const unsigned int* words, int nw, int start) {
    const int R = nw * 32;
    start %= R;
    if (start < 0) start += R;
    const int wi = start >> 5, o = start & 31;
    const int next = wi + 1 == nw ? 0 : wi + 1;
    const unsigned long long v = ((unsigned long long)words[wi] << 32) | (unsigned long long)words[next];
    return (unsigned int)((v >> (48 - o)) & 0xFFFFu);
}

/// Row-major element (r * 16 + c) of a 16x16 tile at tensor-core position `pos`: for t = pos / 8, r0 = (t % 4) * 2 and
/// c0 = t / 4, the eight positions of t are rows r0, r0 + 1, r0 + 8, r0 + 9 at columns c0, then c0 + 8.
__device__ __forceinline__ int tensor_core_element(int pos) {
    const int t = pos >> 3, j = pos & 7;
    const int r0 = (t & 3) * 2;
    const int row = r0 + ((j & 3) >> 1) * 8 + (j & 1);
    const int col = (t >> 2) + (j >> 2) * 8;
    return row * 16 + col;
}

constexpr float kTrellisInvSqrt128 = 0.08838834764831845f;

/// The 128-point Sylvester Hadamard, H[i][j] = (-1)^popcount(i & j) / sqrt(128), of the 128 floats at `s` (shared
/// memory), by the butterfly stages h = 1, 2, ..., 64: pair i = ((t / h) * 2h) + (t % h) becomes (a + b, a - b); then
/// the scale. Called by the 128 threads t = 0..127 of a block together (it synchronizes the block); several groups of
/// 128 threads may work on disjoint arrays as long as every thread of the block calls it the same number of times.
__device__ __forceinline__ void trellis_hadamard128(float* s, int t) {
    __syncthreads();
    for (int h = 1; h < 128; h <<= 1) {
        if (t < 64) {
            const int i = ((t / h) * 2 * h) + (t % h);
            const float a = s[i], b = s[i + h];
            s[i] = a + b;
            s[i + h] = a - b;
        }
        __syncthreads();
    }
    s[t] = s[t] * kTrellisInvSqrt128;
    __syncthreads();
}

/// The same Hadamard as trellis_hadamard128 on 128 floats held by one warp, four consecutive per lane (element
/// 4 * lane + j in v[j]), in registers: every stage forms the same a + b and a - b, so the results are bitwise those of
/// trellis_hadamard128. No block synchronization.
__device__ __forceinline__ void trellis_hadamard128_warp(float (&v)[4], int lane) {
    // h = 1 and 2 pair elements within a lane.
    float t0 = v[0] + v[1], t1 = v[0] - v[1], t2 = v[2] + v[3], t3 = v[2] - v[3];
    v[0] = t0 + t2;
    v[1] = t1 + t3;
    v[2] = t0 - t2;
    v[3] = t1 - t3;
    // h = 4 .. 64 pair lane l with l ^ (h / 4); the lower lane holds a, the upper b.
#pragma unroll
    for (int h = 1; h < 32; h <<= 1) {
        const bool upper = (lane & h) != 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float other = __shfl_xor_sync(0xFFFFFFFFu, v[j], h);
            v[j] = upper ? other - v[j] : v[j] + other;
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = v[j] * kTrellisInvSqrt128;
}

#endif
