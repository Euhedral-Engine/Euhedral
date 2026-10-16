// Trellis payload to BF16 for the prefill route.
#ifndef EUHEDRAL_TRELLIS_DECODE_CUH
#define EUHEDRAL_TRELLIS_DECODE_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "trellis/codebook.cuh"

// The largest tile ring: w = 128 uint16 words (8 bits per weight).
constexpr int kTrellisMaxWords = 64;

// out[c][r] = bf16(Q[r][c]) for the rotated weight Q [k, n], so out is [n][k] (weights of a BF16 linear). One block per
// 16x16 tile, one thread per position.
//   grid (k / 16) * (n / 16), block 256.
extern "C" __global__ __launch_bounds__(256) void euhedral_trellis_decode_bf16(
        const unsigned short* __restrict__ words, unsigned short* __restrict__ out, int k, int n, int w) {
    __shared__ unsigned int ring[kTrellisMaxWords];
    const int nw = w >> 1, ka = w >> 4;
    const unsigned int mask = (w & 15) == 8 ? 0xAAAAu : 0u;
    const int tiles_n = n >> 4;
    const int ti = blockIdx.x / tiles_n, tj = blockIdx.x % tiles_n;
    const unsigned int* tile = (const unsigned int*)(words + (size_t)blockIdx.x * w);
    const int p = threadIdx.x;
    if (p < nw) ring[p] = tile[p];
    __syncthreads();
    const float value = __half2float(decode_mul1(frac_window(ring, nw, frac_s(p, ka, mask, w) - 16)));
    const int e = tensor_core_element(p);
    out[(size_t)(tj * 16 + (e & 15)) * k + ti * 16 + (e >> 4)] = __bfloat16_as_ushort(__float2bfloat16_rn(value));
}

#endif
