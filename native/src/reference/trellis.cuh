// Scalar reference of a trellis linear: FP32 throughout, no FP16 rounding of the activations. It is compiled into the
// trellis module (trellis/kernels.cu), so the shared linear table loads one module for all of its trellis entries.
#ifndef EUHEDRAL_REFERENCE_TRELLIS_CUH
#define EUHEDRAL_REFERENCE_TRELLIS_CUH

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include "trellis/codebook.cuh"

// One CTA per output element (r, c): the block's rotated input u = H128(x * suh[seg(c)]) lives in dynamic shared
// memory (4 * k bytes); thread t computes z[t] = sum_k u[k] * Q[k][c0 + t] for the 128-column block c0 holding c, the
// block applies the output Hadamard and the thread of c writes svh[c] * H(z)[c].
//   grid rows * n, block 128, dynamic shared 4 * k.
extern "C" __global__ __launch_bounds__(128) void euhedral_trellis_reference(
        const unsigned short* __restrict__ x, const unsigned short* __restrict__ words,
        const unsigned short* __restrict__ suh, const unsigned short* __restrict__ svh, const int* __restrict__ ends,
        int s, unsigned short* __restrict__ out, int rows, int k, int n, int w) {
    extern __shared__ float u[];
    __shared__ float z[128];
    const int t = threadIdx.x;
    const size_t index = blockIdx.x;
    const int r = (int)(index / n), c = (int)(index % n);
    if (r >= rows) return;
    const int c0 = c & ~127;
    int g = 0;
    while (g + 1 < s && c0 >= ends[g]) ++g;
    for (int b = 0; b < k / 128; ++b) {
        const size_t column = (size_t)b * 128 + t;
        z[t] = __uint_as_float((unsigned int)x[(size_t)r * k + column] << 16)
                * __half2float(__ushort_as_half(suh[(size_t)g * k + column]));
        trellis_hadamard128(z, t);
        u[b * 128 + t] = z[t];
        __syncthreads();
    }
    const int nw = w >> 1, ka = w >> 4;
    const unsigned int mask = (w & 15) == 8 ? 0xAAAAu : 0u;
    const int tiles_n = n >> 4;
    const int column = c0 + t;
    const int tj = column >> 4, cl = column & 15;
    float sum = 0.0f;
    for (int row = 0; row < k; ++row) {
        const int ti = row >> 4, rl = row & 15;
        // The position whose tensor-core element is (rl, cl): the inverse of tensor_core_element.
        const int j = (rl & 1) | (((rl >> 3) & 1) << 1) | ((cl >> 3) << 2);
        const int p = ((((cl & 7) << 2) | ((rl >> 1) & 3)) << 3) | j;
        const unsigned int* tile = (const unsigned int*)(words + ((size_t)ti * tiles_n + tj) * w);
        const float q = __half2float(decode_mul1(frac_window(tile, nw, frac_s(p, ka, mask, w) - 16)));
        sum += u[row] * q;
    }
    z[t] = sum;
    trellis_hadamard128(z, t);
    if (t == (c & 127))
        out[(size_t)r * n + c] =
                __bfloat16_as_ushort(__float2bfloat16_rn(__half2float(__ushort_as_half(svh[c])) * z[t]));
}

#endif
