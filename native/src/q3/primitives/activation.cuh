#pragma once
#include "../numeric.cuh"

namespace q3 {
// Stage a ROWS x K_TILE activation tile from global BF16 into a borrowed shared
// tile (row-major, physical stride STRIDE), zero-filling rows >= `rows` and
// columns >= `in_features`. The logical K tile is always K_TILE wide.
// SCOPE: CTA-cooperative over THREADS threads; no barrier. The strategy must
// barrier before any consumer reads `tile`, and again before it is re-staged.
template<int ROWS, int K_TILE, int THREADS, int STRIDE = K_TILE>
static __device__ __forceinline__ void stage_activation_tile(
        __nv_bfloat16* tile, const unsigned short* input, unsigned int rows, unsigned int in_features,
        unsigned int row_start, unsigned int k_base, unsigned int thread) {
    static_assert(STRIDE >= K_TILE && STRIDE % 8 == 0, "WMMA A stride must cover the K tile");
    for (unsigned int i = thread; i < ROWS * K_TILE; i += THREADS) {
        unsigned int r = row_start + i / K_TILE, k = k_base + i % K_TILE;
        tile[(i / K_TILE) * STRIDE + i % K_TILE] = __float2bfloat16(r < rows && k < in_features
                ? bf16_to_float(input[(unsigned long long)r * in_features + k]) : 0.0f);
    }
}

// Load a lane's K-stripe activations into registers for R rows: x[r][s] holds
// column k_base + lane + 32*s, zero outside the matrix.
// SCOPE: thread-local reads; OWNS `x` (caller registers) until the next K step.
template<int R, int STRIPES>
static __device__ __forceinline__ void load_activation_stripes(
        float (&x)[R][STRIPES], const unsigned short* input, unsigned int rows, unsigned int in_features,
        unsigned int first_row, unsigned int k_base, unsigned int lane) {
    #pragma unroll
    for (int r = 0; r < R; r++) {
        #pragma unroll
        for (int s = 0; s < STRIPES; s++) {
            unsigned int column = k_base + lane + s * 32;
            unsigned long long offset = (unsigned long long)(first_row + r) * in_features + column;
            x[r][s] = first_row + r < rows && column < in_features ? bf16_to_float(input[offset]) : 0;
        }
    }
}


}  // namespace q3
