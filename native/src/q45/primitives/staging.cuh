#pragma once
#include <cuda_bf16.h>
#include "packed_load.cuh"

namespace q45 {
// hi/lo split of code * scale, packed as (hi << 16) | lo. hi = rn(weight),
// lo = rn(weight - hi), the same two BF16 components the original kernel
// staged. SCOPE: thread-local; pure.
static __device__ __forceinline__ unsigned int split_entry(int code, float scale) {
    float weight = apply_scale(code, scale);
    __nv_bfloat16 hi = __float2bfloat16(weight);
    __nv_bfloat16 lo = __float2bfloat16(weight - __bfloat162float(hi));
    return ((unsigned int)__bfloat16_as_ushort(hi) << 16) | __bfloat16_as_ushort(lo);
}

// Stage a lane's two weights (K = i, i + 1) from its code pair. Both splits
// complete in registers before any shared store.
// SCOPE: thread-local; elements i and i + 1 are owned by the calling thread.
template<int BITS>
static __device__ __forceinline__ void stage_split_pair(
        __nv_bfloat16* hi, __nv_bfloat16* lo, unsigned int i, unsigned int pair, float scale) {
    unsigned int e0 = split_entry(unpack_code<BITS>(pair, 0), scale);
    unsigned int e1 = split_entry(unpack_code<BITS>(pair, 1), scale);
    hi[i] = __ushort_as_bfloat16((unsigned short)(e0 >> 16));
    lo[i] = __ushort_as_bfloat16((unsigned short)e0);
    hi[i + 1] = __ushort_as_bfloat16((unsigned short)(e1 >> 16));
    lo[i + 1] = __ushort_as_bfloat16((unsigned short)e1);
}

// Stage one decoded G64 word set for column `col` (column-major B tile with
// physical stride STRIDE). Out-of-matrix columns pass an all-zero word set,
// which stages zero weights. SCOPE: warp-collective.
template<int BITS, int STRIDE>
static __device__ __forceinline__ void stage_group_words(
        __nv_bfloat16* hi, __nv_bfloat16* lo, unsigned int col, unsigned int word, unsigned int lane) {
    unsigned int pair = group_pair<BITS>(word, lane);
    float scale = group_scale<BITS>(word);
    stage_split_pair<BITS>(hi, lo, col * STRIDE + lane * 2, pair, scale);
}

// Copy a ROWS x kGroup BF16 activation tile into a borrowed shared tile (row
// major, stride STRIDE) without numeric conversion; rows >= `rows` are zero.
// in_features is a multiple of 128, so K never leaves the matrix.
// SCOPE: CTA-cooperative; no barrier.
template<int ROWS, int THREADS, int STRIDE>
static __device__ __forceinline__ void stage_activation_tile(
        __nv_bfloat16* tile, const unsigned short* input, unsigned int rows, unsigned int in_features,
        unsigned int row_start, unsigned int k_base, unsigned int thread) {
    static_assert(STRIDE >= (int)kGroup && STRIDE % 16 == 0, "MMA A stride must cover the K group");
    for (unsigned int i = thread; i < ROWS * kGroup; i += THREADS) {
        unsigned int r = row_start + i / kGroup, k = k_base + i % kGroup;
        tile[(i / kGroup) * STRIDE + i % kGroup] =
                __ushort_as_bfloat16(r < rows ? input[(unsigned long long)r * in_features + k] : (unsigned short)0);
    }
}

// Decode and stage one K group for COLS output columns, loading as it goes.
// Each warp stages columns warp, warp + WARPS, ... SCOPE: per-warp collective.
template<int BITS, int COLS, int WARPS, int STRIDE>
static __device__ __forceinline__ void stage_weight_tile(
        __nv_bfloat16* hi, __nv_bfloat16* lo, const Layout<BITS>& w, unsigned int out_start,
        unsigned int out_features, unsigned int k_base, unsigned int warp, unsigned int lane) {
    static_assert(STRIDE >= (int)kGroup && STRIDE % 16 == 0, "MMA B stride must cover the K group");
    for (unsigned int col = warp; col < COLS; col += WARPS) {
        unsigned int word = out_start + col < out_features
                ? load_group_word<BITS>(w, w.group(out_start + col, k_base / kGroup), lane) : 0u;
        stage_group_words<BITS, STRIDE>(hi, lo, col, word, lane);
    }
}

}  // namespace q45
