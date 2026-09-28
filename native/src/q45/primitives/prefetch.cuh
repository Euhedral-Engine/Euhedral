#pragma once
#include "staging.cuh"

namespace q45 {
// The next compact K group lives in the same compute warps' registers: each
// thread owns its BF16 activation pairs, and each warp owns one G64 word set
// (codes, Q5 fifth bits and scale) per assigned output column.
template<class Tile, int BITS>
struct CompactPrefetch {
    static constexpr int kThreads = Tile::kWarps * 32;
    static constexpr int kPairs = Tile::kRows * (int)kGroup / (kThreads * 2);
    static constexpr int kColumns = Tile::kCols / Tile::kWarps;
    unsigned int activation[kPairs];
    unsigned int words[kColumns];
};

// Requires a 4-byte-aligned input base; in_features is a multiple of 128, so
// every row stride is even and a pair never straddles rows.
template<class Tile, int BITS>
static __device__ __forceinline__ void prefetch_compact_tile(
        CompactPrefetch<Tile, BITS>& next, const unsigned short* input, const Layout<BITS>& w,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned int row_start, unsigned int out_start, unsigned int k_base,
        unsigned int thread, unsigned int warp, unsigned int lane) {
    constexpr int kThreads = CompactPrefetch<Tile, BITS>::kThreads;
    #pragma unroll
    for (int j = 0; j < CompactPrefetch<Tile, BITS>::kPairs; ++j) {
        unsigned int i = thread * 2 + j * kThreads * 2;
        unsigned int r = row_start + i / kGroup, k = k_base + i % kGroup;
        next.activation[j] = r < rows
                ? *(const unsigned int*)(input + (unsigned long long)r * in_features + k) : 0u;
    }
    #pragma unroll
    for (int j = 0; j < CompactPrefetch<Tile, BITS>::kColumns; ++j) {
        unsigned int col = warp + j * Tile::kWarps;
        next.words[j] = out_start + col < out_features
                ? load_group_word<BITS>(w, w.group(out_start + col, k_base / kGroup), lane) : 0u;
    }
}

template<class Tile, int BITS, int STRIDE>
static __device__ __forceinline__ void stage_prefetched_activation(
        __nv_bfloat16* a, const CompactPrefetch<Tile, BITS>& next, unsigned int thread) {
    constexpr int kThreads = CompactPrefetch<Tile, BITS>::kThreads;
    #pragma unroll
    for (int j = 0; j < CompactPrefetch<Tile, BITS>::kPairs; ++j) {
        unsigned int i = thread * 2 + j * kThreads * 2;
        unsigned int row = i / kGroup, k = i % kGroup;
        unsigned int bits = next.activation[j];
        a[row * STRIDE + k] = __ushort_as_bfloat16((unsigned short)bits);
        a[row * STRIDE + k + 1] = __ushort_as_bfloat16((unsigned short)(bits >> 16));
    }
}

template<class Tile, int BITS, int STRIDE>
static __device__ __forceinline__ void stage_prefetched_weights(
        __nv_bfloat16* hi, __nv_bfloat16* lo, const CompactPrefetch<Tile, BITS>& next,
        unsigned int warp, unsigned int lane) {
    #pragma unroll
    for (int j = 0; j < CompactPrefetch<Tile, BITS>::kColumns; ++j)
        stage_group_words<BITS, STRIDE>(hi, lo, warp + j * Tile::kWarps, next.words[j], lane);
}

}  // namespace q45
