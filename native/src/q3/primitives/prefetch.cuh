#pragma once
#include "staging.cuh"
#include "activation.cuh"

namespace q3 {
// The next compact K group lives in the same four compute warps' registers.
// Each thread owns its activation pairs; six lanes per warp own the packed
// words for each assigned output column, and lane zero owns its FP16 scale.
template<class Tile>
struct CompactPrefetch {
    static constexpr int kThreads = Tile::kWarps * 32;
    static constexpr int kPairs = Tile::kRows * kGroup / (kThreads * 2);
    static constexpr int kColumns = Tile::kCols / Tile::kWarps;
    unsigned int activation[kPairs];
    unsigned int words[kColumns];
    unsigned short scales[kColumns];
};

template<class Tile>
static __device__ __forceinline__ void prefetch_compact_tile(
        CompactPrefetch<Tile>& next, const unsigned short* input, const Layout& w,
        unsigned int rows, unsigned int in_features, unsigned int out_features,
        unsigned int row_start, unsigned int out_start, unsigned int k_base,
        unsigned int thread, unsigned int warp, unsigned int lane) {
    constexpr int kThreads = CompactPrefetch<Tile>::kThreads;
    #pragma unroll
    for (int j = 0; j < CompactPrefetch<Tile>::kPairs; ++j) {
        unsigned int i = thread * 2 + j * kThreads * 2;
        unsigned int r = row_start + i / kGroup, k = k_base + i % kGroup;
        unsigned int pair = 0;
        if (r < rows && k < in_features) {
            // The caller guarantees a 4-byte-aligned base and even row width.
            // k is even, so an in-range pair always contains two BF16 elements.
            unsigned long long offset = (unsigned long long)r * in_features + k;
            asm volatile("ld.global.u32 %0, [%1];" : "=r"(pair) : "l"(input + offset));
        }
        next.activation[j] = pair;
    }
    #pragma unroll
    for (int j = 0; j < CompactPrefetch<Tile>::kColumns; ++j) {
        unsigned int col = warp + j * Tile::kWarps;
        if (out_start + col < out_features) {
            unsigned long long g = w.group(out_start + col, k_base / kGroup);
            next.words[j] = lane < 6 ? ((const unsigned int*)w.group_bytes(g))[lane] : 0;
            next.scales[j] = lane == 0 ? w.scales[g] : 0;
        } else {
            next.words[j] = 0;
            next.scales[j] = 0;
        }
    }
}

template<class Tile, int STRIDE>
static __device__ __forceinline__ void stage_prefetched_activation(
        __nv_bfloat16* a, const CompactPrefetch<Tile>& next, unsigned int thread) {
    constexpr int kThreads = CompactPrefetch<Tile>::kThreads;
    #pragma unroll
    for (int j = 0; j < CompactPrefetch<Tile>::kPairs; ++j) {
        unsigned int i = thread * 2 + j * kThreads * 2;
        unsigned int row = i / kGroup, k = i % kGroup;
        unsigned int bits = next.activation[j];
        a[row * STRIDE + k] = __float2bfloat16(bf16_to_float((unsigned short)bits));
        a[row * STRIDE + k + 1] = __float2bfloat16(bf16_to_float((unsigned short)(bits >> 16)));
    }
}

template<class Tile, int STRIDE, int PARTS = 2>
static __device__ __forceinline__ void stage_prefetched_weights(
        __nv_bfloat16* hi, __nv_bfloat16* lo, const CompactPrefetch<Tile>& next,
        unsigned int warp, unsigned int lane) {
    #pragma unroll
    for (int j = 0; j < CompactPrefetch<Tile>::kColumns; ++j) {
        unsigned int col = warp + j * Tile::kWarps;
        unsigned int bit = lane * 6u;
        unsigned int low = __shfl_sync(0xffffffffu, next.words[j], bit >> 5);
        unsigned int high = __shfl_sync(0xffffffffu, next.words[j], (bit >> 5) + 1);
        unsigned int codes = (unsigned int)((((unsigned long long)high << 32) | low) >> (bit & 31u)) & 63u;
        float scale = fp16_to_float(__shfl_sync(0xffffffffu, (unsigned int)next.scales[j], 0));
        stage_split_pair<PARTS>(hi, lo, col * STRIDE + lane * 2, codes, scale);
    }
}

}  // namespace q3
