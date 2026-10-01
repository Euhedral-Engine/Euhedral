#pragma once
#include "packed_load.cuh"
#include "decode.cuh"

namespace q3 {
// Store an FP32 weight as two BF16 components (hi + lo) in borrowed shared tiles.
// hi = rn(weight), lo = rn(weight - hi). For Q3 weights this pair is exact, so the
// MMA path does not silently round the FP16-scaled value to BF16.
// SCOPE: thread-local; the element index is owned by the calling thread.
static __device__ __forceinline__ void stage_split_weight(
        __nv_bfloat16* hi, __nv_bfloat16* lo, unsigned int i, float weight) {
    hi[i] = __float2bfloat16(weight);
    lo[i] = __float2bfloat16(weight - __bfloat162float(hi[i]));
}

// hi/lo split of decode(raw) * scale, packed as (hi << 16) | lo. Bitwise
// identical to stage_split_weight (exact for finite scales).
// SCOPE: thread-local; pure.
static __device__ __forceinline__ unsigned int split_entry(unsigned int raw, float scale) {
    float weight = apply_scale((int)raw - ((raw & 4u) ? 8 : 0), scale);
    __nv_bfloat16 hi = __float2bfloat16(weight);
    __nv_bfloat16 lo = __float2bfloat16(weight - __bfloat162float(hi));
    return ((unsigned int)__bfloat16_as_ushort(hi) << 16) | __bfloat16_as_ushort(lo);
}

// Stage a lane's two weights (K = i, i + 1) from its 6-bit code pair.
// Both splits complete in registers before any shared store, so no store
// waits on the other element's conversion chain.
// SCOPE: thread-local; elements i and i + 1 are owned by the calling thread.
// PARTS 2 stages hi and lo (hi + lo is code * scale exactly); PARTS 1 stages hi only, the BF16
// rounding of code * scale, and leaves `lo` untouched.
template<int PARTS = 2>
static __device__ __forceinline__ void stage_split_pair(
        __nv_bfloat16* hi, __nv_bfloat16* lo, unsigned int i, unsigned int codes, float scale) {
    unsigned int e0 = split_entry(codes & 7u, scale);
    unsigned int e1 = split_entry((codes >> 3) & 7u, scale);
    hi[i] = __ushort_as_bfloat16((unsigned short)(e0 >> 16));
    hi[i + 1] = __ushort_as_bfloat16((unsigned short)(e1 >> 16));
    if (PARTS > 1) {
        lo[i] = __ushort_as_bfloat16((unsigned short)e0);
        lo[i + 1] = __ushort_as_bfloat16((unsigned short)e1);
    }
}

// Decode and stage one K group for COLS output columns into borrowed hi/lo tiles
// (column-major: column c occupies [c * STRIDE, c * STRIDE + kGroup)). Each warp
// stages columns warp, warp + WARPS, ...; columns >= out_features stage zeros.
// SCOPE: each warp is warp-collective over its columns; no barrier. The strategy
// must barrier before MMA consumption and again before re-staging.
template<int COLS, int WARPS, int STRIDE = kGroup, int PARTS = 2>
static __device__ __forceinline__ void stage_weight_tile(
        __nv_bfloat16* hi, __nv_bfloat16* lo, const Layout& w, unsigned int out_start,
        unsigned int out_features, unsigned int k_base, unsigned int warp, unsigned int lane) {
    static_assert(STRIDE >= kGroup && STRIDE % 8 == 0, "WMMA B stride must cover the K group");
    for (unsigned int col = warp; col < COLS; col += WARPS) {
        unsigned int codes = 0;
        float scale = 0.0f;
        if (out_start + col < out_features) {
            unsigned long long g = w.group(out_start + col, k_base / kGroup);
            codes = load_packed_pair(w, g, lane);
            scale = load_group_scale(w, g, lane);
        }
        stage_split_pair<PARTS>(hi, lo, col * STRIDE + lane * 2, codes, scale);
    }
}


}  // namespace q3
