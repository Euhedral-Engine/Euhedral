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

// Decode and stage one K group for COLS output columns into borrowed hi/lo tiles
// (column-major: column c occupies [c * STRIDE, c * STRIDE + kGroup)). Each warp
// stages columns warp, warp + WARPS, ...; columns >= out_features stage zeros.
// SCOPE: each warp is warp-collective over its columns; no barrier. The strategy
// must barrier before MMA consumption and again before re-staging.
template<int COLS, int WARPS, int STRIDE = kGroup>
static __device__ __forceinline__ void stage_weight_tile(
        __nv_bfloat16* hi, __nv_bfloat16* lo, const Layout& w, unsigned int out_start,
        unsigned int out_features, unsigned int k_base, unsigned int warp, unsigned int lane) {
    static_assert(STRIDE >= kGroup && STRIDE % 16 == 0, "WMMA B stride must cover the K group");
    for (unsigned int col = warp; col < COLS; col += WARPS) {
        unsigned int codes = 0;
        float scale = 0.0f;
        if (out_start + col < out_features) {
            unsigned long long g = w.group(out_start + col, k_base / kGroup);
            codes = load_packed_pair(w, g, lane);
            scale = load_group_scale(w, g, lane);
        }
        #pragma unroll
        for (int p = 0; p < 2; p++)
            stage_split_weight(hi, lo, col * STRIDE + lane * 2 + p, apply_scale(decode_code(codes, p), scale));
    }
}


}  // namespace q3
