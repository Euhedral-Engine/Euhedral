#ifndef EUHEDRAL_Q45_DECODE_POLICY_H
#define EUHEDRAL_Q45_DECODE_POLICY_H

#include <stdint.h>

// Single-row Q4/Q5 decode uses the warp-local wide kernel where its layout requirements hold
// (K % 512 == 0, K <= 8192, whole 8-column CTAs) and the grid exceeds 512 CTAs. With weights
// streaming from DRAM it measured 14-20% faster at 896 and 1536 CTAs (Q5 5120 -> 7168 and
// 12288, Q4 5120 -> 7168) and 6% slower at 512 CTAs (Q4 5120 -> 4096), where the per-warp row
// staging is not amortized within one wave.
static inline int euhedral_q45_decode_wide_shape(uint32_t rows, uint32_t in_features, uint32_t out_features) {
    return rows == 1 && in_features != 0 && in_features % 512u == 0 && in_features <= 8192u
            && out_features % 8u == 0 && out_features / 8u > 512u;
}

// Single-row Q4/Q5 decode uses the contiguous-ownership kernel (8 rows per CTA, 1024-K slices) unless
// exact numerics are selected. With cold weights: Q5 5120 -> 12288 64 -> 52 us, 5120 -> 7168
// 42 -> 31 us; Q4 5120 -> 4096 21 -> 15 us, 5120 -> 7168 34 -> 25 us.
static inline int euhedral_q45_decode_contiguous_shape(uint32_t rows, uint32_t in_features, uint32_t out_features) {
    return rows == 1 && in_features != 0 && in_features % 1024u == 0 && out_features != 0 && out_features % 8u == 0;
}

// Q4/Q5 prefill quanta use the balanced tile engine unless exact numerics are selected: 128-row tiles
// from 256 rows (512 rows: Q5 5120 -> 12288 1898 -> 1286 us, 5120 -> 7168 1147 -> 900 us; Q4
// 5120 -> 7168 898 -> 787 us), 64-row tiles from 65 rows (128 rows: Q5 5120 -> 7168 367 -> 258 us,
// Q4 292 -> 212 us). At 64 rows the 64 x 32 kernel fills more CTAs and stays. Returns the tile rows, or 0.
static inline uint32_t euhedral_q45_prefill_wide_rows(uint32_t bits, uint32_t rows, uint32_t in_features,
        uint32_t out_features) {
    if ((bits != 4 && bits != 5) || rows <= 64u || in_features == 0 || in_features % 32u != 0
            || out_features == 0 || out_features % 32u != 0)
        return 0u;
    return rows >= 256u ? 128u : 64u;
}

#endif
