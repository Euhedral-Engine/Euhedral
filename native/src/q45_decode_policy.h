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

// Q5 prefill quanta from 256 rows use the 128 x 64 tile engine unless exact numerics are selected
// (512 rows: 5120 -> 12288 1899 -> 1401 us, 5120 -> 7168 1149 -> 990 us). Q4 keeps the 64 x 32
// kernel, which the engine only tied at 128 x 64 and lost to at 64 x 64. Returns the tile rows, or 0.
static inline uint32_t euhedral_q45_prefill_wide_rows(uint32_t bits, uint32_t rows, uint32_t in_features,
        uint32_t out_features) {
    return bits == 5 && rows >= 256u && in_features != 0 && in_features % 32u == 0
            && out_features != 0 && out_features % 32u == 0 ? 128u : 0u;
}

#endif
