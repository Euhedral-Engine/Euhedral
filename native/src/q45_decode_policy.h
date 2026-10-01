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

#endif
