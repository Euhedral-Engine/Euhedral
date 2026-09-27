#ifndef EUHEDRAL_Q3_PREFILL_POLICY_H
#define EUHEDRAL_Q3_PREFILL_POLICY_H

#include <stdint.h>

// The explicit 64-row entry point remains independent of AUTO's shape policy.
static inline int euhedral_q3_wide_prefill(int mode, uint32_t rows,
        uint32_t in_features, uint32_t out_features) {
    if (mode == 3) return 1;
    if (mode != 2) return 0;
    // Smaller mixer batches win at the operator level but have inconsistent
    // full-model gains; route only repeatably faster prefill chunks to tile64.
    if (in_features == 6144 && out_features == 5120) return rows >= 512;
    return rows >= 64 && ((in_features == 5120 && out_features == 34816)
            || (in_features == 17408 && out_features == 5120));
}

// AUTO keeps the WMMA reference leaf where the explicit mma.sync 64-row leaf
// measured slower: the down projection at exactly 64 rows, the only down size
// that is both wide-routed (rows >= 64) and a single 64-row band. Explicit
// mode 3 requests always use the default 64-row kernel.
static inline int euhedral_q3_wmma_prefill64(int mode, uint32_t rows,
        uint32_t in_features, uint32_t out_features) {
    return mode == 2 && rows <= 64 && in_features == 17408 && out_features == 5120
            && euhedral_q3_wide_prefill(mode, rows, in_features, out_features);
}

// Prefill kernel selection for modes 2 and 3, given which optional symbols the
// loaded module provides. Returns the chosen kernel; the grid tile is 64 rows
// for PREFILL64 and PREFILL64_WMMA and 32 rows otherwise. AUTO falls back to
// the 32-row kernel when tile64 is absent; explicit mode 3 then has none.
enum euhedral_q3_prefill_kernel {
    EUHEDRAL_Q3_PREFILL_NONE = 0,
    EUHEDRAL_Q3_PREFILL32,
    EUHEDRAL_Q3_PREFILL64,
    EUHEDRAL_Q3_PREFILL64_WMMA,
};
static inline enum euhedral_q3_prefill_kernel euhedral_q3_select_prefill(int mode, uint32_t rows,
        uint32_t in_features, uint32_t out_features, int has_prefill64, int has_prefill64_wmma) {
    if (!euhedral_q3_wide_prefill(mode, rows, in_features, out_features))
        return mode == 2 ? EUHEDRAL_Q3_PREFILL32 : EUHEDRAL_Q3_PREFILL_NONE;
    if (!has_prefill64) return mode == 2 ? EUHEDRAL_Q3_PREFILL32 : EUHEDRAL_Q3_PREFILL_NONE;
    return has_prefill64_wmma && euhedral_q3_wmma_prefill64(mode, rows, in_features, out_features)
            ? EUHEDRAL_Q3_PREFILL64_WMMA : EUHEDRAL_Q3_PREFILL64;
}

#endif
