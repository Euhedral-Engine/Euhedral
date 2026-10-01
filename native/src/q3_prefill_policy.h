#ifndef EUHEDRAL_Q3_PREFILL_POLICY_H
#define EUHEDRAL_Q3_PREFILL_POLICY_H

#include <stdint.h>

// Single-row decode streams weights with the warp-local wide kernel wherever its layout
// requirements hold: whole 16-byte scale vectors (K % 512 == 0), a scale row that fits its
// 256-word shared slot (K <= 32768) and whole 8-column CTAs. With weights streaming from DRAM it
// measured 25-29% faster than the cooperative kernel on every model shape (5120 -> 34816,
// 17408 -> 5120, 6144 -> 5120, and the LM head).
static inline int euhedral_q3_decode_wide_shape(uint32_t rows, uint32_t in_features, uint32_t out_features) {
    return rows == 1 && in_features != 0 && in_features % 512u == 0 && in_features <= 32768u
            && out_features != 0 && out_features % 8u == 0;
}

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
// for PREFILL64 variants and 32 rows otherwise. AUTO falls back to the 32-row
// kernel when tile64 is absent; explicit mode 3 then has none.
enum euhedral_q3_prefill_kernel {
    EUHEDRAL_Q3_PREFILL_NONE = 0,
    EUHEDRAL_Q3_PREFILL32,
    EUHEDRAL_Q3_PREFILL64,
    EUHEDRAL_Q3_PREFILL64_WMMA,
    EUHEDRAL_Q3_PREFILL64_K32_CB,
    EUHEDRAL_Q3_PREFILL32_S104,
};
static inline enum euhedral_q3_prefill_kernel euhedral_q3_select_prefill(int mode, uint32_t rows,
        uint32_t in_features, uint32_t out_features, int has_prefill64, int has_prefill64_wmma) {
    if (!euhedral_q3_wide_prefill(mode, rows, in_features, out_features))
        return mode == 2 ? EUHEDRAL_Q3_PREFILL32 : EUHEDRAL_Q3_PREFILL_NONE;
    if (!has_prefill64) return mode == 2 ? EUHEDRAL_Q3_PREFILL32 : EUHEDRAL_Q3_PREFILL_NONE;
    return has_prefill64_wmma && euhedral_q3_wmma_prefill64(mode, rows, in_features, out_features)
            ? EUHEDRAL_Q3_PREFILL64_WMMA : EUHEDRAL_Q3_PREFILL64;
}

#define EUHEDRAL_Q3_HAS_K32_COMPACT_B_POLICY 1
static inline int euhedral_q3_use_k32_compact_b_prefill(int mode, uint32_t rows,
        uint32_t in_features, uint32_t out_features) {
    if (mode != 2) return 0;
    if (in_features == 6144 && out_features == 5120)
        return rows == 256 || rows == 512 || rows == 1024;
    if ((in_features == 5120 && out_features == 34816)
            || (in_features == 17408 && out_features == 5120))
        return rows == 64 || rows == 256 || rows == 1024;
    return 0;
}

static inline enum euhedral_q3_prefill_kernel euhedral_q3_select_prefill_with_k32_compact_b(
        int mode, uint32_t rows, uint32_t in_features, uint32_t out_features,
        int has_prefill64_k32_cb, int has_prefill64, int has_prefill64_wmma) {
    if (has_prefill64_k32_cb
            && euhedral_q3_use_k32_compact_b_prefill(mode, rows, in_features, out_features))
        return EUHEDRAL_Q3_PREFILL64_K32_CB;
    return euhedral_q3_select_prefill(mode, rows, in_features, out_features,
            has_prefill64, has_prefill64_wmma);
}

// The 104-element-stride 32-row kernel wins on the GDN output mixer only where the grid has one or
// three 32-row tiles; everywhere else the 88-element kernel is at least as fast.
static inline int euhedral_q3_use_s104_prefill(int mode, uint32_t rows,
        uint32_t in_features, uint32_t out_features) {
    return mode == 2 && in_features == 6144 && out_features == 5120
            && (rows <= 32 || (rows >= 65 && rows <= 96));
}

#define EUHEDRAL_Q3_HAS_CB_ALIGNMENT_POLICY 1
static inline enum euhedral_q3_prefill_kernel euhedral_q3_select_prefill_for_input(
        int mode, uint32_t rows, uint32_t in_features, uint32_t out_features,
        int has_prefill64_k32_cb, int input_aligned_16, int has_prefill64, int has_prefill64_wmma) {
    return euhedral_q3_select_prefill_with_k32_compact_b(mode, rows, in_features, out_features,
            has_prefill64_k32_cb && input_aligned_16, has_prefill64, has_prefill64_wmma);
}

static inline enum euhedral_q3_prefill_kernel euhedral_q3_select_prefill_s104(
        int mode, uint32_t rows, uint32_t in_features, uint32_t out_features,
        int has_prefill64_k32_cb, int input_aligned_16, int has_prefill64, int has_prefill64_wmma,
        int has_s104) {
    enum euhedral_q3_prefill_kernel selected = euhedral_q3_select_prefill_for_input(mode, rows,
            in_features, out_features, has_prefill64_k32_cb, input_aligned_16, has_prefill64,
            has_prefill64_wmma);
    if (selected == EUHEDRAL_Q3_PREFILL32 && has_s104
            && euhedral_q3_use_s104_prefill(mode, rows, in_features, out_features))
        return EUHEDRAL_Q3_PREFILL32_S104;
    return selected;
}

#define EUHEDRAL_Q3_HAS_K32_PREFILL_TILE_POLICY 1
static inline uint32_t euhedral_q3_prefill_tile_rows(enum euhedral_q3_prefill_kernel selected) {
    return selected == EUHEDRAL_Q3_PREFILL64
            || selected == EUHEDRAL_Q3_PREFILL64_WMMA
            || selected == EUHEDRAL_Q3_PREFILL64_K32_CB ? 64u : 32u;
}

#endif
