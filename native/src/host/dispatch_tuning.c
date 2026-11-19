#include "dispatch_tuning.h"
#include "euhedral_cuda.h"
#include <stdatomic.h>
#include <stddef.h>

#define FUSED_ROWS 9u

/* Measured on an RTX 5070 Ti: the output pass of a one-row norm over 40 CTAs. */
static _Atomic uint32_t rms_norm_slices = 40u;
/* Fewest keys at which the fused verifier kernel is faster than the row twins, by verified rows (6 query heads per KV
 * head, RTX 5070 Ti): rows that fill their 8-column mma tiles win from 2048 keys, the others only on longer caches. */
static _Atomic uint32_t fused_from[FUSED_ROWS] = {UINT32_MAX, UINT32_MAX, 49152u, 16384u, 4096u, 2048u, 16384u,
        16384u, 12288u};

int euhedral_cuda_set_dispatch(uint32_t rms_norm_decode_slices, const uint32_t* fused_verify_from,
        uint32_t fused_verify_from_count) {
    if (rms_norm_decode_slices == 0u || fused_verify_from == NULL || fused_verify_from_count != FUSED_ROWS)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    for (uint32_t row = 0; row < FUSED_ROWS; row++)
        if (fused_verify_from[row] == 0u) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    atomic_store_explicit(&rms_norm_slices, rms_norm_decode_slices, memory_order_relaxed);
    for (uint32_t row = 0; row < FUSED_ROWS; row++)
        atomic_store_explicit(&fused_from[row], fused_verify_from[row], memory_order_relaxed);
    return EUHEDRAL_CUDA_SUCCESS;
}

uint32_t euhedral_dispatch_rms_norm_slices(void) {
    return atomic_load_explicit(&rms_norm_slices, memory_order_relaxed);
}

uint32_t euhedral_dispatch_fused_from(uint32_t rows) {
    return rows < FUSED_ROWS ? atomic_load_explicit(&fused_from[rows], memory_order_relaxed) : UINT32_MAX;
}
