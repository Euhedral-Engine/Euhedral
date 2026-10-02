#ifndef EUHEDRAL_Q3_P2E2_GEOMETRY_H
#define EUHEDRAL_Q3_P2E2_GEOMETRY_H

#include "euhedral_cuda.h"
#include <stdint.h>

/* P2E2 tensor geometry (native/src/q3/p2e2.cuh, docs/COMPRESSED_Q3.md). The payload plane's length
 * depends on the codes, so a byte size is valid when it lies between an empty and a full payload. */
#define EUHEDRAL_Q3_P2E2_SLICE 1024u
#define EUHEDRAL_Q3_P2E2_PAD_WORDS 80u

static inline uint64_t euhedral_q3_p2e2_align256(uint64_t value) {
    return (value + 255u) & ~255ull;
}

/* Byte offset of the payload plane, or 0 when the shape overflows 64 bits. */
static inline uint64_t euhedral_q3_p2e2_payload_offset(uint32_t rows, uint32_t in_features) {
    uint64_t codes = (uint64_t)rows * in_features;
    uint64_t base = euhedral_q3_p2e2_align256(codes / 4u);
    uint64_t scale = euhedral_q3_p2e2_align256(base + 4ull * rows);
    return euhedral_q3_p2e2_align256(scale + codes / 32u);
}

static inline int euhedral_q3_p2e2_geometry(uint32_t rows, uint32_t in_features, uint64_t byte_size) {
    if (rows == 0 || in_features == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    if (in_features % EUHEDRAL_Q3_P2E2_SLICE != 0) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    uint64_t payload = euhedral_q3_p2e2_payload_offset(rows, in_features);
    uint64_t pad = 4ull * EUHEDRAL_Q3_P2E2_PAD_WORDS;
    uint64_t full = payload + 4ull * (((uint64_t)rows * in_features + 15u) / 16u) + pad;
    if (byte_size < payload + pad || byte_size > full || (byte_size - payload) % 4u != 0) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    return EUHEDRAL_CUDA_SUCCESS;
}

/* Size of the row-split Q3 tensor a P2E2 tensor expands to. */
static inline uint64_t euhedral_q3_row_split_size(uint32_t rows, uint32_t in_features) {
    uint64_t groups = ((uint64_t)in_features + 127u) / 128u * 2u;
    return euhedral_q3_p2e2_align256((uint64_t)rows * groups * 24u) + (uint64_t)rows * groups * 2u;
}

#endif
