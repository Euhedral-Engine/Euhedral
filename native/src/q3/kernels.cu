#include "contiguous.cuh"
#include "p2e2.cuh"
#include "common/pdl.cuh"

// Q3G64_F16S decode kernels (docs/COMPACT_Q3_REFERENCE.md). Prefill runs on block-scaled FP8 tensor cores
// (q3_mx/kernels.cu); the scalar numerical reference lives in reference/kernels.cu.

// Single-row decode with contiguous lane ownership (contiguous.cuh): each lane owns 32 whole codes.
// 128 threads, 16 rows per CTA; in_features a multiple of 1024, out_features a multiple of 16, a 16-byte
// aligned input and a 4-byte aligned weight base.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_decode_contiguous(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) {
    q3::contiguous_decode(input, weights, output, in_features, out_features, scale_offset);
}

// Multi-row twins of euhedral_q3_decode_contiguous for 2 to 8 rows: row t is bit for bit the one-row
// kernel's output for that row, so speculative verification and short batches see one numerical result
// however the rows are grouped. Same grid (out_features / 16) and per-row requirements.
#define EUHEDRAL_Q3_DECODE_CONTIGUOUS_ROWS(M) \
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_decode_contiguous_rows##M( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features, unsigned long long scale_offset) { \
    q3::contiguous_decode_rows<M>(input, weights, output, in_features, out_features, scale_offset); \
}
EUHEDRAL_Q3_DECODE_CONTIGUOUS_ROWS(2)
EUHEDRAL_Q3_DECODE_CONTIGUOUS_ROWS(3)
EUHEDRAL_Q3_DECODE_CONTIGUOUS_ROWS(4)
EUHEDRAL_Q3_DECODE_CONTIGUOUS_ROWS(5)
EUHEDRAL_Q3_DECODE_CONTIGUOUS_ROWS(6)
EUHEDRAL_Q3_DECODE_CONTIGUOUS_ROWS(7)
EUHEDRAL_Q3_DECODE_CONTIGUOUS_ROWS(8)
#undef EUHEDRAL_Q3_DECODE_CONTIGUOUS_ROWS

// P2E2 tensors (p2e2.cuh, docs/COMPRESSED_Q3.md).
// Single-row decode from the compressed layout, bitwise identical to euhedral_q3_decode_contiguous on
// the row-split tensor it encodes. 128 threads, 16 rows per CTA; same shape requirements.
extern "C" __global__ __launch_bounds__(128, 5) void euhedral_q3_p2e2_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int in_features, unsigned int out_features) {
    q3::p2e2::p2e2_decode<1>(input, weights, output, in_features, out_features);
}
// Two to four rows (speculative verification and drafting): row t is bit for bit the one-row kernel's output
// for it, as the contiguous rows kernels are for the row-split tensor.
#define EUHEDRAL_Q3_P2E2_DECODE_ROWS(M, BLOCKS) \
extern "C" __global__ __launch_bounds__(128, BLOCKS) void euhedral_q3_p2e2_decode_rows##M( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int in_features, unsigned int out_features) { \
    q3::p2e2::p2e2_decode<M>(input, weights, output, in_features, out_features); \
}
EUHEDRAL_Q3_P2E2_DECODE_ROWS(2, 4)
EUHEDRAL_Q3_P2E2_DECODE_ROWS(3, 3)
EUHEDRAL_Q3_P2E2_DECODE_ROWS(4, 2)
#undef EUHEDRAL_Q3_P2E2_DECODE_ROWS
// Expands rows [first_row, first_row + row_count) of a P2E2 tensor of `rows` rows into the row-split
// tensor of row_count rows they encode, byte for byte. One warp per row.
extern "C" __global__ __launch_bounds__(128) void euhedral_q3_p2e2_expand(
        const unsigned char* weights, unsigned char* out, unsigned int rows, unsigned int in_features,
        unsigned int first_row, unsigned int row_count) {
    q3::p2e2::p2e2_expand(weights, out, rows, in_features, first_row, row_count);
}
