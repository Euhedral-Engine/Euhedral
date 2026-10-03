#include "contiguous.cuh"
#include "common/pdl.cuh"

// Q4/Q5 decode kernels for the GDN input projections and the attention projections that carry more than
// three bits. Prefill runs on block-scaled FP8 tensor cores (q3_mx/kernels.cu); the scalar numerical
// reference lives in reference/kernels.cu.

#define EUHEDRAL_Q45_DECODE_CONTIGUOUS_KERNEL(B) \
extern "C" __global__ __launch_bounds__(128) void euhedral_q##B##_decode_contiguous( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features) { \
    q45::contiguous_decode<B>(input, weights, output, in_features, out_features); \
}
EUHEDRAL_Q45_DECODE_CONTIGUOUS_KERNEL(4)
EUHEDRAL_Q45_DECODE_CONTIGUOUS_KERNEL(5)
#undef EUHEDRAL_Q45_DECODE_CONTIGUOUS_KERNEL

// Multi-row twins of the contiguous kernels for 2 to 8 rows: row t is bit for bit the one-row kernel's
// output for that row. Same grid (out_features / 8) and per-row requirements.
#define EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS(B, M) \
extern "C" __global__ __launch_bounds__(128) void euhedral_q##B##_decode_contiguous_rows##M( \
        const unsigned short* input, const unsigned char* weights, unsigned short* output, \
        unsigned int rows, unsigned int in_features, unsigned int out_features) { \
    q45::contiguous_decode_rows<B, M>(input, weights, output, in_features, out_features); \
}
#define EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS_ALL(B) \
    EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS(B, 2) EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS(B, 3) \
    EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS(B, 4) EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS(B, 5) \
    EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS(B, 6) EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS(B, 7) \
    EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS(B, 8)
EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS_ALL(4)
EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS_ALL(5)
#undef EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS_ALL
#undef EUHEDRAL_Q45_DECODE_CONTIGUOUS_ROWS
