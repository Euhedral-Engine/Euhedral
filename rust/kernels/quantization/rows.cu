// Row gathers for calibration: token embeddings and the draft head's rows of the LM head.
#include <cuda_bf16.h>

// out[i, :] = table[ids[i], :] for BF16 rows of `width` elements (width * 2 a multiple of 16).
extern "C" __global__ void gather_rows(const uint4* table, const int* ids, long long count, int width, uint4* out) {
    const int vectors = width / 8;
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count * vectors) return;
    long long row = i / vectors;
    int column = (int)(i % vectors);
    out[i] = table[(long long)ids[row] * vectors + column];
}
