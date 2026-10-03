#include "nvfp4.cuh"

// NVFP4 decode kernels (nvfp4.cuh), FP32 accumulation. Prefill runs on native FP4 tensor cores
// (nvfp4_native/kernels.cu); the scalar numerical reference lives in reference/kernels.cu.

// One row, 128 threads, 16 rows per CTA.
extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int in_features, unsigned int out_features) {
    nvfp4::decode(input, weights, output, in_features, out_features);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_decode_sd4(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int in_features, unsigned int out_features) {
    nvfp4::decode<true>(input, weights, output, in_features, out_features);
}

// 2..8 token rows, each bit for bit as euhedral_nvfp4_decode on that row.
#define EUHEDRAL_NVFP4_DECODE_ROWS(M)                                                                       \
    extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_decode_rows##M(                         \
            const unsigned short* input, const unsigned char* weights, unsigned short* output,             \
            unsigned int in_features, unsigned int out_features) {                                          \
        nvfp4::decode_rows<M>(input, weights, output, in_features, out_features);                           \
    }                                                                                                       \
    extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_decode_rows##M##_sd4(                   \
            const unsigned short* input, const unsigned char* weights, unsigned short* output,             \
            unsigned int in_features, unsigned int out_features) {                                          \
        nvfp4::decode_rows<M, true>(input, weights, output, in_features, out_features);                     \
    }
EUHEDRAL_NVFP4_DECODE_ROWS(2)
EUHEDRAL_NVFP4_DECODE_ROWS(3)
EUHEDRAL_NVFP4_DECODE_ROWS(4)
EUHEDRAL_NVFP4_DECODE_ROWS(5)
EUHEDRAL_NVFP4_DECODE_ROWS(6)
EUHEDRAL_NVFP4_DECODE_ROWS(7)
EUHEDRAL_NVFP4_DECODE_ROWS(8)
