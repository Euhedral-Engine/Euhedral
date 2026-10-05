#include "nvfp4.cuh"

// NVFP4 decode kernels (nvfp4.cuh), BF16 tensor cores with FP32 accumulation. Prefill runs on native FP4 tensor cores
// (nvfp4_native/kernels.cu); the scalar numerical reference lives in reference/kernels.cu.

// 1 to 8 token rows (nvfp4.cuh decode_rows), each bit for bit the one-row call on that row. Two tiles, chosen
// by shape: _w8 (16 rows per CTA, 8 warps) for in_features up to 5120, _w4 (32 rows per CTA, 4 warps) above.
#define EUHEDRAL_NVFP4_DECODE_ROWS(M)                                                                       \
    extern "C" __global__ __launch_bounds__(256) void euhedral_nvfp4_decode_rows##M##_w8(                    \
            const unsigned short* input, const unsigned char* weights, unsigned short* output,             \
            unsigned int in_features, unsigned int out_features) {                                          \
        nvfp4::decode_rows<M, false, 1, 8>(input, weights, output, in_features, out_features);              \
    }                                                                                                       \
    extern "C" __global__ __launch_bounds__(256) void euhedral_nvfp4_decode_rows##M##_w8_sd4(                \
            const unsigned short* input, const unsigned char* weights, unsigned short* output,             \
            unsigned int in_features, unsigned int out_features) {                                          \
        nvfp4::decode_rows<M, true, 1, 8>(input, weights, output, in_features, out_features);               \
    }                                                                                                       \
    extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_decode_rows##M##_w4(                    \
            const unsigned short* input, const unsigned char* weights, unsigned short* output,             \
            unsigned int in_features, unsigned int out_features) {                                          \
        nvfp4::decode_rows<M, false, 2, 4>(input, weights, output, in_features, out_features);              \
    }                                                                                                       \
    extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_decode_rows##M##_w4_sd4(                \
            const unsigned short* input, const unsigned char* weights, unsigned short* output,             \
            unsigned int in_features, unsigned int out_features) {                                          \
        nvfp4::decode_rows<M, true, 2, 4>(input, weights, output, in_features, out_features);               \
    }
EUHEDRAL_NVFP4_DECODE_ROWS(1)
EUHEDRAL_NVFP4_DECODE_ROWS(2)
EUHEDRAL_NVFP4_DECODE_ROWS(3)
EUHEDRAL_NVFP4_DECODE_ROWS(4)
EUHEDRAL_NVFP4_DECODE_ROWS(5)
EUHEDRAL_NVFP4_DECODE_ROWS(6)
EUHEDRAL_NVFP4_DECODE_ROWS(7)
EUHEDRAL_NVFP4_DECODE_ROWS(8)
