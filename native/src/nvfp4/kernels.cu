#include "nvfp4.cuh"

// NVFP4 weight kernels (nvfp4.cuh). FP32 accumulation; the tile kernels stage BF16(weight).

// One row, 128 threads, 16 rows per CTA. Required symbol of the module.
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

// 2..8 token rows, each bit for bit as euhedral_nvfp4_decode on that row (speculative verification).
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

// Any rows on the balanced tile engine: 128 or 64 rows x 64 outputs per CTA.
#define EUHEDRAL_NVFP4_PREFILL(suffix, producer)                                                            \
    extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_prefill_128x64##suffix(                 \
            const unsigned short* input, const unsigned char* weights, unsigned short* output,             \
            unsigned int rows, unsigned int in_features, unsigned int out_features) {                      \
        __shared__ balanced::Storage<4> stage;                                                              \
        balanced::run<producer>(input, weights, output, rows, in_features, out_features, 0ull, stage);      \
    }                                                                                                       \
    extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_prefill_64x64##suffix(                  \
            const unsigned short* input, const unsigned char* weights, unsigned short* output,             \
            unsigned int rows, unsigned int in_features, unsigned int out_features) {                      \
        __shared__ balanced::Storage<2> stage;                                                              \
        balanced::run<producer, 2>(input, weights, output, rows, in_features, out_features, 0ull, stage);   \
    }                                                                                                       \
    /* Gate/up projection fused with SwiGLU: `outputs` weight rows, gate rows first; writes outputs / 2. */ \
    extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_gate_up_swiglu_128x32##suffix(          \
            const unsigned short* x, const unsigned char* w, unsigned short* y, unsigned int m, unsigned int k, \
            unsigned int n) {                                                                               \
        __shared__ balanced::Storage<4> s;                                                                  \
        balanced::run_paired<producer>(x, w, y, m, k, n, 0ull, s, 0, n / 2u);                               \
    }                                                                                                       \
    extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_gate_up_swiglu_64x32##suffix(           \
            const unsigned short* x, const unsigned char* w, unsigned short* y, unsigned int m, unsigned int k, \
            unsigned int n) {                                                                               \
        __shared__ balanced::Storage<2> s;                                                                  \
        balanced::run_paired<producer, 2>(x, w, y, m, k, n, 0ull, s, 0, n / 2u);                            \
    }
EUHEDRAL_NVFP4_PREFILL(, nvfp4::B)
EUHEDRAL_NVFP4_PREFILL(_sd4, nvfp4::BT<true>)
