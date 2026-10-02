#include "nvfp4.cuh"

// NVFP4 weight kernels (nvfp4.cuh). FP32 accumulation; the tile kernels stage BF16(weight).

// One row, 128 threads, 16 rows per CTA. Required symbol of the module.
extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_decode(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int in_features, unsigned int out_features) {
    nvfp4::decode(input, weights, output, in_features, out_features);
}

// Any rows on the balanced tile engine: 128 or 64 rows x 64 outputs per CTA.
extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_prefill_128x64(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features) {
    __shared__ balanced::Storage<4> stage;
    balanced::run<nvfp4::B>(input, weights, output, rows, in_features, out_features, 0ull, stage);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_prefill_64x64(
        const unsigned short* input, const unsigned char* weights, unsigned short* output,
        unsigned int rows, unsigned int in_features, unsigned int out_features) {
    __shared__ balanced::Storage<2> stage;
    balanced::run<nvfp4::B, 2>(input, weights, output, rows, in_features, out_features, 0ull, stage);
}

// Gate/up projection fused with SwiGLU: `outputs` weight rows, gate rows first; writes outputs / 2.
extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_gate_up_swiglu_128x32(
        const unsigned short* x, const unsigned char* w, unsigned short* y, unsigned int m, unsigned int k, unsigned int n) {
    __shared__ balanced::Storage<4> s;
    balanced::run_paired<nvfp4::B>(x, w, y, m, k, n, 0ull, s, 0, n / 2u);
}
extern "C" __global__ __launch_bounds__(128) void euhedral_nvfp4_gate_up_swiglu_64x32(
        const unsigned short* x, const unsigned char* w, unsigned short* y, unsigned int m, unsigned int k, unsigned int n) {
    __shared__ balanced::Storage<2> s;
    balanced::run_paired<nvfp4::B, 2>(x, w, y, m, k, n, 0ull, s, 0, n / 2u);
}
