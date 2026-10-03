#include "native.cuh"

// Activation quantization: one CTA per row; two NVFP4 planes, the quantized value and its quantized
// residual (nvfp4n::ActivationLayout, docs/NVFP4_NATIVE.md).
extern "C" __global__ void __launch_bounds__(nvfp4n::kQuantizeThreads) euhedral_nvfp4n_quantize_rows(
        const __nv_bfloat16* input, unsigned char* output, unsigned int rows, unsigned int k) {
    nvfp4n::quantize_rows<2>(input, output, rows, k);
}

// Tile linear: gridDim.x = tiles_m * tiles_n (M fastest), 288 threads (eight consumer warps and a producer warp),
// nvfp4n::Pipeline<2>::kSharedBytes of dynamic shared memory; the three tensor maps describe the activation code
// planes, the weight codes and the weight scales (docs/NVFP4_NATIVE.md).
#define EUHEDRAL_NVFP4N_LINEAR(name, terms, paired, sd4)                                                    \
    extern "C" __global__ void __launch_bounds__(nvfp4n::kThreads, 1) name(                                 \
            const __grid_constant__ nvfp4n::TensorMap tm_a, const __grid_constant__ nvfp4n::TensorMap tm_b, \
            const __grid_constant__ nvfp4n::TensorMap tm_bs,                                                \
            const unsigned char* activations, const unsigned char* weights, __nv_bfloat16* output,          \
            unsigned int rows, unsigned int k, unsigned int cols) {                                        \
        nvfp4n::linear<terms, paired, sd4>(&tm_a, &tm_b, &tm_bs, activations, weights, output, rows, k, cols); \
    }
// Linear: `cols` output features (128 per tile). Paired gate/up: `cols` weight rows (gate, then up), cols / 2
// SwiGLU outputs (64 per tile). The _sd4 kernels take row-split-k128-sd4-v1 weights.
EUHEDRAL_NVFP4N_LINEAR(euhedral_nvfp4n_linear_128x128, 2, false, false)
EUHEDRAL_NVFP4N_LINEAR(euhedral_nvfp4n_gate_up_swiglu_128x64, 2, true, false)
EUHEDRAL_NVFP4N_LINEAR(euhedral_nvfp4n_linear_128x128_sd4, 2, false, true)
EUHEDRAL_NVFP4N_LINEAR(euhedral_nvfp4n_gate_up_swiglu_128x64_sd4, 2, true, true)

// Skinny (decode-like) linear: gridDim.x = ceil(cols / 64), gridDim.y = K splits.
#define EUHEDRAL_NVFP4N_SKINNY(name, terms, mf, sd4)                                                        \
    extern "C" __global__ void __launch_bounds__(nvfp4n::kSkinnyThreads) name(                              \
            const unsigned char* activations, const unsigned char* weights, __nv_bfloat16* output,          \
            float* partials, unsigned int rows, unsigned int k, unsigned int cols) {                       \
        nvfp4n::skinny_linear<terms, mf, sd4>(activations, weights, output, partials, rows, k, cols);       \
    }
EUHEDRAL_NVFP4N_SKINNY(euhedral_nvfp4n_skinny_16, 2, 1, false)
EUHEDRAL_NVFP4N_SKINNY(euhedral_nvfp4n_skinny_32, 2, 2, false)
EUHEDRAL_NVFP4N_SKINNY(euhedral_nvfp4n_skinny_64, 2, 4, false)
EUHEDRAL_NVFP4N_SKINNY(euhedral_nvfp4n_skinny_16_sd4, 2, 1, true)
EUHEDRAL_NVFP4N_SKINNY(euhedral_nvfp4n_skinny_32_sd4, 2, 2, true)
EUHEDRAL_NVFP4N_SKINNY(euhedral_nvfp4n_skinny_64_sd4, 2, 4, true)

// FP32 partials (splits x count) summed in split order, to BF16.
extern "C" __global__ void euhedral_nvfp4n_skinny_finish(
        const float* partials, __nv_bfloat16* output, unsigned int count, unsigned int splits) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float sum = partials[i];
    for (unsigned int split = 1; split < splits; ++split) sum += partials[(unsigned long long)split * count + i];
    output[i] = __float2bfloat16_rn(sum);
}
