#include "native.cuh"

// Activation quantization: one CTA per row; `terms` NVFP4 planes (nvfp4n::ActivationLayout).
extern "C" __global__ void __launch_bounds__(nvfp4n::kQuantizeThreads) euhedral_nvfp4n_quantize_rows(
        const __nv_bfloat16* input, unsigned char* output, unsigned int rows, unsigned int k) {
    nvfp4n::quantize_rows<1>(input, output, rows, k);
}
extern "C" __global__ void __launch_bounds__(nvfp4n::kQuantizeThreads) euhedral_nvfp4n_quantize_rows_x2(
        const __nv_bfloat16* input, unsigned char* output, unsigned int rows, unsigned int k) {
    nvfp4n::quantize_rows<2>(input, output, rows, k);
}

#define EUHEDRAL_NVFP4N_LINEAR(name, terms, paired)                                                         \
    extern "C" __global__ void __launch_bounds__(nvfp4n::kThreads) name(                                    \
            const unsigned char* activations, const unsigned char* weights, __nv_bfloat16* output,          \
            unsigned int rows, unsigned int k, unsigned int cols) {                                        \
        nvfp4n::linear<terms, paired>(activations, weights, output, rows, k, cols);                         \
    }
// Linear: `cols` output features. Paired gate/up: `cols` weight rows (gate, then up), cols / 2 SwiGLU outputs.
EUHEDRAL_NVFP4N_LINEAR(euhedral_nvfp4n_linear_128x128, 1, false)
EUHEDRAL_NVFP4N_LINEAR(euhedral_nvfp4n_gate_up_swiglu_128x64, 1, true)
EUHEDRAL_NVFP4N_LINEAR(euhedral_nvfp4n_linear_x2_128x128, 2, false)
EUHEDRAL_NVFP4N_LINEAR(euhedral_nvfp4n_gate_up_swiglu_x2_128x64, 2, true)

// Skinny (decode-like) linear: gridDim.x = ceil(cols / 64), gridDim.y = K splits.
#define EUHEDRAL_NVFP4N_SKINNY(name, terms, mf)                                                             \
    extern "C" __global__ void __launch_bounds__(nvfp4n::kSkinnyThreads) name(                              \
            const unsigned char* activations, const unsigned char* weights, __nv_bfloat16* output,          \
            float* partials, unsigned int rows, unsigned int k, unsigned int cols) {                       \
        nvfp4n::skinny_linear<terms, mf>(activations, weights, output, partials, rows, k, cols);            \
    }
EUHEDRAL_NVFP4N_SKINNY(euhedral_nvfp4n_skinny_16, 1, 1)
EUHEDRAL_NVFP4N_SKINNY(euhedral_nvfp4n_skinny_32, 1, 2)
EUHEDRAL_NVFP4N_SKINNY(euhedral_nvfp4n_skinny_64, 1, 4)
EUHEDRAL_NVFP4N_SKINNY(euhedral_nvfp4n_skinny_x2_16, 2, 1)
EUHEDRAL_NVFP4N_SKINNY(euhedral_nvfp4n_skinny_x2_32, 2, 2)
EUHEDRAL_NVFP4N_SKINNY(euhedral_nvfp4n_skinny_x2_64, 2, 4)

// FP32 partials (splits x count) summed in split order, to BF16.
extern "C" __global__ void euhedral_nvfp4n_skinny_finish(
        const float* partials, __nv_bfloat16* output, unsigned int count, unsigned int splits) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= count) return;
    float sum = partials[i];
    for (unsigned int split = 1; split < splits; ++split) sum += partials[(unsigned long long)split * count + i];
    output[i] = __float2bfloat16_rn(sum);
}
