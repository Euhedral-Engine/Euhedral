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

