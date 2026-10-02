#include "cuda_kernel_loader.h"
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* NVFP4 weights (native/src/nvfp4/nvfp4.cuh): one-row decode, the balanced tile engine for every
 * other row count, and the paired gate/up + SwiGLU region. */
#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static CUmodule module;
static CUfunction decode, prefill128, prefill64, gate_up128, gate_up64;
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

static CUfunction optional_kernel(const char* name) {
    CUfunction loaded = NULL;
    return cuModuleGetFunction(&loaded, module, name) == CUDA_SUCCESS ? loaded : NULL;
}

static void initialize(void) {
    init_status = euhedral_cuda_load_kernel((const void*)&once, "nvfp4/kernels.cu", "euhedral_nvfp4_decode", &module, &decode);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    prefill128 = optional_kernel("euhedral_nvfp4_prefill_128x64");
    prefill64 = optional_kernel("euhedral_nvfp4_prefill_64x64");
    gate_up128 = optional_kernel("euhedral_nvfp4_gate_up_swiglu_128x32");
    gate_up64 = optional_kernel("euhedral_nvfp4_gate_up_swiglu_64x32");
    euhedral_cuda_pdl_register(decode);
}

#ifdef _WIN32
static BOOL CALLBACK initialize_once(PINIT_ONCE state, PVOID parameter, PVOID* context) {
    (void)state; (void)parameter; (void)context;
    initialize();
    return TRUE;
}
#endif

static int ensure_initialized(void) {
#ifdef _WIN32
    if (!InitOnceExecuteOnce(&once, initialize_once, NULL, NULL)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    if (pthread_once(&once, initialize) != 0) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#endif
    return init_status;
}

static uint64_t align256(uint64_t value) { return (value + 255u) & ~255ull; }

/* Byte size of an NVFP4 tensor of `rows` rows of `in_features` values (Nvfp4Layout.byteSize). */
static uint64_t nvfp4_size(uint32_t rows, uint32_t in_features) {
    uint64_t k = ((uint64_t)in_features + 127u) / 128u * 128u;
    uint64_t scales = align256((uint64_t)rows * k / 2u);
    return align256(scales + (uint64_t)rows * k / 16u) + 4u;
}

static int finish(CUresult status) {
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

static int prepare(const void* input, const void* weights, const void* output, uint32_t rows, uint32_t in_features,
        uint32_t weight_rows, uint64_t weights_byte_size) {
    if (input == NULL || weights == NULL || output == NULL || rows == 0 || in_features == 0 || weight_rows == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    if (weights_byte_size != nvfp4_size(weight_rows, in_features)) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    if (in_features % 32u != 0 || ((uintptr_t)input & 15u) != 0 || ((uintptr_t)weights & 15u) != 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    return ensure_initialized();
}

int euhedral_cuda_linear_nvfp4_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size) {
    int status = prepare(input, weights, output, rows, in_features, out_features, weights_byte_size);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int rows_arg = rows, in_arg = in_features, out_arg = out_features;
    if (rows == 1u && in_features % 1024u == 0 && out_features % 16u == 0) {
        void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &in_arg, &out_arg};
        return finish(euhedral_launch_kernel(decode, out_features / 16u, 1, 1, 128, 1, 1, 0,
                euhedral_cuda_submission_stream(), params, NULL));
    }
    CUfunction kernel = rows >= 128u && prefill128 != NULL ? prefill128 : prefill64;
    if (kernel == NULL) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    uint64_t tile = kernel == prefill128 ? 128u : 64u;
    uint64_t grid = ((uint64_t)rows + tile - 1u) / tile * (((uint64_t)out_features + 63u) / 64u);
    if (grid > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &rows_arg, &in_arg, &out_arg};
    return finish(euhedral_launch_kernel(kernel, (unsigned int)grid, 1, 1, 128, 1, 1, 0,
            euhedral_cuda_submission_stream(), params, NULL));
}

int euhedral_cuda_nvfp4_gate_up_swiglu_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t width, uint32_t outputs, uint64_t weight_bytes) {
    int status = prepare(input, weights, output, rows, width, outputs, weight_bytes);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    if (outputs % 64u != 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    CUfunction kernel = rows >= 128u && gate_up128 != NULL ? gate_up128 : gate_up64;
    if (kernel == NULL) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    uint64_t tile = kernel == gate_up128 ? 128u : 64u;
    uint64_t grid = ((uint64_t)rows + tile - 1u) / tile * (outputs / 2u / 32u);
    if (grid > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int rows_arg = rows, width_arg = width, outputs_arg = outputs;
    void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &rows_arg, &width_arg, &outputs_arg};
    return finish(euhedral_launch_kernel(kernel, (unsigned int)grid, 1, 1, 128, 1, 1, 0,
            euhedral_cuda_submission_stream(), params, NULL));
}

/* Native Blackwell NVFP4 (native/src/nvfp4_native, docs/NVFP4_NATIVE.md): the BF16 activations are
 * quantized to NVFP4 into caller scratch, then multiplied with block-scaled FP4 tensor-core MMA
 * (OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X). The module is compiled for this device's sm_12xa target and
 * exists only there; EUHEDRAL_NVFP4_NATIVE=0 selects the BF16-expansion kernels above instead. */
#define NATIVE_SHARED_BYTES 67584u  /* nvfp4n::Pipeline<1 or 2>::kSharedBytes */
#ifdef _WIN32
static INIT_ONCE native_once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t native_once = PTHREAD_ONCE_INIT;
#endif
static CUmodule native_module;
static CUfunction native_quantize, native_linear, native_gate_up;
static int native_status = EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
/* Activation terms: 2 (the default) adds a quantized residual, about 1% error per linear, at the
 * relaxed-numerics floor of the drift harness; 1 (EUHEDRAL_NVFP4_NATIVE=1) quantizes once, about 10%,
 * doubling teacher-forced KL, for about 45% more prefill throughput. docs/NVFP4_NATIVE.md. */
static unsigned int native_terms = 2u;

static void initialize_native(void) {
    const char* selected = getenv("EUHEDRAL_NVFP4_NATIVE");
    if (selected != NULL && strcmp(selected, "0") == 0) return;
    native_terms = selected != NULL && strcmp(selected, "1") == 0 ? 1u : 2u;
    const int two = native_terms == 2u;
    int status = euhedral_cuda_load_native_kernel((const void*)&native_once, "nvfp4_native/kernels.cu",
            two ? "euhedral_nvfp4n_linear_x2_128x128" : "euhedral_nvfp4n_linear_128x128", &native_module, &native_linear);
    if (status == EUHEDRAL_CUDA_SUCCESS
            && (cuModuleGetFunction(&native_quantize, native_module,
                        two ? "euhedral_nvfp4n_quantize_rows_x2" : "euhedral_nvfp4n_quantize_rows") != CUDA_SUCCESS
                    || cuModuleGetFunction(&native_gate_up, native_module,
                               two ? "euhedral_nvfp4n_gate_up_swiglu_x2_128x64" : "euhedral_nvfp4n_gate_up_swiglu_128x64")
                            != CUDA_SUCCESS))
        status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    if (status == EUHEDRAL_CUDA_SUCCESS
            && (cuFuncSetAttribute(native_linear, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)NATIVE_SHARED_BYTES)
                            != CUDA_SUCCESS
                    || cuFuncSetAttribute(native_gate_up, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               (int)NATIVE_SHARED_BYTES) != CUDA_SUCCESS))
        status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    native_status = status;
}

#ifdef _WIN32
static BOOL CALLBACK initialize_native_once(PINIT_ONCE state, PVOID parameter, PVOID* context) {
    (void)state; (void)parameter; (void)context;
    initialize_native();
    return TRUE;
}
#endif

static int ensure_native(void) {
    // The capability check and module load need this thread's device context.
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
#ifdef _WIN32
    if (!InitOnceExecuteOnce(&native_once, initialize_native_once, NULL, NULL)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    if (pthread_once(&native_once, initialize_native) != 0) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#endif
    return native_status;
}

int euhedral_cuda_nvfp4_native_available(void) {
    return ensure_native() == EUHEDRAL_CUDA_SUCCESS;
}

/* Scratch for `rows` activation rows of `in_features` values in the selected number of terms: codes, a
 * 256-aligned scale plane and one FP32 global per row (nvfp4n::ActivationLayout). */
uint64_t euhedral_cuda_nvfp4_activation_bytes(uint32_t rows, uint32_t in_features) {
    uint64_t k = ((uint64_t)in_features + 127u) / 128u * 128u, planes = (uint64_t)rows * native_terms;
    uint64_t scales = align256(planes * k / 2u);
    return align256(scales + planes * k / 16u) + 4ull * rows;
}

static int launch_native(CUfunction kernel, uint64_t column_tile, const void* input, const void* weights, void* output,
        void* scratch, uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size,
        uint64_t scratch_byte_size);

int euhedral_cuda_linear_nvfp4_native_bf16(const void* input, const void* weights, void* output, void* scratch,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size,
        uint64_t scratch_byte_size) {
    int status = ensure_native();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    return launch_native(native_linear, out_features, input, weights, output, scratch, rows, in_features, out_features,
            weights_byte_size, scratch_byte_size);
}

int euhedral_cuda_nvfp4_native_gate_up_swiglu_bf16(const void* input, const void* weights, void* output, void* scratch,
        uint32_t rows, uint32_t width, uint32_t outputs, uint64_t weight_bytes, uint64_t scratch_byte_size) {
    int status = ensure_native();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    if (outputs % 2u != 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    return launch_native(native_gate_up, outputs, input, weights, output, scratch, rows, width, outputs, weight_bytes,
            scratch_byte_size);
}

/* `columns` B-tile rows to cover: out_features for a linear (128 per tile) and all gate + up rows for
 * the paired region (64 outputs, 128 weight rows, per tile). */
static int launch_native(CUfunction kernel, uint64_t columns, const void* input, const void* weights, void* output,
        void* scratch, uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size,
        uint64_t scratch_byte_size) {
    int status;
    // Activations are quantized, so exact numerics keep the BF16-expansion kernels (the reference twin).
    if (in_features % 128u != 0 || euhedral_cuda_exact_numerics()) return EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
    status = prepare(input, weights, output, rows, in_features, out_features, weights_byte_size);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    if (scratch == NULL || scratch_byte_size < euhedral_cuda_nvfp4_activation_bytes(rows, in_features))
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    uint64_t grid = ((uint64_t)rows + 127u) / 128u * ((columns + 127u) / 128u);
    if (grid > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    CUstream stream = euhedral_cuda_submission_stream();
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output, scratch_ptr = (CUdeviceptr)(uintptr_t)scratch;
    unsigned int rows_arg = rows, in_arg = in_features, out_arg = out_features;
    void* quantize_params[] = {&input_ptr, &scratch_ptr, &rows_arg, &in_arg};
    CUresult result = euhedral_launch_kernel(native_quantize, rows, 1, 1, 128, 1, 1, 0, stream, quantize_params, NULL);
    if (result != CUDA_SUCCESS) return finish(result);
    void* linear_params[] = {&scratch_ptr, &weights_ptr, &output_ptr, &rows_arg, &in_arg, &out_arg};
    return finish(euhedral_launch_kernel(kernel, (unsigned int)grid, 1, 1, 256, 1, 1, NATIVE_SHARED_BYTES,
            stream, linear_params, NULL));
}
