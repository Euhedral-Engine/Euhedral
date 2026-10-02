#include "cuda_kernel_loader.h"
#include <cuda_runtime_api.h>
#include <stdint.h>
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
