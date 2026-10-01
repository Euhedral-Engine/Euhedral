#include "cuda_kernel_loader.h"
#include <cuda_runtime_api.h>
#include <math.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static CUmodule module;
static CUfunction function;
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

static void initialize(void) {
    init_status = euhedral_cuda_load_kernel((const void*)&once, "norm/kernels.cu", "euhedral_rms_norm_bf16", &module, &function);
    if (init_status == EUHEDRAL_CUDA_SUCCESS) euhedral_cuda_pdl_register(function);
}
#ifdef _WIN32
static BOOL CALLBACK initialize_once(PINIT_ONCE state, PVOID parameter, PVOID* context) {
    (void)state; (void)parameter; (void)context;
    initialize();
    return TRUE;
}
#endif

static int rms_norm_bf16_with_offset(const void* input, const void* weight, void* output,
        uint32_t rows, uint32_t width, float epsilon, float weight_offset) {
    if (input == NULL || weight == NULL || output == NULL || rows == 0 || width == 0 || !isfinite(epsilon) || epsilon < 0.0f)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int context_status = euhedral_cuda_bind_thread_context();
    if (context_status != EUHEDRAL_CUDA_SUCCESS) return context_status;
#ifdef _WIN32
    if (!InitOnceExecuteOnce(&once, initialize_once, NULL, NULL)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    if (pthread_once(&once, initialize) != 0) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#endif
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return init_status;
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weight_ptr = (CUdeviceptr)(uintptr_t)weight;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int rows_arg = rows, width_arg = width;
    void* params[] = {&input_ptr, &weight_ptr, &output_ptr, &rows_arg, &width_arg, &epsilon, &weight_offset};
    // A single row (decode) is otherwise one CTA on one SM; its output pass spreads over 40.
    unsigned int slices = rows == 1 ? 40u : 1u;
    CUresult status = euhedral_launch_kernel(function, rows, slices, 1, 128, 1, 1, 0, euhedral_cuda_submission_stream(), params, NULL);
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

int euhedral_cuda_rms_norm_bf16(const void* input, const void* weight, void* output,
        uint32_t rows, uint32_t width, float epsilon) {
    return rms_norm_bf16_with_offset(input, weight, output, rows, width, epsilon, 0.0f);
}

int euhedral_cuda_rms_norm_unit_offset_bf16(const void* input, const void* weight, void* output,
        uint32_t rows, uint32_t width, float epsilon) {
    return rms_norm_bf16_with_offset(input, weight, output, rows, width, epsilon, 1.0f);
}
