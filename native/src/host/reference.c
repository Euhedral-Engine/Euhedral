#include "reference.h"
#include "cuda_kernel_loader.h"
#include <cuda_runtime_api.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* Launches of the scalar reference kernels. The module compiles on first use, so a process that never
 * selects exact numerics and never meets an unqualified shape never pays for it. */
#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static CUmodule module;
static CUfunction q3_reference, q45_reference, nvfp4_reference[2];
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

static void initialize(void) {
    init_status = euhedral_cuda_load_kernel((const void*)&once, "reference/kernels.cu", "euhedral_q3_reference",
            &module, &q3_reference);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    if (cuModuleGetFunction(&q45_reference, module, "euhedral_q45_reference") != CUDA_SUCCESS
            || cuModuleGetFunction(&nvfp4_reference[0], module, "euhedral_nvfp4_reference") != CUDA_SUCCESS
            || cuModuleGetFunction(&nvfp4_reference[1], module, "euhedral_nvfp4_reference_sd4") != CUDA_SUCCESS)
        init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
}

#ifdef _WIN32
static BOOL CALLBACK initialize_once(PINIT_ONCE state, PVOID parameter, PVOID* context) {
    (void)state; (void)parameter; (void)context;
    initialize();
    return TRUE;
}
#endif

static int ensure(void) {
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
#ifdef _WIN32
    if (!InitOnceExecuteOnce(&once, initialize_once, NULL, NULL)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    if (pthread_once(&once, initialize) != 0) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#endif
    return init_status;
}

static int finish(CUresult status) {
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

int euhedral_reference_q3(const void* input, const void* weights, void* output, uint32_t rows,
        uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size) {
    if (input == NULL || weights == NULL || output == NULL || rows == 0 || in_features == 0 || out_features == 0
            || weights_byte_size == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t k_pad = ((uint64_t)in_features + 127u) / 128u * 128u, groups = k_pad / 64u;
    const uint64_t scale_offset = (((uint64_t)out_features * groups * 24u) + 255u) & ~255ull;
    if (scale_offset + (uint64_t)out_features * groups * 2u != weights_byte_size) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    const uint64_t grid = (uint64_t)rows * out_features;
    if (grid > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = ensure();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int rows_arg = rows, in_arg = in_features, out_arg = out_features;
    unsigned long long scale_arg = scale_offset;
    void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &rows_arg, &in_arg, &out_arg, &scale_arg};
    return finish(euhedral_launch_kernel(q3_reference, (unsigned int)grid, 1, 1, 128, 1, 1, 0,
            euhedral_cuda_submission_stream(), params, NULL));
}

int euhedral_reference_q45(const void* input, const void* weights, void* output, uint32_t rows,
        uint32_t in_features, uint32_t out_features, uint32_t bits) {
    const uint64_t count = (uint64_t)rows * out_features;
    int status = ensure();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int rows_arg = rows, in_arg = in_features, out_arg = out_features, bits_arg = bits;
    void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &rows_arg, &in_arg, &out_arg, &bits_arg};
    /* The kernel strides over outputs, so the grid is capped at the CUDA grid-x limit. */
    const unsigned int grid = count > 2147483647u ? 2147483647u : (unsigned int)count;
    return finish(euhedral_launch_kernel(q45_reference, grid, 1, 1, 128, 1, 1, 0,
            euhedral_cuda_submission_stream(), params, NULL));
}

int euhedral_reference_nvfp4(const void* input, const void* weights, void* output, uint32_t rows,
        uint32_t in_features, uint32_t out_features, int sd4) {
    const uint64_t count = (uint64_t)rows * out_features;
    int status = ensure();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int rows_arg = rows, in_arg = in_features, out_arg = out_features;
    void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &rows_arg, &in_arg, &out_arg};
    const unsigned int grid = count > 2147483647u ? 2147483647u : (unsigned int)count;
    return finish(euhedral_launch_kernel(nvfp4_reference[sd4 ? 1 : 0], grid, 1, 1, 128, 1, 1, 0,
            euhedral_cuda_submission_stream(), params, NULL));
}

/* The scalar Q3 oracle, for numerical comparisons. */
int euhedral_cuda_linear_q3_reference_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size) {
    return euhedral_reference_q3(input, weights, output, rows, in_features, out_features, weights_byte_size);
}
