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

/* Q3 prefill on block-scaled FP8 tensor cores (native/src/q3_mx, docs/PREFILL_MX.md): the BF16 activations are
 * split into two E4M3 terms with a power-of-two block scale (exact for BF16), the Q3 codes are exact E4M3, and the
 * MMA accumulates in FP32 at the full FP8 rate, twice the BF16 rate on GeForce Blackwell. Needs an sm_12x device;
 * EUHEDRAL_Q3_MX=0 disables it, exact numerics decline it. */
#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static CUmodule module;
static CUfunction quantize, linear, gate_up;
static int init_status = EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;

/* q3mx::kSharedBytes of native/src/q3_mx/kernels.cu: three stages of (A hi, A lo, expanded W, 128 FP32 scales). */
#define Q3MX_SHARED_BYTES (3u * (3u * 128u * 64u + 128u * 4u))

static void initialize(void) {
    const char* selected = getenv("EUHEDRAL_Q3_MX");
    if (selected != NULL && strcmp(selected, "0") == 0) return;
    int status = euhedral_cuda_load_native_kernel((const void*)&once, "q3_mx/kernels.cu", "euhedral_q3mx_quantize", &module, &quantize);
    if (status == EUHEDRAL_CUDA_SUCCESS
            && (cuModuleGetFunction(&linear, module, "euhedral_q3mx_linear_128x128") != CUDA_SUCCESS
                    || cuModuleGetFunction(&gate_up, module, "euhedral_q3mx_gate_up_swiglu_128x64") != CUDA_SUCCESS
                    || cuFuncSetAttribute(linear, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)Q3MX_SHARED_BYTES) != CUDA_SUCCESS
                    || cuFuncSetAttribute(gate_up, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)Q3MX_SHARED_BYTES) != CUDA_SUCCESS))
        status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    init_status = status;
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

static uint64_t align256(uint64_t value) { return (value + 255u) & ~255ull; }

int euhedral_cuda_q3_mx_available(void) {
    return ensure() == EUHEDRAL_CUDA_SUCCESS;
}

/* The hi plane, the lo plane (rows x width bytes each) and the UE8M0 scales (rows x width / 32), 256-aligned. */
uint64_t euhedral_cuda_q3_mx_scratch_bytes(uint32_t rows, uint32_t width) {
    uint64_t plane = align256((uint64_t)rows * width);
    return 2u * plane + align256((uint64_t)rows * (width / 32u));
}

static int finish(CUresult status) {
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

/* `weight_rows` Q3 rows (Layout: 24-byte groups, FP16 scales after a 256-aligned code plane) against `rows` BF16
 * activation rows of `width` values. paired: the weight rows are gate rows then up rows and the output is the
 * SwiGLU, weight_rows / 2 values per row. */
static int run(int paired, const void* input, const void* weights, void* output, void* scratch, uint32_t rows,
        uint32_t width, uint32_t weight_rows, uint64_t weight_bytes, uint64_t scratch_bytes) {
    if (rows == 0 || width % 128u != 0 || weight_rows % 128u != 0
            || ((uintptr_t)input & 15u) != 0 || ((uintptr_t)weights & 3u) != 0 || ((uintptr_t)output & 3u) != 0)
        return EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
    /* Activations are quantized, so exact numerics and row-exact execution keep the BF16 kernels. */
    if (euhedral_cuda_exact_numerics() || euhedral_cuda_row_exact()) return EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
    int status = ensure();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    const uint64_t groups = (uint64_t)weight_rows * (width / 64u);
    const uint64_t scale_offset = align256(groups * 24u);
    if (weight_bytes != scale_offset + groups * 2u) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    if (scratch == NULL || ((uintptr_t)scratch & 255u) != 0 || scratch_bytes < euhedral_cuda_q3_mx_scratch_bytes(rows, width))
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t grid = (((uint64_t)rows + 127u) / 128u) * (weight_rows / 128u), blocks = ((uint64_t)rows * (width / 32u) + 255u) / 256u;
    if (grid > 2147483647u || blocks > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    CUstream stream = euhedral_cuda_submission_stream();
    const uint64_t plane = align256((uint64_t)rows * width);
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output, hi_ptr = (CUdeviceptr)(uintptr_t)scratch;
    CUdeviceptr lo_ptr = hi_ptr + plane, scale_ptr = hi_ptr + 2u * plane;
    unsigned int rows_arg = rows, width_arg = width, weight_rows_arg = weight_rows;
    unsigned long long scale_offset_arg = scale_offset;
    void* quantize_params[] = {&input_ptr, &hi_ptr, &lo_ptr, &scale_ptr, &rows_arg, &width_arg};
    CUresult result = euhedral_launch_kernel(quantize, (unsigned int)blocks, 1, 1, 256, 1, 1, 0, stream, quantize_params, NULL);
    if (result != CUDA_SUCCESS) return finish(result);
    void* gemm_params[] = {&hi_ptr, &lo_ptr, &scale_ptr, &weights_ptr, &output_ptr, &rows_arg, &weight_rows_arg, &width_arg,
            &scale_offset_arg};
    return finish(euhedral_launch_kernel(paired ? gate_up : linear, (unsigned int)grid, 1, 1, 256, 1, 1, Q3MX_SHARED_BYTES, stream,
            gemm_params, NULL));
}

int euhedral_cuda_linear_q3_mx_bf16(const void* input, const void* weights, void* output, void* scratch, uint32_t rows,
        uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size, uint64_t scratch_byte_size) {
    return run(0, input, weights, output, scratch, rows, in_features, out_features, weights_byte_size, scratch_byte_size);
}

int euhedral_cuda_q3_mx_gate_up_swiglu_bf16(const void* input, const void* weights, void* output, void* scratch,
        uint32_t rows, uint32_t width, uint32_t outputs, uint64_t weight_bytes, uint64_t scratch_byte_size) {
    return run(1, input, weights, output, scratch, rows, width, outputs, weight_bytes, scratch_byte_size);
}
