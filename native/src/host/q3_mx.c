#include "cuda_kernel_loader.h"
#include <cuda_runtime_api.h>
#include <stdint.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* Q3 prefill on block-scaled FP8 tensor cores (native/src/q3_mx, docs/kernels/PREFILL_MX.md): the BF16 activations are
 * split into two E4M3 terms with a power-of-two block scale (exact for BF16), the Q3 codes are exact E4M3, and the
 * MMA accumulates in FP32 at the full FP8 rate of GeForce Blackwell. Needs an sm_12x device. The Rust engine launches
 * the kernels itself; the library answers whether the module loads on this device. */
#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static CUmodule module;
static CUfunction quantize, linear, gate_up, linear_q4, linear_q5, reduce;
static int init_status = EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;

/* q3mx::kSharedBytes of native/src/q3_mx/kernels.cu: three stages of (A hi, A lo, expanded W, 128 FP32 scales). */
#define Q3MX_SHARED_BYTES (3u * (3u * 128u * 64u + 128u * 4u))

static void initialize(void) {
    int status = euhedral_cuda_load_native_kernel((const void*)&once, "q3_mx/kernels.cu", "euhedral_q3mx_quantize", &module, &quantize);
    if (status == EUHEDRAL_CUDA_SUCCESS
            && (cuModuleGetFunction(&linear, module, "euhedral_q3mx_linear_128x128") != CUDA_SUCCESS
                    || cuModuleGetFunction(&gate_up, module, "euhedral_q3mx_gate_up_swiglu_128x64") != CUDA_SUCCESS
                    || cuModuleGetFunction(&linear_q4, module, "euhedral_q4mx_linear_128x128") != CUDA_SUCCESS
                    || cuModuleGetFunction(&linear_q5, module, "euhedral_q5mx_linear_128x128") != CUDA_SUCCESS
                    || cuFuncSetAttribute(linear_q4, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)Q3MX_SHARED_BYTES) != CUDA_SUCCESS
                    || cuFuncSetAttribute(linear_q5, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)Q3MX_SHARED_BYTES) != CUDA_SUCCESS
                    || cuFuncSetAttribute(linear, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)Q3MX_SHARED_BYTES) != CUDA_SUCCESS
                    || cuFuncSetAttribute(gate_up, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)Q3MX_SHARED_BYTES) != CUDA_SUCCESS))
        status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    if (status == EUHEDRAL_CUDA_SUCCESS && cuModuleGetFunction(&reduce, module, "euhedral_q3mx_reduce") != CUDA_SUCCESS)
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

int euhedral_cuda_q3_mx_available(void) {
    return ensure() == EUHEDRAL_CUDA_SUCCESS;
}
