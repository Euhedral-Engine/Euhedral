#include "cuda_kernel_loader.h"
#include "submission.h"
#include "euhedral_cuda.h"
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* The Flash-Next (qwen4_exp) kernels of native/src/qwen4/kernels.cu behind one launch entry point.
 *
 * The model's operators are many small kernels that differ only in their arguments, so instead of one C wrapper and
 * one FFM descriptor each, the host exposes a table of kernels and a single launcher. A launch passes the kernel's
 * index, its geometry and its arguments as 8-byte words with the byte size (4 or 8) each argument has in the kernel's
 * signature. The launcher checks the argument count and every size against the compiled kernel
 * (cuFuncGetParamInfo), so a mismatch between the Java caller and the CUDA source fails the call instead of
 * corrupting it. The order of EUHEDRAL_QWEN4_KERNELS is the ABI: Qwen4Kernel on the Java side lists the same names in
 * the same order and a test compares them. */
#define QWEN4_MAX_PARAMETERS 32

typedef struct {
    const char* name;
    unsigned int max_dynamic_shared_bytes;
} Qwen4KernelEntry;

static const Qwen4KernelEntry kernels[] = {
    {"euhedral_q4_linear_bf16", 0},
    {"euhedral_q4_grouped_rms_norm_bf16", 0},
    {"euhedral_q4_scaled_silu_bf16", 0},
    {"euhedral_q4_hc_mix_bf16", 0},
    {"euhedral_q4_hc_inject_bf16", 0},
    {"euhedral_q4_repeat_streams_bf16", 0},
    {"euhedral_q4_ngram_expand_bf16", 0},
    {"euhedral_q4_ple_gate_bf16", 0},
    {"euhedral_q4_ple_conv_bf16", 0},
    {"euhedral_q4_conv_history_bf16", 0},
    {"euhedral_q4_gdn_conv_bf16", 0},
    {"euhedral_q4_gdn_control_bf16", 0},
    {"euhedral_q4_gdn_gated_norm_bf16", 0},
    {"euhedral_q4_head_norm_rope_bf16", 0},
    {"euhedral_q4_qsa_pool_keys_bf16", 0},
    {"euhedral_q4_qsa_tail_bf16", 0},
    {"euhedral_q4_qsa_scores", 0},
    {"euhedral_q4_qsa_select", 0},
    /* One unit's three warps (q4qsa::kUnitSharedBytes). */
    {"euhedral_q4_qsa_attention", 28224},
    {"euhedral_q4_qsa_merge", 0},
    {"euhedral_q4_qsa_kv_append", 0},
    {"euhedral_q4_router_bf16", 0},
    {"euhedral_q4_swiglu_bf16", 0},
    {"euhedral_q4_moe_finish_bf16", 0},
    {"euhedral_q4_embedding_bf16", 0},
    {"euhedral_q4_expert_gate_up_swiglu_bf16", 0},
    {"euhedral_q4_expert_down_bf16", 0},
    {"euhedral_q4_expert_combine_bf16", 0},
};
#define KERNEL_COUNT ((int)(sizeof(kernels) / sizeof(kernels[0])))

#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static int anchor;
static CUmodule module;
static CUfunction functions[sizeof(kernels) / sizeof(kernels[0])];
static int parameter_count[sizeof(kernels) / sizeof(kernels[0])];
static uint8_t parameter_size[sizeof(kernels) / sizeof(kernels[0])][QWEN4_MAX_PARAMETERS];
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

static void initialize(void) {
    init_status = euhedral_cuda_load_kernel(&anchor, "qwen4/kernels.cu", kernels[0].name, &module, &functions[0]);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    for (int index = 1; index < KERNEL_COUNT; index++) {
        init_status = (int)cuModuleGetFunction(&functions[index], module, kernels[index].name);
        if (init_status != CUDA_SUCCESS) return;
    }
    for (int index = 0; index < KERNEL_COUNT; index++) {
        if (kernels[index].max_dynamic_shared_bytes != 0) {
            init_status = (int)cuFuncSetAttribute(functions[index], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                    (int)kernels[index].max_dynamic_shared_bytes);
            if (init_status != CUDA_SUCCESS) return;
        }
        int count = 0;
        for (; count < QWEN4_MAX_PARAMETERS; count++) {
            size_t offset = 0, size = 0;
            if (cuFuncGetParamInfo(functions[index], (size_t)count, &offset, &size) != CUDA_SUCCESS) break;
            if (size != 4 && size != 8) {
                init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
                return;
            }
            parameter_size[index][count] = (uint8_t)size;
        }
        parameter_count[index] = count;
    }
}

#ifdef _WIN32
static BOOL CALLBACK initialize_once(PINIT_ONCE state, PVOID parameter, PVOID* context) {
    (void)state; (void)parameter; (void)context;
    initialize();
    return TRUE;
}
#endif

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_qwen4_kernel_count(void) {
    return KERNEL_COUNT;
}

EUHEDRAL_CUDA_EXPORT const char* euhedral_cuda_qwen4_kernel_name(int index) {
    return index >= 0 && index < KERNEL_COUNT ? kernels[index].name : NULL;
}

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_qwen4_launch(int32_t kernel, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        uint32_t block_x, uint32_t block_y, uint32_t block_z, uint32_t shared_bytes,
        const uint64_t* words, const uint8_t* sizes, int32_t count) {
    if (kernel < 0 || kernel >= KERNEL_COUNT || words == NULL || sizes == NULL || count < 0 || count > QWEN4_MAX_PARAMETERS
            || grid_x == 0 || grid_y == 0 || grid_z == 0 || block_x == 0 || block_y == 0 || block_z == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int context_status = euhedral_cuda_bind_thread_context();
    if (context_status != EUHEDRAL_CUDA_SUCCESS) return context_status;
#ifdef _WIN32
    if (!InitOnceExecuteOnce(&once, initialize_once, NULL, NULL)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    if (pthread_once(&once, initialize) != 0) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#endif
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return init_status;
    if (count != parameter_count[kernel]) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    void* parameters[QWEN4_MAX_PARAMETERS];
    uint32_t narrow[QWEN4_MAX_PARAMETERS];
    for (int index = 0; index < count; index++) {
        if (sizes[index] != parameter_size[kernel][index]) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
        if (sizes[index] == 8) {
            parameters[index] = (void*)&words[index];
        } else {
            narrow[index] = (uint32_t)words[index];
            parameters[index] = &narrow[index];
        }
    }
    CUresult status = euhedral_launch_kernel(functions[kernel], grid_x, grid_y, grid_z, block_x, block_y, block_z,
            shared_bytes, euhedral_cuda_submission_stream(), parameters, NULL);
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}
