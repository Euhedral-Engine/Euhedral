#include "cuda_kernel_loader.h"
#include "decode_shapes.h"
#include "reference.h"
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* Q4 and Q5 decode (native/src/q45/kernels.cu): the contiguous one-row kernels and their 2 to 8 row twins.
 * Prefill (9 or more rows) runs on block-scaled FP8 tensor cores (q3_mx.c); exact numerics and shapes no
 * kernel takes run the scalar reference (reference.c). */
#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static CUmodule module;
/* [format]: 0 Q4, 1 Q5. */
static CUfunction decode_one[2], decode_rows[2][EUHEDRAL_DECODE_MAX_ROWS + 1];
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

static void initialize(void) {
    init_status = euhedral_cuda_load_kernel((const void*)&once, "q45/kernels.cu", "euhedral_q4_decode_contiguous", &module, &decode_one[0]);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    if (cuModuleGetFunction(&decode_one[1], module, "euhedral_q5_decode_contiguous") != CUDA_SUCCESS) {
        init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
        return;
    }
    for (int format = 0; format < 2; format++) {
        euhedral_cuda_pdl_register(decode_one[format]);
        for (unsigned int m = 2; m <= EUHEDRAL_DECODE_MAX_ROWS; m++) {
            char name[48];
            snprintf(name, sizeof(name), "euhedral_q%d_decode_contiguous_rows%u", 4 + format, m);
            if (cuModuleGetFunction(&decode_rows[format][m], module, name) != CUDA_SUCCESS)
                init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
            euhedral_cuda_pdl_register(decode_rows[format][m]);
        }
    }
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

static uint64_t align256(uint64_t size) { return (size + 255u) & ~UINT64_C(255); }

/* Byte size of a Q4 or Q5 tensor (the code plane, for Q5 the fifth-bit plane, then FP16 group scales,
 * each plane 256-aligned). */
static int quantized_byte_size(uint32_t in_features, uint32_t out_features, uint32_t bits, uint64_t* required) {
    if (in_features == 0 || in_features % 128 != 0 || out_features == 0 || (bits != 4 && bits != 5))
        return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    const uint64_t groups = in_features / 64;
    if ((uint64_t)out_features > UINT64_MAX / groups / 32) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    const uint64_t code_plane = align256((uint64_t)out_features * groups * 32);
    const uint64_t high_plane = bits == 5 ? align256((uint64_t)out_features * groups * 8) : 0;
    const uint64_t scale_offset = code_plane + high_plane;
    if ((uint64_t)out_features > (UINT64_MAX - scale_offset) / groups / 2) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    *required = scale_offset + (uint64_t)out_features * groups * 2;
    return EUHEDRAL_CUDA_SUCCESS;
}

int euhedral_cuda_linear_quantized_bf16(
        const void* device_input, const void* device_weights, void* device_output,
        uint32_t rows, uint32_t in_features, uint32_t out_features,
        uint64_t weights_byte_size, uint32_t bits) {
    if (device_input == NULL || device_weights == NULL || device_output == NULL || rows == 0 || out_features == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    uint64_t required;
    int status = quantized_byte_size(in_features, out_features, bits, &required);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    if (required != weights_byte_size) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    if ((uint64_t)rows * out_features > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    const int qualified = rows <= EUHEDRAL_DECODE_MAX_ROWS && euhedral_q45_decode_shape(in_features, out_features)
            && (((uintptr_t)device_weights | (uintptr_t)device_input) & 15u) == 0u;
    if (euhedral_cuda_exact_numerics() || !qualified)
        return euhedral_reference_q45(device_input, device_weights, device_output, rows, in_features, out_features, bits);
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    const int format = bits == 5;
    CUdeviceptr input = (CUdeviceptr)(uintptr_t)device_input;
    CUdeviceptr weights = (CUdeviceptr)(uintptr_t)device_weights;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, in_arg = in_features, out_arg = out_features;
    void* parameters[] = {&input, &weights, &output, &rows_arg, &in_arg, &out_arg};
    CUresult result = euhedral_launch_kernel(rows == 1 ? decode_one[format] : decode_rows[format][rows], out_features / 8u,
            1, 1, 128, 1, 1, 0, euhedral_cuda_submission_stream(), parameters, NULL);
    if (result != CUDA_SUCCESS) return (int)result;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}
