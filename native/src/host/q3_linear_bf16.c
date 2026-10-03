#include "cuda_kernel_loader.h"
#include "decode_shapes.h"
#include "q3_p2e2_geometry.h"
#include "reference.h"
#include <cuda_runtime_api.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* Q3G64_F16S decode (native/src/q3/kernels.cu): the contiguous one-row kernel, its 2 to 8 row twins and the
 * P2E2 decode and expansion kernels. Prefill (9 or more rows) runs on block-scaled FP8 tensor cores
 * (q3_mx.c); exact numerics and shapes no kernel takes run the scalar reference (reference.c). */
#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static CUmodule module;
static CUfunction decode_one, decode_rows[EUHEDRAL_DECODE_MAX_ROWS + 1];  /* [M]: 2..8 rows */
static CUfunction p2e2_decode_rows[5], p2e2_expand;  /* [M]: 1..4 rows */
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

static int get_function(CUfunction* function, const char* name) {
    return cuModuleGetFunction(function, module, name) == CUDA_SUCCESS ? EUHEDRAL_CUDA_SUCCESS : EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
}

static void initialize(void) {
    init_status = euhedral_cuda_load_kernel((const void*)&once, "q3/kernels.cu", "euhedral_q3_decode_contiguous", &module, &decode_one);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    for (unsigned int m = 2; m <= EUHEDRAL_DECODE_MAX_ROWS; m++) {
        char name[48];
        snprintf(name, sizeof(name), "euhedral_q3_decode_contiguous_rows%u", m);
        if (get_function(&decode_rows[m], name) != EUHEDRAL_CUDA_SUCCESS) init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
        euhedral_cuda_pdl_register(decode_rows[m]);
    }
    if (get_function(&p2e2_decode_rows[1], "euhedral_q3_p2e2_decode") != EUHEDRAL_CUDA_SUCCESS
            || get_function(&p2e2_expand, "euhedral_q3_p2e2_expand") != EUHEDRAL_CUDA_SUCCESS)
        init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    for (unsigned int m = 2; m <= 4; m++) {
        char name[48];
        snprintf(name, sizeof(name), "euhedral_q3_p2e2_decode_rows%u", m);
        if (get_function(&p2e2_decode_rows[m], name) != EUHEDRAL_CUDA_SUCCESS) init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
    // The decode kernels begin with euhedral_pdl_begin() (see cuda_kernel_loader.h).
    euhedral_cuda_pdl_register(decode_one);
    for (unsigned int m = 1; m <= 4; m++) euhedral_cuda_pdl_register(p2e2_decode_rows[m]);
}
#ifdef _WIN32
static BOOL CALLBACK initialize_once(PINIT_ONCE state, PVOID parameter, PVOID* context) {
    (void)state; (void)parameter; (void)context;
    initialize();
    return TRUE;
}
#endif

static int ensure_initialized(void);

/* Production dispatch: 1 to 8 rows on the contiguous decode kernels, every row bit for bit as a one-row call.
 * Larger batches belong to the FP8 route, so what reaches here is the scalar reference. */
int euhedral_cuda_linear_q3_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size) {
    if (input == NULL || weights == NULL || output == NULL || rows == 0 || in_features == 0 || out_features == 0 || weights_byte_size == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int context_status = euhedral_cuda_bind_thread_context();
    if (context_status != EUHEDRAL_CUDA_SUCCESS) return context_status;
    const uint64_t k_pad = ((uint64_t)in_features + 127u) / 128u * 128u, groups = k_pad / 64u;
    const uint64_t scale_offset = (((uint64_t)out_features * groups * 24u) + 255u) & ~255ull;
    if (scale_offset + (uint64_t)out_features * groups * 2u != weights_byte_size) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    const int qualified = rows <= EUHEDRAL_DECODE_MAX_ROWS && euhedral_q3_decode_shape(in_features, out_features)
            && ((uintptr_t)input & 15u) == 0u && ((uintptr_t)weights & 3u) == 0u;
    if (euhedral_cuda_exact_numerics() || !qualified)
        return euhedral_reference_q3(input, weights, output, rows, in_features, out_features, weights_byte_size);
    int status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int rows_arg = rows, in_arg = in_features, out_arg = out_features;
    unsigned long long scale_arg = scale_offset;
    void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &rows_arg, &in_arg, &out_arg, &scale_arg};
    CUresult result = euhedral_launch_kernel(rows == 1 ? decode_one : decode_rows[rows], out_features / 16u, 1, 1, 128, 1, 1, 0,
            euhedral_cuda_submission_stream(), params, NULL);
    if (result != CUDA_SUCCESS) return (int)result;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

static int ensure_initialized(void) {
#ifdef _WIN32
    if (!InitOnceExecuteOnce(&once, initialize_once, NULL, NULL)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    if (pthread_once(&once, initialize) != 0) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#endif
    return init_status;
}

static int finish_launch(CUresult status) {
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

int euhedral_cuda_linear_q3_p2e2_decode_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size) {
    if (input == NULL || weights == NULL || output == NULL || rows == 0 || out_features == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int geometry = euhedral_q3_p2e2_geometry(out_features, in_features, weights_byte_size);
    if (geometry != EUHEDRAL_CUDA_SUCCESS) return geometry;
    int context_status = euhedral_cuda_bind_thread_context();
    if (context_status != EUHEDRAL_CUDA_SUCCESS) return context_status;
    int status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    // The fused route reproduces the contiguous decode kernels, which serve 1 to 8 rows; every other route expands.
    if (euhedral_cuda_exact_numerics() || rows > EUHEDRAL_DECODE_MAX_ROWS || ((uintptr_t)input & 15u) != 0u
            || ((uintptr_t)weights & 15u) != 0u || !euhedral_q3_decode_shape(in_features, out_features))
        return EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
    /* Up to four rows per launch; each row is the one-row result whichever launch computes it. */
    for (uint32_t first = 0; first < rows; first += 4u) {
        uint32_t count = rows - first < 4u ? rows - first : 4u;
        CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input + (CUdeviceptr)first * in_features * 2u;
        CUdeviceptr weights_ptr = (CUdeviceptr)(uintptr_t)weights;
        CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output + (CUdeviceptr)first * out_features * 2u;
        unsigned int in_arg = in_features, out_arg = out_features;
        void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &in_arg, &out_arg};
        CUresult result = euhedral_launch_kernel(p2e2_decode_rows[count], out_features / 16u, 1, 1, 128, 1, 1, 0,
                euhedral_cuda_submission_stream(), params, NULL);
        if (result != CUDA_SUCCESS) return (int)result;
    }
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

int euhedral_cuda_q3_p2e2_expand(const void* weights, uint64_t weights_byte_size, uint32_t rows,
        uint32_t in_features, uint32_t first_row, uint32_t row_count, void* destination, uint64_t destination_byte_size) {
    if (weights == NULL || destination == NULL || row_count == 0 || first_row >= rows || row_count > rows - first_row)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int geometry = euhedral_q3_p2e2_geometry(rows, in_features, weights_byte_size);
    if (geometry != EUHEDRAL_CUDA_SUCCESS) return geometry;
    if (destination_byte_size < euhedral_q3_row_split_size(row_count, in_features)) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    if (((uintptr_t)weights & 15u) != 0u || ((uintptr_t)destination & 15u) != 0u) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int context_status = euhedral_cuda_bind_thread_context();
    if (context_status != EUHEDRAL_CUDA_SUCCESS) return context_status;
    int status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr weights_ptr = (CUdeviceptr)(uintptr_t)weights, destination_ptr = (CUdeviceptr)(uintptr_t)destination;
    unsigned int rows_arg = rows, in_arg = in_features, first_arg = first_row, count_arg = row_count;
    void* params[] = {&weights_ptr, &destination_ptr, &rows_arg, &in_arg, &first_arg, &count_arg};
    return finish_launch(euhedral_launch_kernel(p2e2_expand, (row_count + 3u) / 4u, 1, 1, 128, 1, 1, 0,
            euhedral_cuda_submission_stream(), params, NULL));
}
