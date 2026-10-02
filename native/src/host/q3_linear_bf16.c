#include "cuda_kernel_loader.h"
#include "q3_prefill_policy.h"
#include "q3_p2e2_geometry.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdatomic.h>
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

#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static CUmodule module;
static CUfunction function;
static CUfunction decode1, decode2, decode4, decode_wide, decode_contiguous, prefill, prefill64, prefill64_wmma, prefill64_k32_cb, prefill_s104;
static CUfunction prefill_exact, prefill64_exact, prefill64_k32_cb_exact, prefill_s104_exact;
static CUfunction prefill_engine, prefill_engine64;
static CUfunction p2e2_decode, p2e2_expand;
static CUfunction decode_contiguous_rows[9];  /* [M]: row-exact twins for 2..8 rows */
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
/* Alias of euhedral_cuda_select_exact_numerics, kept for existing callers. */
int euhedral_cuda_q3_decode_select_exact(int exact) {
    return euhedral_cuda_select_exact_numerics(exact);
}
static CUfunction optional_kernel(const char* name) {
    CUfunction loaded = NULL;
    return cuModuleGetFunction(&loaded, module, name) == CUDA_SUCCESS ? loaded : NULL;
}
static void initialize(void) {
    init_status = euhedral_cuda_load_kernel((const void*)&once, "q3/kernels.cu", "euhedral_q3_linear_bf16", &module, &function);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    decode1 = optional_kernel("euhedral_q3_decode_1");
    decode2 = optional_kernel("euhedral_q3_decode_2");
    decode4 = optional_kernel("euhedral_q3_decode_4");
    decode_wide = optional_kernel("euhedral_q3_decode_wide");
    decode_contiguous = optional_kernel("euhedral_q3_decode_contiguous");
    prefill = optional_kernel("euhedral_q3_prefill");
    prefill_s104 = optional_kernel("euhedral_q3_prefill_s104");
    prefill64 = optional_kernel("euhedral_q3_prefill_64");
    prefill64_wmma = optional_kernel("euhedral_q3_prefill_64_wmma");
    prefill64_k32_cb = optional_kernel("euhedral_q3_prefill_64_k32_cb");
    prefill_exact = optional_kernel("euhedral_q3_prefill_exact");
    prefill_s104_exact = optional_kernel("euhedral_q3_prefill_s104_exact");
    prefill64_exact = optional_kernel("euhedral_q3_prefill_64_exact");
    prefill64_k32_cb_exact = optional_kernel("euhedral_q3_prefill_64_k32_cb_exact");
    prefill_engine = optional_kernel("euhedral_q3_prefill_128x64");
    prefill_engine64 = optional_kernel("euhedral_q3_prefill_64x64");
    p2e2_decode = optional_kernel("euhedral_q3_p2e2_decode");
    p2e2_expand = optional_kernel("euhedral_q3_p2e2_expand");
    for (int m = 2; m <= 8; m++) {
        char name[48];
        snprintf(name, sizeof(name), "euhedral_q3_decode_contiguous_rows%d", m);
        decode_contiguous_rows[m] = optional_kernel(name);
        euhedral_cuda_pdl_register(decode_contiguous_rows[m]);
    }
    // The decode kernels begin with euhedral_pdl_begin() (see cuda_kernel_loader.h).
    euhedral_cuda_pdl_register(decode1);
    euhedral_cuda_pdl_register(decode2);
    euhedral_cuda_pdl_register(decode4);
    euhedral_cuda_pdl_register(decode_wide);
    euhedral_cuda_pdl_register(decode_contiguous);
    euhedral_cuda_pdl_register(p2e2_decode);
}
#ifdef _WIN32
static BOOL CALLBACK initialize_once(PINIT_ONCE state, PVOID parameter, PVOID* context) {
    (void)state; (void)parameter; (void)context;
    initialize();
    return TRUE;
}
#endif

static int linear_q3(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size, int mode) {
    if (input == NULL || weights == NULL || output == NULL || rows == 0 || in_features == 0 || out_features == 0 || weights_byte_size == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int context_status = euhedral_cuda_bind_thread_context();
    if (context_status != EUHEDRAL_CUDA_SUCCESS) return context_status;
    uint64_t k_pad = ((uint64_t)in_features + 127u) / 128u * 128u;
    uint64_t groups = k_pad / 64u;
    uint64_t base = (uint64_t)out_features * groups * 24u;
    uint64_t scale_offset = (base + 255u) & ~255ull;
    uint64_t expected = scale_offset + (uint64_t)out_features * groups * 2u;
    if (expected != weights_byte_size) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    uint64_t grid = (uint64_t)rows * out_features;
    uint32_t row_tile = rows == 1 ? 1 : rows == 2 ? 2 : 4;
    if (mode == 1) grid = (((uint64_t)rows + row_tile - 1) / row_tile) * (((uint64_t)out_features + 7) / 8);
    // Match the prefill grid to the selected CTA tile. The optional tile64
    // symbol is checked below before normal dispatch can use this grid.
    int wide_prefill = euhedral_q3_wide_prefill(mode, rows, in_features, out_features);
    if (mode == 2 || mode == 3) grid = (((uint64_t)rows + (wide_prefill ? 63 : 31)) / (wide_prefill ? 64 : 32))
            * (((uint64_t)out_features + 31) / 32);
    if (grid > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
#ifdef _WIN32
    if (!InitOnceExecuteOnce(&once, initialize_once, NULL, NULL)) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#else
    if (pthread_once(&once, initialize) != 0) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
#endif
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return init_status;
    // Row-exact verification (euhedral_cuda_row_exact): every row exactly as a one-row decode call. Where
    // that call runs the contiguous kernel, its multi-row twin computes all rows in one launch, bit for
    // bit; otherwise the rows run one at a time.
    if (rows > 1 && euhedral_cuda_row_exact()) {
        int one_row_contiguous = decode_contiguous != NULL && !euhedral_cuda_exact_numerics()
                && ((uintptr_t)input & 15u) == 0u && ((uintptr_t)weights & 3u) == 0u
                && euhedral_q3_decode_contiguous_shape(1, in_features, out_features);
        if (one_row_contiguous && rows <= 8u && decode_contiguous_rows[rows] != NULL) {
            CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
            CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
            unsigned int rows_arg = rows, in_arg = in_features, out_arg = out_features;
            unsigned long long scale_arg = scale_offset;
            void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &rows_arg, &in_arg, &out_arg, &scale_arg};
            CUresult status = euhedral_launch_kernel(decode_contiguous_rows[rows], out_features / 16u, 1, 1, 128, 1, 1, 0,
                    euhedral_cuda_submission_stream(), params, NULL);
            if (status != CUDA_SUCCESS) return (int)status;
            if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
            cudaError_t sync = cudaDeviceSynchronize();
            return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
        }
        for (uint32_t row = 0; row < rows; row++) {
            int status = linear_q3((const unsigned char*)input + (uint64_t)row * in_features * 2u, weights,
                    (unsigned char*)output + (uint64_t)row * out_features * 2u, 1, in_features, out_features,
                    weights_byte_size, 1);
            if (status != EUHEDRAL_CUDA_SUCCESS) return status;
        }
        return EUHEDRAL_CUDA_SUCCESS;
    }
    // A matched older NVRTC source may lack the optional tile64 kernels. AUTO then
    // retains the 32-row route; explicit 64-row requests fail below.
    enum euhedral_q3_prefill_kernel prefill_kernel = EUHEDRAL_Q3_PREFILL_NONE;
    if (mode == 2 || mode == 3) {
        // CB uses 16-byte vector loads; qualified K row strides are multiples of 16.
        int input_aligned_16 = ((uintptr_t)input & 15u) == 0u;
        prefill_kernel = euhedral_q3_select_prefill_s104(mode, rows, in_features, out_features,
                prefill64_k32_cb != NULL, input_aligned_16,
                prefill64 != NULL, prefill64_wmma != NULL, prefill_s104 != NULL);
        wide_prefill = euhedral_q3_prefill_tile_rows(prefill_kernel) == 64u;
        grid = (((uint64_t)rows + (wide_prefill ? 63 : 31)) / (wide_prefill ? 64 : 32))
                * (((uint64_t)out_features + 31) / 32);
        if (grid > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    }
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int rows_arg = rows, in_arg = in_features, out_arg = out_features;
    unsigned long long scale_arg = scale_offset;
    void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &rows_arg, &in_arg, &out_arg, &scale_arg};
    int contiguous = mode == 1 && decode_contiguous != NULL && !euhedral_cuda_exact_numerics()
            && ((uintptr_t)input & 15u) == 0u && ((uintptr_t)weights & 3u) == 0u
            && euhedral_q3_decode_contiguous_shape(rows, in_features, out_features);
    if (contiguous) grid = out_features / 16u;
    int wide_decode = !contiguous && mode == 1 && decode_wide != NULL && ((uintptr_t)weights & 15u) == 0u
            && euhedral_q3_decode_wide_shape(rows, in_features, out_features);
    CUfunction selected = contiguous ? decode_contiguous : wide_decode ? decode_wide
            : mode == 1 ? (row_tile == 1 ? decode1 : row_tile == 2 ? decode2 : decode4)
            : mode == 0 ? function
            : prefill_kernel == EUHEDRAL_Q3_PREFILL64_K32_CB ? prefill64_k32_cb
            : prefill_kernel == EUHEDRAL_Q3_PREFILL64_WMMA ? prefill64_wmma
            : prefill_kernel == EUHEDRAL_Q3_PREFILL64 ? prefill64
            : prefill_kernel == EUHEDRAL_Q3_PREFILL32_S104 ? prefill_s104
            : prefill_kernel == EUHEDRAL_Q3_PREFILL32 ? prefill : NULL;
    // Exact numerics select the hi + lo twins of the prefill kernels.
    if ((mode == 2 || mode == 3) && euhedral_cuda_exact_numerics()) {
        CUfunction exact = selected == prefill64_k32_cb ? prefill64_k32_cb_exact
                : selected == prefill64 ? prefill64_exact
                : selected == prefill_s104 ? prefill_s104_exact
                : selected == prefill ? prefill_exact : NULL;
        if (exact != NULL) selected = exact;
    }
    // Relaxed numerics: the qualified mixer shape runs on the balanced tile engine.
    else if (mode == 2 && ((uintptr_t)input & 15u) == 0u && ((uintptr_t)weights & 3u) == 0u) {
        uint32_t engine_rows = euhedral_q3_prefill_engine_rows(mode, rows, in_features, out_features);
        CUfunction engine = engine_rows == 128u ? prefill_engine : engine_rows == 64u ? prefill_engine64 : NULL;
        if (engine != NULL) {
            selected = engine;
            grid = (((uint64_t)rows + engine_rows - 1u) / engine_rows) * (((uint64_t)out_features + 63u) / 64u);
        }
    }
    if (selected == NULL) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    CUresult status = euhedral_launch_kernel(selected, (unsigned int)grid, 1, 1, 128, 1, 1, 0, euhedral_cuda_submission_stream(), params, NULL);
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

int euhedral_cuda_linear_q3_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size) {
    return linear_q3(input, weights, output, rows, in_features, out_features, weights_byte_size, 0);
}

int euhedral_cuda_linear_q3_decode_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size) {
    return linear_q3(input, weights, output, rows, in_features, out_features, weights_byte_size, 1);
}

int euhedral_cuda_linear_q3_prefill_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size) {
    return linear_q3(input, weights, output, rows, in_features, out_features, weights_byte_size, 2);
}

int euhedral_cuda_linear_q3_prefill_64_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size) {
    return linear_q3(input, weights, output, rows, in_features, out_features, weights_byte_size, 3);
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
    // The fused route reproduces euhedral_q3_decode_contiguous only; every other route expands.
    if (p2e2_decode == NULL || euhedral_cuda_exact_numerics() || rows != 1u
            || ((uintptr_t)input & 15u) != 0u || ((uintptr_t)weights & 15u) != 0u
            || !euhedral_q3_decode_contiguous_shape(rows, in_features, out_features))
        return EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int in_arg = in_features, out_arg = out_features;
    void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &in_arg, &out_arg};
    return finish_launch(euhedral_launch_kernel(p2e2_decode, out_features / 16u, 1, 1, 128, 1, 1, 0,
            euhedral_cuda_submission_stream(), params, NULL));
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
    if (p2e2_expand == NULL) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    CUdeviceptr weights_ptr = (CUdeviceptr)(uintptr_t)weights, destination_ptr = (CUdeviceptr)(uintptr_t)destination;
    unsigned int rows_arg = rows, in_arg = in_features, first_arg = first_row, count_arg = row_count;
    void* params[] = {&weights_ptr, &destination_ptr, &rows_arg, &in_arg, &first_arg, &count_arg};
    return finish_launch(euhedral_launch_kernel(p2e2_expand, (row_count + 3u) / 4u, 1, 1, 128, 1, 1, 0,
            euhedral_cuda_submission_stream(), params, NULL));
}
