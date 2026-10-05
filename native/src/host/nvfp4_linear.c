#include "cuda_kernel_loader.h"
#include "decode_shapes.h"
#include "reference.h"
#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* NVFP4 weights (native/src/nvfp4/nvfp4.cuh): the 1 to 8 row tensor-core decode on BF16 activations, and
 * (from two rows) the native Blackwell tensor-core route below. Every kernel has an _sd4 twin for
 * row-split-k128-sd4-v1 tensors (table-indexed block scales), told apart by their byte size. Exact numerics and
 * shapes no kernel takes run the scalar reference (reference.c). */
#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static CUmodule module;
/* [layout][tile][M]: layout 0 plain NVFP4, 1 SD4; tile 0 _w8, 1 _w4 (decode_tile); 1 to 8 token rows. */
static CUfunction decode_rows[2][2][EUHEDRAL_DECODE_MAX_ROWS + 1];
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

static void initialize(void) {
    init_status = euhedral_cuda_load_kernel((const void*)&once, "nvfp4/kernels.cu", "euhedral_nvfp4_decode_rows1_w8", &module,
            &decode_rows[0][0][1]);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    for (int layout = 0; layout < 2; ++layout) {
        for (int tile = 0; tile < 2; ++tile) {
            for (unsigned int m = 1; m <= EUHEDRAL_DECODE_MAX_ROWS; ++m) {
                char name[64];
                snprintf(name, sizeof(name), "euhedral_nvfp4_decode_rows%u_%s%s", m, tile ? "w4" : "w8", layout ? "_sd4" : "");
                if (cuModuleGetFunction(&decode_rows[layout][tile][m], module, name) != CUDA_SUCCESS)
                    init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
                euhedral_cuda_pdl_register(decode_rows[layout][tile][m]);
            }
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

static uint64_t align256(uint64_t value) { return (value + 255u) & ~255ull; }

/* Byte size of an NVFP4 tensor of `rows` rows of `in_features` values (Nvfp4Layout.byteSize): plain
 * (one E4M3 scale per 16 values, then the FP32 global) or SD4 (a 4-bit scale index per 16 values, then
 * the 16-code table and the global). */
static uint64_t nvfp4_size(uint32_t rows, uint32_t in_features, int sd4) {
    uint64_t k = ((uint64_t)in_features + 127u) / 128u * 128u;
    uint64_t scales = align256((uint64_t)rows * k / 2u);
    return align256(scales + (uint64_t)rows * k / (sd4 ? 32u : 16u)) + (sd4 ? 20u : 4u);
}

/* 0 for a plain tensor, 1 for SD4, -1 when the size matches neither. The SD4 scale plane is half the
 * plain one, at least 4 bytes per row, so the sizes never coincide. */
static int nvfp4_layout(uint32_t rows, uint32_t in_features, uint64_t byte_size) {
    if (byte_size == nvfp4_size(rows, in_features, 0)) return 0;
    if (byte_size == nvfp4_size(rows, in_features, 1)) return 1;
    return -1;
}

static int finish(CUresult status) {
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

static int prepare(const void* input, const void* weights, const void* output, uint32_t rows, uint32_t in_features,
        uint32_t weight_rows, uint64_t weights_byte_size, int* layout) {
    if (input == NULL || weights == NULL || output == NULL || rows == 0 || in_features == 0 || weight_rows == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    *layout = nvfp4_layout(weight_rows, in_features, weights_byte_size);
    if (*layout < 0) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    if (in_features % 32u != 0 || ((uintptr_t)input & 15u) != 0 || ((uintptr_t)weights & 15u) != 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    return ensure_initialized();
}

/* Decode tile of a weight shape (nvfp4/kernels.cu): 1 (_w4, 32 rows per CTA, 4 warps over K) for in_features above
 * 5120 when out_features allows it, else 0 (_w8, 16 rows per CTA, 8 warps). A function of the shape alone, so
 * one-row and multi-row calls on a tensor share their summation order. */
static int decode_tile(uint32_t in_features, uint32_t out_features) {
    return in_features > 5120u && out_features % 32u == 0;
}

/* Production dispatch for BF16 activations: 1 to 8 rows (decode, speculative verification and drafting) on the
 * tensor-core decode kernels, each row bit for bit as a one-row call. Other row counts run on the native route
 * (euhedral_cuda_linear_nvfp4_native_bf16) or, when it declines, the scalar reference. */
int euhedral_cuda_linear_nvfp4_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size) {
    int layout;
    int status = prepare(input, weights, output, rows, in_features, out_features, weights_byte_size, &layout);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    const int qualified = rows <= EUHEDRAL_DECODE_MAX_ROWS && in_features % 128u == 0 && out_features % 16u == 0;
    if (euhedral_cuda_exact_numerics() || !qualified)
        return euhedral_reference_nvfp4(input, weights, output, rows, in_features, out_features, layout);
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int in_arg = in_features, out_arg = out_features;
    void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &in_arg, &out_arg};
    const int tile = decode_tile(in_features, out_features);
    return finish(euhedral_launch_kernel(decode_rows[layout][tile][rows], out_features / (tile ? 32u : 16u), 1, 1,
            tile ? 128u : 256u, 1, 1, 0, euhedral_cuda_submission_stream(), params, NULL));
}

/* Native Blackwell NVFP4 (native/src/nvfp4_native, docs/NVFP4_NATIVE.md): the BF16 activations are
 * quantized to two NVFP4 terms (the value and its quantized residual) into caller scratch, then multiplied
 * with block-scaled FP4 tensor-core MMA (OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X). The module is compiled for
 * this device's sm_12xa target and exists only there. */
#define NATIVE_SHARED_BYTES 85040u  /* nvfp4n::Pipeline<2>::kSharedBytes */
#define NATIVE_TILE_THREADS 288u   /* nvfp4n::kThreads: eight consumer warps and a producer warp */
#define SKINNY_MAX_ROWS 64u
#define SKINNY_COLUMNS 64u
#define SKINNY_TARGET_CTAS 140u  /* two per SM */

/* nvfp4n::Skinny<terms, rows / 16>::kSharedBytes: stages of (A rows + 64 weight rows) x 160 bytes. */
static uint32_t skinny_shared_bytes(uint32_t terms, uint32_t rows) {
    uint32_t stage = (terms * rows + SKINNY_COLUMNS) * 160u, stages = 101376u / stage;
    return (stages > 4u ? 4u : stages) * stage;
}

/* K splits for the skinny kernel: enough CTAs to keep every SM streaming weights. */
static uint32_t skinny_splits(uint32_t in_features, uint32_t out_features) {
    uint32_t tiles = (out_features + SKINNY_COLUMNS - 1u) / SKINNY_COLUMNS, k_tiles = in_features / 256u;
    uint32_t splits = (SKINNY_TARGET_CTAS + tiles - 1u) / tiles, limit = k_tiles / 2u;
    if (splits > limit) splits = limit;
    return splits == 0u ? 1u : splits;
}

static int skinny_route(uint32_t rows, uint32_t in_features) {
    return rows <= SKINNY_MAX_ROWS && in_features % 256u == 0;
}
#ifdef _WIN32
static INIT_ONCE native_once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t native_once = PTHREAD_ONCE_INIT;
#endif
static CUmodule native_module;
static CUfunction native_quantize, native_finish;
static CUfunction native_linear[2], native_gate_up[2];  /* [layout] */
static CUfunction native_skinny[2][3];  /* [layout][up to 16, 32 and 64 rows] */
static int native_status = EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
/* Activation terms: the residual term brings the error per linear to about 1%, the relaxed-numerics floor
 * of the drift harness (docs/NVFP4_NATIVE.md). */
#define NATIVE_TERMS 2u

static void initialize_native(void) {
    int status = euhedral_cuda_load_native_kernel((const void*)&native_once, "nvfp4_native/kernels.cu",
            "euhedral_nvfp4n_linear_128x128", &native_module, &native_linear[0]);
    if (status == EUHEDRAL_CUDA_SUCCESS
            && (cuModuleGetFunction(&native_quantize, native_module, "euhedral_nvfp4n_quantize_rows") != CUDA_SUCCESS
                    || cuModuleGetFunction(&native_finish, native_module, "euhedral_nvfp4n_skinny_finish") != CUDA_SUCCESS))
        status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    static const char* skinny_names[3] = {"euhedral_nvfp4n_skinny_16", "euhedral_nvfp4n_skinny_32", "euhedral_nvfp4n_skinny_64"};
    for (int layout = 0; layout < 2 && status == EUHEDRAL_CUDA_SUCCESS; ++layout) {
        const char* suffix = layout ? "_sd4" : "";
        char name[64];
        snprintf(name, sizeof(name), "euhedral_nvfp4n_linear_128x128%s", suffix);
        if (cuModuleGetFunction(&native_linear[layout], native_module, name) != CUDA_SUCCESS) status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
        snprintf(name, sizeof(name), "euhedral_nvfp4n_gate_up_swiglu_128x64%s", suffix);
        if (status == EUHEDRAL_CUDA_SUCCESS && cuModuleGetFunction(&native_gate_up[layout], native_module, name) != CUDA_SUCCESS)
            status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
        for (int i = 0; i < 3 && status == EUHEDRAL_CUDA_SUCCESS; ++i) {
            snprintf(name, sizeof(name), "%s%s", skinny_names[i], suffix);
            if (cuModuleGetFunction(&native_skinny[layout][i], native_module, name) != CUDA_SUCCESS
                    || cuFuncSetAttribute(native_skinny[layout][i], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               (int)skinny_shared_bytes(NATIVE_TERMS, 16u << i)) != CUDA_SUCCESS)
                status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
        }
        if (status == EUHEDRAL_CUDA_SUCCESS
                && (cuFuncSetAttribute(native_linear[layout], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                            (int)NATIVE_SHARED_BYTES) != CUDA_SUCCESS
                        || cuFuncSetAttribute(native_gate_up[layout], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                                   (int)NATIVE_SHARED_BYTES) != CUDA_SUCCESS))
            status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    }
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

/* Activations of `rows` rows of `in_features` values in the selected number of terms: codes, the tile-major
 * scales (rows rounded up to 128 per tile and term, 8 bytes per row) and one FP32 global per row
 * (nvfp4n::ActivationLayout). */
static uint64_t activation_bytes(uint32_t rows, uint32_t in_features) {
    uint64_t k = ((uint64_t)in_features + 127u) / 128u * 128u, planes = (uint64_t)rows * NATIVE_TERMS;
    uint64_t pad = ((uint64_t)rows + 127u) / 128u * 128u;
    uint64_t scales = align256(planes * k / 2u);
    return align256(scales + k / 128u * NATIVE_TERMS * pad * 8u) + 4ull * rows;
}

/* Scratch for one native linear: the activations, then (split-K skinny shapes) FP32 partials. */
uint64_t euhedral_cuda_nvfp4_native_scratch_bytes(uint32_t rows, uint32_t in_features, uint32_t out_features) {
    uint64_t bytes = activation_bytes(rows, in_features);
    if (skinny_route(rows, in_features) && skinny_splits(in_features, out_features) > 1u)
        bytes = align256(bytes) + 4ull * skinny_splits(in_features, out_features) * rows * out_features;
    return bytes;
}

/* A tiled tensor map over bytes (the TMA descriptors of the tile kernels). */
static CUresult byte_map(CUtensorMap* map, CUdeviceptr address, cuuint32_t rank, const cuuint64_t* dims,
        const cuuint64_t* strides, const cuuint32_t* box, CUtensorMapSwizzle swizzle) {
    const cuuint32_t element_strides[3] = {1u, 1u, 1u};
    return cuTensorMapEncodeTiled(map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, (void*)(uintptr_t)address, dims, strides, box,
            element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle, CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
            CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

static int launch_native(int paired, uint64_t columns, const void* input, const void* weights, void* output,
        void* scratch, uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size,
        uint64_t scratch_byte_size);

int euhedral_cuda_linear_nvfp4_native_bf16(const void* input, const void* weights, void* output, void* scratch,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size,
        uint64_t scratch_byte_size) {
    int status = ensure_native();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    return launch_native(0, out_features, input, weights, output, scratch, rows, in_features, out_features,
            weights_byte_size, scratch_byte_size);
}

int euhedral_cuda_nvfp4_native_gate_up_swiglu_bf16(const void* input, const void* weights, void* output, void* scratch,
        uint32_t rows, uint32_t width, uint32_t outputs, uint64_t weight_bytes, uint64_t scratch_byte_size) {
    int status = ensure_native();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    if (outputs % 2u != 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    return launch_native(1, outputs, input, weights, output, scratch, rows, width, outputs, weight_bytes,
            scratch_byte_size);
}

/* `columns` B-tile rows to cover: out_features for a linear (128 per tile) and all gate + up rows for
 * the paired region (64 outputs, 128 weight rows, per tile). */
static int launch_native(int paired, uint64_t columns, const void* input, const void* weights, void* output,
        void* scratch, uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size,
        uint64_t scratch_byte_size) {
    int status, layout;
    // Activations are quantized, so exact numerics keep the scalar reference.
    if (in_features % 128u != 0 || euhedral_cuda_exact_numerics()) return EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
    status = prepare(input, weights, output, rows, in_features, out_features, weights_byte_size, &layout);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    /* The tile kernels' TMA descriptors need 16-byte strides: the scale rows (K / 16 bytes, or K / 32 for SD4) and
     * a 16-byte aligned scratch. */
    const int skinny_shape = !paired && skinny_route(rows, in_features);
    if (!skinny_shape && (in_features % (layout ? 512u : 256u) != 0 || ((uintptr_t)scratch & 15u) != 0))
        return EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
    CUfunction kernel = paired ? native_gate_up[layout] : native_linear[layout];
    const int skinny = !paired && skinny_route(rows, in_features);
    if (scratch == NULL || scratch_byte_size < (skinny ? euhedral_cuda_nvfp4_native_scratch_bytes(rows, in_features, out_features)
                                                       : activation_bytes(rows, in_features)))
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
    if (skinny) {
        /* Decode-like rows: weight streaming dominates, so 64-column CTAs, split over K when the
         * columns alone cannot occupy the GPU; each split stores FP32 partials, summed in split order
         * and converted once, so results are deterministic. */
        const uint32_t index = rows <= 16u ? 0u : rows <= 32u ? 1u : 2u, splits = skinny_splits(in_features, out_features);
        CUdeviceptr partials = splits > 1u ? scratch_ptr + align256(activation_bytes(rows, in_features)) : 0;
        void* skinny_params[] = {&scratch_ptr, &weights_ptr, &output_ptr, &partials, &rows_arg, &in_arg, &out_arg};
        result = euhedral_launch_kernel(native_skinny[layout][index], (out_features + SKINNY_COLUMNS - 1u) / SKINNY_COLUMNS, splits, 1,
                128, 1, 1, skinny_shared_bytes(NATIVE_TERMS, 16u << index), stream, skinny_params, NULL);
        if (result != CUDA_SUCCESS || splits == 1u) return finish(result);
        unsigned int count = rows * out_features, splits_arg = splits;
        void* finish_params[] = {&partials, &output_ptr, &count, &splits_arg};
        return finish(euhedral_launch_kernel(native_finish, (count + 255u) / 256u, 1, 1, 256, 1, 1, 0, stream,
                finish_params, NULL));
    }
    /* Tile kernels: TMA descriptors over the activation code planes (K / 2 bytes, rows, terms), the weight codes
     * (K / 2 bytes, weight rows) and the weight scales (row_scales bytes, weight rows). */
    const uint64_t row_bytes = in_features / 2u, row_scales = in_features / (layout ? 32u : 16u);
    const uint64_t scale_offset = align256((uint64_t)out_features * row_bytes);
    const cuuint32_t b_rows = paired ? 64u : 128u;
    CUtensorMap tm_a, tm_b, tm_bs;
    const cuuint64_t a_dims[3] = {row_bytes, rows, NATIVE_TERMS}, a_strides[2] = {row_bytes, row_bytes * rows};
    const cuuint32_t a_box[3] = {64u, 128u, 1u};
    const cuuint64_t b_dims[2] = {row_bytes, out_features}, b_strides[1] = {row_bytes};
    const cuuint32_t b_box[2] = {64u, b_rows};
    const cuuint64_t s_dims[2] = {row_scales, out_features}, s_strides[1] = {row_scales};
    const cuuint32_t s_box[2] = {16u, b_rows};
    result = byte_map(&tm_a, scratch_ptr, 3, a_dims, a_strides, a_box, CU_TENSOR_MAP_SWIZZLE_64B);
    if (result == CUDA_SUCCESS) result = byte_map(&tm_b, weights_ptr, 2, b_dims, b_strides, b_box, CU_TENSOR_MAP_SWIZZLE_64B);
    if (result == CUDA_SUCCESS)
        result = byte_map(&tm_bs, weights_ptr + scale_offset, 2, s_dims, s_strides, s_box, CU_TENSOR_MAP_SWIZZLE_NONE);
    if (result != CUDA_SUCCESS) return finish(result);
    void* linear_params[] = {&tm_a, &tm_b, &tm_bs, &scratch_ptr, &weights_ptr, &output_ptr, &rows_arg, &in_arg, &out_arg};
    return finish(euhedral_launch_kernel(kernel, (unsigned int)grid, 1, 1, NATIVE_TILE_THREADS, 1, 1, NATIVE_SHARED_BYTES,
            stream, linear_params, NULL));
}
