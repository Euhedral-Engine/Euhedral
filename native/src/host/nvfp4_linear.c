#include "cuda_kernel_loader.h"
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

/* NVFP4 weights (native/src/nvfp4/nvfp4.cuh): one-row decode, the balanced tile engine for every
 * other row count, and the paired gate/up + SwiGLU region. Every kernel has an _sd4 twin for
 * row-split-k128-sd4-v1 tensors (table-indexed block scales), told apart by their byte size. */
#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static CUmodule module;
/* [layout]: 0 plain NVFP4, 1 SD4. */
static CUfunction decode[2], prefill128[2], prefill64[2], gate_up128[2], gate_up64[2];
static CUfunction decode_rows[2][9];  /* [layout][M] for 2..8 token rows */
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

static CUfunction optional_kernel(const char* name) {
    CUfunction loaded = NULL;
    return cuModuleGetFunction(&loaded, module, name) == CUDA_SUCCESS ? loaded : NULL;
}

static void initialize(void) {
    init_status = euhedral_cuda_load_kernel((const void*)&once, "nvfp4/kernels.cu", "euhedral_nvfp4_decode", &module, &decode[0]);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    for (int layout = 0; layout < 2; ++layout) {
        const char* suffix = layout ? "_sd4" : "";
        char name[64];
        snprintf(name, sizeof(name), "euhedral_nvfp4_decode%s", suffix);
        decode[layout] = optional_kernel(name);
        snprintf(name, sizeof(name), "euhedral_nvfp4_prefill_128x64%s", suffix);
        prefill128[layout] = optional_kernel(name);
        snprintf(name, sizeof(name), "euhedral_nvfp4_prefill_64x64%s", suffix);
        prefill64[layout] = optional_kernel(name);
        snprintf(name, sizeof(name), "euhedral_nvfp4_gate_up_swiglu_128x32%s", suffix);
        gate_up128[layout] = optional_kernel(name);
        snprintf(name, sizeof(name), "euhedral_nvfp4_gate_up_swiglu_64x32%s", suffix);
        gate_up64[layout] = optional_kernel(name);
        euhedral_cuda_pdl_register(decode[layout]);
        for (int m = 2; m <= 8; ++m) {
            snprintf(name, sizeof(name), "euhedral_nvfp4_decode_rows%d%s", m, suffix);
            decode_rows[layout][m] = optional_kernel(name);
            euhedral_cuda_pdl_register(decode_rows[layout][m]);
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

int euhedral_cuda_linear_nvfp4_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size) {
    int layout;
    int status = prepare(input, weights, output, rows, in_features, out_features, weights_byte_size, &layout);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int rows_arg = rows, in_arg = in_features, out_arg = out_features;
    /* Row-exact (speculative verification): rows computed bit for bit as one-row decode, the weights
     * streamed once for up to 8 rows. */
    if (rows > 1u && euhedral_cuda_row_exact()) {
        if (rows <= 8u && in_features % 1024u == 0 && out_features % 16u == 0 && decode_rows[layout][rows] != NULL) {
            void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &in_arg, &out_arg};
            return finish(euhedral_launch_kernel(decode_rows[layout][rows], out_features / 16u, 1, 1, 128, 1, 1, 0,
                    euhedral_cuda_submission_stream(), params, NULL));
        }
        for (uint32_t row = 0; row < rows; ++row) {
            status = euhedral_cuda_linear_nvfp4_bf16((const char*)input + (uint64_t)row * in_features * 2u, weights,
                    (char*)output + (uint64_t)row * out_features * 2u, 1u, in_features, out_features, weights_byte_size);
            if (status != EUHEDRAL_CUDA_SUCCESS) return status;
        }
        return EUHEDRAL_CUDA_SUCCESS;
    }
    if (rows == 1u && in_features % 1024u == 0 && out_features % 16u == 0) {
        if (decode[layout] == NULL) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
        void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &in_arg, &out_arg};
        return finish(euhedral_launch_kernel(decode[layout], out_features / 16u, 1, 1, 128, 1, 1, 0,
                euhedral_cuda_submission_stream(), params, NULL));
    }
    CUfunction kernel = rows >= 128u && prefill128[layout] != NULL ? prefill128[layout] : prefill64[layout];
    if (kernel == NULL) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    uint64_t tile = kernel == prefill128[layout] ? 128u : 64u;
    uint64_t grid = ((uint64_t)rows + tile - 1u) / tile * (((uint64_t)out_features + 63u) / 64u);
    if (grid > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &rows_arg, &in_arg, &out_arg};
    return finish(euhedral_launch_kernel(kernel, (unsigned int)grid, 1, 1, 128, 1, 1, 0,
            euhedral_cuda_submission_stream(), params, NULL));
}

int euhedral_cuda_nvfp4_gate_up_swiglu_bf16(const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t width, uint32_t outputs, uint64_t weight_bytes) {
    int layout;
    int status = prepare(input, weights, output, rows, width, outputs, weight_bytes, &layout);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    if (outputs % 64u != 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    CUfunction kernel = rows >= 128u && gate_up128[layout] != NULL ? gate_up128[layout] : gate_up64[layout];
    if (kernel == NULL) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    uint64_t tile = kernel == gate_up128[layout] ? 128u : 64u;
    uint64_t grid = ((uint64_t)rows + tile - 1u) / tile * (outputs / 2u / 32u);
    if (grid > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    CUdeviceptr input_ptr = (CUdeviceptr)(uintptr_t)input, weights_ptr = (CUdeviceptr)(uintptr_t)weights;
    CUdeviceptr output_ptr = (CUdeviceptr)(uintptr_t)output;
    unsigned int rows_arg = rows, width_arg = width, outputs_arg = outputs;
    void* params[] = {&input_ptr, &weights_ptr, &output_ptr, &rows_arg, &width_arg, &outputs_arg};
    return finish(euhedral_launch_kernel(kernel, (unsigned int)grid, 1, 1, 128, 1, 1, 0,
            euhedral_cuda_submission_stream(), params, NULL));
}

/* Native Blackwell NVFP4 (native/src/nvfp4_native, docs/NVFP4_NATIVE.md): the BF16 activations are
 * quantized to NVFP4 into caller scratch, then multiplied with block-scaled FP4 tensor-core MMA
 * (OMMA.SF.16864.F32.E2M1.E2M1.UE4M3.4X). The module is compiled for this device's sm_12xa target and
 * exists only there; EUHEDRAL_NVFP4_NATIVE=0 selects the BF16-expansion kernels above instead. */
#define NATIVE_SHARED_BYTES 67584u  /* nvfp4n::Pipeline<1 or 2>::kSharedBytes */
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
/* Activation terms: 2 (the default) adds a quantized residual, about 1% error per linear, at the
 * relaxed-numerics floor of the drift harness; 1 (EUHEDRAL_NVFP4_NATIVE=1) quantizes once, about 10%,
 * doubling teacher-forced KL, for about 45% more prefill throughput. docs/NVFP4_NATIVE.md. */
static unsigned int native_terms = 2u;

static void initialize_native(void) {
    const char* selected = getenv("EUHEDRAL_NVFP4_NATIVE");
    if (selected != NULL && strcmp(selected, "0") == 0) return;
    native_terms = selected != NULL && strcmp(selected, "1") == 0 ? 1u : 2u;
    const int two = native_terms == 2u;
    int status = euhedral_cuda_load_native_kernel((const void*)&native_once, "nvfp4_native/kernels.cu",
            two ? "euhedral_nvfp4n_linear_x2_128x128" : "euhedral_nvfp4n_linear_128x128", &native_module, &native_linear[0]);
    if (status == EUHEDRAL_CUDA_SUCCESS
            && (cuModuleGetFunction(&native_quantize, native_module,
                        two ? "euhedral_nvfp4n_quantize_rows_x2" : "euhedral_nvfp4n_quantize_rows") != CUDA_SUCCESS
                    || cuModuleGetFunction(&native_finish, native_module, "euhedral_nvfp4n_skinny_finish") != CUDA_SUCCESS))
        status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    static const char* skinny_names[2][3] = {
            {"euhedral_nvfp4n_skinny_16", "euhedral_nvfp4n_skinny_32", "euhedral_nvfp4n_skinny_64"},
            {"euhedral_nvfp4n_skinny_x2_16", "euhedral_nvfp4n_skinny_x2_32", "euhedral_nvfp4n_skinny_x2_64"}};
    for (int layout = 0; layout < 2 && status == EUHEDRAL_CUDA_SUCCESS; ++layout) {
        const char* suffix = layout ? "_sd4" : "";
        char name[64];
        snprintf(name, sizeof(name), "%s%s", two ? "euhedral_nvfp4n_linear_x2_128x128" : "euhedral_nvfp4n_linear_128x128",
                suffix);
        if (cuModuleGetFunction(&native_linear[layout], native_module, name) != CUDA_SUCCESS) status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
        snprintf(name, sizeof(name), "%s%s",
                two ? "euhedral_nvfp4n_gate_up_swiglu_x2_128x64" : "euhedral_nvfp4n_gate_up_swiglu_128x64", suffix);
        if (status == EUHEDRAL_CUDA_SUCCESS && cuModuleGetFunction(&native_gate_up[layout], native_module, name) != CUDA_SUCCESS)
            status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
        for (int i = 0; i < 3 && status == EUHEDRAL_CUDA_SUCCESS; ++i) {
            snprintf(name, sizeof(name), "%s%s", skinny_names[two][i], suffix);
            if (cuModuleGetFunction(&native_skinny[layout][i], native_module, name) != CUDA_SUCCESS
                    || cuFuncSetAttribute(native_skinny[layout][i], CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
                               (int)skinny_shared_bytes(native_terms, 16u << i)) != CUDA_SUCCESS)
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

/* Activations of `rows` rows of `in_features` values in the selected number of terms: codes, a
 * 256-aligned scale plane and one FP32 global per row (nvfp4n::ActivationLayout). */
static uint64_t activation_bytes(uint32_t rows, uint32_t in_features) {
    uint64_t k = ((uint64_t)in_features + 127u) / 128u * 128u, planes = (uint64_t)rows * native_terms;
    uint64_t scales = align256(planes * k / 2u);
    return align256(scales + planes * k / 16u) + 4ull * rows;
}

/* Scratch for one native linear: the activations, then (split-K skinny shapes) FP32 partials. */
uint64_t euhedral_cuda_nvfp4_native_scratch_bytes(uint32_t rows, uint32_t in_features, uint32_t out_features) {
    uint64_t bytes = activation_bytes(rows, in_features);
    if (skinny_route(rows, in_features) && skinny_splits(in_features, out_features) > 1u)
        bytes = align256(bytes) + 4ull * skinny_splits(in_features, out_features) * rows * out_features;
    return bytes;
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
    // Activations are quantized, so exact numerics keep the BF16-expansion kernels (the reference twin).
    if (in_features % 128u != 0 || euhedral_cuda_exact_numerics()) return EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
    status = prepare(input, weights, output, rows, in_features, out_features, weights_byte_size, &layout);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
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
                128, 1, 1, skinny_shared_bytes(native_terms, 16u << index), stream, skinny_params, NULL);
        if (result != CUDA_SUCCESS || splits == 1u) return finish(result);
        unsigned int count = rows * out_features, splits_arg = splits;
        void* finish_params[] = {&partials, &output_ptr, &count, &splits_arg};
        return finish(euhedral_launch_kernel(native_finish, (count + 255u) / 256u, 1, 1, 256, 1, 1, 0, stream,
                finish_params, NULL));
    }
    void* linear_params[] = {&scratch_ptr, &weights_ptr, &output_ptr, &rows_arg, &in_arg, &out_arg};
    return finish(euhedral_launch_kernel(kernel, (unsigned int)grid, 1, 1, 256, 1, 1, NATIVE_SHARED_BYTES,
            stream, linear_params, NULL));
}
