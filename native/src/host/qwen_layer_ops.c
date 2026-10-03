#include "cuda_kernel_loader.h"
#include "euhedral_cuda.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#else
#include <pthread.h>
#endif

/* The layer operators besides the quantized linears: GDN, attention over the NVFP4 KV cache, residual and norm
 * elementwise kernels, SwiGLU and greedy selection. Every kernel here is required; a module that fails to
 * load fails the first call with EUHEDRAL_CUDA_KERNEL_UNAVAILABLE. */
#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static int linear_anchor;
static int gdn_anchor;
static int elementwise_anchor;
static int sampling_anchor;
static int attention_anchor;
static CUmodule linear_module;
static CUmodule gdn_module;
static CUmodule elementwise_module;
static CUmodule attention_module;
static CUmodule sampling_module;
static CUfunction linear_bf16_to_float;
static CUfunction gdn_control;
static CUfunction gdn_convolution;
static CUfunction gdn_recurrence, gdn_recurrence_c8;
static CUfunction gdn_gated_rms_norm;
static CUfunction gdn_project_control, gdn_project_control_tiled;
static CUfunction residual_add;
static CUfunction argmax_bf16;
static CUfunction residual_rms_norm, residual_rms_norm_row;
static CUfunction swiglu;
static CUfunction attention_qk_norm_rope;
static CUfunction attention_qk_norm_rope_rows;
static CUfunction attention_append_nvfp4, attention_prefill_nvfp4, attention_decode_nvfp4, attention_merge_nvfp4;
/* fa2::kSharedBytes of native/src/attention/nvfp4_prefill_fa2.cuh: the K and V tiles, double buffered. */
#define FA2_SHARED_BYTES 67584u
static CUfunction attention_prefill_nvfp4_exact, attention_decode_nvfp4_exact, attention_prefill_fa2, attention_decode_gqa;
/* Multi-row twins of decode attention and its merge: row j is one-row decode at its own position. */
static CUfunction attention_decode_rows, attention_decode_gqa_rows, attention_merge_rows;
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

static int get_function(CUmodule module, CUfunction* function, const char* name) {
    return (int)cuModuleGetFunction(function, module, name);
}

/* Loads every kernel of a module; the first failure is the status. */
static int load_functions(CUmodule module, const char* const* names, CUfunction* const* functions, int count) {
    for (int index = 0; index < count; index++) {
        int status = get_function(module, functions[index], names[index]);
        if (status != CUDA_SUCCESS) return status;
    }
    return CUDA_SUCCESS;
}

static void initialize_modules(void) {
    init_status = euhedral_cuda_load_kernel(&linear_anchor, "linear/kernels.cu", "euhedral_linear_bf16_to_float",
            &linear_module, &linear_bf16_to_float);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;

    init_status = euhedral_cuda_load_kernel(&gdn_anchor, "gdn/kernels.cu", "euhedral_gdn_control_fp32", &gdn_module, &gdn_control);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    {
        const char* const names[] = {"euhedral_gdn_project_control_fp32", "euhedral_gdn_project_control_8x4_fp32",
                "euhedral_gdn_convolution_bf16", "euhedral_gdn_recurrence_bf16", "euhedral_gdn_recurrence_c8_bf16",
                "euhedral_gdn_gated_rms_norm_bf16"};
        CUfunction* const functions[] = {&gdn_project_control, &gdn_project_control_tiled, &gdn_convolution,
                &gdn_recurrence, &gdn_recurrence_c8, &gdn_gated_rms_norm};
        init_status = load_functions(gdn_module, names, functions, 6);
        if (init_status != CUDA_SUCCESS) return;
    }

    init_status = euhedral_cuda_load_kernel(&elementwise_anchor, "elementwise/kernels.cu", "euhedral_residual_add_bf16",
            &elementwise_module, &residual_add);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    {
        const char* const names[] = {"euhedral_residual_rms_norm_bf16", "euhedral_residual_rms_norm_row_bf16", "euhedral_swiglu_bf16"};
        CUfunction* const functions[] = {&residual_rms_norm, &residual_rms_norm_row, &swiglu};
        init_status = load_functions(elementwise_module, names, functions, 3);
        if (init_status != CUDA_SUCCESS) return;
    }
    init_status = euhedral_cuda_load_kernel(&sampling_anchor, "sampling/kernels.cu", "euhedral_argmax_bf16", &sampling_module, &argmax_bf16);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;

    init_status = euhedral_cuda_load_kernel(&attention_anchor, "attention/kernels.cu", "euhedral_attention_qk_norm_rope_bf16",
            &attention_module, &attention_qk_norm_rope);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    {
        const char* const names[] = {"euhedral_attention_qk_norm_rope_rows_bf16", "euhedral_attention_kv_append_nvfp4",
                "euhedral_attention_prefill32_nvfp4", "euhedral_attention_prefill32_nvfp4_exact",
                "euhedral_attention_prefill_fa2_nvfp4", "euhedral_attention_decode_gqa_nvfp4",
                "euhedral_attention_decode_nvfp4", "euhedral_attention_decode_nvfp4_exact",
                "euhedral_attention_merge_nvfp4", "euhedral_attention_decode_nvfp4_rows",
                "euhedral_attention_decode_gqa_nvfp4_rows", "euhedral_attention_merge_nvfp4_rows"};
        CUfunction* const functions[] = {&attention_qk_norm_rope_rows, &attention_append_nvfp4, &attention_prefill_nvfp4,
                &attention_prefill_nvfp4_exact, &attention_prefill_fa2, &attention_decode_gqa, &attention_decode_nvfp4,
                &attention_decode_nvfp4_exact, &attention_merge_nvfp4, &attention_decode_rows,
                &attention_decode_gqa_rows, &attention_merge_rows};
        init_status = load_functions(attention_module, names, functions, 12);
        if (init_status != CUDA_SUCCESS) return;
    }
    if (cuFuncSetAttribute(attention_prefill_fa2, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, (int)FA2_SHARED_BYTES) != CUDA_SUCCESS)
        init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
}

// Every kernel registered here begins with euhedral_pdl_begin(), so stream-serialized
// programmatic dependent launch is safe for it (see cuda_kernel_loader.h).
static void initialize(void) {
    initialize_modules();
    euhedral_cuda_pdl_register(linear_bf16_to_float);
    euhedral_cuda_pdl_register(gdn_control);
    euhedral_cuda_pdl_register(argmax_bf16);
    euhedral_cuda_pdl_register(gdn_project_control);
    euhedral_cuda_pdl_register(residual_rms_norm);
    euhedral_cuda_pdl_register(residual_rms_norm_row);
    euhedral_cuda_pdl_register(gdn_convolution);
    euhedral_cuda_pdl_register(gdn_recurrence);
    euhedral_cuda_pdl_register(gdn_recurrence_c8);
    euhedral_cuda_pdl_register(gdn_gated_rms_norm);
    euhedral_cuda_pdl_register(residual_add);
    euhedral_cuda_pdl_register(swiglu);
    euhedral_cuda_pdl_register(attention_qk_norm_rope);
    euhedral_cuda_pdl_register(attention_qk_norm_rope_rows);
    euhedral_cuda_pdl_register(attention_append_nvfp4);
    euhedral_cuda_pdl_register(attention_decode_nvfp4);
    euhedral_cuda_pdl_register(attention_decode_nvfp4_exact);
    euhedral_cuda_pdl_register(attention_decode_gqa);
    euhedral_cuda_pdl_register(attention_merge_nvfp4);
    euhedral_cuda_pdl_register(attention_decode_rows);
    euhedral_cuda_pdl_register(attention_decode_gqa_rows);
    euhedral_cuda_pdl_register(attention_merge_rows);
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

static int launch_and_synchronize(CUfunction function, uint32_t grid_x, uint32_t block_x, void** parameters) {
    CUresult status = euhedral_launch_kernel(function, grid_x, 1, 1, block_x, 1, 1, 0,
            euhedral_cuda_submission_stream(), parameters, NULL);
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

static int launch_and_synchronize_shared(CUfunction function, uint32_t grid_x, uint32_t block_x, uint32_t shared_bytes,
        void** parameters) {
    CUresult status = euhedral_launch_kernel(function, grid_x, 1, 1, block_x, 1, 1, shared_bytes,
            euhedral_cuda_submission_stream(), parameters, NULL);
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

static int launch_and_synchronize_2d(CUfunction function, uint32_t grid_x, uint32_t grid_y, uint32_t block_x,
        void** parameters) {
    CUresult status = euhedral_launch_kernel(function, grid_x, grid_y, 1, block_x, 1, 1, 0,
            euhedral_cuda_submission_stream(), parameters, NULL);
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

int euhedral_cuda_linear_bf16_to_float(
        const void* device_input, const void* device_weights, void* device_output,
        uint32_t rows, uint32_t in_features, uint32_t out_features) {
    if (device_input == NULL || device_weights == NULL || device_output == NULL || rows == 0 || in_features == 0 || out_features == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * out_features;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr input = (CUdeviceptr)(uintptr_t)device_input;
    CUdeviceptr weights = (CUdeviceptr)(uintptr_t)device_weights;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, in_arg = in_features, out_arg = out_features;
    void* parameters[] = {&input, &weights, &output, &rows_arg, &in_arg, &out_arg};
    return launch_and_synchronize(linear_bf16_to_float, (uint32_t)count, 128, parameters);
}

int euhedral_cuda_gdn_control_fp32(
        const float* device_a_projection, const float* device_b_projection,
        const float* device_a_log, const float* device_dt_bias,
        float* device_alpha_output, float* device_beta_output, uint32_t rows, uint32_t heads) {
    if (device_a_projection == NULL || device_b_projection == NULL || device_a_log == NULL || device_dt_bias == NULL
            || device_alpha_output == NULL || device_beta_output == NULL || rows == 0 || heads == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * heads;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr a = (CUdeviceptr)(uintptr_t)device_a_projection;
    CUdeviceptr b = (CUdeviceptr)(uintptr_t)device_b_projection;
    CUdeviceptr a_log = (CUdeviceptr)(uintptr_t)device_a_log;
    CUdeviceptr dt_bias = (CUdeviceptr)(uintptr_t)device_dt_bias;
    CUdeviceptr alpha = (CUdeviceptr)(uintptr_t)device_alpha_output;
    CUdeviceptr beta = (CUdeviceptr)(uintptr_t)device_beta_output;
    uint32_t rows_arg = rows, heads_arg = heads;
    void* parameters[] = {&a, &b, &a_log, &dt_bias, &alpha, &beta, &rows_arg, &heads_arg};
    return launch_and_synchronize(gdn_control, (uint32_t)((count + 127) / 128), 128, parameters);
}

int euhedral_cuda_gdn_convolution_bf16(
        const void* device_query_key, const void* device_value_z, const void* device_convolution_weights,
        void* device_convolution_state, void* device_output, uint32_t rows,
        uint32_t query_key_width, uint32_t value_width, uint32_t convolution_width, uint32_t kernel_size) {
    if (device_query_key == NULL || device_value_z == NULL || device_convolution_weights == NULL
            || device_convolution_state == NULL || device_output == NULL || rows == 0 || query_key_width == 0
            || value_width == 0 || convolution_width != query_key_width + value_width || kernel_size < 2 || kernel_size > 32)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr qk = (CUdeviceptr)(uintptr_t)device_query_key;
    CUdeviceptr vz = (CUdeviceptr)(uintptr_t)device_value_z;
    CUdeviceptr weights = (CUdeviceptr)(uintptr_t)device_convolution_weights;
    CUdeviceptr state = (CUdeviceptr)(uintptr_t)device_convolution_state;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, qk_arg = query_key_width, value_arg = value_width;
    uint32_t channels_arg = convolution_width, kernel_arg = kernel_size;
    void* parameters[] = {&qk, &vz, &weights, &state, &output, &rows_arg, &qk_arg, &value_arg, &channels_arg, &kernel_arg};
    // Kernel rows are split into 32-row blocks (QWEN_GDN_CONV_ROWS in gdn/kernels.cu).
    return launch_and_synchronize_2d(gdn_convolution, (convolution_width + 127) / 128, (rows + 31) / 32, 128, parameters);
}

int euhedral_cuda_gdn_recurrence_bf16(
        const void* device_convolved, const float* device_alpha, const float* device_beta,
        float* device_recurrent_state, void* device_output, uint32_t rows,
        uint32_t key_heads, uint32_t value_heads, uint32_t key_head_dim,
        uint32_t value_head_dim, float output_scale) {
    if (device_convolved == NULL || device_alpha == NULL || device_beta == NULL || device_recurrent_state == NULL
            || device_output == NULL || rows == 0 || key_heads == 0 || value_heads == 0
            || value_heads % key_heads != 0 || key_head_dim != 128 || value_head_dim != 128
            || !isfinite(output_scale) || output_scale <= 0.0f)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t rows_count = (uint64_t)value_heads * value_head_dim;
    if (rows_count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr convolved = (CUdeviceptr)(uintptr_t)device_convolved;
    CUdeviceptr alpha = (CUdeviceptr)(uintptr_t)device_alpha;
    CUdeviceptr beta = (CUdeviceptr)(uintptr_t)device_beta;
    CUdeviceptr state = (CUdeviceptr)(uintptr_t)device_recurrent_state;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, key_heads_arg = key_heads, value_heads_arg = value_heads;
    uint32_t key_dim_arg = key_head_dim, value_dim_arg = value_head_dim;
    void* parameters[] = {&convolved, &alpha, &beta, &state, &output, &rows_arg,
            &key_heads_arg, &value_heads_arg, &key_dim_arg, &value_dim_arg, &output_scale};
    // Prefill (two or more rows) on column-owned lanes, eight value columns per warp (16-byte aligned rows and
    // state); one-row decode keeps the warp-tree kernel, which measured as fast in the model. Row-exact
    // execution selects the kernel one-row decode would. Both kernels run rows in order with the state in FP32
    // registers, so M rows in one launch equal M one-row launches bit for bit.
    const int aligned = (((uintptr_t)device_convolved | (uintptr_t)device_recurrent_state) & 15u) == 0u;
    if (!euhedral_cuda_exact_numerics() && aligned && rows > 1 && !euhedral_cuda_row_exact())
        return launch_and_synchronize(gdn_recurrence_c8, (uint32_t)(rows_count / 8), 32, parameters);
    // Must match QWEN_GDN_WARP_COLUMNS (8 value columns per warp) in gdn/kernels.cu.
    return launch_and_synchronize(gdn_recurrence, (uint32_t)(rows_count / 8), 32, parameters);
}

int euhedral_cuda_gdn_gated_rms_norm_bf16(
        const void* device_recurrent, const void* device_value_z, const void* device_norm_weight,
        void* device_output, uint32_t rows, uint32_t value_heads, uint32_t head_dim, float epsilon) {
    if (device_recurrent == NULL || device_value_z == NULL || device_norm_weight == NULL || device_output == NULL
            || rows == 0 || value_heads == 0 || head_dim != 128 || !isfinite(epsilon) || epsilon < 0.0f)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * value_heads;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr recurrent = (CUdeviceptr)(uintptr_t)device_recurrent;
    CUdeviceptr value_z = (CUdeviceptr)(uintptr_t)device_value_z;
    CUdeviceptr weight = (CUdeviceptr)(uintptr_t)device_norm_weight;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, heads_arg = value_heads, dim_arg = head_dim;
    void* parameters[] = {&recurrent, &value_z, &weight, &output, &rows_arg, &heads_arg, &dim_arg, &epsilon};
    return launch_and_synchronize(gdn_gated_rms_norm, (uint32_t)count, 128, parameters);
}

int euhedral_cuda_residual_rms_norm_bf16(
        const void* residual, const void* delta, const void* weight,
        void* hidden, void* normalized, uint32_t rows, uint32_t width, float epsilon) {
    // Row-exact: where one row runs the row-owned kernel (one CTA per row, blockIdx.x is the row), a
    // rows-CTA launch of it computes every row exactly as one-row launches would.
    const int row_exact = rows > 1 && euhedral_cuda_row_exact();
    const int row_kernel = residual && delta && weight && hidden && normalized && width % 8u == 0u && width <= 8192u
            && !euhedral_cuda_exact_numerics() && (((uintptr_t)residual | (uintptr_t)delta | (uintptr_t)weight
                    | (uintptr_t)hidden | (uintptr_t)normalized) & 15u) == 0u;
    if (row_exact && !row_kernel && residual && delta && hidden && normalized) {
        const uint64_t stride = (uint64_t)width * 2u;
        for (uint32_t row = 0; row < rows; ++row) {
            int status = euhedral_cuda_residual_rms_norm_bf16((const char*)residual + row * stride, (const char*)delta + row * stride,
                    weight, (char*)hidden + row * stride, (char*)normalized + row * stride, 1u, width, epsilon);
            if (status != 0) return status;
        }
        return 0;
    }
    if (!residual || !delta || !weight || !hidden || !normalized || rows == 0 || width == 0
            || !isfinite(epsilon) || epsilon < 0.0f) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    if (rows > INT32_MAX || (uint64_t)rows * width > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    void* parameters[] = {&residual, &delta, &weight, &hidden, &normalized, &rows, &width, &epsilon};
    // Relaxed numerics: one row (decode) keeps its columns in registers, eight per thread.
    if ((rows == 1 || row_exact) && row_kernel)
        return launch_and_synchronize(residual_rms_norm_row, rows, (width / 8u + 31u) / 32u * 32u, parameters);
    if (row_exact) {
        const uint64_t stride = (uint64_t)width * 2u;
        for (uint32_t row = 0; row < rows; ++row) {
            status = euhedral_cuda_residual_rms_norm_bf16((const char*)residual + row * stride, (const char*)delta + row * stride,
                    weight, (char*)hidden + row * stride, (char*)normalized + row * stride, 1u, width, epsilon);
            if (status != 0) return status;
        }
        return 0;
    }
    return launch_and_synchronize(residual_rms_norm, rows, 128, parameters);
}

int euhedral_cuda_gdn_project_control_fp32(
        const void* input, const void* a_weight, const void* b_weight,
        const float* a_log, const float* dt_bias, float* alpha, float* beta,
        uint32_t rows, uint32_t width, uint32_t heads) {
    if (!input || !a_weight || !b_weight || !a_log || !dt_bias || !alpha || !beta
            || rows == 0 || width == 0 || heads == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * heads;
    if (count > INT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    void* parameters[] = {&input, &a_weight, &b_weight, &a_log, &dt_bias, &alpha, &beta, &rows, &width, &heads};
    // From eight rows, 8-row x 4-head CTAs share activation and weight loads (bitwise equal).
    if (rows >= 8u && heads % 4u == 0u)
        return launch_and_synchronize(gdn_project_control_tiled, (rows + 7u) / 8u * (heads / 4u), 128, parameters);
    return launch_and_synchronize(gdn_project_control, (uint32_t)count, 128, parameters);
}

int euhedral_cuda_residual_add_bf16(
        const void* device_residual, const void* device_delta, void* device_output, uint32_t rows, uint32_t width) {
    if (device_residual == NULL || device_delta == NULL || device_output == NULL || rows == 0 || width == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * width;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr residual = (CUdeviceptr)(uintptr_t)device_residual;
    CUdeviceptr delta = (CUdeviceptr)(uintptr_t)device_delta;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t count_arg = (uint32_t)count;
    void* parameters[] = {&residual, &delta, &output, &count_arg};
    return launch_and_synchronize(residual_add, (uint32_t)((count + 255) / 256), 256, parameters);
}

int euhedral_cuda_argmax_bf16(const void* device_logits, uint32_t count, void* device_result) {
    if (device_logits == NULL || device_result == NULL || count == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr logits = (CUdeviceptr)(uintptr_t)device_logits;
    CUdeviceptr result = (CUdeviceptr)(uintptr_t)device_result;
    void* parameters[] = {&logits, &count, &result};
    return launch_and_synchronize(argmax_bf16, 1, 1024, parameters);
}

int euhedral_cuda_swiglu_bf16(
        const void* device_gate_up, void* device_output, uint32_t rows, uint32_t intermediate_size) {
    if (device_gate_up == NULL || device_output == NULL || rows == 0 || intermediate_size == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * intermediate_size;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr gate_up = (CUdeviceptr)(uintptr_t)device_gate_up;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, width_arg = intermediate_size;
    void* parameters[] = {&gate_up, &output, &rows_arg, &width_arg};
    return launch_and_synchronize(swiglu, (uint32_t)((count + 255) / 256), 256, parameters);
}

int euhedral_cuda_zero_device_memory(void* device_address, uint64_t byte_size) {
    if (device_address == NULL || byte_size == 0) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    cudaStream_t stream = euhedral_cuda_submission_stream();
    if (stream != NULL) {
        cudaError_t result = cudaMemsetAsync(device_address, 0, (size_t)byte_size, stream);
        return result == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)result;
    }
    cudaError_t result = cudaMemset(device_address, 0, (size_t)byte_size);
    if (result != cudaSuccess) return (int)result;
    result = cudaDeviceSynchronize();
    return result == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)result;
}

int euhedral_cuda_attention_kv_append_nvfp4(
        const void* device_query_key,
        const void* device_gate_value,
        void* device_key_cache,
        void* device_value_cache,
        uint32_t rows,
        uint32_t query_width,
        uint32_t key_value_width,
        uint64_t start_position,
        const void* device_position) {
    if (device_query_key == NULL || device_gate_value == NULL || device_key_cache == NULL || device_value_cache == NULL
            || device_position == NULL || rows == 0 || query_width == 0 || key_value_width == 0 || query_width % key_value_width != 0
            || key_value_width % 256 != 0 || query_width % 256 != 0
            || start_position > UINT32_MAX - (uint64_t)rows)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t count = (uint64_t)rows * key_value_width;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr query_key = (CUdeviceptr)(uintptr_t)device_query_key;
    CUdeviceptr gate_value = (CUdeviceptr)(uintptr_t)device_gate_value;
    CUdeviceptr key_cache = (CUdeviceptr)(uintptr_t)device_key_cache;
    CUdeviceptr value_cache = (CUdeviceptr)(uintptr_t)device_value_cache;
    CUdeviceptr position = (CUdeviceptr)(uintptr_t)device_position;
    uint32_t rows_arg = rows, query_width_arg = query_width, key_value_width_arg = key_value_width;
    void* parameters[] = {&query_key, &gate_value, &key_cache, &value_cache,
            &rows_arg, &query_width_arg, &key_value_width_arg, &position};
    return launch_and_synchronize(attention_append_nvfp4, (uint32_t)((count / 256 + 3) / 4), 128, parameters);
}

static int qk_norm_rope(const void* device_query_key, const void* device_query_norm, const void* device_key_norm,
        void* device_output, uint32_t rows, uint32_t query_heads, uint32_t key_value_heads, uint32_t head_dim,
        uint32_t rotary_dim, uint64_t start_position, const void* device_position, uint64_t position_offset,
        float epsilon, double rope_theta);

int euhedral_cuda_attention_qk_norm_rope_bf16(
        const void* device_query_key,
        const void* device_query_norm,
        const void* device_key_norm,
        void* device_output,
        uint32_t rows,
        uint32_t query_heads,
        uint32_t key_value_heads,
        uint32_t head_dim,
        uint32_t rotary_dim,
        uint64_t start_position,
        const void* device_position,
        float epsilon,
        double rope_theta) {
    return qk_norm_rope(device_query_key, device_query_norm, device_key_norm, device_output, rows, query_heads,
            key_value_heads, head_dim, rotary_dim, start_position, device_position, 0, epsilon, rope_theta);
}

/* `start_position` is the host's copy of the quantum's position, which the kernels read from `device_position`;
 * `position_offset` is this call's first row within the quantum. */
static int qk_norm_rope(const void* device_query_key, const void* device_query_norm, const void* device_key_norm,
        void* device_output, uint32_t rows, uint32_t query_heads, uint32_t key_value_heads, uint32_t head_dim,
        uint32_t rotary_dim, uint64_t start_position, const void* device_position, uint64_t position_offset,
        float epsilon, double rope_theta) {
    // Row-exact: the one-row kernel already maps block -> (row, head) with position start + row, so a
    // rows x heads launch of it computes every row exactly as one-row launches would. The in-place query
    // path reads and writes only its own (row, head) block.
    const int row_exact_heads = rows > 1 && euhedral_cuda_row_exact();
    if (row_exact_heads && device_query_key != NULL && device_output != NULL
            && ((uint64_t)rows * ((uint64_t)query_heads + key_value_heads) > UINT32_MAX)) {
        const uint64_t width = ((uint64_t)query_heads + key_value_heads) * head_dim * 2u;
        for (uint32_t row = 0; row < rows; ++row) {
            int status = qk_norm_rope((const char*)device_query_key + row * width, device_query_norm,
                    device_key_norm, (char*)device_output + row * width, 1u, query_heads, key_value_heads, head_dim,
                    rotary_dim, start_position + row, device_position, position_offset + row, epsilon, rope_theta);
            if (status != 0) return status;
        }
        return 0;
    }
    if (device_query_key == NULL || device_query_norm == NULL || device_key_norm == NULL || device_output == NULL
            || device_position == NULL || rows == 0 || query_heads == 0 || key_value_heads == 0 || query_heads % key_value_heads != 0
            || head_dim != 256 || rotary_dim == 0 || rotary_dim > head_dim || (rotary_dim & 1) != 0
            || start_position > UINT64_MAX - rows || !isfinite(epsilon) || epsilon <= 0.0f
            || !isfinite(rope_theta) || rope_theta <= 0.0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t heads = (uint64_t)query_heads + key_value_heads;
    const uint64_t blocks = (uint64_t)rows * heads;
    if (heads > UINT32_MAX || blocks > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr query_key = (CUdeviceptr)(uintptr_t)device_query_key;
    CUdeviceptr query_norm = (CUdeviceptr)(uintptr_t)device_query_norm;
    CUdeviceptr key_norm = (CUdeviceptr)(uintptr_t)device_key_norm;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, query_heads_arg = query_heads, key_value_heads_arg = key_value_heads;
    CUdeviceptr position = (CUdeviceptr)(uintptr_t)device_position;
    uint32_t head_dim_arg = head_dim, rotary_dim_arg = rotary_dim;
    void* parameters[] = {&query_key, &query_norm, &key_norm, &output, &rows_arg, &query_heads_arg,
            &key_value_heads_arg, &head_dim_arg, &rotary_dim_arg, &position, &position_offset, &epsilon, &rope_theta};
    // A single row (decode) keeps one CTA per head; from two rows one CTA per row shares the angles.
    if (rows > 1 && !row_exact_heads)
        return launch_and_synchronize(attention_qk_norm_rope_rows, rows, 256, parameters);
    return launch_and_synchronize(attention_qk_norm_rope, (uint32_t)blocks, head_dim, parameters);
}

static int attention_causal(const void* query_key, const void* gate, const void* keys, const void* values, void* output,
        uint32_t rows, uint32_t query_heads, uint32_t key_heads, uint32_t head_dim, uint32_t cache_length, uint64_t start,
        const void* device_position, uint64_t position_offset, void* scratch);

int euhedral_cuda_attention_causal_nvfp4(
        const void* query_key, const void* gate, const void* keys, const void* values, void* output,
        uint32_t rows, uint32_t query_heads, uint32_t key_heads, uint32_t head_dim,
        uint32_t cache_length, uint64_t start, const void* device_position, void* scratch) {
    if (device_position == NULL) return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    return attention_causal(query_key, gate, keys, values, output, rows, query_heads, key_heads, head_dim, cache_length,
            start, device_position, 0, scratch);
}

/* Decode kernels read the quantum's position from `device_position` (plus `position_offset`, this call's first row
 * within the quantum); `start` is the host's copy, which sizes the grids. Prefill kernels take `start` by value. */
static int attention_causal(const void* query_key, const void* gate, const void* keys, const void* values, void* output,
        uint32_t rows, uint32_t query_heads, uint32_t key_heads, uint32_t head_dim, uint32_t cache_length, uint64_t start,
        const void* device_position, uint64_t position_offset, void* scratch) {
    // The row twins also serve small non-row-exact quanta that bring decode scratch (speculative drafting):
    // a 32-row prefill tile over a long cache is one CTA per head, about 9.5 ms over 32K keys for 2-4 rows
    // against 0.45-0.75 ms for the twins.
    if (rows > 1 && (euhedral_cuda_row_exact() || rows <= 8u) && query_key && gate && keys && values && output && scratch
            && query_heads && key_heads && query_heads % key_heads == 0 && head_dim == 256 && rows <= 64u
            && start <= cache_length && rows <= cache_length - start && start + rows <= UINT32_MAX
            && position_offset == 0 && !euhedral_cuda_exact_numerics()) {
        /* Row-exact twins: row j is one-row decode at position start + j, all rows in one launch each for
         * decode and merge. Rows use the one-row kernel choice at their own length: the GQA kernel from
         * 2048 keys when the group fits, else the per-head kernel. The scratch holds one
         * one-row partial area (query_heads * 64 * 258 floats) per row. */
        int status = euhedral_cuda_bind_thread_context();
        if (status != 0) return status;
        status = ensure_initialized();
        if (status != 0) return status;
        const uint32_t group = query_heads / key_heads;
        const uint32_t from = group <= 8u ? 2048u : UINT32_MAX;
        {
            const uint32_t first = (uint32_t)start + 1u, last = (uint32_t)start + rows;
            unsigned long long row_stride = (unsigned long long)query_heads * 64u * 258u;
            uint32_t qh = query_heads, kh = key_heads, from_arg = from;
            CUdeviceptr position = (CUdeviceptr)(uintptr_t)device_position;
            if (first < from) {
                uint32_t below = last < from ? last : from - 1u, splits = (below + 47u) / 48u;
                if (splits > 64u) splits = 64u;
                void* args[] = {&query_key, &keys, &values, &qh, &kh, &position, &scratch, &row_stride, &from_arg};
                status = launch_and_synchronize_2d(attention_decode_rows, query_heads * splits, rows, 128, args);
                if (status != 0) return status;
            }
            if (last >= from) {
                uint32_t splits = (last + 31u) / 32u;
                if (splits > 64u) splits = 64u;
                void* args[] = {&query_key, &keys, &values, &qh, &kh, &position, &scratch, &row_stride, &from_arg};
                status = launch_and_synchronize_2d(attention_decode_gqa_rows, key_heads * splits, rows, 96, args);
                if (status != 0) return status;
            }
            void* merge_args[] = {&gate, &output, &scratch, &qh, &kh, &position, &row_stride, &from_arg};
            return launch_and_synchronize_2d(attention_merge_rows, query_heads, rows, 128, merge_args);
        }
    }
    if (rows > 1 && euhedral_cuda_row_exact() && query_key && gate && output) {
        /* Row-exact: row j is one-row decode at position start + j over the keys up to it. */
        const uint64_t width = ((uint64_t)query_heads + key_heads) * head_dim * 2u, out = (uint64_t)query_heads * head_dim * 2u;
        for (uint32_t row = 0; row < rows; ++row) {
            int status = attention_causal((const char*)query_key + row * width, (const char*)gate + row * width, keys, values,
                    (char*)output + row * out, 1u, query_heads, key_heads, head_dim, cache_length, start + row, device_position,
                    position_offset + row, scratch);
            if (status != 0) return status;
        }
        return 0;
    }
    if (!query_key || !gate || !keys || !values || !output || !rows || !query_heads || !key_heads
            || query_heads % key_heads != 0 || head_dim != 256 || !cache_length
            || start > cache_length || rows > cache_length - start || (rows == 1 && !scratch))
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    uint64_t grid = ((uint64_t)rows + 31) / 32 * query_heads;
    if (grid > UINT32_MAX || (uint64_t)query_heads * 64 > UINT32_MAX)
        return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != 0) return status;
    status = ensure_initialized();
    if (status != 0) return status;
    if (rows > 1) {
        void* args[] = {&query_key, &gate, &keys, &values, &output, &rows, &query_heads,
                &key_heads, &head_dim, &cache_length, &start};
        // Exact numerics keep the key-ordered softmax sum.
        int exact = euhedral_cuda_exact_numerics();
        // Relaxed numerics: the FlashAttention-2 leaf (16 query rows x one KV head's query-head group per
        // CTA) wherever its grid carries enough work: 512 rows after 3584 keys 3733 -> 1481 us, 128 rows
        // after 3968 keys 1252 -> 774 us; on small grids (64-256 rows, short context) the 32-row tile wins.
        uint32_t group = query_heads / key_heads;
        if (!exact && group >= 1u && group <= 6u
                && (rows >= 512u || (rows >= 128u && (uint64_t)cache_length >= 2048u))) {
            uint64_t fa2_grid = ((uint64_t)rows + 15u) / 16u * key_heads;
            if (fa2_grid <= UINT32_MAX)
                return launch_and_synchronize_shared(attention_prefill_fa2, (uint32_t)fa2_grid, 32u * (group + 2u), FA2_SHARED_BYTES, args);
        }
        CUfunction prefill = exact ? attention_prefill_nvfp4_exact : attention_prefill_nvfp4;
        return launch_and_synchronize(prefill, (uint32_t)grid, 128, args);
    }
    // Decode must only attend the prefix ending at its query, even when a caller
    // supplies a longer physical cache. Scratch reserves 64 * heads * 258 floats.
    // Each CTA scans about 48 keys with one warp per key stream, so the visible prefix spreads over
    // hundreds of CTAs; longer splits leave most SMs idle, shorter ones cost more in the merge.
    uint32_t length = (uint32_t)start + 1;
    uint32_t splits = (uint32_t)(((uint64_t)length + 47) / 48);
    if (splits > 64) splits = 64;
    CUstream stream = euhedral_cuda_submission_stream();
    CUdeviceptr position = (CUdeviceptr)(uintptr_t)device_position;
    unsigned long long offset_arg = position_offset;
    void* args[] = {&query_key, &gate, &keys, &values, &output, &rows, &query_heads,
            &key_heads, &head_dim, &position, &offset_arg, &scratch, &splits};
    void* merge[] = {&gate, &output, &scratch, &query_heads, &key_heads, &splits};
    // Exact numerics keep the per-element (lane + 32 d) kernel.
    int exact = euhedral_cuda_exact_numerics();
    CUfunction decode = exact ? attention_decode_nvfp4_exact : attention_decode_nvfp4;
    uint32_t decode_grid = query_heads * splits, decode_block = 128;
    // Relaxed numerics from 2048 keys: one three-warp tensor-core CTA (96 threads) per (KV head, 32-key split, at most 64)
    // serves the KV head's whole query-head group (docs/ATTENTION_DECODE.md). Shorter contexts keep the per-query-head
    // kernel, which measured faster in decode at 1024 keys.
    uint32_t group = query_heads / key_heads;
    if (!exact && length >= 2048u && group <= 8u) {
        splits = (uint32_t)(((uint64_t)length + 31) / 32);
        if (splits > 64) splits = 64;
        decode = attention_decode_gqa;
        decode_grid = key_heads * splits;
        decode_block = 96;
    }
    status = (int)euhedral_launch_kernel(decode, decode_grid, 1, 1,
            decode_block, 1, 1, 0, stream, args, NULL);
    if (status == 0) status = (int)euhedral_launch_kernel(attention_merge_nvfp4, query_heads, 1, 1,
            128, 1, 1, 0, stream, merge, NULL);
    if (status != 0 || stream == NULL) {
        int drained = (int)cudaStreamSynchronize((cudaStream_t)stream);
        if (status == 0) status = drained;
    }
    return status;
}
