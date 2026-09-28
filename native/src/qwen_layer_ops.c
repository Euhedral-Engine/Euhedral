#include "cuda_kernel_loader.h"
#include "euhedral_cuda.h"
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
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
static int quantized_anchor;
static int q45_anchor;
static int gdn_anchor;
static int elementwise_anchor;
static int attention_anchor;
static CUmodule quantized_module;
static CUmodule q45_module;
static CUmodule gdn_module;
static CUmodule elementwise_module;
static CUmodule attention_module;
static CUfunction linear_quantized;
// Q4/Q5 kernels indexed by [bits == 5]: cooperative decode for 1, 2 and 4
// token rows per CTA, and 32- and 64-row prefill tiles.
static CUfunction q45_decode[2][3];
static CUfunction q45_prefill[2];
static CUfunction q45_prefill64[2];
static int q45_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
static CUfunction linear_bf16_to_float;
static CUfunction gdn_control;
static CUfunction gdn_convolution;
static CUfunction gdn_recurrence;
static CUfunction gdn_gated_rms_norm;
static CUfunction residual_add;
static CUfunction swiglu;
static CUfunction attention_qk_norm_rope;
static CUfunction attention_kv_append;
static CUfunction attention_causal;
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

// AUTO routes Q4/Q5 prefill batches of at least this many rows to the 64-row
// tile, which reuses each decoded weight tile across twice as many rows.
#define Q45_PREFILL64_MIN_ROWS 64u

static int get_function(CUmodule module, CUfunction* function, const char* name) {
    return (int)cuModuleGetFunction(function, module, name);
}

static void initialize(void) {
    init_status = euhedral_cuda_load_kernel(
            &quantized_anchor, "qwen_layer_linear.cu", "euhedral_linear_quantized_bf16", &quantized_module, &linear_quantized);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    CUresult status = get_function(quantized_module, &linear_bf16_to_float, "euhedral_linear_bf16_to_float");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }

    q45_status = euhedral_cuda_load_kernel(
            &q45_anchor, "q45_linear_bf16.cu", "euhedral_q4_decode_1", &q45_module, &q45_decode[0][0]);
    if (q45_status == EUHEDRAL_CUDA_SUCCESS) {
        static const char* const decode_names[2][3] = {
                {"euhedral_q4_decode_1", "euhedral_q4_decode_2", "euhedral_q4_decode_4"},
                {"euhedral_q5_decode_1", "euhedral_q5_decode_2", "euhedral_q5_decode_4"}};
        static const char* const prefill_names[2] = {"euhedral_q4_prefill", "euhedral_q5_prefill"};
        static const char* const prefill64_names[2] = {"euhedral_q4_prefill_64", "euhedral_q5_prefill_64"};
        for (int format = 0; format < 2 && status == CUDA_SUCCESS; format++) {
            for (int tile = 0; tile < 3 && status == CUDA_SUCCESS; tile++)
                status = get_function(q45_module, &q45_decode[format][tile], decode_names[format][tile]);
            if (status == CUDA_SUCCESS) status = get_function(q45_module, &q45_prefill[format], prefill_names[format]);
            if (status == CUDA_SUCCESS) status = get_function(q45_module, &q45_prefill64[format], prefill64_names[format]);
        }
        if (status != CUDA_SUCCESS) q45_status = (int)status;
    }

    init_status = euhedral_cuda_load_kernel(
            &gdn_anchor, "qwen_gdn_ops.cu", "euhedral_gdn_control_fp32", &gdn_module, &gdn_control);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    status = get_function(gdn_module, &gdn_convolution, "euhedral_gdn_convolution_bf16");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }
    status = get_function(gdn_module, &gdn_recurrence, "euhedral_gdn_recurrence_bf16");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }
    status = get_function(gdn_module, &gdn_gated_rms_norm, "euhedral_gdn_gated_rms_norm_bf16");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }

    init_status = euhedral_cuda_load_kernel(
            &elementwise_anchor, "qwen_elementwise.cu", "euhedral_residual_add_bf16", &elementwise_module, &residual_add);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    status = get_function(elementwise_module, &swiglu, "euhedral_swiglu_bf16");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }

    init_status = euhedral_cuda_load_kernel(
            &attention_anchor,
            "qwen_attention_ops.cu",
            "euhedral_attention_qk_norm_rope_bf16",
            &attention_module,
            &attention_qk_norm_rope);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    status = get_function(attention_module, &attention_kv_append, "euhedral_attention_kv_append_bf16");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }
    status = get_function(attention_module, &attention_causal, "euhedral_attention_causal_bf16");
    init_status = status == CUDA_SUCCESS ? EUHEDRAL_CUDA_SUCCESS : (int)status;
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
    CUresult status = cuLaunchKernel(function, grid_x, 1, 1, block_x, 1, 1, 0,
            euhedral_cuda_submission_stream(), parameters, NULL);
    if (status != CUDA_SUCCESS) return (int)status;
    if (euhedral_cuda_submission_stream() != NULL) return EUHEDRAL_CUDA_SUCCESS;
    cudaError_t sync = cudaDeviceSynchronize();
    return sync == cudaSuccess ? EUHEDRAL_CUDA_SUCCESS : (int)sync;
}

static int align_plane(uint64_t size, uint64_t* aligned) {
    if (size > UINT64_MAX - 255) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    *aligned = (size + 255) & ~UINT64_C(255);
    return EUHEDRAL_CUDA_SUCCESS;
}

static int quantized_byte_size(uint32_t in_features, uint32_t out_features, uint32_t bits, uint64_t* required) {
    if (in_features == 0 || in_features % 128 != 0 || out_features == 0 || (bits != 4 && bits != 5))
        return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    const uint64_t groups = in_features / 64;
    if ((uint64_t)out_features > UINT64_MAX / groups / 32) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    const uint64_t code_bytes = (uint64_t)out_features * groups * 32;
    uint64_t code_plane_bytes;
    int status = align_plane(code_bytes, &code_plane_bytes);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    uint64_t high_plane_bytes = 0;
    if (bits == 5) {
        if ((uint64_t)out_features > UINT64_MAX / groups / 8) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
        const uint64_t high_bytes = (uint64_t)out_features * groups * 8;
        status = align_plane(high_bytes, &high_plane_bytes);
        if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    }
    if (code_plane_bytes > UINT64_MAX - high_plane_bytes) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    const uint64_t scale_offset = code_plane_bytes + high_plane_bytes;
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
    const uint64_t count = (uint64_t)rows * out_features;
    if (count > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr input = (CUdeviceptr)(uintptr_t)device_input;
    CUdeviceptr weights = (CUdeviceptr)(uintptr_t)device_weights;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, in_arg = in_features, out_arg = out_features, bits_arg = bits;
    void* parameters[] = {&input, &weights, &output, &rows_arg, &in_arg, &out_arg, &bits_arg};
    // The scalar kernel strides over outputs, so its grid is capped at the CUDA
    // grid-x limit rather than rejecting counts above it.
    const uint32_t scalar_grid = count > 2147483647u ? 2147483647u : (uint32_t)count;
    const char* mode = getenv("EUHEDRAL_Q45_DISPATCH");
    if (mode != NULL && strcmp(mode, "SCALAR") == 0)
        return launch_and_synchronize(linear_quantized, scalar_grid, 128, parameters);
    // DECODE, PREFILL and PREFILL64 force one optimized route; AUTO (the default)
    // selects decode through the per-format row threshold, then the 64-row
    // prefill tile from Q45_PREFILL64_MIN_ROWS rows and the 32-row tile below it.
    enum { ROUTE_DECODE, ROUTE_PREFILL, ROUTE_PREFILL64 } route;
    if (mode != NULL && strcmp(mode, "DECODE") == 0) route = ROUTE_DECODE;
    else if (mode != NULL && strcmp(mode, "PREFILL") == 0) route = ROUTE_PREFILL;
    else if (mode != NULL && strcmp(mode, "PREFILL64") == 0) route = ROUTE_PREFILL64;
    else if (mode == NULL || strcmp(mode, "AUTO") == 0) {
        const char* threshold_string = getenv(bits == 4 ? "EUHEDRAL_Q4_DECODE_MAX_ROWS" : "EUHEDRAL_Q5_DECODE_MAX_ROWS");
        uint32_t threshold = bits == 4 ? 9 : 4;
        if (threshold_string != NULL) {
            char* end;
            unsigned long parsed = strtoul(threshold_string, &end, 10);
            if (*threshold_string == '\0' || *threshold_string == '-' || *end != '\0' || parsed > UINT32_MAX)
                return EUHEDRAL_CUDA_INVALID_ARGUMENT;
            threshold = (uint32_t)parsed;
        }
        route = rows <= threshold ? ROUTE_DECODE
                : rows >= Q45_PREFILL64_MIN_ROWS ? ROUTE_PREFILL64 : ROUTE_PREFILL;
    } else return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    // The optimized kernels read code, fifth-bit and scale-pair words as 32-bit
    // values, which needs a 4-byte-aligned weight base. Other bases (a valid
    // payload may start at an allocation + 2) keep the byte-oriented scalar
    // reference, which has no alignment requirement beyond its FP16 scales.
    if (((uintptr_t)device_weights & 3u) != 0u)
        return launch_and_synchronize(linear_quantized, scalar_grid, 128, parameters);
    if (q45_status != EUHEDRAL_CUDA_SUCCESS) return q45_status;
    const int format = bits == 5;
    CUfunction function;
    uint64_t grid64;
    const uint64_t decode_tiles = ((uint64_t)out_features + 7u) / 8u;
    const uint64_t prefill_tiles = ((uint64_t)out_features + 31u) / 32u;
    if (route == ROUTE_DECODE) {
        // Rows per CTA: 1 and 2 exactly, otherwise 4 (the last CTA may be partial).
        const uint32_t row_tile = rows == 1 ? 1 : rows == 2 ? 2 : 4;
        function = q45_decode[format][row_tile == 1 ? 0 : row_tile == 2 ? 1 : 2];
        grid64 = (((uint64_t)rows + row_tile - 1u) / row_tile) * decode_tiles;
    } else if (route == ROUTE_PREFILL64) {
        function = q45_prefill64[format];
        grid64 = (((uint64_t)rows + 63u) / 64u) * prefill_tiles;
    } else {
        function = q45_prefill[format];
        grid64 = (((uint64_t)rows + 31u) / 32u) * prefill_tiles;
    }
    if (grid64 > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    uint32_t grid = (uint32_t)grid64;
    void* optimized_parameters[] = {&input, &weights, &output, &rows_arg, &in_arg, &out_arg};
    return launch_and_synchronize(function, grid, 128, optimized_parameters);
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
        float* device_g_output, float* device_beta_output, uint32_t rows, uint32_t heads) {
    if (device_a_projection == NULL || device_b_projection == NULL || device_a_log == NULL || device_dt_bias == NULL
            || device_g_output == NULL || device_beta_output == NULL || rows == 0 || heads == 0)
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
    CUdeviceptr g = (CUdeviceptr)(uintptr_t)device_g_output;
    CUdeviceptr beta = (CUdeviceptr)(uintptr_t)device_beta_output;
    uint32_t rows_arg = rows, heads_arg = heads;
    void* parameters[] = {&a, &b, &a_log, &dt_bias, &g, &beta, &rows_arg, &heads_arg};
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
    return launch_and_synchronize(gdn_convolution, (convolution_width + 127) / 128, 128, parameters);
}

int euhedral_cuda_gdn_recurrence_bf16(
        const void* device_convolved, const float* device_g, const float* device_beta,
        float* device_recurrent_state, void* device_output, uint32_t rows,
        uint32_t key_heads, uint32_t value_heads, uint32_t key_head_dim,
        uint32_t value_head_dim, float output_scale) {
    if (device_convolved == NULL || device_g == NULL || device_beta == NULL || device_recurrent_state == NULL
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
    CUdeviceptr g = (CUdeviceptr)(uintptr_t)device_g;
    CUdeviceptr beta = (CUdeviceptr)(uintptr_t)device_beta;
    CUdeviceptr state = (CUdeviceptr)(uintptr_t)device_recurrent_state;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, key_heads_arg = key_heads, value_heads_arg = value_heads;
    uint32_t key_dim_arg = key_head_dim, value_dim_arg = value_head_dim;
    void* parameters[] = {&convolved, &g, &beta, &state, &output, &rows_arg,
            &key_heads_arg, &value_heads_arg, &key_dim_arg, &value_dim_arg, &output_scale};
    return launch_and_synchronize(gdn_recurrence, (uint32_t)rows_count, key_head_dim, parameters);
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
        float epsilon,
        double rope_theta) {
    if (device_query_key == NULL || device_query_norm == NULL || device_key_norm == NULL || device_output == NULL
            || rows == 0 || query_heads == 0 || key_value_heads == 0 || query_heads % key_value_heads != 0
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
    uint32_t head_dim_arg = head_dim, rotary_dim_arg = rotary_dim;
    void* parameters[] = {&query_key, &query_norm, &key_norm, &output, &rows_arg, &query_heads_arg,
            &key_value_heads_arg, &head_dim_arg, &rotary_dim_arg, &start_position, &epsilon, &rope_theta};
    return launch_and_synchronize(attention_qk_norm_rope, (uint32_t)blocks, head_dim, parameters);
}

int euhedral_cuda_attention_kv_append_bf16(
        const void* device_query_key,
        const void* device_gate_value,
        void* device_key_cache,
        void* device_value_cache,
        uint32_t rows,
        uint32_t query_width,
        uint32_t key_value_width,
        uint64_t start_position) {
    if (device_query_key == NULL || device_gate_value == NULL || device_key_cache == NULL || device_value_cache == NULL
            || rows == 0 || query_width == 0 || key_value_width == 0 || query_width % key_value_width != 0
            || start_position > UINT64_MAX - rows)
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
    uint32_t rows_arg = rows, query_width_arg = query_width, key_value_width_arg = key_value_width;
    void* parameters[] = {&query_key, &gate_value, &key_cache, &value_cache,
            &rows_arg, &query_width_arg, &key_value_width_arg, &start_position};
    return launch_and_synchronize(attention_kv_append, (uint32_t)((count + 255) / 256), 256, parameters);
}

int euhedral_cuda_attention_causal_bf16(
        const void* device_query_key,
        const void* device_gate_value,
        const void* device_key_cache,
        const void* device_value_cache,
        void* device_output,
        uint32_t rows,
        uint32_t query_heads,
        uint32_t key_value_heads,
        uint32_t head_dim,
        uint32_t cache_length,
        uint64_t start_position) {
    if (device_query_key == NULL || device_gate_value == NULL || device_key_cache == NULL || device_value_cache == NULL
            || device_output == NULL || rows == 0 || query_heads == 0 || key_value_heads == 0
            || query_heads % key_value_heads != 0 || head_dim != 256 || cache_length == 0
            || start_position > UINT64_MAX - rows || start_position + rows > cache_length)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const uint64_t blocks = (uint64_t)rows * query_heads;
    if (blocks > UINT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    CUdeviceptr query_key = (CUdeviceptr)(uintptr_t)device_query_key;
    CUdeviceptr gate_value = (CUdeviceptr)(uintptr_t)device_gate_value;
    CUdeviceptr key_cache = (CUdeviceptr)(uintptr_t)device_key_cache;
    CUdeviceptr value_cache = (CUdeviceptr)(uintptr_t)device_value_cache;
    CUdeviceptr output = (CUdeviceptr)(uintptr_t)device_output;
    uint32_t rows_arg = rows, query_heads_arg = query_heads, key_value_heads_arg = key_value_heads;
    uint32_t head_dim_arg = head_dim, cache_length_arg = cache_length;
    void* parameters[] = {&query_key, &gate_value, &key_cache, &value_cache, &output, &rows_arg,
            &query_heads_arg, &key_value_heads_arg, &head_dim_arg, &cache_length_arg, &start_position};
    return launch_and_synchronize(attention_causal, (uint32_t)blocks, head_dim, parameters);
}
