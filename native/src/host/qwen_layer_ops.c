#include "cuda_kernel_loader.h"
#include "euhedral_cuda.h"
#include "qwen_ffn_policy.h"
#include "q45_decode_policy.h"
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

#ifdef _WIN32
static INIT_ONCE once = INIT_ONCE_STATIC_INIT;
#else
static pthread_once_t once = PTHREAD_ONCE_INIT;
#endif
static int quantized_anchor;
static int q45_anchor;
static int gdn_anchor;
static int elementwise_anchor;
static int sampling_anchor;
static int attention_anchor;
static int ffn_anchor;
static CUmodule ffn_module;
static CUfunction gate_up_swiglu;
static CUfunction gate_up64x32, gate_up128x32, ffn_down128x64, ffn_down64x64;
static CUfunction gate_up64x32_exact, gate_up128x32_exact, ffn_down128x64_exact, ffn_down64x64_exact;
static CUfunction ffn_down_split64, ffn_down_split128, ffn_down_reduce;
static CUfunction ffn_stream_gate;
static CUfunction ffn_stream_down;
static CUfunction stream_gate128, stream_down128, stream_gate128_exact, stream_down128_exact;
static CUmodule quantized_module;
static CUmodule q45_module;
static CUmodule gdn_module;
static CUmodule elementwise_module;
static CUmodule attention_module;
static CUmodule sampling_module;
static CUfunction linear_quantized;
// Q4/Q5 kernels indexed by [bits == 5]: cooperative decode for 1, 2 and 4
// token rows per CTA, and 32- and 64-row prefill tiles.
static CUfunction q45_decode[2][3];
static CUfunction q45_decode_wide[2];
static CUfunction q45_decode_contiguous[2];
static CUfunction q45_decode_contiguous_rows[2][9];  /* [format][M]: row-exact twins for 2..8 rows */
static CUfunction q45_prefill[2];
static CUfunction q45_prefill64[2];
static CUfunction q45_grouped64;
static CUfunction q45_prefill_exact[2], q45_prefill64_exact[2], q45_grouped64_exact;
static CUfunction q45_grouped_balanced[2]; /* 64-row and 128-row tiles */
static CUfunction q45_prefill_wide[2][2]; /* [format][64-row, 128-row tile] */

/* Exact numerics select the hi + lo twins of the Q4/Q5 prefill kernels. */
static CUfunction q45_prefill_kernel(int format, int wide) {
    CUfunction exact = wide ? q45_prefill64_exact[format] : q45_prefill_exact[format];
    if (exact != NULL && euhedral_cuda_exact_numerics()) return exact;
    return wide ? q45_prefill64[format] : q45_prefill[format];
}

/* Relaxed numerics: the balanced tile engine for a qualified 64-row prefill route. Sets the grid
   and returns NULL where the 64 x 32 kernel applies. */
static CUfunction q45_prefill_wide_kernel(int format, const void* input, uint32_t rows, uint32_t in_features,
        uint32_t out_features, uint64_t* grid) {
    uint32_t tile_rows = euhedral_q45_prefill_wide_rows(format ? 5u : 4u, rows, in_features, out_features);
    if (tile_rows == 0u || euhedral_cuda_exact_numerics() || ((uintptr_t)input & 15u) != 0u) return NULL;
    CUfunction function = q45_prefill_wide[format][tile_rows == 128u];
    if (function != NULL)
        *grid = (((uint64_t)rows + tile_rows - 1u) / tile_rows) * (((uint64_t)out_features + 63u) / 64u);
    return function;
}
static int q45_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
static CUfunction linear_bf16_to_float;
static CUfunction gdn_control;
static CUfunction gdn_convolution;
static CUfunction gdn_recurrence, gdn_recurrence_c8, gdn_recurrence_c4;
static CUfunction gdn_gated_rms_norm;
static CUfunction residual_add;
static CUfunction argmax_bf16;
static CUfunction residual_rms_norm, residual_rms_norm_row;
static CUfunction gdn_project_control, gdn_project_control_tiled;
static CUfunction swiglu;
static CUfunction attention_qk_norm_rope;
static CUfunction attention_qk_norm_rope_rows;
static CUfunction attention_kv_append;
static CUfunction attention_causal;
static CUfunction attention_append_nvfp4, attention_prefill_nvfp4, attention_decode_nvfp4, attention_merge_nvfp4;
static CUfunction attention_prefill_nvfp4_exact, attention_decode_nvfp4_exact, attention_prefill_fa2, attention_decode_gqa;
/* Row-exact multi-row twins of decode attention and its merge (speculative verification). */
static CUfunction attention_decode_rows, attention_decode_gqa_rows, attention_merge_rows;
static int init_status = EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;

// The grouped Q4+Q5 launch of the GDN input projections wins over two launches
// only in this row range: below it the 32-row tile of the separate route is
// faster, above it the grouped kernel's register use costs occupancy.
#define Q45_GROUPED_MIN_ROWS 33u
#define Q45_GROUPED_MAX_ROWS 192u

// AUTO routes Q4/Q5 prefill batches of at least this many rows to the 64-row
// tile, which reuses each decoded weight tile across twice as many rows.
#define Q45_PREFILL64_MIN_ROWS 64u

static int get_function(CUmodule module, CUfunction* function, const char* name) {
    return (int)cuModuleGetFunction(function, module, name);
}

static void initialize_modules(void) {
    init_status = euhedral_cuda_load_kernel(
            &quantized_anchor, "linear/kernels.cu", "euhedral_linear_quantized_bf16", &quantized_module, &linear_quantized);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    CUresult status = get_function(quantized_module, &linear_bf16_to_float, "euhedral_linear_bf16_to_float");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }

    q45_status = euhedral_cuda_load_kernel(
            &q45_anchor, "q45/kernels.cu", "euhedral_q4_decode_1", &q45_module, &q45_decode[0][0]);
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
        // Optional: without them single-row decode keeps the cooperative kernels.
        get_function(q45_module, &q45_decode_wide[0], "euhedral_q4_decode_wide");
        get_function(q45_module, &q45_decode_wide[1], "euhedral_q5_decode_wide");
        get_function(q45_module, &q45_decode_contiguous[0], "euhedral_q4_decode_contiguous");
        get_function(q45_module, &q45_decode_contiguous[1], "euhedral_q5_decode_contiguous");
        for (int format = 0; format < 2; format++)
            for (int m = 2; m <= 8; m++) {
                char name[48];
                snprintf(name, sizeof(name), "euhedral_q%d_decode_contiguous_rows%d", 4 + format, m);
                if (cuModuleGetFunction(&q45_decode_contiguous_rows[format][m], q45_module, name) != CUDA_SUCCESS)
                    q45_decode_contiguous_rows[format][m] = NULL;
            }
        // Optional: without it the GDN projection pair falls back to two launches.
        get_function(q45_module, &q45_grouped64, "euhedral_q45_prefill_64_grouped");
        get_function(q45_module, &q45_grouped64_exact, "euhedral_q45_prefill_64_grouped_exact");
        get_function(q45_module, &q45_grouped_balanced[0], "euhedral_q45_grouped_64x64");
        get_function(q45_module, &q45_grouped_balanced[1], "euhedral_q45_grouped_128x64");
        get_function(q45_module, &q45_prefill_exact[0], "euhedral_q4_prefill_exact");
        get_function(q45_module, &q45_prefill_exact[1], "euhedral_q5_prefill_exact");
        get_function(q45_module, &q45_prefill64_exact[0], "euhedral_q4_prefill_64_exact");
        get_function(q45_module, &q45_prefill64_exact[1], "euhedral_q5_prefill_64_exact");
        // Optional: without them relaxed prefill keeps the 64 x 32 kernels.
        get_function(q45_module, &q45_prefill_wide[0][0], "euhedral_q4_prefill_64x64");
        get_function(q45_module, &q45_prefill_wide[0][1], "euhedral_q4_prefill_128x64");
        get_function(q45_module, &q45_prefill_wide[1][0], "euhedral_q5_prefill_64x64");
        get_function(q45_module, &q45_prefill_wide[1][1], "euhedral_q5_prefill_128x64");
        if (status != CUDA_SUCCESS) q45_status = (int)status;
    }

    init_status = euhedral_cuda_load_kernel(
            &gdn_anchor, "gdn/kernels.cu", "euhedral_gdn_control_fp32", &gdn_module, &gdn_control);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    get_function(gdn_module, &gdn_project_control, "euhedral_gdn_project_control_fp32");
    get_function(gdn_module, &gdn_project_control_tiled, "euhedral_gdn_project_control_8x4_fp32");
    status = get_function(gdn_module, &gdn_convolution, "euhedral_gdn_convolution_bf16");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }
    status = get_function(gdn_module, &gdn_recurrence, "euhedral_gdn_recurrence_bf16");
    if (status == CUDA_SUCCESS) {
        // Optional relaxed column-owned recurrences; exact numerics keep the warp-tree kernel.
        get_function(gdn_module, &gdn_recurrence_c8, "euhedral_gdn_recurrence_c8_bf16");
        get_function(gdn_module, &gdn_recurrence_c4, "euhedral_gdn_recurrence_c4_bf16");
    }
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }
    status = get_function(gdn_module, &gdn_gated_rms_norm, "euhedral_gdn_gated_rms_norm_bf16");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }

    init_status = euhedral_cuda_load_kernel(
            &elementwise_anchor, "elementwise/kernels.cu", "euhedral_residual_add_bf16", &elementwise_module, &residual_add);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    get_function(elementwise_module, &residual_rms_norm, "euhedral_residual_rms_norm_bf16");
    get_function(elementwise_module, &residual_rms_norm_row, "euhedral_residual_rms_norm_row_bf16");
    status = get_function(elementwise_module, &swiglu, "euhedral_swiglu_bf16");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }
    // Optional: without it greedy selection stays on the host.
    euhedral_cuda_load_kernel(&sampling_anchor, "sampling/kernels.cu", "euhedral_argmax_bf16", &sampling_module,
            &argmax_bf16);

    init_status = euhedral_cuda_load_kernel(
            &attention_anchor,
            "attention/kernels.cu",
            "euhedral_attention_qk_norm_rope_bf16",
            &attention_module,
            &attention_qk_norm_rope);
    if (init_status != EUHEDRAL_CUDA_SUCCESS) return;
    status = get_function(attention_module, &attention_append_nvfp4, "euhedral_attention_kv_append_nvfp4");
    if (status == CUDA_SUCCESS) status = get_function(attention_module, &attention_prefill_nvfp4, "euhedral_attention_prefill32_nvfp4");
    if (status == CUDA_SUCCESS)
        get_function(attention_module, &attention_prefill_nvfp4_exact, "euhedral_attention_prefill32_nvfp4_exact");
    if (status == CUDA_SUCCESS)
        get_function(attention_module, &attention_prefill_fa2, "euhedral_attention_prefill_fa2_nvfp4");
    if (status == CUDA_SUCCESS)
        get_function(attention_module, &attention_decode_gqa, "euhedral_attention_decode_gqa_nvfp4");
    if (status == CUDA_SUCCESS) status = get_function(attention_module, &attention_decode_nvfp4, "euhedral_attention_decode_nvfp4");
    if (status == CUDA_SUCCESS)
        get_function(attention_module, &attention_decode_nvfp4_exact, "euhedral_attention_decode_nvfp4_exact");
    if (status == CUDA_SUCCESS) status = get_function(attention_module, &attention_merge_nvfp4, "euhedral_attention_merge_nvfp4");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }
    if (get_function(attention_module, &attention_decode_rows, "euhedral_attention_decode_nvfp4_rows") != CUDA_SUCCESS
            || get_function(attention_module, &attention_merge_rows, "euhedral_attention_merge_nvfp4_rows") != CUDA_SUCCESS)
        attention_decode_rows = attention_merge_rows = NULL;
    if (get_function(attention_module, &attention_decode_gqa_rows, "euhedral_attention_decode_gqa_nvfp4_rows") != CUDA_SUCCESS)
        attention_decode_gqa_rows = NULL;
    // Optional row-owned variant (256-dimension heads), bitwise equal to the per-head kernel.
    get_function(attention_module, &attention_qk_norm_rope_rows, "euhedral_attention_qk_norm_rope_rows_bf16");
    status = get_function(attention_module, &attention_kv_append, "euhedral_attention_kv_append_bf16");
    if (status != CUDA_SUCCESS) { init_status = (int)status; return; }
    status = get_function(attention_module, &attention_causal, "euhedral_attention_causal_bf16");
    init_status = status == CUDA_SUCCESS ? EUHEDRAL_CUDA_SUCCESS : (int)status;
    if (init_status == EUHEDRAL_CUDA_SUCCESS) {
        euhedral_cuda_load_kernel(&ffn_anchor, "ffn/kernels.cu", "euhedral_q3_gate_up_swiglu_bf16",
                &ffn_module, &gate_up_swiglu);
        if (ffn_module) {
            get_function(ffn_module, &gate_up64x32, "euhedral_q3_gate_up_swiglu_64x32");
            get_function(ffn_module, &gate_up128x32, "euhedral_q3_gate_up_swiglu_128x32");
            get_function(ffn_module, &ffn_down128x64, "euhedral_q3_ffn_down_128x64");
            get_function(ffn_module, &ffn_down64x64, "euhedral_q3_ffn_down_64x64");
            get_function(ffn_module, &gate_up64x32_exact, "euhedral_q3_gate_up_swiglu_64x32_exact");
            get_function(ffn_module, &gate_up128x32_exact, "euhedral_q3_gate_up_swiglu_128x32_exact");
            get_function(ffn_module, &ffn_down64x64_exact, "euhedral_q3_ffn_down_64x64_exact");
            get_function(ffn_module, &ffn_down128x64_exact, "euhedral_q3_ffn_down_128x64_exact");
            get_function(ffn_module, &stream_gate128_exact, "stream_gate_up_128x32_exact");
            get_function(ffn_module, &stream_down128_exact, "stream_down_128x64_exact");
            get_function(ffn_module, &ffn_down_split64, "euhedral_q3_ffn_down_split_64x64");
            get_function(ffn_module, &ffn_down_split128, "euhedral_q3_ffn_down_split_128x64");
            get_function(ffn_module, &ffn_down_reduce, "euhedral_q3_ffn_down_reduce");
            get_function(ffn_module, &stream_gate128, "stream_gate_up_128x32");
            get_function(ffn_module, &stream_down128, "stream_down_128x64");
            get_function(ffn_module, &ffn_stream_gate, "stream_gate_up");
            get_function(ffn_module, &ffn_stream_down, "stream_down");
        }
    }
}

// Every kernel registered here begins with euhedral_pdl_begin(), so stream-serialized
// programmatic dependent launch is safe for it (see cuda_kernel_loader.h).
static void initialize(void) {
    initialize_modules();
    euhedral_cuda_pdl_register(linear_bf16_to_float);
    for (int format = 0; format < 2; format++)
        for (int tile = 0; tile < 3; tile++) euhedral_cuda_pdl_register(q45_decode[format][tile]);
    for (int format = 0; format < 2; format++) euhedral_cuda_pdl_register(q45_decode_wide[format]);
    for (int format = 0; format < 2; format++) euhedral_cuda_pdl_register(q45_decode_contiguous[format]);
    for (int format = 0; format < 2; format++)
        for (int m = 2; m <= 8; m++) euhedral_cuda_pdl_register(q45_decode_contiguous_rows[format][m]);
    euhedral_cuda_pdl_register(gdn_control);
    euhedral_cuda_pdl_register(argmax_bf16);
    euhedral_cuda_pdl_register(gdn_project_control);
    euhedral_cuda_pdl_register(residual_rms_norm);
    euhedral_cuda_pdl_register(residual_rms_norm_row);
    euhedral_cuda_pdl_register(gdn_convolution);
    euhedral_cuda_pdl_register(gdn_recurrence);
    euhedral_cuda_pdl_register(gdn_recurrence_c8);
    euhedral_cuda_pdl_register(gdn_recurrence_c4);
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

static int launch_and_synchronize_2d(CUfunction function, uint32_t grid_x, uint32_t grid_y, uint32_t block_x,
        void** parameters) {
    CUresult status = euhedral_launch_kernel(function, grid_x, grid_y, 1, block_x, 1, 1, 0,
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
    // Row-exact verification (euhedral_cuda_row_exact): every row exactly as a one-row call. Where that
    // call runs the contiguous decode kernel, its multi-row twin computes all rows in one launch, bit
    // for bit; otherwise the rows run one at a time.
    if (rows > 1 && euhedral_cuda_row_exact()) {
        const int format = bits == 5;
        const int decode_route = mode == NULL || strcmp(mode, "AUTO") == 0 || strcmp(mode, "DECODE") == 0;
        if (decode_route && q45_status == EUHEDRAL_CUDA_SUCCESS && rows <= 8u
                && q45_decode_contiguous[format] != NULL && q45_decode_contiguous_rows[format][rows] != NULL
                && !euhedral_cuda_exact_numerics()
                && (((uintptr_t)device_weights | (uintptr_t)device_input) & 15u) == 0u
                && euhedral_q45_decode_contiguous_shape(1, in_features, out_features)) {
            void* twin_parameters[] = {&input, &weights, &output, &rows_arg, &in_arg, &out_arg};
            return launch_and_synchronize(q45_decode_contiguous_rows[format][rows], out_features / 8u, 128,
                    twin_parameters);
        }
        for (uint32_t row = 0; row < rows; row++) {
            status = euhedral_cuda_linear_quantized_bf16((const unsigned char*)device_input + (uint64_t)row * in_features * 2u,
                    device_weights, (unsigned char*)device_output + (uint64_t)row * out_features * 2u, 1, in_features,
                    out_features, weights_byte_size, bits);
            if (status != EUHEDRAL_CUDA_SUCCESS) return status;
        }
        return EUHEDRAL_CUDA_SUCCESS;
    }
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
        if (q45_decode_contiguous[format] != NULL && !euhedral_cuda_exact_numerics()
                && (((uintptr_t)device_weights | (uintptr_t)device_input) & 15u) == 0u
                && euhedral_q45_decode_contiguous_shape(rows, in_features, out_features))
            function = q45_decode_contiguous[format];
        else if (q45_decode_wide[format] != NULL && ((uintptr_t)device_weights & 15u) == 0u
                && euhedral_q45_decode_wide_shape(rows, in_features, out_features))
            function = q45_decode_wide[format];
        grid64 = (((uint64_t)rows + row_tile - 1u) / row_tile) * decode_tiles;
    } else if (route == ROUTE_PREFILL64) {
        function = q45_prefill_wide_kernel(format, device_input, rows, in_features, out_features, &grid64);
        if (function == NULL) {
            function = q45_prefill_kernel(format, 1);
            grid64 = (((uint64_t)rows + 63u) / 64u) * prefill_tiles;
        }
    } else {
        function = q45_prefill_kernel(format, 0);
        grid64 = (((uint64_t)rows + 31u) / 32u) * prefill_tiles;
    }
    if (grid64 > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    uint32_t grid = (uint32_t)grid64;
    void* optimized_parameters[] = {&input, &weights, &output, &rows_arg, &in_arg, &out_arg};
    return launch_and_synchronize(function, grid, 128, optimized_parameters);
}

int euhedral_cuda_gdn_projections_bf16(
        const void* input, const void* q4, const void* q5, void* qk_output, void* value_z_output,
        uint32_t rows, uint32_t hidden, uint32_t qk_width, uint32_t value_z_width,
        uint64_t q4_bytes, uint64_t q5_bytes) {
    if (!input || !q4 || !q5 || !qk_output || !value_z_output || rows == 0 || qk_width == 0 || value_z_width == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    const char* mode = getenv("EUHEDRAL_Q45_DISPATCH");
    const int automatic = mode == NULL || strcmp(mode, "AUTO") == 0;
    // Explicit decode thresholds keep the ordinary per-projection routing.
    const int thresholds = getenv("EUHEDRAL_Q4_DECODE_MAX_ROWS") != NULL || getenv("EUHEDRAL_Q5_DECODE_MAX_ROWS") != NULL;
    int status = EUHEDRAL_CUDA_SUCCESS;
    if (automatic && !thresholds && rows >= Q45_GROUPED_MIN_ROWS && rows <= Q45_GROUPED_MAX_ROWS
            && (((uintptr_t)q4 | (uintptr_t)q5) & 3u) == 0u) {
        uint64_t expected4, expected5;
        status = quantized_byte_size(hidden, qk_width, 4, &expected4);
        if (status != EUHEDRAL_CUDA_SUCCESS) return status;
        status = quantized_byte_size(hidden, value_z_width, 5, &expected5);
        if (status != EUHEDRAL_CUDA_SUCCESS) return status;
        if (expected4 != q4_bytes || expected5 != q5_bytes) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
        status = euhedral_cuda_bind_thread_context();
        if (status != EUHEDRAL_CUDA_SUCCESS) return status;
        status = ensure_initialized();
        if (status != EUHEDRAL_CUDA_SUCCESS) return status;
        if (q45_status == EUHEDRAL_CUDA_SUCCESS && q45_grouped64) {
            const uint64_t row_tiles = ((uint64_t)rows + 63u) / 64u;
            const uint64_t grid64 = row_tiles * (((uint64_t)qk_width + 31u) / 32u) + row_tiles * (((uint64_t)value_z_width + 31u) / 32u);
            if (grid64 > 2147483647u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
            void* parameters[] = {&input, &q4, &qk_output, &q5, &value_z_output, &rows, &hidden, &qk_width,
                    &value_z_width};
            CUfunction grouped = q45_grouped64_exact != NULL && euhedral_cuda_exact_numerics()
                    ? q45_grouped64_exact : q45_grouped64;
            // Relaxed numerics: the balanced engine, 64-row tiles up to 64 rows and 128-row tiles above.
            int wide = rows > 64u;
            if (!euhedral_cuda_exact_numerics() && q45_grouped_balanced[wide] != NULL
                    && ((uintptr_t)input & 15u) == 0u && hidden % 32u == 0u) {
                const uint64_t tiles = ((uint64_t)rows + (wide ? 127u : 63u)) / (wide ? 128u : 64u);
                const uint64_t balanced_grid = tiles * (((uint64_t)qk_width + 63u) / 64u)
                        + tiles * (((uint64_t)value_z_width + 63u) / 64u);
                return launch_and_synchronize(q45_grouped_balanced[wide], (uint32_t)balanced_grid, 128, parameters);
            }
            return launch_and_synchronize(grouped, (uint32_t)grid64, 128, parameters);
        }
    }
    status = euhedral_cuda_linear_quantized_bf16(input, q4, qk_output, rows, hidden, qk_width, q4_bytes, 4);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    return euhedral_cuda_linear_quantized_bf16(input, q5, value_z_output, rows, hidden, value_z_width, q5_bytes, 5);
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
    // Relaxed numerics: column-owned lanes (C value columns per warp; 16-byte aligned rows and state).
    const char* mode = getenv("EUHEDRAL_GDN_RECURRENCE");
    int aligned = (((uintptr_t)device_convolved | (uintptr_t)device_recurrent_state) & 15u) == 0u;
    // From two rows (prefill); single-row decode keeps the exact kernel, which measured as fast in the
    // model (the one-row operator gain, 16.5 -> 14.8 us with four columns, did not survive). Eight
    // columns per warp: 512 rows 580 -> 327 us. EUHEDRAL_GDN_RECURRENCE=C4|C8|EXACT overrides.
    // Row-exact selects the kernel one-row decode would. Both kernels run rows in order with the state in
    // FP32 registers, so M rows in one launch equal M one-row launches bit for bit.
    if (!euhedral_cuda_exact_numerics() && aligned
            && (mode != NULL ? strcmp(mode, "EXACT") != 0 : rows > 1 && !euhedral_cuda_row_exact())) {
        int four = mode != NULL && strcmp(mode, "C4") == 0;
        CUfunction relaxed = four ? gdn_recurrence_c4 : gdn_recurrence_c8;
        if (relaxed != NULL)
            return launch_and_synchronize(relaxed, (uint32_t)(rows_count / (four ? 4 : 8)), 32, parameters);
    }
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

// One compound FFN instruction owns both slots and the FP32 continuation matrix.
// CUDA event edges name slot readiness/release; no consumer spins on global progress flags.
int euhedral_cuda_q3_ffn_streamed_bf16(
        const void* input, const void* gate_weights, const void* down_weights, void* output,
        void* slots, float* accumulators, uint32_t rows, uint32_t hidden, uint32_t intermediate,
        uint64_t gate_bytes, uint64_t down_bytes) {
    if (!input || !gate_weights || !down_weights || !output || !slots || !accumulators
            || !euhedral_ffn_streamed_rows(rows) || hidden != 5120 || intermediate != 17408
            || ((uintptr_t)input & 15u) != 0 || ((uintptr_t)slots & 15u) != 0
            || ((uintptr_t)gate_weights & 3u) != 0 || ((uintptr_t)down_weights & 3u) != 0
            || ((uintptr_t)accumulators & 3u) != 0 || ((uintptr_t)output & 1u) != 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    uint32_t gate_outputs = intermediate * 2u;
    uint64_t gate_groups = (uint64_t)gate_outputs * (hidden / 64u);
    uint64_t down_groups = (uint64_t)hidden * (intermediate / 64u);
    uint64_t gate_scale = (gate_groups * 24u + 255u) & ~UINT64_C(255);
    uint64_t down_scale = (down_groups * 24u + 255u) & ~UINT64_C(255);
    if (gate_bytes != gate_scale + gate_groups * 2u || down_bytes != down_scale + down_groups * 2u)
        return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    // Exact numerics stage BF16 hi + lo weights; the default stages hi only.
    int exact = euhedral_cuda_exact_numerics();
    CUfunction gate = exact ? stream_gate128_exact : stream_gate128;
    CUfunction down = exact ? stream_down128_exact : stream_down128;
    uint32_t row_tile = 128u, gate_tile = 32u, down_tile = 64u;
    if (!gate || !down) {
        // An older source bundle retains the old K32 continuation shape. Select its
        // functions and all three matching launch dimensions together.
        gate = ffn_stream_gate; down = ffn_stream_down;
        row_tile = 64u; gate_tile = 16u; down_tile = 32u;
    }
    if (!gate || !down) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    cudaStream_t producer = (cudaStream_t)euhedral_cuda_submission_stream();
    cudaStream_t consumer = NULL;
    cudaEvent_t ready[2] = {NULL, NULL}, release[2] = {NULL, NULL};
#define FFN_TRY(call) do { status = (int)(call); if (status != 0) goto cleanup; } while (0)
    FFN_TRY(cudaStreamCreateWithFlags(&consumer, cudaStreamNonBlocking));
    for (int slot = 0; slot < 2; slot++) {
        FFN_TRY(cudaEventCreateWithFlags(&ready[slot], cudaEventDisableTiming));
        FFN_TRY(cudaEventCreateWithFlags(&release[slot], cudaEventDisableTiming));
    }
    uint32_t part = 0, last_slot = 0;
    for (uint32_t begin = 0; begin < intermediate; begin += 4096u, part++) {
        uint32_t count = intermediate - begin;
        if (count > 4096u) count = 4096u;
        uint32_t slot = part & 1u;
        last_slot = slot;
        void* staging = (unsigned char*)slots + (uint64_t)slot * rows * 4096u * 2u;
        if (part >= 2u) FFN_TRY(cudaStreamWaitEvent(producer, release[slot], 0));
        void* gate_args[] = {&input, &gate_weights, &staging, &rows, &hidden, &gate_outputs, &gate_scale, &begin, &count};
        FFN_TRY(euhedral_launch_kernel(gate, (rows / row_tile) * (count / gate_tile), 1, 1, 128, 1, 1, 0,
                (CUstream)producer, gate_args, NULL));
        FFN_TRY(cudaEventRecord(ready[slot], producer));
        FFN_TRY(cudaStreamWaitEvent(consumer, ready[slot], 0));
        void* down_args[] = {&staging, &down_weights, &output, &rows, &intermediate, &hidden, &down_scale,
                &accumulators, &begin, &count};
        FFN_TRY(euhedral_launch_kernel(down, (rows / row_tile) * (hidden / down_tile), 1, 1, 128, 1, 1, 0,
                (CUstream)consumer, down_args, NULL));
        FFN_TRY(cudaEventRecord(release[slot], consumer));
    }
    FFN_TRY(cudaStreamWaitEvent(producer, release[last_slot], 0));
    if (producer == NULL) FFN_TRY(cudaStreamSynchronize(producer));
cleanup:
    if (status != 0) {
        // Best-effort branch drain. Any error still requires the caller to prove device
        // completion or quarantine borrowed storage, even with a NULL submission stream.
        if (consumer != NULL) cudaStreamSynchronize(consumer);
        cudaStreamSynchronize(producer);
    }
    for (int slot = 0; slot < 2; slot++) {
        if (ready[slot]) { int error = (int)cudaEventDestroy(ready[slot]); if (status == 0) status = error; }
        if (release[slot]) { int error = (int)cudaEventDestroy(release[slot]); if (status == 0) status = error; }
    }
    if (consumer) { int error = (int)cudaStreamDestroy(consumer); if (status == 0) status = error; }
#undef FFN_TRY
    return status;
}

int euhedral_cuda_q3_gate_up_swiglu_bf16(
        const void* input, const void* weights, void* output, uint32_t rows,
        uint32_t width, uint32_t outputs, uint64_t weight_bytes) {
    if (!input || !weights || !output || rows == 0 || width == 0 || outputs == 0
            || width % 128 != 0 || outputs % 32 != 0 || ((uintptr_t)input & 15u) != 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    uint64_t groups = (uint64_t)outputs * (width / 64u);
    if (groups > (UINT64_MAX - 255) / 26u) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    uint64_t scale_offset = (groups * 24u + 255u) & ~UINT64_C(255);
    if (weight_bytes != scale_offset + groups * 2u) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    uint64_t grid = (((uint64_t)rows + 63u) / 64u) * (outputs / 32u);
    if (grid > INT32_MAX) return EUHEDRAL_CUDA_SIZE_OVERFLOW;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    uint32_t tile_rows = euhedral_ffn_gate_tile_rows(rows, width, outputs);
    int exact = euhedral_cuda_exact_numerics();
    CUfunction selected = tile_rows == 64u ? (exact ? gate_up64x32_exact : gate_up64x32)
            : tile_rows == 128u ? (exact ? gate_up128x32_exact : gate_up128x32) : NULL;
    // Older source bundles keep the matching old symbol AND its old launch tile.
    if (selected) grid = (((uint64_t)rows + tile_rows - 1u) / tile_rows) * (outputs / 64u);
    else selected = gate_up_swiglu;
    if (!selected) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    void* parameters[] = {&input, &weights, &output, &rows, &width, &outputs, &scale_offset};
    return launch_and_synchronize(selected, (uint32_t)grid, 128, parameters);
}

int euhedral_cuda_q3_ffn_down_bf16(
        const void* input, const void* weights, void* output, uint32_t rows,
        uint32_t width, uint32_t outputs, uint64_t weight_bytes) {
    if (!input || !weights || !output || !rows || !width || !outputs || !weight_bytes)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    uint32_t tile_rows = euhedral_ffn_down_tile_rows(rows, width, outputs);
    if (tile_rows == 0u || ((uintptr_t)input & 15u) != 0 || ((uintptr_t)weights & 3u) != 0)
        return euhedral_cuda_linear_q3_prefill_bf16(input, weights, output, rows, width, outputs, weight_bytes);
    uint64_t groups = (uint64_t)outputs * (width / 64u);
    uint64_t scale_offset = (groups * 24u + 255u) & ~UINT64_C(255);
    if (weight_bytes != scale_offset + groups * 2u) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    // Older source bundles lack the 64-row entry point; they keep the generic route for those rows.
    int exact = euhedral_cuda_exact_numerics();
    CUfunction selected = tile_rows == 64u ? (exact ? ffn_down64x64_exact : ffn_down64x64)
            : (exact ? ffn_down128x64_exact : ffn_down128x64);
    if (!selected)
        return euhedral_cuda_linear_q3_prefill_bf16(input, weights, output, rows, width, outputs, weight_bytes);
    uint32_t grid = ((rows + tile_rows - 1u) / tile_rows) * (outputs / 64u);
    void* parameters[] = {&input, &weights, &output, &rows, &width, &outputs, &scale_offset};
    return launch_and_synchronize(selected, grid, 128, parameters);
}

int euhedral_cuda_q3_ffn_down_split_bf16(
        const void* input, const void* weights, void* output, float* partials,
        uint32_t rows, uint32_t width, uint32_t outputs, uint64_t weight_bytes) {
    if (!input || !weights || !output || rows == 0 || width == 0 || outputs == 0)
        return EUHEDRAL_CUDA_INVALID_ARGUMENT;
    uint32_t splits = euhedral_ffn_down_splits(rows, width, outputs);
    if (splits == 0u || !partials || euhedral_cuda_exact_numerics() || ((uintptr_t)input & 15u) != 0 || ((uintptr_t)weights & 3u) != 0
            || ((uintptr_t)partials & 3u) != 0)
        return euhedral_cuda_q3_ffn_down_bf16(input, weights, output, rows, width, outputs, weight_bytes);
    uint64_t groups = (uint64_t)outputs * (width / 64u);
    uint64_t scale_offset = (groups * 24u + 255u) & ~UINT64_C(255);
    if (weight_bytes != scale_offset + groups * 2u) return EUHEDRAL_CUDA_FORMAT_MISMATCH;
    int status = euhedral_cuda_bind_thread_context();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    status = ensure_initialized();
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    uint32_t tile_rows = rows <= 64u ? 64u : 128u;
    CUfunction split = tile_rows == 64u ? ffn_down_split64 : ffn_down_split128;
    if (!split || !ffn_down_reduce)
        return euhedral_cuda_q3_ffn_down_bf16(input, weights, output, rows, width, outputs, weight_bytes);
    uint32_t grid = ((rows + tile_rows - 1u) / tile_rows) * (outputs / 64u);
    void* split_parameters[] = {&input, &weights, &partials, &rows, &width, &outputs, &scale_offset, &splits};
    status = launch_and_synchronize_2d(split, grid, splits, 128, split_parameters);
    if (status != EUHEDRAL_CUDA_SUCCESS) return status;
    uint32_t count = rows * outputs;
    void* reduce_parameters[] = {&partials, &output, &count, &splits};
    return launch_and_synchronize(ffn_down_reduce, (count + 255u) / 256u, 256, reduce_parameters);
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
    if (!residual_rms_norm) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    void* parameters[] = {&residual, &delta, &weight, &hidden, &normalized, &rows, &width, &epsilon};
    // Relaxed numerics: one row (decode) keeps its columns in registers, eight per thread.
    if ((rows == 1 || row_exact) && residual_rms_norm_row != NULL && row_kernel)
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
    if (!gdn_project_control) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
    void* parameters[] = {&input, &a_weight, &b_weight, &a_log, &dt_bias, &alpha, &beta, &rows, &width, &heads};
    // From eight rows, 8-row x 4-head CTAs share activation and weight loads (bitwise equal).
    if (rows >= 8u && heads % 4u == 0u && gdn_project_control_tiled != NULL)
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
    if (argmax_bf16 == NULL) return EUHEDRAL_CUDA_KERNEL_UNAVAILABLE;
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
        uint64_t start_position) {
    if (device_query_key == NULL || device_gate_value == NULL || device_key_cache == NULL || device_value_cache == NULL
            || rows == 0 || query_width == 0 || key_value_width == 0 || query_width % key_value_width != 0
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
    uint32_t rows_arg = rows, query_width_arg = query_width, key_value_width_arg = key_value_width;
    void* parameters[] = {&query_key, &gate_value, &key_cache, &value_cache,
            &rows_arg, &query_width_arg, &key_value_width_arg, &start_position};
    return launch_and_synchronize(attention_append_nvfp4, (uint32_t)((count / 256 + 3) / 4), 128, parameters);
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
    // Row-exact: the one-row kernel already maps block -> (row, head) with position start + row, so a
    // rows x heads launch of it computes every row exactly as one-row launches would. The in-place query
    // path reads and writes only its own (row, head) block.
    const int row_exact_heads = rows > 1 && euhedral_cuda_row_exact();
    if (row_exact_heads && device_query_key != NULL && device_output != NULL
            && ((uint64_t)rows * ((uint64_t)query_heads + key_value_heads) > UINT32_MAX)) {
        const uint64_t width = ((uint64_t)query_heads + key_value_heads) * head_dim * 2u;
        for (uint32_t row = 0; row < rows; ++row) {
            int status = euhedral_cuda_attention_qk_norm_rope_bf16((const char*)device_query_key + row * width, device_query_norm,
                    device_key_norm, (char*)device_output + row * width, 1u, query_heads, key_value_heads, head_dim,
                    rotary_dim, start_position + row, epsilon, rope_theta);
            if (status != 0) return status;
        }
        return 0;
    }
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
    // A single row (decode) keeps one CTA per head; from two rows one CTA per row shares the angles.
    if (rows > 1 && !row_exact_heads && attention_qk_norm_rope_rows != NULL)
        return launch_and_synchronize(attention_qk_norm_rope_rows, rows, 256, parameters);
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

int euhedral_cuda_attention_causal_nvfp4(
        const void* query_key, const void* gate, const void* keys, const void* values, void* output,
        uint32_t rows, uint32_t query_heads, uint32_t key_heads, uint32_t head_dim,
        uint32_t cache_length, uint64_t start, void* scratch) {
    // The row twins also serve small non-row-exact quanta that bring decode scratch (speculative drafting):
    // a 32-row prefill tile over a long cache is one CTA per head, about 9.5 ms over 32K keys for 2-4 rows
    // against 0.45-0.75 ms for the twins.
    if (rows > 1 && (euhedral_cuda_row_exact() || rows <= 8u) && query_key && gate && keys && values && output && scratch
            && query_heads && key_heads && query_heads % key_heads == 0 && head_dim == 256 && rows <= 64u
            && start <= cache_length && rows <= cache_length - start && start + rows <= UINT32_MAX
            && !euhedral_cuda_exact_numerics()) {
        /* Row-exact twins: row j is one-row decode at position start + j, all rows in one launch each for
         * decode and merge. Rows use the one-row kernel choice at their own length: the GQA kernel from
         * 2048 keys when it is loaded and the group fits, else the per-head kernel. The scratch holds one
         * one-row partial area (query_heads * 64 * 258 floats) per row. */
        int status = euhedral_cuda_bind_thread_context();
        if (status != 0) return status;
        status = ensure_initialized();
        if (status != 0) return status;
        const uint32_t group = query_heads / key_heads;
        const uint32_t from = attention_decode_gqa != NULL && group <= 8u ? 2048u : UINT32_MAX;
        if (attention_decode_rows != NULL && attention_merge_rows != NULL
                && (from == UINT32_MAX || attention_decode_gqa_rows != NULL)) {
            const uint32_t first = (uint32_t)start + 1u, last = (uint32_t)start + rows;
            unsigned long long row_stride = (unsigned long long)query_heads * 64u * 258u;
            uint32_t qh = query_heads, kh = key_heads, from_arg = from;
            unsigned long long start_arg = start;
            if (first < from) {
                uint32_t below = last < from ? last : from - 1u, splits = (below + 47u) / 48u;
                if (splits > 64u) splits = 64u;
                void* args[] = {&query_key, &keys, &values, &qh, &kh, &start_arg, &scratch, &row_stride, &from_arg};
                status = launch_and_synchronize_2d(attention_decode_rows, query_heads * splits, rows, 128, args);
                if (status != 0) return status;
            }
            if (last >= from) {
                uint32_t splits = (last + 31u) / 32u;
                if (splits > 64u) splits = 64u;
                void* args[] = {&query_key, &keys, &values, &qh, &kh, &start_arg, &scratch, &row_stride, &from_arg};
                status = launch_and_synchronize_2d(attention_decode_gqa_rows, key_heads * splits, rows, 32, args);
                if (status != 0) return status;
            }
            void* merge_args[] = {&gate, &output, &scratch, &qh, &kh, &start_arg, &row_stride, &from_arg};
            return launch_and_synchronize_2d(attention_merge_rows, query_heads, rows, 128, merge_args);
        }
    }
    if (rows > 1 && euhedral_cuda_row_exact() && query_key && gate && output) {
        /* Row-exact: row j is one-row decode at position start + j over the keys up to it. */
        const uint64_t width = ((uint64_t)query_heads + key_heads) * head_dim * 2u, out = (uint64_t)query_heads * head_dim * 2u;
        for (uint32_t row = 0; row < rows; ++row) {
            int status = euhedral_cuda_attention_causal_nvfp4((const char*)query_key + row * width, (const char*)gate + row * width,
                    keys, values, (char*)output + row * out, 1u, query_heads, key_heads, head_dim, cache_length, start + row, scratch);
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
        if (!exact && attention_prefill_fa2 != NULL && group >= 1u && group <= 8u
                && (rows >= 512u || (rows >= 128u && (uint64_t)cache_length >= 2048u))) {
            uint64_t fa2_grid = ((uint64_t)rows + 15u) / 16u * key_heads;
            if (fa2_grid <= UINT32_MAX)
                return launch_and_synchronize(attention_prefill_fa2, (uint32_t)fa2_grid, 32u * group, args);
        }
        CUfunction prefill = attention_prefill_nvfp4_exact != NULL && exact
                ? attention_prefill_nvfp4_exact : attention_prefill_nvfp4;
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
    void* args[] = {&query_key, &gate, &keys, &values, &output, &rows, &query_heads,
            &key_heads, &head_dim, &length, &start, &scratch, &splits};
    void* merge[] = {&gate, &output, &scratch, &query_heads, &key_heads, &splits};
    // Exact numerics keep the per-element (lane + 32 d) kernel.
    int exact = euhedral_cuda_exact_numerics();
    CUfunction decode = attention_decode_nvfp4_exact != NULL && exact
            ? attention_decode_nvfp4_exact : attention_decode_nvfp4;
    uint32_t decode_grid = query_heads * splits, decode_block = 128;
    // Relaxed numerics from 2048 keys: one tensor-core warp per (KV head, 32-key split, at most 64) serves
    // the KV head's whole query-head group. With the merge: 2048 keys 52.0 -> 41.7 us, 4096 keys 77 -> 54 us,
    // 16K keys 212 -> 120 us per layer. At 1024 keys it won as an operator (35.6 -> 33.6 us) but measured
    // -0.6% in decode, so shorter contexts keep the per-query-head kernel.
    uint32_t group = query_heads / key_heads;
    if (!exact && attention_decode_gqa != NULL && length >= 2048u && group <= 8u) {
        splits = (uint32_t)(((uint64_t)length + 31) / 32);
        if (splits > 64) splits = 64;
        decode = attention_decode_gqa;
        decode_grid = key_heads * splits;
        decode_block = 32;
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
