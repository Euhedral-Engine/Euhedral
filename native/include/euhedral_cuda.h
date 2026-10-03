#ifndef EUHEDRAL_CUDA_H
#define EUHEDRAL_CUDA_H

#include <stdint.h>

/* Thread-local launch selection; NULL preserves the synchronous ABI. */
void* euhedral_cuda_submission_stream(void);

#ifdef _WIN32
#define EUHEDRAL_CUDA_EXPORT __declspec(dllexport)
#else
#define EUHEDRAL_CUDA_EXPORT __attribute__((visibility("default")))
#endif

#define EUHEDRAL_CUDA_SUCCESS 0
#define EUHEDRAL_CUDA_INVALID_ARGUMENT (-1)
#define EUHEDRAL_CUDA_SIZE_OVERFLOW (-2)
#define EUHEDRAL_CUDA_FORMAT_MISMATCH (-3)
#define EUHEDRAL_CUDA_KERNEL_UNAVAILABLE (-4)
/* The requested specialized route does not apply (numerics, shape or alignment); use another. */
#define EUHEDRAL_CUDA_ROUTE_UNAVAILABLE (-5)

#ifdef __cplusplus
extern "C" {
#endif

EUHEDRAL_CUDA_EXPORT void* euhedral_cuda_malloc(uint64_t byte_size);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_free(void* address);
EUHEDRAL_CUDA_EXPORT void* euhedral_cuda_host_malloc(uint64_t byte_size);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_host_free(void* address);
/// Pinned, huge-page-backed host memory for weights that are staged to the device on use.
EUHEDRAL_CUDA_EXPORT void* euhedral_cuda_host_weights_malloc(uint64_t byte_size);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_host_weights_free(void* address);
/// The device address at which kernels read host weights in place (zero-copy, over PCIe).
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_host_weights_device_pointer(const void* host_address, uint64_t* device_address);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_device_memory_info(uint64_t* free_byte_size, uint64_t* total_byte_size);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_copy_host_to_device(
        void* device_address,
        const void* host_address,
        uint64_t byte_size);
/// Requires a pinned host allocation retained through stream completion; the copy is queued on the
/// selected stream. With no stream selected it completes before returning.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_copy_upload_to_device(
        void* device_address,
        const void* host_address,
        uint64_t byte_size);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_copy_device_to_host(
        void* host_address,
        const void* device_address,
        uint64_t byte_size);

/// Requires a pinned host destination retained, and read, only after the selected stream's work
/// retires; the copy is queued on that stream. With no stream selected it completes before returning.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_copy_device_to_readback(
        void* host_address,
        const void* device_address,
        uint64_t byte_size);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_copy_device_to_device(
        void* destination_address,
        const void* source_address,
        uint64_t byte_size);

/// Copies `rows` rows of `row_bytes` between pitched device regions; queued like
/// euhedral_cuda_copy_device_to_device.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_copy_device_to_device_2d(
        void* destination_address,
        uint64_t destination_pitch,
        const void* source_address,
        uint64_t source_pitch,
        uint64_t row_bytes,
        uint64_t rows);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_embed_q3(
        const int32_t* device_token_ids,
        const void* device_embedding_weights,
        void* device_hidden_state,
        uint32_t token_count,
        uint32_t vocabulary_size,
        uint32_t hidden_size,
        uint64_t embedding_byte_size);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_rms_norm_bf16(
        const void* device_input,
        const void* device_weight,
        void* device_output,
        uint32_t rows,
        uint32_t width,
        float epsilon);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_rms_norm_unit_offset_bf16(
        const void* device_input,
        const void* device_weight,
        void* device_output,
        uint32_t rows,
        uint32_t width,
        float epsilon);

/* Q3G64_F16S linear on BF16 rows, the production dispatch: 1 to 8 rows on the contiguous decode kernels (every row
 * bit for bit as a one-row call). Larger batches run on the FP8 route below; exact numerics, and shapes or alignments
 * no decode kernel takes, run the scalar reference. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_q3_bf16(
        const void* device_input,
        const void* device_weights,
        void* device_output,
        uint32_t rows,
        uint32_t in_features,
        uint32_t out_features,
        uint64_t weights_byte_size);
/* The scalar Q3 reference: the numerical oracle the production kernels are measured against. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_q3_reference_bf16(
        const void* device_input,
        const void* device_weights,
        void* device_output,
        uint32_t rows,
        uint32_t in_features,
        uint32_t out_features,
        uint64_t weights_byte_size);
/* Selects exact numerics (nonzero) for later launches in this process and returns the previous selection: every
 * quantized linear runs its scalar reference and every relaxed-order operator its exact twin. The default is relaxed.
 * For numerical comparisons against the oracle. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_select_exact_numerics(int exact);

/* NVFP4 weights (Nvfp4Layout) linear on BF16 rows, `in_features` x `out_features` (weight rows): one row on the decode
 * kernel, 2 to 8 rows on its row twins (each bit for bit as a one-row call), exact numerics and other shapes on the
 * scalar reference. Larger batches run on the native route below. FP32 accumulation. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_nvfp4_bf16(
        const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size);
/* Native Blackwell NVFP4 linear (docs/NVFP4_NATIVE.md): quantizes the BF16 input rows to two NVFP4 terms in `scratch`
 * (euhedral_cuda_nvfp4_native_scratch_bytes) and multiplies with block-scaled FP4 tensor-core MMA. Returns
 * EUHEDRAL_CUDA_ROUTE_UNAVAILABLE under exact numerics or when in_features is not a multiple of 128; the activations
 * are quantized, so results differ from euhedral_cuda_linear_nvfp4_bf16. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_nvfp4_native_available(void);
/* Scratch for one native linear or gate/up region: quantized activations, plus FP32 split-K partials for
 * decode-like row counts (2 to 64 rows run the skinny weight-streaming kernel). */
EUHEDRAL_CUDA_EXPORT uint64_t euhedral_cuda_nvfp4_native_scratch_bytes(
        uint32_t rows, uint32_t in_features, uint32_t out_features);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_nvfp4_native_bf16(
        const void* input, const void* weights, void* output, void* scratch,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size,
        uint64_t scratch_byte_size);
/* The paired gate/up region on native NVFP4: `outputs` weight rows (gate first), outputs / 2 SwiGLU
 * values per row, gate and up rounded to BF16 before SwiGLU. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_nvfp4_native_gate_up_swiglu_bf16(
        const void* input, const void* weights, void* output, void* scratch,
        uint32_t rows, uint32_t width, uint32_t outputs, uint64_t weight_bytes, uint64_t scratch_byte_size);

/* Prefill (9 or more rows) on block-scaled FP8 tensor cores (docs/PREFILL_MX.md): quantizes the BF16 input rows into two E4M3
 * terms and a power-of-two block scale in `scratch` (euhedral_cuda_q3_mx_scratch_bytes, 256-aligned) and multiplies
 * with the Q3 codes (exact in E4M3) at the full FP8 MMA rate, FP32 accumulation, FP16 group scales applied in
 * FP32. Returns EUHEDRAL_CUDA_ROUTE_UNAVAILABLE off sm_12x, under exact numerics or row-exact execution, for
 * unaligned operands, and when in_features or the weight rows are not a multiple of 128. The BF16 activations are
 * represented exactly (down to 2^-9 of a 32-block's maximum). */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_q3_mx_available(void);
EUHEDRAL_CUDA_EXPORT uint64_t euhedral_cuda_q3_mx_scratch_bytes(uint32_t rows, uint32_t width, uint32_t weight_rows);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_q3_mx_bf16(
        const void* input, const void* weights, void* output, void* scratch,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size,
        uint64_t scratch_byte_size);
/* Names, for the calling thread, the number of weight rows of the whole tensor a linear's output-row chunk belongs to
 * (0 clears it), so the split-K choice, and with it the summation order, is that of the whole tensor. */
EUHEDRAL_CUDA_EXPORT void euhedral_cuda_q3_mx_select_split_rows(uint32_t weight_rows);
/* Q4 and Q5 tensors (bits = 4 or 5) on the same route. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_q45_mx_bf16(
        int bits, const void* input, const void* weights, void* output, void* scratch,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size,
        uint64_t scratch_byte_size);
/* The paired gate/up region: `outputs` weight rows (gate first), outputs / 2 SwiGLU values per row. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_q3_mx_gate_up_swiglu_bf16(
        const void* input, const void* weights, void* output, void* scratch,
        uint32_t rows, uint32_t width, uint32_t outputs, uint64_t weight_bytes, uint64_t scratch_byte_size);

/* P2E2 Q3 tensors (layout row-split-p2e2-v1, docs/COMPRESSED_Q3.md): the same Q3G64_F16S values in
 * a smaller, entropy-coded layout. The decode route runs one row on the shapes of the contiguous decode
 * kernel, bitwise identical to it, and returns
 * EUHEDRAL_CUDA_ROUTE_UNAVAILABLE otherwise. Every other route expands the tensor into the row-split
 * layout (`destination` holds at least that many bytes) and runs the row-split entry points. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_q3_p2e2_decode_bf16(
        const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size);
/* Expands rows [first_row, first_row + row_count) into a row-split tensor of row_count rows. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_q3_p2e2_expand(
        const void* weights, uint64_t weights_byte_size, uint32_t rows, uint32_t in_features,
        uint32_t first_row, uint32_t row_count, void* destination, uint64_t destination_byte_size);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_embed_q3_p2e2(
        const int32_t* device_token_ids,
        const void* device_embedding_weights,
        void* device_hidden_state,
        uint32_t token_count,
        uint32_t vocabulary_size,
        uint32_t hidden_size,
        uint64_t embedding_byte_size);

/* Q4 (bits = 4) and Q5 (bits = 5) linear: 1 to 8 rows on the contiguous decode kernels (every row bit for bit as a
 * one-row call); exact numerics and other shapes on the scalar reference. Larger batches run on the FP8 route. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_quantized_bf16(
        const void* device_input,
        const void* device_weights,
        void* device_output,
        uint32_t rows,
        uint32_t in_features,
        uint32_t out_features,
        uint64_t weights_byte_size,
        uint32_t bits);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_bf16_to_float(
        const void* device_input,
        const void* device_weights,
        void* device_output,
        uint32_t rows,
        uint32_t in_features,
        uint32_t out_features);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_gdn_control_fp32(
        const float* device_a_projection,
        const float* device_b_projection,
        const float* device_a_log,
        const float* device_dt_bias,
        float* device_alpha_output,
        float* device_beta_output,
        uint32_t rows,
        uint32_t heads);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_gdn_convolution_bf16(
        const void* device_query_key,
        const void* device_value_z,
        const void* device_convolution_weights,
        void* device_convolution_state,
        void* device_output,
        uint32_t rows,
        uint32_t query_key_width,
        uint32_t value_width,
        uint32_t convolution_width,
        uint32_t kernel_size);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_gdn_recurrence_bf16(
        const void* device_convolved,
        const float* device_alpha,
        const float* device_beta,
        float* device_recurrent_state,
        void* device_output,
        uint32_t rows,
        uint32_t key_heads,
        uint32_t value_heads,
        uint32_t key_head_dim,
        uint32_t value_head_dim,
        float output_scale);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_gdn_gated_rms_norm_bf16(
        const void* device_recurrent,
        const void* device_value_z,
        const void* device_norm_weight,
        void* device_output,
        uint32_t rows,
        uint32_t value_heads,
        uint32_t head_dim,
        float epsilon);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_residual_rms_norm_bf16(
        const void* residual, const void* delta, const void* weight,
        void* hidden, void* normalized, uint32_t rows, uint32_t width, float epsilon);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_gdn_project_control_fp32(
        const void* input, const void* a_weight, const void* b_weight,
        const float* a_log, const float* dt_bias, float* alpha, float* beta,
        uint32_t rows, uint32_t width, uint32_t heads);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_residual_add_bf16(
        const void* device_residual,
        const void* device_delta,
        void* device_output,
        uint32_t rows,
        uint32_t width);

/* Greedy selection over `count` BF16 logits: writes one 64-bit key to `device_result` whose low word
 * is 0xFFFFFFFF minus the selected token ID (the lowest ID among equal maxima, never NaN or negative
 * infinity), or 0 when no logit is selectable. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_argmax_bf16(const void* device_logits, uint32_t count, void* device_result);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_swiglu_bf16(
        const void* device_gate_up,
        void* device_output,
        uint32_t rows,
        uint32_t intermediate_size);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_zero_device_memory(void* device_address, uint64_t byte_size);

// NVFP4 K/V arguments are device page-address tables. Pages contain 256 tokens,
// token-major heads, with 128 E2M1 code bytes and 16 E4M3 scales per D256 head.
// Position-dependent kernels read the quantum's start position from `position` (one uint64 in device memory,
// written before the quantum's first launch); `start` is the host's copy, which validates and sizes the launch.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_attention_kv_append_nvfp4(
        const void* query_key, const void* gate, void* keys, void* values,
        uint32_t rows, uint32_t query_width, uint32_t key_width, uint64_t start, const void* position);
// Scratch: query_heads * 64 * 258 FP32 elements for single-token decode; null for prefill.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_attention_causal_nvfp4(
        const void* query_key, const void* gate, const void* keys, const void* values, void* output,
        uint32_t rows, uint32_t query_heads, uint32_t key_heads, uint32_t head_dim,
        uint32_t cache_length, uint64_t start, const void* position, void* scratch);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_attention_qk_norm_rope_bf16(
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
        const void* position,
        float epsilon,
        double rope_theta);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_synchronize(void);
EUHEDRAL_CUDA_EXPORT uint64_t euhedral_cuda_stream_create(void);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_stream_destroy(uint64_t stream);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_stream_synchronize(uint64_t stream);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_stream_select(uint64_t stream);
EUHEDRAL_CUDA_EXPORT void euhedral_cuda_stream_clear(void);
EUHEDRAL_CUDA_EXPORT uint64_t euhedral_cuda_completion_event_create(void);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_completion_event_record(uint64_t event, uint64_t stream);
/// Orders work submitted to `stream` afterwards behind the work recorded in `event` (device side).
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_stream_wait_event(uint64_t stream, uint64_t event);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_completion_event_query(uint64_t event);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_completion_event_destroy(uint64_t event);
/// Schedules a notification after preceding stream work and the recorded event.
/// The callback must not call CUDA APIs or finalize frames.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_completion_notify(
        uint64_t stream, void (*callback)(uint64_t, int), uint64_t token);

#ifdef __cplusplus
}
#endif

#endif
