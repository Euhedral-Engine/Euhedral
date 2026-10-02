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

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_q3_bf16(
        const void* device_input,
        const void* device_weights,
        void* device_output,
        uint32_t rows,
        uint32_t in_features,
        uint32_t out_features,
        uint64_t weights_byte_size);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_q3_decode_bf16(
        const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size);
/* Selects exact numerics (nonzero) for later launches in this process and returns the previous
 * selection: every kernel with a relaxed FP32 accumulation order (contiguous Q3 decode, split-K FFN
 * down, ...) is replaced by its bitwise-exact counterpart. The default is relaxed unless the
 * environment sets EUHEDRAL_EXACT=1 (or EUHEDRAL_Q3_DECODE=EXACT). The second name is an alias. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_select_exact_numerics(int exact);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_q3_decode_select_exact(int exact);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_q3_prefill_bf16(
        const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_linear_q3_prefill_64_bf16(
        const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t in_features, uint32_t out_features, uint64_t weights_byte_size);

/* P2E2 Q3 tensors (layout row-split-p2e2-v1, docs/COMPRESSED_Q3.md): the same Q3G64_F16S values in
 * a smaller, entropy-coded layout. The decode route runs one row with relaxed numerics on the shapes
 * of euhedral_cuda_linear_q3_decode_bf16's contiguous kernel, bitwise identical to it, and returns
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

/* Split-K FFN down for the measured prefill shapes: K splits accumulate into `partials`
 * (splits x rows x outputs FP32, splits = 4) and a reduction writes BF16 `output`. Other shapes, or
 * a NULL `partials`, run euhedral_cuda_q3_ffn_down_bf16. FP32 order differs from the unsplit leaf. */
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_q3_ffn_down_split_bf16(
        const void* input, const void* weights, void* output, float* partials,
        uint32_t rows, uint32_t width, uint32_t outputs, uint64_t weight_bytes);
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_q3_ffn_down_bf16(
        const void* input, const void* weights, void* output,
        uint32_t rows, uint32_t width, uint32_t outputs, uint64_t weight_bytes);

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

// Owns internal streams even with synchronous outer submission. On ANY error, the caller
// must prove device completion or quarantine all borrowed buffers before releasing them.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_q3_ffn_streamed_bf16(
        const void* input, const void* gate_weights, const void* down_weights, void* output,
        void* slots, float* accumulators, uint32_t rows, uint32_t hidden, uint32_t intermediate,
        uint64_t gate_bytes, uint64_t down_bytes);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_q3_gate_up_swiglu_bf16(
        const void* input, const void* weights, void* output, uint32_t rows,
        uint32_t width, uint32_t outputs, uint64_t weight_bytes);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_residual_rms_norm_bf16(
        const void* residual, const void* delta, const void* weight,
        void* hidden, void* normalized, uint32_t rows, uint32_t width, float epsilon);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_gdn_project_control_fp32(
        const void* input, const void* a_weight, const void* b_weight,
        const float* a_log, const float* dt_bias, float* alpha, float* beta,
        uint32_t rows, uint32_t width, uint32_t heads);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_gdn_projections_bf16(
        const void* input, const void* q4, const void* q5, void* qk_output, void* value_z_output,
        uint32_t rows, uint32_t hidden, uint32_t qk_width, uint32_t value_z_width,
        uint64_t q4_bytes, uint64_t q5_bytes);

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
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_attention_kv_append_nvfp4(
        const void* query_key, const void* gate, void* keys, void* values,
        uint32_t rows, uint32_t query_width, uint32_t key_width, uint64_t start);
// Scratch: query_heads * 64 * 258 FP32 elements for single-token decode; null for prefill.
EUHEDRAL_CUDA_EXPORT int euhedral_cuda_attention_causal_nvfp4(
        const void* query_key, const void* gate, const void* keys, const void* values, void* output,
        uint32_t rows, uint32_t query_heads, uint32_t key_heads, uint32_t head_dim,
        uint32_t cache_length, uint64_t start, void* scratch);

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
        float epsilon,
        double rope_theta);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_attention_kv_append_bf16(
        const void* device_query_key,
        const void* device_gate_value,
        void* device_key_cache,
        void* device_value_cache,
        uint32_t rows,
        uint32_t query_width,
        uint32_t key_value_width,
        uint64_t start_position);

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_attention_causal_bf16(
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
        uint64_t start_position);

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
