#include "euhedral_cuda.h"

#include <stddef.h>

/* The names of the Flash-Next (qwen4_exp) kernels of native/src/qwen4/kernels.cu. The Rust engine launches them
 * itself; the library reports the names, in this order. */
static const char* const kernels[] = {
    "euhedral_q4_linear_bf16",
    "euhedral_q4_grouped_rms_norm_bf16",
    "euhedral_q4_scaled_silu_bf16",
    "euhedral_q4_hc_mix_bf16",
    "euhedral_q4_hc_inject_bf16",
    "euhedral_q4_repeat_streams_bf16",
    "euhedral_q4_ngram_expand_bf16",
    "euhedral_q4_ple_gate_bf16",
    "euhedral_q4_ple_conv_bf16",
    "euhedral_q4_conv_history_bf16",
    "euhedral_q4_gdn_conv_bf16",
    "euhedral_q4_gdn_control_bf16",
    "euhedral_q4_gdn_gated_norm_bf16",
    "euhedral_q4_head_norm_rope_bf16",
    "euhedral_q4_qsa_pool_keys_bf16",
    "euhedral_q4_qsa_tail_bf16",
    "euhedral_q4_qsa_scores",
    "euhedral_q4_qsa_select",
    "euhedral_q4_qsa_attention",
    "euhedral_q4_qsa_merge",
    "euhedral_q4_qsa_kv_append",
    "euhedral_q4_router_bf16",
    "euhedral_q4_swiglu_bf16",
    "euhedral_q4_moe_finish_bf16",
    "euhedral_q4_embedding_bf16",
    "euhedral_q4_expert_gate_up_swiglu_bf16",
    "euhedral_q4_expert_down_bf16",
    "euhedral_q4_expert_combine_bf16",
    "euhedral_q4_swiglu_padded_bf16",
    "euhedral_q4_nvfp4_pad_k",
    "euhedral_q4_linear_tc_bf16",
    "euhedral_q4_linear_tc_split_bf16",
    "euhedral_q4_linear_tc_rows_bf16",
    "euhedral_q4_linear_tc_rows_split_bf16",
    "euhedral_q4_linear_split4_bf16",
    "euhedral_q4_linear_split8_bf16",
};

#define KERNEL_COUNT ((int)(sizeof(kernels) / sizeof(kernels[0])))

EUHEDRAL_CUDA_EXPORT const char* euhedral_cuda_qwen4_kernel_name(int index) {
    return index >= 0 && index < KERNEL_COUNT ? kernels[index] : NULL;
}
