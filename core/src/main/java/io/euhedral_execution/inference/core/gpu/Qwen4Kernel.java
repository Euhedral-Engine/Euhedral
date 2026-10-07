package io.euhedral_execution.inference.core.gpu;

/// The Flash-Next (qwen4_exp) kernels behind `euhedral_cuda_qwen4_launch`. The declaration order is the native
/// table's order (native/src/host/qwen4_ops.c); a test compares the names.
public enum Qwen4Kernel {
    LINEAR_BF16("euhedral_q4_linear_bf16"),
    GROUPED_RMS_NORM_BF16("euhedral_q4_grouped_rms_norm_bf16"),
    SCALED_SILU_BF16("euhedral_q4_scaled_silu_bf16"),
    HC_MIX_BF16("euhedral_q4_hc_mix_bf16"),
    HC_INJECT_BF16("euhedral_q4_hc_inject_bf16"),
    REPEAT_STREAMS_BF16("euhedral_q4_repeat_streams_bf16"),
    NGRAM_EXPAND_BF16("euhedral_q4_ngram_expand_bf16"),
    PLE_GATE_BF16("euhedral_q4_ple_gate_bf16"),
    PLE_CONV_BF16("euhedral_q4_ple_conv_bf16"),
    CONV_HISTORY_BF16("euhedral_q4_conv_history_bf16"),
    GDN_CONV_BF16("euhedral_q4_gdn_conv_bf16"),
    GDN_CONTROL_BF16("euhedral_q4_gdn_control_bf16"),
    GDN_GATED_NORM_BF16("euhedral_q4_gdn_gated_norm_bf16"),
    HEAD_NORM_ROPE_BF16("euhedral_q4_head_norm_rope_bf16"),
    QSA_POOL_KEYS_BF16("euhedral_q4_qsa_pool_keys_bf16"),
    QSA_TAIL_BF16("euhedral_q4_qsa_tail_bf16"),
    QSA_SCORES("euhedral_q4_qsa_scores"),
    QSA_SELECT("euhedral_q4_qsa_select"),
    QSA_ATTENTION("euhedral_q4_qsa_attention"),
    QSA_MERGE("euhedral_q4_qsa_merge"),
    QSA_KV_APPEND("euhedral_q4_qsa_kv_append"),
    ROUTER_BF16("euhedral_q4_router_bf16"),
    SWIGLU_BF16("euhedral_q4_swiglu_bf16"),
    MOE_FINISH_BF16("euhedral_q4_moe_finish_bf16"),
    EMBEDDING_BF16("euhedral_q4_embedding_bf16"),
    EXPERT_GATE_UP_SWIGLU_BF16("euhedral_q4_expert_gate_up_swiglu_bf16"),
    EXPERT_DOWN_BF16("euhedral_q4_expert_down_bf16"),
    EXPERT_COMBINE_BF16("euhedral_q4_expert_combine_bf16"),
    SWIGLU_PADDED_BF16("euhedral_q4_swiglu_padded_bf16"),
    NVFP4_PAD_K("euhedral_q4_nvfp4_pad_k"),
    LINEAR_TC_BF16("euhedral_q4_linear_tc_bf16"),
    LINEAR_TC_SPLIT_BF16("euhedral_q4_linear_tc_split_bf16"),
    LINEAR_TC_ROWS_BF16("euhedral_q4_linear_tc_rows_bf16"),
    LINEAR_TC_ROWS_SPLIT_BF16("euhedral_q4_linear_tc_rows_split_bf16");

    private final String symbol;

    Qwen4Kernel(String symbol) {
        this.symbol = symbol;
    }

    /// The kernel's name in the CUDA source.
    public String symbol() {
        return this.symbol;
    }
}
