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
    GDN_GATED_NORM_BF16("euhedral_q4_gdn_gated_norm_bf16");

    private final String symbol;

    Qwen4Kernel(String symbol) {
        this.symbol = symbol;
    }

    /// The kernel's name in the CUDA source.
    public String symbol() {
        return this.symbol;
    }
}
