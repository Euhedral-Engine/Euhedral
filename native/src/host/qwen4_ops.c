#include "kernel_table.h"

/* The Flash-Next (qwen4_exp) kernels of native/src/qwen4/kernels.cu behind one launch entry point.
 *
 * The model's operators are many small kernels that differ only in their arguments, so instead of one C wrapper and
 * one FFM descriptor each, the host exposes a table of kernels and a single launcher. A launch passes the kernel's
 * index, its geometry and its arguments as 8-byte words with the byte size (4 or 8) each argument has in the kernel's
 * signature. The launcher (kernel_table.c) checks the argument count and every size against the compiled kernel
 * (cuFuncGetParamInfo), so a mismatch between the Java caller and the CUDA source fails the call instead of
 * corrupting it. The order of `kernels` is the ABI: model.qwen4.Kernel on the Java side lists the same names in
 * the same order and a test compares them. */
static const EuhedralKernelEntry kernels[] = {
    {"euhedral_q4_linear_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_grouped_rms_norm_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_scaled_silu_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_hc_mix_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_hc_inject_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_repeat_streams_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_ngram_expand_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_ple_gate_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_ple_conv_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_conv_history_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_gdn_conv_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_gdn_control_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_gdn_gated_norm_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_head_norm_rope_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_qsa_pool_keys_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_qsa_tail_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_qsa_scores", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_qsa_select", 0, "qwen4/kernels.cu"},
    /* One unit's three warps (q4qsa::kUnitSharedBytes). */
    {"euhedral_q4_qsa_attention", 28224, "qwen4/kernels.cu"},
    {"euhedral_q4_qsa_merge", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_qsa_kv_append", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_router_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_swiglu_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_moe_finish_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_embedding_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_expert_gate_up_swiglu_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_expert_down_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_expert_combine_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_swiglu_padded_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_nvfp4_pad_k", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_linear_tc_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_linear_tc_split_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_linear_tc_rows_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_linear_tc_rows_split_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_linear_split4_bf16", 0, "qwen4/kernels.cu"},
    {"euhedral_q4_linear_split8_bf16", 0, "qwen4/kernels.cu"},
};

static EuhedralKernelTable table = EUHEDRAL_KERNEL_TABLE_INIT(kernels);
EUHEDRAL_KERNEL_TABLE_READY(table_ready, table)

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_qwen4_kernel_count(void) {
    return table.count;
}

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_qwen4_kernel_available(int32_t kernel) {
    return euhedral_kernel_table_available(&table, table_ready, kernel);
}

EUHEDRAL_CUDA_EXPORT const char* euhedral_cuda_qwen4_kernel_name(int index) {
    return index >= 0 && index < table.count ? kernels[index].name : NULL;
}

EUHEDRAL_CUDA_EXPORT int euhedral_cuda_qwen4_launch(int32_t kernel, uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
        uint32_t block_x, uint32_t block_y, uint32_t block_z, uint32_t shared_bytes,
        const uint64_t* words, const uint8_t* sizes, int32_t count) {
    return euhedral_kernel_table_launch(&table, table_ready, kernel, grid_x, grid_y, grid_z, block_x, block_y, block_z,
            shared_bytes, words, sizes, count);
}
