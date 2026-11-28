//! The kernels of the Flash-Next table the dense part, the
//! sparse attention and the mixture of experts launch. The discriminant is the kernel's index in the table, which is its ABI; a test compares
//! every name with the catalog's.

use euhedral_inference_cuda::launch::catalog::qwen4_table;

use crate::formats::shapes::{KernelArguments, Launch, Table};

/// A kernel of the Flash-Next table, by its table index.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(i32)]
pub enum Qwen4Kernel {
    LinearBf16 = 0,
    GroupedRmsNorm = 1,
    ScaledSilu = 2,
    HcMix = 3,
    HcInject = 4,
    RepeatStreams = 5,
    NgramExpand = 6,
    PleGate = 7,
    PleConv = 8,
    ConvHistory = 9,
    GdnConv = 10,
    GdnControl = 11,
    GdnGatedNorm = 12,
    HeadNormRope = 13,
    QsaPoolKeys = 14,
    QsaTail = 15,
    QsaScores = 16,
    QsaSelect = 17,
    QsaAttention = 18,
    QsaMerge = 19,
    QsaKvAppend = 20,
    Router = 21,
    Swiglu = 22,
    MoeFinish = 23,
    Embedding = 24,
    ExpertGateUpSwiglu = 25,
    ExpertDown = 26,
    ExpertCombine = 27,
    SwigluPadded = 28,
    Nvfp4PadK = 29,
    LinearTc = 30,
    LinearTcSplit = 31,
    LinearTcRows = 32,
    LinearTcRowsSplit = 33,
    LinearSplit4 = 34,
    LinearSplit8 = 35,
}

impl Qwen4Kernel {
    /// Every kernel the engine launches from the table.
    pub const ALL: [Qwen4Kernel; 36] = [
        Qwen4Kernel::LinearBf16,
        Qwen4Kernel::GroupedRmsNorm,
        Qwen4Kernel::ScaledSilu,
        Qwen4Kernel::HcMix,
        Qwen4Kernel::HcInject,
        Qwen4Kernel::RepeatStreams,
        Qwen4Kernel::NgramExpand,
        Qwen4Kernel::PleGate,
        Qwen4Kernel::PleConv,
        Qwen4Kernel::ConvHistory,
        Qwen4Kernel::GdnConv,
        Qwen4Kernel::GdnControl,
        Qwen4Kernel::GdnGatedNorm,
        Qwen4Kernel::HeadNormRope,
        Qwen4Kernel::QsaPoolKeys,
        Qwen4Kernel::QsaTail,
        Qwen4Kernel::QsaScores,
        Qwen4Kernel::QsaSelect,
        Qwen4Kernel::QsaAttention,
        Qwen4Kernel::QsaMerge,
        Qwen4Kernel::QsaKvAppend,
        Qwen4Kernel::Router,
        Qwen4Kernel::Swiglu,
        Qwen4Kernel::MoeFinish,
        Qwen4Kernel::Embedding,
        Qwen4Kernel::ExpertGateUpSwiglu,
        Qwen4Kernel::ExpertDown,
        Qwen4Kernel::ExpertCombine,
        Qwen4Kernel::SwigluPadded,
        Qwen4Kernel::Nvfp4PadK,
        Qwen4Kernel::LinearTc,
        Qwen4Kernel::LinearTcSplit,
        Qwen4Kernel::LinearTcRows,
        Qwen4Kernel::LinearTcRowsSplit,
        Qwen4Kernel::LinearSplit4,
        Qwen4Kernel::LinearSplit8,
    ];

    /// The kernel's index in the table.
    pub fn index(self) -> i32 {
        self as i32
    }

    /// The kernel's name.
    pub fn symbol(self) -> &'static str {
        match self {
            Qwen4Kernel::LinearBf16 => "euhedral_q4_linear_bf16",
            Qwen4Kernel::GroupedRmsNorm => "euhedral_q4_grouped_rms_norm_bf16",
            Qwen4Kernel::ScaledSilu => "euhedral_q4_scaled_silu_bf16",
            Qwen4Kernel::HcMix => "euhedral_q4_hc_mix_bf16",
            Qwen4Kernel::HcInject => "euhedral_q4_hc_inject_bf16",
            Qwen4Kernel::RepeatStreams => "euhedral_q4_repeat_streams_bf16",
            Qwen4Kernel::NgramExpand => "euhedral_q4_ngram_expand_bf16",
            Qwen4Kernel::PleGate => "euhedral_q4_ple_gate_bf16",
            Qwen4Kernel::PleConv => "euhedral_q4_ple_conv_bf16",
            Qwen4Kernel::ConvHistory => "euhedral_q4_conv_history_bf16",
            Qwen4Kernel::GdnConv => "euhedral_q4_gdn_conv_bf16",
            Qwen4Kernel::GdnControl => "euhedral_q4_gdn_control_bf16",
            Qwen4Kernel::GdnGatedNorm => "euhedral_q4_gdn_gated_norm_bf16",
            Qwen4Kernel::HeadNormRope => "euhedral_q4_head_norm_rope_bf16",
            Qwen4Kernel::QsaPoolKeys => "euhedral_q4_qsa_pool_keys_bf16",
            Qwen4Kernel::QsaTail => "euhedral_q4_qsa_tail_bf16",
            Qwen4Kernel::QsaScores => "euhedral_q4_qsa_scores",
            Qwen4Kernel::QsaSelect => "euhedral_q4_qsa_select",
            Qwen4Kernel::QsaAttention => "euhedral_q4_qsa_attention",
            Qwen4Kernel::QsaMerge => "euhedral_q4_qsa_merge",
            Qwen4Kernel::QsaKvAppend => "euhedral_q4_qsa_kv_append",
            Qwen4Kernel::Router => "euhedral_q4_router_bf16",
            Qwen4Kernel::Swiglu => "euhedral_q4_swiglu_bf16",
            Qwen4Kernel::MoeFinish => "euhedral_q4_moe_finish_bf16",
            Qwen4Kernel::Embedding => "euhedral_q4_embedding_bf16",
            Qwen4Kernel::ExpertGateUpSwiglu => "euhedral_q4_expert_gate_up_swiglu_bf16",
            Qwen4Kernel::ExpertDown => "euhedral_q4_expert_down_bf16",
            Qwen4Kernel::ExpertCombine => "euhedral_q4_expert_combine_bf16",
            Qwen4Kernel::SwigluPadded => "euhedral_q4_swiglu_padded_bf16",
            Qwen4Kernel::Nvfp4PadK => "euhedral_q4_nvfp4_pad_k",
            Qwen4Kernel::LinearTc => "euhedral_q4_linear_tc_bf16",
            Qwen4Kernel::LinearTcSplit => "euhedral_q4_linear_tc_split_bf16",
            Qwen4Kernel::LinearTcRows => "euhedral_q4_linear_tc_rows_bf16",
            Qwen4Kernel::LinearTcRowsSplit => "euhedral_q4_linear_tc_rows_split_bf16",
            Qwen4Kernel::LinearSplit4 => "euhedral_q4_linear_split4_bf16",
            Qwen4Kernel::LinearSplit8 => "euhedral_q4_linear_split8_bf16",
        }
    }

    /// The dynamic shared bytes the catalog opts the kernel in to: what its launches ask for (0 for a kernel that
    /// uses none).
    pub fn shared_bytes(self) -> u32 {
        qwen4_table()[self.index() as usize].shared_opt_in_bytes
    }

    /// A launch of this kernel with no dynamic shared memory (Java launches none).
    pub fn launch(self, grid: [u32; 3], block: [u32; 3], args: KernelArguments) -> Launch {
        self.launch_shared(grid, block, 0, args)
    }

    /// A launch of this kernel with `shared_bytes` of dynamic shared memory.
    pub fn launch_shared(self, grid: [u32; 3], block: [u32; 3], shared_bytes: u32, args: KernelArguments) -> Launch {
        Launch {
            table: Table::Qwen4,
            kernel: self.index(),
            symbol: self.symbol(),
            grid,
            block,
            shared_bytes,
            args,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_kernel_is_the_catalogs_entry_at_its_index() {
        for kernel in Qwen4Kernel::ALL {
            let entry = &qwen4_table()[kernel.index() as usize];
            assert_eq!(entry.name, kernel.symbol(), "table index {}", kernel.index());
        }
    }
}
