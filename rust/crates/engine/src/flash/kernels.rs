//! The kernels of the Flash-Next table (`euhedral_cuda_qwen4_launch`, native/src/host/qwen4_ops.c) the dense part
//! launches. The discriminant is the kernel's index in the table, which is its ABI; a test compares every name with the
//! catalog's.

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
    Embedding = 24,
    LinearTc = 30,
    LinearTcSplit = 31,
    LinearTcRows = 32,
    LinearTcRowsSplit = 33,
    LinearSplit4 = 34,
    LinearSplit8 = 35,
}

impl Qwen4Kernel {
    /// Every kernel the dense part launches.
    pub const ALL: [Qwen4Kernel; 20] = [
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
        Qwen4Kernel::Embedding,
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
            Qwen4Kernel::Embedding => "euhedral_q4_embedding_bf16",
            Qwen4Kernel::LinearTc => "euhedral_q4_linear_tc_bf16",
            Qwen4Kernel::LinearTcSplit => "euhedral_q4_linear_tc_split_bf16",
            Qwen4Kernel::LinearTcRows => "euhedral_q4_linear_tc_rows_bf16",
            Qwen4Kernel::LinearTcRowsSplit => "euhedral_q4_linear_tc_rows_split_bf16",
            Qwen4Kernel::LinearSplit4 => "euhedral_q4_linear_split4_bf16",
            Qwen4Kernel::LinearSplit8 => "euhedral_q4_linear_split8_bf16",
        }
    }

    /// A launch of this kernel with no dynamic shared memory (Java launches none).
    pub fn launch(self, grid: [u32; 3], block: [u32; 3], args: KernelArguments) -> Launch {
        Launch {
            table: Table::Qwen4,
            kernel: self.index(),
            symbol: self.symbol(),
            grid,
            block,
            shared_bytes: 0,
            args,
        }
    }
}

#[cfg(test)]
mod tests {
    use euhedral_inference_cuda::launch::catalog::qwen4_table;

    use super::*;

    #[test]
    fn every_kernel_is_the_catalogs_entry_at_its_index() {
        for kernel in Qwen4Kernel::ALL {
            let entry = &qwen4_table()[kernel.index() as usize];
            assert_eq!(entry.name, kernel.symbol(), "table index {}", kernel.index());
        }
    }
}
