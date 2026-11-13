//! The catalog of every kernel the Rust engine launches, and the planned launch the family files produce.
//!
//! One [`KernelSpec`] per kernel, in a fixed order: the entries of the shared linear table, then those of the
//! Flash-Next table (each in the table's order, which is the native ABI), then the other families. A kernel is named by
//! its [`KernelId`], its index in [`KERNELS`]. The fields are what the native library does for the kernel today: the
//! name it registers, the source it compiles from, the architecture-specific compile target of that source, the
//! dynamic shared memory it sets as the function's maximum, and whether it registers the kernel for programmatic
//! dependent launch (`euhedral_cuda_pdl_register`).
//!
//! The families are the ones the Rust engine calls (`ops.rs`). The native library has more kernels (the Q3 and Q4/Q5
//! linears, the multi-expert GEMMs, the drafter's, `euhedral_linear_bf16_to_float`, `euhedral_gdn_control_fp32`,
//! `euhedral_bf16_embedding`, the native gate-up SwiGLU); they stay out until the engine calls them.

use crate::DevicePtr;

/// The most arguments one launch takes: the native launcher's limit (`EUHEDRAL_KERNEL_TABLE_MAX_PARAMETERS`).
pub const MAX_ARGUMENTS: usize = 32;

/// The source root a kernel compiles from: a file under the installed kernel directory, as the native loader names it.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Module {
    Trellis,
    /// Block-scaled FP8 tensor cores; compiled for the architecture-specific target.
    TrellisFp8,
    Embedding,
    Qwen4,
    Elementwise,
    Norm,
    Sampling,
    Gdn,
    Attention,
    Nvfp4,
    /// Native Blackwell NVFP4; compiled for the architecture-specific target.
    Nvfp4Native,
    Reference,
}

impl Module {
    /// The source root (the NVRTC program name).
    pub const fn root(self) -> &'static str {
        match self {
            Module::Trellis => "trellis/kernels.cu",
            Module::TrellisFp8 => "trellis_fp8/kernels.cu",
            Module::Embedding => "embedding/kernels.cu",
            Module::Qwen4 => "qwen4/kernels.cu",
            Module::Elementwise => "elementwise/kernels.cu",
            Module::Norm => "norm/kernels.cu",
            Module::Sampling => "sampling/kernels.cu",
            Module::Gdn => "gdn/kernels.cu",
            Module::Attention => "attention/kernels.cu",
            Module::Nvfp4 => "nvfp4/kernels.cu",
            Module::Nvfp4Native => "nvfp4_native/kernels.cu",
            Module::Reference => "reference/kernels.cu",
        }
    }

    /// Whether the module needs the architecture-specific target (`sm_XYa`), which exists on compute capability 12.x
    /// only; on another device its kernels are unavailable.
    pub const fn architecture_specific(self) -> bool {
        matches!(self, Module::TrellisFp8 | Module::Nvfp4Native)
    }
}

/// The group a kernel belongs to; the family files plan one group each (the two tables share one file).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// The shared linear table (`euhedral_cuda_linear_launch`).
    LinearTable,
    /// The Flash-Next table (`euhedral_cuda_qwen4_launch`).
    Qwen4Table,
    /// Residual, SwiGLU, RMS norm, argmax and selected token.
    Elementwise,
    Gdn,
    Attention,
    /// The NVFP4 linears: decode, reference and native.
    Nvfp4,
    /// The embedding lookups outside the linear table.
    Embedding,
}

/// One kernel, as the native library has it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KernelSpec {
    pub family: Family,
    /// The kernel's name, as the native library registers it.
    pub name: &'static str,
    pub module: Module,
    /// The dynamic shared bytes the native library sets as the function's maximum (0 when it sets none). Some are
    /// below the 48 KiB default, which the native library sets all the same.
    pub shared_opt_in_bytes: u32,
    /// Whether the native library registers the kernel for programmatic dependent launch.
    pub pdl_registered: bool,
}

impl KernelSpec {
    /// Whether the kernel's module needs the architecture-specific compile target.
    pub const fn architecture_specific(&self) -> bool {
        self.module.architecture_specific()
    }
}

/// A kernel of [`KERNELS`], by index.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct KernelId(u16);

impl KernelId {
    /// The index in [`KERNELS`].
    pub const fn index(self) -> usize {
        self.0 as usize
    }

    pub const fn spec(self) -> &'static KernelSpec {
        &SPECS[self.0 as usize]
    }

    pub const fn name(self) -> &'static str {
        self.spec().name
    }
}

const fn spec(
    family: Family,
    name: &'static str,
    module: Module,
    shared_opt_in_bytes: u32,
    pdl_registered: bool,
) -> KernelSpec {
    KernelSpec {
        family,
        name,
        module,
        shared_opt_in_bytes,
        pdl_registered,
    }
}

/// The kernel called `name`; in a constant it fails the build when there is none.
pub const fn find(name: &str) -> KernelId {
    match lookup(name) {
        Some(id) => id,
        None => panic!("no kernel of this name in the catalog"),
    }
}

/// The kernel called `name`, if there is one.
pub const fn lookup(name: &str) -> Option<KernelId> {
    let wanted = name.as_bytes();
    let mut index = 0;
    while index < SPECS.len() {
        let candidate = SPECS[index].name.as_bytes();
        if candidate.len() == wanted.len() {
            let mut at = 0;
            while at < wanted.len() && candidate[at] == wanted[at] {
                at += 1;
            }
            if at == wanted.len() {
                return Some(KernelId(index as u16));
            }
        }
        index += 1;
    }
    None
}

/// The entries of the shared linear table, in the table's order (the position is the table index).
pub fn linear_table() -> &'static [KernelSpec] {
    &KERNELS[..LINEAR_TABLE_LEN]
}

/// The entries of the Flash-Next table, in the table's order.
pub fn qwen4_table() -> &'static [KernelSpec] {
    &KERNELS[LINEAR_TABLE_LEN..LINEAR_TABLE_LEN + QWEN4_TABLE_LEN]
}

/// Every kernel the Rust engine launches.
pub static KERNELS: [KernelSpec; SPECS.len()] = SPECS;

/// `native/src/host/linear_kernels.c:7-24`.
const LINEAR_TABLE_LEN: usize = 11;
/// `native/src/host/qwen4_ops.c:12-50`.
const QWEN4_TABLE_LEN: usize = 36;

const SPECS: [KernelSpec; 148] = [
    spec(
        Family::LinearTable,
        "euhedral_trellis_prepass",
        Module::Trellis,
        0,
        false,
    ),
    spec(Family::LinearTable, "euhedral_trellis_mma", Module::Trellis, 0, false),
    spec(
        Family::LinearTable,
        "euhedral_trellis_mma_rows",
        Module::Trellis,
        0,
        false,
    ),
    spec(
        Family::LinearTable,
        "euhedral_trellis_mma_rows64",
        Module::Trellis,
        0,
        false,
    ),
    spec(
        Family::LinearTable,
        "euhedral_trellis_epilogue",
        Module::Trellis,
        0,
        false,
    ),
    spec(
        Family::LinearTable,
        "euhedral_trellis_decode_bf16",
        Module::Trellis,
        0,
        false,
    ),
    spec(
        Family::LinearTable,
        "euhedral_trellis_epilogue_bf16",
        Module::Trellis,
        0,
        false,
    ),
    spec(
        Family::LinearTable,
        "euhedral_trellis_reference",
        Module::Trellis,
        69632,
        false,
    ),
    spec(
        Family::LinearTable,
        "euhedral_trellis_mma_rows_fp8",
        Module::TrellisFp8,
        0,
        false,
    ),
    spec(
        Family::LinearTable,
        "euhedral_trellis_mma_rows64_fp8",
        Module::TrellisFp8,
        0,
        false,
    ),
    spec(
        Family::LinearTable,
        "euhedral_nvfp4_embedding",
        Module::Embedding,
        0,
        false,
    ),
    spec(Family::Qwen4Table, "euhedral_q4_linear_bf16", Module::Qwen4, 0, false),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_grouped_rms_norm_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_scaled_silu_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(Family::Qwen4Table, "euhedral_q4_hc_mix_bf16", Module::Qwen4, 0, false),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_hc_inject_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_repeat_streams_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_ngram_expand_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(Family::Qwen4Table, "euhedral_q4_ple_gate_bf16", Module::Qwen4, 0, false),
    spec(Family::Qwen4Table, "euhedral_q4_ple_conv_bf16", Module::Qwen4, 0, false),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_conv_history_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(Family::Qwen4Table, "euhedral_q4_gdn_conv_bf16", Module::Qwen4, 0, false),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_gdn_control_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_gdn_gated_norm_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_head_norm_rope_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_qsa_pool_keys_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(Family::Qwen4Table, "euhedral_q4_qsa_tail_bf16", Module::Qwen4, 0, false),
    spec(Family::Qwen4Table, "euhedral_q4_qsa_scores", Module::Qwen4, 0, false),
    spec(Family::Qwen4Table, "euhedral_q4_qsa_select", Module::Qwen4, 0, false),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_qsa_attention",
        Module::Qwen4,
        28224,
        false,
    ),
    spec(Family::Qwen4Table, "euhedral_q4_qsa_merge", Module::Qwen4, 0, false),
    spec(Family::Qwen4Table, "euhedral_q4_qsa_kv_append", Module::Qwen4, 0, false),
    spec(Family::Qwen4Table, "euhedral_q4_router_bf16", Module::Qwen4, 0, false),
    spec(Family::Qwen4Table, "euhedral_q4_swiglu_bf16", Module::Qwen4, 0, false),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_moe_finish_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_embedding_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_expert_gate_up_swiglu_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_expert_down_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_expert_combine_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_swiglu_padded_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(Family::Qwen4Table, "euhedral_q4_nvfp4_pad_k", Module::Qwen4, 0, false),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_linear_tc_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_linear_tc_split_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_linear_tc_rows_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_linear_tc_rows_split_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_linear_split4_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Qwen4Table,
        "euhedral_q4_linear_split8_bf16",
        Module::Qwen4,
        0,
        false,
    ),
    spec(
        Family::Elementwise,
        "euhedral_residual_add_bf16",
        Module::Elementwise,
        0,
        true,
    ),
    spec(
        Family::Elementwise,
        "euhedral_swiglu_bf16",
        Module::Elementwise,
        0,
        true,
    ),
    spec(
        Family::Elementwise,
        "euhedral_residual_rms_norm_bf16",
        Module::Elementwise,
        0,
        true,
    ),
    spec(
        Family::Elementwise,
        "euhedral_residual_rms_norm_row_bf16",
        Module::Elementwise,
        0,
        true,
    ),
    spec(Family::Elementwise, "euhedral_rms_norm_bf16", Module::Norm, 0, true),
    spec(Family::Elementwise, "euhedral_argmax_bf16", Module::Sampling, 0, true),
    spec(
        Family::Elementwise,
        "euhedral_argmax_logprob_bf16",
        Module::Sampling,
        0,
        true,
    ),
    spec(
        Family::Elementwise,
        "euhedral_selected_token",
        Module::Sampling,
        0,
        true,
    ),
    spec(Family::Gdn, "euhedral_gdn_project_control_fp32", Module::Gdn, 0, true),
    spec(
        Family::Gdn,
        "euhedral_gdn_project_control_8x4_fp32",
        Module::Gdn,
        0,
        false,
    ),
    spec(Family::Gdn, "euhedral_gdn_convolution_bf16", Module::Gdn, 0, true),
    spec(Family::Gdn, "euhedral_gdn_recurrence_bf16", Module::Gdn, 0, true),
    spec(Family::Gdn, "euhedral_gdn_recurrence_c8_bf16", Module::Gdn, 0, true),
    spec(Family::Gdn, "euhedral_gdn_gated_rms_norm_bf16", Module::Gdn, 0, true),
    spec(
        Family::Attention,
        "euhedral_attention_qk_norm_rope_bf16",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_qk_norm_rope_rows_bf16",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_kv_append_nvfp4",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_prefill32_nvfp4",
        Module::Attention,
        0,
        false,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_prefill32_nvfp4_exact",
        Module::Attention,
        0,
        false,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_prefill_fa2_nvfp4",
        Module::Attention,
        67584,
        false,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_nvfp4",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_nvfp4_exact",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_gqa_nvfp4",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_merge_nvfp4",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_nvfp4_rows",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_gqa_nvfp4_rows",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_merge_nvfp4_rows",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_gqa_nvfp4_fused2",
        Module::Attention,
        40096,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_gqa_nvfp4_fused3",
        Module::Attention,
        41920,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_gqa_nvfp4_fused4",
        Module::Attention,
        43744,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_gqa_nvfp4_fused5",
        Module::Attention,
        45568,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_gqa_nvfp4_fused6",
        Module::Attention,
        47392,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_kv_append_bf16",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_prefill32_bf16",
        Module::Attention,
        0,
        false,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_prefill32_bf16_exact",
        Module::Attention,
        0,
        false,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_prefill_fa2_bf16",
        Module::Attention,
        67584,
        false,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_bf16",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_bf16_exact",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_gqa_bf16",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_merge_bf16",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_bf16_rows",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_decode_gqa_bf16_rows",
        Module::Attention,
        0,
        true,
    ),
    spec(
        Family::Attention,
        "euhedral_attention_merge_bf16_rows",
        Module::Attention,
        0,
        true,
    ),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows1_w8", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows2_w8", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows3_w8", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows4_w8", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows5_w8", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows6_w8", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows7_w8", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows8_w8", Module::Nvfp4, 0, true),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_groups2_w8",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_groups3_w8",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_groups4_w8",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows1_w4", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows2_w4", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows3_w4", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows4_w4", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows5_w4", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows6_w4", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows7_w4", Module::Nvfp4, 0, true),
    spec(Family::Nvfp4, "euhedral_nvfp4_decode_rows8_w4", Module::Nvfp4, 0, true),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_groups2_w4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_groups3_w4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_groups4_w4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows1_w8_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows2_w8_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows3_w8_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows4_w8_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows5_w8_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows6_w8_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows7_w8_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows8_w8_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_groups2_w8_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_groups3_w8_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_groups4_w8_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows1_w4_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows2_w4_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows3_w4_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows4_w4_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows5_w4_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows6_w4_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows7_w4_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_rows8_w4_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_groups2_w4_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_groups3_w4_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_decode_groups4_w4_sd4",
        Module::Nvfp4,
        0,
        true,
    ),
    spec(Family::Nvfp4, "euhedral_nvfp4_reference", Module::Reference, 0, false),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4_reference_sd4",
        Module::Reference,
        0,
        false,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4n_quantize_rows",
        Module::Nvfp4Native,
        0,
        false,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4n_skinny_finish",
        Module::Nvfp4Native,
        0,
        false,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4n_linear_128x128",
        Module::Nvfp4Native,
        85040,
        false,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4n_skinny_16",
        Module::Nvfp4Native,
        61440,
        false,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4n_skinny_32",
        Module::Nvfp4Native,
        81920,
        false,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4n_skinny_64",
        Module::Nvfp4Native,
        92160,
        false,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4n_linear_128x128_sd4",
        Module::Nvfp4Native,
        85040,
        false,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4n_skinny_16_sd4",
        Module::Nvfp4Native,
        61440,
        false,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4n_skinny_32_sd4",
        Module::Nvfp4Native,
        81920,
        false,
    ),
    spec(
        Family::Nvfp4,
        "euhedral_nvfp4n_skinny_64_sd4",
        Module::Nvfp4Native,
        92160,
        false,
    ),
    spec(Family::Embedding, "euhedral_q3_embedding", Module::Embedding, 0, false),
    spec(
        Family::Embedding,
        "euhedral_q3_p2e2_embedding",
        Module::Embedding,
        0,
        false,
    ),
];

/// One launch argument: a 64-bit word and the byte size (4 or 8) the kernel declares for it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arguments {
    words: [u64; MAX_ARGUMENTS],
    sizes: [u8; MAX_ARGUMENTS],
    count: usize,
}

impl Default for Arguments {
    fn default() -> Self {
        Self::new()
    }
}

impl Arguments {
    pub const fn new() -> Self {
        Self {
            words: [0; MAX_ARGUMENTS],
            sizes: [0; MAX_ARGUMENTS],
            count: 0,
        }
    }

    /// A device address.
    pub fn pointer(self, address: DevicePtr) -> Self {
        self.word(address.get(), 8)
    }

    /// A 32-bit value.
    pub fn u32(self, value: u32) -> Self {
        self.word(u64::from(value), 4)
    }

    /// A 64-bit value.
    pub fn u64(self, value: u64) -> Self {
        self.word(value, 8)
    }

    /// A 32-bit float, carried as its bits.
    pub fn f32(self, value: f32) -> Self {
        self.word(u64::from(value.to_bits()), 4)
    }

    /// A 64-bit float, carried as its bits.
    pub fn f64(self, value: f64) -> Self {
        self.word(value.to_bits(), 8)
    }

    /// A word of `size` bytes (4 or 8).
    pub fn word(mut self, word: u64, size: u8) -> Self {
        // An argument past the limit is counted and not stored; a launch with too many is refused by its consumer.
        if self.count < MAX_ARGUMENTS {
            self.words[self.count] = word;
            self.sizes[self.count] = size;
        }
        self.count += 1;
        self
    }

    /// How many arguments were added (more than [`MAX_ARGUMENTS`] means the launch is invalid).
    pub const fn count(&self) -> usize {
        self.count
    }

    /// The stored words, in order.
    pub fn words(&self) -> &[u64] {
        &self.words[..self.stored()]
    }

    /// The byte size of each stored word.
    pub fn sizes(&self) -> &[u8] {
        &self.sizes[..self.stored()]
    }

    /// Argument `index` as a 32-bit float; `None` when it is not a 4-byte argument.
    pub fn f32_at(&self, index: usize) -> Option<f32> {
        (self.sizes().get(index) == Some(&4)).then(|| f32::from_bits(self.words[index] as u32))
    }

    /// Argument `index` as a 64-bit float; `None` when it is not an 8-byte argument.
    pub fn f64_at(&self, index: usize) -> Option<f64> {
        (self.sizes().get(index) == Some(&8)).then(|| f64::from_bits(self.words[index]))
    }

    const fn stored(&self) -> usize {
        if self.count < MAX_ARGUMENTS {
            self.count
        } else {
            MAX_ARGUMENTS
        }
    }
}

/// One planned kernel launch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Launch {
    pub kernel: KernelId,
    pub grid: [u32; 3],
    pub block: [u32; 3],
    /// Dynamic shared bytes.
    pub shared_bytes: u32,
    pub args: Arguments,
    /// Whether the launch may overlap its predecessor (programmatic dependent launch): true only for a kernel the
    /// native library registers for it, and when the plan's caller allows it.
    pub programmatic: bool,
}

impl Launch {
    /// A launch of `kernel`; programmatic dependent launch is eligible exactly when the native library registers the
    /// kernel for it.
    pub const fn new(kernel: KernelId, grid: [u32; 3], block: [u32; 3], shared_bytes: u32, args: Arguments) -> Self {
        Self {
            kernel,
            grid,
            block,
            shared_bytes,
            args,
            programmatic: kernel.spec().pdl_registered,
        }
    }

    pub const fn spec(&self) -> &'static KernelSpec {
        self.kernel.spec()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use super::*;

    #[test]
    fn every_kernel_is_listed_once() {
        let names: BTreeSet<&str> = KERNELS.iter().map(|kernel| kernel.name).collect();
        assert_eq!(names.len(), KERNELS.len());
        for (index, kernel) in KERNELS.iter().enumerate() {
            assert_eq!(lookup(kernel.name).map(KernelId::index), Some(index), "{}", kernel.name);
        }
        assert_eq!(lookup("euhedral_not_a_kernel"), None);
    }

    #[test]
    fn catalog_pdl_set_equals_the_native_registrations() {
        // The registrations of the native library, for the kernels the Rust engine launches, by name.
        let mut registered: BTreeSet<String> = BTreeSet::new();
        // qwen_layer_ops.c:170-185 (`initialize`), the lines in the comments.
        for name in [
            "euhedral_argmax_bf16",                      // :172
            "euhedral_argmax_logprob_bf16",              // :173
            "euhedral_selected_token",                   // :174
            "euhedral_gdn_project_control_fp32",         // :175
            "euhedral_residual_rms_norm_bf16",           // :176
            "euhedral_residual_rms_norm_row_bf16",       // :177
            "euhedral_gdn_convolution_bf16",             // :178
            "euhedral_gdn_recurrence_bf16",              // :179
            "euhedral_gdn_recurrence_c8_bf16",           // :180
            "euhedral_gdn_gated_rms_norm_bf16",          // :181
            "euhedral_residual_add_bf16",                // :182
            "euhedral_swiglu_bf16",                      // :183
            "euhedral_attention_qk_norm_rope_bf16",      // :184
            "euhedral_attention_qk_norm_rope_rows_bf16", // :185
        ] {
            registered.insert(name.to_string());
        }
        // qwen_layer_ops.c:186-198, once per cache format: append (:188), decode, decode_exact, decode_gqa, merge,
        // decode_rows, decode_gqa_rows, merge_rows (:189-195), and the fused kernels of 2 to 6 tiles (:196-197), which
        // only nvfp4 has (bf16's are NULL and skipped).
        for format in ["nvfp4", "bf16"] {
            for pattern in [
                "kv_append_F",
                "decode_F",
                "decode_F_exact",
                "decode_gqa_F",
                "merge_F",
                "decode_F_rows",
                "decode_gqa_F_rows",
                "merge_F_rows",
            ] {
                registered.insert(format!("euhedral_attention_{}", pattern.replace('F', format)));
            }
        }
        for tiles in 2..=6 {
            registered.insert(format!("euhedral_attention_decode_gqa_nvfp4_fused{tiles}"));
        }
        // rms_norm_bf16.c:22.
        registered.insert("euhedral_rms_norm_bf16".to_string());
        // nvfp4_linear.c:44 (rows 1 to 8) and :51 (groups 2 to 4), every layout and tile.
        for layout in ["", "_sd4"] {
            for tile in ["w8", "w4"] {
                for rows in 1..=8 {
                    registered.insert(format!("euhedral_nvfp4_decode_rows{rows}_{tile}{layout}"));
                }
                for groups in 2..=4 {
                    registered.insert(format!("euhedral_nvfp4_decode_groups{groups}_{tile}{layout}"));
                }
            }
        }
        // Registered natively and not launched by the Rust engine, so not in the catalog: qwen_layer_ops.c:170
        // (`euhedral_linear_bf16_to_float`) and :171 (`euhedral_gdn_control_fp32`).
        for name in ["euhedral_linear_bf16_to_float", "euhedral_gdn_control_fp32"] {
            assert_eq!(lookup(name), None);
        }

        let flagged: BTreeSet<String> = KERNELS
            .iter()
            .filter(|kernel| kernel.pdl_registered)
            .map(|kernel| kernel.name.to_string())
            .collect();
        assert_eq!(flagged, registered);
    }

    #[test]
    fn the_linear_table_entries_are_the_native_table_in_order() {
        // native/src/host/linear_kernels.c:7-24, which `LinearKernel::ALL` in the engine lists in the same order
        // (`crates/engine/src/formats/shapes.rs:204`; the engine's `catalog` test compares them directly).
        let names = [
            "euhedral_trellis_prepass",
            "euhedral_trellis_mma",
            "euhedral_trellis_mma_rows",
            "euhedral_trellis_mma_rows64",
            "euhedral_trellis_epilogue",
            "euhedral_trellis_decode_bf16",
            "euhedral_trellis_epilogue_bf16",
            "euhedral_trellis_reference",
            "euhedral_trellis_mma_rows_fp8",
            "euhedral_trellis_mma_rows64_fp8",
            "euhedral_nvfp4_embedding",
        ];
        let table: Vec<&str> = linear_table().iter().map(|kernel| kernel.name).collect();
        assert_eq!(table, names);
        assert!(linear_table().iter().all(|kernel| kernel.family == Family::LinearTable));
        // :18 the reference's opt-in, :20-21 the FP8 module.
        assert_eq!(find("euhedral_trellis_reference").spec().shared_opt_in_bytes, 4 * 17408);
        assert!(find("euhedral_trellis_mma_rows_fp8").spec().architecture_specific());
        assert!(find("euhedral_trellis_mma_rows64_fp8").spec().architecture_specific());
        assert!(!find("euhedral_trellis_mma").spec().architecture_specific());
    }

    #[test]
    fn the_qwen4_table_entries_are_the_native_table_in_order() {
        // native/src/host/qwen4_ops.c:12-50, in order.
        let names = [
            "linear_bf16",
            "grouped_rms_norm_bf16",
            "scaled_silu_bf16",
            "hc_mix_bf16",
            "hc_inject_bf16",
            "repeat_streams_bf16",
            "ngram_expand_bf16",
            "ple_gate_bf16",
            "ple_conv_bf16",
            "conv_history_bf16",
            "gdn_conv_bf16",
            "gdn_control_bf16",
            "gdn_gated_norm_bf16",
            "head_norm_rope_bf16",
            "qsa_pool_keys_bf16",
            "qsa_tail_bf16",
            "qsa_scores",
            "qsa_select",
            "qsa_attention",
            "qsa_merge",
            "qsa_kv_append",
            "router_bf16",
            "swiglu_bf16",
            "moe_finish_bf16",
            "embedding_bf16",
            "expert_gate_up_swiglu_bf16",
            "expert_down_bf16",
            "expert_combine_bf16",
            "swiglu_padded_bf16",
            "nvfp4_pad_k",
            "linear_tc_bf16",
            "linear_tc_split_bf16",
            "linear_tc_rows_bf16",
            "linear_tc_rows_split_bf16",
            "linear_split4_bf16",
            "linear_split8_bf16",
        ];
        let table: Vec<String> = qwen4_table().iter().map(|kernel| kernel.name.to_string()).collect();
        let expected: Vec<String> = names.iter().map(|name| format!("euhedral_q4_{name}")).collect();
        assert_eq!(table, expected);
        for kernel in qwen4_table() {
            assert_eq!(kernel.family, Family::Qwen4Table);
            assert_eq!(kernel.module, Module::Qwen4);
            assert!(!kernel.pdl_registered);
        }
        // :32, one unit's three warps.
        assert_eq!(find("euhedral_q4_qsa_attention").spec().shared_opt_in_bytes, 28224);
    }

    #[test]
    fn only_the_native_modules_are_architecture_specific() {
        let specific: BTreeSet<&str> = KERNELS
            .iter()
            .filter(|kernel| kernel.architecture_specific())
            .map(|kernel| kernel.module.root())
            .collect();
        assert_eq!(
            specific,
            BTreeSet::from(["trellis_fp8/kernels.cu", "nvfp4_native/kernels.cu"])
        );
    }

    #[test]
    fn the_opt_in_shared_bytes_are_the_native_ones() {
        // qwen_layer_ops.c:50 and :101.
        assert_eq!(
            find("euhedral_attention_prefill_fa2_nvfp4").spec().shared_opt_in_bytes,
            67584
        );
        assert_eq!(
            find("euhedral_attention_prefill_fa2_bf16").spec().shared_opt_in_bytes,
            67584
        );
        // qwen_layer_ops.c:61-63 (`fused_shared_bytes`) for 2 and 6 tiles, set at :110.
        assert_eq!(
            find("euhedral_attention_decode_gqa_nvfp4_fused2")
                .spec()
                .shared_opt_in_bytes,
            40096
        );
        assert_eq!(
            find("euhedral_attention_decode_gqa_nvfp4_fused6")
                .spec()
                .shared_opt_in_bytes,
            47392
        );
        // nvfp4_linear.c:152 (`NATIVE_SHARED_BYTES`) and `skinny_shared_bytes` (:159-162, set at :208-209) for 16, 32 and 64 rows with two terms.
        assert_eq!(find("euhedral_nvfp4n_linear_128x128").spec().shared_opt_in_bytes, 85040);
        assert_eq!(find("euhedral_nvfp4n_skinny_16").spec().shared_opt_in_bytes, 61440);
        assert_eq!(find("euhedral_nvfp4n_skinny_32_sd4").spec().shared_opt_in_bytes, 81920);
        assert_eq!(find("euhedral_nvfp4n_skinny_64").spec().shared_opt_in_bytes, 92160);
        // Everything else sets none.
        let with: usize = KERNELS.iter().filter(|kernel| kernel.shared_opt_in_bytes != 0).count();
        assert_eq!(with, 1 + 1 + 2 + 5 + 2 + 6);
    }

    #[test]
    fn floats_round_trip_as_their_exact_bits() {
        let single = f32::from_bits(0x7fc0_1234); // a NaN with a payload
        let double = f64::from_bits(0xfff0_0000_dead_beef); // a NaN with a payload and the sign set
        let args = Arguments::new()
            .f32(single)
            .f64(double)
            .f32(-0.0)
            .f64(f64::MIN_POSITIVE / 2.0);
        assert_eq!(args.words()[0], 0x7fc0_1234);
        assert_eq!(args.words()[1], 0xfff0_0000_dead_beef);
        assert_eq!(args.sizes(), [4, 8, 4, 8]);
        assert_eq!(args.f32_at(0).map(f32::to_bits), Some(single.to_bits()));
        assert_eq!(args.f64_at(1).map(f64::to_bits), Some(double.to_bits()));
        assert_eq!(args.f32_at(2).map(f32::to_bits), Some((-0.0f32).to_bits()));
        assert_eq!(
            args.f64_at(3).map(f64::to_bits),
            Some((f64::MIN_POSITIVE / 2.0).to_bits())
        );
        // A float is not read as the other width.
        assert_eq!(args.f64_at(0), None);
        assert_eq!(args.f32_at(1), None);
    }

    #[test]
    fn a_launch_is_eligible_for_overlap_exactly_when_the_kernel_is_registered() {
        let add = Launch::new(
            find("euhedral_residual_add_bf16"),
            [1, 1, 1],
            [256, 1, 1],
            0,
            Arguments::new(),
        );
        assert!(add.programmatic);
        let prefill = Launch::new(
            find("euhedral_attention_prefill32_bf16"),
            [1, 1, 1],
            [128, 1, 1],
            0,
            Arguments::new(),
        );
        assert!(!prefill.programmatic);
    }

    #[test]
    fn arguments_past_the_limit_are_counted_and_not_stored() {
        let mut args = Arguments::new();
        for value in 0..(MAX_ARGUMENTS as u32 + 2) {
            args = args.u32(value);
        }
        assert_eq!(args.count(), MAX_ARGUMENTS + 2);
        assert_eq!(args.words().len(), MAX_ARGUMENTS);
    }
}
