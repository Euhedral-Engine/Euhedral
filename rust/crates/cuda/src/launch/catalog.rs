//! The catalog of every kernel the Rust engine launches, and the planned launch the family files produce.
//!
//! One [`KernelSpec`] per kernel, in a fixed order: the entries of the shared linear table, then those of the
//! Flash-Next table (each in the table's order, which is the native ABI), then the other families. A kernel is named by
//! its [`KernelId`], its index in [`KERNELS`]. The fields are what the native library does for the kernel today: the
//! name it registers, the source it compiles from, the architecture-specific compile target of that source, the
//! dynamic shared memory it sets as the function's maximum, and whether it registers the kernel for programmatic
//! dependent launch (`pdl_register`).
//!
//! The families are the ones the Rust engine calls (`ops.rs`), the Flash-Next MoE kernels among them (the router, the
//! shared expert's SwiGLU, and the expert gate-up, down and combine kernels, which the layer's MoE stages launch). The
//! native library has more kernels (the Q3 gate-up SwiGLU, `euhedral_linear_bf16_to_float`,
//! `euhedral_gdn_control_fp32`, the native gate-up SwiGLU); they stay out until the engine calls them.

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
    /// The Q3 decode kernels and the P2E2 decode and expansion.
    Q3,
    /// The Q4 and Q5 decode kernels.
    Q45,
    /// Block-scaled FP8 tensor cores for the Q3, Q4 and Q5 prefill; compiled for the architecture-specific target.
    Q3Mx,
    /// The DFlash2 drafter's operators.
    Dflash,
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
            Module::Q3 => "q3/kernels.cu",
            Module::Q45 => "q45/kernels.cu",
            Module::Q3Mx => "q3_mx/kernels.cu",
            Module::Dflash => "dflash/kernels.cu",
            Module::Reference => "reference/kernels.cu",
        }
    }

    /// Whether the module needs the architecture-specific target (`sm_XYa`), which exists on compute capability 12.x
    /// only; on another device its kernels are unavailable.
    pub const fn architecture_specific(self) -> bool {
        matches!(self, Module::TrellisFp8 | Module::Nvfp4Native | Module::Q3Mx)
    }
}

/// The group a kernel belongs to; the family files plan one group each (the two tables share one file).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Family {
    /// The shared linear table.
    LinearTable,
    /// The Flash-Next table.
    Qwen4Table,
    /// Residual, SwiGLU, RMS norm, argmax and selected token.
    Elementwise,
    Gdn,
    Attention,
    /// The NVFP4 linears: decode, reference and native.
    Nvfp4,
    /// The embedding lookups outside the linear table.
    Embedding,
    /// The Q3, Q4 and Q5 grouped linears: decode, reference, FP8 prefill and the P2E2 expansion.
    Q3,
    /// The DFlash2 drafter's operators.
    Dflash,
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
    /// Whether the kernel takes tensor maps (128-byte `__grid_constant__` parameters, carried by [`Arguments`]): the
    /// native NVFP4 linear kernels.
    pub fn tensor_map_arguments(&self) -> bool {
        self.module == Module::Nvfp4Native
            && (self.name.starts_with("euhedral_nvfp4n_linear_")
                || self.name.starts_with("euhedral_nvfp4n_gate_up_swiglu_"))
    }

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

/// The kernel called `name`, failing the build when evaluated in a constant and there is none. Not for direct use:
/// name a kernel by literal with [`kernel!`], which forces the evaluation at build time.
#[doc(hidden)]
pub const fn find_const(name: &str) -> KernelId {
    match lookup(name) {
        Some(id) => id,
        None => panic!("no kernel of this name in the catalog"),
    }
}

/// The [`KernelId`] of the kernel called by the literal, checked at build time: a name the catalog does not list is a
/// compile error. This is the only way to name a kernel by literal; [`lookup`] is for names that are not literals.
///
/// ```
/// use euhedral_inference_cuda::kernel;
/// let id = kernel!("euhedral_residual_add_bf16");
/// assert_eq!(id.name(), "euhedral_residual_add_bf16");
/// ```
///
/// ```compile_fail
/// use euhedral_inference_cuda::kernel;
/// let id = kernel!("euhedral_residual_ad_bf16");
/// ```
#[macro_export]
macro_rules! kernel {
    ($name:literal) => {
        const { $crate::launch::catalog::find_const($name) }
    };
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

/// Every catalog kernel, in catalog order.
pub fn ids() -> impl Iterator<Item = KernelId> {
    (0..SPECS.len()).map(|index| KernelId(index as u16))
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

/// The number of entries of the shared linear table.
const LINEAR_TABLE_LEN: usize = 11;
/// The number of entries of the Flash-Next table.
const QWEN4_TABLE_LEN: usize = 36;

/// The dynamic shared memory of `euhedral_dflash_attention_tc_bf16`: the expression of the kernel's
/// shared size, whose terms are the kernel's `attentionShared` layout (`dflash/kernels.cu`): 32
/// score rows of 276 floats, the two BF16 probability planes of 32 rows by 280, the 16 rows of 136 BF16 values, and
/// 64 floats of row maxima and sums. A name, not the expression, in the table: `native/tests/pdl_wait_check.py`
/// reads each entry's size as one word.
const DFLASH_ATTENTION_SHARED_BYTES: u32 = 32 * 276 * 4 + 2 * 32 * 280 * 2 + 16 * 136 * 2 + 64 * 4;

const SPECS: [KernelSpec; 203] = [
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
    spec(
        Family::Embedding,
        "euhedral_bf16_embedding",
        Module::Embedding,
        0,
        false,
    ),
    spec(Family::Q3, "euhedral_q3_decode_contiguous", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_decode_contiguous_rows2", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_decode_contiguous_rows3", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_decode_contiguous_rows4", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_decode_contiguous_rows5", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_decode_contiguous_rows6", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_decode_contiguous_rows7", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_decode_contiguous_rows8", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_decode_contiguous_tokens", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_p2e2_decode", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_p2e2_decode_rows2", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_p2e2_decode_rows3", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_p2e2_decode_rows4", Module::Q3, 0, true),
    spec(Family::Q3, "euhedral_q3_p2e2_expand", Module::Q3, 0, false),
    spec(Family::Q3, "euhedral_q4_decode_contiguous", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q4_decode_contiguous_rows2", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q4_decode_contiguous_rows3", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q4_decode_contiguous_rows4", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q4_decode_contiguous_rows5", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q4_decode_contiguous_rows6", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q4_decode_contiguous_rows7", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q4_decode_contiguous_rows8", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q4_decode_contiguous_tokens", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q5_decode_contiguous", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q5_decode_contiguous_rows2", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q5_decode_contiguous_rows3", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q5_decode_contiguous_rows4", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q5_decode_contiguous_rows5", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q5_decode_contiguous_rows6", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q5_decode_contiguous_rows7", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q5_decode_contiguous_rows8", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q5_decode_contiguous_tokens", Module::Q45, 0, true),
    spec(Family::Q3, "euhedral_q3mx_quantize", Module::Q3Mx, 0, false),
    spec(Family::Q3, "euhedral_q3mx_linear_128x128", Module::Q3Mx, 75264, false),
    spec(Family::Q3, "euhedral_q4mx_linear_128x128", Module::Q3Mx, 75264, false),
    spec(Family::Q3, "euhedral_q5mx_linear_128x128", Module::Q3Mx, 75264, false),
    spec(
        Family::Q3,
        "euhedral_q3mx_gate_up_swiglu_128x64",
        Module::Q3Mx,
        75264,
        false,
    ),
    spec(Family::Q3, "euhedral_q3mx_reduce", Module::Q3Mx, 0, false),
    spec(Family::Q3, "euhedral_q3_reference", Module::Reference, 0, false),
    spec(Family::Q3, "euhedral_q45_reference", Module::Reference, 0, false),
    spec(Family::Dflash, "euhedral_dflash_linear_bf16", Module::Dflash, 0, true),
    spec(
        Family::Dflash,
        "euhedral_dflash_linear_rows_bf16",
        Module::Dflash,
        0,
        true,
    ),
    spec(
        Family::Dflash,
        "euhedral_dflash_linear_split_bf16",
        Module::Dflash,
        0,
        true,
    ),
    spec(
        Family::Dflash,
        "euhedral_dflash_linear_rows_split_bf16",
        Module::Dflash,
        0,
        true,
    ),
    spec(Family::Dflash, "euhedral_dflash_rms_norm_bf16", Module::Dflash, 0, true),
    spec(Family::Dflash, "euhedral_dflash_conv_bf16", Module::Dflash, 0, true),
    spec(
        Family::Dflash,
        "euhedral_dflash_context_kv_bf16",
        Module::Dflash,
        0,
        true,
    ),
    spec(Family::Dflash, "euhedral_dflash_block_qk_bf16", Module::Dflash, 0, true),
    spec(
        Family::Dflash,
        "euhedral_dflash_attention_tc_bf16",
        Module::Dflash,
        DFLASH_ATTENTION_SHARED_BYTES,
        true,
    ),
    spec(
        Family::Dflash,
        "euhedral_dflash_attention_merge_bf16",
        Module::Dflash,
        0,
        true,
    ),
    spec(Family::Dflash, "euhedral_dflash_swiglu_bf16", Module::Dflash, 0, true),
    spec(
        Family::Dflash,
        "euhedral_dflash_topk_partial_bf16",
        Module::Dflash,
        0,
        true,
    ),
    spec(
        Family::Dflash,
        "euhedral_dflash_topk_merge_bf16",
        Module::Dflash,
        0,
        true,
    ),
    spec(Family::Dflash, "euhedral_dflash_select_bf16", Module::Dflash, 0, true),
];

/// The parameters of one tensor map, as `cuTensorMapEncodeTiled` takes them. Every map is over bytes (`UINT8`) with unit element strides, no interleave, 128-byte L2
/// promotion and no out-of-bounds fill; those do not vary and are not fields.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct TensorMapPlan {
    /// The global address (`CUdeviceptr`).
    pub address: u64,
    /// The rank (2 or 3): how many of `dims` and `box_dims`, and one fewer of `strides`, are used.
    pub rank: u32,
    /// Extents in elements (bytes), innermost first.
    pub dims: [u64; 3],
    /// Byte strides of dimensions 1 and up.
    pub strides: [u64; 2],
    /// Box extents in elements.
    pub box_dims: [u32; 3],
    pub swizzle: TensorMapSwizzle,
}

/// The swizzle of a tensor map (`CUtensorMapSwizzle`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorMapSwizzle {
    None,
    B64,
}

impl TensorMapPlan {
    /// The extents in use.
    pub fn dims(&self) -> &[u64] {
        &self.dims[..self.rank as usize]
    }

    /// The strides in use.
    pub fn strides(&self) -> &[u64] {
        &self.strides[..self.rank as usize - 1]
    }

    /// The box extents in use.
    pub fn box_dims(&self) -> &[u32] {
        &self.box_dims[..self.rank as usize]
    }
}

/// The most tensor maps one launch takes (the native tile kernels take three).
pub const MAX_TENSOR_MAPS: usize = 3;

/// The byte size of a tensor-map parameter (`CUtensorMap`).
pub const TENSOR_MAP_BYTES: u8 = 128;

/// The arguments of one launch: tensor maps by value, then words, each word with the byte size (4 or 8) the kernel
/// declares for it. The maps are the kernel's leading parameters, in the order they were added.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Arguments {
    maps: [Option<TensorMapPlan>; MAX_TENSOR_MAPS],
    map_count: usize,
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
            maps: [None; MAX_TENSOR_MAPS],
            map_count: 0,
            words: [0; MAX_ARGUMENTS],
            sizes: [0; MAX_ARGUMENTS],
            count: 0,
        }
    }

    /// A tensor map, passed by value in front of the words. More than [`MAX_TENSOR_MAPS`] are counted and not stored, as
    /// words past the limit are.
    pub fn tensor_map(mut self, map: TensorMapPlan) -> Self {
        if self.map_count < MAX_TENSOR_MAPS {
            self.maps[self.map_count] = Some(map);
        }
        self.map_count += 1;
        self
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

    /// How many tensor maps were added (more than [`MAX_TENSOR_MAPS`] means the launch is invalid).
    pub const fn map_count(&self) -> usize {
        self.map_count
    }

    /// The stored tensor maps, in order.
    pub fn maps(&self) -> impl Iterator<Item = &TensorMapPlan> {
        self.maps.iter().flatten()
    }

    /// How many words were added (more than [`MAX_ARGUMENTS`] means the launch is invalid).
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
    /// Whether the launch is a programmatic dependent launch (it may overlap its predecessor's tail): true for every
    /// kernel the native library registers for it. The queue launches every flagged launch so; the native library does
    /// so only while the kernel is among its first 64 registrations and the thread selected it. Output is the same,
    /// because each such kernel begins by waiting on its predecessor; only the overlap timing can differ.
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
    fn the_kernels_that_take_tensor_maps_are_the_native_nvfp4_linears() {
        let names: BTreeSet<&str> = KERNELS
            .iter()
            .filter(|kernel| kernel.tensor_map_arguments())
            .map(|kernel| kernel.name)
            .collect();
        let expected = BTreeSet::from(["euhedral_nvfp4n_linear_128x128", "euhedral_nvfp4n_linear_128x128_sd4"]);
        assert_eq!(names, expected);
    }

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
        for name in [
            "euhedral_argmax_bf16",
            "euhedral_argmax_logprob_bf16",
            "euhedral_selected_token",
            "euhedral_gdn_project_control_fp32",
            "euhedral_residual_rms_norm_bf16",
            "euhedral_residual_rms_norm_row_bf16",
            "euhedral_gdn_convolution_bf16",
            "euhedral_gdn_recurrence_bf16",
            "euhedral_gdn_recurrence_c8_bf16",
            "euhedral_gdn_gated_rms_norm_bf16",
            "euhedral_residual_add_bf16",
            "euhedral_swiglu_bf16",
            "euhedral_attention_qk_norm_rope_bf16",
            "euhedral_attention_qk_norm_rope_rows_bf16",
        ] {
            registered.insert(name.to_string());
        }
        // Once per cache format: append, decode, decode_exact, decode_gqa, merge, decode_rows, decode_gqa_rows,
        // merge_rows, and the fused kernels of 2 to 6 tiles, which only nvfp4 has.
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
        registered.insert("euhedral_rms_norm_bf16".to_string());
        // Rows 1 to 8 and groups 2 to 4, every layout and tile.
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
        // The one-row decode kernel, its 2 to 8 row twins and the token-row twin, and the
        // P2E2 decode kernels of 1 to 4 rows (the expansion is not registered).
        registered.insert("euhedral_q3_decode_contiguous".to_string());
        registered.insert("euhedral_q3_decode_contiguous_tokens".to_string());
        for rows in 2..=8 {
            registered.insert(format!("euhedral_q3_decode_contiguous_rows{rows}"));
        }
        registered.insert("euhedral_q3_p2e2_decode".to_string());
        for rows in 2..=4 {
            registered.insert(format!("euhedral_q3_p2e2_decode_rows{rows}"));
        }
        // Q4 and Q5.
        for bits in [4, 5] {
            registered.insert(format!("euhedral_q{bits}_decode_contiguous"));
            registered.insert(format!("euhedral_q{bits}_decode_contiguous_tokens"));
            for rows in 2..=8 {
                registered.insert(format!("euhedral_q{bits}_decode_contiguous_rows{rows}"));
            }
        }
        // Every drafter kernel.
        for name in [
            "linear",
            "linear_rows",
            "linear_split",
            "linear_rows_split",
            "rms_norm",
            "conv",
            "context_kv",
            "block_qk",
            "attention_tc",
            "attention_merge",
            "swiglu",
            "topk_partial",
            "topk_merge",
            "select",
        ] {
            registered.insert(format!("euhedral_dflash_{name}_bf16"));
        }
        // Registered natively and not launched by the Rust engine, so not in the catalog:
        // `euhedral_linear_bf16_to_float` and `euhedral_gdn_control_fp32`.
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
        // The native table, which `LinearKernel::ALL` in the engine lists in the same order (the engine's
        // `catalog` test compares them directly).
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
        // The reference's opt-in, and the FP8 module.
        assert_eq!(
            kernel!("euhedral_trellis_reference").spec().shared_opt_in_bytes,
            4 * 17408
        );
        assert!(kernel!("euhedral_trellis_mma_rows_fp8").spec().architecture_specific());
        assert!(
            kernel!("euhedral_trellis_mma_rows64_fp8")
                .spec()
                .architecture_specific()
        );
        assert!(!kernel!("euhedral_trellis_mma").spec().architecture_specific());
    }

    #[test]
    fn the_qwen4_table_entries_are_the_native_table_in_order() {
        // The Flash-Next table, in order.
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
        // One unit's three warps.
        assert_eq!(kernel!("euhedral_q4_qsa_attention").spec().shared_opt_in_bytes, 28224);
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
            BTreeSet::from(["trellis_fp8/kernels.cu", "nvfp4_native/kernels.cu", "q3_mx/kernels.cu"])
        );
    }

    #[test]
    fn the_opt_in_shared_bytes_are_the_native_ones() {
        assert_eq!(
            kernel!("euhedral_attention_prefill_fa2_nvfp4")
                .spec()
                .shared_opt_in_bytes,
            67584
        );
        assert_eq!(
            kernel!("euhedral_attention_prefill_fa2_bf16")
                .spec()
                .shared_opt_in_bytes,
            67584
        );
        // The fused kernels for 2 and 6 tiles.
        assert_eq!(
            kernel!("euhedral_attention_decode_gqa_nvfp4_fused2")
                .spec()
                .shared_opt_in_bytes,
            40096
        );
        assert_eq!(
            kernel!("euhedral_attention_decode_gqa_nvfp4_fused6")
                .spec()
                .shared_opt_in_bytes,
            47392
        );
        // The native tile kernel, and the skinny kernels for 16, 32 and 64 rows with two terms.
        assert_eq!(
            kernel!("euhedral_nvfp4n_linear_128x128").spec().shared_opt_in_bytes,
            85040
        );
        assert_eq!(kernel!("euhedral_nvfp4n_skinny_16").spec().shared_opt_in_bytes, 61440);
        // The drafter's attention kernel.
        assert_eq!(
            kernel!("euhedral_dflash_attention_tc_bf16").spec().shared_opt_in_bytes,
            75776
        );
        assert_eq!(
            kernel!("euhedral_nvfp4n_skinny_32_sd4").spec().shared_opt_in_bytes,
            81920
        );
        assert_eq!(kernel!("euhedral_nvfp4n_skinny_64").spec().shared_opt_in_bytes, 92160);
        // Everything else sets none.
        let with: usize = KERNELS.iter().filter(|kernel| kernel.shared_opt_in_bytes != 0).count();
        // The three FP8 prefill linears of Q3, Q4 and Q5.
        assert_eq!(
            kernel!("euhedral_q3mx_linear_128x128").spec().shared_opt_in_bytes,
            75264
        );
        assert_eq!(
            kernel!("euhedral_q4mx_linear_128x128").spec().shared_opt_in_bytes,
            75264
        );
        assert_eq!(
            kernel!("euhedral_q5mx_linear_128x128").spec().shared_opt_in_bytes,
            75264
        );
        assert_eq!(
            kernel!("euhedral_q3mx_gate_up_swiglu_128x64")
                .spec()
                .shared_opt_in_bytes,
            75264
        );
        assert_eq!(with, 1 + 1 + 2 + 5 + 2 + 6 + 3 + 1 + 1);
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
            kernel!("euhedral_residual_add_bf16"),
            [1, 1, 1],
            [256, 1, 1],
            0,
            Arguments::new(),
        );
        assert!(add.programmatic);
        let prefill = Launch::new(
            kernel!("euhedral_attention_prefill32_bf16"),
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
