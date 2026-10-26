//! The model's dimensions, from the artifact's metadata, and the profile its tensors give it.

use euhedral_inference_edrl::{LayerType, ModelMetadata, TensorTable, WeightFormat, WeightLayout};
use euhedral_inference_settings::KvFormat;

use crate::error::PlanError;

/// What a decoder layer mixes tokens with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerKind {
    FullAttention,
    GatedDeltaNet,
}

/// The dimensions the engine reads from the artifact (`Qwen38Config`).
#[derive(Clone, Debug, PartialEq)]
pub struct ModelConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub attention_head_dim: usize,
    pub intermediate_size: usize,
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,
    pub rms_norm_epsilon: f64,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub max_position_embeddings: usize,
    /// One entry per decoder layer.
    pub layer_kinds: Vec<LayerKind>,
    pub num_experts: usize,
    pub tie_word_embeddings: bool,
    pub attention_output_gate: bool,
    pub mtp_layer_count: usize,
    /// The MTP draft depth the artifact stores.
    pub draft_depth: Option<u8>,
}

impl From<&ModelMetadata> for ModelConfig {
    fn from(metadata: &ModelMetadata) -> Self {
        let size = |value: u32| value as usize;
        Self {
            vocab_size: size(metadata.vocab_size),
            hidden_size: size(metadata.hidden_size),
            num_hidden_layers: size(metadata.num_hidden_layers),
            num_attention_heads: size(metadata.num_attention_heads),
            num_key_value_heads: size(metadata.num_key_value_heads),
            attention_head_dim: size(metadata.attention_head_dim),
            intermediate_size: size(metadata.intermediate_size),
            linear_num_key_heads: size(metadata.linear_num_key_heads),
            linear_num_value_heads: size(metadata.linear_num_value_heads),
            linear_key_head_dim: size(metadata.linear_key_head_dim),
            linear_value_head_dim: size(metadata.linear_value_head_dim),
            linear_conv_kernel_dim: size(metadata.linear_conv_kernel_dim),
            rms_norm_epsilon: metadata.rms_norm_epsilon,
            rope_theta: metadata.rope_theta,
            partial_rotary_factor: metadata.partial_rotary_factor,
            max_position_embeddings: size(metadata.max_position_embeddings),
            layer_kinds: metadata
                .layer_types
                .iter()
                .map(|kind| match kind {
                    LayerType::FullAttention => LayerKind::FullAttention,
                    LayerType::GatedDeltaNet => LayerKind::GatedDeltaNet,
                })
                .collect(),
            num_experts: size(metadata.num_experts),
            tie_word_embeddings: metadata.tie_word_embeddings,
            attention_output_gate: metadata.attention_output_gate,
            mtp_layer_count: size(metadata.mtp_layer_count),
            draft_depth: metadata.draft_depth,
        }
    }
}

impl ModelConfig {
    /// Refuses a geometry the engine does not run: the dense Qwen text model with 64 layers and at most one MTP
    /// layer (`CompactWeightLoader.validateConfig`), and the attention and GDN geometry `ExecutionPlan` checks.
    pub fn check_supported(&self) -> Result<(), PlanError> {
        let unsupported = |why: String| Err(PlanError::Unsupported(why));
        if self.layer_kinds.len() != 64 || self.num_hidden_layers != 64 || self.mtp_layer_count > 1 {
            return unsupported("the engine runs a 64-layer model with at most one MTP layer".to_string());
        }
        if self.num_experts != 0 {
            return unsupported("the engine runs the dense Qwen topology, not a mixture of experts".to_string());
        }
        let rotary = (self.attention_head_dim as f64 * self.partial_rotary_factor).round();
        let bad = self.hidden_size == 0
            || !self.hidden_size.is_multiple_of(128)
            || self.intermediate_size == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || !self.num_attention_heads.is_multiple_of(self.num_key_value_heads)
            || self.attention_head_dim != 256
            || !self.attention_output_gate
            || !self.rope_theta.is_finite()
            || self.rope_theta <= 0.0
            || !self.partial_rotary_factor.is_finite()
            || self.partial_rotary_factor <= 0.0
            || self.partial_rotary_factor > 1.0
            || rotary <= 0.0
            || rotary % 2.0 != 0.0
            || self.linear_num_key_heads == 0
            || self.linear_num_value_heads == 0
            || !self.linear_num_value_heads.is_multiple_of(self.linear_num_key_heads)
            || self.linear_key_head_dim != 128
            || self.linear_value_head_dim != 128
            || self.linear_conv_kernel_dim < 2
            || self.linear_conv_kernel_dim > 32
            || !self.rms_norm_epsilon.is_finite()
            || !(self.rms_norm_epsilon as f32).is_finite()
            || (self.rms_norm_epsilon as f32) <= 0.0;
        if bad {
            return unsupported("unsupported Qwen text model geometry".to_string());
        }
        Ok(())
    }

    /// Full-attention layers of the base model: the layers that keep a KV cache.
    pub fn attention_layers(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|kind| **kind == LayerKind::FullAttention)
            .count()
    }

    /// Gated DeltaNet layers.
    pub fn gdn_layers(&self) -> usize {
        self.layer_kinds
            .iter()
            .filter(|kind| **kind == LayerKind::GatedDeltaNet)
            .count()
    }

    /// Width of one token's key (or value) rows in the cache: KV heads times the head dimension.
    pub fn kv_width(&self) -> usize {
        self.num_key_value_heads * self.attention_head_dim
    }

    /// Width of the attention queries: query heads times the head dimension.
    pub fn attention_query_width(&self) -> usize {
        self.num_attention_heads * self.attention_head_dim
    }

    /// Width of the GDN query and key rows together: two per key head.
    pub fn gdn_query_key_width(&self) -> usize {
        2 * self.linear_num_key_heads * self.linear_key_head_dim
    }

    /// Width of the GDN value rows.
    pub fn gdn_value_width(&self) -> usize {
        self.linear_num_value_heads * self.linear_value_head_dim
    }

    /// Channels the GDN convolution runs over: query, key and value.
    pub fn gdn_convolution_channels(&self) -> usize {
        self.gdn_query_key_width() + self.gdn_value_width()
    }
}

/// The projections' quantization family (`ArtifactProfile.Quantization`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Quantization {
    Q3,
    Nvfp4,
    Bf16,
    Trellis,
    /// Projections of more than one quantization, each object routed by its own format.
    Mixed,
}

/// The speculative decoding the artifact's drafter selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Speculation {
    None,
    Mtp,
    Dflash2,
}

/// What the artifact's projections and drafter make of it (`ArtifactProfile`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ArtifactProfile {
    pub quantization: Quantization,
    pub compressed: bool,
    pub speculation: Speculation,
    pub draft_depth: Option<u8>,
}

impl ArtifactProfile {
    /// Reads the profile from the text layers' projections and from the presence of the MTP layer and draft head
    /// (`ArtifactProfile.of`). The BF16 projections are ignored when any other quantization is present, since the
    /// small GDN projections stay BF16 in every quantized artifact.
    pub fn of(table: &TensorTable, config: &ModelConfig) -> Result<Self, PlanError> {
        let mut families: Vec<Quantization> = Vec::new();
        let mut probe: Option<(WeightFormat, WeightLayout)> = None;
        let (mut mtp, mut draft_head, mut dflash2) = (false, false, false);
        for entry in table.entries() {
            let name = entry.name();
            if probe.is_none() && name.starts_with("text/layers/") && name.ends_with("/mlp/gate_up") {
                probe = Some((entry.format(), entry.layout()));
            }
            mtp |= name.starts_with("mtp/");
            draft_head |= name.starts_with("text/draft_head");
            dflash2 |= name.starts_with("dflash2/");
            if !name.starts_with("text/layers/") || is_companion(name) {
                continue;
            }
            if let Some((family, _)) = registered(entry.format(), entry.layout())
                && !families.contains(&family)
            {
                families.push(family);
            }
        }
        if families.len() > 1 {
            families.retain(|family| *family != Quantization::Bf16);
        }
        let (format, layout) =
            probe.ok_or_else(|| PlanError::Inventory("the artifact has no text layer FFN projection".to_string()))?;
        let speculation = if dflash2 {
            Speculation::Dflash2
        } else if mtp && draft_head {
            Speculation::Mtp
        } else {
            Speculation::None
        };
        if families.len() > 1 {
            return Ok(Self {
                quantization: Quantization::Mixed,
                compressed: false,
                speculation,
                draft_depth: config.draft_depth,
            });
        }
        let (quantization, compressed) = registered(format, layout).ok_or_else(|| {
            PlanError::Unsupported(format!(
                "unsupported projection format {format:?}/{layout:?}; expected Q3, NVFP4, BF16 or trellis"
            ))
        })?;
        Ok(Self {
            quantization,
            compressed,
            speculation,
            draft_depth: config.draft_depth,
        })
    }

    /// The attention cache format a sequence takes unless the settings select one: BF16 for the unquantized model,
    /// NVFP4 for every quantized one (`ArtifactProfile.kvFormat`).
    pub fn kv_format(&self) -> KvFormat {
        if self.quantization == Quantization::Bf16 {
            KvFormat::Bf16
        } else {
            KvFormat::Nvfp4
        }
    }

    /// Drafts per verification: the stored MTP depth, else 2 for Q3 and 4 for every other pure artifact; a mixed
    /// artifact must store its depth (`ArtifactProfile.draftLength`). DFlash2 verifies the first 6 of a block's 7
    /// drafts. Zero without a drafter.
    pub fn speculative_depth(&self) -> Result<usize, PlanError> {
        match self.speculation {
            Speculation::None => Ok(0),
            Speculation::Dflash2 => Ok(6),
            Speculation::Mtp => match (self.draft_depth, self.quantization) {
                (Some(depth), _) => Ok(usize::from(depth)),
                (None, Quantization::Mixed) => Err(PlanError::Inventory(
                    "a mixed artifact stores no MTP draft depth".to_string(),
                )),
                (None, Quantization::Q3) => Ok(2),
                (None, _) => Ok(4),
            },
        }
    }

    /// Whether the load places the named object on the device (`CompactWeightLoader.uploads`): the MTP layer and
    /// the draft head only for MTP, the DFlash2 drafter's weights only for DFlash2.
    pub fn uploads(&self, name: &str) -> bool {
        if name.starts_with("mtp/") || name.starts_with("text/draft_head") {
            return self.speculation == Speculation::Mtp;
        }
        if name.starts_with("dflash2/") {
            return self.speculation == Speculation::Dflash2 && name != "dflash2/config";
        }
        true
    }
}

fn is_companion(name: &str) -> bool {
    name.ends_with(".suh") || name.ends_with(".svh") || name.ends_with(".segments")
}

/// The family and compression of a registered linear format, if `(format, layout)` is one.
fn registered(format: WeightFormat, layout: WeightLayout) -> Option<(Quantization, bool)> {
    match (format, layout) {
        (WeightFormat::Q3G64Fp16 | WeightFormat::Q4G64Fp16 | WeightFormat::Q5G64Fp16, WeightLayout::RowSplitK128V1) => {
            Some((Quantization::Q3, false))
        }
        (WeightFormat::Q3G64Fp16, WeightLayout::RowSplitP2e2V1) => Some((Quantization::Q3, true)),
        (WeightFormat::Nvfp4, WeightLayout::RowSplitK128V1) => Some((Quantization::Nvfp4, false)),
        (WeightFormat::Nvfp4, WeightLayout::RowSplitK128Sd4V1) => Some((Quantization::Nvfp4, true)),
        (WeightFormat::Bf16, WeightLayout::ContiguousLeV1) => Some((Quantization::Bf16, false)),
        (WeightFormat::Trellis, WeightLayout::TrellisTile16V1) => Some((Quantization::Trellis, false)),
        _ => None,
    }
}
