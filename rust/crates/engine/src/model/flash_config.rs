//! The dimensions of a Flash-Next (`qwen4_exp`) model, from the version-3 artifact's metadata (`Qwen4Config`).
//!
//! The artifact reader owns the file's configuration and refuses what is malformed in it. This type holds the numbers
//! the engine sizes memory from and refuses the geometries the engine's sizing cannot honour ([`check_supported`]).
//!
//! [`check_supported`]: FlashModelConfig::check_supported

use euhedral_inference_edrl::{FlashConfig, FlashLayerType, FlashMetadata};

use crate::error::PlanError;
use crate::residency::flash::{LARGEST_PREFILL_CHUNK_TOKENS, SequenceState};

/// The width of the values one NVFP4 row of the sequence's KV pages holds: a token stores one key row and one value row
/// per this many values of each KV head.
pub const KV_ROW_VALUES: u64 = 256;

/// The dimensions the engine reads from a Flash-Next artifact. Every count is widened to `u64` so the sizing's
/// products are checked, not wrapped.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlashModelConfig {
    pub vocab_size: u64,
    pub hidden_size: u64,
    pub num_layers: u64,
    pub max_position_embeddings: u64,
    /// Text layers that keep a KV cache and an indexer.
    pub sparse_attention_layers: u64,
    pub attention_heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    /// Tokens an indexer key stands for.
    pub qsa_compress_ratio: u64,
    pub qsa_kv_heads: u64,
    pub qsa_head_dim: u64,
    pub gdn_key_heads: u64,
    pub gdn_value_heads: u64,
    pub gdn_key_head_dim: u64,
    pub gdn_value_head_dim: u64,
    pub gdn_conv_kernel_dim: u64,
    /// Routed experts of one layer.
    pub num_experts: u64,
    pub experts_per_token: u64,
    pub expert_intermediate_size: u64,
    pub shared_expert_intermediate_size: u64,
    /// Parallel residual streams.
    pub hyper_connections: u64,
}

impl FlashModelConfig {
    /// Reads the configuration from the version-3 metadata and refuses what the reader or [`check_supported`] refuses,
    /// naming it.
    ///
    /// [`check_supported`]: FlashModelConfig::check_supported
    pub fn from_metadata(metadata: &FlashMetadata) -> Result<Self, PlanError> {
        let flash = FlashConfig::from_metadata(metadata).map_err(|error| PlanError::Config(error.to_string()))?;
        Self::from_flash(&flash)
    }

    /// The dimensions of an artifact's configuration, refused when [`check_supported`] refuses them.
    ///
    /// [`check_supported`]: FlashModelConfig::check_supported
    pub fn from_flash(flash: &FlashConfig) -> Result<Self, PlanError> {
        let sparse = flash
            .text
            .layer_types
            .iter()
            .filter(|kind| **kind == FlashLayerType::SparseAttention)
            .count();
        let wide = |value: u32| u64::from(value);
        let config = Self {
            vocab_size: wide(flash.text.vocab_size),
            hidden_size: wide(flash.text.hidden_size),
            num_layers: wide(flash.text.num_layers),
            max_position_embeddings: wide(flash.text.max_position_embeddings),
            sparse_attention_layers: sparse as u64,
            attention_heads: wide(flash.attention.num_heads),
            kv_heads: wide(flash.attention.num_kv_heads),
            head_dim: wide(flash.attention.head_dim),
            qsa_compress_ratio: wide(flash.qsa.compress_ratio),
            qsa_kv_heads: wide(flash.qsa.kv_heads),
            qsa_head_dim: wide(flash.qsa.head_dim),
            gdn_key_heads: wide(flash.gdn.num_key_heads),
            gdn_value_heads: wide(flash.gdn.num_value_heads),
            gdn_key_head_dim: wide(flash.gdn.key_head_dim),
            gdn_value_head_dim: wide(flash.gdn.value_head_dim),
            gdn_conv_kernel_dim: wide(flash.gdn.conv_kernel_dim),
            num_experts: wide(flash.moe.num_experts),
            experts_per_token: wide(flash.moe.experts_per_token),
            expert_intermediate_size: wide(flash.moe.intermediate_size),
            shared_expert_intermediate_size: wide(flash.moe.shared_expert_intermediate_size),
            hyper_connections: wide(flash.hyper_connection.count),
        };
        config.check_supported()?;
        Ok(config)
    }

    /// Text layers that run Gated DeltaNet.
    pub fn gdn_layers(&self) -> u64 {
        self.num_layers - self.sparse_attention_layers
    }

    /// Channels the GDN convolution runs over: query, key and value.
    pub fn gdn_convolution_channels(&self) -> Option<u64> {
        let keys = 2u64
            .checked_mul(self.gdn_key_heads)?
            .checked_mul(self.gdn_key_head_dim)?;
        keys.checked_add(self.gdn_value_heads.checked_mul(self.gdn_value_head_dim)?)
    }

    /// Refuses a geometry the engine's sizing and kernels do not run: the artifact reader has proved the relations of
    /// the file's own configuration; this proves the divisions and products the plan makes. The sequence state's bytes
    /// of a context as long as the model's positions, and the workspace of the longest prefill chunk, must be
    /// computable without overflow, so the planner's sums never wrap.
    pub fn check_supported(&self) -> Result<(), PlanError> {
        let unsupported = |why: String| Err(PlanError::Unsupported(why));
        let row_values = self.kv_heads.saturating_mul(self.head_dim);
        if row_values == 0 || !row_values.is_multiple_of(KV_ROW_VALUES) {
            return unsupported(format!(
                "attention KV heads {} times head dimension {} is {row_values} values, not a positive multiple of \
                 the {KV_ROW_VALUES} values of one NVFP4 KV row",
                self.kv_heads, self.head_dim
            ));
        }
        if self.qsa_compress_ratio == 0 || self.qsa_kv_heads == 0 || self.qsa_head_dim == 0 {
            return unsupported(format!(
                "the indexer needs a compress ratio, KV heads and a head dimension above 0, not {}, {} and {}",
                self.qsa_compress_ratio, self.qsa_kv_heads, self.qsa_head_dim
            ));
        }
        if self.gdn_conv_kernel_dim < 2 {
            return unsupported(format!(
                "the GDN convolution kernel of {} taps keeps no history to size",
                self.gdn_conv_kernel_dim
            ));
        }
        if self.shared_expert_intermediate_size == 0 || self.expert_intermediate_size == 0 {
            return unsupported(format!(
                "expert intermediate sizes must be above 0, not {} routed and {} shared",
                self.expert_intermediate_size, self.shared_expert_intermediate_size
            ));
        }
        let overflow = |what: &str| PlanError::Unsupported(format!("the {what} of this geometry overflows 64 bits"));
        let longest = usize::try_from(self.max_position_embeddings).unwrap_or(usize::MAX);
        SequenceState::new(self)
            .context_bytes(longest)
            .ok_or_else(|| overflow("sequence state"))?;
        SequenceState::new(self)
            .workspace_bytes(LARGEST_PREFILL_CHUNK_TOKENS)
            .ok_or_else(|| overflow("workspace"))?;
        SequenceState::new(self)
            .shared_down_padding_bytes()
            .ok_or_else(|| overflow("shared expert padding"))?;
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod fixtures {
    use euhedral_inference_edrl::{
        AttentionConfig, FlashConfig, FlashLayerType, GdnConfig, HyperConnectionConfig, MoeConfig, MtpConfig,
        NgramConfig, PleConfig, QsaConfig, TextConfig, VisionConfig,
    };

    /// A small configuration the artifact reader accepts: two text layers, one of each mixer, 3 routed experts per
    /// layer of which 2 serve a token.
    pub(crate) fn flash_config() -> FlashConfig {
        FlashConfig {
            text: TextConfig {
                vocab_size: 32,
                hidden_size: 128,
                num_layers: 2,
                max_position_embeddings: 4096,
                rms_norm_epsilon: 1e-6,
                hidden_activation: "silu".into(),
                tie_word_embeddings: false,
                bos_token_id: 1,
                eos_token_id: 2,
                layer_types: vec![FlashLayerType::GatedDeltaNet, FlashLayerType::SparseAttention],
            },
            attention: AttentionConfig {
                num_heads: 2,
                num_kv_heads: 1,
                head_dim: 256,
                partial_rotary_factor: 0.25,
                rope_theta: 10000.0,
                mrope_section: vec![1, 1, 1],
                mrope_interleaved: true,
                output_gate: "sigmoid".into(),
                full_attention_interval: 2,
            },
            gdn: GdnConfig {
                num_key_heads: 1,
                num_value_heads: 2,
                key_head_dim: 16,
                value_head_dim: 64,
                conv_kernel_dim: 4,
                ssm_dtype: "float32".into(),
            },
            qsa: QsaConfig {
                indexer_budget: 8,
                compress_ratio: 4,
                head_dim: 16,
                kv_heads: 1,
                num_heads: 2,
            },
            moe: MoeConfig {
                num_experts: 3,
                experts_per_token: 2,
                intermediate_size: 128,
                shared_expert_intermediate_size: 128,
            },
            hyper_connection: HyperConnectionConfig { count: 2, lowrank: 4 },
            ngram: NgramConfig {
                size: 2,
                vocab_size_base: 100,
                heads_per_ngram: 2,
                vocab_divisible_by: 1,
                split_parts: 2,
                shard_rows: 16,
                layer_multipliers: vec![3, 5],
                heads_offsets: vec![0, 10],
                heads_vocab_sizes: vec![10, 12],
            },
            ple: PleConfig {
                layers: vec![1],
                embed_dim: 64,
                conv_kernel_size: 4,
            },
            mtp: MtpConfig {
                num_layers: 1,
                hybrid: true,
                layer_types: vec![FlashLayerType::GatedDeltaNet],
                rope_theta: 10000.0,
                use_dedicated_embeddings: false,
                use_hidden_state_from_layer: -1,
            },
            vision: VisionConfig {
                depth: 1,
                hidden_size: 16,
                intermediate_size: 32,
                num_heads: 2,
                in_channels: 3,
                patch_size: 2,
                spatial_merge_size: 2,
                temporal_patch_size: 2,
                out_hidden_size: 24,
                num_position_embeddings: 8,
                hidden_activation: "gelu".into(),
                deepstack_visual_indexes: vec![0],
                image_token_id: 3,
                video_token_id: 4,
                start_token_id: 5,
                end_token_id: 6,
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::fixtures::flash_config;
    use super::*;

    #[test]
    fn the_dimensions_are_the_metadata_s_and_the_layer_split_is_counted() {
        let flash = flash_config();
        let config = FlashModelConfig::from_metadata(&flash.to_metadata()).unwrap();
        assert_eq!(config, FlashModelConfig::from_flash(&flash).unwrap());
        assert_eq!(config.num_layers, 2);
        assert_eq!(config.sparse_attention_layers, 1);
        assert_eq!(config.gdn_layers(), 1);
        assert_eq!(config.gdn_convolution_channels(), Some(2 * 16 + 2 * 64));
        assert_eq!((config.num_experts, config.experts_per_token), (3, 2));
    }

    #[test]
    fn metadata_the_reader_refuses_is_refused_by_the_key_it_names() {
        let mut metadata = flash_config().to_metadata();
        metadata.remove("moe.experts_per_token").unwrap();
        let PlanError::Config(why) = FlashModelConfig::from_metadata(&metadata).unwrap_err() else {
            panic!("a missing key is a malformed configuration");
        };
        assert!(why.contains("moe.experts_per_token"), "{why}");
    }

    #[test]
    fn a_kv_row_that_is_not_whole_nvfp4_rows_is_refused_with_its_width() {
        let mut flash = flash_config();
        flash.attention.head_dim = 64;
        let PlanError::Unsupported(why) = FlashModelConfig::from_flash(&flash).unwrap_err() else {
            panic!("an unsupported geometry");
        };
        assert!(why.contains("64 values") && why.contains("256"), "{why}");
    }

    #[test]
    fn a_geometry_whose_sizes_overflow_is_refused_before_a_plan_wraps() {
        let mut flash = flash_config();
        flash.text.max_position_embeddings = u32::MAX;
        flash.attention.num_kv_heads = 1 << 30;
        flash.attention.num_heads = 1 << 30;
        flash.attention.head_dim = 1 << 30;
        let PlanError::Unsupported(why) = FlashModelConfig::from_flash(&flash).unwrap_err() else {
            panic!("an unsupported geometry");
        };
        assert!(why.contains("overflows"), "{why}");
    }

    #[test]
    fn indexer_and_convolution_dimensions_the_sizing_divides_by_are_refused_at_zero() {
        let mut ratio = flash_config();
        ratio.qsa.compress_ratio = 0;
        assert!(matches!(
            FlashModelConfig::from_flash(&ratio),
            Err(PlanError::Unsupported(why)) if why.contains("compress ratio")
        ));
        let mut taps = flash_config();
        taps.gdn.conv_kernel_dim = 1;
        assert!(matches!(
            FlashModelConfig::from_flash(&taps),
            Err(PlanError::Unsupported(why)) if why.contains("1 taps")
        ));
    }

    #[test]
    fn a_zero_routed_expert_width_is_refused_naming_it() {
        let mut flash = flash_config();
        flash.moe.intermediate_size = 0;
        assert!(matches!(
            FlashModelConfig::from_flash(&flash),
            Err(PlanError::Unsupported(why)) if why.contains("0 routed")
        ));
    }
}
