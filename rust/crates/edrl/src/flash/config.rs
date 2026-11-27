use crate::error::EdrlError;
use crate::flash::metadata::{FlashMetadata, MetadataReader, MetadataValue};

/// The architecture name the metadata records.
pub const ARCHITECTURE: &str = "qwen4_exp";

/// The most text layers the format counts.
pub const MAX_LAYERS: u32 = 4096;
/// The most multi-token-prediction layers.
pub const MAX_MTP_LAYERS: u32 = 64;
/// The most layers that hold n-gram tables.
pub const MAX_NGRAM_LAYERS: usize = 64;
/// The most shards one n-gram table splits into.
pub const MAX_SPLIT_PARTS: u32 = 4096;
/// The most routed experts per layer.
pub const MAX_EXPERTS: u32 = 65536;
/// The most heads or values per head a GDN layer may name: the conv channel count is derived from them.
pub const MAX_GDN_DIMENSION: u32 = 1 << 20;

/// The token mixer of a layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FlashLayerType {
    /// Gated DeltaNet (`linear_attention` in the source configuration).
    GatedDeltaNet,
    /// Qwen sparse attention: attention over the keys an indexer selects (`full_attention` in the source).
    SparseAttention,
}

impl FlashLayerType {
    /// The name the metadata stores.
    pub fn source_name(self) -> &'static str {
        match self {
            FlashLayerType::GatedDeltaNet => "linear_attention",
            FlashLayerType::SparseAttention => "full_attention",
        }
    }

    fn parse(name: &str) -> Result<FlashLayerType, EdrlError> {
        match name {
            "linear_attention" => Ok(FlashLayerType::GatedDeltaNet),
            "full_attention" => Ok(FlashLayerType::SparseAttention),
            _ => Err(EdrlError::invalid(format!("unknown layer type '{name}'"))),
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TextConfig {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_layers: u32,
    pub max_position_embeddings: u32,
    pub rms_norm_epsilon: f64,
    pub hidden_activation: String,
    pub tie_word_embeddings: bool,
    pub bos_token_id: i32,
    pub eos_token_id: i32,
    pub layer_types: Vec<FlashLayerType>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct AttentionConfig {
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub partial_rotary_factor: f64,
    pub rope_theta: f64,
    pub mrope_section: Vec<i32>,
    pub mrope_interleaved: bool,
    pub output_gate: String,
    pub full_attention_interval: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GdnConfig {
    pub num_key_heads: u32,
    pub num_value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub conv_kernel_dim: u32,
    pub ssm_dtype: String,
}

impl GdnConfig {
    /// The channels the input convolution runs over: query, key and value.
    pub fn conv_channels(&self) -> u64 {
        2 * u64::from(self.num_key_heads) * u64::from(self.key_head_dim)
            + u64::from(self.num_value_heads) * u64::from(self.value_head_dim)
    }
}

/// Sparse attention: each query attends to at most `indexer_budget` keys chosen by an indexer with `num_heads` query
/// heads and `kv_heads` key heads of `head_dim`, over keys compressed by `compress_ratio`.
#[derive(Debug, Clone, PartialEq)]
pub struct QsaConfig {
    pub indexer_budget: u32,
    pub compress_ratio: u32,
    pub head_dim: u32,
    pub kv_heads: u32,
    pub num_heads: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MoeConfig {
    pub num_experts: u32,
    pub experts_per_token: u32,
    pub intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
}

/// Hyper-connections: `count` parallel residual streams, mixed through a rank-`lowrank` projection.
#[derive(Debug, Clone, PartialEq)]
pub struct HyperConnectionConfig {
    pub count: u32,
    pub lowrank: u32,
}

/// The n-gram embedding: `(size - 1) * heads_per_ngram` hash heads, each addressing a slice of one table that is split
/// row-wise into `split_parts` shards of `shard_rows` rows.
#[derive(Debug, Clone, PartialEq)]
pub struct NgramConfig {
    pub size: u32,
    pub vocab_size_base: i64,
    pub heads_per_ngram: u32,
    pub vocab_divisible_by: u32,
    pub split_parts: u32,
    pub shard_rows: i64,
    pub layer_multipliers: Vec<i64>,
    pub heads_offsets: Vec<i64>,
    pub heads_vocab_sizes: Vec<i64>,
}

impl NgramConfig {
    /// The hash heads: `(size - 1) * heads_per_ngram`.
    pub fn heads(&self) -> u64 {
        u64::from(self.size - 1) * u64::from(self.heads_per_ngram)
    }
}

/// The layers that hold the n-gram tables and their projections, counted from 0.
#[derive(Debug, Clone, PartialEq)]
pub struct PleConfig {
    pub layers: Vec<i32>,
    pub embed_dim: u32,
    pub conv_kernel_size: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct MtpConfig {
    pub num_layers: u32,
    pub hybrid: bool,
    pub layer_types: Vec<FlashLayerType>,
    pub rope_theta: f64,
    pub use_dedicated_embeddings: bool,
    pub use_hidden_state_from_layer: i32,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VisionConfig {
    pub depth: u32,
    pub hidden_size: u32,
    pub intermediate_size: u32,
    pub num_heads: u32,
    pub in_channels: u32,
    pub patch_size: u32,
    pub spatial_merge_size: u32,
    pub temporal_patch_size: u32,
    pub out_hidden_size: u32,
    pub num_position_embeddings: u32,
    pub hidden_activation: String,
    pub deepstack_visual_indexes: Vec<i32>,
    pub image_token_id: i32,
    pub video_token_id: i32,
    pub start_token_id: i32,
    pub end_token_id: i32,
}

/// The topology and hyper-parameters of a `qwen4_exp` model, as an artifact records them. [`FlashConfig::from_metadata`]
/// is the only way to read one from a file and refuses a configuration whose relations do not hold.
#[derive(Debug, Clone, PartialEq)]
pub struct FlashConfig {
    pub text: TextConfig,
    pub attention: AttentionConfig,
    pub gdn: GdnConfig,
    pub qsa: QsaConfig,
    pub moe: MoeConfig,
    pub hyper_connection: HyperConnectionConfig,
    pub ngram: NgramConfig,
    pub ple: PleConfig,
    pub mtp: MtpConfig,
    pub vision: VisionConfig,
}

impl FlashConfig {
    pub fn layer_type(&self, layer: usize) -> FlashLayerType {
        self.text.layer_types[layer]
    }

    /// Whether `layer` holds the n-gram tables.
    pub fn has_ngram(&self, layer: usize) -> bool {
        self.ple
            .layers
            .iter()
            .any(|candidate| usize::try_from(*candidate).is_ok_and(|candidate| candidate == layer))
    }

    /// Reads the configuration from the decoded metadata: exactly the keys it knows, each of its type.
    pub fn from_metadata(metadata: &FlashMetadata) -> Result<FlashConfig, EdrlError> {
        let mut input = MetadataReader::new(metadata);
        if input.string("architecture")? != ARCHITECTURE {
            return Err(EdrlError::invalid(format!(
                "metadata architecture is not {ARCHITECTURE}"
            )));
        }
        input.string("source.model_type")?;
        let layer_types = parse_layer_types(&input.strings("text.layer_types")?)?;
        let text = TextConfig {
            vocab_size: input.count("text.vocab_size")?,
            hidden_size: input.count("text.hidden_size")?,
            num_layers: input.count("text.num_hidden_layers")?,
            max_position_embeddings: input.count("text.max_position_embeddings")?,
            rms_norm_epsilon: input.real("text.rms_norm_eps")?,
            hidden_activation: input.string("text.hidden_act")?,
            tie_word_embeddings: input.boolean("text.tie_word_embeddings")?,
            bos_token_id: input.signed("text.bos_token_id")?,
            eos_token_id: input.signed("text.eos_token_id")?,
            layer_types,
        };
        let attention = AttentionConfig {
            num_heads: input.count("attention.num_heads")?,
            num_kv_heads: input.count("attention.num_kv_heads")?,
            head_dim: input.count("attention.head_dim")?,
            partial_rotary_factor: input.real("attention.partial_rotary_factor")?,
            rope_theta: input.real("attention.rope_theta")?,
            mrope_section: input.ints("attention.mrope_section")?,
            mrope_interleaved: input.boolean("attention.mrope_interleaved")?,
            output_gate: input.string("attention.output_gate")?,
            full_attention_interval: input.count("attention.full_attention_interval")?,
        };
        let gdn = GdnConfig {
            num_key_heads: input.count("gdn.num_key_heads")?,
            num_value_heads: input.count("gdn.num_value_heads")?,
            key_head_dim: input.count("gdn.key_head_dim")?,
            value_head_dim: input.count("gdn.value_head_dim")?,
            conv_kernel_dim: input.count("gdn.conv_kernel_dim")?,
            ssm_dtype: input.string("gdn.ssm_dtype")?,
        };
        let qsa = QsaConfig {
            indexer_budget: input.count("qsa.indexer_budget")?,
            compress_ratio: input.count("qsa.compress_ratio")?,
            head_dim: input.count("qsa.head_dim")?,
            kv_heads: input.count("qsa.kv_heads")?,
            num_heads: input.count("qsa.num_heads")?,
        };
        let moe = MoeConfig {
            num_experts: input.count("moe.num_experts")?,
            experts_per_token: input.count("moe.experts_per_token")?,
            intermediate_size: input.count("moe.intermediate_size")?,
            shared_expert_intermediate_size: input.count("moe.shared_expert_intermediate_size")?,
        };
        let hyper_connection = HyperConnectionConfig {
            count: input.count("hc.count")?,
            lowrank: input.count("hc.lowrank")?,
        };
        let ngram = NgramConfig {
            size: input.count("ngram.size")?,
            vocab_size_base: input.integer("ngram.vocab_size_base")?,
            heads_per_ngram: input.count("ngram.heads_per_ngram")?,
            vocab_divisible_by: input.count("ngram.vocab_divisible_by")?,
            split_parts: input.count("ngram.split_parts")?,
            shard_rows: input.integer("ngram.shard_rows")?,
            layer_multipliers: input.longs("ngram.layer_multipliers")?,
            heads_offsets: input.longs("ngram.heads_offsets")?,
            heads_vocab_sizes: input.longs("ngram.heads_vocab_sizes")?,
        };
        let ple = PleConfig {
            layers: input.ints("ple.layers")?,
            embed_dim: input.count("ple.embed_dim")?,
            conv_kernel_size: input.count("ple.conv_kernel_size")?,
        };
        let mtp = MtpConfig {
            num_layers: input.count("mtp.num_layers")?,
            hybrid: input.boolean("mtp.hybrid")?,
            layer_types: parse_layer_types(&input.strings("mtp.layer_types")?)?,
            rope_theta: input.real("mtp.rope_theta")?,
            use_dedicated_embeddings: input.boolean("mtp.use_dedicated_embeddings")?,
            use_hidden_state_from_layer: input.signed("mtp.use_hidden_state_from_layer")?,
        };
        let vision = VisionConfig {
            depth: input.count("vision.depth")?,
            hidden_size: input.count("vision.hidden_size")?,
            intermediate_size: input.count("vision.intermediate_size")?,
            num_heads: input.count("vision.num_heads")?,
            in_channels: input.count("vision.in_channels")?,
            patch_size: input.count("vision.patch_size")?,
            spatial_merge_size: input.count("vision.spatial_merge_size")?,
            temporal_patch_size: input.count("vision.temporal_patch_size")?,
            out_hidden_size: input.count("vision.out_hidden_size")?,
            num_position_embeddings: input.count("vision.num_position_embeddings")?,
            hidden_activation: input.string("vision.hidden_act")?,
            deepstack_visual_indexes: input.ints("vision.deepstack_visual_indexes")?,
            image_token_id: input.signed("vision.image_token_id")?,
            video_token_id: input.signed("vision.video_token_id")?,
            start_token_id: input.signed("vision.start_token_id")?,
            end_token_id: input.signed("vision.end_token_id")?,
        };
        let format = input.string("quant.format")?;
        let block = input.integer("quant.block")?;
        let scale = input.string("quant.scale_dtype")?;
        if format != "nvfp4" || block != 16 || scale != "e4m3" {
            return Err(EdrlError::invalid(format!(
                "unsupported quantization {format} block {block} scales {scale}"
            )));
        }
        input.finish()?;
        let config = FlashConfig {
            text,
            attention,
            gdn,
            qsa,
            moe,
            hyper_connection,
            ngram,
            ple,
            mtp,
            vision,
        };
        config.validate()?;
        Ok(config)
    }

    /// The metadata an artifact stores for this configuration: [`FlashConfig::from_metadata`] reads it back.
    pub fn to_metadata(&self) -> FlashMetadata {
        let mut m = FlashMetadata::new();
        let int = |value: u32| MetadataValue::Int(i64::from(value));
        let signed = |value: i32| MetadataValue::Int(i64::from(value));
        let text = |value: &str| MetadataValue::Text(value.to_owned());
        let widen = |values: &[i32]| MetadataValue::Ints(values.iter().map(|value| i64::from(*value)).collect());
        let names = |types: &[FlashLayerType]| {
            MetadataValue::Texts(types.iter().map(|kind| kind.source_name().to_owned()).collect())
        };
        let (t, a, g, q, e, h, n, p, x, v) = (
            &self.text,
            &self.attention,
            &self.gdn,
            &self.qsa,
            &self.moe,
            &self.hyper_connection,
            &self.ngram,
            &self.ple,
            &self.mtp,
            &self.vision,
        );
        m.insert("architecture", text(ARCHITECTURE));
        m.insert("source.model_type", text(ARCHITECTURE));
        m.insert("text.vocab_size", int(t.vocab_size));
        m.insert("text.hidden_size", int(t.hidden_size));
        m.insert("text.num_hidden_layers", int(t.num_layers));
        m.insert("text.max_position_embeddings", int(t.max_position_embeddings));
        m.insert("text.rms_norm_eps", MetadataValue::Real(t.rms_norm_epsilon));
        m.insert("text.hidden_act", text(&t.hidden_activation));
        m.insert("text.tie_word_embeddings", MetadataValue::Bool(t.tie_word_embeddings));
        m.insert("text.bos_token_id", signed(t.bos_token_id));
        m.insert("text.eos_token_id", signed(t.eos_token_id));
        m.insert("text.layer_types", names(&t.layer_types));
        m.insert("attention.num_heads", int(a.num_heads));
        m.insert("attention.num_kv_heads", int(a.num_kv_heads));
        m.insert("attention.head_dim", int(a.head_dim));
        m.insert(
            "attention.partial_rotary_factor",
            MetadataValue::Real(a.partial_rotary_factor),
        );
        m.insert("attention.rope_theta", MetadataValue::Real(a.rope_theta));
        m.insert("attention.mrope_section", widen(&a.mrope_section));
        m.insert("attention.mrope_interleaved", MetadataValue::Bool(a.mrope_interleaved));
        m.insert("attention.output_gate", text(&a.output_gate));
        m.insert("attention.full_attention_interval", int(a.full_attention_interval));
        m.insert("gdn.num_key_heads", int(g.num_key_heads));
        m.insert("gdn.num_value_heads", int(g.num_value_heads));
        m.insert("gdn.key_head_dim", int(g.key_head_dim));
        m.insert("gdn.value_head_dim", int(g.value_head_dim));
        m.insert("gdn.conv_kernel_dim", int(g.conv_kernel_dim));
        m.insert("gdn.ssm_dtype", text(&g.ssm_dtype));
        m.insert("qsa.indexer_budget", int(q.indexer_budget));
        m.insert("qsa.compress_ratio", int(q.compress_ratio));
        m.insert("qsa.head_dim", int(q.head_dim));
        m.insert("qsa.kv_heads", int(q.kv_heads));
        m.insert("qsa.num_heads", int(q.num_heads));
        m.insert("moe.num_experts", int(e.num_experts));
        m.insert("moe.experts_per_token", int(e.experts_per_token));
        m.insert("moe.intermediate_size", int(e.intermediate_size));
        m.insert(
            "moe.shared_expert_intermediate_size",
            int(e.shared_expert_intermediate_size),
        );
        m.insert("hc.count", int(h.count));
        m.insert("hc.lowrank", int(h.lowrank));
        m.insert("ngram.size", int(n.size));
        m.insert("ngram.vocab_size_base", MetadataValue::Int(n.vocab_size_base));
        m.insert("ngram.heads_per_ngram", int(n.heads_per_ngram));
        m.insert("ngram.vocab_divisible_by", int(n.vocab_divisible_by));
        m.insert("ngram.split_parts", int(n.split_parts));
        m.insert("ngram.shard_rows", MetadataValue::Int(n.shard_rows));
        m.insert(
            "ngram.layer_multipliers",
            MetadataValue::Ints(n.layer_multipliers.clone()),
        );
        m.insert("ngram.heads_offsets", MetadataValue::Ints(n.heads_offsets.clone()));
        m.insert(
            "ngram.heads_vocab_sizes",
            MetadataValue::Ints(n.heads_vocab_sizes.clone()),
        );
        m.insert("ple.layers", widen(&p.layers));
        m.insert("ple.embed_dim", int(p.embed_dim));
        m.insert("ple.conv_kernel_size", int(p.conv_kernel_size));
        m.insert("mtp.num_layers", int(x.num_layers));
        m.insert("mtp.hybrid", MetadataValue::Bool(x.hybrid));
        m.insert("mtp.layer_types", names(&x.layer_types));
        m.insert("mtp.rope_theta", MetadataValue::Real(x.rope_theta));
        m.insert(
            "mtp.use_dedicated_embeddings",
            MetadataValue::Bool(x.use_dedicated_embeddings),
        );
        m.insert("mtp.use_hidden_state_from_layer", signed(x.use_hidden_state_from_layer));
        m.insert("vision.depth", int(v.depth));
        m.insert("vision.hidden_size", int(v.hidden_size));
        m.insert("vision.intermediate_size", int(v.intermediate_size));
        m.insert("vision.num_heads", int(v.num_heads));
        m.insert("vision.in_channels", int(v.in_channels));
        m.insert("vision.patch_size", int(v.patch_size));
        m.insert("vision.spatial_merge_size", int(v.spatial_merge_size));
        m.insert("vision.temporal_patch_size", int(v.temporal_patch_size));
        m.insert("vision.out_hidden_size", int(v.out_hidden_size));
        m.insert("vision.num_position_embeddings", int(v.num_position_embeddings));
        m.insert("vision.hidden_act", text(&v.hidden_activation));
        m.insert("vision.deepstack_visual_indexes", widen(&v.deepstack_visual_indexes));
        m.insert("vision.image_token_id", signed(v.image_token_id));
        m.insert("vision.video_token_id", signed(v.video_token_id));
        m.insert("vision.start_token_id", signed(v.start_token_id));
        m.insert("vision.end_token_id", signed(v.end_token_id));
        m.insert("quant.format", text("nvfp4"));
        m.insert("quant.block", MetadataValue::Int(16));
        m.insert("quant.scale_dtype", text("e4m3"));
        m
    }

    /// The relations between the fields that every later size and shape computation relies on. The divisors are
    /// checked positive first: a zero would otherwise be a division by zero, not a refusal.
    pub(crate) fn validate(&self) -> Result<(), EdrlError> {
        let (t, a, g, e, h, n, p, x) = (
            &self.text,
            &self.attention,
            &self.gdn,
            &self.moe,
            &self.hyper_connection,
            &self.ngram,
            &self.ple,
            &self.mtp,
        );
        require(
            t.num_layers > 0 && t.num_layers <= MAX_LAYERS && t.layer_types.len() == t.num_layers as usize,
            || {
                format!(
                    "{} layer types for {} layers (1 to {MAX_LAYERS})",
                    t.layer_types.len(),
                    t.num_layers
                )
            },
        )?;
        require(t.hidden_size > 0 && t.hidden_size.is_multiple_of(128), || {
            "hidden size must be a positive multiple of 128".into()
        })?;
        require(t.vocab_size > 0, || "vocabulary size must be positive".into())?;
        require(a.num_kv_heads > 0 && a.num_heads.is_multiple_of(a.num_kv_heads), || {
            "attention heads are not a positive multiple of the KV heads".into()
        })?;
        require(a.output_gate == "sigmoid", || {
            format!("unsupported attention output gate {}", a.output_gate)
        })?;
        require(
            g.num_key_heads > 0 && g.num_value_heads.is_multiple_of(g.num_key_heads),
            || "GDN value heads are not a positive multiple of the key heads".into(),
        )?;
        require(
            [g.num_key_heads, g.num_value_heads, g.key_head_dim, g.value_head_dim]
                .iter()
                .all(|dimension| *dimension <= MAX_GDN_DIMENSION),
            || format!("GDN heads and head dimensions are at most {MAX_GDN_DIMENSION}"),
        )?;
        require(g.ssm_dtype == "float32", || {
            format!("unsupported GDN state type {}", g.ssm_dtype)
        })?;
        require(e.experts_per_token > 0 && e.experts_per_token <= e.num_experts, || {
            "experts per token is out of range".into()
        })?;
        require(e.num_experts <= MAX_EXPERTS, || "expert count is out of range".into())?;
        require(h.count > 0 && h.lowrank > 0, || {
            "hyper-connection sizes must be positive".into()
        })?;
        require(
            n.size >= 2
                && n.heads_per_ngram > 0
                && n.split_parts > 0
                && n.split_parts <= MAX_SPLIT_PARTS
                && n.shard_rows > 0,
            || "n-gram configuration is invalid".into(),
        )?;
        let heads = n.heads();
        require(
            n.heads_offsets.len() as u64 == heads && n.heads_vocab_sizes.len() as u64 == heads,
            || format!("n-gram head tables do not have {heads} entries"),
        )?;
        require(n.layer_multipliers.len() == n.size as usize, || {
            "n-gram multipliers do not have one entry per n-gram size".into()
        })?;
        require(u64::from(t.hidden_size).is_multiple_of(heads), || {
            "hidden size is not a multiple of the n-gram heads".into()
        })?;
        require(!p.layers.is_empty() && p.layers.len() <= MAX_NGRAM_LAYERS, || {
            format!(
                "{} layers hold the n-gram tables (1 to {MAX_NGRAM_LAYERS})",
                p.layers.len()
            )
        })?;
        for layer in &p.layers {
            require(u32::try_from(*layer).is_ok_and(|layer| layer < t.num_layers), || {
                format!("n-gram layer {layer} is out of range")
            })?;
        }
        let total_rows = (n.shard_rows as u64).saturating_mul(u64::from(n.split_parts));
        let mut end: i64 = 0;
        for (head, (offset, size)) in n.heads_offsets.iter().zip(&n.heads_vocab_sizes).enumerate() {
            require(*offset == end, || {
                format!("n-gram head offsets are not cumulative at head {head}")
            })?;
            require(*size > 0, || format!("n-gram head {head} has an empty vocabulary"))?;
            end = end
                .checked_add(*size)
                .ok_or_else(|| EdrlError::invalid(format!("invalid configuration: n-gram head {head} overflows")))?;
        }
        require(end as u64 <= total_rows, || {
            format!("n-gram heads address {end} rows but the shards hold {total_rows}")
        })?;
        require(
            x.num_layers <= MAX_MTP_LAYERS && x.layer_types.len() == x.num_layers as usize,
            || "MTP layer types do not match the MTP layer count".into(),
        )?;
        require(self.vision.depth <= MAX_LAYERS, || {
            "vision depth is out of range".into()
        })?;
        require(t.max_position_embeddings > 0, || {
            "maximum position embeddings must be positive".into()
        })
    }
}

fn require(condition: bool, message: impl FnOnce() -> String) -> Result<(), EdrlError> {
    if condition {
        Ok(())
    } else {
        Err(EdrlError::invalid(format!("invalid configuration: {}", message())))
    }
}

fn parse_layer_types(names: &[String]) -> Result<Vec<FlashLayerType>, EdrlError> {
    names.iter().map(|name| FlashLayerType::parse(name)).collect()
}
