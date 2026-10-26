use crate::error::EdrlError;
use crate::header::MAX_COUNT;
use crate::reader::Reader;

/// The most draft tokens per step an artifact may record (`MtpDecoder.MAX_DRAFTS`).
pub const MAX_DRAFT_DEPTH: u8 = 16;

const MAX_STRING_BYTES: usize = 1 << 20;

/// The kind of a decoder layer; the ordinal is what the artifact stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    FullAttention,
    GatedDeltaNet,
}

/// The model configuration an artifact stores (`Qwen38Config`), in the order the metadata encodes it. Every count
/// and dimension is at most 1,000,000.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelMetadata {
    pub vocab_size: u32,
    pub hidden_size: u32,
    pub num_hidden_layers: u32,
    pub num_attention_heads: u32,
    pub num_key_value_heads: u32,
    pub attention_head_dim: u32,
    pub intermediate_size: u32,
    pub linear_num_key_heads: u32,
    pub linear_num_value_heads: u32,
    pub linear_key_head_dim: u32,
    pub linear_value_head_dim: u32,
    pub linear_conv_kernel_dim: u32,
    pub rms_norm_epsilon: f64,
    pub rope_theta: f64,
    pub partial_rotary_factor: f64,
    pub max_position_embeddings: u32,
    pub hidden_activation: String,
    pub layer_types: Vec<LayerType>,
    pub num_experts: u32,
    pub num_experts_per_token: u32,
    pub moe_intermediate_size: u32,
    pub shared_expert_intermediate_size: u32,
    pub tie_word_embeddings: bool,
    pub attention_output_gate: bool,
    pub mtp_layer_count: u32,
    /// The MTP draft depth the artifact was built for, when it stores one (1 to [`MAX_DRAFT_DEPTH`]).
    pub draft_depth: Option<u8>,
}

/// The metadata region.
#[derive(Debug)]
pub struct Metadata;

impl Metadata {
    /// Decodes the metadata region. The optional trailing `int32` is the draft depth; any other trailing byte is an
    /// error.
    pub fn parse(bytes: &[u8]) -> Result<ModelMetadata, EdrlError> {
        let mut reader = Reader::new(bytes);
        let vocab_size = count(&mut reader, "vocab size")?;
        let hidden_size = count(&mut reader, "hidden size")?;
        let num_hidden_layers = count(&mut reader, "hidden layer count")?;
        let num_attention_heads = count(&mut reader, "attention head count")?;
        let num_key_value_heads = count(&mut reader, "key/value head count")?;
        let attention_head_dim = count(&mut reader, "attention head dimension")?;
        let intermediate_size = count(&mut reader, "intermediate size")?;
        let linear_num_key_heads = count(&mut reader, "linear key head count")?;
        let linear_num_value_heads = count(&mut reader, "linear value head count")?;
        let linear_key_head_dim = count(&mut reader, "linear key head dimension")?;
        let linear_value_head_dim = count(&mut reader, "linear value head dimension")?;
        let linear_conv_kernel_dim = count(&mut reader, "linear convolution kernel dimension")?;
        let rms_norm_epsilon = finite(&mut reader, "RMS norm epsilon")?;
        let rope_theta = finite(&mut reader, "rope theta")?;
        let partial_rotary_factor = finite(&mut reader, "partial rotary factor")?;
        let max_position_embeddings = count(&mut reader, "maximum position embeddings")?;
        let hidden_activation = string(&mut reader, "hidden activation")?;
        let layer_count = count(&mut reader, "layer type count")?;
        let mut layer_types = Vec::with_capacity(layer_count.min(4096) as usize);
        for _ in 0..layer_count {
            layer_types.push(match reader.i32("layer type")? {
                0 => LayerType::FullAttention,
                1 => LayerType::GatedDeltaNet,
                other => {
                    return Err(EdrlError::invalid(format!(
                        "layer type value is outside the supported range: {other}"
                    )));
                }
            });
        }
        let num_experts = count(&mut reader, "expert count")?;
        let num_experts_per_token = count(&mut reader, "experts per token")?;
        let moe_intermediate_size = count(&mut reader, "MoE intermediate size")?;
        let shared_expert_intermediate_size = count(&mut reader, "shared expert intermediate size")?;
        let tie_word_embeddings = boolean(&mut reader, "tie word embeddings")?;
        let attention_output_gate = boolean(&mut reader, "attention output gate")?;
        let mtp_layer_count = count(&mut reader, "MTP layer count")?;
        let draft_depth = if reader.remaining() > 0 {
            let depth = reader.i32("MTP draft depth")?;
            let depth = u8::try_from(depth)
                .ok()
                .filter(|depth| (1..=MAX_DRAFT_DEPTH).contains(depth))
                .ok_or_else(|| EdrlError::invalid("MTP draft depth is outside the supported range"))?;
            Some(depth)
        } else {
            None
        };
        if reader.remaining() > 0 {
            return Err(EdrlError::invalid("metadata contains trailing bytes"));
        }
        if layer_types.len() != num_hidden_layers as usize {
            return Err(EdrlError::invalid("layer type count does not match hidden layer count"));
        }
        Ok(ModelMetadata {
            vocab_size,
            hidden_size,
            num_hidden_layers,
            num_attention_heads,
            num_key_value_heads,
            attention_head_dim,
            intermediate_size,
            linear_num_key_heads,
            linear_num_value_heads,
            linear_key_head_dim,
            linear_value_head_dim,
            linear_conv_kernel_dim,
            rms_norm_epsilon,
            rope_theta,
            partial_rotary_factor,
            max_position_embeddings,
            hidden_activation,
            layer_types,
            num_experts,
            num_experts_per_token,
            moe_intermediate_size,
            shared_expert_intermediate_size,
            tie_word_embeddings,
            attention_output_gate,
            mtp_layer_count,
            draft_depth,
        })
    }
}

/// An `int32` count or dimension: 0 to 1,000,000.
fn count(reader: &mut Reader<'_>, what: &'static str) -> Result<u32, EdrlError> {
    let value = reader.i32(what)?;
    u32::try_from(value)
        .ok()
        .filter(|value| *value <= MAX_COUNT)
        .ok_or_else(|| EdrlError::invalid(format!("{what} is outside the supported range: {value}")))
}

fn finite(reader: &mut Reader<'_>, what: &'static str) -> Result<f64, EdrlError> {
    let value = reader.f64(what)?;
    if !value.is_finite() {
        return Err(EdrlError::invalid("config floating-point values must be finite"));
    }
    Ok(value)
}

fn boolean(reader: &mut Reader<'_>, what: &'static str) -> Result<bool, EdrlError> {
    match reader.u8(what)? {
        0 => Ok(false),
        1 => Ok(true),
        other => Err(EdrlError::invalid(format!("{what} must be 0 or 1, but was {other}"))),
    }
}

fn string(reader: &mut Reader<'_>, what: &'static str) -> Result<String, EdrlError> {
    let length = reader.i32(what)?;
    let length = usize::try_from(length)
        .ok()
        .filter(|length| *length <= MAX_STRING_BYTES)
        .ok_or_else(|| EdrlError::invalid(format!("{what} length is outside the supported range: {length}")))?;
    let bytes = reader.take(length, what)?;
    String::from_utf8(bytes.to_vec()).map_err(|_| EdrlError::invalid(format!("{what} is not valid UTF-8")))
}
