//! The model's device tensors, found by layer and role, and the names the artifact stores them under.

use std::collections::HashMap;
use std::sync::Arc;

use euhedral_inference_cuda::{DeviceBuffer, DevicePtr};
use euhedral_inference_edrl::{TensorDataType, WeightFormat, WeightLayout};

/// A layer of the model. The decoder layers count from 0; the MTP layer follows them (index 64 in the 64-layer
/// model, as its KV cache does); the tensors that belong to no layer (embedding, heads, final norm, the MTP stem)
/// take [`LayerIndex::GLOBAL`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LayerIndex(pub u16);

impl LayerIndex {
    /// The tensors that belong to no layer.
    pub const GLOBAL: LayerIndex = LayerIndex(u16::MAX);
}

/// The vector a norm layer scales by.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum NormKind {
    Input,
    PostAttention,
    /// The attention queries' norm.
    Query,
    /// The attention keys' norm.
    Key,
    /// The GDN output norm.
    Gdn,
    Final,
}

/// The tensors of the MTP stem, which belong to no decoder layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum MtpRole {
    InputProjection,
    EmbeddingNorm,
    HiddenNorm,
    FinalNorm,
}

/// What a tensor is for within its layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Role {
    /// The attention query and key rows.
    QueryKey,
    /// The attention gate and value rows.
    GateValue,
    /// The GDN query and key rows.
    Qk,
    /// The GDN value and z rows.
    ValueZ,
    GdnA,
    GdnB,
    /// The output projection of the attention or GDN mixer.
    Out,
    GateUp,
    Down,
    Norm(NormKind),
    /// The GDN decay base.
    ALog,
    /// The GDN time-step bias.
    DtBias,
    /// The GDN convolution kernel.
    Convolution,
    LmHead,
    DraftHead,
    DraftTokenIds,
    Embedding,
    Mtp(MtpRole),
}

/// The three companions of a trellis object, which stay on the device with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TrellisParts {
    /// The input scale vectors, one row per segment.
    pub suh: DevicePtr,
    /// The output scale vector.
    pub svh: DevicePtr,
    /// The segment ends as the device reads them.
    pub segments: DevicePtr,
    /// The segment ends, known on the host from the artifact: strictly increasing multiples of 128, the last `n`.
    pub segment_ends: Arc<[u32]>,
    /// Rows of `suh`: the segment count.
    pub suh_rows: u64,
}

/// One tensor on the device: where it is, how it is stored, and what goes with it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WeightRef {
    /// The artifact's name for the tensor (the name of the half for a split MTP projection).
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: TensorDataType,
    pub format: WeightFormat,
    pub layout: WeightLayout,
    pub ptr: DevicePtr,
    pub bytes: u64,
    /// The companions of a trellis object.
    pub trellis: Option<TrellisParts>,
}

/// The device tensors of a loaded model, in one arena, keyed by `(LayerIndex, Role)`.
#[derive(Debug)]
pub struct Weights {
    tensors: HashMap<(LayerIndex, Role), WeightRef>,
    // Declared last: the tensors above address this memory.
    arena: DeviceBuffer,
}

impl Weights {
    /// The weights of `tensors`, which address `arena`.
    pub fn new(arena: DeviceBuffer, tensors: HashMap<(LayerIndex, Role), WeightRef>) -> Self {
        Self { tensors, arena }
    }

    /// The tensor of `layer` in `role`.
    pub fn get(&self, layer: LayerIndex, role: Role) -> Option<&WeightRef> {
        self.tensors.get(&(layer, role))
    }

    /// Every tensor with its key.
    pub fn iter(&self) -> impl Iterator<Item = (&(LayerIndex, Role), &WeightRef)> {
        self.tensors.iter()
    }

    /// How many tensors are loaded.
    pub fn len(&self) -> usize {
        self.tensors.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tensors.is_empty()
    }

    /// The bytes the arena holds on the device.
    pub fn device_bytes(&self) -> u64 {
        self.arena.bytes() as u64
    }
}

/// The key of the object an artifact tensor name stands for, for a model of `layers` decoder layers; `None` for a
/// name that is not an object (a companion, or a name the engine does not know).
///
/// The MTP layer's tensors take [`LayerIndex`]`(layers)` and the roles of a base layer. Its attention projection
/// is stored packed (`query_key_gate_value`); [`split_mtp_pack`] names its two halves.
pub fn object_key(name: &str, layers: usize) -> Option<(LayerIndex, Role)> {
    let global = LayerIndex::GLOBAL;
    match name {
        "text/token_embedding" => return Some((global, Role::Embedding)),
        "text/output_head" => return Some((global, Role::LmHead)),
        "text/draft_head" => return Some((global, Role::DraftHead)),
        "text/draft_head_token_ids" => return Some((global, Role::DraftTokenIds)),
        "text/final_norm" => return Some((global, Role::Norm(NormKind::Final))),
        "mtp/input_projection" => return Some((global, Role::Mtp(MtpRole::InputProjection))),
        "mtp/embedding_norm" => return Some((global, Role::Mtp(MtpRole::EmbeddingNorm))),
        "mtp/hidden_norm" => return Some((global, Role::Mtp(MtpRole::HiddenNorm))),
        "mtp/final_norm" => return Some((global, Role::Mtp(MtpRole::FinalNorm))),
        _ => {}
    }
    let (layer, rest) = if let Some(rest) = name.strip_prefix("mtp/layer/") {
        (LayerIndex(u16::try_from(layers).ok()?), rest)
    } else {
        let rest = name.strip_prefix("text/layers/")?;
        let (index, rest) = rest.split_once('/')?;
        (LayerIndex(index.parse().ok()?), rest)
    };
    let role = match rest {
        "input_norm" => Role::Norm(NormKind::Input),
        "post_attention_norm" => Role::Norm(NormKind::PostAttention),
        "mlp/gate_up" => Role::GateUp,
        "mlp/down" => Role::Down,
        "attention/query_key" => Role::QueryKey,
        "attention/gate_value" => Role::GateValue,
        "attention/query_norm" => Role::Norm(NormKind::Query),
        "attention/key_norm" => Role::Norm(NormKind::Key),
        "attention/output" | "gdn/output" => Role::Out,
        "gdn/a_log" => Role::ALog,
        "gdn/dt_bias" => Role::DtBias,
        "gdn/convolution" => Role::Convolution,
        "gdn/a_projection" => Role::GdnA,
        "gdn/b_projection" => Role::GdnB,
        "gdn/query_key" => Role::Qk,
        "gdn/value_z" => Role::ValueZ,
        "gdn/norm" => Role::Norm(NormKind::Gdn),
        _ => return None,
    };
    Some((layer, role))
}

/// The MTP layer's packed attention projection (`query_key_gate_value`): the base layers' query and key rows followed
/// by their gate and value rows. The loader cuts it into two halves of equal rows.
pub const MTP_ATTENTION_PACK: &str = "mtp/layer/attention/query_key_gate_value";

/// The roles of the two halves of [`MTP_ATTENTION_PACK`], in row order.
pub const MTP_PACK_HALVES: [Role; 2] = [Role::QueryKey, Role::GateValue];

/// The name a trellis object's companion has: `suffix` is `.suh`, `.svh` or `.segments`.
pub fn companion_name(object: &str, suffix: &str) -> String {
    format!("{object}{suffix}")
}

/// Whether `name` is a trellis companion by its suffix.
pub fn is_companion(name: &str) -> bool {
    [".suh", ".svh", ".segments"]
        .iter()
        .any(|suffix| name.ends_with(suffix))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_map_to_layers_and_roles() {
        let key = |name: &str| object_key(name, 64);
        assert_eq!(key("text/token_embedding"), Some((LayerIndex::GLOBAL, Role::Embedding)));
        assert_eq!(key("text/layers/7/gdn/value_z"), Some((LayerIndex(7), Role::ValueZ)));
        assert_eq!(key("text/layers/3/attention/output"), Some((LayerIndex(3), Role::Out)));
        assert_eq!(key("text/layers/63/mlp/gate_up"), Some((LayerIndex(63), Role::GateUp)));
        assert_eq!(
            key("mtp/layer/attention/query_norm"),
            Some((LayerIndex(64), Role::Norm(NormKind::Query)))
        );
        assert_eq!(
            key("mtp/input_projection"),
            Some((LayerIndex::GLOBAL, Role::Mtp(MtpRole::InputProjection)))
        );
        assert_eq!(key("text/layers/3/mlp/down.suh"), None, "companions are not objects");
        assert_eq!(key(MTP_ATTENTION_PACK), None, "the pack is split into its halves");
        assert_eq!(key("text/layers/x/mlp/down"), None);
    }

    #[test]
    fn companions_are_told_by_suffix() {
        assert!(is_companion("text/layers/0/mlp/down.suh"));
        assert!(is_companion("a.segments"));
        assert!(!is_companion("text/layers/0/mlp/down"));
        assert_eq!(companion_name("a", ".svh"), "a.svh");
    }
}
