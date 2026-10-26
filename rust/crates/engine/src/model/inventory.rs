//! The objects an artifact must hold for the model, and the shapes they must have
//! (`CompactWeightLoader.validateInventory` and `expectedDescriptor`). Checked on the table alone, so a malformed
//! artifact is refused before any payload is read or any device memory is allocated.

use std::collections::HashSet;

use euhedral_inference_edrl::{TensorDataType, TensorEntry, TensorTable, WeightFormat, WeightLayout};

use crate::error::PlanError;
use crate::model::config::{LayerKind, ModelConfig};
use crate::model::weights::{MTP_ATTENTION_PACK, companion_name, is_companion};

/// The suffixes of a trellis object's companions, in the order its format lists them.
pub const TRELLIS_COMPANIONS: [&str; 3] = [".suh", ".svh", ".segments"];

/// What the table must hold for a tensor named so.
enum Expected {
    /// A vector or small matrix stored as it is: the shape, the source type and the format.
    Direct {
        shape: Vec<u64>,
        dtype: TensorDataType,
        format: WeightFormat,
    },
    /// A projection stored in any quantized format (or BF16): the logical `[n, k]` shape.
    Projection { shape: Vec<u64> },
}

fn is_trellis(entry: &TensorEntry) -> bool {
    entry.format() == WeightFormat::Trellis && entry.layout() == WeightLayout::TrellisTile16V1
}

/// The logical `[n, k]` of a trellis payload `[k/16, n/16, w]`.
pub fn trellis_logical(shape: &[u64]) -> Option<(u64, u64)> {
    match shape {
        [tiles_k, tiles_n, width] if *tiles_k > 0 && *tiles_n > 0 && *width > 0 => Some((tiles_n * 16, tiles_k * 16)),
        _ => None,
    }
}

/// The names of every object the model holds in an artifact of `config`.
pub fn expected_names(config: &ModelConfig) -> Vec<String> {
    let mut names: Vec<String> = [
        "text/token_embedding",
        "text/final_norm",
        "text/output_head",
        "text/draft_head",
        "text/draft_head_token_ids",
    ]
    .iter()
    .map(|name| (*name).to_string())
    .collect();
    for (index, kind) in config.layer_kinds.iter().enumerate() {
        let prefix = format!("text/layers/{index}");
        for tail in ["input_norm", "post_attention_norm", "mlp/gate_up", "mlp/down"] {
            names.push(format!("{prefix}/{tail}"));
        }
        let mixer: &[&str] = match kind {
            LayerKind::FullAttention => &[
                "attention/query_key",
                "attention/gate_value",
                "attention/query_norm",
                "attention/key_norm",
                "attention/output",
            ],
            LayerKind::GatedDeltaNet => &[
                "gdn/a_log",
                "gdn/dt_bias",
                "gdn/convolution",
                "gdn/a_projection",
                "gdn/b_projection",
                "gdn/query_key",
                "gdn/value_z",
                "gdn/norm",
                "gdn/output",
            ],
        };
        names.extend(mixer.iter().map(|tail| format!("{prefix}/{tail}")));
    }
    if config.mtp_layer_count > 0 {
        for name in [
            "mtp/input_projection",
            "mtp/embedding_norm",
            "mtp/hidden_norm",
            "mtp/final_norm",
            "mtp/layer/input_norm",
            "mtp/layer/post_attention_norm",
            MTP_ATTENTION_PACK,
            "mtp/layer/attention/query_norm",
            "mtp/layer/attention/key_norm",
            "mtp/layer/attention/output",
            "mtp/layer/mlp/gate_up",
            "mtp/layer/mlp/down",
        ] {
            names.push(name.to_string());
        }
    }
    names
}

fn direct(shape: &[usize], dtype: TensorDataType, format: WeightFormat) -> Expected {
    Expected::Direct {
        shape: shape.iter().map(|dimension| *dimension as u64).collect(),
        dtype,
        format,
    }
}

fn projection(rows: usize, columns: usize) -> Expected {
    Expected::Projection {
        shape: vec![rows as u64, columns as u64],
    }
}

/// What `name` must be (`CompactWeightLoader.expectedDescriptor`, with the dimensions taken from `config`).
fn expected(config: &ModelConfig, name: &str) -> Option<Expected> {
    let hidden = config.hidden_size;
    let attention_query = config.attention_query_width();
    let attention_qk = attention_query + config.kv_width();
    let norm = || direct(&[hidden], TensorDataType::Bf16, WeightFormat::Bf16);
    let head_norm = || direct(&[config.attention_head_dim], TensorDataType::Bf16, WeightFormat::Bf16);
    Some(match name {
        "text/token_embedding" | "text/output_head" => projection(config.vocab_size, hidden),
        "text/final_norm" => norm(),
        "mtp/input_projection" => projection(hidden, 2 * hidden),
        MTP_ATTENTION_PACK => projection(2 * attention_qk, hidden),
        "mtp/embedding_norm" | "mtp/hidden_norm" | "mtp/final_norm" => norm(),
        "mtp/layer/input_norm" | "mtp/layer/post_attention_norm" => norm(),
        "mtp/layer/attention/query_norm" | "mtp/layer/attention/key_norm" => head_norm(),
        "mtp/layer/attention/output" => projection(hidden, attention_query),
        "mtp/layer/mlp/gate_up" => projection(2 * config.intermediate_size, hidden),
        "mtp/layer/mlp/down" => projection(hidden, config.intermediate_size),
        _ => {
            let tail = name.strip_prefix("text/layers/")?.split_once('/')?.1;
            match tail {
                "input_norm" | "post_attention_norm" => norm(),
                "mlp/gate_up" => projection(2 * config.intermediate_size, hidden),
                "mlp/down" => projection(hidden, config.intermediate_size),
                "attention/query_key" | "attention/gate_value" => projection(attention_qk, hidden),
                "attention/query_norm" | "attention/key_norm" => head_norm(),
                "attention/output" => projection(hidden, attention_query),
                "gdn/a_log" | "gdn/dt_bias" => direct(
                    &[config.linear_num_value_heads],
                    TensorDataType::Bf16,
                    WeightFormat::Fp32,
                ),
                "gdn/convolution" => direct(
                    &[config.linear_conv_kernel_dim, config.gdn_convolution_channels()],
                    TensorDataType::Bf16,
                    WeightFormat::Bf16,
                ),
                "gdn/a_projection" | "gdn/b_projection" => direct(
                    &[config.linear_num_value_heads, hidden],
                    TensorDataType::Bf16,
                    WeightFormat::Bf16,
                ),
                "gdn/query_key" => projection(config.gdn_query_key_width(), hidden),
                "gdn/value_z" => projection(2 * config.gdn_value_width(), hidden),
                "gdn/norm" => direct(
                    &[config.linear_value_head_dim],
                    TensorDataType::Bf16,
                    WeightFormat::Bf16,
                ),
                "gdn/output" => projection(hidden, config.gdn_value_width()),
                _ => return None,
            }
        }
    })
}

/// Whether `entry` may stand for `expected`.
fn accepts(expected: &Expected, entry: &TensorEntry) -> bool {
    match expected {
        Expected::Direct { shape, dtype, format } => {
            entry.shape() == shape.as_slice()
                && entry.dtype() == *dtype
                && entry.format() == *format
                && entry.layout() == WeightLayout::ContiguousLeV1
        }
        Expected::Projection { shape } => {
            if entry.dtype() != TensorDataType::Bf16 {
                return false;
            }
            if is_trellis(entry) {
                return trellis_logical(entry.shape()).is_some_and(|(n, k)| [n, k] == shape.as_slice());
            }
            // A quantized or BF16 projection: its stored shape is the logical one.
            entry.shape() == shape.as_slice()
        }
    }
}

/// Checks that `table` holds exactly the model's objects with the shapes the model gives them, every trellis object
/// with its three well-formed companions, and nothing else.
pub fn validate_inventory(config: &ModelConfig, table: &TensorTable) -> Result<(), PlanError> {
    let invalid = |why: String| Err(PlanError::Inventory(why));
    let mut companions: Vec<String> = Vec::new();
    for entry in table.entries() {
        if !is_trellis(entry) {
            continue;
        }
        let name = entry.name();
        let Some((n, k)) = trellis_logical(entry.shape()) else {
            return invalid(format!("trellis object '{name}' is not [k/16, n/16, w]"));
        };
        let mut found = Vec::with_capacity(3);
        for suffix in TRELLIS_COMPANIONS {
            let companion = companion_name(name, suffix);
            let Some(tensor) = table.get(&companion) else {
                return invalid(format!(
                    "trellis object '{name}' is missing its companion '{companion}'"
                ));
            };
            let (dtype, format) = if suffix == ".segments" {
                (TensorDataType::Int32, WeightFormat::I32)
            } else {
                (TensorDataType::Fp16, WeightFormat::Fp16)
            };
            if tensor.format() != format || tensor.dtype() != dtype || tensor.layout() != WeightLayout::ContiguousLeV1 {
                return invalid(format!(
                    "trellis companion '{companion}' must be {dtype:?}/{format:?}/ContiguousLeV1"
                ));
            }
            found.push((companion, tensor));
        }
        let (suh, svh, segments) = (&found[0].1, &found[1].1, &found[2].1);
        if suh.shape().len() != 2 || suh.shape()[1] != k || suh.shape()[0] < 1 {
            return invalid(format!(
                "trellis companion '{}' must have shape [segments, {k}]",
                found[0].0
            ));
        }
        if svh.shape() != [n] {
            return invalid(format!("trellis companion '{}' must have shape [{n}]", found[1].0));
        }
        if segments.shape() != [suh.shape()[0]] {
            return invalid(format!(
                "trellis companion '{}' must have one end per suh row ({})",
                found[2].0,
                suh.shape()[0]
            ));
        }
        companions.extend(found.into_iter().map(|(companion, _)| companion));
    }
    // No companion stands without its object.
    for entry in table.entries() {
        let name = entry.name();
        for suffix in TRELLIS_COMPANIONS {
            let Some(owner) = name.strip_suffix(suffix) else {
                continue;
            };
            if !table.get(owner).is_some_and(is_trellis) {
                return invalid(format!("trellis companion '{name}' has no trellis object"));
            }
        }
    }

    let expected_names = expected_names(config);
    let known: HashSet<&str> = expected_names
        .iter()
        .map(String::as_str)
        .chain(companions.iter().map(String::as_str))
        .collect();
    let want = expected_names.len() + companions.len();
    if table.len() != want {
        return invalid(format!(
            "a compact Qwen artifact must contain {want} runtime objects, found {}",
            table.len()
        ));
    }
    for name in &expected_names {
        let Some(entry) = table.get(name) else {
            return invalid(format!("the artifact is missing runtime object '{name}'"));
        };
        if let Some(expected) = expected(config, name)
            && !accepts(&expected, entry)
        {
            return invalid(format!(
                "runtime metadata of '{name}' conflicts with the model: shape {:?}, {:?}/{:?}",
                entry.shape(),
                entry.format(),
                entry.layout()
            ));
        }
        // The token embedding and the heads are read by row or in place: they have no trellis route.
        if is_trellis(entry)
            && matches!(
                name.as_str(),
                "text/token_embedding" | "text/output_head" | "text/draft_head"
            )
        {
            return invalid(format!(
                "'{name}' cannot be trellis: the token embedding and the heads are Q3, NVFP4 or BF16"
            ));
        }
    }
    for entry in table.entries() {
        if !known.contains(entry.name())
            && !(is_companion(entry.name()) && companions.iter().any(|c| c == entry.name()))
        {
            return invalid(format!(
                "the artifact contains unknown runtime object '{}'",
                entry.name()
            ));
        }
    }
    Ok(())
}
