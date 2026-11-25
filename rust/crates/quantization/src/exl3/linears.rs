//! The text linears of a Qwen 3.8 model: which object's Hessian each one shares, where its rows sit in the layer's
//! fused objects, and the allocation inputs (priority and promotion group) of the budgeted ones.
//!
//! Portions ported from exllamav3 (MIT License, Copyright (c) 2025 Turboderp).

use super::allocation::Linear as Budgeted;
use crate::checkpoint::CheckpointError;
use crate::checkpoint::objects::Qwen38Objects;

const TEXT: &str = "model.language_model.layers.";

/// The eight linears of the MTP layer (tensor names without `.weight`).
pub const MTP_LINEARS: [&str; 8] = [
    "mtp.fc",
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.mlp.down_proj",
];

/// The shapes `[n, k]` of the linears of the 64-layer Swift 1.5 model, by role: what [`budget`] needs when there is
/// no checkpoint to read them from.
pub const SWIFT_SHAPES: [(&str, [u64; 2]); 10] = [
    ("q_proj", [12288, 5120]),
    ("k_proj", [1024, 5120]),
    ("v_proj", [1024, 5120]),
    ("o_proj", [5120, 6144]),
    ("in_proj_qkv", [10240, 5120]),
    ("in_proj_z", [6144, 5120]),
    ("out_proj", [5120, 6144]),
    ("gate_proj", [17408, 5120]),
    ("up_proj", [17408, 5120]),
    ("down_proj", [5120, 17408]),
];

/// A linear of a decoder layer.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Linear {
    /// The tensor's name without `.weight`.
    pub tensor: String,
    /// `n`: the output rows.
    pub rows: u64,
    /// `k`: the input columns.
    pub columns: u64,
    /// The object whose observed Hessian it shares.
    pub hessian_object: String,
    /// False when `n` or `k` is not a multiple of 128: such a linear is stored as F16.
    pub quantized: bool,
}

/// Where `rows` rows of a linear (from `linear_row`) sit in an object (from `object_row`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    pub object: String,
    pub object_row: u64,
    pub linear_row: u64,
    pub rows: u64,
}

/// The linears of decoder layer `index`, in module order.
pub fn layer(checkpoint: &Qwen38Objects, index: usize) -> Result<Vec<Linear>, CheckpointError> {
    let full = *checkpoint
        .config
        .full_attention
        .get(index)
        .ok_or_else(|| CheckpointError::Invalid(format!("the model has no layer {index}")))?;
    let p = format!("{TEXT}{index}.");
    let o = format!("text/layers/{index}/");
    let mut all = Vec::new();
    let mut add = |tensor: String, object: &str| -> Result<(), CheckpointError> {
        all.push(linear(checkpoint, tensor, format!("{o}{object}"))?);
        Ok(())
    };
    if full {
        for role in ["q_proj", "k_proj", "v_proj"] {
            add(format!("{p}self_attn.{role}"), "attention/query_key")?;
        }
        add(format!("{p}self_attn.o_proj"), "attention/output")?;
    } else {
        add(format!("{p}linear_attn.in_proj_qkv"), "gdn/query_key")?;
        add(format!("{p}linear_attn.in_proj_z"), "gdn/query_key")?;
        add(format!("{p}linear_attn.out_proj"), "gdn/output")?;
        add(format!("{p}linear_attn.in_proj_a"), "gdn/query_key")?;
        add(format!("{p}linear_attn.in_proj_b"), "gdn/query_key")?;
    }
    add(format!("{p}mlp.gate_proj"), "mlp/gate_up")?;
    add(format!("{p}mlp.up_proj"), "mlp/gate_up")?;
    add(format!("{p}mlp.down_proj"), "mlp/down")?;
    Ok(all)
}

fn linear(checkpoint: &Qwen38Objects, tensor: String, hessian_object: String) -> Result<Linear, CheckpointError> {
    let found = checkpoint.store().tensor(&format!("{tensor}.weight"))?;
    let [rows, columns] = found.shape[..] else {
        return Err(CheckpointError::Invalid(format!("{tensor} is not a matrix")));
    };
    Ok(Linear {
        tensor,
        rows,
        columns,
        hessian_object,
        quantized: rows.is_multiple_of(128) && columns.is_multiple_of(128),
    })
}

/// The rows of `tensor` (a name from [`layer`]) inside the layer's objects, in object order. Empty for the GDN a/b
/// projections, which no object holds.
pub fn placements(checkpoint: &Qwen38Objects, index: usize, tensor: &str) -> Result<Vec<Placement>, CheckpointError> {
    let weight = format!("{tensor}.weight");
    let mut found = Vec::new();
    for name in checkpoint.layer_objects(index)? {
        let mut object_row = 0;
        for piece in checkpoint.object(&name)?.pieces {
            if piece.tensor == weight {
                found.push(Placement {
                    object: name.clone(),
                    object_row,
                    linear_row: piece.first_row,
                    rows: piece.rows,
                });
            }
            object_row += piece.rows;
        }
    }
    Ok(found)
}

/// The budgeted linears of a Qwen 3.8 text model with `layers` layers, in exllamav3's module-tree order, with their
/// allocation inputs: `full_attention(i)` says whether layer `i` is full attention, and `shape(tensor)` gives the
/// `[n, k]` of a linear by its name without `.weight`.
///
/// - The attention module has priority 2 and promotes q, k and v together (`.qkv`) and o alone (`.o`).
/// - The GDN module has priority 1 and promotes in_proj_qkv and in_proj_z together (`.qkvz`) and out_proj alone
///   (`.o`).
/// - The gated MLP has priority 0 and promotes up and gate together (`.gu`) and down alone (`.d`).
///
/// The GDN a/b projections are not quantized and are not listed.
pub fn budget<E>(
    layers: usize,
    full_attention: impl Fn(usize) -> bool,
    mut shape: impl FnMut(&str) -> Result<[u64; 2], E>,
) -> Result<Vec<Budgeted>, E> {
    let mut all = Vec::new();
    for layer in 0..layers {
        let p = format!("{TEXT}{layer}.");
        let full = full_attention(layer);
        let attention = format!("{p}{}", if full { "self_attn" } else { "linear_attn" });
        let priority = if full { 2 } else { 1 };
        let projections: &[(&str, &str)] = if full {
            &[
                ("q_proj", ".qkv"),
                ("k_proj", ".qkv"),
                ("v_proj", ".qkv"),
                ("o_proj", ".o"),
            ]
        } else {
            &[("in_proj_qkv", ".qkvz"), ("in_proj_z", ".qkvz"), ("out_proj", ".o")]
        };
        let mut push = |name: String, priority: i32, group: String| -> Result<(), E> {
            let [n, k] = shape(&name)?;
            all.push(Budgeted {
                name,
                layer,
                numel: n * k,
                priority,
                group,
            });
            Ok(())
        };
        for (role, group) in projections {
            push(format!("{attention}.{role}"), priority, format!("{attention}{group}"))?;
        }
        let mlp = format!("{p}mlp");
        push(format!("{mlp}.up_proj"), 0, format!("{mlp}.gu"))?;
        push(format!("{mlp}.gate_proj"), 0, format!("{mlp}.gu"))?;
        push(format!("{mlp}.down_proj"), 0, format!("{mlp}.d"))?;
    }
    Ok(all)
}

/// [`budget`] for a model of `layers` layers whose every fourth layer, from the fourth, is full attention, with the
/// shapes of [`SWIFT_SHAPES`].
pub fn qwen38(layers: usize) -> Result<Vec<Budgeted>, CheckpointError> {
    budget(
        layers,
        |layer| layer % 4 == 3,
        |tensor| {
            let role = tensor.rsplit('.').next().unwrap_or(tensor);
            SWIFT_SHAPES
                .iter()
                .find_map(|(name, shape)| (*name == role).then_some(*shape))
                .ok_or_else(|| CheckpointError::Invalid(format!("no built-in shape for {tensor}")))
        },
    )
}

/// [`budget`] for the checkpoint's own text model: the layer count and which layers are full attention come from its
/// config, the shapes from its tensor headers.
pub fn qwen38_of(checkpoint: &Qwen38Objects) -> Result<Vec<Budgeted>, CheckpointError> {
    let full = &checkpoint.config.full_attention;
    budget(
        checkpoint.config.layers,
        |layer| full.get(layer).copied().unwrap_or(false),
        |tensor| {
            let found = checkpoint.store().tensor(&format!("{tensor}.weight"))?;
            match found.shape[..] {
                [n, k] => Ok([n, k]),
                _ => Err(CheckpointError::Invalid(format!("{tensor} is not a matrix"))),
            }
        },
    )
}

#[cfg(test)]
mod tests;
