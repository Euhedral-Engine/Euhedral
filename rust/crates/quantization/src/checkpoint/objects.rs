//! The quantized objects of a Qwen 3.8 artifact as row ranges of the BF16 checkpoint tensors, fused as the converter
//! fuses them:
//! - full attention: `q_proj` interleaves per head [query 256 rows; gate 256 rows], split into `query_key` =
//!   [query (6144); `k_proj` (1024)] and `gate_value` = [gate (6144); `v_proj` (1024)];
//! - GDN: `query_key` = `in_proj_qkv` rows 0..4095, `value_z` = [`in_proj_qkv` rows 4096..10239; `in_proj_z`];
//! - MLP: `gate_up` = [`gate_proj`; `up_proj`];
//! - MTP: `query_key_gate_value` = [query; k; gate; v], `input_projection` = `mtp.fc`.

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use euhedral_io::IoMemory;
use serde_json::Value;

use super::io::read_file;
use super::safetensors::{ReadRange, Safetensors};
use super::{CheckpointError, Done};

const TEXT: &str = "model.language_model.";

/// The quantized objects of the MTP layer, in the order the drafter reads them.
pub const MTP_OBJECTS: [&str; 5] = [
    "mtp/input_projection",
    "mtp/layer/attention/query_key_gate_value",
    "mtp/layer/attention/output",
    "mtp/layer/mlp/gate_up",
    "mtp/layer/mlp/down",
];

/// Rows `[first_row, first_row + rows)` of a checkpoint tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Piece {
    pub tensor: String,
    pub first_row: u64,
    pub rows: u64,
}

/// An object: its pieces in order, and its shape [rows, columns] of BF16 elements.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectSource {
    pub name: String,
    pub pieces: Vec<Piece>,
    pub rows: u64,
    pub columns: u64,
}

impl ObjectSource {
    /// The size of the object in bytes, or `None` when it does not fit in 64 bits.
    pub fn bytes(&self) -> Option<u64> {
        self.rows.checked_mul(self.columns)?.checked_mul(2)
    }
}

/// The dimensions the calibration needs, from `config.json`'s `text_config`.
#[derive(Clone, Debug, PartialEq)]
pub struct Config {
    pub hidden: u64,
    pub intermediate: u64,
    pub layers: usize,
    pub heads: u64,
    pub kv_heads: u64,
    pub head_dim: u64,
    pub rotary_dim: u64,
    pub key_heads: u64,
    pub value_heads: u64,
    pub key_dim: u64,
    pub value_dim: u64,
    pub conv_kernel: u64,
    pub eps: f64,
    pub theta: f64,
    pub vocabulary: u64,
    pub full_attention: Vec<bool>,
}

impl Config {
    /// Reads the dimensions from the bytes of a `config.json`.
    pub fn parse(json: &[u8]) -> Result<Config, CheckpointError> {
        let root: Value = serde_json::from_slice(json)
            .map_err(|error| CheckpointError::Format(format!("config.json is not JSON: {error}")))?;
        let text = root
            .get("text_config")
            .ok_or_else(|| CheckpointError::Format("config.json has no text_config".to_owned()))?;
        let missing = |key: &str| CheckpointError::Format(format!("text_config has no usable {key}"));
        let number = |key: &str| text.get(key).and_then(Value::as_u64).ok_or_else(|| missing(key));
        let real = |value: Option<&Value>, key: &str| value.and_then(Value::as_f64).ok_or_else(|| missing(key));
        let layers = usize::try_from(number("num_hidden_layers")?).map_err(|_| missing("num_hidden_layers"))?;
        let types = text
            .get("layer_types")
            .and_then(Value::as_array)
            .filter(|types| types.len() >= layers)
            .ok_or_else(|| missing("layer_types"))?;
        let full_attention = types
            .iter()
            .take(layers)
            .map(|kind| kind.as_str() == Some("full_attention"))
            .collect();
        let head_dim = number("head_dim")?;
        let rotary_dim =
            (head_dim as f64 * real(text.get("partial_rotary_factor"), "partial_rotary_factor")?).round() as u64;
        Ok(Config {
            hidden: number("hidden_size")?,
            intermediate: number("intermediate_size")?,
            layers,
            heads: number("num_attention_heads")?,
            kv_heads: number("num_key_value_heads")?,
            head_dim,
            rotary_dim,
            key_heads: number("linear_num_key_heads")?,
            value_heads: number("linear_num_value_heads")?,
            key_dim: number("linear_key_head_dim")?,
            value_dim: number("linear_value_head_dim")?,
            conv_kernel: number("linear_conv_kernel_dim")?,
            eps: real(text.get("rms_norm_eps"), "rms_norm_eps")?,
            theta: real(
                text.get("rope_parameters").and_then(|rope| rope.get("rope_theta")),
                "rope_theta",
            )?,
            vocabulary: number("vocab_size")?,
            full_attention,
        })
    }
}

/// The objects of one checkpoint.
#[derive(Debug)]
pub struct Qwen38Objects {
    store: Arc<Safetensors>,
    pub config: Config,
}

impl Qwen38Objects {
    pub fn new(store: Arc<Safetensors>, config: Config) -> Self {
        Self { store, config }
    }

    /// Reads `config.json` of `checkpoint`; `then` gets the objects of `store`. `Err` started nothing.
    pub fn open(store: Arc<Safetensors>, checkpoint: &Path, then: Done<Qwen38Objects>) -> Result<(), CheckpointError> {
        let io = Arc::clone(store.io());
        read_file(
            &io,
            &checkpoint.join("config.json"),
            Arc::new(move |bytes| {
                then(
                    bytes
                        .and_then(|bytes| Config::parse(&bytes))
                        .map(|config| Qwen38Objects {
                            store: Arc::clone(&store),
                            config,
                        }),
                );
            }),
        )
    }

    pub fn store(&self) -> &Arc<Safetensors> {
        &self.store
    }

    /// The quantized object names of decoder layer `index`, in the order the layer reads them.
    pub fn layer_objects(&self, index: usize) -> Result<Vec<String>, CheckpointError> {
        let full = *self
            .config
            .full_attention
            .get(index)
            .ok_or_else(|| CheckpointError::Invalid(format!("the model has no layer {index}")))?;
        let roles = if full {
            [
                "attention/query_key",
                "attention/gate_value",
                "attention/output",
                "mlp/gate_up",
                "mlp/down",
            ]
        } else {
            ["gdn/query_key", "gdn/value_z", "gdn/output", "mlp/gate_up", "mlp/down"]
        };
        Ok(roles.iter().map(|role| format!("text/layers/{index}/{role}")).collect())
    }

    /// The pieces of object `name`.
    pub fn object(&self, name: &str) -> Result<ObjectSource, CheckpointError> {
        let unknown = || CheckpointError::Invalid(format!("unknown object {name}"));
        if let Some(rest) = name.strip_prefix("text/layers/") {
            let (index, role) = rest.split_once('/').ok_or_else(unknown)?;
            let index: usize = index.parse().map_err(|_| unknown())?;
            return self.layer_object(name, &format!("{TEXT}layers.{index}."), role);
        }
        if let Some(role) = name.strip_prefix("mtp/layer/") {
            return self.layer_object(name, "mtp.layers.0.", role);
        }
        match name {
            "mtp/input_projection" => self.whole(name, "mtp.fc.weight"),
            "text/output_head" => self.whole(name, "lm_head.weight"),
            _ => Err(unknown()),
        }
    }

    fn layer_object(&self, name: &str, prefix: &str, role: &str) -> Result<ObjectSource, CheckpointError> {
        let config = &self.config;
        let weight = |tensor: &str| format!("{prefix}{tensor}.weight");
        match role {
            "attention/query_key" => {
                let mut pieces = self.attention_half(prefix, false);
                pieces.push(self.full(&weight("self_attn.k_proj"))?);
                self.fuse(name, pieces)
            }
            "attention/gate_value" => {
                let mut pieces = self.attention_half(prefix, true);
                pieces.push(self.full(&weight("self_attn.v_proj"))?);
                self.fuse(name, pieces)
            }
            "attention/query_key_gate_value" => {
                let mut pieces = self.attention_half(prefix, false);
                pieces.push(self.full(&weight("self_attn.k_proj"))?);
                pieces.extend(self.attention_half(prefix, true));
                pieces.push(self.full(&weight("self_attn.v_proj"))?);
                self.fuse(name, pieces)
            }
            "attention/output" => self.whole(name, &weight("self_attn.o_proj")),
            "gdn/query_key" => self.fuse(
                name,
                vec![Piece {
                    tensor: weight("linear_attn.in_proj_qkv"),
                    first_row: 0,
                    rows: 2 * config.key_heads * config.key_dim,
                }],
            ),
            "gdn/value_z" => self.fuse(
                name,
                vec![
                    Piece {
                        tensor: weight("linear_attn.in_proj_qkv"),
                        first_row: 2 * config.key_heads * config.key_dim,
                        rows: config.value_heads * config.value_dim,
                    },
                    self.full(&weight("linear_attn.in_proj_z"))?,
                ],
            ),
            "gdn/output" => self.whole(name, &weight("linear_attn.out_proj")),
            "mlp/gate_up" => self.fuse(
                name,
                vec![self.full(&weight("mlp.gate_proj"))?, self.full(&weight("mlp.up_proj"))?],
            ),
            "mlp/down" => self.whole(name, &weight("mlp.down_proj")),
            _ => Err(CheckpointError::Invalid(format!("unknown layer object {name}"))),
        }
    }

    /// The query (or gate) rows of a per-head interleaved `q_proj`: head h owns rows h * 2 * head_dim + [0, head_dim)
    /// for the query and + [head_dim, 2 * head_dim) for the gate.
    fn attention_half(&self, prefix: &str, gate: bool) -> Vec<Piece> {
        let head_dim = self.config.head_dim;
        (0..self.config.heads)
            .map(|head| Piece {
                tensor: format!("{prefix}self_attn.q_proj.weight"),
                first_row: head * 2 * head_dim + if gate { head_dim } else { 0 },
                rows: head_dim,
            })
            .collect()
    }

    fn full(&self, tensor: &str) -> Result<Piece, CheckpointError> {
        let rows = *self
            .store
            .tensor(tensor)?
            .shape
            .first()
            .ok_or_else(|| CheckpointError::Format(format!("{tensor} is a scalar")))?;
        Ok(Piece {
            tensor: tensor.to_owned(),
            first_row: 0,
            rows,
        })
    }

    fn whole(&self, name: &str, tensor: &str) -> Result<ObjectSource, CheckpointError> {
        self.fuse(name, vec![self.full(tensor)?])
    }

    fn fuse(&self, name: &str, pieces: Vec<Piece>) -> Result<ObjectSource, CheckpointError> {
        let mut columns = None;
        let mut rows = 0u64;
        for piece in &pieces {
            let tensor = self.store.tensor(&piece.tensor)?;
            if tensor.dtype != "BF16" || tensor.shape.len() != 2 {
                return Err(CheckpointError::Invalid(format!(
                    "{} is not a BF16 matrix",
                    piece.tensor
                )));
            }
            if columns.is_some_and(|columns| columns != tensor.shape[1]) {
                return Err(CheckpointError::Invalid(format!("{name}: mixed widths")));
            }
            columns = Some(tensor.shape[1]);
            if piece
                .first_row
                .checked_add(piece.rows)
                .is_none_or(|end| end > tensor.shape[0])
            {
                return Err(CheckpointError::Invalid(format!("{name}: rows out of range")));
            }
            rows = rows
                .checked_add(piece.rows)
                .ok_or_else(|| CheckpointError::Invalid(format!("{name}: too many rows")))?;
        }
        let columns = columns.ok_or_else(|| CheckpointError::Invalid(format!("{name}: no pieces")))?;
        let source = ObjectSource {
            name: name.to_owned(),
            pieces,
            rows,
            columns,
        };
        source
            .bytes()
            .ok_or_else(|| CheckpointError::Invalid(format!("{name}: the object is too large")))?;
        Ok(source)
    }

    /// Reads object `source` (BF16, row-major [rows, columns]): `region(range)` supplies the memory for the bytes
    /// `range` of the object, one call per piece in order, and `then` gets the memories in that order. A piece that
    /// follows another in the object follows it in memory only if the caller makes the regions adjacent. `Err` started
    /// nothing.
    pub fn read<B: IoMemory>(
        &self,
        source: &ObjectSource,
        region: &mut dyn FnMut(Range<u64>) -> B,
        then: Done<Vec<B>>,
    ) -> Result<(), CheckpointError> {
        let too_large = || CheckpointError::Invalid(format!("{}: the object is too large", source.name));
        let row_bytes = source.columns.checked_mul(2).ok_or_else(too_large)?;
        source.bytes().ok_or_else(too_large)?;
        let mut at = 0;
        let mut ranges = Vec::with_capacity(source.pieces.len());
        for piece in &source.pieces {
            let bytes = piece.rows.checked_mul(row_bytes).ok_or_else(too_large)?;
            let memory = region(at..at + bytes);
            if memory.len() as u64 != bytes {
                return Err(CheckpointError::Invalid(format!(
                    "{}: a region of {} bytes for {bytes}",
                    source.name,
                    memory.len()
                )));
            }
            ranges.push(ReadRange {
                tensor: piece.tensor.clone(),
                skip: piece.first_row.checked_mul(row_bytes).ok_or_else(too_large)?,
                memory,
            });
            at = at.checked_add(bytes).ok_or_else(too_large)?;
        }
        self.store.read_ranges(ranges, then)
    }

    /// Reads object `source` into one `Vec` of its bytes.
    pub fn read_bytes(&self, source: &ObjectSource, then: Done<Vec<u8>>) -> Result<(), CheckpointError> {
        let mut region = |range: Range<u64>| vec![0u8; (range.end - range.start) as usize];
        self.read(
            source,
            &mut region,
            Arc::new(move |pieces| then(pieces.map(|pieces| pieces.concat()))),
        )
    }
}
