//! What the EXL3 layout is, as plain values: the tensors of a quantized linear, the ledger of what has been written
//! (which gives the index and `tensor_storage`), the two JSON documents and the pretty printer that writes them, the
//! BF16 to F16 conversion and the choice of auxiliary files. Nothing here touches a file.
//!
//! Portions ported from exllamav3 (MIT License, Copyright (c) 2025 Turboderp).

use std::collections::HashMap;
use std::fmt::Write as _;

use serde_json::{Map, Number, Value, json};

use crate::checkpoint::CheckpointError;

/// The multiplier of the `mul1` codebook, as the signed I32 exllamav3 stores.
pub const MUL1: i32 = 0x83DC_D12D_u32 as i32;

/// Bits of the MTP linears.
pub const MTP_BITS: u32 = 4;

pub(super) const VERSION: &str = "1.6.1";
pub(super) const TEXT: &str = "model.language_model.";

/// Module order inside a decoder layer, as exllamav3 lists the leaf modules of `tensor_storage`.
pub(super) const LAYER_MODULES: [&str; 17] = [
    "input_layernorm",
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.in_proj_b",
    "linear_attn.in_proj_a",
    "linear_attn.out_proj",
    "linear_attn.norm",
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "self_attn.q_norm",
    "self_attn.k_norm",
    "post_attention_layernorm",
    "mlp.up_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
];

/// The parts of a module `tensor_storage` lists, in its order.
const STORED_ORDER: [&str; 5] = ["weight", "suh", "svh", "mul1", "trellis"];

const TORCH_DTYPES: [(&str, &str); 5] = [
    ("I16", "torch.int16"),
    ("F16", "torch.float16"),
    ("BF16", "torch.bfloat16"),
    ("F32", "torch.float32"),
    ("I32", "torch.int32"),
];

/// One tensor of a file written: its data type, shape and size.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Stored {
    pub dtype: String,
    pub shape: Vec<u64>,
    pub bytes: u64,
}

/// What the quantizer produced for one linear.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Quantized {
    /// The trellis words: `k / 16 * (n / 16) * 16 * bits` of 16 bits.
    pub trellis: Vec<i16>,
    /// The input scales, as F16 bits: `k` of them.
    pub suh: Vec<u16>,
    /// The output scales, as F16 bits: `n` of them.
    pub svh: Vec<u16>,
}

/// A tensor to write: name, data type, shape and bytes.
pub(super) struct Planned {
    pub name: String,
    pub dtype: &'static str,
    pub shape: Vec<u64>,
    pub data: Vec<u8>,
}

fn invalid(message: String) -> CheckpointError {
    CheckpointError::Invalid(message)
}

/// The four tensors of linear `linear` (`[n, k]` weights at `bits` bits per weight) from what the quantizer made.
pub(super) fn quantized_tensors(
    linear: &str,
    n: u64,
    k: u64,
    bits: f64,
    result: &Quantized,
) -> Result<Vec<Planned>, CheckpointError> {
    let words = bits * 16.0;
    if words != words.round() || !(16.0..=128.0).contains(&words) {
        return Err(invalid(format!("{linear}: {bits} bits")));
    }
    let words = words as u64;
    let tiles = k / 16 * (n / 16);
    if result.trellis.len() as u64 != tiles * words {
        return Err(invalid(format!(
            "{linear}: trellis has {} words, {bits} bits over {n}x{k} need {}",
            result.trellis.len(),
            tiles * words
        )));
    }
    if result.suh.len() as u64 != k || result.svh.len() as u64 != n {
        return Err(invalid(format!(
            "{linear}: scales {}/{} do not fit {n}x{k}",
            result.suh.len(),
            result.svh.len()
        )));
    }
    Ok(vec![
        Planned {
            name: format!("{linear}.trellis"),
            dtype: "I16",
            shape: vec![k / 16, n / 16, words],
            data: result.trellis.iter().flat_map(|word| word.to_le_bytes()).collect(),
        },
        Planned {
            name: format!("{linear}.suh"),
            dtype: "F16",
            shape: vec![k],
            data: result.suh.iter().flat_map(|half| half.to_le_bytes()).collect(),
        },
        Planned {
            name: format!("{linear}.svh"),
            dtype: "F16",
            shape: vec![n],
            data: result.svh.iter().flat_map(|half| half.to_le_bytes()).collect(),
        },
        Planned {
            name: format!("{linear}.mul1"),
            dtype: "I32",
            shape: Vec::new(),
            data: MUL1.to_le_bytes().to_vec(),
        },
    ])
}

/// The names of the four tensors a quantized linear is stored as.
pub(super) fn quantized_names(linear: &str) -> [String; 4] {
    ["trellis", "suh", "svh", "mul1"].map(|part| format!("{linear}.{part}"))
}

/// The bits of the half-precision float nearest `value`, ties to even. A NaN keeps
/// its sign and as many of its payload bits as fit.
pub fn half_bits(value: f32) -> u16 {
    let raw = value.to_bits();
    let sign = ((raw & 0x8000_0000) >> 16) as u16;
    if value.is_nan() {
        return sign
            | 0x7C00
            | ((raw & 0x007F_E000) >> 13) as u16
            | ((raw & 0x0000_1FF0) >> 4) as u16
            | (raw & 0xF) as u16;
    }
    let magnitude = value.abs();
    // The overflow threshold is the largest half plus half an ulp.
    if magnitude >= 65520.0 {
        return sign | 0x7C00;
    }
    // Half of the smallest subnormal half, and anything smaller, rounds to zero.
    if magnitude <= 0.5 * 2f32.powi(-24) {
        return sign;
    }
    let mut exponent = (raw >> 23 & 0xFF) as i32 - 127;
    let mut delta = 0;
    let mut hidden = 0;
    if exponent < -14 {
        delta = -14 - exponent;
        exponent = -15;
        hidden = 0x0080_0000;
    }
    let significand = raw & 0x007F_FFFF | hidden;
    let mut bits = (significand >> (13 + delta)) as u16;
    let lsb = significand & (1 << (13 + delta));
    let round = significand & (1 << (12 + delta));
    let sticky = significand & ((1 << (12 + delta)) - 1);
    if round != 0 && (lsb | sticky) != 0 {
        bits += 1;
    }
    // A carry out of the significand adds to the exponent.
    sign | (((exponent + 15) << 10) as u16).wrapping_add(bits)
}

/// Converts the BF16 elements in `bytes` (little-endian) to F16 in place; a finite value beyond F16 is an error.
pub(super) fn bf16_to_half(name: &str, bytes: &mut [u8]) -> Result<(), CheckpointError> {
    for pair in bytes.as_chunks_mut::<2>().0 {
        let value = f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16);
        let half = half_bits(value);
        if value.is_finite() && half & 0x7C00 == 0x7C00 {
            return Err(invalid(format!("{name}: {value:e} overflows F16")));
        }
        *pair = half.to_le_bytes();
    }
    Ok(())
}

/// What has been written: which file holds each tensor, how much data each file holds, and the tensors of the modules
/// `tensor_storage` lists.
#[derive(Debug, Default)]
pub(super) struct Ledger {
    /// Tensor and file, in the order the files were written and, within a file, by name.
    weight_map: Vec<(String, String)>,
    modules: HashMap<String, HashMap<String, Stored>>,
    file_bytes: HashMap<String, u64>,
}

impl Ledger {
    /// Notes that `file` now holds `tensors` (sorted by name) and nothing else: an earlier version of the file is
    /// replaced.
    pub fn record_file(&mut self, file: &str, tensors: &[(String, Stored)]) {
        self.weight_map.retain(|(_, held)| held != file);
        let mut bytes = 0;
        for (name, stored) in tensors {
            self.weight_map.push((name.clone(), file.to_owned()));
            bytes += stored.bytes;
            self.record(name, stored);
        }
        self.file_bytes.insert(file.to_owned(), bytes);
    }

    pub fn has_file(&self, file: &str) -> bool {
        self.file_bytes.contains_key(file)
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.weight_map.iter().any(|(held, _)| held == name)
    }

    pub fn files(&self) -> impl Iterator<Item = &str> {
        self.file_bytes.keys().map(String::as_str)
    }

    /// The trellis words per 16x16 tile that module `module` was stored with, if it is a quantized linear.
    pub fn trellis_words(&self, module: &str) -> Option<u64> {
        self.modules.get(module)?.get("trellis")?.shape.get(2).copied()
    }

    /// Notes a written tensor under its `tensor_storage` module, if it belongs to one.
    fn record(&mut self, tensor: &str, stored: &Stored) {
        let Some((module, part)) = tensor.rsplit_once('.') else {
            return;
        };
        if !STORED_ORDER.contains(&part) {
            return;
        }
        let mut listed =
            module == format!("{TEXT}embed_tokens") || module == format!("{TEXT}norm") || module == "lm_head";
        if let Some(rest) = module.strip_prefix(&format!("{TEXT}layers.")) {
            listed = LAYER_MODULES.contains(&rest.split_once('.').map_or(rest, |(_, leaf)| leaf));
        }
        if listed {
            self.modules
                .entry(module.to_owned())
                .or_default()
                .insert(part.to_owned(), stored.clone());
        }
    }

    /// The `model.safetensors.index.json` document.
    pub fn index(&self) -> Value {
        let total: u64 = self.file_bytes.values().sum();
        let mut map = Map::new();
        for (tensor, file) in &self.weight_map {
            map.insert(tensor.clone(), Value::String(file.clone()));
        }
        json!({"metadata": {"total_size": total}, "weight_map": map})
    }

    /// `tensor_storage` for the text modules written so far, in exllamav3's module order.
    pub fn tensor_storage(&self, layers: usize) -> Value {
        let mut keys = vec![format!("{TEXT}embed_tokens")];
        for layer in 0..layers {
            for module in LAYER_MODULES {
                keys.push(format!("{TEXT}layers.{layer}.{module}"));
            }
        }
        keys.push(format!("{TEXT}norm"));
        keys.push("lm_head".to_owned());
        let mut storage = Map::new();
        for key in keys {
            let Some(parts) = self.modules.get(&key) else {
                continue;
            };
            let mut stored = Map::new();
            for part in STORED_ORDER {
                let Some(found) = parts.get(part) else {
                    continue;
                };
                let dtype = TORCH_DTYPES
                    .iter()
                    .find_map(|(name, torch)| (*name == found.dtype).then_some(*torch));
                stored.insert(
                    format!("{key}.{part}"),
                    json!({"shape": found.shape, "n_bytes": found.bytes, "dtype": dtype}),
                );
            }
            let mut module = Map::new();
            module.insert("stored_tensors".to_owned(), Value::Object(stored));
            if let Some(words) = self.trellis_words(&key) {
                let bits = words as f64 / 16.0;
                module.insert("quant_format".to_owned(), json!("exl3"));
                let bits = if bits == bits.round() {
                    Value::from(words / 16)
                } else {
                    Number::from_f64(bits).map_or(Value::Null, Value::Number)
                };
                module.insert("bits_per_weight".to_owned(), bits);
                module.insert("mul1_multiplier".to_owned(), json!(MUL1 as u32));
            }
            storage.insert(key, Value::Object(module));
        }
        Value::Object(storage)
    }
}

/// The `quantization_config` block of `config.json`.
pub(super) fn quantization_config(bits: f64, rows: u64, columns: u64, mtp_written: bool) -> Value {
    let mut block = Map::new();
    block.insert("quant_method".to_owned(), json!("exl3"));
    block.insert("version".to_owned(), json!(VERSION));
    block.insert(
        "bits".to_owned(),
        Number::from_f64(bits).map_or(Value::Null, Value::Number),
    );
    block.insert("head_bits".to_owned(), json!(16));
    block.insert("calibration".to_owned(), json!({"rows": rows, "cols": columns}));
    block.insert("out_scales".to_owned(), json!("always"));
    block.insert("codebook".to_owned(), json!("mul1"));
    if mtp_written {
        block.insert("mtp_bits".to_owned(), json!(MTP_BITS));
    }
    Value::Object(block)
}

/// The files exllamav3 copies next to the weights, from the names in the source directory: every one except
/// `.safetensors` files, `config.json`, the weight index and pickled checkpoints (`.bin`, `.ckpt`, `.pth`, `.pt`).
/// Sorted.
pub fn auxiliary_files<'a>(names: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut chosen: Vec<String> = names
        .into_iter()
        .filter(|name| {
            !name.ends_with(".safetensors")
                && *name != "config.json"
                && *name != "model.safetensors.index.json"
                && ![".bin", ".ckpt", ".pth", ".pt"]
                    .iter()
                    .any(|suffix| name.ends_with(suffix))
        })
        .map(str::to_owned)
        .collect();
    chosen.sort();
    chosen
}

/// The file of decoder layer `index`.
pub fn layer_file(index: usize) -> String {
    format!("{TEXT}layers.{index}.safetensors")
}

/// The file of the MTP layer.
pub const MTP_FILE: &str = "mtp.safetensors";

/// The files [`Exl3Writer::finish`](super::Exl3Writer::finish) writes after the layers and the MTP layer: the embedding,
/// `lm_head`, the vision tower when the source has one, the auxiliary files of `source` (the names of its directory)
/// and the three documents. In the order they are written.
pub fn closing_files<'a>(vision: bool, source: impl IntoIterator<Item = &'a str>) -> Vec<String> {
    let mut files = vec![
        format!("{TEXT}embed_tokens.safetensors"),
        "lm_head.safetensors".to_owned(),
    ];
    if vision {
        files.push("model.visual.safetensors".to_owned());
    }
    files.extend(auxiliary_files(source));
    files.extend(DOCUMENTS.map(str::to_owned));
    files
}

/// The documents `finish` writes last.
const DOCUMENTS: [&str; 3] = [
    "model.safetensors.index.json",
    "config.json",
    "quantization_config.json",
];

/// `value` as the layout's JSON files write a float: the shortest digits that read back as `value`, plain from `1e-3` up to
/// `1e7` and as `d.dddE<exponent>` outside, always with a digit after the point.
pub fn double_text(value: f64) -> String {
    if value == 0.0 {
        return if value.is_sign_negative() { "-0.0" } else { "0.0" }.to_owned();
    }
    // `{:e}` gives the shortest digits as `d.ddde<exp>`.
    let scientific = format!("{:e}", value.abs());
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(|c| *c != '.').collect();
    let sign = if value < 0.0 { "-" } else { "" };
    let mut text = String::from(sign);
    if (-3..7).contains(&exponent) {
        if exponent >= 0 {
            let whole = exponent as usize + 1;
            let padded = format!("{digits:0<whole$}");
            let _ = write!(
                text,
                "{}.{}",
                &padded[..whole],
                if padded.len() > whole { &padded[whole..] } else { "0" }
            );
        } else {
            let _ = write!(text, "0.{}{digits}", "0".repeat((-exponent - 1) as usize));
        }
    } else {
        let (head, tail) = digits.split_at(1);
        let _ = write!(text, "{head}.{}E{exponent}", if tail.is_empty() { "0" } else { tail });
    }
    text
}

/// `value` in the shape the layout's JSON files have: four-space indentation, a space after each colon, every array
/// element and object member on its own line, `[ ]` and `{ }` when empty, no newline at the end.
pub fn pretty(value: &Value) -> String {
    let mut out = String::new();
    write_value(&mut out, value, 0);
    out
}

fn write_value(out: &mut String, value: &Value, depth: usize) {
    match value {
        Value::Null => out.push_str("null"),
        Value::Bool(flag) => out.push_str(if *flag { "true" } else { "false" }),
        Value::Number(number) => match (number.is_f64(), number.as_f64()) {
            (true, Some(float)) => out.push_str(&double_text(float)),
            _ => {
                let _ = write!(out, "{number}");
            }
        },
        Value::String(text) => write_string(out, text),
        Value::Array(items) if items.is_empty() => out.push_str("[ ]"),
        Value::Array(items) => {
            out.push('[');
            for (index, item) in items.iter().enumerate() {
                out.push_str(if index == 0 { "\n" } else { ",\n" });
                indent(out, depth + 1);
                write_value(out, item, depth + 1);
            }
            out.push('\n');
            indent(out, depth);
            out.push(']');
        }
        Value::Object(members) if members.is_empty() => out.push_str("{ }"),
        Value::Object(members) => {
            out.push('{');
            for (index, (key, item)) in members.iter().enumerate() {
                out.push_str(if index == 0 { "\n" } else { ",\n" });
                indent(out, depth + 1);
                write_string(out, key);
                out.push_str(": ");
                write_value(out, item, depth + 1);
            }
            out.push('\n');
            indent(out, depth);
            out.push('}');
        }
    }
}

fn indent(out: &mut String, depth: usize) {
    for _ in 0..depth {
        out.push_str("    ");
    }
}

fn write_string(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if c < ' ' => {
                let _ = write!(out, "\\u{:04X}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

#[cfg(test)]
mod tests;
