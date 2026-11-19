//! What the EXL3 writer tests share: driving a writer call to its callback, listing a directory, opening a source.

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: it lists its own fixture directories and stops on a failed unwrap"
)]

use std::collections::HashMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use euhedral_inference_engine::Handoff;
use euhedral_inference_quantization::checkpoint::CheckpointError;
use euhedral_inference_quantization::checkpoint::objects::Qwen38Objects;
use euhedral_inference_quantization::checkpoint::safetensors::Safetensors;
use euhedral_inference_quantization::exl3::Quantized;
use euhedral_inference_quantization::exl3::{Done, Exl3Settings, Exl3Writer, Listing};

use crate::checkpoint_support::{Spec, World, bf16, safetensors, write};

/// Calls `start` with a callback and waits, outside the lattice, for the writer and the result.
pub fn run(start: impl FnOnce(Done)) -> (Exl3Writer, Result<(), CheckpointError>) {
    let handoff = Handoff::new();
    let sender = Arc::clone(&handoff);
    start(Arc::new(move |writer, result| sender.send((writer, result))));
    handoff.wait(Duration::from_secs(60)).expect("the callback never ran")
}

pub fn names(directory: &Path) -> Vec<String> {
    let mut found: Vec<String> = fs::read_dir(directory)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect();
    found.sort();
    found
}

pub fn listing(source: &Path, out: &Path) -> Listing {
    Listing {
        source: names(source),
        out: names(out),
    }
}

pub fn open(world: &World, source: &Path, out: &Path, bits: f64, rows: u64, columns: u64) -> Exl3Writer {
    let store = Arc::new(world.call(|done| Safetensors::open(&world.io, source, done)).unwrap());
    let objects = world.call(|done| Qwen38Objects::open(store, source, done)).unwrap();
    let mut settings = Exl3Settings::new(bits, rows, columns);
    settings.buffer_bytes = 4096;
    Exl3Writer::new(
        Arc::clone(&world.io),
        Arc::clone(&world.host),
        objects,
        source,
        out,
        settings,
    )
}

const LAYER: &str = "model.language_model.layers.0.";

pub const GDN: [&str; 6] = [
    "linear_attn.in_proj_qkv",
    "linear_attn.in_proj_z",
    "linear_attn.out_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
];

pub const MTP: [&str; 8] = [
    "mtp.layers.0.self_attn.q_proj",
    "mtp.layers.0.self_attn.k_proj",
    "mtp.layers.0.self_attn.v_proj",
    "mtp.layers.0.self_attn.o_proj",
    "mtp.layers.0.mlp.gate_proj",
    "mtp.layers.0.mlp.up_proj",
    "mtp.layers.0.mlp.down_proj",
    "mtp.fc",
];

pub const CONFIG: &str = r#"{"model_type":"qwen3_5","tie_word_embeddings":true,"text_config":{"num_hidden_layers":1,
 "layer_types":["linear_attention"],"hidden_size":128,"intermediate_size":128,"num_attention_heads":2,
 "num_key_value_heads":1,"head_dim":8,"partial_rotary_factor":0.25,"linear_num_key_heads":1,
 "linear_num_value_heads":8,"linear_key_head_dim":16,"linear_value_head_dim":16,
 "linear_conv_kernel_dim":4,"rms_norm_eps":1e-06,"rope_parameters":{"rope_theta":10000000},
 "vocab_size":128}}"#;

/// A miniature source checkpoint: one linear-attention layer, one MTP layer, a vision tensor, every weight BF16 1.0
/// except `lm_head.weight[5]`, which is `lm_head_value`.
pub fn miniature(directory: &Path, config: &str, lm_head_value: f32) {
    fs::create_dir_all(directory).unwrap();
    write(&directory.join("config.json"), config.as_bytes());
    write(&directory.join("tokenizer.json"), b"{}");
    write(&directory.join("LICENSE"), b"x");
    write(&directory.join("old.bin"), b"x");
    write(&directory.join("empty.txt"), b"");
    let mut shapes: Vec<(String, Vec<u64>)> = Vec::new();
    for name in GDN.iter().take(3) {
        shapes.push((format!("{LAYER}{name}.weight"), vec![128, 128]));
    }
    for name in &GDN[3..] {
        shapes.push((format!("{LAYER}{name}.weight"), vec![128, 128]));
    }
    shapes.push((format!("{LAYER}linear_attn.in_proj_a.weight"), vec![8, 128]));
    shapes.push((format!("{LAYER}linear_attn.in_proj_b.weight"), vec![8, 128]));
    shapes.push((format!("{LAYER}linear_attn.A_log"), vec![8]));
    shapes.push((format!("{LAYER}linear_attn.dt_bias"), vec![8]));
    shapes.push((format!("{LAYER}linear_attn.conv1d.weight"), vec![128, 1, 4]));
    shapes.push((format!("{LAYER}linear_attn.norm.weight"), vec![16]));
    shapes.push((format!("{LAYER}input_layernorm.weight"), vec![128]));
    shapes.push((format!("{LAYER}post_attention_layernorm.weight"), vec![128]));
    shapes.push(("model.language_model.embed_tokens.weight".to_owned(), vec![128, 128]));
    shapes.push(("model.language_model.norm.weight".to_owned(), vec![128]));
    shapes.push(("lm_head.weight".to_owned(), vec![128, 128]));
    shapes.push(("model.visual.blocks.0.attn.qkv.weight".to_owned(), vec![48, 16]));
    shapes.push(("model.visual.blocks.0.attn.qkv.bias".to_owned(), vec![48]));
    shapes.push(("mtp.norm.weight".to_owned(), vec![128]));
    shapes.push(("mtp.layers.0.input_layernorm.weight".to_owned(), vec![128]));
    shapes.push(("mtp.fc.weight".to_owned(), vec![128, 256]));
    for name in &MTP[..7] {
        shapes.push((format!("{name}.weight"), vec![128, 128]));
    }
    let specs: Vec<Spec> = shapes
        .into_iter()
        .map(|(name, shape)| {
            let count: u64 = shape.iter().product();
            let mut values = vec![1.0f32; count as usize];
            if name == "lm_head.weight" {
                values[5] = lm_head_value;
            }
            Spec {
                name: Box::leak(name.into_boxed_str()),
                dtype: "BF16",
                shape,
                bytes: bf16(&values),
            }
        })
        .collect();
    write(&directory.join("model.safetensors"), &safetensors(&specs));
}

/// Quantizer output of the miniature's layer 0 at `bits`, and the rates by linear. The first linear's scales are
/// F16 1.0 (`suh`) and F16 2.0 (`svh`).
pub fn layer_results(bits: f64) -> (HashMap<String, Quantized>, HashMap<String, f64>) {
    let mut results = HashMap::new();
    let mut rates = HashMap::new();
    for (position, name) in GDN.iter().enumerate() {
        let name = format!("{LAYER}{name}");
        let words = 8 * 8 * (bits * 16.0) as usize;
        let (suh, svh) = if position == 0 { (0x3C00, 0x4000) } else { (0, 0) };
        results.insert(
            name.clone(),
            Quantized {
                trellis: vec![0; words],
                suh: vec![suh; 128],
                svh: vec![svh; 128],
            },
        );
        rates.insert(name, bits);
    }
    (results, rates)
}

pub fn mtp_results() -> HashMap<String, Quantized> {
    MTP.iter()
        .map(|name| {
            let k = if *name == "mtp.fc" { 256 } else { 128 };
            let words = (k / 16) * 8 * 64;
            (
                (*name).to_owned(),
                Quantized {
                    trellis: vec![0; words],
                    suh: vec![0; k],
                    svh: vec![0; 128],
                },
            )
        })
        .collect()
}

/// The dtype, shape and bytes of tensor `name` of the safetensors file `path`.
pub fn tensor_of(path: &Path, name: &str) -> (String, Vec<u64>, Vec<u8>) {
    let file = fs::read(path).unwrap();
    let length = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&file[8..8 + length]).unwrap();
    let node = &header[name];
    let [start, end] = [0, 1].map(|i| node["data_offsets"][i].as_u64().unwrap() as usize);
    let shape = node["shape"]
        .as_array()
        .unwrap()
        .iter()
        .map(|d| d.as_u64().unwrap())
        .collect();
    (
        node["dtype"].as_str().unwrap().to_owned(),
        shape,
        file[8 + length + start..8 + length + end].to_vec(),
    )
}

/// The tensor names of the safetensors file `path`, in file order.
pub fn tensor_names(path: &Path) -> Vec<String> {
    let file = fs::read(path).unwrap();
    let length = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    let header: serde_json::Value = serde_json::from_slice(&file[8..8 + length]).unwrap();
    let mut found: Vec<(u64, String)> = header
        .as_object()
        .unwrap()
        .iter()
        .filter(|(name, _)| *name != "__metadata__")
        .map(|(name, node)| (node["data_offsets"][0].as_u64().unwrap(), name.clone()))
        .collect();
    found.sort();
    found.into_iter().map(|(_, name)| name).collect()
}
