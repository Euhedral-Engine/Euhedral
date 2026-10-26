//! The tensor table of the RCO artifact and the model around it, for the tests that need neither the file nor a GPU.

#![allow(dead_code, reason = "each test file uses some of these helpers")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test helper: a failed unwrap is the failure"
)]

use euhedral_inference_edrl::{HEADER_BYTES, Header, LayerType, ModelMetadata, TensorTable};
use euhedral_inference_engine::model::ModelConfig;
use euhedral_inference_settings::{ModelSettings, Settings};

pub fn fixture() -> &'static str {
    include_str!("../fixtures/rco_table.tsv")
}

pub fn put_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

pub fn put_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// The RCO artifact's table, rebuilt in the container's binary form with synthetic offsets, and parsed.
pub fn rco_table_without(skip: impl Fn(&str) -> bool) -> TensorTable {
    let rows: Vec<Vec<&str>> = fixture()
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .filter(|columns| !skip(columns[0]))
        .collect();
    let mut table = Vec::new();
    let mut entries: Vec<(Vec<u8>, u64)> = Vec::new();
    for columns in &rows {
        let shape: Vec<u64> = columns[1].split('x').map(|d| d.parse().unwrap()).collect();
        let mut entry = Vec::new();
        put_u32(&mut entry, columns[0].len() as u32);
        entry.extend_from_slice(columns[0].as_bytes());
        put_u32(&mut entry, shape.len() as u32);
        for dimension in &shape {
            put_u64(&mut entry, *dimension);
        }
        for ordinal in &columns[2..5] {
            put_u32(&mut entry, ordinal.parse().unwrap());
        }
        entries.push((entry, columns[5].parse().unwrap()));
    }
    let table_bytes: usize = entries.iter().map(|(entry, _)| entry.len() + 16).sum();
    let table_offset = HEADER_BYTES as u64;
    let data_offset = table_offset + table_bytes as u64;
    let mut offset = data_offset.div_ceil(4096) * 4096;
    for (entry, bytes) in &entries {
        table.extend_from_slice(entry);
        put_u64(&mut table, offset);
        put_u64(&mut table, *bytes);
        offset = (offset + bytes).div_ceil(4096) * 4096;
    }
    let mut header = Vec::new();
    put_u32(&mut header, 0x5157_454E);
    put_u32(&mut header, 2);
    put_u64(&mut header, HEADER_BYTES as u64);
    put_u64(&mut header, 0);
    put_u64(&mut header, table_offset);
    put_u32(&mut header, entries.len() as u32);
    put_u32(&mut header, 0);
    put_u64(&mut header, data_offset);
    let header = Header::parse(header.as_slice().try_into().unwrap()).unwrap();
    TensorTable::parse(&table, &header, offset).unwrap()
}

pub fn rco_table() -> TensorTable {
    rco_table_without(|_| false)
}

pub fn metadata() -> ModelMetadata {
    ModelMetadata {
        vocab_size: 248_320,
        hidden_size: 5120,
        num_hidden_layers: 64,
        num_attention_heads: 24,
        num_key_value_heads: 4,
        attention_head_dim: 256,
        intermediate_size: 17_408,
        linear_num_key_heads: 16,
        linear_num_value_heads: 48,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        rms_norm_epsilon: 1e-6,
        rope_theta: 1e7,
        partial_rotary_factor: 0.25,
        max_position_embeddings: 262_144,
        hidden_activation: "silu".to_string(),
        layer_types: (0..64)
            .map(|layer| {
                if layer % 4 == 3 {
                    LayerType::FullAttention
                } else {
                    LayerType::GatedDeltaNet
                }
            })
            .collect(),
        num_experts: 0,
        num_experts_per_token: 0,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        tie_word_embeddings: false,
        attention_output_gate: true,
        mtp_layer_count: 1,
        draft_depth: Some(4),
    }
}

pub fn config() -> ModelConfig {
    ModelConfig::from(&metadata())
}

pub fn settings(max_tokens: u32) -> Settings {
    let mut settings = Settings::with_model(ModelSettings::default());
    settings.context.max_tokens = max_tokens;
    settings
}
