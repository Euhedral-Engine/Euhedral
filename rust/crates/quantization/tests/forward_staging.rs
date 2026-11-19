//! A layer read from a tiny checkpoint into host memory: the objects, and the tensors as the kernels take them.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it writes its fixture files and stops on a failed unwrap"
)]

mod checkpoint_support;

use std::path::Path;
use std::sync::Arc;

use checkpoint_support::{Spec, bf16, drive, safetensors, write};
use euhedral_inference_quantization::checkpoint::CheckpointError;
use euhedral_inference_quantization::checkpoint::objects::Qwen38Objects;
use euhedral_inference_quantization::checkpoint::safetensors::Safetensors;
use euhedral_inference_quantization::forward::{Part, StagedLayer, Target, stage};
use euhedral_io::AsyncIo;

const CONFIG: &str = r#"{"text_config": {
    "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
    "hidden_size": 4, "intermediate_size": 6, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 2, "partial_rotary_factor": 0.25, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
    "linear_key_head_dim": 2, "linear_value_head_dim": 2, "linear_conv_kernel_dim": 4,
    "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10000000.0}, "vocab_size": 16}}"#;

const CHANNELS: usize = 8;
const TAPS: usize = 4;

/// A BF16 tensor whose element `i` is `base + i`.
fn tensor(name: &str, base: f32, shape: &[u64]) -> Spec {
    let count: u64 = shape.iter().product();
    let values: Vec<f32> = (0..count).map(|i| base + i as f32).collect();
    Spec {
        name: Box::leak(name.to_owned().into_boxed_str()),
        dtype: "BF16",
        shape: shape.to_vec(),
        bytes: bf16(&values),
    }
}

fn layer_tensors(prefix: &str, base: f32) -> Vec<Spec> {
    let name = |tail: &str| format!("{prefix}{tail}");
    vec![
        tensor(&name("input_layernorm.weight"), base, &[4]),
        tensor(&name("post_attention_layernorm.weight"), base + 10.0, &[4]),
        tensor(&name("mlp.gate_proj.weight"), base + 20.0, &[6, 4]),
        tensor(&name("mlp.up_proj.weight"), base + 30.0, &[6, 4]),
        tensor(&name("mlp.down_proj.weight"), base + 40.0, &[4, 6]),
    ]
}

fn attention_tensors(prefix: &str, base: f32) -> Vec<Spec> {
    let name = |tail: &str| format!("{prefix}{tail}");
    let mut all = layer_tensors(prefix, base);
    all.extend([
        tensor(&name("self_attn.q_proj.weight"), base + 50.0, &[8, 4]),
        tensor(&name("self_attn.k_proj.weight"), base + 60.0, &[2, 4]),
        tensor(&name("self_attn.v_proj.weight"), base + 70.0, &[2, 4]),
        tensor(&name("self_attn.o_proj.weight"), base + 80.0, &[4, 4]),
        tensor(&name("self_attn.q_norm.weight"), base + 90.0, &[2]),
        tensor(&name("self_attn.k_norm.weight"), base + 100.0, &[2]),
    ]);
    all
}

fn tensors() -> Vec<Spec> {
    let gdn = "model.language_model.layers.0.";
    let mut all = layer_tensors(gdn, 1.0);
    all.extend([
        tensor(&format!("{gdn}linear_attn.in_proj_qkv.weight"), 200.0, &[8, 4]),
        tensor(&format!("{gdn}linear_attn.in_proj_z.weight"), 300.0, &[4, 4]),
        tensor(&format!("{gdn}linear_attn.out_proj.weight"), 400.0, &[4, 4]),
        tensor(&format!("{gdn}linear_attn.in_proj_a.weight"), 500.0, &[2, 4]),
        tensor(&format!("{gdn}linear_attn.in_proj_b.weight"), 510.0, &[2, 4]),
        tensor(&format!("{gdn}linear_attn.norm.weight"), 60.0, &[2]),
        tensor(&format!("{gdn}linear_attn.A_log"), 5.0, &[2]),
        tensor(&format!("{gdn}linear_attn.dt_bias"), 7.0, &[2]),
        tensor(
            &format!("{gdn}linear_attn.conv1d.weight"),
            0.0,
            &[CHANNELS as u64, 1, TAPS as u64],
        ),
    ]);
    all.extend(attention_tensors("model.language_model.layers.1.", 2.0));
    all.extend(attention_tensors("mtp.layers.0.", 3.0));
    all.push(tensor("mtp.fc.weight", 600.0, &[4, 8]));
    all.push(tensor("lm_head.weight", 10.0, &[16, 4]));
    all
}

fn open(directory: &Path) -> (Arc<AsyncIo>, Qwen38Objects) {
    write(&directory.join("model.safetensors"), &safetensors(&tensors()));
    write(&directory.join("config.json"), CONFIG.as_bytes());
    let io = AsyncIo::open().unwrap();
    let store = Arc::new(drive(&io, |done| Safetensors::open(&io, directory, done)).unwrap());
    let objects = drive(&io, |done| Qwen38Objects::open(store, directory, done)).unwrap();
    (io, objects)
}

fn staged(io: &Arc<AsyncIo>, objects: &Qwen38Objects, target: Target) -> Result<StagedLayer, CheckpointError> {
    drive(io, |done| stage(objects, target, done))
}

fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_le_bytes(*word))
        .collect()
}

fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect()
}

#[test]
fn a_gdn_layer_is_staged_with_its_tensors_as_the_kernels_take_them() {
    let directory = tempfile::tempdir().unwrap();
    let (io, objects) = open(directory.path());
    let layer = staged(&io, &objects, Target::Layer(0)).unwrap();
    assert_eq!((layer.prefix(), layer.attention()), ("text/layers/0/", false));

    // The objects are the checkpoint's fused rows, in the order the layer reads them.
    let names: Vec<&str> = layer
        .parts()
        .filter_map(|(part, _)| match part {
            Part::Object(name) => Some(name.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(names, objects.layer_objects(0).unwrap());
    for name in &names {
        let source = objects.object(name).unwrap();
        let expected = drive(&io, |done| objects.read_bytes(&source, done)).unwrap();
        assert_eq!(
            layer.bytes(&Part::Object((*name).to_owned())).unwrap(),
            expected,
            "{name}"
        );
    }

    // A_log and dt_bias are FP32 for the control kernel.
    assert_eq!(floats(layer.bytes(&Part::ALog).unwrap()), [5.0, 6.0]);
    assert_eq!(floats(layer.bytes(&Part::DtBias).unwrap()), [7.0, 8.0]);
    // The convolution is [taps, channels]: element (tap, channel) is the checkpoint's (channel, tap).
    let convolution = bf16_values(layer.bytes(&Part::Convolution).unwrap());
    assert_eq!(convolution.len(), CHANNELS * TAPS);
    for channel in 0..CHANNELS {
        for tap in 0..TAPS {
            assert_eq!(
                convolution[tap * CHANNELS + channel],
                (channel * TAPS + tap) as f32,
                "channel {channel}, tap {tap}"
            );
        }
    }
    // The norms and projections are the tensors themselves.
    assert_eq!(
        bf16_values(layer.bytes(&Part::InputNorm).unwrap()),
        [1.0, 2.0, 3.0, 4.0]
    );
    assert_eq!(
        bf16_values(layer.bytes(&Part::PostNorm).unwrap()),
        [11.0, 12.0, 13.0, 14.0]
    );
    assert_eq!(bf16_values(layer.bytes(&Part::GdnNorm).unwrap()), [60.0, 61.0]);
    assert_eq!(bf16_values(layer.bytes(&Part::AProjection).unwrap())[0], 500.0);
    assert_eq!(bf16_values(layer.bytes(&Part::BProjection).unwrap())[0], 510.0);
    assert!(layer.bytes(&Part::QueryNorm).is_none());
}

#[test]
fn an_attention_layer_and_the_mtp_block_carry_the_query_and_key_norms() {
    let directory = tempfile::tempdir().unwrap();
    let (io, objects) = open(directory.path());
    let layer = staged(&io, &objects, Target::Layer(1)).unwrap();
    assert_eq!((layer.prefix(), layer.attention()), ("text/layers/1/", true));
    assert_eq!(bf16_values(layer.bytes(&Part::QueryNorm).unwrap()), [92.0, 93.0]);
    assert_eq!(bf16_values(layer.bytes(&Part::KeyNorm).unwrap()), [102.0, 103.0]);
    assert!(layer.bytes(&Part::Convolution).is_none() && layer.bytes(&Part::ALog).is_none());

    let mtp = staged(&io, &objects, Target::Mtp).unwrap();
    assert_eq!((mtp.prefix(), mtp.attention()), ("mtp/layer/", true));
    assert_eq!(bf16_values(mtp.bytes(&Part::QueryNorm).unwrap()), [93.0, 94.0]);
    let fused = mtp
        .bytes(&Part::Object("mtp/layer/attention/query_key_gate_value".to_owned()))
        .expect("the MTP block's fused object");
    // Twelve rows of the four hidden columns, two bytes each.
    assert_eq!(fused.len(), 12 * 4 * 2);
}

#[test]
fn a_layer_the_model_lacks_is_refused_before_any_read() {
    let directory = tempfile::tempdir().unwrap();
    let (io, objects) = open(directory.path());
    assert!(matches!(
        staged(&io, &objects, Target::Layer(2)),
        Err(CheckpointError::Invalid(_))
    ));
}
