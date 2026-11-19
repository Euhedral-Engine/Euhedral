//! The fused objects of a tiny checkpoint: which rows of which tensors make each object, and what is refused.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it writes its fixture files and stops on a failed unwrap"
)]

mod checkpoint_support;

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use euhedral_inference_quantization::checkpoint::CheckpointError;
use euhedral_inference_quantization::checkpoint::objects::{Config, MTP_OBJECTS, ObjectSource, Piece, Qwen38Objects};
use euhedral_inference_quantization::checkpoint::safetensors::Safetensors;
use euhedral_io::AsyncIo;
use checkpoint_support::{Spec, bf16, drive, safetensors, write};

const CONFIG: &str = r#"{"text_config": {
    "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
    "hidden_size": 4, "intermediate_size": 6, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 2, "partial_rotary_factor": 0.25, "linear_num_key_heads": 1, "linear_num_value_heads": 2,
    "linear_key_head_dim": 2, "linear_value_head_dim": 2, "linear_conv_kernel_dim": 4,
    "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10000000.0}, "vocab_size": 16}}"#;

/// A BF16 matrix whose row `r` holds the value `base + r` throughout.
fn matrix(name: &'static str, base: f32, rows: u64, columns: u64) -> Spec {
    let values: Vec<f32> = (0..rows)
        .flat_map(|row| vec![base + row as f32; columns as usize])
        .collect();
    Spec {
        name,
        dtype: "BF16",
        shape: vec![rows, columns],
        bytes: bf16(&values),
    }
}

fn attention(prefix: &str) -> Vec<Spec> {
    let name = |tensor: &str| -> &'static str { Box::leak(format!("{prefix}{tensor}.weight").into_boxed_str()) };
    vec![
        matrix(name("self_attn.q_proj"), 0.0, 8, 4),
        matrix(name("self_attn.k_proj"), 10.0, 2, 4),
        matrix(name("self_attn.v_proj"), 20.0, 2, 4),
        matrix(name("self_attn.o_proj"), 30.0, 4, 4),
        matrix(name("mlp.gate_proj"), 40.0, 6, 4),
        matrix(name("mlp.up_proj"), 50.0, 6, 4),
        matrix(name("mlp.down_proj"), 60.0, 4, 6),
    ]
}

fn tensors() -> Vec<Spec> {
    let mut all = vec![
        matrix(
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
            70.0,
            8,
            4,
        ),
        matrix("model.language_model.layers.0.linear_attn.in_proj_z.weight", 80.0, 4, 4),
        matrix("model.language_model.layers.0.linear_attn.out_proj.weight", 90.0, 4, 4),
        matrix("model.language_model.layers.0.mlp.gate_proj.weight", 40.0, 6, 4),
        matrix("model.language_model.layers.0.mlp.up_proj.weight", 50.0, 6, 4),
        matrix("model.language_model.layers.0.mlp.down_proj.weight", 60.0, 4, 6),
        matrix("mtp.fc.weight", 100.0, 4, 8),
        matrix("lm_head.weight", 10.0, 16, 4),
    ];
    all.extend(attention("model.language_model.layers.1."));
    all.extend(attention("mtp.layers.0."));
    all
}

fn open(directory: &Path, specs: &[Spec]) -> (Arc<AsyncIo>, Qwen38Objects) {
    write(&directory.join("model.safetensors"), &safetensors(specs));
    write(&directory.join("config.json"), CONFIG.as_bytes());
    let io = AsyncIo::open().unwrap();
    let store = Arc::new(drive(&io, |done| Safetensors::open(&io, directory, done)).unwrap());
    let objects = drive(&io, |done| Qwen38Objects::open(store, directory, done)).unwrap();
    (io, objects)
}

/// The first element of each row of an object's bytes.
fn row_values(objects: &Qwen38Objects, io: &Arc<AsyncIo>, name: &str) -> Vec<f32> {
    let source = objects.object(name).unwrap();
    let bytes = drive(io, |done| objects.read_bytes(&source, done)).unwrap();
    assert_eq!(Some(bytes.len() as u64), source.bytes(), "{name}");
    bytes
        .chunks_exact(source.columns as usize * 2)
        .map(|row| f32::from_bits(u32::from(u16::from_le_bytes([row[0], row[1]])) << 16))
        .collect()
}

#[test]
fn the_dimensions_come_from_the_text_config() {
    let config = Config::parse(CONFIG.as_bytes()).unwrap();
    assert_eq!((config.hidden, config.intermediate, config.layers), (4, 6, 2));
    assert_eq!(
        (config.heads, config.kv_heads, config.head_dim, config.rotary_dim),
        (2, 1, 2, 1)
    );
    assert_eq!(
        (config.key_heads, config.value_heads, config.key_dim, config.value_dim),
        (1, 2, 2, 2)
    );
    assert_eq!((config.conv_kernel, config.vocabulary), (4, 16));
    assert_eq!(config.theta, 10_000_000.0);
    assert_eq!(config.full_attention, [false, true]);
    assert!(matches!(Config::parse(b"{}"), Err(CheckpointError::Format(_))));
    assert!(matches!(Config::parse(b"nope"), Err(CheckpointError::Format(_))));
    let short = CONFIG.replace("\"linear_attention\", ", "");
    assert!(matches!(Config::parse(short.as_bytes()), Err(CheckpointError::Format(m)) if m.contains("layer_types")));
}

#[test]
fn layers_list_the_objects_their_kind_reads() {
    let directory = tempfile::tempdir().unwrap();
    let (_io, objects) = open(directory.path(), &tensors());
    assert_eq!(
        objects.layer_objects(0).unwrap(),
        ["gdn/query_key", "gdn/value_z", "gdn/output", "mlp/gate_up", "mlp/down"]
            .map(|role| format!("text/layers/0/{role}"))
    );
    assert_eq!(
        objects.layer_objects(1).unwrap(),
        [
            "attention/query_key",
            "attention/gate_value",
            "attention/output",
            "mlp/gate_up",
            "mlp/down"
        ]
        .map(|role| format!("text/layers/1/{role}"))
    );
    assert!(matches!(objects.layer_objects(2), Err(CheckpointError::Invalid(_))));
}

#[test]
fn an_interleaved_q_proj_splits_into_query_and_gate_rows() {
    let directory = tempfile::tempdir().unwrap();
    let (io, objects) = open(directory.path(), &tensors());
    // Head h owns q_proj rows 4h..4h+2 for the query and 4h+2..4h+4 for the gate; k and v follow.
    assert_eq!(
        row_values(&objects, &io, "text/layers/1/attention/query_key"),
        [0.0, 1.0, 4.0, 5.0, 10.0, 11.0]
    );
    assert_eq!(
        row_values(&objects, &io, "text/layers/1/attention/gate_value"),
        [2.0, 3.0, 6.0, 7.0, 20.0, 21.0]
    );
    assert_eq!(
        row_values(&objects, &io, "mtp/layer/attention/query_key_gate_value"),
        [0.0, 1.0, 4.0, 5.0, 10.0, 11.0, 2.0, 3.0, 6.0, 7.0, 20.0, 21.0]
    );
    assert_eq!(
        row_values(&objects, &io, "text/layers/1/attention/output"),
        [30.0, 31.0, 32.0, 33.0]
    );
}

#[test]
fn gdn_and_mlp_objects_take_the_rows_the_layer_reads() {
    let directory = tempfile::tempdir().unwrap();
    let (io, objects) = open(directory.path(), &tensors());
    assert_eq!(
        row_values(&objects, &io, "text/layers/0/gdn/query_key"),
        [70.0, 71.0, 72.0, 73.0]
    );
    assert_eq!(
        row_values(&objects, &io, "text/layers/0/gdn/value_z"),
        [74.0, 75.0, 76.0, 77.0, 80.0, 81.0, 82.0, 83.0]
    );
    assert_eq!(
        row_values(&objects, &io, "text/layers/0/gdn/output"),
        [90.0, 91.0, 92.0, 93.0]
    );
    let gate_up: Vec<f32> = (0..6)
        .map(|r| 40.0 + r as f32)
        .chain((0..6).map(|r| 50.0 + r as f32))
        .collect();
    assert_eq!(row_values(&objects, &io, "text/layers/0/mlp/gate_up"), gate_up);
    assert_eq!(
        row_values(&objects, &io, "mtp/layer/mlp/down"),
        [60.0, 61.0, 62.0, 63.0]
    );
    assert_eq!(
        row_values(&objects, &io, "mtp/input_projection"),
        [100.0, 101.0, 102.0, 103.0]
    );
    assert_eq!(row_values(&objects, &io, "text/output_head").len(), 16);
}

#[test]
fn every_mtp_object_resolves_and_pieces_name_their_rows() {
    let directory = tempfile::tempdir().unwrap();
    let (_io, objects) = open(directory.path(), &tensors());
    for name in MTP_OBJECTS {
        let source = objects.object(name).unwrap();
        assert!(source.rows > 0 && source.columns > 0, "{name}");
    }
    let source = objects.object("text/layers/0/gdn/value_z").unwrap();
    assert_eq!(
        source.pieces[0],
        Piece {
            tensor: "model.language_model.layers.0.linear_attn.in_proj_qkv.weight".to_owned(),
            first_row: 4,
            rows: 4
        }
    );
    assert_eq!((source.rows, source.columns), (8, 4));
}

#[test]
fn read_hands_each_piece_the_region_asked_for_in_order() {
    let directory = tempfile::tempdir().unwrap();
    let (io, objects) = open(directory.path(), &tensors());
    let source = objects.object("text/layers/0/mlp/gate_up").unwrap();
    let mut asked = Vec::new();
    let mut region = |range: Range<u64>| {
        asked.push(range.clone());
        vec![0xAAu8; (range.end - range.start) as usize]
    };
    let pieces = drive(&io, |done| objects.read(&source, &mut region, done)).unwrap();
    assert_eq!(asked, [0..48, 48..96]);
    assert_eq!(pieces.len(), 2);
    assert_eq!(pieces[1][..2], bf16(&[50.0]));

    // A region of the wrong size is refused before anything is read.
    let mut wrong = |_: Range<u64>| vec![0u8; 3];
    assert!(matches!(
        objects.read(
            &source,
            &mut wrong,
            Arc::new(|_: Result<Vec<Vec<u8>>, _>| panic!("nothing was started"))
        ),
        Err(CheckpointError::Invalid(_))
    ));
}

#[test]
fn what_cannot_be_fused_is_refused() {
    let directory = tempfile::tempdir().unwrap();
    let (_io, objects) = open(directory.path(), &tensors());
    for name in [
        "nope",
        "text/layers/x/mlp/down",
        "text/layers/0/mlp/other",
        "text/layers/0",
        "mtp/layer/bad",
    ] {
        assert!(
            matches!(objects.object(name), Err(CheckpointError::Invalid(_))),
            "{name}"
        );
    }
    // A GDN layer has no attention tensors.
    assert!(objects.object("text/layers/0/attention/output").is_err());

    // A tensor that is not BF16, one that is narrower than its partner, and one with too few rows.
    let mut specs = tensors();
    specs[3] = Spec {
        name: "model.language_model.layers.0.mlp.gate_proj.weight",
        dtype: "F16",
        shape: vec![6, 4],
        bytes: vec![0; 48],
    };
    specs[4] = matrix("model.language_model.layers.0.mlp.up_proj.weight", 50.0, 6, 3);
    specs[0] = matrix(
        "model.language_model.layers.0.linear_attn.in_proj_qkv.weight",
        70.0,
        5,
        4,
    );
    let broken = tempfile::tempdir().unwrap();
    let (_io, objects) = open(broken.path(), &specs);
    let message = |name: &str| objects.object(name).unwrap_err().to_string();
    assert!(message("text/layers/0/mlp/gate_up").contains("not a BF16 matrix"));
    assert!(message("text/layers/0/gdn/value_z").contains("rows out of range"));
    specs[3] = matrix("model.language_model.layers.0.mlp.gate_proj.weight", 40.0, 6, 4);
    let mixed = tempfile::tempdir().unwrap();
    let (_io, objects) = open(mixed.path(), &specs);
    assert!(message_of(&objects, "text/layers/0/mlp/gate_up").contains("mixed widths"));
}

fn message_of(objects: &Qwen38Objects, name: &str) -> String {
    objects.object(name).unwrap_err().to_string()
}

#[test]
fn an_object_too_large_to_count_is_refused_not_wrapped() {
    let directory = tempfile::tempdir().unwrap();
    let (_io, objects) = open(directory.path(), &tensors());
    let huge = ObjectSource {
        name: "huge".to_owned(),
        pieces: vec![Piece {
            tensor: "lm_head.weight".to_owned(),
            first_row: 0,
            rows: 1 << 62,
        }],
        rows: 1 << 62,
        columns: 4,
    };
    assert_eq!(huge.bytes(), None);
    let mut region = |_: Range<u64>| Vec::<u8>::new();
    let refused = objects.read(
        &huge,
        &mut region,
        Arc::new(|_: Result<Vec<Vec<u8>>, _>| panic!("nothing was started")),
    );
    assert!(matches!(refused, Err(CheckpointError::Invalid(message)) if message.contains("too large")));
    let fits = objects.object("text/output_head").unwrap();
    assert_eq!(fits.bytes(), Some(16 * 4 * 2));
}
