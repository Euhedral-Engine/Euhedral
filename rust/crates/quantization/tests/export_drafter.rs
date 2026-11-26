//! The drafter export: which weights the MTP layer is made of, in which order its objects are written, what is
//! refused, and that a run that ends early leaves nothing under the output's name.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it stops on a failed unwrap and prepares and reads its files directly"
)]

mod checkpoint_support;

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use checkpoint_support::{Spec, World, bf16, safetensors};
use euhedral_inference_edrl::tiny::TinyArtifact;
use euhedral_inference_edrl::{WeightFormat, WeightLayout};
use euhedral_inference_engine::Handoff;
use euhedral_inference_quantization::assemble::{ArtifactFile, AssembleError, Done};
use euhedral_inference_quantization::mtp::export::{ExportError, Settings, export};
use euhedral_inference_quantization::mtp::nvfp4;
use tempfile::TempDir;

/// A drafter layer of the smallest shape NVFP4 takes: rows of 128 columns.
const CONFIG: &str = r#"{"text_config": {
    "num_hidden_layers": 1, "layer_types": ["full_attention"],
    "hidden_size": 128, "intermediate_size": 128, "num_attention_heads": 2, "num_key_value_heads": 1,
    "head_dim": 64, "partial_rotary_factor": 0.25, "linear_num_key_heads": 1, "linear_num_value_heads": 1,
    "linear_key_head_dim": 2, "linear_value_head_dim": 2, "linear_conv_kernel_dim": 4,
    "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10000000.0}, "vocab_size": 16}}"#;

/// The MTP objects in the order the converter writes them, with their rows and columns (a norm has one row).
const ORDER: [(&str, u64, u64); 12] = [
    ("mtp/input_projection", 128, 256),
    ("mtp/embedding_norm", 1, 128),
    ("mtp/hidden_norm", 1, 128),
    ("mtp/layer/input_norm", 1, 128),
    ("mtp/layer/attention/query_key_gate_value", 384, 128),
    ("mtp/layer/attention/query_norm", 1, 64),
    ("mtp/layer/attention/key_norm", 1, 64),
    ("mtp/layer/attention/output", 128, 128),
    ("mtp/layer/post_attention_norm", 1, 128),
    ("mtp/layer/mlp/gate_up", 256, 128),
    ("mtp/layer/mlp/down", 128, 128),
    ("mtp/final_norm", 1, 128),
];

/// The parameter file's order: the projections, then the norms.
const PARAMETERS: [&str; 12] = [
    "mtp/input_projection",
    "mtp/layer/attention/query_key_gate_value",
    "mtp/layer/attention/output",
    "mtp/layer/mlp/gate_up",
    "mtp/layer/mlp/down",
    "mtp/embedding_norm",
    "mtp/hidden_norm",
    "mtp/layer/input_norm",
    "mtp/layer/post_attention_norm",
    "mtp/layer/attention/query_norm",
    "mtp/layer/attention/key_norm",
    "mtp/final_norm",
];

/// Values a BF16 holds exactly, different for every seed and index.
fn values(seed: u64, count: u64) -> Vec<f32> {
    (0..count)
        .map(|index| ((index * (2 * seed + 5) + seed) % 255) as f32 / 16.0 - 8.0)
        .collect()
}

fn tensor(name: &'static str, seed: u64, shape: &[u64]) -> (Spec, Vec<f32>) {
    let held = values(seed, shape.iter().product());
    let spec = Spec {
        name,
        dtype: "BF16",
        shape: shape.to_vec(),
        bytes: bf16(&held),
    };
    (spec, held)
}

/// The checkpoint's MTP tensors, and the values of each object as the converter fuses them.
struct Checkpoint {
    specs: Vec<Spec>,
    objects: Vec<(&'static str, Vec<f32>)>,
}

fn checkpoint() -> Checkpoint {
    let (mut specs, mut held) = (Vec::new(), Vec::new());
    let mut add = |name: &'static str, seed: u64, shape: &[u64]| {
        let (spec, floats) = tensor(name, seed, shape);
        specs.push(spec);
        held.push((name, floats));
    };
    add("mtp.fc.weight", 1, &[128, 256]);
    add("mtp.layers.0.self_attn.q_proj.weight", 2, &[256, 128]);
    add("mtp.layers.0.self_attn.k_proj.weight", 3, &[64, 128]);
    add("mtp.layers.0.self_attn.v_proj.weight", 4, &[64, 128]);
    add("mtp.layers.0.self_attn.o_proj.weight", 5, &[128, 128]);
    add("mtp.layers.0.mlp.gate_proj.weight", 6, &[128, 128]);
    add("mtp.layers.0.mlp.up_proj.weight", 7, &[128, 128]);
    add("mtp.layers.0.mlp.down_proj.weight", 8, &[128, 128]);
    add("mtp.pre_fc_norm_embedding.weight", 9, &[128]);
    add("mtp.pre_fc_norm_hidden.weight", 10, &[128]);
    add("mtp.layers.0.input_layernorm.weight", 11, &[128]);
    add("mtp.layers.0.post_attention_layernorm.weight", 12, &[128]);
    add("mtp.layers.0.self_attn.q_norm.weight", 13, &[64]);
    add("mtp.layers.0.self_attn.k_norm.weight", 14, &[64]);
    add("mtp.norm.weight", 15, &[128]);
    let of = |name: &str| held.iter().find(|(known, _)| *known == name).unwrap().1.clone();
    // The query and gate rows of each head are interleaved in `q_proj`: head h owns rows 128 h + [0, 64) for the query
    // and 128 h + [64, 128) for the gate.
    let q = of("mtp.layers.0.self_attn.q_proj.weight");
    let heads = |gate: usize| -> Vec<f32> {
        (0..2)
            .flat_map(|head| q[(head * 128 + 64 * gate) * 128..(head * 128 + 64 * gate + 64) * 128].to_vec())
            .collect()
    };
    let fused = [
        heads(0),
        of("mtp.layers.0.self_attn.k_proj.weight"),
        heads(1),
        of("mtp.layers.0.self_attn.v_proj.weight"),
    ]
    .concat();
    let gate_up = [
        of("mtp.layers.0.mlp.gate_proj.weight"),
        of("mtp.layers.0.mlp.up_proj.weight"),
    ]
    .concat();
    let objects = vec![
        ("mtp/input_projection", of("mtp.fc.weight")),
        ("mtp/embedding_norm", of("mtp.pre_fc_norm_embedding.weight")),
        ("mtp/hidden_norm", of("mtp.pre_fc_norm_hidden.weight")),
        ("mtp/layer/input_norm", of("mtp.layers.0.input_layernorm.weight")),
        ("mtp/layer/attention/query_key_gate_value", fused),
        (
            "mtp/layer/attention/query_norm",
            of("mtp.layers.0.self_attn.q_norm.weight"),
        ),
        (
            "mtp/layer/attention/key_norm",
            of("mtp.layers.0.self_attn.k_norm.weight"),
        ),
        ("mtp/layer/attention/output", of("mtp.layers.0.self_attn.o_proj.weight")),
        (
            "mtp/layer/post_attention_norm",
            of("mtp.layers.0.post_attention_layernorm.weight"),
        ),
        ("mtp/layer/mlp/gate_up", gate_up),
        ("mtp/layer/mlp/down", of("mtp.layers.0.mlp.down_proj.weight")),
        ("mtp/final_norm", of("mtp.norm.weight")),
    ];
    Checkpoint { specs, objects }
}

struct Fixture {
    world: World,
    dir: TempDir,
    model: PathBuf,
    base: PathBuf,
    checkpoint: Checkpoint,
}

impl Fixture {
    fn new() -> Fixture {
        let dir = tempfile::tempdir().unwrap();
        let model = dir.path().join("model");
        fs::create_dir(&model).unwrap();
        let checkpoint = checkpoint();
        fs::write(model.join("model.safetensors"), safetensors(&checkpoint.specs)).unwrap();
        fs::write(model.join("config.json"), CONFIG).unwrap();
        let base = dir.path().join("base.edrl");
        fs::write(&base, TinyArtifact::new(1, true).bytes().unwrap()).unwrap();
        Fixture {
            world: World::start(),
            dir,
            model,
            base,
            checkpoint,
        }
    }

    fn settings(&self, parameters: Option<&Path>, out: &Path) -> Settings {
        Settings {
            parameters: parameters.map(Path::to_path_buf),
            model: self.model.clone(),
            artifact: self.base.clone(),
            out: out.to_path_buf(),
        }
    }

    fn export(&self, parameters: Option<&Path>, out: &Path) -> Result<(), ExportError> {
        export(
            &self.settings(parameters, out),
            &self.world.io,
            &self.world.host,
            1 << 20,
        )
    }

    fn open(&self, path: &Path) -> ArtifactFile {
        let handoff = Handoff::new();
        let sender = Arc::clone(&handoff);
        let done: Done<ArtifactFile> = Arc::new(move |result: Result<ArtifactFile, AssembleError>| sender.send(result));
        ArtifactFile::open(&self.world.io, path, true, done).unwrap();
        handoff
            .wait(Duration::from_secs(20))
            .expect("the artifact never opened")
            .unwrap()
    }

    /// The names a directory holds, to see that nothing but the output was left in it.
    fn left(&self) -> Vec<String> {
        let mut names: Vec<String> = fs::read_dir(self.dir.path())
            .unwrap()
            .map(|entry| entry.unwrap().file_name().into_string().unwrap())
            .collect();
        names.sort();
        names
    }

    /// A parameter file: a header, then every object's values in the parameter order.
    fn parameter_file(&self, seed: u64) -> (PathBuf, Vec<(&'static str, Vec<f32>)>) {
        let held: Vec<(&'static str, Vec<f32>)> = PARAMETERS
            .iter()
            .enumerate()
            .map(|(index, name)| {
                let (_, rows, columns) = ORDER.iter().find(|(known, _, _)| known == name).unwrap();
                (*name, values(seed + index as u64, rows * columns))
            })
            .collect();
        let mut bytes = vec![0u8; 4096];
        for (_, floats) in &held {
            bytes.extend(floats.iter().flat_map(|value| value.to_le_bytes()));
        }
        let path = self.dir.path().join("best.bin");
        fs::write(&path, bytes).unwrap();
        (path, held)
    }
}

/// The payload of `name` in an artifact.
fn payload(file: &ArtifactFile, name: &str) -> Vec<u8> {
    let entry = file.table().get(name).unwrap_or_else(|| panic!("no {name}"));
    let bytes = fs::read(file.path()).unwrap();
    bytes[entry.offset() as usize..(entry.offset() + entry.bytes()) as usize].to_vec()
}

/// What the export must have written for `name` from `held`: NVFP4 for a projection, BF16 for a norm.
fn expected(name: &str, held: &[(&str, Vec<f32>)]) -> Vec<u8> {
    let (_, rows, columns) = ORDER.iter().find(|(known, _, _)| *known == name).unwrap();
    let floats = &held.iter().find(|(known, _)| *known == name).unwrap().1;
    if *rows == 1 {
        bf16(floats)
    } else {
        nvfp4::round(floats, *rows, *columns as u32).unwrap()
    }
}

/// Checks the exported artifact against the base and the values the MTP layer must hold.
fn check(fixture: &Fixture, out: &Path, held: &[(&str, Vec<f32>)]) {
    let (base, written) = (fixture.open(&fixture.base), fixture.open(out));
    let names = |file: &ArtifactFile| -> Vec<String> {
        file.table()
            .entries()
            .iter()
            .map(|entry| entry.name().to_owned())
            .collect()
    };
    // The new layer takes the place of the base's first MTP object; the rest of the base is as it was.
    let mut want: Vec<String> = Vec::new();
    for name in names(&base) {
        if !name.starts_with("mtp/") {
            want.push(name);
        } else if !want.iter().any(|known| known == "mtp/input_projection") {
            want.extend(ORDER.map(|(name, _, _)| name.to_owned()));
        }
    }
    assert_eq!(names(&written), want);
    for name in names(&base).iter().filter(|name| !name.starts_with("mtp/")) {
        assert_eq!(payload(&written, name), payload(&base, name), "{name}");
    }
    for (name, rows, columns) in ORDER {
        let entry = written.table().get(name).unwrap();
        let projection = rows > 1;
        assert_eq!(
            entry.shape(),
            if projection { vec![rows, columns] } else { vec![columns] },
            "{name}"
        );
        assert_eq!(
            (entry.format(), entry.layout()),
            if projection {
                (WeightFormat::Nvfp4, WeightLayout::RowSplitK128V1)
            } else {
                (WeightFormat::Bf16, WeightLayout::ContiguousLeV1)
            },
            "{name}"
        );
        assert_eq!(payload(&written, name), expected(name, held), "{name}");
    }
    assert!(written.tokenizer().is_some() && base.tokenizer() == written.tokenizer());
}

#[test]
fn without_parameters_the_checkpoints_mtp_weights_are_written_in_the_converters_order() {
    let fixture = Fixture::new();
    let out = fixture.dir.path().join("out.edrl");
    fixture.export(None, &out).unwrap();
    check(&fixture, &out, &fixture.checkpoint.objects);
    assert_eq!(fixture.left(), ["base.edrl", "model", "out.edrl"]);
    fixture.world.finish();
}

#[test]
fn with_parameters_their_values_are_written_and_the_checkpoints_are_not() {
    let fixture = Fixture::new();
    let (parameters, held) = fixture.parameter_file(40);
    let out = fixture.dir.path().join("out.edrl");
    fixture.export(Some(&parameters), &out).unwrap();
    check(&fixture, &out, &held);
    assert_ne!(held[0].1, fixture.checkpoint.objects[0].1);
    fixture.world.finish();
}

#[test]
fn a_parameter_file_of_another_shape_is_refused_and_nothing_is_written() {
    let fixture = Fixture::new();
    let (parameters, _) = fixture.parameter_file(40);
    let whole = fs::read(&parameters).unwrap();
    let out = fixture.dir.path().join("out.edrl");
    for (what, bytes) in [
        ("too short", whole[..whole.len() - 4].to_vec()),
        ("too long", [whole.clone(), vec![0; 4]].concat()),
        ("no header", whole[..100].to_vec()),
    ] {
        fs::write(&parameters, bytes).unwrap();
        let error = fixture.export(Some(&parameters), &out).unwrap_err();
        assert!(
            matches!(&error, ExportError::Invalid(message) if message.contains("needs")),
            "{what}: {error}"
        );
        assert_eq!(fixture.left(), ["base.edrl", "best.bin", "model"], "{what}");
    }
    fixture.world.finish();
}

#[test]
fn a_run_that_cannot_complete_leaves_the_output_name_and_the_base_as_they_were() {
    let fixture = Fixture::new();
    let before = fs::read(&fixture.base).unwrap();
    // The output is the base itself: the writer refuses to overwrite its own source, after the new objects are written.
    let error = fixture.export(None, &fixture.base.clone()).unwrap_err();
    assert!(error.to_string().contains("would overwrite"), "{error}");
    assert_eq!(fs::read(&fixture.base).unwrap(), before);
    assert_eq!(
        fixture.left(),
        ["base.edrl", "model"],
        "no .part and no payload file is left"
    );
    fixture.world.finish();
}

#[test]
fn a_checkpoint_without_the_drafter_is_refused() {
    let fixture = Fixture::new();
    let kept: Vec<Spec> = fixture
        .checkpoint
        .specs
        .iter()
        .take(1)
        .map(|spec| Spec {
            name: spec.name,
            dtype: spec.dtype,
            shape: spec.shape.clone(),
            bytes: spec.bytes.clone(),
        })
        .collect();
    fs::write(fixture.model.join("model.safetensors"), safetensors(&kept)).unwrap();
    let out = fixture.dir.path().join("out.edrl");
    assert!(fixture.export(None, &out).is_err());
    assert_eq!(fixture.left(), ["base.edrl", "model"]);
    fixture.world.finish();
}
