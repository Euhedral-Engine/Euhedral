//! The EXL3 writer on a miniature checkpoint: what a full write holds, what it refuses, and what it does with the
//! files an earlier run left.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it reads and writes its fixture files and stops on a failed unwrap"
)]

mod checkpoint_support;
mod exl3_support;

use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::{Path, PathBuf};

use checkpoint_support::World;
use euhedral_inference_quantization::exl3::Exl3Writer;
use exl3_support::{
    CONFIG, GDN, MTP, layer_results, listing, miniature, mtp_results, names, open, run, tensor_names, tensor_of,
};
use serde_json::Value;

const LAYER: &str = "model.language_model.layers.0.";

struct Fixture {
    _directory: tempfile::TempDir,
    source: PathBuf,
    out: PathBuf,
    world: World,
}

fn fixture(config: &str, lm_head_value: f32) -> Fixture {
    let directory = tempfile::tempdir().unwrap();
    let (source, out) = (directory.path().join("source"), directory.path().join("out"));
    miniature(&source, config, lm_head_value);
    fs::create_dir_all(&out).unwrap();
    Fixture {
        _directory: directory,
        source,
        out,
        world: World::start(),
    }
}

fn writer(fixture: &Fixture) -> Exl3Writer {
    open(&fixture.world, &fixture.source, &fixture.out, 2.0, 4, 8)
}

fn rates(bits: f64) -> HashMap<String, f64> {
    layer_results(bits).1
}

fn finish(fixture: &Fixture, writer: Exl3Writer, bits: &HashMap<String, f64>) -> (Exl3Writer, Result<(), String>) {
    let list = listing(&fixture.source, &fixture.out);
    let (writer, result) = run(|done| writer.finish(bits, list, done));
    (writer, result.map_err(|error| error.to_string()))
}

fn write_layer_and_mtp(fixture: &Fixture, writer: Exl3Writer) -> Exl3Writer {
    let (results, bits) = layer_results(2.0);
    let (writer, result) = run(|done| writer.layer(0, &results, &bits, done));
    result.unwrap();
    let (writer, result) = run(|done| writer.mtp(&mtp_results(), done));
    result.unwrap();
    let _ = fixture;
    writer
}

fn json(path: &Path) -> Value {
    serde_json::from_slice(&fs::read(path).unwrap()).unwrap()
}

// Mutants caught: a quantized linear stored with the wrong dtype or shape; the a/b projections or lm_head left BF16;
// the embedding or vision tensors converted; an index that lists a tensor twice, misses one or adds sizes wrongly;
// a config that keeps tied embeddings or drops the block; a source file not copied or an old checkpoint copied.
#[test]
fn a_full_write_holds_every_source_tensor_in_the_layout() {
    let f = fixture(CONFIG, 1.0);
    let writer = write_layer_and_mtp(&f, writer(&f));
    let (_, result) = finish(&f, writer, &rates(2.0));
    result.unwrap();

    let layer = f.out.join("model.language_model.layers.0.safetensors");
    let first = format!("{LAYER}{}", GDN[0]);
    let (dtype, shape, _) = tensor_of(&layer, &format!("{first}.trellis"));
    assert_eq!((dtype.as_str(), shape), ("I16", vec![8, 8, 32]));
    let (dtype, shape, suh) = tensor_of(&layer, &format!("{first}.suh"));
    assert_eq!((dtype.as_str(), shape), ("F16", vec![128]));
    assert_eq!(&suh[34..36], &0x3C00u16.to_le_bytes(), "the scales are the ones given");
    let (dtype, shape, mul1) = tensor_of(&layer, &format!("{first}.mul1"));
    assert_eq!((dtype.as_str(), shape.len()), ("I32", 0));
    assert_eq!(mul1, 0x83DC_D12Du32.to_le_bytes());
    let (dtype, shape, _) = tensor_of(&f.out.join("mtp.safetensors"), "mtp.layers.0.self_attn.q_proj.trellis");
    assert_eq!((dtype.as_str(), shape), ("I16", vec![8, 8, 64]));
    let (_, shape, _) = tensor_of(&f.out.join("mtp.safetensors"), "mtp.fc.trellis");
    assert_eq!(shape, vec![16, 8, 64]);

    for (file, name) in [
        (
            "model.language_model.layers.0.safetensors",
            format!("{LAYER}linear_attn.in_proj_a.weight"),
        ),
        ("lm_head.safetensors", "lm_head.weight".to_owned()),
    ] {
        let (dtype, _, bytes) = tensor_of(&f.out.join(file), &name);
        assert_eq!(dtype, "F16", "{name}");
        assert_eq!(&bytes[..2], &0x3C00u16.to_le_bytes(), "BF16 1.0 is F16 1.0 in {name}");
    }
    for (file, name) in [
        (
            "model.language_model.embed_tokens.safetensors",
            "model.language_model.embed_tokens.weight",
        ),
        ("model.visual.safetensors", "model.visual.blocks.0.attn.qkv.weight"),
        (
            "model.language_model.layers.0.safetensors",
            "model.language_model.layers.0.linear_attn.A_log",
        ),
    ] {
        let (dtype, _, bytes) = tensor_of(&f.out.join(file), name);
        assert_eq!(dtype, "BF16", "{name}");
        assert_eq!(&bytes[..2], &0x3F80u16.to_le_bytes());
    }
    let (_, _, norm) = tensor_of(&f.out.join("lm_head.safetensors"), "model.language_model.norm.weight");
    assert_eq!(norm.len(), 256, "the final norm is stored with lm_head");

    // The tensors of a file are sorted by name.
    let listed = tensor_names(&layer);
    let mut sorted = listed.clone();
    sorted.sort();
    assert_eq!(listed, sorted);

    // The index lists every written tensor once, in the file that holds it, and its total is their size.
    let index = json(&f.out.join("model.safetensors.index.json"));
    let mut total = 0;
    let mut mapped = BTreeSet::new();
    for (name, file) in index["weight_map"].as_object().unwrap() {
        let file = file.as_str().unwrap();
        assert!(
            tensor_names(&f.out.join(file)).contains(name),
            "{name} is not in {file}"
        );
        total += tensor_of(&f.out.join(file), name).2.len() as u64;
        mapped.insert(name.clone());
    }
    assert_eq!(index["metadata"]["total_size"].as_u64().unwrap(), total);
    let mut written = BTreeSet::new();
    for file in names(&f.out).iter().filter(|file| file.ends_with(".safetensors")) {
        written.extend(tensor_names(&f.out.join(file)));
    }
    assert_eq!(mapped, written);
    let source_names: BTreeSet<String> = {
        let source = f.source.join("model.safetensors");
        tensor_names(&source).into_iter().collect()
    };
    for name in &source_names {
        let stem = name.strip_suffix(".weight").unwrap_or(name);
        assert!(
            mapped.contains(name) || mapped.contains(&format!("{stem}.trellis")),
            "{name} is missing"
        );
    }

    let config = json(&f.out.join("config.json"));
    assert_eq!(config["tie_word_embeddings"], false);
    assert_eq!(config["text_config"]["num_hidden_layers"], 1);
    let quantization = &config["quantization_config"];
    assert_eq!(quantization["quant_method"], "exl3");
    assert_eq!(quantization["version"], "1.6.1");
    assert_eq!(quantization["bits"], 2.0);
    assert_eq!(quantization["mtp_bits"], 4);
    let full = json(&f.out.join("quantization_config.json"));
    assert_eq!(
        full["tensor_storage"].as_object().unwrap().len(),
        14,
        "the 11 modules of the layer, the embedding, the final norm and lm_head"
    );

    let files: BTreeSet<String> = names(&f.out).into_iter().collect();
    for copied in ["tokenizer.json", "LICENSE"] {
        assert!(files.contains(copied), "{copied}");
    }
    assert!(!files.contains("old.bin"));
    assert!(
        !files.iter().any(|file| file.ends_with(".part")),
        "no partial file is left"
    );
    f.world.finish();
}

// Mutant caught: a finish that writes the large files before it knows a layer is missing.
#[test]
fn finish_refuses_a_checkpoint_missing_a_layer_before_it_writes_anything() {
    let f = fixture(CONFIG, 1.0);
    let (writer, result) = finish(&f, writer(&f), &HashMap::new());
    assert!(result.unwrap_err().contains("not written"));
    assert!(names(&f.out).is_empty(), "{:?}", names(&f.out));
    // The writer is given back whole: the same one goes on.
    let writer = write_layer_and_mtp(&f, writer);
    let (_, result) = finish(&f, writer, &rates(2.0));
    result.unwrap();
    f.world.finish();
}

// Mutant caught: a conversion that clamps to the largest half, or lets the value become infinity.
#[test]
fn a_value_beyond_half_precision_in_lm_head_is_refused_and_no_file_is_left() {
    let f = fixture(CONFIG, 1.0e38);
    let writer = write_layer_and_mtp(&f, writer(&f));
    let (_, result) = finish(&f, writer, &rates(2.0));
    assert!(result.unwrap_err().contains("lm_head.weight"));
    let files = names(&f.out);
    assert!(!files.contains(&"lm_head.safetensors".to_owned()), "{files:?}");
    assert!(!files.contains(&"model.safetensors.index.json".to_owned()), "{files:?}");
    assert!(!files.iter().any(|file| file.ends_with(".part")), "{files:?}");
    f.world.finish();
}

// Mutant caught: a half-integer rate written as an integer, or a rate that rounds.
#[test]
fn half_integer_rates_are_declared_and_stored() {
    let f = fixture(CONFIG, 1.0);
    let (results, bits) = layer_results(2.5);
    let (writer, result) = run(|done| writer(&f).layer(0, &results, &bits, done));
    result.unwrap();
    let (_, shape, _) = tensor_of(
        &f.out.join("model.language_model.layers.0.safetensors"),
        &format!("{LAYER}{}.trellis", GDN[0]),
    );
    assert_eq!(shape, vec![8, 8, 40]);
    assert_eq!(
        writer.quantization_config()["bits"],
        2.0,
        "the declared average is the writer's setting"
    );
    f.world.finish();
}

// Mutant caught: a layer result with a trellis of the wrong length, or scales that do not fit, being written.
#[test]
fn a_result_that_does_not_fit_its_linear_is_refused_and_nothing_is_written() {
    let f = fixture(CONFIG, 1.0);
    let (mut results, bits) = layer_results(2.0);
    results.get_mut(&format!("{LAYER}{}", GDN[2])).unwrap().trellis.pop();
    let (writer, result) = run(|done| writer(&f).layer(0, &results, &bits, done));
    assert!(result.is_err());
    assert!(names(&f.out).is_empty());
    let (mut results, bits) = layer_results(2.0);
    results.get_mut(&format!("{LAYER}{}", GDN[1])).unwrap().svh.pop();
    let (_, result) = run(|done| writer.layer(0, &results, &bits, done));
    assert!(result.is_err());
    assert!(names(&f.out).is_empty());
    f.world.finish();
}

// Mutant caught: a result keyed by an unquantized linear being ignored instead of refused.
#[test]
fn a_result_for_a_linear_that_is_not_quantized_is_refused() {
    let f = fixture(CONFIG, 1.0);
    let (mut results, mut bits) = layer_results(2.0);
    let stray = format!("{LAYER}linear_attn.in_proj_a");
    results.insert(stray.clone(), results[&format!("{LAYER}{}", GDN[0])].clone());
    bits.insert(stray, 2.0);
    let (_, result) = run(|done| writer(&f).layer(0, &results, &bits, done));
    assert!(result.unwrap_err().to_string().contains("in_proj_a"));
    assert!(names(&f.out).is_empty());
    f.world.finish();
}

// Mutant caught: an MTP write that accepts a partial or extra set of linears.
#[test]
fn mtp_needs_exactly_its_eight_linears() {
    let f = fixture(CONFIG, 1.0);
    let mut results = mtp_results();
    results.remove("mtp.fc");
    let (writer, result) = run(|done| writer(&f).mtp(&results, done));
    assert!(result.is_err());
    let mut results = mtp_results();
    results.insert(format!("{LAYER}{}", GDN[0]), results[MTP[0]].clone());
    let (_, result) = run(|done| writer.mtp(&results, done));
    assert!(result.is_err());
    assert!(names(&f.out).is_empty());
    f.world.finish();
}

// Mutants caught: an earlier layer file that is not adopted (finish then misses its tensors); one that is adopted
// without checking its tensors.
#[test]
fn files_an_earlier_run_left_are_adopted_when_whole_and_refused_when_not() {
    let f = fixture(CONFIG, 1.0);
    let first = write_layer_and_mtp(&f, writer(&f));
    drop(first);
    // A new writer finishes the checkpoint from what is on disk.
    let (_, result) = finish(&f, writer(&f), &rates(2.0));
    result.unwrap();
    let index = json(&f.out.join("model.safetensors.index.json"));
    assert_eq!(
        index["weight_map"][format!("{LAYER}{}.suh", GDN[0])],
        "model.language_model.layers.0.safetensors"
    );
    assert_eq!(index["weight_map"]["mtp.fc.trellis"], "mtp.safetensors");
    assert_eq!(
        json(&f.out.join("config.json"))["quantization_config"]["mtp_bits"],
        4,
        "the adopted MTP file is declared"
    );
    assert_eq!(
        json(&f.out.join("quantization_config.json"))["tensor_storage"]
            .as_object()
            .unwrap()
            .len(),
        14
    );
    f.world.finish();
}

// Mutant caught: adopting a layer file without comparing its tensors to what the layer needs.
#[test]
fn an_earlier_layer_file_missing_tensors_fails_finish() {
    let f = fixture(CONFIG, 1.0);
    let first = write_layer_and_mtp(&f, writer(&f));
    drop(first);
    // Replace the layer file with one that lacks most of its tensors.
    let layer = f.out.join("model.language_model.layers.0.safetensors");
    let spec = vec![checkpoint_support::Spec {
        name: "model.language_model.layers.0.input_layernorm.weight",
        dtype: "BF16",
        shape: vec![128],
        bytes: checkpoint_support::bf16(&[1.0; 128]),
    }];
    checkpoint_support::write(&layer, &checkpoint_support::safetensors(&spec));
    let (_, result) = finish(&f, writer(&f), &rates(2.0));
    assert!(result.unwrap_err().contains("earlier run"));
    assert!(!names(&f.out).contains(&"lm_head.safetensors".to_owned()));
    f.world.finish();
}

// Mutant caught: a finish that lets a safetensors file the index would not list stay in the output.
#[test]
fn a_stray_safetensors_file_is_refused_before_the_large_files() {
    let f = fixture(CONFIG, 1.0);
    let writer = write_layer_and_mtp(&f, writer(&f));
    checkpoint_support::write(&f.out.join("model.language_model.layers.7.safetensors"), b"x");
    let (_, result) = finish(&f, writer, &rates(2.0));
    assert!(result.unwrap_err().contains("layers.7"));
    let files = names(&f.out);
    assert!(!files.contains(&"lm_head.safetensors".to_owned()), "{files:?}");
    f.world.finish();
}

// Mutant caught: a rate check that compares the wrong number of words, or none.
#[test]
fn a_rate_that_differs_from_the_stored_one_is_refused() {
    let f = fixture(CONFIG, 1.0);
    let writer = write_layer_and_mtp(&f, writer(&f));
    let (_, result) = finish(&f, writer, &rates(3.0));
    assert!(result.unwrap_err().contains("different rate"));
    assert!(!names(&f.out).contains(&"lm_head.safetensors".to_owned()));
    f.world.finish();
}

// Mutant caught: a source that is already quantized being written over.
#[test]
fn a_source_that_is_already_quantized_is_refused_before_the_large_files() {
    let quantized = CONFIG.replacen('{', r#"{"quantization_config":{"quant_method":"gptq"},"#, 1);
    let f = fixture(&quantized, 1.0);
    let writer = write_layer_and_mtp(&f, writer(&f));
    let (_, result) = finish(&f, writer, &rates(2.0));
    assert!(result.unwrap_err().contains("already quantized"));
    assert!(!names(&f.out).contains(&"lm_head.safetensors".to_owned()));
    f.world.finish();
}

// Mutant caught: a rewritten layer whose tensors count twice in the index total.
#[test]
fn writing_a_layer_again_replaces_it() {
    let f = fixture(CONFIG, 1.0);
    let writer = write_layer_and_mtp(&f, writer(&f));
    let (results, bits) = layer_results(2.0);
    let (writer, result) = run(|done| writer.layer(0, &results, &bits, done));
    result.unwrap();
    let (_, result) = finish(&f, writer, &rates(2.0));
    result.unwrap();
    let index = json(&f.out.join("model.safetensors.index.json"));
    let mut total = 0;
    for (name, file) in index["weight_map"].as_object().unwrap() {
        total += tensor_of(&f.out.join(file.as_str().unwrap()), name).2.len() as u64;
    }
    assert_eq!(index["metadata"]["total_size"].as_u64().unwrap(), total);
    f.world.finish();
}
