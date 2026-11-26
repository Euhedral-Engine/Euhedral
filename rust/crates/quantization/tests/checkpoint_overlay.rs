//! An overlay checkpoint reads its replaced tensors from its own shard and every other tensor from the source.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it reads and writes its fixture files and stops on a failed unwrap"
)]

mod checkpoint_support;

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::sync::Arc;

use checkpoint_support::{Spec, World, bf16, drive, safetensors, write};
use euhedral_inference_quantization::checkpoint::CheckpointError;
use euhedral_inference_quantization::checkpoint::overlay;
use euhedral_inference_quantization::checkpoint::safetensors::Safetensors;

/// A source of one shard holding `a.weight` and `b.weight`, an index naming it, and a config file.
fn source(directory: &Path) {
    fs::create_dir_all(directory).unwrap();
    let shard = safetensors(&[
        Spec {
            name: "a.weight",
            dtype: "BF16",
            shape: vec![3],
            bytes: bf16(&[1.0, 2.0, 3.0]),
        },
        Spec {
            name: "b.weight",
            dtype: "BF16",
            shape: vec![2],
            bytes: bf16(&[4.0, 5.0]),
        },
    ]);
    write(&directory.join("model-00001.safetensors"), &shard);
    write(
        &directory.join("model.safetensors.index.json"),
        br#"{"metadata":{},"weight_map":{"a.weight":"model-00001.safetensors","b.weight":"model-00001.safetensors"}}"#,
    );
    write(&directory.join("config.json"), b"{}");
}

fn values(world: &World, directory: &Path, name: &str) -> Vec<f32> {
    let store = Arc::new(
        world
            .call(|done| Safetensors::open(&world.io, directory, done))
            .unwrap(),
    );
    let bytes = store.tensor(name).unwrap().bytes as usize;
    let bytes = drive(&world.io, |done| store.read_tensor(name, vec![0u8; bytes], done)).unwrap();
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes(*pair)) << 16))
        .collect()
}

fn overlay_of(world: &World, from: &Path, to: &Path, tensors: &[(&str, &[f32])]) -> Result<(), CheckpointError> {
    let tensors: BTreeMap<String, Vec<f32>> = tensors
        .iter()
        .map(|(name, values)| ((*name).to_owned(), values.to_vec()))
        .collect();
    world.call(|done| overlay::write(&world.io, &world.host, from, to, tensors, done))
}

#[test]
fn a_replaced_tensor_shadows_the_source_and_the_rest_falls_through() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let (from, to) = (directory.path().join("source"), directory.path().join("overlay"));
    source(&from);
    overlay_of(&world, &from, &to, &[("b.weight", &[-0.5, 6.5])]).unwrap();

    assert_eq!(
        values(&world, &to, "a.weight"),
        [1.0, 2.0, 3.0],
        "a tensor that was not replaced is the source's"
    );
    assert_eq!(
        values(&world, &to, "b.weight"),
        [-0.5, 6.5],
        "a replaced tensor is the overlay's"
    );
    assert_eq!(
        fs::read(to.join("config.json")).unwrap(),
        b"{}",
        "the source's other files are reachable from the overlay"
    );
    assert_eq!(values(&world, &from, "b.weight"), [4.0, 5.0], "the source is untouched");
    world.finish();
}

#[test]
fn replacements_are_rounded_to_nearest_even() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let (from, to) = (directory.path().join("source"), directory.path().join("overlay"));
    source(&from);
    // Halfway between two BF16 values: the one with an even last bit.
    let down = f32::from_bits(0x3F80_8000);
    let up = f32::from_bits(0x3F81_8000);
    overlay_of(&world, &from, &to, &[("a.weight", &[down, up, 3.0])]).unwrap();
    assert_eq!(
        values(&world, &to, "a.weight"),
        [f32::from_bits(0x3F80_0000), f32::from_bits(0x3F82_0000), 3.0]
    );
    world.finish();
}

#[test]
fn a_name_the_source_does_not_have_is_refused_before_anything_is_written() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let (from, to) = (directory.path().join("source"), directory.path().join("overlay"));
    source(&from);
    let error = overlay_of(
        &world,
        &from,
        &to,
        &[("b.weight", &[1.0, 2.0]), ("nope.weight", &[1.0])],
    )
    .unwrap_err();
    assert!(error.to_string().contains("nope.weight"), "{error}");
    assert!(!to.exists(), "nothing was written for a refused overlay");
    world.finish();
}
