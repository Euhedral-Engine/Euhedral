//! The linears of a Qwen 3.8 checkpoint: where their rows sit in the fused objects (on a miniature one), and that the
//! built-in shapes are the Swift 1.5 checkpoint's own (skipped without `euhedral.quantization.checkpoint`).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it stops on a failed unwrap"
)]

mod checkpoint_support;
mod exl3_support;

use std::path::{Path, PathBuf};
use std::sync::Arc;

use checkpoint_support::World;
use euhedral_inference_quantization::checkpoint::objects::Qwen38Objects;
use euhedral_inference_quantization::checkpoint::safetensors::Safetensors;
use euhedral_inference_quantization::exl3::linears::{layer, placements, qwen38, qwen38_of};
use exl3_support::{CONFIG, miniature};
use test_support::require_property;

fn objects_of(world: &World, directory: &Path) -> Qwen38Objects {
    let store = Arc::new(
        world
            .call(|done| Safetensors::open(&world.io, directory, done))
            .unwrap(),
    );
    world.call(|done| Qwen38Objects::open(store, directory, done)).unwrap()
}

/// The miniature checkpoint: a linear-attention layer and a full-attention one.
fn synthetic(world: &World) -> (tempfile::TempDir, Qwen38Objects) {
    let directory = tempfile::tempdir().unwrap();
    miniature(directory.path(), CONFIG, 1.0);
    let objects = objects_of(world, directory.path());
    (directory, objects)
}

fn open(world: &World) -> Option<Qwen38Objects> {
    let directory = PathBuf::from(require_property("euhedral.quantization.checkpoint")?);
    Some(objects_of(world, &directory))
}

// Mutant caught: a placement that skips a piece, or counts a piece's rows from the wrong object row, leaves a row
// uncovered or covered twice.
#[test]
fn placements_cover_every_row_of_a_linear_once() {
    let world = World::start();
    let (_directory, objects) = synthetic(&world);
    for index in [0, 1] {
        for linear in layer(&objects, index)
            .unwrap()
            .into_iter()
            .filter(|linear| linear.quantized)
        {
            let mut seen = vec![false; linear.rows as usize];
            for place in placements(&objects, index, &linear.tensor).unwrap() {
                for row in place.linear_row..place.linear_row + place.rows {
                    assert!(!seen[row as usize], "{} row {row}", linear.tensor);
                    seen[row as usize] = true;
                }
            }
            assert!(seen.iter().all(|covered| *covered), "{}", linear.tensor);
        }
    }
    world.finish();
}

// Mutant caught: treating the GDN a/b projections (8 or 48 rows) as quantized linears.
#[test]
fn the_gdn_a_and_b_projections_are_not_quantized_and_sit_in_no_object() {
    let world = World::start();
    let (_directory, objects) = synthetic(&world);
    let linears = layer(&objects, 0).unwrap();
    let small: Vec<_> = linears
        .iter()
        .filter(|linear| linear.tensor.ends_with("in_proj_a") || linear.tensor.ends_with("in_proj_b"))
        .collect();
    assert_eq!(small.len(), 2);
    for linear in small {
        assert!(!linear.quantized, "{}", linear.tensor);
        assert!(
            placements(&objects, 0, &linear.tensor).unwrap().is_empty(),
            "{}",
            linear.tensor
        );
    }
    assert_eq!(linears.iter().filter(|linear| linear.quantized).count(), 6);
    world.finish();
}

// Mutant caught: a built-in shape that differs from the checkpoint's.
#[test]
fn the_built_in_shapes_are_the_checkpoints() {
    let world = World::start();
    let Some(objects) = open(&world) else {
        world.finish();
        return;
    };
    assert_eq!(qwen38_of(&objects).unwrap(), qwen38(objects.config.layers).unwrap());
    world.finish();
}
