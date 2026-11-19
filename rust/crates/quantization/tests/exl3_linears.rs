//! The linears of the Swift 1.5 checkpoint: where their rows sit in the fused objects, and that the built-in shapes
//! are the checkpoint's own. Skipped without `euhedral.quantization.checkpoint`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it stops on a failed unwrap"
)]

mod checkpoint_support;

use std::path::PathBuf;
use std::sync::Arc;

use checkpoint_support::World;
use euhedral_inference_quantization::checkpoint::objects::Qwen38Objects;
use euhedral_inference_quantization::checkpoint::safetensors::Safetensors;
use euhedral_inference_quantization::exl3::linears::{layer, placements, qwen38, qwen38_of};
use test_support::require_property;

fn open(world: &World) -> Option<Qwen38Objects> {
    let directory = PathBuf::from(require_property("euhedral.quantization.checkpoint")?);
    let store = Arc::new(
        world
            .call(|done| Safetensors::open(&world.io, &directory, done))
            .unwrap(),
    );
    Some(world.call(|done| Qwen38Objects::open(store, &directory, done)).unwrap())
}

// Mutant caught: a placement that skips a piece, or counts a piece's rows from the wrong object row, leaves a row
// uncovered or covered twice.
#[test]
fn placements_cover_every_row_of_a_linear_once() {
    let world = World::start();
    let Some(objects) = open(&world) else {
        world.finish();
        return;
    };
    for index in [0, 3] {
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
    let Some(objects) = open(&world) else {
        world.finish();
        return;
    };
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
