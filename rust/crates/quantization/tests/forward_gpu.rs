//! Device tests of the layer forward's lane operations and of a layer on the real checkpoint. `#[ignore]`d like the
//! crate's other device tests:
//! `cargo test -p euhedral-inference-quantization --test forward_gpu -- --ignored --nocapture`.
//!
//! The engine library comes from `euhedral.cuda.library`, the math libraries from
//! `euhedral.quantization.cuda-libraries`, the checkpoint from `euhedral.quantization.checkpoint` and the forward
//! input from `euhedral.quantization.forward-fixture`. A missing setting fails the test: a device test that skips is
//! no check. The builtins library of NVRTC must be on `LD_LIBRARY_PATH`.
#![cfg(target_os = "linux")]
#![allow(
    clippy::disallowed_methods,
    clippy::print_stdout,
    reason = "a test: reads the fixture and checks the device with synchronous calls, and prints what it measured"
)]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

mod checkpoint_support;
mod forward_support;

use std::fs::read;
use std::path::PathBuf;
use std::sync::Arc;

use checkpoint_support::drive;
use euhedral_inference_quantization::checkpoint::objects::Qwen38Objects;
use euhedral_inference_quantization::checkpoint::safetensors::Safetensors;
use euhedral_inference_quantization::forward::{
    LaneContext, LayerForward, LayerWeights, Observation, StagedLayer, Target, stage,
};
use euhedral_io::AsyncIo;
use forward_support::{Rig, bf16_to_f32, gaussians, npy_payload, rig};

const UPLOAD_CHUNK: usize = 64 << 20;

fn to_bf16(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

fn bytes_of(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// The symmetric Hessian (upper blocks, then mirrored) against a float64 X^T X on the host, for a width that is not a
/// multiple of the block, accumulated over two calls.
#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn the_accumulated_hessian_is_x_transpose_x_on_both_triangles() {
    let mut rig = rig();
    let (rows, k) = (300usize, 1100usize);
    let x: Vec<u16> = gaussians(rows * k, 3).into_iter().map(to_bf16).collect();
    let x_device = rig.upload(&bytes_of(&x));
    let hessian = rig.upload(&vec![0u8; 4 * k * k]);
    let half = rows / 2;
    let (x_ptr, h_ptr) = (x_device.ptr(), hessian.ptr());
    rig.run(|context| {
        context.accumulate_hessian(h_ptr, x_ptr, half as u32, k as u32)?;
        context.accumulate_hessian(
            h_ptr,
            x_ptr.offset((2 * half * k) as u64),
            (rows - half) as u32,
            k as u32,
        )?;
        context.symmetrize(h_ptr, k as u32)
    });
    let bytes = rig.download(&hessian, 4 * k * k);
    let got = |row: usize, column: usize| {
        let at = 4 * (column * k + row);
        f32::from_le_bytes(bytes[at..at + 4].try_into().unwrap())
    };
    let mut worst = 0f64;
    // Every row of the matrix is probed on a stride that crosses the block boundary at 1024, both triangles.
    for a in (0..k).step_by(7) {
        for b in (0..k).step_by(5) {
            let want: f64 = (0..rows)
                .map(|r| f64::from(bf16_to_f32(x[r * k + a])) * f64::from(bf16_to_f32(x[r * k + b])))
                .sum();
            worst = worst.max((f64::from(got(a, b)) - want).abs() / (want.abs() + 1.0));
        }
    }
    println!("worst relative difference {worst:.2e}");
    assert!(worst < 1e-4, "worst relative difference {worst}");
    rig.finish();
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn gathered_rows_are_the_table_rows_the_ids_name_in_order() {
    let mut rig = rig();
    let (entries, width) = (10usize, 24usize);
    let table: Vec<u16> = (0..entries * width)
        .map(|i| to_bf16((i / width * 10 + i % width) as f32))
        .collect();
    let ids = [3i32, 0, 9, 3, 7];
    let table_device = rig.upload(&bytes_of(&table));
    let ids_device = rig.upload(&ids.iter().flat_map(|id| id.to_le_bytes()).collect::<Vec<_>>());
    let out = rig.upload(&vec![0u8; ids.len() * width * 2]);
    let (t, i, o) = (table_device.ptr(), ids_device.ptr(), out.ptr());
    rig.run(|context| context.gather_rows(t, i, ids.len() as u32, width as u32, o));
    let bytes = rig.download(&out, ids.len() * width * 2);
    for (row, id) in ids.iter().enumerate() {
        for column in 0..width {
            let at = 2 * (row * width + column);
            let value = bf16_to_f32(u16::from_le_bytes([bytes[at], bytes[at + 1]]));
            assert_eq!(value, (*id as usize * 10 + column) as f32, "row {row}, column {column}");
        }
    }
    rig.finish();
}

/// The checkpoint and the first-run input of the real-model tests.
struct Model {
    io: Arc<AsyncIo>,
    objects: Qwen38Objects,
    input: Vec<u8>,
}

fn model() -> Model {
    let checkpoint = PathBuf::from(
        test_support::gradle_property("euhedral.quantization.checkpoint")
            .expect("set euhedral.quantization.checkpoint (EUHEDRAL_TEST_QUANTIZATION_CHECKPOINT)"),
    );
    let fixture = PathBuf::from(
        test_support::gradle_property("euhedral.quantization.forward-fixture")
            .expect("set euhedral.quantization.forward-fixture (EUHEDRAL_TEST_QUANTIZATION_FORWARD_FIXTURE)"),
    );
    let io = AsyncIo::open().unwrap();
    let store = Arc::new(drive(&io, |done| Safetensors::open(&io, &checkpoint, done)).unwrap());
    let objects = drive(&io, |done| Qwen38Objects::open(store, &checkpoint, done)).unwrap();
    let input = read(fixture.join("input.npy")).unwrap();
    Model {
        io,
        objects,
        input: npy_payload(&input).to_vec(),
    }
}

/// What a forward run of one layer gives: the hidden states after it, and what the observer saw.
#[derive(Debug)]
struct Run {
    hidden: Vec<u8>,
    observed: Vec<(String, u32, u32)>,
}

/// Runs `layer` over two sequences of 256 rows twice on `forward` from `model.input`, each run's hidden states and
/// observations returned.
fn forward_twice(rig: &mut Rig, model: &Model, staged: &StagedLayer) -> [Run; 2] {
    let (sequences, length) = (2u32, 256u32);
    let rows = sequences * length;
    let hidden = rig.upload(&model.input);
    let (library, pool) = (Arc::clone(&rig.library), Arc::clone(&rig.pool));
    let config = &model.objects.config;
    let mut retained = Vec::new();
    let (mut forward, weights) = rig.run(|context| {
        let forward = LayerForward::new(context.ops(), &library, &pool, config, rows)?;
        let weights = LayerWeights::upload(context.ops(), &pool, staged, UPLOAD_CHUNK, &mut retained)?;
        Ok((forward, weights))
    });
    drop(retained);
    let hidden_ptr = hidden.ptr();
    let mut runs = Vec::new();
    for _ in 0..2 {
        rig.overwrite(&hidden, &model.input);
        let mut observed = Vec::new();
        rig.run(|context| {
            let mut observer = |seen: &Observation<'_>, _: &mut LaneContext<'_, _>| {
                observed.push((seen.object.to_owned(), seen.rows, seen.columns));
                Ok(())
            };
            forward.forward(context, &weights, hidden_ptr, sequences, length, Some(&mut observer))
        });
        forward.retire();
        let bytes = rig.download(&hidden, model.input.len());
        runs.push(Run {
            hidden: bytes,
            observed,
        });
    }
    rig.run(|context| {
        weights.release(context.ops())?;
        forward.release(context.ops())
    });
    let [first, second]: [Run; 2] = runs.try_into().ok().unwrap();
    [first, second]
}

fn staged(model: &Model, target: Target) -> StagedLayer {
    drive(&model.io, |done| stage(&model.objects, target, done)).unwrap()
}

/// A second run over the same input gives the same hidden states bit for bit: the GDN state and the attention cache
/// start empty on every run. The observer sees every quantized object once, in the order the layer applies them, with
/// the rows of both sequences.
#[test]
#[ignore = "needs a GPU, the bundled NVRTC, the math libraries and the checkpoint"]
fn a_rerun_is_identical_and_the_observer_sees_each_object_once_in_order() {
    let mut rig = rig();
    let model = model();
    for index in [0usize, 3] {
        let layer = staged(&model, Target::Layer(index));
        let [first, second] = forward_twice(&mut rig, &model, &layer);
        assert_eq!(first.hidden, second.hidden, "layer {index}: a rerun differs");
        assert_ne!(
            first.hidden, model.input,
            "layer {index}: the hidden states did not change"
        );
        let names: Vec<&str> = first.observed.iter().map(|(name, _, _)| name.as_str()).collect();
        assert_eq!(names, model.objects.layer_objects(index).unwrap(), "layer {index}");
        assert!(
            first.observed.iter().all(|(_, rows, _)| *rows == 512),
            "layer {index}: rows"
        );
        assert_eq!(first.observed, second.observed);
    }
    rig.finish();
}
