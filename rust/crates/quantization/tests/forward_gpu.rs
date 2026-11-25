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
use euhedral_inference_cuda::KernelOps;
use euhedral_inference_quantization::checkpoint::objects::Qwen38Objects;
use euhedral_inference_quantization::checkpoint::safetensors::Safetensors;
use euhedral_inference_quantization::forward::{
    LaneContext, LayerForward, LayerWeights, Observation, Part, StagedLayer, Target, stage,
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

/// Runs `layer` over two sequences of 256 rows once per input in `inputs` on one `LayerForward`, each run's hidden
/// states and observations returned.
fn forward_runs(rig: &mut Rig, model: &Model, staged: &StagedLayer, inputs: &[&[u8]]) -> Vec<Run> {
    let (sequences, length) = (2u32, 256u32);
    let rows = sequences * length;
    let hidden = rig.upload(&model.input);
    let (library, pool) = (Arc::clone(&rig.library), Arc::clone(&rig.pool));
    let config = &model.objects.config;
    let (mut forward, weights) = rig.run(|context| {
        let forward = LayerForward::new(context.ops(), &library, &pool, config, rows)?;
        // SAFETY: `staged` is borrowed for the whole of `forward_runs`, and `rig.run` waits for the lane to retire
        // the copies before it returns.
        let weights = unsafe { LayerWeights::upload(context.ops(), staged, UPLOAD_CHUNK) }?;
        Ok((forward, weights))
    });
    let hidden_ptr = hidden.ptr();
    let mut runs = Vec::new();
    for input in inputs {
        rig.overwrite(&hidden, input);
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
    runs
}

fn forward_twice(rig: &mut Rig, model: &Model, staged: &StagedLayer) -> [Run; 2] {
    let runs = forward_runs(rig, model, staged, &[&model.input, &model.input]);
    runs.try_into().ok().unwrap()
}

/// `target` read into pinned memory from the rig's pool.
fn staged(rig: &Rig, model: &Model, target: Target) -> StagedLayer {
    let pool = Arc::clone(&rig.pool);
    drive(&model.io, |done| {
        stage(&model.objects, target, |bytes| pool.acquire(bytes), done)
    })
    .unwrap()
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
        let layer = staged(&rig, &model, Target::Layer(index));
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

/// Uploading with a chunk that divides none of the parts puts every part on the device exactly as it was staged: the
/// chunks of a part land at their own offsets, and the pieces of a multi-piece object follow one another.
#[test]
#[ignore = "needs a GPU, the bundled NVRTC, the math libraries and the checkpoint"]
fn an_upload_in_odd_chunks_puts_each_part_on_the_device_as_staged() {
    let mut rig = rig();
    let model = model();
    let pool = Arc::clone(&rig.pool);
    for index in [0usize, 3] {
        let layer = staged(&rig, &model, Target::Layer(index));
        // SAFETY: `layer` outlives the call, and `rig.run` waits for the lane to retire the copies.
        let weights = rig.run(|context| unsafe { LayerWeights::upload(context.ops(), &layer, 1_000_003) });
        let mut loans = Vec::new();
        for (part, bytes) in layer.parts() {
            loans.push((part, pool.acquire(bytes).unwrap(), weights.ptr(part).unwrap()));
        }
        rig.run(|context| {
            for (_, loan, source) in &mut loans {
                let bytes = loan.len();
                // SAFETY: the loans outlive the run, which returns once the lane has retired the copies.
                unsafe {
                    context
                        .ops()
                        .copy_device_to_readback(loan.buffer_mut(), 0, *source, bytes)
                }?;
            }
            Ok(())
        });
        for (part, loan, _) in &loans {
            assert!(
                loan.buffer().as_slice()[..loan.len()] == layer.bytes(part).unwrap()[..],
                "layer {index}: {part:?} differs after the upload"
            );
        }
        rig.run(|context| weights.release(context.ops()));
    }
    rig.finish();
}

/// A sequence's result does not depend on where it sits among the others: the sequences A, B and then B, A give the
/// same rows with their places swapped.
#[test]
#[ignore = "needs a GPU, the bundled NVRTC, the math libraries and the checkpoint"]
fn a_sequence_gives_the_same_rows_wherever_it_sits() {
    let mut rig = rig();
    let model = model();
    let half = model.input.len() / 2;
    let swapped = [&model.input[half..], &model.input[..half]].concat();
    assert_ne!(
        model.input[..half],
        model.input[half..],
        "the two sequences must differ"
    );
    for index in [0usize, 3] {
        let layer = staged(&rig, &model, Target::Layer(index));
        let [straight, crossed]: [Run; 2] = forward_runs(&mut rig, &model, &layer, &[&model.input, &swapped])
            .try_into()
            .ok()
            .unwrap();
        assert!(
            straight.hidden[..half] == crossed.hidden[half..] && straight.hidden[half..] == crossed.hidden[..half],
            "layer {index}: a sequence's rows depend on its place"
        );
        assert!(
            straight.hidden[..half] != straight.hidden[half..],
            "layer {index}: the sequences came out equal"
        );
    }
    rig.finish();
}

/// The MTP block's one fused query_key_gate_value object is two projections: the first half of its rows, then the
/// second. After the head, the projections are the normalized rows times each half, taken from the object's own
/// shape.
#[test]
#[ignore = "needs a GPU, the bundled NVRTC, the math libraries and the checkpoint"]
fn the_fused_mtp_object_is_two_projections_one_after_the_other() {
    let mut rig = rig();
    let model = model();
    let layer = staged(&rig, &model, Target::Mtp);
    let (library, pool) = (Arc::clone(&rig.library), Arc::clone(&rig.pool));
    let config = &model.objects.config;
    let rows = 8u32;
    let hidden = rig.upload(&model.input[..rows as usize * config.hidden as usize * 2]);
    let (mut forward, weights) = rig.run(|context| {
        let forward = LayerForward::new(context.ops(), &library, &pool, config, rows)?;
        // SAFETY: `layer` outlives the call, and `rig.run` waits for the lane to retire the copies.
        let weights = unsafe { LayerWeights::upload(context.ops(), &layer, UPLOAD_CHUNK) }?;
        Ok((forward, weights))
    });
    let name = "mtp/layer/attention/query_key_gate_value";
    let (fused_rows, columns) = weights.shapes().iter().find(|(held, _)| held == name).unwrap().1;
    let (width, k) = ((fused_rows / 2) as u32, columns as u32);
    let first_half = weights.object(name).unwrap();
    let second_half = first_half.offset(fused_rows / 2 * columns * 2);
    let bytes = rows as usize * width as usize * 2;
    let scratch = rig.upload(&vec![0u8; 3 * bytes + rows as usize * k as usize * 2]);
    let (normalized, want_first, want_second) = (
        scratch.ptr().offset(3 * bytes as u64),
        scratch.ptr(),
        scratch.ptr().offset(bytes as u64),
    );
    let (got_first, got_second) = forward.projections();
    let (got_first_loan, got_second_loan) = (pool.acquire(bytes).unwrap(), pool.acquire(bytes).unwrap());
    let (want_first_loan, want_second_loan) = (pool.acquire(bytes).unwrap(), pool.acquire(bytes).unwrap());
    let mut loans = [got_first_loan, got_second_loan, want_first_loan, want_second_loan];
    let hidden_ptr = hidden.ptr();
    rig.run(|context| {
        forward.head(context, &weights, hidden_ptr, rows, None)?;
        let norm = weights.ptr(&Part::InputNorm).unwrap();
        context
            .ops()
            .rms_norm_unit_offset_bf16(hidden_ptr, norm, normalized, rows, k, config.eps as f32)?;
        context.linear(normalized, first_half, want_first, rows, k, width)?;
        context.linear(normalized, second_half, want_second, rows, k, width)?;
        let sources = [got_first, got_second, want_first, want_second];
        for (loan, source) in loans.iter_mut().zip(sources) {
            // SAFETY: the loans outlive the run, which returns once the lane has retired the copies.
            unsafe {
                context
                    .ops()
                    .copy_device_to_readback(loan.buffer_mut(), 0, source, bytes)
            }?;
        }
        Ok(())
    });
    assert!(
        loans[0].buffer().as_slice()[..bytes] == loans[2].buffer().as_slice()[..bytes],
        "the first projection is not the first half"
    );
    assert!(
        loans[1].buffer().as_slice()[..bytes] == loans[3].buffer().as_slice()[..bytes],
        "the second projection is not the second half"
    );
    assert!(
        loans[2].buffer().as_slice()[..bytes] != loans[3].buffer().as_slice()[..bytes],
        "the halves came out equal"
    );
    rig.run(|context| {
        weights.release(context.ops())?;
        forward.release(context.ops())
    });
    rig.finish();
}
