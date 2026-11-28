//! Device tests of the drafter. `#[ignore]`d like the crate's other device tests:
//! `cargo test -p euhedral-inference-quantization --test drafter_gpu -- --ignored --nocapture`.
//!
//! The engine library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY`); the math libraries
//! from `euhedral.quantization.cuda-libraries`.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "a test: waits for the lane to retire the work, and copies synchronously to check results"
)]

mod support;

use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_gpu::GpuError;
use euhedral_gpu::cuda::CudaModule;
use euhedral_inference_cuda::DeviceBuffer;
use euhedral_inference_cuda::math::MathLibraries;
use euhedral_inference_quantization::kernels::Module;
use euhedral_inference_quantization::mtp::{
    Config, Drafter, DrafterKernels, Lineage, Shape, Window, Work, plan_lineage_dot, plan_lineage_gather,
    plan_lineage_spread, plan_nvfp4_round,
};
use euhedral_inference_quantization::training::plan_squared_error;
use support::{Device, device_with, gaussians};

const ROWS: u32 = 3;
const CONTEXT: u32 = 2;

fn config(depths: u32) -> Config {
    Config {
        shape: Shape {
            hidden: 32,
            intermediate: 32,
            heads: 2,
            kv_heads: 1,
            head_dim: 32,
            rotary_dim: 16,
            theta: 1e7,
            eps: 1e-6,
        },
        depths,
        rows: ROWS,
        max_keys: 8,
        max_positions: 16,
    }
}

struct Rig {
    device: Device,
    math: Arc<MathLibraries>,
    kernels: DrafterKernels<CudaModule>,
}

impl Deref for Rig {
    type Target = Device;

    fn deref(&self) -> &Device {
        &self.device
    }
}

fn rig() -> Rig {
    let directory = test_support::test_setting("euhedral.quantization.cuda-libraries")
        .expect("set euhedral.quantization.cuda-libraries (EUHEDRAL_TEST_QUANTIZATION_CUDA_LIBRARIES)");
    let math = MathLibraries::load(&PathBuf::from(directory)).expect("the math libraries load");
    let (device, compiled) = device_with(&[Module::Train, Module::Drafter], "drafter");
    let kernels = DrafterKernels::new(&compiled[0], &compiled[1]);
    Rig { device, math, kernels }
}

impl Rig {
    /// Runs `work` in order on one submission and returns when the lane has retired it.
    fn run(&self, work: &[Work]) {
        let mut handles = self.math.lane_handles().expect("handles");
        self.submit_and_wait(|queue| {
            // SAFETY: the work names live buffers of the test, which outlive the wait, on the lane's own handles.
            unsafe { self.kernels.enqueue(queue, &mut handles, work) }
                .map_err(|error| GpuError::Invalid(error.to_string()))
        });
        handles.close().expect("the handles close");
    }
}

/// A drafter with its arena, parameters from fixed seeds, and the buffers of one run.
struct Model {
    drafter: Drafter,
    arena: DeviceBuffer,
    inputs: Inputs,
    loss: DeviceBuffer,
    /// The output buffer of each depth, from the last forward pass.
    outputs: Vec<u64>,
}

/// The rows of a context and of a window, and the targets of the window's depths.
struct Inputs {
    context_hidden: DeviceBuffer,
    context_embedded: DeviceBuffer,
    hidden: DeviceBuffer,
    embedded: Vec<DeviceBuffer>,
    targets: Vec<DeviceBuffer>,
}

fn rows(rig: &Rig, count: u32, seed: u64) -> DeviceBuffer {
    rig.upload_f32(&gaussians(count as usize * 32, seed))
}

const CONTEXT_WINDOW: Window = Window {
    position: 0,
    first_key: 0,
    count: CONTEXT,
};
const WINDOW: Window = Window {
    position: CONTEXT as u64,
    first_key: 0,
    count: ROWS,
};

impl Model {
    fn new(rig: &Rig, depths: u32, rounded: bool) -> Self {
        let bytes = usize::try_from(Drafter::arena_bytes(&config(depths)).unwrap()).unwrap();
        let arena = rig.buffer(bytes);
        let mut drafter = Drafter::new(&config(depths), arena.addr()).unwrap();
        drafter.set_rounded(rounded);
        for (index, parameter) in drafter.parameters().iter().enumerate() {
            let scale = if parameter.rows == 1 {
                0.1
            } else {
                1.0 / (parameter.columns as f32).sqrt()
            };
            let values: Vec<f32> = gaussians(parameter.count() as usize, 100 + index as u64)
                .into_iter()
                .map(|value| value * scale)
                .collect();
            let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            arena
                .copy_from_host((parameter.values - arena.addr()) as usize, &bytes)
                .unwrap();
        }
        rig.run(&drafter.initialise().unwrap());
        rig.run(&drafter.measure().unwrap());
        let inputs = Inputs {
            context_hidden: rows(rig, CONTEXT, 1),
            context_embedded: rows(rig, CONTEXT, 2),
            hidden: rows(rig, ROWS, 3),
            embedded: (0..depths).map(|k| rows(rig, ROWS, 10 + u64::from(k))).collect(),
            targets: (0..depths).map(|k| rows(rig, ROWS, 20 + u64::from(k))).collect(),
        };
        Self {
            drafter,
            arena,
            inputs,
            loss: rig.buffer(8),
            outputs: Vec::new(),
        }
    }

    fn offset(&self, address: u64) -> usize {
        (address - self.arena.addr()) as usize
    }

    fn read(&self, address: u64, count: usize) -> Vec<f32> {
        let mut bytes = vec![0u8; 4 * count];
        self.arena.copy_to_host(self.offset(address), &mut bytes).unwrap();
        let (words, _) = bytes.as_chunks::<4>();
        words.iter().map(|word| f32::from_le_bytes(*word)).collect()
    }

    fn write(&self, address: u64, values: &[f32]) {
        let bytes: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
        self.arena.copy_from_host(self.offset(address), &bytes).unwrap();
    }

    /// The depth-1 forward over the context rows, which leaves their keys and values.
    fn context(&mut self, rig: &Rig) {
        let input = &self.inputs;
        let forward = self
            .drafter
            .forward(
                1,
                &CONTEXT_WINDOW,
                input.context_hidden.addr(),
                input.context_embedded.addr(),
            )
            .unwrap();
        rig.run(&forward.work);
    }

    /// The forward pass of every depth over the window; the outputs of each depth.
    fn forward(&mut self, rig: &Rig) -> Vec<Vec<f32>> {
        let mut input = self.inputs.hidden.addr();
        let mut outputs = Vec::new();
        self.outputs.clear();
        for depth in 1..=self.drafter.config().depths {
            let embedded = self.inputs.embedded[depth as usize - 1].addr();
            let forward = self.drafter.forward(depth, &WINDOW, input, embedded).unwrap();
            rig.run(&forward.work);
            outputs.push(self.read(forward.output, ROWS as usize * 32));
            input = forward.output;
            self.outputs.push(forward.output);
        }
        outputs
    }

    /// The summed squared error of the depths' outputs against their targets, with their gradients into the
    /// drafter's output gradients when `with_gradient`.
    fn loss(&self, rig: &Rig, with_gradient: bool) -> f64 {
        self.loss.copy_from_host(0, &[0u8; 8]).unwrap();
        let mut work = Vec::new();
        for depth in 1..=self.drafter.config().depths {
            let output = self.outputs[depth as usize - 1];
            let gradient = with_gradient.then(|| self.drafter.output_gradient(depth).unwrap());
            let target = self.inputs.targets[depth as usize - 1].addr();
            let count = u64::from(ROWS) * 32;
            work.push(Work::Steps(
                plan_squared_error(output, target, count, 1.0, gradient, self.loss.addr()).unwrap(),
            ));
        }
        rig.run(&work);
        let mut bytes = [0u8; 8];
        self.loss.copy_to_host(0, &mut bytes).unwrap();
        f64::from_le_bytes(bytes)
    }

    /// Forward, loss and backward over the window: the gradients of the loss.
    fn gradients(&mut self, rig: &Rig) {
        self.forward(rig);
        rig.run(&self.drafter.start_backward(&WINDOW).unwrap());
        self.loss(rig, true);
        for depth in (1..=self.drafter.config().depths).rev() {
            rig.run(&self.drafter.backward(depth, &WINDOW).unwrap());
        }
    }
}

#[test]
#[ignore = "needs the device"]
fn depth_one_of_a_deeper_drafter_equals_a_drafter_of_one_depth() {
    let rig = rig();
    for rounded in [false, true] {
        let mut deep = Model::new(&rig, 3, rounded);
        let mut single = Model::new(&rig, 1, rounded);
        deep.context(&rig);
        single.context(&rig);
        let deep_first = deep.forward(&rig).remove(0);
        let single_first = single.forward(&rig).remove(0);
        assert!(
            deep_first.iter().any(|v| *v != 0.0),
            "rounded {rounded}: an output of zeros proves nothing"
        );
        assert_eq!(deep_first, single_first, "rounded {rounded}");
    }
}

/// The loss of the window after its forward pass.
fn loss_of(model: &mut Model, rig: &Rig) -> f64 {
    model.forward(rig);
    model.loss(rig, false)
}

/// The relative error of the directional derivative of the loss along each parameter's own gradient, by central
/// differences, at depths 3; the largest of them and the parameter it is of.
#[allow(clippy::print_stdout, reason = "the measured errors are the result a run reports")]
fn directional_error(rig: &Rig) -> (f64, &'static str) {
    const STEP: f32 = 2e-3;
    let mut model = Model::new(rig, 3, false);
    model.context(rig);
    model.gradients(rig);
    let mut worst = (0.0f64, "");
    for parameter in model.drafter.parameters().to_vec() {
        let count = parameter.count() as usize;
        let gradient = model.read(parameter.gradient, count);
        let largest = gradient.iter().fold(0.0f32, |m, g| m.max(g.abs()));
        assert!(largest > 1e-4, "{}: a gradient of zeros proves nothing", parameter.name);
        // The step is along the gradient, scaled so that its largest element moves by STEP.
        let direction: Vec<f32> = gradient.iter().map(|g| g / largest).collect();
        let analytic: f64 = gradient
            .iter()
            .zip(&direction)
            .map(|(g, d)| f64::from(*g) * f64::from(*d))
            .sum();
        let values = model.read(parameter.values, count);
        let moved = |sign: f32| -> Vec<f32> {
            values
                .iter()
                .zip(&direction)
                .map(|(v, d)| v + sign * STEP * d)
                .collect()
        };
        model.write(parameter.values, &moved(1.0));
        let up = loss_of(&mut model, rig);
        model.write(parameter.values, &moved(-1.0));
        let down = loss_of(&mut model, rig);
        model.write(parameter.values, &values);
        let numeric = (up - down) / (2.0 * f64::from(STEP));
        let error = (numeric - analytic).abs() / analytic.abs();
        println!(
            "{}: analytic {analytic:.4} numeric {numeric:.4} error {error:.2e}",
            parameter.name
        );
        if error > worst.0 {
            worst = (error, parameter.name);
        }
    }
    worst
}

#[test]
#[ignore = "needs the device"]
fn gradients_match_finite_differences_of_the_loss() {
    let rig = rig();
    let (error, name) = directional_error(&rig);
    // A correct drafter is within 1.1e-3 on every parameter (FP32 round-off over the step); a skipped lineage dot in
    // the forward pass reaches 4.3e-1 and one in the backward pass 2.1e-1.
    assert!(error < 1e-2, "{name}: the directional derivative is off by {error}");
}

#[test]
#[ignore = "needs the device"]
fn rounded_gradients_are_the_straight_through_gradients_of_the_rounded_weights() {
    let rig = rig();
    let mut rounded = Model::new(&rig, 3, true);
    let mut plain = Model::new(&rig, 3, false);
    // The plain drafter holds the rounded drafter's weights as its values, so what it computes forward and back is what
    // the straight-through pass computes: the gradients with respect to the rounded weights.
    let mut moved = 0;
    for parameter in rounded
        .drafter
        .parameters()
        .to_vec()
        .iter()
        .filter(|p| p.maximum.is_some())
    {
        let count = parameter.count() as usize;
        let output = rig.buffer(4 * count);
        rig.run(&[Work::Steps(
            plan_nvfp4_round(
                parameter.values,
                output.addr(),
                None,
                parameter.rows,
                parameter.columns,
                parameter.maximum.unwrap(),
            )
            .unwrap(),
        )]);
        let weights = rig.download_f32(&output, count);
        moved += usize::from(weights != rounded.read(parameter.values, count));
        plain.write(plain.drafter.parameter(parameter.name).unwrap().values, &weights);
    }
    assert_eq!(moved, 5, "rounding changes every projection, or the test shows nothing");
    for model in [&mut rounded, &mut plain] {
        model.context(&rig);
        model.gradients(&rig);
    }
    for parameter in rounded.drafter.parameters().to_vec() {
        let count = parameter.count() as usize;
        let got = rounded.read(parameter.gradient, count);
        let expected = plain.read(plain.drafter.parameter(parameter.name).unwrap().gradient, count);
        let largest = expected.iter().fold(0.0f32, |m, g| m.max(g.abs()));
        assert!(largest > 1e-4, "{}: a gradient of zeros proves nothing", parameter.name);
        // The norms' gradients add with atomics, so their last bits depend on the order the blocks ran in.
        let tolerance = if parameter.maximum.is_some() { 0.0 } else { 1e-5 };
        let worst = got.iter().zip(&expected).fold(0.0f32, |m, (a, b)| m.max((a - b).abs()));
        assert!(
            worst <= tolerance * largest,
            "{}: differs by {worst} of {largest}",
            parameter.name
        );
    }
}

/// Lineage tensors `[heads, rows, stride]` and rows `[rows, width]` of the test below.
const HEADS: usize = 4;
const GROUP: usize = 2;
const D: usize = 32;
const STRIDE: usize = 3;
const SLOT: usize = 1;

#[test]
#[ignore = "needs the device"]
fn lineage_kernels_read_and_write_only_their_slot() {
    let rig = rig();
    let rows = ROWS as usize;
    let (query_width, key_width) = (HEADS * D, HEADS / GROUP * D);
    let lineage = Lineage {
        heads: HEADS as u32,
        group: GROUP as u32,
        d: D as u32,
        rows: ROWS,
        stride: STRIDE as u32,
        slot: SLOT as u32,
    };
    let queries = gaussians(rows * query_width, 1);
    let keys = gaussians(rows * key_width, 2);
    let tensor = gaussians(HEADS * rows * STRIDE, 3);
    let at = |h: usize, r: usize, slot: usize| (h * rows + r) * STRIDE + slot;
    let (queries_buffer, keys_buffer) = (rig.upload_f32(&queries), rig.upload_f32(&keys));

    // Dot: slot SLOT of the output gets the score, the others keep what they held.
    let output = rig.upload_f32(&tensor);
    rig.run(&[Work::Steps(
        plan_lineage_dot(
            (queries_buffer.addr(), query_width as u32),
            (keys_buffer.addr(), key_width as u32),
            output.addr(),
            &lineage,
        )
        .unwrap(),
    )]);
    let got = rig.download_f32(&output, tensor.len());
    for h in 0..HEADS {
        for r in 0..rows {
            let dot: f32 = (0..D)
                .map(|i| queries[r * query_width + h * D + i] * keys[r * key_width + h / GROUP * D + i])
                .sum();
            for slot in 0..STRIDE {
                let expected = if slot == SLOT { dot } else { tensor[at(h, r, slot)] };
                assert!(
                    (got[at(h, r, slot)] - expected).abs() <= 1e-5,
                    "dot h{h} r{r} slot{slot}"
                );
            }
        }
    }

    // Spread: y += c[slot] x over each head, reading nothing of the other slots.
    let target = gaussians(rows * query_width, 4);
    let spread = rig.upload_f32(&target);
    let coefficients = rig.upload_f32(&tensor);
    rig.run(&[Work::Steps(
        plan_lineage_spread(
            coefficients.addr(),
            (keys_buffer.addr(), key_width as u32),
            (spread.addr(), query_width as u32),
            &lineage,
        )
        .unwrap(),
    )]);
    let got = rig.download_f32(&spread, target.len());
    for r in 0..rows {
        for h in 0..HEADS {
            for i in 0..D {
                let expected = target[r * query_width + h * D + i]
                    + tensor[at(h, r, SLOT)] * keys[r * key_width + h / GROUP * D + i];
                assert!(
                    (got[r * query_width + h * D + i] - expected).abs() <= 1e-5,
                    "spread r{r} h{h} i{i}"
                );
            }
        }
    }

    // Gather: y[kv] += the group's sum of c[slot] x.
    let target = gaussians(rows * key_width, 5);
    let gathered = rig.upload_f32(&target);
    rig.run(&[Work::Steps(
        plan_lineage_gather(
            coefficients.addr(),
            (queries_buffer.addr(), query_width as u32),
            (gathered.addr(), key_width as u32),
            &lineage,
        )
        .unwrap(),
    )]);
    let got = rig.download_f32(&gathered, target.len());
    for r in 0..rows {
        for kv in 0..HEADS / GROUP {
            for i in 0..D {
                let sum: f32 = (kv * GROUP..(kv + 1) * GROUP)
                    .map(|h| tensor[at(h, r, SLOT)] * queries[r * query_width + h * D + i])
                    .sum();
                let expected = target[r * key_width + kv * D + i] + sum;
                assert!(
                    (got[r * key_width + kv * D + i] - expected).abs() <= 1e-5,
                    "gather r{r} kv{kv} i{i}"
                );
            }
        }
    }
}
