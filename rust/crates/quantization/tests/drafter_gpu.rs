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
use euhedral_inference_quantization::mtp::{Config, Drafter, DrafterKernels, Shape, Window, Work};
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
    let directory = test_support::gradle_property("euhedral.quantization.cuda-libraries")
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

    fn poke(&self, address: u64, value: f32) {
        self.arena
            .copy_from_host(self.offset(address), &value.to_le_bytes())
            .unwrap();
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

#[test]
#[ignore = "needs the device"]
fn gradients_match_finite_differences_of_the_loss() {
    const STEP: f32 = 1e-2;
    let rig = rig();
    let mut model = Model::new(&rig, 2, false);
    model.context(&rig);
    model.gradients(&rig);
    let parameters = model.drafter.parameters().to_vec();
    for parameter in &parameters {
        let gradient = model.read(parameter.gradient, parameter.count() as usize);
        let largest = gradient.iter().fold(0.0f32, |m, g| m.max(g.abs()));
        assert!(largest > 1e-4, "{}: a gradient of zeros proves nothing", parameter.name);
        let mut indices: Vec<usize> = (0..gradient.len()).collect();
        indices.sort_by(|a, b| gradient[*b].abs().total_cmp(&gradient[*a].abs()));
        for index in indices.into_iter().take(3) {
            let address = parameter.values + 4 * index as u64;
            let value = model.read(address, 1)[0];
            model.poke(address, value + STEP);
            model.forward(&rig);
            let up = model.loss(&rig, false);
            model.poke(address, value - STEP);
            model.forward(&rig);
            let down = model.loss(&rig, false);
            model.poke(address, value);
            let numeric = (up - down) / (2.0 * f64::from(STEP));
            let error = (numeric - f64::from(gradient[index])).abs() / f64::from(largest);
            assert!(
                error < 0.03,
                "{}[{index}]: analytic {} numeric {numeric} (error {error} of the largest {largest})",
                parameter.name,
                gradient[index]
            );
        }
    }
}
