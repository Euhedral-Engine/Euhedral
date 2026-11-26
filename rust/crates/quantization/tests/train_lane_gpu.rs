//! Device test of the training lane: a window and an optimiser step through the lane's own staging and chains equal the
//! same planned work run directly on a second lane, gradients and values included; and the draft head decoded on the
//! lane equals a host decode. `#[ignore]`d like the crate's other
//! device tests:
//! `cargo test -p euhedral-inference-quantization --test train_lane_gpu -- --ignored --nocapture`.
//!
//! The engine library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY`), the math libraries
//! from `euhedral.quantization.cuda-libraries`; without them the test fails: a device test that skips is no check.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test file: the pump is the worker, and a failed unwrap is the failure"
)]

mod support;

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::generics::LatticeSource;
use euhedral_gpu::cuda::{CudaGpu, CudaModule, CudaStream};
use euhedral_gpu::{Gpu, StreamOptions};
use euhedral_graph::{Completions, LanePool};
use euhedral_inference_cuda::math::MathLibraries;
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, HostWeightsArena};
use euhedral_inference_quantization::kernels::{Compiled, Module, ModuleOptions, Modules};
use euhedral_inference_quantization::mtp::train::lane::{Finished, Job, Lane, PackedHead, Resources, TrainLane};
use euhedral_inference_quantization::mtp::train::load::Keep;
use euhedral_inference_quantization::mtp::train::progress::Update;
use euhedral_inference_quantization::mtp::train::settings::Window;
use euhedral_inference_quantization::mtp::train::window::{Addresses, Mode, plan};
use euhedral_inference_quantization::mtp::{Config, Drafter, DrafterKernels, Shape};
use support::gaussians;

const HIDDEN: u32 = 32;
const DEPTHS: u32 = 2;
const ROWS: u32 = 6;
const DRAFT_ROWS: u32 = 16;
const VOCABULARY: u32 = 12;
const COUNT: u32 = 5;
/// What unwritten device memory holds: a NaN in every float format.
const POISON: u8 = 0xff;
const WINDOW: Window = Window {
    a: 0,
    count: COUNT,
    first_key: 0,
    labelled: true,
};
const UPDATE: Update = Update {
    step: 1,
    projection_rate: 1e-3,
    norm_rate: 1e-3,
};

fn config() -> Config {
    Config {
        shape: Shape {
            hidden: HIDDEN,
            intermediate: 32,
            heads: 2,
            kv_heads: 1,
            head_dim: 32,
            rotary_dim: 16,
            theta: 1e7,
            eps: 1e-6,
        },
        depths: DEPTHS,
        rows: ROWS,
        max_keys: 8,
        max_positions: 16,
    }
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn float_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// Everything a lane and the host planning of its work are made of, with the same contents in every call.
struct Model {
    lane: TrainLane,
    drafter: Drafter,
    addresses: Addresses,
}

struct Rig {
    gpu: Arc<CudaGpu>,
    pool: Arc<LanePool>,
    library: Arc<CudaLibrary>,
    math: Arc<MathLibraries>,
    train: Arc<Compiled<CudaModule>>,
    drafter: Arc<Compiled<CudaModule>>,
    completions: Arc<Completions>,
    streams: Vec<CudaStream>,
}

fn take(library: &Arc<CudaLibrary>, bytes: usize, contents: &[u8]) -> DeviceBuffer {
    let buffer = DeviceBuffer::alloc(library, bytes.max(1)).unwrap();
    // NaN in every float format, so that a read of what nothing wrote shows in the results. Only the drafter's
    // gradients and moments start at zero, and `prepare` writes those.
    buffer.copy_from_host(0, &vec![POISON; bytes.max(1)]).unwrap();
    if !contents.is_empty() {
        buffer.copy_from_host(0, contents).unwrap();
    }
    buffer
}

impl Rig {
    /// A model; `preload` writes a job's inputs straight into its buffers, for the run that does not stage them.
    fn model(&mut self, preload: Option<&Job>) -> Model {
        let library = &self.library;
        let config = config();
        let arena = take(
            library,
            usize::try_from(Drafter::arena_bytes(&config).unwrap()).unwrap(),
            &[],
        );
        let drafter = Drafter::new(&config, arena.addr()).unwrap();
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
            arena
                .copy_from_host((parameter.values - arena.addr()) as usize, &float_bytes(&values))
                .unwrap();
        }
        // A Q3 table of `VOCABULARY` rows: 48 bytes of codes a row, then two FP16 scales a row.
        let codes = VOCABULARY as usize * 48;
        let mut table = HostWeightsArena::alloc(library, codes + VOCABULARY as usize * 4).unwrap();
        for (at, byte) in table.as_mut_slice().iter_mut().enumerate() {
            *byte = if at < codes {
                (at * 37 % 251) as u8
            } else if at % 2 == 0 {
                0x00
            } else {
                0x38
            };
        }
        // SAFETY: the table is kept by the lane for as long as the addresses are used, and nothing writes it after this.
        let table_address = unsafe { table.device_address() }.unwrap().get();
        let head = take(
            library,
            2 * (DRAFT_ROWS * HIDDEN) as usize,
            &bf16_bytes(&gaussians((DRAFT_ROWS * HIDDEN) as usize, 7)),
        );
        let (rows, depths, wide) = (u64::from(ROWS), u64::from(DEPTHS), u64::from(HIDDEN));
        let mut owned = vec![];
        let mut make_with = |bytes: u64, contents: &[u8]| {
            let buffer = take(library, bytes as usize, contents);
            let address = buffer.addr();
            owned.push(buffer);
            address
        };
        let words = |values: &[i32]| -> Vec<u8> { values.iter().flat_map(|value| value.to_le_bytes()).collect() };
        let (hidden_bytes, id_bytes, label_bytes) = match preload {
            Some(job) => (job.hidden.clone(), words(&job.ids), words(&job.labels)),
            None => Default::default(),
        };
        let head_rows = 128;
        let addresses = Addresses {
            hidden: HIDDEN,
            draft_rows: DRAFT_ROWS,
            depths: DEPTHS,
            scale: 1.0 / 16.0,
            head: head.addr(),
            table: table_address,
            scale_offset: codes as u64,
            hidden_bf16: make_with(2 * rows * wide, &hidden_bytes),
            hidden_input: make_with(4 * rows * wide, &[]),
            ids: make_with(4 * rows * depths, &id_bytes),
            labels: make_with(4 * rows * depths, &label_bytes),
            predicted: make_with(4 * rows * depths, &[]),
            label_log_probabilities: make_with(4 * rows * depths, &[]),
            loss: make_with(8 * depths, &[]),
            embedded: (0..DEPTHS).map(|_| make_with(4 * rows * wide, &[])).collect(),
            head_input: make_with(2 * rows * wide, &[]),
            logits: make_with(4 * head_rows * u64::from(DRAFT_ROWS), &[]),
            logits_bf16: make_with(2 * head_rows * u64::from(DRAFT_ROWS), &[]),
        };
        owned.push(head);
        let kernels = DrafterKernels::new(&self.train, &self.drafter);
        let handles = self
            .math
            .lane_handles()
            .unwrap_or_else(|failed| panic!("{}", failed.error));
        let resources = Resources {
            kernels,
            handles,
            drafter: Drafter::new(&config, arena.addr()).unwrap(),
            addresses: addresses.clone(),
            rows: ROWS,
            arena,
            buffers: owned,
            table,
            completions: Arc::clone(&self.completions),
            keep: Keep::new(
                Arc::clone(&self.gpu),
                Arc::clone(&self.pool),
                Arc::clone(&self.math),
                Arc::clone(&self.library),
            ),
            poison: Some(POISON),
        };
        // The copies above come from pageable memory and may return before they land; the lane's stream does not wait
        // for them, so the device is waited for before the lane first runs.
        library.synchronize().unwrap();
        let mut lane = TrainLane::new(self.streams.remove(0), library, resources).unwrap();
        // The lane's own drafter has its parameters measured and its gradients zeroed; so has the planning copy.
        lane.prepare(true).unwrap();
        let mut planning = Drafter::new(&config, lane.drafter().parameters()[0].values).unwrap();
        planning.initialise().unwrap();
        planning.measure().unwrap();
        Model {
            lane,
            drafter: planning,
            addresses,
        }
    }

    /// Runs `start` and pumps the lattice until its `done` has run.
    fn wait<T: Send + 'static>(
        &self,
        start: impl FnOnce(Box<dyn FnOnce(Finished<TrainLane, T>) + Send>),
    ) -> (TrainLane, T) {
        let slot: Arc<Mutex<Option<Finished<TrainLane, T>>>> = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&slot);
        start(Box::new(move |finished| *sink.lock().unwrap() = Some(finished)));
        let deadline = Instant::now() + Duration::from_secs(60);
        loop {
            self.completions.reap();
            self.completions.pull(
                &mut |mut frame| {
                    frame.execute().unwrap();
                    frame.do_finally();
                },
                &mut |_| false,
                u64::MAX,
            );
            if let Some(finished) = slot.lock().unwrap().take() {
                let value = finished
                    .result
                    .unwrap_or_else(|error| panic!("the call failed: {error}"));
                return (finished.lane, value);
            }
            assert!(
                Instant::now() < deadline,
                "the call never finished: its retirement did not arrive"
            );
        }
    }

    fn read(&self, lane: TrainLane, address: u64, bytes: usize) -> (TrainLane, Vec<u8>) {
        self.wait(|done| {
            lane.read(address, bytes, done);
        })
    }
}

fn job() -> Job {
    let hidden = bf16_bytes(&gaussians((COUNT * HIDDEN) as usize, 3));
    let ids = (0..DEPTHS * COUNT).map(|at| (at * 5 % VOCABULARY) as i32).collect();
    let labels = (0..DEPTHS * COUNT).map(|at| (at * 7 % DRAFT_ROWS) as i32).collect();
    Job {
        window: WINDOW,
        mode: Mode::Train,
        hidden,
        ids,
        labels,
    }
}

/// Every parameter's values, then its gradient, as floats.
fn snapshot(rig: &Rig, mut lane: TrainLane) -> (TrainLane, Vec<f32>) {
    let mut floats = Vec::new();
    for parameter in lane.drafter().parameters().to_vec() {
        for address in [parameter.values, parameter.gradient] {
            let (back, bytes) = rig.read(lane, address, 4 * parameter.count() as usize);
            lane = back;
            floats.extend(bytes.as_chunks::<4>().0.iter().map(|word| f32::from_le_bytes(*word)));
        }
    }
    (lane, floats)
}

fn close(label: &str, actual: &[f32], wanted: &[f32]) {
    assert_eq!(actual.len(), wanted.len(), "{label}: the lengths differ");
    for (at, (a, w)) in actual.iter().zip(wanted).enumerate() {
        // Atomic additions in the norm gradients make the last bits depend on the order.
        assert!(
            (a - w).abs() <= 1e-5 * w.abs().max(1e-3),
            "{label}[{at}]: the lane gave {a}, the direct run {w}"
        );
    }
}

/// The engine library, the math libraries, the device and two lanes' streams.
fn rig() -> Rig {
    let path = test_support::gradle_property("euhedral.cuda.library")
        .expect("set euhedral.cuda.library (EUHEDRAL_TEST_CUDA_LIBRARY) to the engine library");
    let library = CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads");
    let directory = test_support::gradle_property("euhedral.quantization.cuda-libraries")
        .expect("set euhedral.quantization.cuda-libraries (EUHEDRAL_TEST_QUANTIZATION_CUDA_LIBRARIES)");
    let math = MathLibraries::load(&PathBuf::from(directory)).expect("the math libraries load");
    let gpu = CudaGpu::open(0).expect("the device opens");
    let options = ModuleOptions::beside(&library).expect("the product directory");
    let mut modules = Modules::<CudaModule>::open(&gpu, options).expect("the compiler gives machine code");
    let train = modules.load(Module::Train).unwrap();
    let drafter_module = modules.load(Module::Drafter).unwrap();
    let (pool, streams) = gpu
        .open_lanes(&["lane", "direct"], StreamOptions::default())
        .expect("two lanes open");
    let completions = Completions::new(&pool);
    Rig {
        gpu,
        pool,
        library,
        math,
        train,
        drafter: drafter_module,
        completions,
        streams,
    }
}

#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn a_window_and_a_step_through_the_lane_equal_the_same_work_run_directly() {
    let mut rig = rig();
    let through = rig.model(None);
    let direct = rig.model(Some(&job()));

    // Through the lane: the window, then the optimiser step.
    let (lane, outcome) = rig.wait(|done| {
        through.lane.window(job(), done);
    });
    let (lane, after_window) = snapshot(&rig, lane);
    let (lane, ()) = rig.wait(|done| lane.update(UPDATE, done));
    let (lane, after_step) = snapshot(&rig, lane);

    // Directly: the same plan run on the second lane's stream, read the same way.
    let Model {
        lane: mut direct_lane,
        mut drafter,
        addresses,
    } = direct;
    direct_lane
        .run_blocking(&plan(&mut drafter, &addresses, &WINDOW, Mode::Train).unwrap())
        .unwrap();
    let (back, predicted) = rig.read(direct_lane, addresses.predicted, 4 * (DEPTHS * COUNT) as usize);
    let (back, losses) = rig.read(back, addresses.loss, 8 * DEPTHS as usize);
    let (back, direct_window) = snapshot(&rig, back);
    assert_eq!(
        outcome.predicted,
        predicted
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| i32::from_le_bytes(*word))
            .collect::<Vec<_>>(),
        "the argmaxes"
    );
    let direct_losses: Vec<f64> = losses
        .as_chunks::<8>()
        .0
        .iter()
        .map(|word| f64::from_le_bytes(*word))
        .collect();
    assert_eq!(outcome.losses.len(), direct_losses.len());
    for (depth, (a, w)) in outcome.losses.iter().zip(&direct_losses).enumerate() {
        assert!(
            (a - w).abs() <= 1e-6 * w.abs().max(1.0),
            "depth {depth}: loss {a} against {w}"
        );
    }
    close("after the window", &after_window, &direct_window);

    let mut direct_lane = back;
    let step = drafter
        .step(UPDATE.projection_rate, UPDATE.norm_rate, UPDATE.step)
        .unwrap();
    direct_lane.run_blocking(&step).unwrap();
    let (direct_lane, direct_step) = snapshot(&rig, direct_lane);
    close("after the step", &after_step, &direct_step);
    direct_lane.close().unwrap();
    lane.close().unwrap();
}

/// The decode of `rows` NVFP4 rows of 40 values (padded to 128), on the host. The scale codes are E4M3 values that are
/// powers of two times 1 or 1.5, and the global scale is 0.5, so every decoded value is exact in BF16.
fn host_decode(bytes: &[u8], head: &PackedHead) -> Vec<u8> {
    const E2M1: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
    let e4m3 = |code: u8| {
        let (exponent, mantissa) = (i32::from(code >> 3 & 15), f32::from(code & 7));
        2f32.powi(exponent - 7) * (1.0 + mantissa / 8.0)
    };
    let global = f32::from_le_bytes(bytes[head.global_offset as usize..][..4].try_into().unwrap());
    let (k, k_pad) = (head.k as usize, head.k_pad as usize);
    let mut out = Vec::new();
    for row in 0..head.rows as usize {
        for column in 0..k {
            let byte = bytes[row * k_pad / 2 + column / 2];
            let code = if column % 2 == 1 { byte >> 4 } else { byte & 15 };
            let scale = e4m3(bytes[head.scale_offset as usize + row * k_pad / 16 + column / 16]) * global;
            let sign = if code & 8 == 0 { 1.0 } else { -1.0 };
            let value = E2M1[usize::from(code & 7)] * sign * scale;
            out.extend_from_slice(&(value.to_bits() >> 16).to_le_bytes()[..2]);
        }
    }
    out
}

#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn the_head_decoded_on_the_lane_equals_a_host_decode_into_poisoned_memory() {
    let mut rig = rig();
    let library = Arc::clone(&rig.library);
    let model = rig.model(None);
    let (rows, k, k_pad) = (5u32, 40u32, 128u32);
    let codes = (rows * k_pad / 2) as usize;
    let scale_offset = codes.next_multiple_of(256);
    let global_offset = (scale_offset + (rows * k_pad / 16) as usize).next_multiple_of(256);
    let mut bytes = vec![0u8; global_offset + 4];
    for (at, byte) in bytes[..codes].iter_mut().enumerate() {
        *byte = (at * 37 % 251) as u8;
    }
    for (at, byte) in bytes[scale_offset..global_offset].iter_mut().enumerate() {
        *byte = [0x30, 0x38, 0x40, 0x44][at % 4];
    }
    bytes[global_offset..].copy_from_slice(&0.5f32.to_le_bytes());
    let head = PackedHead {
        bytes,
        rows,
        k,
        k_pad,
        scale_offset: scale_offset as u64,
        global_offset: global_offset as u64,
    };
    let wanted = host_decode(&head.bytes, &head);
    // Never written but by the decode, so that a decode that did not run, or ran before its input landed, shows.
    let destination = take(&library, wanted.len(), &[]);
    library.synchronize().unwrap();
    let Model { mut lane, .. } = model;
    lane.decode_head(&head, destination.addr()).unwrap();
    let (lane, decoded) = rig.read(lane, destination.addr(), wanted.len());
    assert_eq!(decoded, wanted, "the decoded head");
    lane.close().unwrap();
}
