//! The device rig the trellis device tests share: the engine library, the compiled trellis module, one lane, and
//! helpers to move values and wait for the lane.
#![allow(dead_code, reason = "each test file uses some of the helpers")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "a test: waits for the lane to retire the work, and copies synchronously to check results"
)]

use std::f64::consts::TAU;
use std::path::PathBuf;
use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaModule, CudaQueue};
use euhedral_gpu::{GpuError, Lane, SharedLanes, StreamOptions, open_shared_lanes};
use euhedral_graph::LaneId;
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer};
use euhedral_inference_quantization::kernels::{Module, ModuleOptions, Modules};
use euhedral_inference_quantization::trellis::{Step, TrellisError, TrellisKernels};

pub struct Rig {
    pub library: Arc<CudaLibrary>,
    pub kernels: TrellisKernels<CudaModule>,
    lanes: SharedLanes<CudaQueue>,
    lane: Arc<CudaQueue>,
    // The device stays open for the lanes.
    _gpu: Arc<CudaGpu>,
}

/// The rig, or a panic naming the missing setting: a device test that skips is no check.
pub fn rig() -> Rig {
    let path = test_support::gradle_property("euhedral.cuda.library")
        .expect("set euhedral.cuda.library (EUHEDRAL_TEST_CUDA_LIBRARY) to the engine library");
    let library = CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads");
    let options = ModuleOptions::beside(&library).expect("the product directory");
    let gpu = CudaGpu::open(0).expect("the device opens");
    let mut modules = Modules::<CudaModule>::open(&gpu, options).expect("the compiler gives machine code");
    let compiled = modules.load(Module::Trellis).unwrap_or_else(|error| panic!("{error}"));
    let kernels = TrellisKernels::new(&compiled).expect("the trellis kernels resolve");
    let lanes = open_shared_lanes(&gpu, &[("trellis", StreamOptions::default())]).expect("a lane opens");
    let lane = Arc::clone(lanes.pool.lane(LaneId(0)));
    Rig {
        library,
        kernels,
        lanes,
        lane,
        _gpu: gpu,
    }
}

impl Rig {
    /// Runs `steps` on the lane and returns when the lane has retired them.
    pub fn run(&self, steps: &[Step]) {
        let ((), marker) = self
            .lane
            .submit_marked(|queue| {
                // SAFETY: the steps name live buffers of the test, which outlive the wait below.
                unsafe { self.kernels.enqueue(queue, steps) }.map_err(|error| match error {
                    TrellisError::Gpu(gpu) => gpu,
                    other => GpuError::Invalid(other.to_string()),
                })
            })
            .expect("submit");
        let start = Instant::now();
        while !self.lanes.queues.reached(marker) {
            assert!(start.elapsed() < Duration::from_secs(60), "the lane did not retire");
            sleep(Duration::from_micros(100));
        }
    }

    pub fn buffer(&self, bytes: usize) -> DeviceBuffer {
        let buffer = DeviceBuffer::alloc(&self.library, bytes.max(1)).expect("device allocation");
        buffer.copy_from_host(0, &vec![0u8; bytes]).expect("clear");
        buffer
    }

    pub fn upload(&self, bytes: &[u8]) -> DeviceBuffer {
        let buffer = self.buffer(bytes.len());
        buffer.copy_from_host(0, bytes).expect("upload");
        buffer
    }

    pub fn upload_f32(&self, values: &[f32]) -> DeviceBuffer {
        self.upload(&values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
    }

    pub fn download(&self, buffer: &DeviceBuffer, bytes: usize) -> Vec<u8> {
        let mut host = vec![0u8; bytes];
        buffer.copy_to_host(0, &mut host).expect("download");
        host
    }

    pub fn download_f32(&self, buffer: &DeviceBuffer, count: usize) -> Vec<f32> {
        let bytes = self.download(buffer, count * 4);
        let (words, _) = bytes.as_chunks::<4>();
        words.iter().map(|word| f32::from_le_bytes(*word)).collect()
    }
}

/// Deterministic standard-normal values.
pub fn gaussians(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    let mut uniform = move || {
        state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((state >> 11) as f64 + 0.5) / (1u64 << 53) as f64
    };
    (0..count)
        .map(|_| ((-2.0 * uniform().ln()).sqrt() * (TAU * uniform()).cos()) as f32)
        .collect()
}
