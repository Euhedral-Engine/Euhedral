//! The device rig of the layer forward tests: the engine library and kernels, the math libraries, one lane with its
//! cuBLAS handles, and helpers to move values and wait for the lane.
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
use euhedral_graph::{DeviceQueue, LaneId, Marker};
use euhedral_inference_cuda::launch::Modes;
use euhedral_inference_cuda::launch::load::{KernelSources, KernelTable, load};
use euhedral_inference_cuda::launch::ops::RusteeOps;
use euhedral_inference_cuda::math::{LaneHandles, MathLibraries};
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, PinnedPool};
use euhedral_inference_quantization::forward::{ForwardError, ForwardKernels, LaneContext};
use euhedral_inference_quantization::kernels::{Module, ModuleOptions, Modules};

pub struct Rig {
    pub library: Arc<CudaLibrary>,
    pub pool: Arc<PinnedPool>,
    table: KernelTable,
    kernels: ForwardKernels,
    handles: Option<LaneHandles>,
    lanes: SharedLanes<CudaQueue>,
    lane: Arc<CudaQueue>,
    // The device stays open for the lanes.
    _gpu: Arc<CudaGpu>,
}

/// The rig, or a panic naming the missing setting: a device test that skips is no check.
pub fn rig() -> Rig {
    let path = test_support::gradle_property("euhedral.cuda.library")
        .expect("set euhedral.cuda.library (EUHEDRAL_TEST_CUDA_LIBRARY) to the engine library");
    let math_directory = test_support::gradle_property("euhedral.quantization.cuda-libraries").expect(
        "set euhedral.quantization.cuda-libraries (EUHEDRAL_TEST_QUANTIZATION_CUDA_LIBRARIES) to the math libraries",
    );
    let library = CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads");
    let math = MathLibraries::load(&PathBuf::from(math_directory)).expect("the math libraries load");
    let gpu = CudaGpu::open(0).expect("the device opens");
    let table = load(&gpu, &KernelSources::beside(&library).expect("the kernel sources")).expect("the kernels load");
    let mut modules =
        Modules::<CudaModule>::open(&gpu, ModuleOptions::beside(&library).expect("the product directory"))
            .expect("the compiler gives machine code");
    let gptq = modules.load(Module::Gptq).unwrap_or_else(|error| panic!("{error}"));
    let rows = modules.load(Module::Rows).unwrap_or_else(|error| panic!("{error}"));
    let kernels = ForwardKernels::new(&gptq, &rows).expect("the forward kernels resolve");
    let lanes = open_shared_lanes(&gpu, &[("forward", StreamOptions::default())]).expect("a lane opens");
    let lane = Arc::clone(lanes.pool.lane(LaneId(0)));
    let handles = math.lane_handles().expect("the lane's handles");
    Rig {
        pool: PinnedPool::new(&library, 1 << 30),
        library,
        table,
        kernels,
        handles: Some(handles),
        lanes,
        lane,
        _gpu: gpu,
    }
}

impl Rig {
    /// Waits until the lane has retired `marker`.
    pub fn retired(&self, marker: Marker) {
        let start = Instant::now();
        while !self.lanes.queues.reached(marker) {
            assert!(start.elapsed() < Duration::from_secs(120), "the lane did not retire");
            sleep(Duration::from_micros(100));
        }
    }

    /// Runs `work` as one submission on the lane, with the engine's kernels and the lane's handles, and returns
    /// once the lane has retired everything it queued.
    pub fn run<R>(
        &mut self,
        work: impl for<'q, 'c> FnOnce(&mut LaneContext<'c, <CudaQueue as Lane>::Enqueue<'q>>) -> Result<R, ForwardError>,
    ) -> R {
        let lane = Arc::clone(&self.lane);
        let home = Marker::new(DeviceQueue::id(&*lane), Lane::lane(&*lane), 0);
        let (table, kernels, library) = (&self.table, &self.kernels, &self.library);
        let handles = self.handles.as_mut().expect("handles");
        let (result, marker) = lane
            .submit_marked(|queue| {
                let ops = RusteeOps::new(queue, table, Modes::default()).on_lane(library, home);
                let mut context = LaneContext::new(ops, handles, kernels);
                work(&mut context).map_err(|error| GpuError::Invalid(error.to_string()))
            })
            .expect("the work is submitted");
        self.retired(marker);
        result
    }

    /// Closes the lane's handles.
    pub fn finish(mut self) {
        self.handles
            .take()
            .expect("handles")
            .close()
            .expect("the handles close");
    }

    pub fn upload(&self, bytes: &[u8]) -> DeviceBuffer {
        let buffer = DeviceBuffer::alloc(&self.library, bytes.len().max(1)).expect("device allocation");
        buffer.copy_from_host(0, bytes).expect("upload");
        buffer
    }

    pub fn download(&self, buffer: &DeviceBuffer, bytes: usize) -> Vec<u8> {
        let mut host = vec![0u8; bytes];
        buffer.copy_to_host(0, &mut host).expect("download");
        host
    }
}

/// The data of an `.npy` file of 16-bit values: the bytes after the header.
pub fn npy_payload(bytes: &[u8]) -> &[u8] {
    let header = usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
    &bytes[10 + header..]
}

pub fn bf16_to_f32(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
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
