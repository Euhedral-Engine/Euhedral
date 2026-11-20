//! The device rig of the GPTQ tests: one lane, the module and the math libraries, and a pump that stands in for the
//! lattice's worker, delivering the lane's retirements until a chain's `done` call has run.
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: the pump is the worker, and a failed unwrap is the failure"
)]
#![allow(dead_code, reason = "each test file uses part of the rig")]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::generics::LatticeSource;
use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_gpu::{Gpu, Stream, StreamOptions};
use euhedral_graph::{Completions, LanePool};
use euhedral_inference_cuda::math::MathLibraries;
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, PinnedPool};
use euhedral_inference_quantization::gptq::format::Format;
use euhedral_inference_quantization::gptq::{
    DeviceRange, Factor, Finished, GptqError, GptqKernels, GptqLane, GptqSettings, Refused, TensorScales,
};
use euhedral_inference_quantization::gptq::{RoundRequest, state::QuantState};
use euhedral_inference_quantization::kernels::{ModuleOptions, Modules};
use euhedral_inference_quantization::memory::{Claims, LaneMemory, StepBytes};
use euhedral_memory::{LocationId, LocationKind, LocationSpec, OwnerId, OwnerSpec, Planner, Priority, Scope};

pub struct Rig {
    pub library: Arc<CudaLibrary>,
    pub completions: Arc<Completions>,
    /// The pool the lane's pinned staging comes from.
    pub pinned: Arc<PinnedPool>,
    lane: Option<GptqLane>,
    planner: Arc<Planner>,
    owner: OwnerId,
    /// Bytes the next claims are short by, device and pinned: a step must refuse a claim too small for it.
    pub short: (u64, u64),
    // The device and the lane pool stay open for the lane.
    _gpu: Arc<CudaGpu>,
    _pool: Arc<LanePool>,
}

/// Bytes each planner location holds: more than any test asks for.
const CAPACITY: u64 = 8 << 30;
const DEVICE: LocationId = LocationId(0);
const PINNED: LocationId = LocationId(1);

/// The rig, or a panic naming the missing setting: a device test that skips is no check.
pub fn rig(settings: GptqSettings) -> Rig {
    let library = test_support::gradle_property("euhedral.cuda.library")
        .expect("set euhedral.cuda.library (EUHEDRAL_TEST_CUDA_LIBRARY)");
    let directory = test_support::gradle_property("euhedral.quantization.cuda-libraries")
        .expect("set euhedral.quantization.cuda-libraries (EUHEDRAL_TEST_QUANTIZATION_CUDA_LIBRARIES)");
    let library = CudaLibrary::load(&PathBuf::from(library)).expect("the CUDA library loads");
    let math = MathLibraries::load(&PathBuf::from(directory)).expect("the math libraries load");
    let gpu = CudaGpu::open(0).expect("the device opens");
    let options = ModuleOptions::beside(&library).expect("the product directory");
    let mut modules = Modules::<CudaModule>::open(&gpu, options).expect("the compiler gives machine code");
    let kernels = GptqKernels::load(&mut modules).expect("gptq.cu compiles");
    let (pool, mut streams) = gpu
        .open_lanes(&["gptq"], StreamOptions::default())
        .expect("a lane opens");
    let mut stream = streams.remove(0);
    let completions = Completions::new(&pool);
    let mut handles = None;
    Stream::submit(&mut stream, |_| {
        handles = Some(math.lane_handles());
        Ok(())
    })
    .expect("the handles are made on the lane");
    let handles = handles.unwrap().expect("the math handles");
    let pinned = PinnedPool::new(&library, 256 << 20);
    let memory = LaneMemory::new(Arc::clone(&library), Arc::clone(&pinned));
    let lane = GptqLane::new(stream, handles, kernels, memory, Arc::clone(&completions), settings).expect("the lane");
    let planner = Planner::new([
        LocationSpec {
            kind: LocationKind::Device { ordinal: 0 },
            capacity: CAPACITY,
            reserve: 0,
        },
        LocationSpec {
            kind: LocationKind::HostPinned,
            capacity: CAPACITY,
            reserve: 0,
        },
    ]);
    let owner = planner.open_owner(OwnerSpec::new("gptq-test", Priority(0), Scope::Process));
    Rig {
        library,
        completions,
        pinned,
        lane: Some(lane),
        planner,
        owner,
        short: (0, 0),
        _gpu: gpu,
        _pool: pool,
    }
}

type Slot<T> = Arc<Mutex<Option<Finished<T>>>>;

impl Rig {
    /// Starts a chain with `start` and pumps the lane's completions until it finishes (a minute at most).
    pub fn run<T: Send + 'static>(
        &mut self,
        start: impl FnOnce(GptqLane, Box<dyn FnOnce(Finished<T>) + Send>) -> Result<(), Box<Refused>>,
    ) -> Result<T, GptqError> {
        let slot: Slot<T> = Arc::new(Mutex::new(None));
        let sink = Arc::clone(&slot);
        let lane = self.lane.take().expect("the lane is idle");
        let begun = start(
            lane,
            Box::new(move |finished| {
                *sink.lock().unwrap() = Some(finished);
            }),
        );
        if let Err(refused) = begun {
            let Refused { lane, error } = *refused;
            self.lane = Some(lane);
            return Err(error);
        }
        let deadline = Instant::now() + Duration::from_secs(60);
        let finished = loop {
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
                break finished;
            }
            assert!(Instant::now() < deadline, "the chain did not finish in a minute");
        };
        let Finished {
            mut lane,
            scratch,
            result,
        } = finished;
        drop(scratch);
        lane.reclaim().expect("the buffers are freed in stream order");
        self.lane = Some(lane);
        result
    }

    /// Claims of exactly `bytes` from the rig's planner.
    pub fn claims(&self, bytes: StepBytes) -> Claims {
        let take = |location, bytes| {
            self.planner
                .claim(self.owner, location, bytes, "test")
                .expect("the planner has the bytes")
        };
        Claims {
            device: Some(take(DEVICE, bytes.device.saturating_sub(self.short.0))),
            pinned: Some(take(PINNED, bytes.pinned.saturating_sub(self.short.1))),
        }
    }

    /// The bytes the planner has not handed out, device and pinned.
    pub fn free(&self) -> (u64, u64) {
        (self.planner.free(DEVICE), self.planner.free(PINNED))
    }

    pub fn capacity(&self) -> u64 {
        CAPACITY
    }

    /// Frees the buffers given up, in stream order.
    pub fn settle(&mut self) {
        self.lane
            .as_mut()
            .expect("the lane is idle")
            .reclaim()
            .expect("reclaim");
    }

    /// Arms a retirement on the lane's stream that nobody collects, so the next chain cannot arm its own and fails.
    pub fn block_arming(&mut self) {
        self.lane_mut().block_arming().expect("a retirement arms");
    }

    /// Collects the retirement [`block_arming`](Self::block_arming) armed, when the stream has run up to it.
    pub fn unblock_arming(&mut self) {
        let deadline = Instant::now() + Duration::from_secs(60);
        while !self.lane_mut().unblock_arming() {
            assert!(Instant::now() < deadline, "the retirement did not arrive in a minute");
        }
    }

    /// Takes the lane apart as a shutdown does, calling `probe` just before it waits for the stream.
    pub fn shutdown(&mut self, probe: Box<dyn Fn() + Send>) -> Result<u64, GptqError> {
        let mut lane = self.lane.take().expect("the lane is idle");
        lane.set_shutdown_probe(probe);
        let (_stream, _handles, reclaimed) = lane.into_parts();
        reclaimed
    }

    /// Failed chains whose state the lane still holds.
    pub fn unsettled(&mut self) -> usize {
        self.lane_mut().unsettled()
    }

    fn lane_mut(&mut self) -> &mut GptqLane {
        self.lane.as_mut().expect("the lane is idle")
    }

    pub fn factor(
        &mut self,
        hessian: &DeviceBuffer,
        k: usize,
        damp: f32,
        act_order: bool,
    ) -> Result<Factor, GptqError> {
        let range = range_of(hessian);
        let bytes = self.lane_mut().factor_bytes(k)?;
        let claims = self.claims(bytes);
        self.run(move |lane, done| lane.factor(range, k, damp, act_order, claims, done))
    }

    pub fn tensor_scales(
        &mut self,
        weight: &DeviceBuffer,
        rows: usize,
        factor: &Factor,
        format: &Format,
    ) -> Result<TensorScales, GptqError> {
        let range = range_of(weight);
        let bytes = self.lane_mut().scales_bytes(rows, factor.k(), format)?;
        let claims = self.claims(bytes);
        self.run(|lane, done| lane.tensor_scales(range, rows, factor, format, claims, done))
    }

    pub fn round(&mut self, request: RoundRequest<'_>) -> Result<QuantState, GptqError> {
        let bytes = self.lane_mut().round_bytes(request.n, request.factor, request.format);
        let claims = self.claims(bytes);
        self.run(|lane, done| lane.round(&request, claims, done))
    }

    /// The bytes [`GptqLane::scales_bytes`] gives for these sizes.
    pub fn scales_bytes(&mut self, rows: usize, k: usize, format: &Format) -> Result<StepBytes, GptqError> {
        self.lane_mut().scales_bytes(rows, k, format)
    }

    pub fn upload(&self, bytes: &[u8]) -> DeviceBuffer {
        let buffer = DeviceBuffer::alloc(&self.library, bytes.len()).expect("device allocation");
        buffer.copy_from_host(0, bytes).expect("upload");
        buffer
    }
}

/// The whole of `buffer` as a step's input.
pub fn range_of(buffer: &DeviceBuffer) -> DeviceRange {
    // SAFETY: the buffer is live device memory the test keeps until its steps are done, and nothing else touches it.
    unsafe { DeviceRange::new(buffer.addr(), buffer.bytes() as u64) }
}
