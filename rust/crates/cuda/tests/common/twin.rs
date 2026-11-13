//! Temporary: removed together with the native launch path, once the kernels run through RustEE.
//!
//! The twin harness: one kernel family's case run twice in one CUDA context, once through the native C library and
//! once through a planned launch on a RustEE lane, and the outputs compared byte for byte.
//!
//! One context. The native library binds the device's primary context with the CUDA runtime; RustEE's
//! `CudaGpu::open(ordinal)` retains the same device's primary context through the driver API. So device addresses mean
//! the same in both, the case's buffers are allocated once per run by the native library and handed to both sides,
//! and nothing is copied through the host to cross between them (the host is used only to read the results back).
//!
//! A case is data: the buffers' initial contents, an optional native step that makes inputs with the C library, the
//! native step under test, and the plan (the launches a planner produced for the same buffers). The harness runs the
//! native step on one set of buffers and the plan on a second set that holds the same inputs, so neither run sees the
//! other's output.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test helper: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: a test synchronizes both lanes and copies synchronously to compare results"
)]
#![allow(dead_code, reason = "each test binary uses a part of the harness")]

use std::error::Error;
use std::sync::Arc;

use euhedral_gpu::cuda::{CudaGpu, CudaStream};
use euhedral_gpu::{Gpu, GpuError, Stream, StreamOptions};
use euhedral_graph::LaneId;
use euhedral_inference_cuda::launch::catalog::Launch;
use euhedral_inference_cuda::launch::load::{KernelSources, KernelTable, Slot, load};
use euhedral_inference_cuda::launch::queue::enqueue;
use euhedral_inference_cuda::{CudaError, CudaLane, CudaLibrary, DeviceBuffer, DevicePtr, Submission, SubmitMode};

pub type Failure = Box<dyn Error>;

/// A step run through the native library on the buffers' device addresses.
pub type Native<'a> = Box<dyn Fn(&Submission<'_>, &[DevicePtr]) -> Result<(), CudaError> + 'a>;

/// The launches a planner produced for buffers at the given addresses.
pub type Plan<'a> = Box<dyn Fn(&[DevicePtr]) -> Vec<Launch> + 'a>;

/// One case: the same work through the C wrapper and through its plan.
pub struct Case<'a> {
    pub name: &'a str,
    /// The buffers' initial contents.
    pub buffers: Vec<Vec<u8>>,
    /// A native step that makes inputs (run once, on the native set; every buffer it leaves is copied to the twin set).
    pub prepare: Option<Native<'a>>,
    /// The C wrapper under test.
    pub native: Native<'a>,
    /// The planned launches for the buffers at these addresses.
    pub plan: Plan<'a>,
    /// The buffers compared.
    pub outputs: Vec<usize>,
}

/// What a case came to.
#[derive(Debug, PartialEq, Eq)]
pub enum Verdict {
    /// Every compared buffer is identical: how many bytes were compared.
    Identical { bytes: usize },
    /// The first difference.
    Differs {
        buffer: usize,
        offset: usize,
        native: u8,
        rustee: u8,
    },
    /// The plan has a launch that cannot run at the pinned RustEE revision (the kernel names and why); nothing ran.
    Skipped(Vec<String>),
}

/// The native library's lane and a RustEE lane on the same device and context.
pub struct Twin {
    library: Arc<CudaLibrary>,
    table: KernelTable,
    lane: Arc<CudaLane>,
    stream: CudaStream,
}

impl Twin {
    /// Opens the device the native library bound through RustEE, loads the kernel modules and opens a lane on each
    /// side. The library reports the device's properties and not its ordinal, so the device is the first whose
    /// compute capability is the library's.
    pub fn open(library: &Arc<CudaLibrary>) -> Result<Twin, Failure> {
        let (major, minor) = library.device().compute_capability;
        let wanted = (i32::try_from(major)?, i32::try_from(minor)?);
        let mut gpu = None;
        for ordinal in 0..16 {
            let Ok(candidate) = CudaGpu::open(ordinal) else { break };
            if candidate.arch() == wanted {
                gpu = Some(candidate);
                break;
            }
        }
        let gpu = gpu.ok_or("RustEE found no device of the library's compute capability")?;
        let table = load(&gpu, &KernelSources::beside(library)?)?;
        let lane = CudaLane::new(library, LaneId(0), 4)?;
        let mut streams = gpu.open_lanes_at(LaneId(1), &["twin"], &[StreamOptions::default()])?;
        let stream = streams.pop().ok_or("RustEE opened no lane")?;
        Ok(Twin {
            library: Arc::clone(library),
            table,
            lane,
            stream,
        })
    }

    pub fn table(&self) -> &KernelTable {
        &self.table
    }

    pub fn gpu(&self) -> &Arc<CudaGpu> {
        self.stream.gpu()
    }

    pub fn run(&mut self, case: &Case<'_>) -> Result<Verdict, Failure> {
        let native = self.upload(&case.buffers)?;
        let twin = self.upload(&case.buffers)?;
        let native_ptrs: Vec<DevicePtr> = native.iter().map(DeviceBuffer::ptr).collect();
        let twin_ptrs: Vec<DevicePtr> = twin.iter().map(DeviceBuffer::ptr).collect();

        let plan = (case.plan)(&twin_ptrs);
        let refused: Vec<String> = plan
            .iter()
            .filter_map(|launch| match self.table.get(launch.kernel) {
                Slot::Unavailable(reason) => Some(format!("{}: {reason}", launch.kernel.name())),
                Slot::Ready(resolved) if launch.shared_bytes > resolved.max_shared_bytes() => Some(format!(
                    "{}: needs {} bytes of dynamic shared memory",
                    launch.kernel.name(),
                    launch.shared_bytes
                )),
                Slot::Ready(_) => None,
            })
            .collect();
        if !refused.is_empty() {
            return Ok(Verdict::Skipped(refused));
        }

        if let Some(prepare) = &case.prepare {
            self.run_native(prepare, &native_ptrs)?;
            for (made, copy) in native.iter().zip(&twin) {
                let mut bytes = vec![0u8; made.bytes()];
                made.copy_to_host(0, &mut bytes)?;
                copy.copy_from_host(0, &bytes)?;
            }
        }
        self.run_native(&case.native, &native_ptrs)?;
        self.run_rustee(&plan)?;

        let mut compared = 0;
        for &index in &case.outputs {
            let (left, right) = (download(&native[index])?, download(&twin[index])?);
            if let Some(offset) = left.iter().zip(&right).position(|(a, b)| a != b) {
                return Ok(Verdict::Differs {
                    buffer: index,
                    offset,
                    native: left[offset],
                    rustee: right[offset],
                });
            }
            compared += left.len();
        }
        Ok(Verdict::Identical { bytes: compared })
    }

    fn upload(&self, buffers: &[Vec<u8>]) -> Result<Vec<DeviceBuffer>, Failure> {
        buffers
            .iter()
            .map(|bytes| {
                let buffer = DeviceBuffer::alloc(&self.library, bytes.len())?;
                buffer.copy_from_host(0, bytes)?;
                Ok(buffer)
            })
            .collect::<Result<Vec<_>, Failure>>()
            .and_then(|buffers| {
                self.lane.synchronize()?;
                Ok(buffers)
            })
    }

    fn run_native(&self, step: &Native<'_>, pointers: &[DevicePtr]) -> Result<(), Failure> {
        let submission = Submission::begin(&self.lane, SubmitMode::default())?;
        step(&submission, pointers)?;
        submission.finish()?;
        self.lane.synchronize()?;
        Ok(())
    }

    fn run_rustee(&mut self, plan: &[Launch]) -> Result<(), Failure> {
        let table = &self.table;
        self.stream.submit(|queue| {
            for launch in plan {
                // SAFETY: the plan names the case's own buffers, which stay allocated until the results are read.
                unsafe { enqueue(queue, table, launch) }.map_err(|error| GpuError::Invalid(error.to_string()))?;
            }
            Ok(())
        })?;
        self.stream.quiesce();
        Ok(())
    }
}

fn download(buffer: &DeviceBuffer) -> Result<Vec<u8>, Failure> {
    let mut bytes = vec![0u8; buffer.bytes()];
    buffer.copy_to_host(0, &mut bytes)?;
    Ok(bytes)
}
