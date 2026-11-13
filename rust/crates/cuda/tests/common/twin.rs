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

use std::collections::BTreeMap;
use std::error::Error;
use std::process::abort;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread;
use std::time::Duration;

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
    /// Whether both sides run row-exact (the native submission's mode; the plan's modes are the planner's own).
    pub row_exact: bool,
    /// Device tables of page addresses: the buffer that is the table, and the buffers it lists. Each set gets a table
    /// of its own pages, and a table is never copied between sets.
    pub tables: Vec<(usize, Vec<usize>)>,
    /// Buffers that are only allocated: scratch the kernels write before they read, too large to be worth filling.
    pub scratch: Vec<usize>,
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
        let native = self.upload(&case.buffers, &case.scratch)?;
        let twin = self.upload(&case.buffers, &case.scratch)?;
        let native_ptrs: Vec<DevicePtr> = native.iter().map(DeviceBuffer::ptr).collect();
        let twin_ptrs: Vec<DevicePtr> = twin.iter().map(DeviceBuffer::ptr).collect();
        for (set, pointers) in [(&native, &native_ptrs), (&twin, &twin_ptrs)] {
            for (table, pages) in &case.tables {
                let bytes: Vec<u8> = pages
                    .iter()
                    .flat_map(|&page| pointers[page].get().to_le_bytes())
                    .collect();
                set[*table].copy_from_host(0, &bytes)?;
            }
        }

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
            self.run_native(prepare, &native_ptrs, case.row_exact)?;
            for (index, (made, copy)) in native.iter().zip(&twin).enumerate() {
                if case.scratch.contains(&index) || case.tables.iter().any(|(table, _)| *table == index) {
                    continue;
                }
                let mut bytes = vec![0u8; made.bytes()];
                made.copy_to_host(0, &mut bytes)?;
                copy.copy_from_host(0, &bytes)?;
            }
        }
        self.run_native(&case.native, &native_ptrs, case.row_exact)?;
        self.run_rustee(&plan)?;

        let mut compared = 0;
        let mut written = false;
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
            written |= left.iter().any(|&byte| byte != 0);
        }
        if !written {
            return Err(format!(
                "{}: every compared buffer is zero, so the case proves nothing",
                case.name
            )
            .into());
        }
        Ok(Verdict::Identical { bytes: compared })
    }

    fn upload(&self, buffers: &[Vec<u8>], scratch: &[usize]) -> Result<Vec<DeviceBuffer>, Failure> {
        buffers
            .iter()
            .enumerate()
            .map(|(index, bytes)| {
                let buffer = DeviceBuffer::alloc(&self.library, bytes.len())?;
                if !scratch.contains(&index) {
                    buffer.copy_from_host(0, bytes)?;
                }
                Ok(buffer)
            })
            .collect::<Result<Vec<_>, Failure>>()
            .and_then(|buffers| {
                self.lane.synchronize()?;
                Ok(buffers)
            })
    }

    fn run_native(&self, step: &Native<'_>, pointers: &[DevicePtr], row_exact: bool) -> Result<(), Failure> {
        let submission = Submission::begin(
            &self.lane,
            SubmitMode {
                row_exact,
                ..SubmitMode::default()
            },
        )?;
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

/// Cases are unavailable for these reasons at the pinned RustEE revision; any other reason is a failure.
const EXPECTED_SKIPS: [(&str, &str); 3] = [
    (
        "architecture-specific compile target",
        "needs the architecture-specific compile target",
    ),
    ("tensor-map arguments", "takes tensor-map arguments"),
    ("dynamic shared memory", "dynamic shared memory"),
];

/// What a sweep came to, per planner: cases compared, cases the planner refused, cases skipped by reason.
#[derive(Default)]
pub struct Tally {
    planners: BTreeMap<String, Counts>,
    failures: Vec<String>,
}

#[derive(Default)]
struct Counts {
    compared: usize,
    refused: usize,
    skipped: BTreeMap<String, usize>,
}

impl Tally {
    pub fn refused(&mut self, planner: &str) {
        self.planners.entry(planner.to_string()).or_default().refused += 1;
    }

    /// Records a verdict; a difference or an unexpected unavailability is a failure, reported at the end.
    pub fn record(&mut self, planner: &str, shape: &str, verdict: Verdict) {
        let counts = self.planners.entry(planner.to_string()).or_default();
        match verdict {
            Verdict::Identical { .. } => counts.compared += 1,
            Verdict::Differs {
                buffer,
                offset,
                native,
                rustee,
            } => {
                let line = format!(
                    "MISMATCH {planner} {shape}: buffer {buffer} first differs at byte {offset}: native {native:#04x}, rustee {rustee:#04x}"
                );
                println!("{line}");
                self.failures.push(line);
            }
            Verdict::Skipped(reasons) => {
                // A case counts once for each reason it is skipped for.
                let mut names = Vec::new();
                for reason in reasons {
                    match EXPECTED_SKIPS.iter().find(|(_, text)| reason.contains(text)) {
                        Some((name, _)) => {
                            if !names.contains(name) {
                                names.push(*name);
                                *counts.skipped.entry((*name).to_string()).or_default() += 1;
                            }
                        }
                        None => {
                            let line = format!("UNAVAILABLE {planner} {shape}: {reason}");
                            println!("{line}");
                            self.failures.push(line);
                        }
                    }
                }
            }
        }
    }

    /// Prints the totals and fails when anything mismatched or was unavailable for an unexpected reason.
    pub fn finish(self) {
        for (planner, counts) in &self.planners {
            println!(
                "{planner}: {} compared, {} refused by the planner, skipped: {:?}",
                counts.compared, counts.refused, counts.skipped
            );
        }
        assert!(
            self.failures.is_empty(),
            "{} failures, first: {}",
            self.failures.len(),
            self.failures[0]
        );
    }
}

/// Aborts the process when the test is still running after `seconds`, so a hung device call fails fast.
pub fn watchdog(seconds: u64) -> Arc<AtomicBool> {
    let done = Arc::new(AtomicBool::new(false));
    let flag = Arc::clone(&done);
    thread::spawn(move || {
        for _ in 0..seconds {
            thread::sleep(Duration::from_secs(1));
            if flag.load(Ordering::Relaxed) {
                return;
            }
        }
        println!("watchdog: the test did not finish in {seconds} s");
        abort();
    });
    done
}

/// A fixed-seed generator (splitmix64) of the test's input bytes.
pub struct Random(pub u64);

impl Random {
    pub fn word(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// `count` BF16 values in about [-1, 1): finite, never NaN.
    pub fn bf16(&mut self, count: usize) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(count * 2);
        // A large buffer repeats a block of 64 Ki values: the data is only to be varied, and cheap to make.
        let block = count.min(1 << 16);
        for _ in 0..block {
            let r = self.word();
            // sign, exponent 0x74..0x7f (values 2^-11 to 1), a random mantissa.
            let exponent = 0x74 + (r >> 8) % 12;
            let value = (((r >> 20) & 1) << 15) | (exponent << 7) | (r & 0x7f);
            bytes.extend_from_slice(&(value as u16).to_le_bytes());
        }
        while bytes.len() < count * 2 {
            let more = (count * 2 - bytes.len()).min(bytes.len());
            bytes.extend_from_within(..more);
        }
        bytes
    }

    /// `count` raw bytes.
    pub fn bytes(&mut self, count: usize) -> Vec<u8> {
        let mut bytes: Vec<u8> = (0..count.min(1 << 17)).map(|_| self.word() as u8).collect();
        while bytes.len() < count {
            let more = (count - bytes.len()).min(bytes.len());
            bytes.extend_from_within(..more);
        }
        bytes
    }
}
