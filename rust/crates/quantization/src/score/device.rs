//! The score kernel on a lane: a group of logits rows and their reference records go up, `score_rows` reduces each
//! row to five float64 values, and the values come back. One chain of two stages on the lane's stream; nothing waits on
//! the device (see [`chain`](crate::chain)).
//!
//! The lane owns its staging: a pinned buffer and a device buffer for each of the logits, the reference records, the
//! targets and the values, sized once for the largest group. A group takes the lane by value and gives it back in
//! [`Finished`], so two groups never share the staging.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

use euhedral_gpu::cuda::{CudaKernel, CudaModule, CudaStream};
use euhedral_gpu::{Args, Dim3, GpuError, LaneEnqueue, Launch, Stream};
use euhedral_graph::Completions;
use euhedral_inference_cuda::{CudaError, CudaLibrary, DeviceBuffer, PinnedBuffer};
use thiserror::Error;

use super::summary::FIELDS;
use crate::chain::{self, ChainError, Next, Parts, Unsettled};
use crate::kernels::{Module, ModuleError, Modules};

/// The threads of a block: the kernel's reductions are written for this many.
const THREADS: u32 = 256;
/// The bytes of one result: [`FIELDS`] float64 values.
const RESULT_BYTES: usize = FIELDS * size_of::<f64>();

/// Why a group could not be scored.
#[derive(Debug, Error)]
pub enum ScoreError {
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error(transparent)]
    Module(#[from] ModuleError),
    /// The group does not fit the lane's sizes.
    #[error("{0}")]
    Request(String),
    #[error("the kernel {0} is not in the module")]
    Kernel(&'static str),
    #[error("{0}")]
    Broken(String),
}

impl ChainError for ScoreError {
    fn broken(message: &str) -> Self {
        ScoreError::Broken(message.to_owned())
    }
}

/// The compiled score kernel.
#[derive(Clone)]
pub struct ScoreKernels {
    score_rows: CudaKernel,
}

impl Debug for ScoreKernels {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("ScoreKernels").finish_non_exhaustive()
    }
}

impl ScoreKernels {
    /// Compiles (or finds) the module. This waits for the compiler: call it at startup.
    pub fn load(modules: &mut Modules<CudaModule>) -> Result<Arc<ScoreKernels>, ScoreError> {
        let compiled = modules.load(Module::Score)?;
        let score_rows = compiled
            .kernel("score_rows")
            .cloned()
            .ok_or(ScoreError::Kernel("score_rows"))?;
        Ok(Arc::new(ScoreKernels { score_rows }))
    }
}

/// The sizes a lane's staging is made for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    /// The logits of a row: the model's vocabulary.
    pub vocabulary: usize,
    /// The bytes of a reference record: [`Reference::row_bytes`](crate::kld::Reference::row_bytes).
    pub record_bytes: usize,
    /// The most rows of a group.
    pub group: usize,
}

impl Geometry {
    fn logits_bytes(&self) -> usize {
        2 * self.vocabulary
    }
}

/// One row to score.
#[derive(Debug)]
pub struct ScoreRow {
    /// The candidate's BF16 logits, `2 * vocabulary` bytes.
    pub logits: Vec<u8>,
    /// The reference record of the position.
    pub record: Vec<u8>,
    /// The token the position predicts.
    pub target: u32,
}

struct Pair {
    host: PinnedBuffer,
    device: DeviceBuffer,
}

impl Pair {
    fn new(library: &Arc<CudaLibrary>, bytes: usize) -> Result<Pair, CudaError> {
        Ok(Pair {
            host: PinnedBuffer::alloc(library, bytes)?,
            device: DeviceBuffer::alloc(library, bytes)?,
        })
    }

    fn upload<Q: LaneEnqueue>(&self, queue: &mut Q, bytes: usize) -> Result<(), GpuError> {
        // SAFETY: both buffers are live, at least `bytes` long, and owned by the lane, which keeps them until the
        // chain's retirement is confirmed; nothing writes the host side meanwhile.
        unsafe { queue.copy_to_device(self.device.addr(), self.host.as_ptr(), bytes) }?;
        Ok(())
    }

    fn download<Q: LaneEnqueue>(&mut self, queue: &mut Q, bytes: usize) -> Result<(), GpuError> {
        // SAFETY: as `upload`; nothing reads the host side before the retirement.
        unsafe { queue.copy_to_host(self.host.as_mut_ptr(), self.device.addr(), bytes) }?;
        Ok(())
    }
}

struct Staging {
    logits: Pair,
    records: Pair,
    targets: Pair,
    values: Pair,
}

/// Everything of a lane except its stream.
struct LaneParts {
    kernels: Arc<ScoreKernels>,
    geometry: Geometry,
    staging: Staging,
    completions: Arc<Completions>,
    unsettled: Unsettled,
}

impl Parts for LaneParts {
    fn completions(&self) -> &Arc<Completions> {
        &self.completions
    }

    fn unsettled(&mut self) -> &mut Unsettled {
        &mut self.unsettled
    }
}

/// A lane's stream with the score kernel and its staging. A group takes the lane by value and gives it back in
/// [`Finished`] (or in [`Refused`], when it could not start).
pub struct ScoreLane {
    stream: CudaStream,
    parts: LaneParts,
}

impl Debug for ScoreLane {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("ScoreLane").finish_non_exhaustive()
    }
}

/// A group that did not start: the lane comes back untouched.
#[derive(Debug)]
pub struct Refused {
    pub lane: ScoreLane,
    pub error: ScoreError,
}

/// What a group hands to its `done` call: the lane, free again, and the values of every row, [`FIELDS`] each.
pub type Finished = chain::Finished<ScoreLane, Vec<f64>, ScoreError>;

impl ScoreLane {
    /// A lane from its stream, the kernel, the sizes of its groups and the completions that deliver its retirements.
    pub fn new(
        stream: CudaStream,
        kernels: Arc<ScoreKernels>,
        library: &Arc<CudaLibrary>,
        geometry: Geometry,
        completions: Arc<Completions>,
    ) -> Result<ScoreLane, ScoreError> {
        if geometry.group == 0 || geometry.vocabulary == 0 || geometry.record_bytes < 8 {
            return Err(ScoreError::Request(format!("a lane cannot hold {geometry:?}")));
        }
        let group = geometry.group;
        let staging = Staging {
            logits: Pair::new(library, group * geometry.logits_bytes())?,
            records: Pair::new(library, group * geometry.record_bytes)?,
            targets: Pair::new(library, group * size_of::<i32>())?,
            values: Pair::new(library, group * RESULT_BYTES)?,
        };
        Ok(ScoreLane {
            stream,
            parts: LaneParts {
                kernels,
                geometry,
                staging,
                completions,
                unsettled: Unsettled::default(),
            },
        })
    }

    /// The sizes the lane was made for.
    pub fn geometry(&self) -> Geometry {
        self.parts.geometry
    }

    /// Scores `rows`: stages them, runs the kernel and reads the values back. Returns at once; `done` runs in the frame
    /// that retires the work, with the lane and `rows.len() * FIELDS` values (or the failure).
    pub fn score(self, rows: Vec<ScoreRow>, done: Box<dyn FnOnce(Finished) + Send>) -> Result<(), Box<Refused>> {
        if let Err(error) = self.check(&rows) {
            return Err(Box::new(Refused { lane: self, error }));
        }
        let ScoreLane { stream, parts } = self;
        chain::start(
            stream,
            parts,
            (),
            |stream, parts| ScoreLane { stream, parts },
            done,
            Box::new(move |stream, parts, ()| stage(stream, parts, &rows)),
        );
        Ok(())
    }

    fn check(&self, rows: &[ScoreRow]) -> Result<(), ScoreError> {
        let geometry = self.parts.geometry;
        if rows.is_empty() || rows.len() > geometry.group {
            return Err(ScoreError::Request(format!(
                "a group of {} rows on a lane of {}",
                rows.len(),
                geometry.group
            )));
        }
        for (index, row) in rows.iter().enumerate() {
            if row.logits.len() != geometry.logits_bytes() || row.record.len() != geometry.record_bytes {
                return Err(ScoreError::Request(format!(
                    "row {index} has {} logits bytes and {} record bytes, not {} and {}",
                    row.logits.len(),
                    row.record.len(),
                    geometry.logits_bytes(),
                    geometry.record_bytes
                )));
            }
            if row.target as usize >= geometry.vocabulary {
                return Err(ScoreError::Request(format!(
                    "row {index} predicts token {} of a vocabulary of {}",
                    row.target, geometry.vocabulary
                )));
            }
        }
        Ok(())
    }

    /// Ends the lane at shutdown: waits for the stream, which a frame never does, and frees the staging.
    pub fn close(mut self) {
        self.parts.unsettled.settle(&mut self.stream, || {});
        Stream::quiesce(&mut self.stream);
    }
}

/// The first stage: stage the rows, upload, launch, download the values.
fn stage(
    stream: &mut CudaStream,
    parts: &mut LaneParts,
    rows: &[ScoreRow],
) -> Result<Next<LaneParts, (), Vec<f64>, ScoreError>, ScoreError> {
    let geometry = parts.geometry;
    let count = rows.len();
    let staging = &mut parts.staging;
    let (logits_bytes, record_bytes) = (geometry.logits_bytes(), geometry.record_bytes);
    for (index, row) in rows.iter().enumerate() {
        fill(staging.logits.host.as_mut_slice(), index * logits_bytes, &row.logits)?;
        fill(staging.records.host.as_mut_slice(), index * record_bytes, &row.record)?;
        fill(
            staging.targets.host.as_mut_slice(),
            index * size_of::<i32>(),
            &(row.target as i32).to_le_bytes(),
        )?;
    }
    let width = i32::try_from(record_bytes / 2).map_err(|_| ScoreError::Request("a record is too long".to_owned()))?;
    let vocabulary = i32::try_from(geometry.vocabulary)
        .map_err(|_| ScoreError::Request("the vocabulary is too large".to_owned()))?;
    let args = Args::new()
        .arg(staging.logits.device.addr())
        .arg(staging.records.device.addr())
        .arg(width)
        .arg(vocabulary)
        .arg(staging.targets.device.addr())
        .arg(staging.values.device.addr());
    let launch = Launch {
        grid: Dim3::x(u32::try_from(count).map_err(|_| ScoreError::Request("too many rows".to_owned()))?),
        block: Dim3::x(THREADS),
        shared_bytes: 0,
        programmatic: false,
    };
    let kernel = parts.kernels.score_rows.clone();
    chain::submit(stream, |queue| {
        staging.logits.upload(queue, count * logits_bytes)?;
        staging.records.upload(queue, count * record_bytes)?;
        staging.targets.upload(queue, count * size_of::<i32>())?;
        // SAFETY: the arguments are checked against the kernel's parameter sizes by the launch, and name buffers the
        // lane owns until the retirement is confirmed.
        unsafe { queue.launch(&kernel, launch, &args) }?;
        staging.values.download(queue, count * RESULT_BYTES)?;
        Ok::<(), ScoreError>(())
    })?;
    Ok(Next::Wait(Box::new(move |_, parts, ()| {
        let bytes = parts
            .staging
            .values
            .host
            .as_slice()
            .get(..count * RESULT_BYTES)
            .ok_or_else(|| ScoreError::broken("the values are shorter than the group"))?;
        let values = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|word| f64::from_le_bytes(*word))
            .collect();
        Ok(Next::Done(values))
    })))
}

fn fill(host: &mut [u8], at: usize, bytes: &[u8]) -> Result<(), ScoreError> {
    host.get_mut(at..at + bytes.len())
        .ok_or_else(|| ScoreError::broken("the staging is shorter than the group"))?
        .copy_from_slice(bytes);
    Ok(())
}
