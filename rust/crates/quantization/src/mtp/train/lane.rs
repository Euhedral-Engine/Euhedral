//! The training run's lane: one stream with the drafter, its kernels, the cuBLAS handles and the buffers of a window.
//! A window, an update or a read of device memory is one chain of stages on the lane's stream (see
//! [`chain`](crate::chain)); each takes the lane by value and gives it back in [`Finished`], so no two overlap and
//! nothing waits on the device. The calls that do wait ([`run_blocking`](TrainLane::run_blocking),
//! [`put`](TrainLane::put) and [`close`](TrainLane::close)) are for startup and shutdown, outside any frame.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

use euhedral_gpu::cuda::{CudaModule, CudaStream};
use euhedral_gpu::{LaneEnqueue, Stream};
use euhedral_graph::Completions;
use euhedral_inference_cuda::math::LaneHandles;
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, HostWeightsArena, PinnedBuffer};

use super::error::RunError;
use super::progress::Update;
use super::settings::Window;
use super::window::{Addresses, Mode, plan};
use crate::chain::{self, Next, Parts, Unsettled};
use crate::mtp::{Drafter, DrafterKernels, Work};

/// The bytes of device memory a read of the lane moves at a time.
pub const CHUNK_BYTES: usize = 32 << 20;

/// What a lane is made of, built by the loader: the drafter and the buffers it plans over.
pub struct Resources {
    pub kernels: DrafterKernels<CudaModule>,
    pub handles: LaneHandles,
    pub drafter: Drafter,
    /// Where the window's work reads and writes.
    pub addresses: Addresses,
    /// The most rows of a window.
    pub rows: u32,
    /// The drafter's arena, which the lane writes parameters into at startup.
    pub arena: DeviceBuffer,
    /// The device buffers the addresses name, kept alive by the lane.
    pub buffers: Vec<DeviceBuffer>,
    /// The embedding table the addresses name.
    pub table: HostWeightsArena,
    pub completions: Arc<Completions>,
}

impl Debug for Resources {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Resources").finish_non_exhaustive()
    }
}

/// Pinned memory the lane's transfers pass through.
struct Staging {
    /// A window's inputs: the hidden rows, the token ids and the labels.
    inputs: PinnedBuffer,
    /// A window's argmaxes and losses.
    results: PinnedBuffer,
    /// A piece of device memory being read.
    chunk: PinnedBuffer,
}

struct LaneParts {
    kernels: DrafterKernels<CudaModule>,
    handles: LaneHandles,
    drafter: Drafter,
    addresses: Addresses,
    rows: u32,
    arena: DeviceBuffer,
    _buffers: Vec<DeviceBuffer>,
    _table: HostWeightsArena,
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

/// A lane's stream with the drafter. A call takes the lane by value and gives it back in [`Finished`].
pub struct TrainLane {
    stream: CudaStream,
    parts: LaneParts,
}

impl Debug for TrainLane {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("TrainLane").finish_non_exhaustive()
    }
}

/// What a call hands to its `done`: the lane, free again, and the result.
pub type Finished<T> = chain::Finished<TrainLane, T, RunError>;

/// One window to run: its rows and inputs, staged on the host.
#[derive(Debug)]
pub struct Job {
    pub window: Window,
    pub mode: Mode,
    /// The base model's hidden rows of the window, BF16.
    pub hidden: Vec<u8>,
    /// The input token of each draft and the head row each is trained towards, depth-major `[depths, count]`.
    pub ids: Vec<i32>,
    pub labels: Vec<i32>,
}

/// What a labelled window's heads returned.
#[derive(Debug, PartialEq)]
pub struct Outcome {
    /// Each draft's argmax over the head's rows, depth-major `[depths, count]`; empty for an unlabelled window.
    pub predicted: Vec<i32>,
    /// The summed loss of each depth.
    pub losses: Vec<f64>,
}

fn pinned(library: &Arc<CudaLibrary>, bytes: usize) -> Result<PinnedBuffer, RunError> {
    let mut buffer = PinnedBuffer::alloc(library, bytes.max(1))?;
    // The allocator leaves it undefined, and a window shorter than the lane's rows uploads the tail too.
    buffer.as_mut_slice().fill(0);
    Ok(buffer)
}

impl TrainLane {
    /// A lane from its stream and what the loader built.
    pub fn new(stream: CudaStream, library: &Arc<CudaLibrary>, resources: Resources) -> Result<TrainLane, RunError> {
        let Resources {
            kernels,
            handles,
            drafter,
            addresses,
            rows,
            arena,
            buffers,
            table,
            completions,
        } = resources;
        let (rows_wide, hidden, depths) = (rows as usize, addresses.hidden as usize, addresses.depths as usize);
        let staging = Staging {
            inputs: pinned(library, rows_wide * hidden * 2 + 2 * 4 * rows_wide * depths)?,
            results: pinned(library, 4 * rows_wide * depths + 8 * depths)?,
            chunk: pinned(library, CHUNK_BYTES)?,
        };
        Ok(TrainLane {
            stream,
            parts: LaneParts {
                kernels,
                handles,
                drafter,
                addresses,
                rows,
                arena,
                _buffers: buffers,
                _table: table,
                staging,
                completions,
                unsettled: Unsettled::default(),
            },
        })
    }

    /// The drafter, for what the loader reads of it.
    pub fn drafter(&self) -> &Drafter {
        &self.parts.drafter
    }

    /// Runs `work` and waits for it. For startup: a frame never waits.
    pub fn run_blocking(&mut self, work: &[Work]) -> Result<(), RunError> {
        let parts = &mut self.parts;
        chain::submit(&mut self.stream, |queue| {
            // SAFETY: the work names buffers the lane owns and keeps until it is closed, and the lane's own handles;
            // nothing else runs on the stream, which is waited for below.
            unsafe { parts.kernels.enqueue(queue, &mut parts.handles, work) }.map_err(RunError::from)
        })?;
        Stream::quiesce(&mut self.stream);
        Ok(())
    }

    /// Plans and runs the drafter's start: its gradients and moments zeroed when `fresh`, and its scales measured from
    /// the values. For startup.
    pub fn prepare(&mut self, fresh: bool) -> Result<(), RunError> {
        let mut work = Vec::new();
        if fresh {
            work.extend(self.parts.drafter.initialise()?);
        }
        work.extend(self.parts.drafter.measure()?);
        self.run_blocking(&work)
    }

    /// Writes `bytes` at `address`, which must lie in the drafter's arena. For startup: it waits.
    #[allow(
        clippy::disallowed_methods,
        reason = "startup: a synchronous copy before any frame runs on the lane"
    )]
    pub fn put(&self, address: u64, bytes: &[u8]) -> Result<(), RunError> {
        let arena = &self.parts.arena;
        let offset = address
            .checked_sub(arena.addr())
            .and_then(|offset| usize::try_from(offset).ok())
            .filter(|offset| offset.checked_add(bytes.len()).is_some_and(|end| end <= arena.bytes()))
            .ok_or_else(|| RunError::Broken(format!("{address:#x} is not in the drafter's arena")))?;
        Ok(arena.copy_from_host(offset, bytes)?)
    }

    /// Runs `job`: uploads its inputs, plans and runs the window, and reads the argmaxes and losses back. Returns at
    /// once; `done` runs in the frame that retires the work.
    pub fn window(self, job: Job, done: Box<dyn FnOnce(Finished<Outcome>) + Send>) -> Result<(), Box<Refused>> {
        if let Err(error) = self.check(&job) {
            return Err(Box::new(Refused { lane: self, error }));
        }
        let TrainLane { stream, parts } = self;
        chain::start(
            stream,
            parts,
            (),
            |stream, parts| TrainLane { stream, parts },
            done,
            Box::new(move |stream, parts, ()| window_stage(stream, parts, &job)),
        );
        Ok(())
    }

    fn check(&self, job: &Job) -> Result<(), RunError> {
        let at = &self.parts.addresses;
        let (count, depths) = (job.window.count as usize, at.depths as usize);
        if job.window.count == 0 || job.window.count > self.parts.rows {
            return Err(RunError::Broken(format!(
                "a window of {} rows on a lane of {}",
                job.window.count, self.parts.rows
            )));
        }
        let wanted = (count * at.hidden as usize * 2, depths * count, depths * count);
        let got = (job.hidden.len(), job.ids.len(), job.labels.len());
        if wanted != got {
            return Err(RunError::Broken(format!(
                "a window of {count} rows has {} hidden bytes, {} ids and {} labels, not {} bytes, {} and {}",
                got.0, got.1, got.2, wanted.0, wanted.1, wanted.2
            )));
        }
        Ok(())
    }

    /// Applies one optimiser step of `update`'s rates: the drafter's parameters move, their gradients are zeroed and
    /// their scales measured.
    pub fn update(self, update: Update, done: Box<dyn FnOnce(Finished<()>) + Send>) {
        let TrainLane { stream, parts } = self;
        chain::start(
            stream,
            parts,
            (),
            |stream, parts| TrainLane { stream, parts },
            done,
            Box::new(move |stream, parts, ()| {
                let work = parts
                    .drafter
                    .step(update.projection_rate, update.norm_rate, update.step)
                    .map_err(RunError::from)?;
                chain::submit(stream, |queue| {
                    // SAFETY: the work names buffers the lane owns, on the lane's own handles, and the lane keeps
                    // them until the retirement is confirmed.
                    unsafe { parts.kernels.enqueue(queue, &mut parts.handles, &work) }.map_err(RunError::from)
                })?;
                Ok(Next::Wait(Box::new(|_, _, ()| Ok(Next::Done(())))))
            }),
        );
    }

    /// Reads `bytes` of device memory at `address`, at most [`CHUNK_BYTES`].
    pub fn read(
        self,
        address: u64,
        bytes: usize,
        done: Box<dyn FnOnce(Finished<Vec<u8>>) + Send>,
    ) -> Result<(), Box<Refused>> {
        if bytes == 0 || bytes > CHUNK_BYTES {
            return Err(Box::new(Refused {
                lane: self,
                error: RunError::Broken(format!("a read of {bytes} bytes, at most {CHUNK_BYTES} fit")),
            }));
        }
        let TrainLane { stream, parts } = self;
        chain::start(
            stream,
            parts,
            (),
            |stream, parts| TrainLane { stream, parts },
            done,
            Box::new(move |stream, parts, ()| {
                let chunk = &mut parts.staging.chunk;
                chain::submit(stream, |queue| {
                    // SAFETY: the chunk is live and `bytes` long, the lane keeps it until the retirement is
                    // confirmed, and nothing else touches it meanwhile; the address is the caller's device memory.
                    unsafe { queue.copy_to_host(chunk.as_mut_ptr(), address, bytes) }?;
                    Ok::<(), RunError>(())
                })?;
                Ok(Next::Wait(Box::new(move |_, parts, ()| {
                    let read = parts
                        .staging
                        .chunk
                        .as_slice()
                        .get(..bytes)
                        .ok_or_else(|| RunError::Broken("the chunk is shorter than the read".to_owned()))?;
                    Ok(Next::Done(read.to_vec()))
                })))
            }),
        );
        Ok(())
    }

    /// Ends the lane at shutdown: waits for the stream, which a frame never does, and frees what it holds.
    pub fn close(mut self) -> Result<(), RunError> {
        self.parts.unsettled.settle(&mut self.stream, || {});
        Stream::quiesce(&mut self.stream);
        let LaneParts { handles, .. } = self.parts;
        handles.close().map_err(RunError::from)
    }
}

/// A call that did not start: the lane comes back untouched.
#[derive(Debug)]
pub struct Refused {
    pub lane: TrainLane,
    pub error: RunError,
}

/// The first stage of a window: stages the inputs, plans the window, enqueues it and the readbacks.
fn window_stage(
    stream: &mut CudaStream,
    parts: &mut LaneParts,
    job: &Job,
) -> Result<Next<LaneParts, (), Outcome, RunError>, RunError> {
    let at = parts.addresses.clone();
    let (count, depths) = (job.window.count as usize, at.depths as usize);
    let (hidden_bytes, id_bytes) = (job.hidden.len(), 4 * job.ids.len());
    let inputs = parts.staging.inputs.as_mut_slice();
    let (head, tail) = inputs.split_at_mut(hidden_bytes);
    head.copy_from_slice(&job.hidden);
    let (ids, labels) = tail.split_at_mut(id_bytes);
    for (slot, id) in ids.as_chunks_mut::<4>().0.iter_mut().zip(&job.ids) {
        *slot = id.to_le_bytes();
    }
    for (slot, label) in labels.as_chunks_mut::<4>().0.iter_mut().zip(&job.labels) {
        *slot = label.to_le_bytes();
    }
    let work = plan(&mut parts.drafter, &at, &job.window, job.mode)?;
    let labelled = job.window.labelled;
    let inputs = parts.staging.inputs.as_ptr();
    let results = &mut parts.staging.results;
    let (predicted_bytes, loss_bytes) = (4 * depths * count, 8 * depths);
    chain::submit(stream, |queue| {
        // SAFETY: the staging and the device buffers are live, at least as long as the copies, and the lane keeps
        // them until the retirement is confirmed; nothing writes the host side meanwhile.
        unsafe {
            queue.copy_to_device(at.hidden_bf16, inputs, hidden_bytes)?;
            queue.copy_to_device(at.ids, inputs.add(hidden_bytes), id_bytes)?;
            queue.copy_to_device(at.labels, inputs.add(hidden_bytes + id_bytes), id_bytes)?;
        }
        // SAFETY: the work names buffers the lane owns and its own handles, kept until the retirement.
        unsafe { parts.kernels.enqueue(queue, &mut parts.handles, &work) }?;
        if labelled {
            let host = results.as_mut_ptr();
            // SAFETY: as for the uploads; the readback buffers are `predicted_bytes + loss_bytes` long.
            unsafe {
                queue.copy_to_host(host, at.predicted, predicted_bytes)?;
                queue.copy_to_host(host.add(predicted_bytes), at.loss, loss_bytes)?;
            }
        }
        Ok::<(), RunError>(())
    })?;
    Ok(Next::Wait(Box::new(move |_, parts, ()| {
        if !labelled {
            return Ok(Next::Done(Outcome {
                predicted: Vec::new(),
                losses: Vec::new(),
            }));
        }
        let bytes = parts
            .staging
            .results
            .as_slice()
            .get(..predicted_bytes + loss_bytes)
            .ok_or_else(|| RunError::Broken("the results are shorter than the window".to_owned()))?;
        let (predicted, losses) = bytes.split_at(predicted_bytes);
        Ok(Next::Done(Outcome {
            predicted: predicted
                .as_chunks::<4>()
                .0
                .iter()
                .map(|word| i32::from_le_bytes(*word))
                .collect(),
            losses: losses
                .as_chunks::<8>()
                .0
                .iter()
                .map(|word| f64::from_le_bytes(*word))
                .collect(),
        }))
    })))
}
