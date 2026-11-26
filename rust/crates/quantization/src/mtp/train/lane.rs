//! The training run's lane: one stream with the drafter, its kernels, the cuBLAS handles and the buffers of a window.
//! A window, an update or a read of device memory is one chain of stages on the lane's stream (see
//! [`chain`](crate::chain)); each takes the lane by value and gives it back in [`Finished`], so no two overlap and
//! nothing waits on the device. A call that cannot run, because its job is wrong, ends the same way: the lane comes
//! back in `done` with the error, so the caller closes it off the lattice workers whatever happened. The calls that do
//! wait ([`run_blocking`](TrainLane::run_blocking), [`put`](TrainLane::put) and [`close`](Lane::close)) are for
//! startup and shutdown, outside any frame.
//!
//! The run is written against the [`Lane`] trait, so that its frames are tested with a lane that has no device.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

use euhedral_gpu::cuda::{CudaModule, CudaStream};
use euhedral_gpu::{LaneEnqueue, Stream};
use euhedral_graph::Completions;
use euhedral_inference_cuda::math::LaneHandles;
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, HostWeightsArena, PinnedBuffer};

use super::error::RunError;
use super::load::Keep;
use super::progress::Update;
use super::settings::Window;
use super::window::{Addresses, Mode, plan};
use crate::chain::{self, Next, Parts, Scratch, Unsettled};
use crate::mtp::{Drafter, DrafterKernels, Work, plan_nvfp4_decode_bf16};

/// The bytes of device memory a read of the lane moves at a time.
pub const CHUNK_BYTES: usize = 32 << 20;

/// An artifact's draft head as stored: NVFP4 rows (row-split-k128-v1) of `k` values padded to `k_pad`, then their scale
/// codes at `scale_offset` and the FP32 global scale at `global_offset`.
#[derive(Debug)]
pub struct PackedHead {
    pub bytes: Vec<u8>,
    pub rows: u32,
    pub k: u32,
    pub k_pad: u32,
    pub scale_offset: u64,
    pub global_offset: u64,
}

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
    /// What the device needs to stay open until the lane is closed.
    pub keep: Keep,
    /// Fills the lane's pinned memory with this byte before use (see `Options::poison`).
    pub poison: Option<u8>,
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
    /// Dropped after everything else, by [`close`](Lane::close): the device outlives everything the lane holds.
    keep: Keep,
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
pub type Finished<L, T> = chain::Finished<L, T, RunError>;

/// What a call's `done` is: run once, with the lane back, in the frame that ends the call.
pub type Done<L, T> = Box<dyn FnOnce(Finished<L, T>) + Send>;

/// What the run needs of a lane. Each call takes the lane and gives it back to `done`, exactly once, whether the call
/// ran, failed or was refused: the lane never gets lost on an error path.
pub trait Lane: Sized + Send + 'static {
    /// Runs `job`: uploads its inputs, plans and runs the window, and reads the argmaxes and losses back. Returns at
    /// once; `done` runs in the frame that retires the work.
    fn window(self, job: Job, done: Done<Self, Outcome>);

    /// Applies one optimiser step of `update`'s rates: the drafter's parameters move, their gradients are zeroed and
    /// their scales measured.
    fn update(self, update: Update, done: Done<Self, ()>);

    /// Reads `bytes` of device memory at `address`, at most [`CHUNK_BYTES`].
    fn read(self, address: u64, bytes: usize, done: Done<Self, Vec<u8>>);

    /// Ends the lane at shutdown: waits for the stream, which a frame never does, and frees what it holds. Whoever
    /// made the lane calls it, outside any frame, once.
    fn close(self) -> Result<(), RunError>;
}

/// Ends a call that cannot run: the lane comes back to `done` with `error`.
fn refuse<L, T>(lane: L, error: RunError, done: Done<L, T>) {
    done(chain::Finished {
        lane,
        scratch: Scratch::new(()),
        result: Err(error),
    });
}

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

/// Pinned memory of `bytes`, which holds whatever the allocator left unless `poison` fills it.
fn pinned(library: &Arc<CudaLibrary>, bytes: usize, poison: Option<u8>) -> Result<PinnedBuffer, RunError> {
    let mut buffer = PinnedBuffer::alloc(library, bytes.max(1))?;
    if let Some(byte) = poison {
        buffer.as_mut_slice().fill(byte);
    }
    Ok(buffer)
}

/// Why a window of `job` cannot run on a lane of `rows` rows, `hidden` wide and `depths` deep.
fn check_window(job: &Job, rows: u32, hidden: u32, depths: u32) -> Result<(), RunError> {
    let (count, depths) = (job.window.count as usize, depths as usize);
    if job.window.count == 0 || job.window.count > rows {
        return Err(RunError::Broken(format!(
            "a window of {} rows on a lane of {rows}",
            job.window.count
        )));
    }
    let wanted = (count * hidden as usize * 2, depths * count, depths * count);
    let got = (job.hidden.len(), job.ids.len(), job.labels.len());
    if wanted != got {
        return Err(RunError::Broken(format!(
            "a window of {count} rows has {} hidden bytes, {} ids and {} labels, not {} bytes, {} and {}",
            got.0, got.1, got.2, wanted.0, wanted.1, wanted.2
        )));
    }
    Ok(())
}

/// Why a read of `bytes` cannot run.
fn check_read(bytes: usize) -> Result<(), RunError> {
    if bytes == 0 || bytes > CHUNK_BYTES {
        return Err(RunError::Broken(format!(
            "a read of {bytes} bytes, between 1 and {CHUNK_BYTES} fit"
        )));
    }
    Ok(())
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
            keep,
            poison,
        } = resources;
        let (rows_wide, hidden, depths) = (rows as usize, addresses.hidden as usize, addresses.depths as usize);
        let staging = Staging {
            inputs: pinned(library, rows_wide * hidden * 2 + 2 * 4 * rows_wide * depths, poison)?,
            results: pinned(library, 4 * rows_wide * depths + 8 * depths, poison)?,
            chunk: pinned(library, CHUNK_BYTES, poison)?,
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
                keep,
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

    /// Decodes an artifact's NVFP4 draft head into BF16 at `head`, on the lane. For startup: it waits.
    ///
    /// The packed bytes go to the device from pinned memory, on the lane's own stream and in the same submission as the
    /// decode kernel, so the stream orders the decode after the upload. (A copy from pageable memory may return before
    /// its transfer lands, and a non-blocking stream does not wait for it.)
    pub fn decode_head(&mut self, stored: &PackedHead, head: u64) -> Result<(), RunError> {
        let library = self.parts.keep.library();
        let mut pinned = PinnedBuffer::alloc(library, stored.bytes.len().max(1))?;
        pinned.as_mut_slice()[..stored.bytes.len()].copy_from_slice(&stored.bytes);
        let device = DeviceBuffer::alloc(library, stored.bytes.len().max(1))?;
        let base = device.addr();
        let work = [Work::Steps(plan_nvfp4_decode_bf16(
            base,
            base + stored.scale_offset,
            base + stored.global_offset,
            stored.rows,
            stored.k_pad,
            stored.k,
            head,
        )?)];
        let parts = &mut self.parts;
        let submitted = chain::submit(&mut self.stream, |queue| {
            // SAFETY: the pinned and device buffers are live and `bytes` long, and are kept until the stream is waited
            // for below; the work names them and the lane's own buffers and handles.
            unsafe {
                queue.copy_to_device(base, pinned.as_ptr(), stored.bytes.len())?;
                parts.kernels.enqueue(queue, &mut parts.handles, &work)?;
            }
            Ok::<(), RunError>(())
        });
        // Waited for whether or not the submission failed: nothing may still be reading the buffers when they drop.
        Stream::quiesce(&mut self.stream);
        submitted
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
}

impl Lane for TrainLane {
    fn window(self, job: Job, done: Done<Self, Outcome>) {
        let at = &self.parts.addresses;
        if let Err(error) = check_window(&job, self.parts.rows, at.hidden, at.depths) {
            return refuse(self, error, done);
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
    }

    fn update(self, update: Update, done: Done<Self, ()>) {
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

    fn read(self, address: u64, bytes: usize, done: Done<Self, Vec<u8>>) {
        if let Err(error) = check_read(bytes) {
            return refuse(self, error, done);
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
    }

    fn close(self) -> Result<(), RunError> {
        let TrainLane { mut stream, mut parts } = self;
        parts.unsettled.settle(&mut stream, || {});
        Stream::quiesce(&mut stream);
        let LaneParts {
            kernels,
            handles,
            drafter,
            addresses,
            rows,
            arena,
            _buffers,
            _table,
            staging,
            completions,
            unsettled,
            keep,
        } = parts;
        // The handles close first, then the rest of the parts, the buffers among them, then the stream; the device
        // outlives everything the lane holds, so `keep` goes last.
        let closed = handles.close().map_err(RunError::from);
        drop((
            kernels,
            drafter,
            addresses,
            rows,
            arena,
            _buffers,
            _table,
            staging,
            completions,
            unsettled,
        ));
        drop(stream);
        drop(keep);
        closed
    }
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

#[cfg(test)]
mod tests {
    #![allow(
        clippy::unwrap_used,
        clippy::disallowed_methods,
        clippy::disallowed_types,
        reason = "a test: a failed unwrap is the failure, and a sink the test thread reads after the call ran"
    )]

    use std::sync::Mutex;

    use super::*;

    const ROWS: u32 = 8;
    const HIDDEN: u32 = 4;
    const DEPTHS: u32 = 2;

    fn job(count: u32) -> Job {
        let (rows, depths) = (count as usize, DEPTHS as usize);
        Job {
            window: Window {
                a: 0,
                count,
                first_key: 0,
                labelled: true,
            },
            mode: Mode::Train,
            hidden: vec![0; rows * HIDDEN as usize * 2],
            ids: vec![0; depths * rows],
            labels: vec![0; depths * rows],
        }
    }

    fn check(job: &Job) -> Result<(), RunError> {
        check_window(job, ROWS, HIDDEN, DEPTHS)
    }

    #[test]
    fn a_window_of_one_to_the_lanes_rows_with_its_inputs_whole_runs() {
        for count in [1, 5, ROWS] {
            check(&job(count)).unwrap();
        }
    }

    #[test]
    fn a_window_of_no_rows_or_more_than_the_lane_has_is_refused() {
        assert!(check(&job(0)).is_err());
        assert!(check(&job(ROWS + 1)).is_err());
    }

    #[test]
    fn a_window_whose_hidden_rows_ids_or_labels_are_the_wrong_length_is_refused() {
        let mut short = job(3);
        short.hidden.pop();
        let mut few_ids = job(3);
        few_ids.ids.pop();
        let mut few_labels = job(3);
        few_labels.labels.pop();
        for refused in [short, few_ids, few_labels] {
            assert!(matches!(check(&refused), Err(RunError::Broken(_))));
        }
    }

    #[test]
    fn a_read_of_one_byte_to_a_chunk_runs_and_anything_else_is_refused() {
        check_read(1).unwrap();
        check_read(CHUNK_BYTES).unwrap();
        assert!(check_read(0).is_err());
        assert!(check_read(CHUNK_BYTES + 1).is_err());
    }

    #[test]
    fn a_refused_call_gives_the_lane_back_to_its_done_with_the_error() {
        let seen = Arc::new(Mutex::new(None));
        let held = Arc::clone(&seen);
        refuse(
            7u8,
            RunError::Broken("no".to_owned()),
            Box::new(move |finished: Finished<u8, ()>| {
                *held.lock().unwrap() = Some((finished.lane, finished.result));
            }),
        );
        let (lane, result) = seen.lock().unwrap().take().unwrap();
        assert_eq!(lane, 7);
        assert!(matches!(result, Err(RunError::Broken(message)) if message == "no"));
    }
}
