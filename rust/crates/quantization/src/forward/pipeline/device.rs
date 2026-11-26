//! The pipeline over the card: the layer forward on a main lane, one sequence per slot lane.
//!
//! The batch's rows travel as in Java: the hidden states of every sequence rest in pinned host memory, a batch is
//! copied to a device buffer on the main lane, run, and copied back when the pass writes. A slot's lane waits, on the
//! device, for the main lane's marker after the head; the main lane waits for every slot lane's marker before the
//! tail. Nothing waits on the host: the pass ends in a frame the lanes' completions emit behind the last marker.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

use euhedral_core::frames::Frame;
use euhedral_gpu::cuda::CudaQueue;
use euhedral_gpu::{GpuError, Lane, LaneEnqueue};
use euhedral_graph::{Completions, ControlError, DeviceQueue, Lake, LaneControl, Marker};
use euhedral_inference_cuda::launch::Modes;
use euhedral_inference_cuda::launch::load::KernelTable;
use euhedral_inference_cuda::launch::ops::RusteeOps;
use euhedral_inference_cuda::math::LaneHandles;
use euhedral_inference_cuda::{CudaLibrary, DevicePtr};
use euhedral_inference_engine::Handoff;

use super::{Attempt, Done, Scheduler, Stages};
use crate::forward::{ForwardError, ForwardKernels, LaneContext, LayerForward, LayerWeights, Observation, Slot};

const BF16_BYTES: u64 = 2;

/// What sees each quantized linear's input on the main lane, before the linear: the layer forward's observer, owned.
pub type PipelineObserver = Box<
    dyn for<'q, 'c> FnMut(
            &Observation<'_>,
            &mut LaneContext<'c, <CudaQueue as Lane>::Enqueue<'q>>,
        ) -> Result<(), ForwardError>
        + Send,
>;

/// The hidden states of every sequence of a pass in host memory, as BF16 rows `[sequences, length, hidden]`.
#[derive(Clone, Copy, Debug)]
pub struct HostStates {
    address: *mut u8,
    bytes: usize,
}

// SAFETY: the pointer names memory the caller keeps live and unused by anything else until the pass ended (the
// contract of `HostStates::new`); the pipeline reads and writes disjoint ranges of it on its lanes.
unsafe impl Send for HostStates {}
// SAFETY: as for `Send`: the value is a plain address; the contract is on the memory.
unsafe impl Sync for HostStates {}

impl HostStates {
    /// The `bytes` at `address`.
    ///
    /// # Safety
    /// The memory is valid for reads and writes of `bytes` bytes, ideally pinned, and neither read nor written by
    /// anything else, from the pass that is given this value until that pass's `done` was called.
    pub unsafe fn new(address: *mut u8, bytes: usize) -> Self {
        HostStates { address, bytes }
    }
}

/// One pass: what the layer runs over and how.
pub struct Request {
    /// The layer's weights.
    pub layer: Arc<LayerWeights>,
    /// The hidden states of every sequence; with `write` they are replaced by the layer's output.
    pub states: HostStates,
    /// The device buffer a batch runs in: BF16 `[batch * length, hidden]`.
    pub batch_states: DevicePtr,
    /// The sequences, each `length` tokens.
    pub sequences: u32,
    pub length: u32,
    /// The sequences run at once.
    pub batch: u32,
    /// Called with each quantized linear's input on the main lane.
    pub observer: Option<PipelineObserver>,
    /// Whether the outputs go back to `states`.
    pub write: bool,
}

/// What every frame of a pass reads: the request, with its observer resting in a cell the main lane's frames take.
pub struct Job {
    layer: Arc<LayerWeights>,
    states: HostStates,
    batch_states: DevicePtr,
    length: u32,
    write: bool,
    observer: Handoff<PipelineObserver>,
}

/// The main lane's own state: its math handles.
#[derive(Debug)]
pub struct MainLane {
    handles: LaneHandles,
}

/// The device calls of a pass.
pub struct DeviceStages {
    forward: LayerForward,
    library: Arc<CudaLibrary>,
    table: Arc<KernelTable>,
    kernels: Arc<ForwardKernels>,
    main: Arc<CudaQueue>,
    slots: Vec<Arc<CudaQueue>>,
    completions: Arc<Completions>,
}

impl Debug for Request {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Request").finish_non_exhaustive()
    }
}

impl Debug for Parts {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Parts").finish_non_exhaustive()
    }
}

impl Debug for Job {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Job").finish_non_exhaustive()
    }
}

impl Debug for DeviceStages {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("DeviceStages").finish_non_exhaustive()
    }
}

fn control(error: ControlError) -> Result<Attempt<Marker>, ForwardError> {
    match error {
        ControlError::Busy => Ok(Attempt::Busy),
        other => Err(ForwardError::Invalid(other.to_string())),
    }
}

/// What a submission came to, with the marker it recorded.
fn settle(made: Result<(Result<(), ForwardError>, Marker), GpuError>) -> Result<Attempt<Marker>, ForwardError> {
    match made {
        Ok((Ok(()), marker)) => Ok(Attempt::Ready(marker)),
        Ok((Err(error), _)) => Err(error),
        Err(GpuError::Busy) => Ok(Attempt::Busy),
        Err(error) => Err(error.into()),
    }
}

impl DeviceStages {
    fn home(lane: &Arc<CudaQueue>) -> Marker {
        Marker::new(DeviceQueue::id(&**lane), Lane::lane(&**lane), 0)
    }

    fn sequence_bytes(&self, length: u32) -> u64 {
        BF16_BYTES * u64::from(self.forward.shape().hidden) * u64::from(length)
    }

    /// The host range of the batch `first..first + count` and its size.
    fn range(&self, job: &Job, first: u32, count: u32) -> Result<(*mut u8, usize), ForwardError> {
        let per = self.sequence_bytes(job.length);
        let start = per * u64::from(first);
        let bytes = per * u64::from(count);
        let end = start + bytes;
        if end > job.states.bytes as u64 {
            return Err(ForwardError::Invalid(format!(
                "the batch ends at byte {end}, past the {} host bytes",
                job.states.bytes
            )));
        }
        let (start, bytes) = (
            usize::try_from(start).map_err(|_| ForwardError::Invalid("the host range overflows".to_owned()))?,
            usize::try_from(bytes).map_err(|_| ForwardError::Invalid("the host range overflows".to_owned()))?,
        );
        Ok((job.states.address.wrapping_add(start), bytes))
    }
}

impl Stages for DeviceStages {
    type Main = MainLane;
    type Slot = Slot;
    type Mark = Marker;
    type Pass = Job;

    fn head(&self, pass: &Job, main: &mut MainLane, first: u32, count: u32) -> Result<Attempt<Marker>, ForwardError> {
        let (host, bytes) = self.range(pass, first, count)?;
        let rows = count * pass.length;
        let mut observer = pass.observer.take();
        let home = Self::home(&self.main);
        let made = self.main.submit_marked(|queue| {
            // SAFETY: `host` is `bytes` bytes of the caller's pinned memory, kept until the pass ended, and the
            // batch buffer holds a batch; the lane orders the copy before the head.
            let copied = unsafe { queue.copy_to_device(pass.batch_states.get(), host, bytes) };
            if let Err(error) = copied {
                return Ok(Err(ForwardError::from(error)));
            }
            let ops = RusteeOps::new(queue, &self.table, Modes::default()).on_lane(&self.library, home);
            let mut context = LaneContext::new(ops, &mut main.handles, &self.kernels);
            let mut call = |observation: &Observation<'_>, context: &mut LaneContext<'_, _>| match observer.as_mut() {
                Some(observer) => observer(observation, context),
                None => Ok(()),
            };
            Ok(self
                .forward
                .head(&mut context, &pass.layer, pass.batch_states, rows, Some(&mut call)))
        });
        if let Some(observer) = observer {
            pass.observer.send(observer);
        }
        settle(made)
    }

    fn sequence(
        &self,
        pass: &Job,
        slot: &mut Slot,
        index: u32,
        after: Marker,
    ) -> Result<Attempt<Marker>, ForwardError> {
        let lane = self
            .slots
            .get(index as usize)
            .ok_or_else(|| ForwardError::Invalid(format!("sequence {index} has no lane")))?;
        if let Err(error) = lane.await_marker(after) {
            return control(error);
        }
        let home = Self::home(lane);
        settle(lane.submit_marked(|queue| {
            let mut ops = RusteeOps::new(queue, &self.table, Modes::default()).on_lane(&self.library, home);
            Ok(self
                .forward
                .sequence_in(&mut ops, &pass.layer, slot, index, pass.length))
        }))
    }

    fn tail(
        &self,
        pass: &Job,
        main: &mut MainLane,
        first: u32,
        count: u32,
        last: bool,
        marks: &[Marker],
    ) -> Result<Attempt<Option<Marker>>, ForwardError> {
        let (host, bytes) = self.range(pass, first, count)?;
        let rows = count * pass.length;
        for mark in marks {
            if let Err(error) = self.main.await_marker(*mark) {
                return match control(error)? {
                    Attempt::Busy => Ok(Attempt::Busy),
                    Attempt::Ready(_) => Ok(Attempt::Busy),
                };
            }
        }
        let mut observer = pass.observer.take();
        let home = Self::home(&self.main);
        let write = pass.write;
        let made = self.main.submit_marked(|queue| {
            {
                let ops = RusteeOps::new(&mut *queue, &self.table, Modes::default()).on_lane(&self.library, home);
                let mut context = LaneContext::new(ops, &mut main.handles, &self.kernels);
                let mut call = |observation: &Observation<'_>, context: &mut LaneContext<'_, _>| match observer.as_mut()
                {
                    Some(observer) => observer(observation, context),
                    None => Ok(()),
                };
                if let Err(error) =
                    self.forward
                        .tail(&mut context, &pass.layer, pass.batch_states, rows, Some(&mut call))
                {
                    return Ok(Err(error));
                }
            }
            if write {
                // SAFETY: as for the copy in, in the other direction; the lane orders the copy after the tail.
                let copied = unsafe { queue.copy_to_host(host, pass.batch_states.get(), bytes) };
                if let Err(error) = copied {
                    return Ok(Err(ForwardError::from(error)));
                }
            }
            Ok(Ok(()))
        });
        if let Some(observer) = observer {
            pass.observer.send(observer);
        }
        Ok(match settle(made)? {
            Attempt::Ready(marker) => Attempt::Ready(last.then_some(marker)),
            Attempt::Busy => Attempt::Busy,
        })
    }

    fn fence(&self, main: &mut MainLane) -> Result<Attempt<Marker>, ForwardError> {
        let _ = main;
        for lane in &self.slots {
            let mark = match lane.mark() {
                Ok(mark) => mark,
                Err(GpuError::Busy) => return Ok(Attempt::Busy),
                Err(error) => return Err(error.into()),
            };
            if let Err(error) = self.main.await_marker(mark) {
                return control(error);
            }
        }
        match self.main.mark() {
            Ok(mark) => Ok(Attempt::Ready(mark)),
            Err(GpuError::Busy) => Ok(Attempt::Busy),
            Err(error) => Err(error.into()),
        }
    }

    fn after(&self, mark: Marker, then: Box<dyn Frame>) -> Result<(), ForwardError> {
        self.completions.after_marker(mark, then);
        Ok(())
    }

    fn retire(&self, slots: &mut [Slot]) {
        for slot in slots {
            slot.retire();
        }
    }
}

/// What a pipeline is made of.
pub struct Parts {
    /// The layer forward (workspace for `batch * length` rows).
    pub forward: LayerForward,
    pub library: Arc<CudaLibrary>,
    pub table: Arc<KernelTable>,
    pub kernels: Arc<ForwardKernels>,
    /// The main lane's math handles; the lane runs the heads, the tails and the copies.
    pub handles: LaneHandles,
    /// The main lane.
    pub main: Arc<CudaQueue>,
    /// One lane per sequence of a batch.
    pub slots: Vec<Arc<CudaQueue>>,
    /// The completions that poll those lanes.
    pub completions: Arc<Completions>,
    /// Where the pass's frames run.
    pub lake: Arc<Lake>,
}

/// The pipelined layer forward: layers in order, up to a batch of sequences in flight, no host wait.
///
/// Java's `ForwardPipeline`. A pass of a layer over every sequence is a chain of frames on the lake: a start that
/// copies a batch in and runs the head on the main lane, one frame per sequence on its own lane behind the head, a
/// finish that runs the tail behind every sequence and copies the batch back, the next batch's start, and the end
/// that the lanes' completions emit once the last marker is reached.
#[derive(Debug)]
pub struct ForwardPipeline {
    scheduler: Scheduler<DeviceStages>,
    main: Arc<CudaQueue>,
}

impl ForwardPipeline {
    /// A pipeline of `slots` sequences in flight. Makes the sequences' states on the main lane, as Java's constructor
    /// does (the first call waits for pinned memory when the pool is cold).
    pub fn new(parts: Parts) -> Result<Self, ForwardError> {
        let Parts {
            mut forward,
            library,
            table,
            kernels,
            handles,
            main,
            slots,
            completions,
            lake,
        } = parts;
        if slots.is_empty() {
            return Err(ForwardError::Invalid("a pipeline of no slots".to_owned()));
        }
        let count = u32::try_from(slots.len()).map_err(|_| ForwardError::Invalid("too many slots".to_owned()))?;
        let home = DeviceStages::home(&main);
        let prepared = main.submit_marked(|queue| {
            let mut ops = RusteeOps::new(queue, &table, Modes::default()).on_lane(&library, home);
            Ok(forward.prepare_slots(&mut ops, count))
        });
        match prepared {
            Ok((Ok(()), _)) => {}
            Ok((Err(error), _)) => return Err(error),
            Err(error) => return Err(error.into()),
        }
        let states = forward.take_slots();
        let stages = DeviceStages {
            forward,
            library,
            table,
            kernels,
            main: Arc::clone(&main),
            slots,
            completions,
        };
        Ok(ForwardPipeline {
            scheduler: Scheduler::new(stages, lake, MainLane { handles }, states),
            main,
        })
    }

    /// Starts one pass of `layer` (see [`Request`]). `done` is called once from a frame, after the device is through
    /// with everything the pass queued, with the pass's result. An `Err` here started nothing.
    pub fn run(&self, request: Request, done: Done) -> Result<(), ForwardError> {
        let Request {
            layer,
            states,
            batch_states,
            sequences,
            length,
            batch,
            observer,
            write,
        } = request;
        let hold = Handoff::empty();
        if let Some(observer) = observer {
            hold.send(observer);
        }
        let job = Job {
            layer,
            states,
            batch_states,
            length,
            write,
            observer: hold,
        };
        self.scheduler.run(job, sequences, batch, done)
    }

    /// Frees the layer's workspace and the states in stream order on the main lane and closes its handles, once no
    /// pass is running (a pass is over when its `done` was called). Returns the marker after the frees; the lane has
    /// retired them when it is reached.
    pub fn finish(self) -> Result<Marker, ForwardError> {
        let main = self.main;
        let Ok((stages, lane, slots)) = self.scheduler.into_parts() else {
            return Err(ForwardError::Invalid("a pass is still running".to_owned()));
        };
        let DeviceStages {
            mut forward,
            library,
            table,
            ..
        } = stages;
        forward.put_slots(slots);
        let home = DeviceStages::home(&main);
        let released = main.submit_marked(|queue| {
            let mut ops = RusteeOps::new(queue, &table, Modes::default()).on_lane(&library, home);
            Ok(forward.release(&mut ops))
        });
        let closed = lane.handles.close();
        let marker = match released {
            Ok((Ok(()), marker)) => marker,
            Ok((Err(error), _)) => return Err(error),
            Err(error) => return Err(error.into()),
        };
        closed?;
        Ok(marker)
    }
}
