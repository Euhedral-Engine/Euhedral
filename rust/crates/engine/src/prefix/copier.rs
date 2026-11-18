//! The device side of the prefix cache's copy jobs: the stage graph that runs a [`CopyJob`] on a transfer lane, the
//! pinned host arena the jobs address, and the addresses of a sequence's state they copy to and from.
//!
//! A job is one stage on one transfer lane: the whole job is one marked submission, whose copies are cut to at most the
//! piece size, and the job ends (and tells the cache) when the lane reaches that marker. The graph's stage is *ordered*
//! on the sequence's own hash ([`Kind::Copy`] of the sequence), so the stage frames and the completions of one
//! sequence's jobs run in the order they were admitted, and the jobs of two sequences never share a chain.

use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use euhedral_graph::{
    EdgeKind, ExternalEdge, GraphError, GraphRuntime, LaneId, Marker, QuantumOutcome, Stage, StageCx, StageError,
    StageGraph, StageGraphSpec, StageId, StageSpec, Submitted,
};
use euhedral_inference_cuda::launch::ops::RusteeOps;
use euhedral_inference_cuda::{CudaError, CudaLibrary, DevicePtr, KernelOps, PinnedBuffer};
use euhedral_inference_settings::KvFormat;

use crate::frames::{Kind, id_hash};
use crate::model::{LayerKind, ModelConfig};
use crate::prefix::copy::{CopyJob, Direction, Ended, HostArena, Resolve};
use crate::prefix::layout::Buffer;
use crate::quantum::{StartError, Started};
use crate::stages::attention::KvSource;
use crate::stages::{LaneTable, SequenceRefs, Unmarked, fatal, submit_on};
use crate::state::SequenceId;

/// The first number a copy job takes. A quantum's number counts up from 1, so the jobs' numbers, which count up from
/// here, never meet a quantum's, and neither meets the "no gate" value ([`u64::MAX`]).
pub(crate) const FIRST_COPY_SERIAL: u64 = 1 << 63;

/// The cache's host arena: one pinned allocation made at startup (a raw pinned buffer is never freed on a running
/// engine), addressed by the offsets the tree's slots carry.
#[derive(Debug)]
pub struct PinnedArena {
    buffer: PinnedBuffer,
}

impl PinnedArena {
    /// Pins `bytes` of host memory. Waits on the allocator: startup only.
    pub fn allocate(library: &Arc<CudaLibrary>, bytes: u64) -> Result<PinnedArena, CudaError> {
        let bytes =
            usize::try_from(bytes).map_err(|_| CudaError::Argument(format!("{bytes} bytes do not fit in memory")))?;
        Ok(PinnedArena {
            buffer: PinnedBuffer::alloc(library, bytes)?,
        })
    }

    /// The arena's size in bytes.
    pub fn bytes(&self) -> u64 {
        self.buffer.len() as u64
    }
}

impl HostArena for PinnedArena {
    fn pinned(&self) -> Option<&PinnedBuffer> {
        Some(&self.buffer)
    }
}

/// The state of one sequence a job copies, by the references a quantum of it would hold: the device addresses of its
/// buffers, as they are when the references were taken. Holding them keeps the pages and buffers alive until the job
/// is dropped.
#[derive(Debug)]
pub struct SequenceAddresses {
    refs: SequenceRefs,
    /// By model layer: the full-attention layer's place among the cache's layers.
    attention: Vec<Option<usize>>,
    format: KvFormat,
}

impl SequenceAddresses {
    /// The addresses of the state `refs` names, for a model of `config`; `None` when `refs` holds no base cache.
    pub fn new(refs: SequenceRefs, config: &ModelConfig) -> Option<SequenceAddresses> {
        let format = refs.kv(KvSource::Base)?.format();
        let mut next = 0;
        let attention = config
            .layer_kinds
            .iter()
            .map(|kind| {
                (*kind == LayerKind::FullAttention).then(|| {
                    next += 1;
                    next - 1
                })
            })
            .collect();
        Some(SequenceAddresses {
            refs,
            attention,
            format,
        })
    }
}

impl SequenceAddresses {
    /// The references the addresses are of.
    pub fn refs(&self) -> &SequenceRefs {
        &self.refs
    }
}

/// The device address of `offset` bytes into `base`, when `bytes` more fit within `size`.
fn within(base: Option<DevicePtr>, size: u64, offset: u64, bytes: u64) -> Option<u64> {
    let base = base.filter(|base| !base.is_null())?;
    (offset.checked_add(bytes)? <= size).then(|| base.offset(offset).get())
}

impl Resolve for SequenceAddresses {
    fn address(&self, buffer: Buffer, offset: u64, bytes: u64) -> Option<u64> {
        let device = self.refs.device();
        match buffer {
            Buffer::Convolution { layer } => {
                let state = device.gdn.layer(layer)?.convolution();
                within(Some(state.ptr()), state.bytes() as u64, offset, bytes)
            }
            Buffer::Recurrent { layer } => {
                let state = device.gdn.layer(layer)?.recurrent();
                within(Some(state.ptr()), state.bytes() as u64, offset, bytes)
            }
            Buffer::KvPage { layer, page } => {
                let view = self.refs.kv(KvSource::Base)?;
                let ordinal = (*self.attention.get(layer)?)?;
                within(view.page(ordinal, page), 2 * view.plane_page_bytes(), offset, bytes)
            }
            Buffer::MtpPage { page } => {
                let view = self.refs.kv(KvSource::Mtp)?;
                within(view.page(0, page), 2 * view.plane_page_bytes(), offset, bytes)
            }
            Buffer::SeedRow => {
                let seed = &device.scratch.draft()?.seed_rows;
                within(Some(seed.ptr()), seed.bytes() as u64, offset, bytes)
            }
        }
    }

    fn kv_format(&self) -> KvFormat {
        self.format
    }

    fn rows(&self) -> u64 {
        self.refs.kv(KvSource::Base).map_or(0, |view| view.reserved())
    }

    fn ready(&self) -> Option<Marker> {
        self.refs.kv(KvSource::Base).and_then(|view| view.ready())
    }
}

/// The one stage of the copy graph.
struct CopyStage {
    lanes: LaneTable,
    unmarked: Unmarked,
}

impl Stage<CopyJob> for CopyStage {
    fn run(&self, job: &CopyJob, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let id = cx.lane.ok_or_else(|| fatal("a copy job was given no lane"))?;
        let lane = self
            .lanes
            .get(id)
            .ok_or_else(|| fatal(format!("lane {} is not in the lane table", id.0)))?;
        let arena = job
            .host()
            .pinned()
            .ok_or_else(|| fatal("a copy job's arena is not pinned"))?;
        let modes = self.lanes.modes(false);
        let table = self.lanes.kernels();
        let waits = cx.waits.iter().copied().chain(job.ready());
        let marker = submit_on(&**lane, waits, &self.unmarked, |queue| {
            let mut ops = RusteeOps::new(queue, table, modes);
            for piece in job.pieces() {
                let (host, bytes) = (piece.host as usize, piece.bytes as usize);
                if host.checked_add(bytes).is_none_or(|end| end > arena.len()) {
                    return Err(fatal(format!("a copy of {bytes} bytes at {host} leaves the arena")));
                }
                // SAFETY: the address came from a buffer the job's `Resolve` holds, which the job keeps alive until
                // it is dropped, after the lane has reached this submission's marker.
                let device = unsafe { DevicePtr::from_raw(piece.device) };
                match job.direction() {
                    // SAFETY: the job holds the arena (an `Arc`) until it is dropped, which is after the marker that
                    // follows this copy was reached; the slot range is the job's alone (pinned or unpublished in the
                    // tree), and the host does not touch the arena except through jobs.
                    Direction::ToHost => unsafe { ops.copy_device_to_readback_shared(arena, host, device, bytes) },
                    // SAFETY: as above; the arena bytes are not modified while a restore reads them (the slots are
                    // pinned by the match).
                    Direction::ToDevice => unsafe { ops.copy_upload_to_device(device, arena, host, bytes) },
                }
                .map_err(fatal)?;
            }
            Ok(())
        })?;
        Ok(Submitted::marker(marker))
    }
}

/// Runs the copy jobs of one sequence.
#[derive(Debug)]
pub struct Copier {
    graph: Arc<StageGraph<CopyJob>>,
}

impl Copier {
    /// The copier of `sequence`, whose jobs run on transfer lane `lane` of `lanes`.
    pub fn new(
        runtime: &GraphRuntime,
        lanes: LaneTable,
        lane: LaneId,
        sequence: SequenceId,
    ) -> Result<Copier, GraphError> {
        let mut spec = StageGraphSpec::new();
        spec.stage(StageSpec {
            name: "prefix-copy",
            ordered: Some(id_hash(Kind::Copy, sequence.0)),
            lanes: vec![lane],
        });
        let unmarked = Unmarked::new(lanes.len());
        let stage: Arc<dyn Stage<CopyJob>> = Arc::new(CopyStage { lanes, unmarked });
        let graph = StageGraph::new(spec, vec![stage], runtime.clone(), |mut job: CopyJob, outcome| {
            job.finish(match outcome {
                QuantumOutcome::Completed => Ended::Done,
                QuantumOutcome::Failed { error, .. } => Ended::Failed(format!("the copy failed: {error:?}")),
                QuantumOutcome::DeviceFailed { cause } => Ended::Failed(format!("the device failed: {cause}")),
                QuantumOutcome::Killed => Ended::Failed("the copy was cancelled".to_string()),
            });
        })?;
        Ok(Copier { graph })
    }

    /// Starts `job` behind each of `after`: its stage waits on the device for their sinks.
    pub fn start(&self, job: CopyJob, kill: Arc<AtomicBool>, after: &[&Started]) -> Result<Started, StartError> {
        let serial = job.serial();
        let external = after.iter().map(|previous| ExternalEdge {
            from: previous.reference.clone(),
            from_stage: previous.sink,
            to_stage: StageId(0),
            kind: EdgeKind::Submission,
        });
        let reference = self.graph.start(job, kill, external);
        Ok(Started {
            serial,
            reference,
            root: StageId(0),
            sink: StageId(0),
        })
    }
}
