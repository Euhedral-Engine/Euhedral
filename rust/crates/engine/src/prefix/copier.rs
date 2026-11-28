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
use euhedral_memory::Claim;

use crate::frames::{Kind, id_hash};
use crate::model::{LayerKind, ModelConfig};
use crate::prefix::copy::{CopyJob, Direction, Ended, HostArena, Resolve};
use crate::prefix::layout::Buffer;
use crate::quantum::{StartError, Started, no_slot_readers};
use crate::stages::attention::KvSource;
use crate::stages::{LaneTable, Retirement, SequenceRefs, Unmarked, fatal, submit_on};
use crate::state::SequenceId;

/// The first number a copy job takes. A quantum's number counts up from 1, so the jobs' numbers, which count up from
/// here, never meet a quantum's, and neither meets the "no gate" value ([`u64::MAX`]).
pub(crate) const FIRST_COPY_SERIAL: u64 = 1 << 63;

/// The cache's host arena: one pinned allocation made at startup (a raw pinned buffer is never freed on a running
/// engine), addressed by the offsets the tree's slots carry.
#[derive(Debug)]
pub struct PinnedArena {
    buffer: PinnedBuffer,
    /// The account's claim for the buffer's bytes, declared after the buffer so that it is released once the buffer is
    /// freed. The copy jobs hold the arena, so an abandoned job keeps the claim as long as the memory.
    claim: Option<Claim>,
}

impl PinnedArena {
    /// Pins `bytes` of host memory. Waits on the allocator: startup only.
    pub fn allocate(library: &Arc<CudaLibrary>, bytes: u64) -> Result<PinnedArena, CudaError> {
        let bytes =
            usize::try_from(bytes).map_err(|_| CudaError::Argument(format!("{bytes} bytes do not fit in memory")))?;
        Ok(PinnedArena {
            buffer: PinnedBuffer::alloc(library, bytes)?,
            claim: None,
        })
    }

    /// The arena, accounted to `claim` for as long as it lives.
    pub fn with_claim(mut self, claim: Claim) -> PinnedArena {
        self.claim = Some(claim);
        self
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
    /// The bytes of one hidden row (one row of the draft seed rows).
    hidden_row_bytes: u64,
    /// The row of the draft seed rows that holds the base hidden row of the checkpoint's last position: the last row
    /// of the chunk that ended at a checkpoint, row 0 for the restore into a sequence about to prefill.
    seed_row: u32,
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
            hidden_row_bytes: config.hidden_size as u64 * 2,
            seed_row: 0,
        })
    }

    /// The seed row the draft's checkpoint state is read from or written to is row `row` of the draft seed rows.
    pub fn with_seed_row(mut self, row: u32) -> SequenceAddresses {
        self.seed_row = row;
        self
    }
}

impl SequenceAddresses {
    /// The references the addresses are of.
    pub fn refs(&self) -> &SequenceRefs {
        &self.refs
    }
}

/// The offset into the draft seed rows of byte `offset` of row `row`.
pub(crate) fn seed_offset(row: u32, row_bytes: u64, offset: u64) -> u64 {
    u64::from(row) * row_bytes + offset
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
            Buffer::DrafterKeys { layer } => {
                let rings = device.scratch.drafter().filter(|rings| layer < rings.layers())?;
                within(Some(rings.ring_keys(layer)), rings.ring_bytes(), offset, bytes)
            }
            Buffer::DrafterValues { layer } => {
                let rings = device.scratch.drafter().filter(|rings| layer < rings.layers())?;
                within(Some(rings.ring_values(layer)), rings.ring_bytes(), offset, bytes)
            }
            Buffer::SeedRow => {
                let rows = &device.scratch.draft()?.seed_rows;
                let offset = seed_offset(self.seed_row, self.hidden_row_bytes, offset);
                within(Some(rows.ptr()), rows.bytes() as u64, offset, bytes)
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
            for piece in job.pieces().iter().filter(|_| !job.skipped()) {
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
    /// The copier of `sequence`, whose jobs run on transfer lane `lane` of `lanes`; `ended` is told the number of each job
    /// once it has ended and been reported.
    pub fn new(
        runtime: &GraphRuntime,
        lanes: LaneTable,
        lane: LaneId,
        sequence: SequenceId,
        ended: Box<dyn Fn(u64) + Send + Sync>,
    ) -> Result<Copier, GraphError> {
        let mut spec = StageGraphSpec::new();
        spec.stage(StageSpec {
            name: "prefix-copy",
            ordered: Some(id_hash(Kind::Copy, sequence.0)),
            lanes: vec![lane],
        });
        let unmarked = Unmarked::new(lanes.len());
        let stage: Arc<dyn Stage<CopyJob>> = Arc::new(CopyStage { lanes, unmarked });
        let graph = StageGraph::new(spec, vec![stage], runtime.clone(), move |mut job: CopyJob, outcome| {
            let (serial, last) = (job.serial(), job.is_last());
            job.finish(match outcome {
                QuantumOutcome::Completed => Ended::Done,
                QuantumOutcome::Failed { error, .. } => Ended::Failed(format!("the copy failed: {error:?}")),
                QuantumOutcome::DeviceFailed { cause } => Ended::Failed(format!("the device failed: {cause}")),
                QuantumOutcome::Killed => Ended::Cancelled,
            });
            // Whatever waited for the job's end (see the order) starts now, and the job concludes in its sequence's order.
            if last {
                ended(serial);
                if let Some(place) = job.take_place() {
                    place.complete(Retirement::Copy);
                }
            }
        })?;
        Ok(Copier { graph })
    }

    /// Starts `job` behind each of `after`: its first group's stage waits on the device for their sinks. The job runs
    /// in groups of at most the piece size, a frame each, each behind the one before; the last, which reports, is the
    /// one returned.
    pub fn start(&self, job: CopyJob, kill: Arc<AtomicBool>, after: &[&Started]) -> Result<Started, StartError> {
        let mut last: Option<Started> = None;
        for group in job.split() {
            let started = match &last {
                Some(previous) => self.start_group(group, Arc::clone(&kill), &[previous]),
                None => self.start_group(group, Arc::clone(&kill), after),
            };
            last = Some(started);
        }
        // A job has at least one group.
        last.ok_or(StartError::Closed)
    }

    fn start_group(&self, group: CopyJob, kill: Arc<AtomicBool>, after: &[&Started]) -> Started {
        let serial = group.serial();
        let external = after.iter().map(|previous| ExternalEdge {
            from: previous.reference.clone(),
            from_stage: previous.sink,
            to_stage: StageId(0),
            kind: EdgeKind::Submission,
        });
        let reference = self.graph.start(group, kill, external);
        Started {
            serial,
            reference,
            root: StageId(0),
            sink: StageId(0),
            slot_readers: no_slot_readers(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::seed_offset;

    #[test]
    fn the_seed_row_of_a_checkpoint_is_the_last_row_of_its_chunk() {
        // A chunk of 512 rows of 5120-byte hidden rows ends at row 511 of the seed rows, not at row 0.
        assert_eq!(seed_offset(511, 5120, 0), 511 * 5120);
        assert_eq!(seed_offset(0, 5120, 0), 0);
        // A copy that starts inside the row stays inside it.
        assert_eq!(seed_offset(3, 5120, 16), 3 * 5120 + 16);
    }
}
