//! Generation: prefill, greedy decode and teacher-forced decoding on the engine's sequences.
//!
//! [`EngineRuntime::generate`](crate::EngineRuntime::generate) starts a greedy request and
//! [`EngineRuntime::teacher_forced`](crate::EngineRuntime::teacher_forced) a teacher-forced run, each on an idle
//! sequence from the pool. Both are carried by the sequence's owner (one for each sequence), which sits in a [`Mailbox`](mailbox::Mailbox): the frames that tell it something (a quantum
//! retired, a logits row arrived, a parked quantum's lane work ran) post the message and run the owner's turn on their
//! own worker if no other turn runs. The caller is told through its listener or sink, on a worker, and waits on a
//! hand-off of its own.

mod acceptance;
mod constraint;
mod draft_length;
mod hidden;
mod listener;
mod mailbox;
mod mtp;
mod order;
mod pool;
mod request;
mod sampled;
mod source;
mod teacher_forced;

use std::fmt::{self, Formatter};
use std::sync::Arc;
#[cfg(feature = "test-oracles")]
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::generics::FrameSink;
use euhedral_data_structures::queues::MpscQueue;
use euhedral_graph::{DEFAULT_RETRY_BACKSTOP, GraphRuntime, RetryCounts};
use euhedral_inference_cuda::{CudaLibrary, DevicePtr};
use euhedral_inference_edrl::WeightFormat;
use euhedral_memory::Claim;

use crate::error::EngineError;
use crate::formats::{Nvfp4Linear, Q3Linear, TrellisLinear};
use crate::frames::{Kind, step};
use crate::model::{LayerIndex, ModelConfig, Role, Weights};
use crate::pass_assembler::PassStatistics;
use crate::plan::{Formats, Plan, StageKind, StagingRing, View};
use crate::prefix::{
    CacheSpec, Copier, POSITION_GRANULE, PrefixCacheHandle, PrefixCacheStats, open_prefix_cache_claimed,
};
#[cfg(feature = "test-oracles")]
use crate::quantum::body_fault::{BodyFault, Defect, HoldEvent, RunSnapshot, Sighting};
use crate::quantum::{Executor, Preparer};
use crate::residency::offload::StagingGeometry;
use crate::residency::planner::ResidencyPlan;
use crate::runtime::{Core, STAGE_RING_QUANTA, Units};
use crate::state::{SequenceId, SequenceState};
use crate::workspace::Workspace;
use crate::workspace::{input_records, input_records_for};
pub use acceptance::{Accepted, accept, accept_tokens};
pub use constraint::{ConstraintError, ReasoningConstraint, TokenConstraint};
pub use draft_length::{DraftLength, DraftLengthError, MAX_DRAFTS, MAX_VERIFIED};
#[cfg(feature = "test-oracles")]
use euhedral_gpu::GraphStats;
pub use hidden::HiddenSink;
pub use listener::GenerationListener;
pub(crate) use mailbox::OPERATION_BOUND;
pub use mailbox::Spawn;
pub(crate) use mailbox::{Address, Mailbox, Turn};
use order::{Order, PassCaps, Reply};
use pool::{Lease, SequencePool};
pub use request::{
    AdmissionError, CancelHandle, FinishReason, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
    Verification,
};
pub(crate) use source::GREEDY_DEPTH;
use source::{Audience, Drafts, Message, Owner, Rejected, Retired, Shared, StartFailed};
pub use teacher_forced::LogitsSink;

/// The engine's admission of a request, as the answer `Owner::begin` takes.
fn admit(core: &Core) -> Result<Units, AdmissionError> {
    core.admit().map_err(|error| AdmissionError::Engine(error.to_string()))
}

/// Why a request is refused once the owner is closed.
fn closed_error() -> AdmissionError {
    AdmissionError::Engine("the engine is closed".to_string())
}

/// The owner a request may begin on: none once the engine stops, so that the request is refused as one that found the
/// owner closed.
fn unless_closing<'a, T>(closing: &AtomicBool, owner: Option<&'a mut T>) -> Option<&'a mut T> {
    owner.filter(|_| !closing.load(Ordering::SeqCst))
}

/// A request that found the owner closed.
fn closed(audience: Audience) -> Rejected {
    Rejected::new(closed_error(), audience)
}

/// What the engine keeps for generating: the plan's graphs on the runtime's lanes, and the owner of the sequence.
pub(crate) struct Generation {
    shared: Arc<Shared>,
    /// The owner, and the messages posted to it.
    mailbox: Arc<Mailbox<Owner>>,
}

impl fmt::Debug for Generation {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Generation").finish_non_exhaustive()
    }
}

/// The staging ring of a plan whose weights are staged, over the device buffer at `base`: `None` when nothing is
/// staged. A ring with no buffer, or a plan that stages with no pinned host arena to copy from, is an error.
pub fn staging_ring(
    geometry: Option<StagingGeometry>,
    base: Option<DevicePtr>,
    has_host_arena: bool,
) -> Result<Option<StagingRing>, String> {
    let Some(geometry) = geometry else {
        return Ok(None);
    };
    let base = base.ok_or_else(|| "the plan stages weights and the engine allocated no staging ring".to_string())?;
    if !has_host_arena {
        return Err("the plan stages weights and holds no pinned host arena to copy them from".to_string());
    }
    Ok(Some(StagingRing {
        base,
        slots: geometry.slots,
        slot_bytes: geometry.slot_bytes,
    }))
}

/// Whether the rings of `ring` events hold the markers `sequences` sequences of `plan` keep in flight, on the stage lanes
/// and on the staging lane. The error names the lane.
pub fn check_marker_rings(plan: &Plan, sequences: usize, ring: usize) -> Result<(), String> {
    for (needed, what) in [
        (markers_in_flight(plan, sequences, false), "stage"),
        (markers_in_flight(plan, sequences, true), "staging"),
    ] {
        if needed > ring {
            return Err(format!(
                "the quanta in flight of {sequences} sequences record {needed} markers on a {what} lane and the \
                 lanes' rings of {ring} events do not hold them"
            ));
        }
    }
    Ok(())
}

/// The markers a lane's ring must hold for `sequences` sequences of `plan`: those of the stages that run on it. A
/// quantum records a marker per stage, so the stage lanes hold the compute stages (`transfers` false) and the staging
/// lane the weight transfers (`transfers` true), which the unstaged stage bound the rings are sized from does not count.
pub fn markers_in_flight(plan: &Plan, sequences: usize, transfers: bool) -> usize {
    let stages_of = |view: View| {
        plan.stages(view).map_or(0, |entries| {
            entries
                .iter()
                .filter(|entry| (entry.kind == StageKind::WeightTransfer) == transfers)
                .count()
        })
    };
    let biggest = [
        View::Decode,
        View::PrefillChunk,
        View::Verify { rows: 2 },
        View::Draft { rows: 1 },
    ]
    .into_iter()
    .map(stages_of)
    .max()
    .unwrap_or(0);
    // A greedy request keeps a few quanta in flight; a speculative step keeps its catch-up, drafts and verification,
    // and the owner prepares one more behind them.
    let step = if plan.drafts() {
        stages_of(View::Verify {
            rows: plan.verified_rows(),
        }) + stages_of(View::Draft { rows: 1 }) * plan.verified_rows()
            + biggest
    } else {
        0
    };
    // Every sequence keeps that many in flight on the same lanes.
    (biggest * STAGE_RING_QUANTA).max(step) * sequences.max(1)
}

impl Generation {
    /// Builds the plan of the loaded model and the graphs of its views, and makes `sequence`, numbered `id`, their
    /// owner's. `address` is where the owner will be (the workspace's owner tells it of quanta that did not start), and
    /// `order` is the engine's, shared by every sequence; `sequences` is how many sequences the engine runs, which the
    /// lanes' rings must hold the quanta of. A refused plan gives the sequence back. `draft_ids` are the token ids of
    /// the draft head's rows, which an artifact with a draft layer needs to speculate.
    #[allow(
        clippy::too_many_arguments,
        reason = "the startup's state, and where this sequence sits in the engine"
    )]
    fn new(
        core: &Arc<Core>,
        config: &Arc<ModelConfig>,
        residency: &ResidencyPlan,
        weights: &Weights,
        workspace: &Arc<Workspace>,
        sequence: SequenceState,
        id: SequenceId,
        address: Arc<Address<Owner>>,
        order: &Arc<Order<Executor>>,
        peers: &Arc<[Arc<Address<Owner>>]>,
        sequences: usize,
        draft_ids: Option<Arc<[u32]>>,
        prefix: Option<PrefixCacheHandle>,
    ) -> Result<Generation, Box<(EngineError, SequenceState)>> {
        let capacity = sequence.kv.planned_rows().min(residency.max_context_tokens as u64);
        // An unbounded queue, whose chunk holds the messages the protocol has in flight: the quanta in flight retire into
        // it, a teacher-forced step posts one row, and a parked quantum posts that its lane work ran. Offering to it
        // cannot fail, so no message is ever dropped.
        let queue = MpscQueue::<Message>::new(GREEDY_DEPTH + 4);
        let (producer, inbox) = (queue.producer(), queue.consumer());
        let copies = prefix.as_ref().map(|_| (id, Arc::clone(order)));
        let executor = match Self::executor(
            core, config, residency, weights, &address, order, peers, sequences, copies,
        ) {
            Ok(executor) => executor,
            Err(error) => return Err(Box::new((error, sequence))),
        };
        let drafts = match (draft_ids, weights.get(LayerIndex::GLOBAL, Role::DraftTokenIds)) {
            (Some(token_ids), Some(device)) if residency.draft_depth > 0 && sequence.mtp_kv.is_some() => Some(Drafts {
                token_ids,
                device_token_ids: device.ptr,
                depth: residency.draft_depth.min(MAX_VERIFIED),
            }),
            _ => None,
        };
        let shared = Arc::new(Shared {
            core: Arc::clone(core),
            config: Arc::clone(config),
            workspace: Arc::clone(workspace),
            executor: Arc::new(executor),
            order: Arc::clone(order),
            address: Arc::clone(&address),
            chunk: residency.rows.rows as usize,
            capacity,
            drafts,
            prefix,
            closing: AtomicBool::new(false),
            #[cfg(feature = "test-oracles")]
            panic_after_admissions: AtomicU64::new(u64::MAX),
            #[cfg(feature = "test-oracles")]
            settle_after_speculation: AtomicBool::new(false),
            #[cfg(feature = "test-oracles")]
            copy_admitted: OnceLock::new(),
            #[cfg(feature = "test-oracles")]
            restore_reserved: OnceLock::new(),
            #[cfg(feature = "test-oracles")]
            quantum_admitted: OnceLock::new(),
        });
        let mailbox = Arc::new(Mailbox::new(
            Owner::new(Arc::clone(&shared), id, sequence, inbox),
            producer,
            Self::spawner(core),
        ));
        address.set(&mailbox);
        Ok(Generation { shared, mailbox })
    }

    /// A frame that drives an owner's mailbox from a thread that must not run its turn, or after a frame's turns are
    /// used: unordered (a random routing hash), so that it may run on any worker, and so that no two owners' frames
    /// share a chain in the lake.
    pub(crate) fn spawner(core: &Arc<Core>) -> Spawn {
        let stages = Arc::clone(&core.stages);
        let seeds = AtomicU64::new(0);
        Box::new(move |work| {
            let mut frame = step(Kind::Owner, 0, work);
            frame.header_mut().randomize_hash(seeds.fetch_add(1, Ordering::Relaxed));
            stages.offer(frame).is_ok()
        })
    }

    /// The stage graphs of the model's plan, whose quanta retire into the owner's mailbox.
    #[allow(
        clippy::too_many_arguments,
        reason = "the startup's state, and where this sequence sits in the engine"
    )]
    fn executor(
        core: &Arc<Core>,
        config: &Arc<ModelConfig>,
        residency: &ResidencyPlan,
        weights: &Weights,
        address: &Arc<Address<Owner>>,
        order: &Arc<Order<Executor>>,
        peers: &Arc<[Arc<Address<Owner>>]>,
        sequences: usize,
        copies: Option<(SequenceId, Arc<Order<Executor>>)>,
    ) -> Result<Executor, EngineError> {
        let library: &Arc<CudaLibrary> = &core.library;
        let mut formats = Formats::new(
            Arc::new(TrellisLinear::from_settings(
                &core.settings.kernels.trellis,
                library.device().sm_count,
            )),
            Arc::new(Nvfp4Linear::new(library)),
        );
        // The grouped format asks the device for its FP8 route, which compiles a module: only an artifact that has
        // grouped tensors pays for it.
        if residency.device.iter().any(|placement| {
            matches!(
                placement.format,
                WeightFormat::Q3G64Fp16 | WeightFormat::Q4G64Fp16 | WeightFormat::Q5G64Fp16
            )
        }) {
            formats = formats.with_grouped(Arc::new(Q3Linear::new(library)));
        }
        // The staging ring and the lane its transfers run on, when the plan holds weights that are staged.
        let ring = staging_ring(
            residency.offload.staging,
            weights.staging_base(),
            weights.host_arena().is_some(),
        )
        .map_err(EngineError::Runtime)?;
        let plan = Plan::build_tiered(
            config,
            weights,
            residency,
            &formats,
            ring.as_ref(),
            ring.as_ref().map(|_| core.staging_lane_id()),
            weights.disk_tier(),
        )
        .map_err(|error| EngineError::Runtime(format!("the execution plan: {error}")))?;
        if plan.fan_out_width() > core.compute_lanes() {
            return Err(EngineError::Runtime(format!(
                "the plan's graphs fan out {} wide and the engine made {} lanes for them",
                plan.fan_out_width(),
                core.compute_lanes()
            )));
        }
        check_marker_rings(&plan, sequences, core.stage_ring()).map_err(EngineError::Runtime)?;
        let runtime = GraphRuntime {
            stages: Arc::clone(&core.stages) as Arc<dyn FrameSink>,
            completions: Arc::clone(core.completions()),
            lanes: Arc::clone(core.lanes()),
            retries: Arc::new(RetryCounts::default()),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        };
        let retiring = Arc::clone(address);
        let (passes, peers) = (Arc::clone(order), Arc::clone(peers));
        let executor = Executor::new(
            Arc::new(plan),
            &runtime,
            core.compute_lanes(),
            Arc::clone(&core.gpu),
            core.settings.kernels.graphs.captures_per_view as usize,
            move |mut quantum, outcome| {
                if quantum.segments().len() < 2 {
                    // Runs the owner's turn on this worker when no other turn runs.
                    retiring.post(Message::Retired(Box::new(Retired::whole(quantum, outcome))));
                    return;
                }
                // A pass of several sequences: the workspace's owner is told first (what waits behind the pass starts, and
                // the passes' rule hears of it), then each member's owner gets its own part.
                passes.pass_retired(quantum.id());
                let leases = quantum.take_leases();
                let guards = quantum.take_flights();
                let batch = Arc::new(quantum);
                for (segment, (lease, guard)) in leases.into_iter().zip(guards).enumerate() {
                    let sequence = batch.segments().get(segment).and_then(|part| part.sequence());
                    let Some(owner) = sequence.and_then(|sequence| peers.get(sequence.0)) else {
                        log::error!("a pass retired with a part of no sequence the engine holds");
                        continue;
                    };
                    owner.post(Message::Retired(Box::new(Retired::part(
                        &batch, segment, lease, guard, &outcome,
                    ))));
                }
            },
        )
        .map_err(|error| EngineError::Runtime(format!("the stage graphs: {error:?}")))?
        .with_preparer(Preparer::new(
            core.lane_table().clone(),
            Arc::clone(&core.pool),
            Arc::clone(core.owner_lane()),
        ));
        let Some((sequence, order)) = copies else {
            return Ok(executor);
        };
        // The sequence's own transfer lane (sequences share them when there are fewer lanes than sequences), so that
        // one sequence's restore does not wait behind another's capture on a lane they would not share.
        let lane = core.transfer_lane_id(sequence.0 % core.copy_lanes(residency.offload.staging.is_some()));
        // A copy job that ended tells the order, as a quantum's retirement does (what waits behind it starts), and then
        // the sequence's owner, which counts it until now.
        let owner = Arc::clone(address);
        let ended = Box::new(move |serial| {
            order.retired(sequence, serial);
            owner.post(Message::CopyEnded);
        });
        let copier = Copier::new(&runtime, core.lane_table().clone(), lane, sequence, ended)
            .map_err(|error| EngineError::Runtime(format!("the prefix cache's copy graph: {error:?}")))?;
        Ok(executor.with_copier(copier))
    }

    /// Starts `request` on this sequence, which `lease` holds until the request ends.
    fn generate(&self, request: GenerationRequest, cancel: CancelHandle, lease: Lease) -> Result<(), AdmissionError> {
        let core = Arc::clone(&self.shared.core);
        self.start(move |owner| match owner {
            Some(owner) => owner.begin(admit(&core), request, cancel, lease),
            None => Err(closed(Audience::Greedy(request.listener))),
        })
    }

    /// Starts `request` on this sequence, which hands every row's hidden state to `sink`.
    fn generate_hidden(
        &self,
        request: GenerationRequest,
        sink: Box<dyn HiddenSink>,
        cancel: CancelHandle,
        lease: Lease,
    ) -> Result<(), AdmissionError> {
        let core = Arc::clone(&self.shared.core);
        self.start(move |owner| match owner {
            Some(owner) => owner.begin_with(admit(&core), request, cancel, lease, Some(sink)),
            None => Err(closed(Audience::Greedy(request.listener))),
        })
    }

    /// Starts a teacher-forced run on this sequence.
    fn teacher_forced(
        &self,
        prefix: Vec<u32>,
        forced: Vec<u32>,
        rows: usize,
        sink: Box<dyn LogitsSink>,
        lease: Lease,
    ) -> Result<(), AdmissionError> {
        let core = Arc::clone(&self.shared.core);
        self.start(move |owner| match owner {
            Some(owner) => owner.begin_forced(admit(&core), prefix, forced, rows, sink, lease),
            None => Err(closed(Audience::Forced(sink))),
        })
    }

    /// Starts the request `begin` makes. Whether it is admitted was decided by the pool that gave the sequence, never by
    /// whether a turn holds the claim. With the claim free the answer is the return value, and
    /// the first turn runs on a worker, never on the caller's thread. With the claim held the request is posted to the
    /// owner and this returns `Ok` at once (no wait: a listener calls this from a worker); a request the owner then
    /// refuses has its listener or sink told, as a failed outcome, after the claim is released.
    fn start(
        &self,
        begin: impl FnOnce(Option<&mut Owner>) -> Result<(), Rejected> + Send + 'static,
    ) -> Result<(), AdmissionError> {
        // A request that begins once the engine stops is refused as a closed one: the stop's cancel is an operation on
        // the owner, so whatever begins after it sees the flag.
        let shared = Arc::clone(&self.shared);
        let begin = move |owner: Option<&mut Owner>| begin(unless_closing(&shared.closing, owner));
        let mut begin = Some(begin);
        let inline = self
            .mailbox
            .try_with(|owner| begin.take().map_or(Ok(()), |begin| begin(Some(owner))));
        match inline {
            Ok(Some(started)) => {
                let started = started.map_err(|rejected| rejected.error);
                if started.is_ok() && !self.mailbox.nudge() {
                    // No worker will take frames: the request would never run. Give it up; applied whoever holds the
                    // claim.
                    self.mailbox.tell(Owner::withdraw);
                    return Err(AdmissionError::Engine("the engine is shutting down".to_string()));
                }
                started
            }
            // The owner is closed: the caller learns from the error.
            Ok(None) => Err(closed_error()),
            Err(_) => {
                let Some(begin) = begin else {
                    return Err(closed_error());
                };
                self.mailbox.defer(move |owner| match owner {
                    Some(owner) => {
                        if let Err(rejected) = begin(Some(owner)) {
                            owner.reject(rejected);
                        }
                        Vec::new()
                    }
                    // Closed while the request waited: its audience is told, once, with the other outcomes.
                    None => begin(None).err().map(Rejected::into_finished).into_iter().collect(),
                });
                Ok(())
            }
        }
    }

    /// The test hook: the owner panics at the end of the turn in which a request has admitted `quanta` quanta.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn panic_after_admissions(&self, quanta: u64) {
        self.shared.panic_after_admissions.store(quanta, Ordering::Release);
    }

    /// The test hook: a speculative request ends with a decode quantum that feeds its last token.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn settle_after_speculation(&self, settle: bool) {
        self.shared.settle_after_speculation.store(settle, Ordering::Release);
    }

    /// The test hook: the engine records no graphs from now on. Only while no request runs.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn disable_graphs(&self) -> bool {
        self.mailbox.ask(true, Owner::disable_graphs).is_some()
    }

    /// Looks at the sequence while no request holds it: `None` only while one does (or the engine is closed), never
    /// because a turn holds the claim at the moment. Waits at most the mailbox's bound: not for a lattice worker.
    pub(crate) fn inspect<R: Send + 'static>(
        &self,
        read: impl FnOnce(&SequenceState) -> R + Send + 'static,
    ) -> Option<R> {
        self.mailbox.ask(true, move |owner| read(owner.sequence()))
    }

    /// Ends the sequence. Nothing may be running: with a request in flight the sequence stays allocated, and says so.
    /// `false` then.
    fn close(&self) -> bool {
        match self.mailbox.close() {
            Some(owner) => {
                owner.close();
                true
            }
            None => {
                log::error!("a generation is running at shutdown: the sequence's device memory stays allocated");
                false
            }
        }
    }

    /// Takes the sequence back from an owner that has run nothing.
    fn take_sequence(&self) -> Option<SequenceState> {
        self.mailbox.close().map(Owner::release)
    }
}

/// Opens the prefix cache the plan granted memory for, once for the engine; `None` (and why, once, in the log) when the
/// engine cannot use it: the chunk is not a multiple of the checkpoint granule or does not divide the interval between
/// checkpoints. Requests that speculate neither restore nor capture (the state of a draft is not stored); the others do,
/// on an engine whose sequences can speculate too. Nothing is
/// opened, and nothing is logged, when the plan granted no memory (`prefix-cache.size: 0`).
fn open_cache(
    core: &Arc<Core>,
    config: &Arc<ModelConfig>,
    residency: &ResidencyPlan,
    claim: Option<Claim>,
) -> Option<PrefixCacheHandle> {
    if residency.prefix_cache_bytes == 0 {
        return None;
    }
    let claim = claim?;
    let settings = core.settings.prefix_cache;
    let chunk = residency.rows.rows;
    if chunk == 0 || !chunk.is_multiple_of(u64::from(POSITION_GRANULE)) {
        log::warn!(
            "the prefix cache is off: the prefill chunk is {chunk} tokens, and it must be a multiple of {POSITION_GRANULE}, the granule of a checkpoint's position"
        );
        return None;
    }
    if !u64::from(settings.checkpoint_tokens).is_multiple_of(chunk) {
        log::warn!(
            "the prefix cache is off: the checkpoint interval is {} tokens, and it must be a multiple of the {chunk}-token prefill chunk",
            settings.checkpoint_tokens
        );
        return None;
    }
    let spec = CacheSpec {
        library: &core.library,
        config,
        format: residency.kv_format,
        settings,
        granted: residency.prefix_cache_bytes,
        chunk: u32::try_from(chunk).ok()?,
        device_tokens: u32::try_from(residency.kv_device_tokens).unwrap_or(u32::MAX),
    };
    open_prefix_cache_claimed(spec, Generation::spawner(core), claim)
}

/// The engine's sequences, each with an owner of its own, the pool that says which of them is idle, and the one order
/// they share on the workspace.
pub(crate) struct Generations {
    generations: Vec<Generation>,
    pool: SequencePool,
    order: Arc<Order<Executor>>,
    /// The prefix cache the sequences share; `None` when it is off (see `open_cache`).
    prefix: Option<PrefixCacheHandle>,
    /// A test hook: the next request admitted arms the owner's panic after this many admissions.
    #[cfg(feature = "test-oracles")]
    panic_next: AtomicU64,
}

impl fmt::Debug for Generations {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Generations")
            .field("sequences", &self.generations.len())
            .finish_non_exhaustive()
    }
}

impl Generations {
    /// Builds an owner and the graphs of the model's plan for each of `sequences`, and the one order they share. A
    /// refused plan gives every sequence back.
    #[allow(clippy::too_many_arguments, reason = "the startup's state, moved into the owners")]
    pub(crate) fn new(
        core: &Arc<Core>,
        config: &Arc<ModelConfig>,
        residency: &ResidencyPlan,
        weights: &Weights,
        workspace: &Arc<Workspace>,
        sequences: Vec<SequenceState>,
        draft_ids: Option<Arc<[u32]>>,
        prefix_claim: Option<Claim>,
    ) -> Result<Generations, Box<(EngineError, Vec<SequenceState>)>> {
        assert!(
            workspace.records() as u64 >= input_records_for(residency.draft_depth, sequences.len()),
            "the workspace holds input records for every sequence"
        );
        // The owners' addresses exist before the owners do: the workspace's owner tells each of its sequence's quanta
        // that did not start.
        let addresses: Vec<Arc<Address<Owner>>> = sequences.iter().map(|_| Address::new()).collect();
        let replies: Vec<Reply> = addresses
            .iter()
            .map(|address| {
                let told = Arc::clone(address);
                Box::new(move |serial, error| {
                    told.post(Message::StartFailed(Box::new(StartFailed { serial, error })));
                }) as Reply
            })
            .collect();
        let caps = match PassCaps::new(residency.pass_rows, residency.pass_parts) {
            Ok(caps) => caps,
            Err(why) => return Err(Box::new((EngineError::Runtime(why), sequences))),
        };
        let order = Arc::new(Order::new(
            replies,
            input_records(residency.draft_depth) as usize,
            caps,
            Generation::spawner(core),
        ));
        let count = sequences.len();
        let peers: Arc<[Arc<Address<Owner>>]> = addresses.iter().map(Arc::clone).collect();
        let prefix = open_cache(core, config, residency, prefix_claim);
        let mut generations: Vec<Generation> = Vec::with_capacity(count);
        let mut waiting = sequences.into_iter().zip(addresses).enumerate();
        while let Some((index, (sequence, address))) = waiting.next() {
            let built = Generation::new(
                core,
                config,
                residency,
                weights,
                workspace,
                sequence,
                SequenceId(index),
                address,
                &order,
                &peers,
                count,
                draft_ids.clone(),
                prefix.clone(),
            );
            match built {
                Ok(generation) => generations.push(generation),
                Err(refused) => {
                    let (error, sequence) = *refused;
                    let mut back: Vec<SequenceState> =
                        generations.iter().filter_map(Generation::take_sequence).collect();
                    back.push(sequence);
                    back.extend(waiting.map(|(_, (sequence, _))| sequence));
                    if let Some(cache) = &prefix {
                        cache.close(OPERATION_BOUND);
                    }
                    drop(order.close());
                    return Err(Box::new((error, back)));
                }
            }
        }
        Ok(Generations {
            pool: SequencePool::new(count),
            generations,
            order,
            prefix,
            #[cfg(feature = "test-oracles")]
            panic_next: AtomicU64::new(u64::MAX),
        })
    }

    /// The number of sequences.
    pub(crate) fn sequences(&self) -> usize {
        self.generations.len()
    }

    /// Takes an idle sequence, or refuses with `Busy` when every sequence runs a generation.
    fn take(&self) -> Result<(&Generation, Lease), AdmissionError> {
        let (id, lease) = self.pool.take().ok_or(AdmissionError::Busy)?;
        let generation = &self.generations[id.0];
        #[cfg(feature = "test-oracles")]
        {
            let quanta = self.panic_next.swap(u64::MAX, Ordering::AcqRel);
            if quanta != u64::MAX {
                generation.panic_after_admissions(quanta);
            }
        }
        Ok((generation, lease))
    }

    /// Starts `request` on an idle sequence.
    pub(crate) fn generate(&self, request: GenerationRequest, cancel: CancelHandle) -> Result<(), AdmissionError> {
        let (generation, lease) = self.take()?;
        generation.generate(request, cancel, lease)
    }

    /// Starts `request` on an idle sequence, which hands the hidden state of every row it runs to `sink`.
    pub(crate) fn generate_hidden(
        &self,
        request: GenerationRequest,
        sink: Box<dyn HiddenSink>,
        cancel: CancelHandle,
    ) -> Result<(), AdmissionError> {
        let (generation, lease) = self.take()?;
        generation.generate_hidden(request, sink, cancel, lease)
    }

    /// What the engine's shared passes did.
    pub(crate) fn pass_statistics(&self) -> PassStatistics {
        self.order.statistics()
    }

    /// The prefix cache's counters; `None` when the cache is off.
    pub(crate) fn prefix_cache_stats(&self) -> Option<PrefixCacheStats> {
        self.prefix.as_ref().map(PrefixCacheHandle::stats)
    }

    /// Starts a teacher-forced run on an idle sequence.
    pub(crate) fn teacher_forced(
        &self,
        prefix: Vec<u32>,
        forced: Vec<u32>,
        rows: usize,
        sink: Box<dyn LogitsSink>,
    ) -> Result<(), AdmissionError> {
        let (generation, lease) = self.take()?;
        generation.teacher_forced(prefix, forced, rows, sink, lease)
    }

    /// The test hook: the owner of the sequence the next request takes panics at the end of the turn in which that
    /// request has admitted `quanta` quanta.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn panic_after_admissions(&self, quanta: u64) {
        self.panic_next.store(quanta, Ordering::Release);
    }

    /// The test hook: a speculative request ends with a decode quantum that feeds its last token.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn settle_after_speculation(&self, settle: bool) {
        for generation in &self.generations {
            generation.settle_after_speculation(settle);
        }
    }

    /// The test hook: `hook` is told of every prefix-cache copy job a sequence's owner admits.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn copy_admitted(&self, hook: Arc<dyn Fn(bool) + Send + Sync>) {
        for generation in &self.generations {
            let hook = Arc::clone(&hook);
            // A hook is set once per engine: the first one stays.
            let _ = generation
                .shared
                .copy_admitted
                .set(Box::new(move |restore| hook(restore)));
        }
    }

    /// The test hook: `hook` is told when a sequence's owner has reserved a restore's rows, before the cache plans it.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn restore_reserved(&self, hook: Arc<dyn Fn() + Send + Sync>) {
        for generation in &self.generations {
            let hook = Arc::clone(&hook);
            let _ = generation.shared.restore_reserved.set(Box::new(move || hook()));
        }
    }

    /// The test hook: `hook` is told of every quantum a sequence's owner admits, with its start position.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn quantum_admitted(&self, hook: Arc<dyn Fn(u64) + Send + Sync>) {
        for generation in &self.generations {
            let hook = Arc::clone(&hook);
            let _ = generation
                .shared
                .quantum_admitted
                .set(Box::new(move |start| hook(start)));
        }
    }

    /// The test hook: the next graph body of the first sequence that runs for a new key ends as `fault`.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn fail_body(&self, fault: BodyFault, sighting: Sighting, token: Option<&Arc<()>>) {
        if let Some(generation) = self.generations.first() {
            generation.shared.executor.body_faults().arm_at(fault, sighting, token);
        }
    }

    /// The test hook: the strong handles of the test's value at each event of the faulted quantum's hold, by sequence.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn body_held_counts(&self) -> Vec<Vec<usize>> {
        self.generations
            .iter()
            .map(|generation| generation.shared.executor.body_faults().held_counts())
            .collect()
    }

    /// The test hook: the graph cache around the faulted quantum's run, by sequence.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn run_snapshots(&self) -> Vec<Option<RunSnapshot>> {
        self.generations
            .iter()
            .map(|generation| generation.shared.executor.body_faults().run_snapshot())
            .collect()
    }

    /// The test hook: faults of the hook itself, by sequence.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn hook_defects(&self) -> Vec<Vec<Defect>> {
        self.generations
            .iter()
            .map(|generation| generation.shared.executor.body_faults().defects())
            .collect()
    }

    /// The test hook: what each sequence's graph cache counted, and how many graphs it holds.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn graph_counts(&self) -> Vec<(GraphStats, usize)> {
        self.generations
            .iter()
            .map(|generation| {
                let executor = &generation.shared.executor;
                (executor.graph_stats(), executor.graphs().0)
            })
            .collect()
    }

    /// The test hook: what became of the hold of the quantum a fault was applied to, by sequence.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn body_events(&self) -> Vec<Vec<HoldEvent>> {
        self.generations
            .iter()
            .map(|generation| generation.shared.executor.body_faults().events())
            .collect()
    }

    /// The test hook: no sequence records graphs from now on. Only while no request runs.
    #[cfg(feature = "test-oracles")]
    pub(crate) fn disable_graphs(&self) -> bool {
        let disabled: Vec<bool> = self.generations.iter().map(Generation::disable_graphs).collect();
        disabled.into_iter().all(|disabled| disabled)
    }

    /// Looks at sequence `index` while no request holds it. `None` while one does, or when there is no such sequence.
    pub(crate) fn inspect<R: Send + 'static>(
        &self,
        index: usize,
        read: impl FnOnce(&SequenceState) -> R + Send + 'static,
    ) -> Option<R> {
        self.generations.get(index)?.inspect(read)
    }

    /// Cancels every request that runs and waits at most `bound` in all until each ended and its listener was told its
    /// last call (a request that ends in a stop is told `Cancelled`). Refuses new requests from now on. Runs while the
    /// lattice still works. `false` when the bound passed with a request still running, and the caller must not free
    /// what it uses; called from inside a turn it returns `false` at once. A request ends only when nothing of its
    /// sequence is in flight (`may_end`): a quantum or job whose guard never drops keeps it from ending, so this returns
    /// `false` at the bound, and the runtime's shutdown forgets the runtime and returns an error. For the engine's owner, outside the lattice.
    pub(crate) fn stop(&self, bound: Duration) -> bool {
        // Every owner is told to stop before any is waited for, so that the requests end side by side.
        let deadline = Instant::now() + bound;
        let idle: Vec<_> = self
            .generations
            .iter()
            .map(|generation| {
                generation.shared.closing.store(true, Ordering::SeqCst);
                generation.mailbox.stopping()
            })
            .collect();
        // A generation whose barrier could not be set (called from inside a turn) cannot be waited for: none is.
        if idle.iter().any(Option::is_none) {
            log::error!("a generation cannot be waited for from inside a turn at shutdown");
            return false;
        }
        let stopped = idle
            .into_iter()
            .flatten()
            .map(|idle| idle.wait(deadline.saturating_duration_since(Instant::now())).is_ok())
            .collect::<Vec<_>>()
            .into_iter()
            .all(|ended| ended);
        if !stopped {
            log::error!("a generation did not end within {bound:?} of the cancel at shutdown");
        }
        stopped
    }

    /// Ends the sequences, then the workspace's order, which they use. Nothing may be running: a sequence with a
    /// request in flight stays allocated, says so, and keeps the order's pins.
    pub(crate) fn close(&self) {
        let closed: Vec<bool> = self.generations.iter().map(Generation::close).collect();
        // The cache is shut once, after every owner (so no owner posts to it) and before the order its copy jobs
        // are admitted through.
        if let Some(cache) = &self.prefix {
            cache.close(OPERATION_BOUND);
        }
        if closed.into_iter().all(|closed| closed) {
            // What the workspace's owner still pins goes with it.
            drop(self.order.close());
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_request_begins_on_an_owner_once_the_engine_stops() {
        let closing = AtomicBool::new(false);
        let mut owner = 7;
        assert_eq!(unless_closing(&closing, Some(&mut owner)).copied(), Some(7));
        closing.store(true, Ordering::SeqCst);
        assert!(unless_closing(&closing, Some(&mut owner)).is_none());
        assert!(unless_closing::<i32>(&closing, None).is_none());
    }
}
