//! The engine runtime: the lattice, the lakes, the lanes and the model on the device.
//!
//! [`EngineRuntime::start`] builds the lattice and what it runs on, on the calling thread, and hands the rest of the
//! startup to frames: the artifact's index is read by `AsyncIo`, the plan is made, the arena and the sequence state
//! are allocated, the weights are uploaded along their chains, and the modules are warmed. The latch is set when the
//! model is ready, or with the error that stopped it. The calling thread waits for the latch and for nothing else.

use std::cell::Cell;
use std::fmt;
use std::mem::{self, ManuallyDrop};
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::config::{FragmentConfig, IdlePolicy, LatticeConfig};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_core::impls::{BaseCloneableObject, DefaultExecutor};
use euhedral_gpu::cuda::{CudaGpu, CudaQueue};
use euhedral_gpu::{Gpu, GpuError, Stream, StreamOptions, open_shared_lanes};
use euhedral_graph::{Completions, DeviceQueue, Lake, LaneId, LanePool, UnitGuard};
use euhedral_hardware_utils::util::CpuSet;
use euhedral_inference_cuda::launch::load::load;
use euhedral_inference_cuda::{CudaLibrary, PinnedPool};
use euhedral_inference_edrl::TensorTable;
use euhedral_inference_settings::{Auto, BatchingSettings, Settings};
use euhedral_io::{AsyncIo, OwnedFile, RawFile};

use crate::error::{EngineError, PlanError};
use crate::generation::{AdmissionError, CancelHandle, GenerationRequest, Generations, LogitsSink};
use crate::generation::{GREEDY_DEPTH, OPERATION_BOUND};
use crate::handoff::Handoff;
use crate::kernels::kernel_sources;
use crate::model::{ArtifactProfile, ModelConfig, Speculation, Weights};
use crate::pass_assembler::PassStatistics;
use crate::plan::Plan;
use crate::prefix::PrefixCacheStats;
use crate::residency::planner::{PlanInputs, ResidencyPlan, pass_rows_limit, sequences_for};
use crate::stages::LaneTable;
use crate::startup;
use crate::state::SequenceState;
use crate::state::released::ReleasedBuffers;
use crate::warmup::Warmed;
use crate::workspace::Workspace;

/// The markers a transfer lane's ring holds: a chain has two copies in flight, and the next waits for the completion of
/// one of them, so the ring has room beyond them.
const TRANSFER_RING: usize = 4;

/// The quanta whose markers a stage lane's ring must hold: those in flight, and the one the owner prepares behind them.
pub(crate) const STAGE_RING_QUANTA: usize = GREEDY_DEPTH + 1;

/// The events a stage lane's ring holds, sized before the model is read: the most markers of the lane that stay in
/// flight, until their last consumer has waited for them. A quantum records one marker per stage it runs, a marker is
/// waited for by the quantum itself and by the ones admitted behind it, so the ring holds the stages of the biggest view
/// of the model the engine runs ([`Plan::stage_bound`]) for `quanta_in_flight` quanta, rounded up to a power of two.
/// The startup refuses a plan that needs more ([`Core::stage_ring`]).
pub(crate) fn stage_ring(quanta_in_flight: usize) -> usize {
    Plan::stage_bound()
        .saturating_mul(quanta_in_flight)
        .max(1)
        .checked_next_power_of_two()
        .map_or(MAX_STAGE_RING, |ring| ring.min(MAX_STAGE_RING))
}

/// The most events a lane's ring holds: its size is a `u32`, a power of two.
const MAX_STAGE_RING: usize = 1 << 31;

/// The quanta in flight when `sequences` sequences each keep [`STAGE_RING_QUANTA`] of them.
fn quanta_for(sequences: usize) -> usize {
    STAGE_RING_QUANTA.saturating_mul(sequences.max(1))
}

/// How many sequences the settings size the lanes' rings for: `batching.max-sequences`, and one when it is 0, which
/// is decided only once the model is read. A count above the rows a pass holds is refused: each sequence keeps its
/// quanta's markers in the rings, which are created with the lanes, before the model is read, and a verification takes
/// a row at least.
fn ringed_sequences(batching: &BatchingSettings) -> Result<usize, EngineError> {
    let sequences = (batching.max_sequences as usize).max(1);
    let rows = pass_rows_limit(batching);
    if sequences > rows {
        return Err(EngineError::Runtime(format!(
            "batching.max-sequences is {sequences}, more than the {rows} rows a pass holds \
             (batching.max-tokens-per-pass, at most batching.prefill-chunk-tokens); set it lower"
        )));
    }
    Ok(sequences)
}

/// The sequences to hold: `batching.max-sequences` when it is at least 1, otherwise (0, automatic) the largest count
/// whose plan fits the device and the host ([`sequences_for`]), at most the rows a shared pass holds
/// (`batching.max-tokens-per-pass`, a prefill chunk by default) over a verification's rows, and 1 for a drafter that
/// is not the artifact's own.
pub fn resolve_sequences(
    config: &ModelConfig,
    table: &TensorTable,
    settings: &Settings,
    inputs: PlanInputs,
) -> Result<usize, PlanError> {
    let requested = settings.batching.max_sequences as usize;
    if requested >= 1 {
        return Ok(requested);
    }
    let profile = ArtifactProfile::of(table, config)?;
    if profile.speculation == Speculation::Dflash2 {
        return Ok(1);
    }
    let rows = pass_rows_limit(&settings.batching);
    let verified = 1 + if profile.speculation == Speculation::Mtp {
        profile.speculative_depth()?
    } else {
        0
    };
    let cap = (rows / verified).max(1);
    let chosen = sequences_for(config, table, settings, inputs, cap)?;
    log::info!(
        "sequences: {chosen} of {} tokens each fit the device (automatic)",
        settings.context.max_tokens
    );
    Ok(chosen)
}

/// What the frames of the engine share: the lattice and everything that feeds it, and the device.
pub(crate) struct Core {
    pub(crate) settings: Settings,
    pub(crate) library: Arc<CudaLibrary>,
    pub(crate) lattice: Arc<ControlPlaneLattice>,
    /// GPU submission frames.
    pub(crate) stages: Arc<Lake>,
    /// CPU tasks: reads' checks, parsing, planning.
    pub(crate) host: Arc<Lake>,
    /// Control: the steps of a startup and of a shutdown.
    pub(crate) control: Arc<Lake>,
    pub(crate) io: Arc<AsyncIo>,
    pub(crate) completions: Arc<Completions>,
    /// The device RustEE's lanes and kernels are on, kept for as long as they are.
    pub(crate) gpu: Arc<CudaGpu>,
    /// The lanes as query queues, which the completions and the graph runtime poll. Its owner and transfer lanes are
    /// reserved: no graph run is homed on them.
    pub(crate) lanes: Arc<LanePool>,
    /// The lanes the stage graphs run on, by lane id: the plan's compute lanes (`0..` [`Plan::lane_width`]), the
    /// owner's lane (reservations and input records), then the transfer
    /// lanes, one per upload chain, the first of which also zeroes the sequence state.
    pub(crate) lane_table: LaneTable,
    /// The events in the ring of each stage lane ([`stage_ring`]).
    pub(crate) stage_ring: usize,
    /// The plan's compute lanes.
    pub(crate) compute: usize,
    pub(crate) pool: Arc<PinnedPool>,
    /// Buffers waiting to be freed in stream order: the pages and page tables of the caches, once their last quantum
    /// is gone. The generation's owner frees them on its lane.
    pub(crate) released: Arc<ReleasedBuffers>,
    /// Pinned host bytes the load may use.
    pub(crate) host_pinned_bytes: u64,
    /// The pinned pool's size (`memory.pinned-buffer-pool`, resolved): what it keeps, and the ring's budget.
    pub(crate) pool_bytes: u64,
    pub(crate) started: Instant,
    /// What stops the lattice and its lakes, shared with the owner.
    pub(crate) teardown: Arc<Teardown>,
}

impl fmt::Debug for Core {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Core").finish_non_exhaustive()
    }
}

/// The raw handle `AsyncIo` reads from.
#[cfg(unix)]
pub(crate) fn raw_file(file: &OwnedFile) -> RawFile {
    use std::os::fd::AsRawFd;

    file.as_raw_fd()
}

/// The raw handle `AsyncIo` reads from.
#[cfg(windows)]
pub(crate) fn raw_file(file: &OwnedFile) -> RawFile {
    use std::os::windows::io::AsRawHandle;

    file.as_raw_handle()
}

/// Units admitted on the three lakes while a startup runs: the lakes take frames only while a unit is admitted.
#[derive(Debug)]
pub(crate) struct Units {
    _stages: UnitGuard,
    _host: UnitGuard,
    _control: UnitGuard,
}

impl Core {
    /// `sequences`, or fewer when the lanes' rings, sized before the model was read, cannot hold the quanta that many
    /// sequences keep in flight. Only an automatic count can exceed them: the rings are sized for the stated count, and
    /// for one when it is automatic, because a ring's markers are device events created with the lane.
    pub(crate) fn hold_sequences(&self, sequences: usize) -> usize {
        let per_sequence = Plan::stage_bound().saturating_mul(STAGE_RING_QUANTA).max(1);
        let held = sequences.min((self.stage_ring / per_sequence).max(1));
        if held < sequences {
            log::warn!(
                "{sequences} sequences fit the device but the lanes' rings hold the quanta of {held}; set \
                 batching.max-sequences to hold more"
            );
        }
        held
    }

    /// The owner's lane: the sequence's reservations and its quanta's input records.
    pub(crate) fn owner_lane(&self) -> &Arc<CudaQueue> {
        self.lane(LaneId(self.compute as u16))
    }

    /// The transfer lane of upload chain `chain`.
    pub(crate) fn transfer_lane(&self, chain: usize) -> &Arc<CudaQueue> {
        self.lane(LaneId((self.compute + 1 + chain) as u16))
    }

    /// The id of the transfer lane of chain `chain`.
    pub(crate) fn transfer_lane_id(&self, chain: usize) -> LaneId {
        LaneId((self.compute + 1 + chain) as u16)
    }

    /// The number of transfer lanes.
    pub(crate) fn transfer_lanes(&self) -> usize {
        self.lane_table.len() - self.compute - 1
    }

    fn lane(&self, id: LaneId) -> &Arc<CudaQueue> {
        match self.lane_table.get(id) {
            Some(lane) => lane,
            None => unreachable!("the runtime opened lane {} and the table holds it", id.0),
        }
    }

    /// The lanes the plan's graphs run on.
    pub(crate) fn compute_lanes(&self) -> usize {
        self.compute
    }

    /// Admits a unit on every lake.
    pub(crate) fn admit(&self) -> Result<Units, EngineError> {
        let closed = |error| EngineError::Runtime(format!("a lake would not take work: {error}"));
        Ok(Units {
            _stages: self.stages.admit().map_err(closed)?,
            _host: self.host.admit().map_err(closed)?,
            _control: self.control.admit().map_err(closed)?,
        })
    }
}

/// The numbers of one startup.
#[derive(Clone, Debug, PartialEq)]
pub struct StartupReport {
    /// From the call of `start` to ready.
    pub ready: Duration,
    /// The weight upload alone.
    pub load: Duration,
    /// Weight bytes read from the artifact and copied to the device.
    pub uploaded_bytes: u64,
    /// Reads and copies the upload was cut into.
    pub pieces: usize,
    /// Upload chains, one transfer lane each.
    pub chains: usize,
    /// The longest piece.
    pub piece_bytes: u64,
    /// Objects on the device (the halves of a split projection counted for two).
    pub objects: usize,
    /// Bytes this runtime allocated on the device: the weight arena, the workspace and the sequence state.
    pub allocated_bytes: u64,
    /// Bytes in use on the device when the model became ready (the whole device's, every process).
    pub device_bytes_in_use: u64,
    /// Pinned bytes the staging pool holds at ready: the upload's ring and the cache's table staging.
    pub pinned_staging_bytes: u64,
    /// Pinned bytes the pool had to allocate while the upload ran; 0, because the ring is acquired before it.
    pub pinned_growth_during_load: u64,
    /// The kernel modules launched once before the model became ready, and those that were not.
    pub warmed: Warmed,
}

/// What stops an engine: its lakes and its lattice, and the count of device bytes that nothing has freed yet. Closing
/// waits for the lattice's workers, so only the engine's owner does it, on `main`.
pub struct Teardown {
    lakes: Vec<Arc<Lake>>,
    lattice: Arc<ControlPlaneLattice>,
    /// Device bytes the runtime allocated and `shutdown` has not freed.
    pub(crate) device_bytes: AtomicU64,
    /// How long `shutdown` waits for a running request to end after its cancel: the lattice's close timeout.
    pub(crate) cancel_bound: Duration,
}

impl Teardown {
    /// The teardown of `lattice` and the `lakes` that feed it.
    pub fn new(lattice: Arc<ControlPlaneLattice>, lakes: Vec<Arc<Lake>>) -> Arc<Teardown> {
        Teardown::with_cancel_bound(lattice, lakes, OPERATION_BOUND)
    }

    /// [`new`](Teardown::new) with the time `shutdown` waits for a running request to end.
    pub(crate) fn with_cancel_bound(
        lattice: Arc<ControlPlaneLattice>,
        lakes: Vec<Arc<Lake>>,
        cancel_bound: Duration,
    ) -> Arc<Teardown> {
        Arc::new(Teardown {
            lakes,
            lattice,
            device_bytes: AtomicU64::new(0),
            cancel_bound,
        })
    }

    /// Closes the lakes, then the lattice. Idempotent. Waits for the workers: never from a worker.
    pub fn close(&self) {
        for lake in &self.lakes {
            lake.complete_gracefully();
        }
        self.lattice.close();
    }
}

impl fmt::Debug for Teardown {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Teardown").finish_non_exhaustive()
    }
}

/// The owner of a started engine's lattice and device memory, held by `main` (or the test): the one place the engine
/// closes the lattice and frees device memory with calls that wait.
///
/// The contract:
/// - [`EngineRuntime::start`] returns it; keep it on the thread that started the engine until the engine is done.
///   That thread must not be a worker of the engine's lattice.
/// - To stop the engine, call [`RuntimeOwner::shutdown`] with the runtime from the latch. It closes the lakes and the
///   lattice, then frees the weights, the workspace and the sequence state.
/// - Dropping the owner always closes the lakes and the lattice, whatever happened to the startup and whatever
///   handles to the runtime are still alive, so that no worker ever runs the last drop of the lattice. Without
///   `shutdown` the device memory is not freed (freeing waits; no `Drop` of the runtime does it): it stays allocated
///   until the process exits, and the drop logs the number of bytes.
#[must_use = "the owner closes the lattice and frees the device memory: hold it until the engine is done"]
pub struct RuntimeOwner {
    teardown: Option<Arc<Teardown>>,
}

impl RuntimeOwner {
    /// The owner of `teardown`. [`EngineRuntime::start`] makes one; this is for tests of the contract.
    pub fn over(teardown: Arc<Teardown>) -> RuntimeOwner {
        RuntimeOwner {
            teardown: Some(teardown),
        }
    }

    /// Stops the engine: closes the lakes and the lattice, waiting for the workers, then frees the device memory of
    /// `runtime`. Call it from the thread that started the engine, once no frame is running. It fails, leaving the
    /// device memory allocated until the process exits (and logging how much), if another handle to `runtime` is
    /// alive.
    pub fn shutdown(mut self, runtime: Arc<EngineRuntime>) -> Result<(), EngineError> {
        let teardown = self.teardown.take();
        let bound = teardown
            .as_ref()
            .map_or(OPERATION_BOUND, |teardown| teardown.cancel_bound);
        // The running request ends on the lattice's workers, so before the lattice closes.
        let stopped = runtime.generations.stop(bound);
        if let Some(teardown) = &teardown {
            teardown.close();
        }
        if !stopped {
            // A quantum may still be in flight: nothing it uses is freed. The runtime is forgotten, not dropped, because
            // dropping it frees the sequence's buffers, which waits for the device.
            if let Some(teardown) = &teardown {
                teardown.log_leak();
            }
            mem::forget(runtime);
            return Err(EngineError::Runtime(
                "a generation did not end after its cancel: the device memory stays allocated until the process exits"
                    .to_string(),
            ));
        }
        match Arc::try_unwrap(runtime) {
            Ok(runtime) => {
                runtime.release();
                if let Some(teardown) = &teardown {
                    teardown.device_bytes.store(0, Ordering::Release);
                }
                Ok(())
            }
            Err(_) => {
                if let Some(teardown) = &teardown {
                    teardown.log_leak();
                }
                Err(EngineError::Runtime(
                    "another handle to the runtime is alive: its device memory stays allocated until the process exits"
                        .to_string(),
                ))
            }
        }
    }
}

/// Stops an engine within `window`: `close` runs (it waits for the workers, up to the lattice's own timeout), and when
/// the window is spent by then, `free` does not run and `left` says what stays. A `free` that has begun is not
/// Stops an engine within `window`: `close` runs (it waits for the workers, up to the lattice's own close timeout, which
/// the start fixed: a `window` shorter than that is not shortened by it, and the free is skipped when the close spent
/// the window, whatever the two values are). When the window is spent, `keep` (the runtime and the teardown share) is
/// forgotten and not dropped, because dropping a runtime frees its sequence's buffers, which waits for the device;
/// the process is ending, and `left` says what stays. A `free` that has begun is not interrupted.
fn stop_within<T>(
    window: Duration,
    keep: T,
    close: impl FnOnce(&T),
    free: impl FnOnce(T) -> Result<(), EngineError>,
    left: impl FnOnce(&T) -> String,
) -> Result<(), EngineError> {
    let begun = Instant::now();
    close(&keep);
    if begun.elapsed() >= window {
        let left = left(&keep);
        mem::forget(keep);
        return Err(EngineError::Unfinished { window, left });
    }
    free(keep)
}

/// Frees a runtime that arrived after its start was abandoned: the arrival carries the release. It runs in the frame
/// from the completions source that delivers the runtime, not in the stages lake, and the free waits for the device
/// there. No hop to the stages lake is possible: the abandoning side's owner has closed the lakes. The lattice is
/// closed by that owner too, so this only frees the device memory.
pub(crate) fn free_arrival(arrival: Result<Arc<EngineRuntime>, EngineError>) {
    let Ok(runtime) = arrival else { return };
    let core = Arc::clone(&runtime.core);
    match Arc::try_unwrap(runtime) {
        Ok(runtime) => {
            runtime.release();
            core.teardown.device_bytes.store(0, Ordering::Release);
        }
        Err(_) => core.teardown.log_leak(),
    }
}

impl Teardown {
    fn log_leak(&self) {
        let bytes = self.device_bytes.load(Ordering::Acquire);
        if bytes > 0 {
            log::error!(
                "{bytes} bytes of device memory stay allocated until the process exits: the engine was not shut down"
            );
        }
    }
}

impl RuntimeOwner {
    /// [`shutdown`](RuntimeOwner::shutdown) bounded by `window`: it returns once the window is spent, saying what it
    /// did not do (`EngineError::Unfinished`), and the device memory it did not free stays allocated until the
    /// process exits (the runtime is forgotten, not dropped, so nothing waits for the device). The lattice's close is
    /// bounded by the timeout given to [`EngineRuntime::start_with`], else RustEE's default; when it spends the
    /// window, the device memory is not freed. A free that has begun is not interrupted.
    pub fn shutdown_within(mut self, runtime: Arc<EngineRuntime>, window: Duration) -> Result<(), EngineError> {
        let teardown = self.teardown.take();
        let stopped = Cell::new(true);
        stop_within(
            window,
            (runtime, teardown),
            |(runtime, teardown)| {
                // The running request ends on the lattice's workers, so before the lattice closes.
                let bound = teardown
                    .as_ref()
                    .map_or(window, |teardown| window.min(teardown.cancel_bound));
                stopped.set(runtime.generations.stop(bound));
                if let Some(teardown) = teardown {
                    teardown.close();
                }
            },
            |(runtime, teardown)| {
                match stopped.get() {
                    false => {
                        // A quantum may still be in flight: nothing it uses is freed, and dropping the runtime would.
                        if let Some(teardown) = &teardown {
                            teardown.log_leak();
                        }
                        mem::forget(runtime);
                        Err(EngineError::Runtime(
                            "a generation did not end after its cancel: the device memory stays allocated until the \
                         process exits"
                                .to_string(),
                        ))
                    }
                    true => Ok((runtime, teardown)),
                }
                .and_then(|(runtime, teardown)| match Arc::try_unwrap(runtime) {
                    Ok(runtime) => {
                        runtime.release();
                        if let Some(teardown) = &teardown {
                            teardown.device_bytes.store(0, Ordering::Release);
                        }
                        Ok(())
                    }
                    Err(_) => {
                        if let Some(teardown) = &teardown {
                            teardown.log_leak();
                        }
                        Err(EngineError::Runtime(
                        "another handle to the runtime is alive: its device memory stays allocated until the process \
                         exits"
                            .to_string(),
                    ))
                    }
                })
            },
            |(_, teardown)| {
                let bytes = teardown
                    .as_ref()
                    .map_or(0, |teardown| teardown.device_bytes.load(Ordering::Acquire));
                let what = if stopped.get() {
                    "the lattice was still closing"
                } else {
                    "a generation did not end after its cancel"
                };
                format!("{what}; {bytes} bytes of device memory stay allocated until the process exits")
            },
        )
    }
}

impl Drop for RuntimeOwner {
    fn drop(&mut self) {
        if let Some(teardown) = self.teardown.take() {
            teardown.close();
            teardown.log_leak();
        }
    }
}

impl fmt::Debug for RuntimeOwner {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("RuntimeOwner").finish_non_exhaustive()
    }
}

/// What the engine keeps on the device. Freed only by [`RuntimeOwner::shutdown`], because freeing waits. The sequences
/// are the generations'.
struct DeviceState {
    weights: Weights,
    workspace: Arc<Workspace>,
}

/// What [`EngineRuntime::start`] hands over: the running engine, or why it did not start.
pub type ReadyHandoff = Arc<Handoff<Result<Arc<EngineRuntime>, EngineError>>>;

/// A started engine: the model on the device and the machinery that runs it.
pub struct EngineRuntime {
    core: Arc<Core>,
    plan: ResidencyPlan,
    config: Arc<ModelConfig>,
    // Never dropped by `Drop`: freeing waits for the device, so only `release` (the owner's shutdown) frees it.
    device: ManuallyDrop<DeviceState>,
    // The sequences and the graphs that run them, each with an owner. Closed by `release`.
    generations: Generations,
    report: StartupReport,
}

impl fmt::Debug for EngineRuntime {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("EngineRuntime")
            .field("report", &self.report)
            .finish_non_exhaustive()
    }
}

impl EngineRuntime {
    /// Starts the engine of `settings` on the CUDA library at `library`, and returns at once: the rest of the startup
    /// runs as frames on the lattice built here. The returned hand-off gets the runtime when the model is on the
    /// device and the modules are warm, or the error that stopped the startup, before anything was allocated on the
    /// device when the artifact or the plan is what failed.
    ///
    /// Call it from the main thread, then [`wait`](Handoff::wait) on the hand-off. Keep the returned [`RuntimeOwner`]
    /// until the engine is done (see its contract): it closes the lattice and frees the device memory, on the calling
    /// thread.
    pub fn start(settings: &Settings, library: &Path) -> (RuntimeOwner, ReadyHandoff) {
        Self::start_with(settings, library, None)
    }

    /// [`start`](EngineRuntime::start) with the longest the lattice's close waits for the workers: `None` keeps
    /// RustEE's default. The server sets it from `server.shutdown-timeout`; nothing else does.
    pub fn start_with(
        settings: &Settings,
        library: &Path,
        close_timeout: Option<Duration>,
    ) -> (RuntimeOwner, ReadyHandoff) {
        let ready = Handoff::new();
        match build(settings, library, close_timeout) {
            Ok(core) => {
                let owner = RuntimeOwner::over(Arc::clone(&core.teardown));
                startup::run(core, Arc::clone(&ready));
                (owner, ready)
            }
            Err(error) => {
                ready.send(Err(error));
                (RuntimeOwner { teardown: None }, ready)
            }
        }
    }

    #[allow(clippy::too_many_arguments, reason = "the startup's state, moved into the runtime")]
    pub(crate) fn assemble(
        core: Arc<Core>,
        plan: ResidencyPlan,
        config: Arc<ModelConfig>,
        weights: Weights,
        workspace: Arc<Workspace>,
        generations: Generations,
        report: StartupReport,
    ) -> EngineRuntime {
        EngineRuntime {
            core,
            plan,
            config,
            device: ManuallyDrop::new(DeviceState { weights, workspace }),
            generations,
            report,
        }
    }

    /// The residency plan the model was loaded by.
    pub fn plan(&self) -> &ResidencyPlan {
        &self.plan
    }

    /// The sequences the engine holds, each of which runs one request at a time: the requests that run at once.
    pub fn sequences(&self) -> usize {
        self.generations.sequences()
    }

    /// What the engine's shared passes did since it started: the passes, their rows and parts, and the parts that shared a
    /// pass with another sequence's.
    pub fn pass_statistics(&self) -> PassStatistics {
        self.generations.pass_statistics()
    }

    /// The prefix cache's counters; `None` when the cache is off (`prefix-cache.size: 0`).
    pub fn prefix_cache_stats(&self) -> Option<PrefixCacheStats> {
        self.generations.prefix_cache_stats()
    }

    /// The model's dimensions.
    pub fn config(&self) -> &ModelConfig {
        &self.config
    }

    /// The model's tensors on the device.
    pub fn weights(&self) -> &Weights {
        &self.device.weights
    }

    /// The workspace, sized for a prefill chunk.
    pub fn workspace(&self) -> &Workspace {
        &self.device.workspace
    }

    /// Reads the device state of the engine's first sequence, when no generation is running on it: `None` while one
    /// is. Not for a lattice worker: when a turn of the owner holds the sequence at the moment, the read waits for it
    /// (a bounded wait). Requests take the lowest idle sequence, so one that runs alone runs on this one.
    pub fn inspect_sequence<R: Send + 'static>(
        &self,
        read: impl FnOnce(&SequenceState) -> R + Send + 'static,
    ) -> Option<R> {
        self.inspect_sequence_at(0, read)
    }

    /// As [`inspect_sequence`](EngineRuntime::inspect_sequence), for sequence `index`; `None` too when there is none.
    pub fn inspect_sequence_at<R: Send + 'static>(
        &self,
        index: usize,
        read: impl FnOnce(&SequenceState) -> R + Send + 'static,
    ) -> Option<R> {
        self.generations.inspect(index, read)
    }

    /// Starts generating from `request`'s prompt on an idle sequence of the engine, and returns at once: the request
    /// runs as frames on the engine's lattice, and its listener is told of it there. A sequence takes one request at a
    /// time and starts each from nothing, whatever the one before left in it; the sequences run their requests
    /// interleaved, each producing the tokens it would alone. [`AdmissionError::Busy`] when every sequence is running
    /// one.
    pub fn generate(&self, request: GenerationRequest) -> Result<(), AdmissionError> {
        self.generations.generate(request, CancelHandle::default())
    }

    /// As [`EngineRuntime::generate`], with `cancel` made by the caller, who keeps a clone: `cancel` ends the generation
    /// from any thread, from the listener too, without waiting, at the next quantum boundary. The generation then ends
    /// through its listener, once, as [`FinishReason::Cancelled`](crate::generation::FinishReason), with the tokens
    /// reported until then, and the sequence takes the next request. A handle already cancelled ends the generation
    /// before its first quantum.
    pub fn generate_cancellable(&self, request: GenerationRequest, cancel: CancelHandle) -> Result<(), AdmissionError> {
        self.generations.generate(request, cancel)
    }

    /// A test hook: the sequence's owner panics at the end of the turn in which the next request has admitted `quanta`
    /// quanta. The request ends with a failure and the engine serves the next.
    #[cfg(feature = "test-oracles")]
    pub fn panic_after_admissions_for_tests(&self, quanta: u64) {
        self.generations.panic_after_admissions(quanta);
    }

    /// A test hook: with `settle`, a speculative request that ends at its token budget ends with one decode quantum that
    /// feeds its last token, so that the sequence holds what greedy decode of one more token leaves (the replay of the
    /// last verification has run).
    #[cfg(feature = "test-oracles")]
    pub fn settle_after_speculation_for_tests(&self, settle: bool) {
        self.generations.settle_after_speculation(settle);
    }

    /// A test hook: `hook` is told, from the owner of the sequence, of each prefix-cache copy job it admits, `true` for
    /// a restore and `false` for a capture, while the job is in flight. It must not block. Set once per engine.
    #[cfg(feature = "test-oracles")]
    pub fn on_copy_admitted_for_tests(&self, hook: impl Fn(bool) + Send + Sync + 'static) {
        self.generations.copy_admitted(Arc::new(hook));
    }

    /// A test hook: `hook` runs on the owner once a restore's rows are reserved, before the cache plans the restore.
    /// It must not block. Set once per engine.
    #[cfg(feature = "test-oracles")]
    pub fn on_restore_reserved_for_tests(&self, hook: impl Fn() + Send + Sync + 'static) {
        self.generations.restore_reserved(Arc::new(hook));
    }

    /// A test hook: `hook` runs on the owner for each quantum it admits, with the quantum's start position. It must not
    /// block. Set once per engine.
    #[cfg(feature = "test-oracles")]
    pub fn on_quantum_admitted_for_tests(&self, hook: impl Fn(u64) + Send + Sync + 'static) {
        self.generations.quantum_admitted(Arc::new(hook));
    }

    /// A test hook: the engine records no graphs from now on, so every quantum runs stage by stage. `false` while a
    /// request runs.
    #[cfg(feature = "test-oracles")]
    pub fn disable_graphs_for_tests(&self) -> bool {
        self.generations.disable_graphs()
    }

    /// Starts a teacher-forced run on the engine's sequence, as [`EngineRuntime::generate`] does: prefills `prefix`,
    /// then decodes the `forced` tokens one row at a time, and hands the logits row of each step to `sink`. Step `k`
    /// of the sink holds the model's prediction of `forced[k]`.
    pub fn teacher_forced(
        &self,
        prefix: Vec<u32>,
        forced: Vec<u32>,
        sink: Box<dyn LogitsSink>,
    ) -> Result<(), AdmissionError> {
        self.generations.teacher_forced(prefix, forced, sink)
    }

    /// What the startup measured.
    pub fn report(&self) -> &StartupReport {
        &self.report
    }

    /// The CUDA library.
    pub fn library(&self) -> &Arc<CudaLibrary> {
        &self.core.library
    }

    /// The pool of pinned buffers, for staging that outlives a call.
    pub fn pinned_pool(&self) -> &Arc<PinnedPool> {
        &self.core.pool
    }

    /// The buffers the sequence's caches gave up when their last holder let go. The engine's own generation frees
    /// them in stream order on each turn of a sequence's owner.
    pub fn released_buffers(&self) -> &Arc<ReleasedBuffers> {
        &self.core.released
    }

    /// The lake of GPU submission frames.
    pub fn stages(&self) -> &Arc<Lake> {
        &self.core.stages
    }

    /// The lake of CPU tasks.
    pub fn host(&self) -> &Arc<Lake> {
        &self.core.host
    }

    /// The lake of control frames.
    pub fn control(&self) -> &Arc<Lake> {
        &self.core.control
    }

    /// The completions that turn lane progress into frames.
    pub fn completions(&self) -> &Arc<Completions> {
        &self.core.completions
    }

    /// The asynchronous file I/O.
    pub fn io(&self) -> &Arc<AsyncIo> {
        &self.core.io
    }

    /// The lanes `completions` watches.
    pub fn lanes(&self) -> &Arc<LanePool> {
        &self.core.lanes
    }

    /// The lattice every frame runs on.
    pub fn lattice(&self) -> &Arc<ControlPlaneLattice> {
        &self.core.lattice
    }

    /// The settings the runtime started with.
    pub fn settings(&self) -> &Settings {
        &self.core.settings
    }
}

impl EngineRuntime {
    /// Frees the device memory. Waits for the device.
    fn release(self) {
        let Self {
            device,
            generations,
            core,
            ..
        } = self;
        let DeviceState { weights, workspace } = ManuallyDrop::into_inner(device);
        // The sequences go first, by `close`, which refuses while a quantum is in flight. The lattice is closed by
        // now, so a quantum still counted will never be dropped: the caches are left allocated (the process is
        // ending) rather than freed under it.
        generations.close();
        // What the caches gave up and the owner had not yet freed. A queued buffer has no holder left, and the plain
        // free waits for the device, so nothing is freed under queued work.
        core.released.drain(drop);
        drop(workspace);
        drop(weights);
    }
}

/// The fragment configuration of the engine's lattice: RustEE's default, except that the idle policy has fixed timing,
/// and with the most frames a worker takes at once from `host.max-batch-size` when that is stated. The policy parks a
/// worker that sits out for [`IdlePolicy::DEFAULT_IDLE_PARK`] (15 us) and keeps the contention history for
/// [`IdlePolicy::DEFAULT_CONTENTION_HALF_LIFE`] (1 ms) whatever the worker's history; the unproductive park and the
/// probe interval stay RustEE's. A decode quantum is a graph of some 290 stages, each a frame handed on once the one
/// before it completes, and nothing wakes a worker for a frame offered to its cache, so each hand-off waits for a worker
/// to come round. The adaptive timing derives a park of up to 0.8 ms from each worker's history once the lattice has a
/// body-cost estimate, and every hand-off then waits out such a park. The fixed park bounds each hand-off by the
/// 15 us park.
pub fn fragment_config(settings: &Settings) -> FragmentConfig {
    let policy = match IdlePolicy::fixed(IdlePolicy::DEFAULT_IDLE_PARK, IdlePolicy::DEFAULT_CONTENTION_HALF_LIFE) {
        Ok(policy) => policy,
        Err(_) => unreachable!("the default contention half-life is not zero"),
    };
    let fragment = FragmentConfig::default().with_idle_policy(policy);
    let stated = match settings.host.max_batch_size {
        Auto::Value(size) => NonZeroU64::new(u64::from(size)),
        Auto::Auto => None,
    };
    match stated {
        Some(size) => fragment.with_max_batch_size(size),
        None => fragment,
    }
}

/// The engine's lattice: [`fragment_config`], frames run by RustEE's default executor, and the workers on the CPUs of
/// `host.cpus` (`auto` is all of them). Nothing is pinned or restricted beyond that list.
pub fn lattice_config(settings: &Settings) -> LatticeConfig {
    let cloneable = BaseCloneableObject::new(fragment_config(settings), DefaultExecutor::new());
    let mut config = LatticeConfig::new(Arc::new(cloneable));
    if let Auto::Value(cpus) = &settings.host.cpus {
        config.allowed_cpus = cpus.ids().map(|cpu| cpu as usize).collect::<CpuSet>();
    }
    config
}

/// The most lanes one RustEE pool holds.
const MAX_LANES: usize = 64;

/// The device the native library bound, opened through RustEE. The library reports the device's properties and not its
/// ordinal, so the device is the first whose compute capability is the library's.
pub fn open_device(library: &CudaLibrary) -> Result<Arc<CudaGpu>, EngineError> {
    let (major, minor) = library.device().compute_capability;
    let wanted = (i32::try_from(major).unwrap_or(0), i32::try_from(minor).unwrap_or(0));
    for ordinal in 0..MAX_LANES as i32 {
        let Ok(candidate) = CudaGpu::open(ordinal) else { break };
        if candidate.arch() == wanted {
            return Ok(candidate);
        }
    }
    Err(EngineError::Runtime(
        "RustEE found no device of the library's compute capability".into(),
    ))
}

/// The lanes of an engine, typed for the engine's own submissions and as queues for the graph runtime and the
/// completions.
#[derive(Debug)]
pub struct LaneSet<L: DeviceQueue> {
    /// The lanes as the engine submits on them.
    pub typed: Arc<LanePool<L>>,
    /// The lanes as queues.
    pub queues: Arc<LanePool>,
}

/// Opens the lanes of an engine on shared RustEE lanes: `compute` lanes for the stage graphs (ids `0..`), the
/// owner's lane, then `transfers` transfer lanes. The lanes after the compute lanes are
/// reserved in both pools, so no run of a graph is ever homed on them.
pub fn open_lane_set<G: Gpu>(
    gpu: &Arc<G>,
    compute: usize,
    ring: usize,
    transfers: usize,
) -> Result<LaneSet<<G::Stream as Stream>::Lane>, GpuError> {
    let stage = StreamOptions {
        markers: ring as u32,
        spans: 0,
        ..StreamOptions::default()
    };
    let transfer = StreamOptions {
        markers: TRANSFER_RING as u32,
        spans: 0,
        ..StreamOptions::default()
    };
    let mut specs: Vec<(&str, StreamOptions)> = Vec::with_capacity(compute + 1 + transfers);
    specs.extend((0..compute + 1).map(|_| ("stage", stage)));
    specs.extend((0..transfers).map(|_| ("transfer", transfer)));
    let shared = open_shared_lanes(gpu, &specs)?;
    let reserved: Vec<LaneId> = (compute..compute + 1 + transfers)
        .map(|lane| LaneId(lane as u16))
        .collect();
    let solely = || GpuError::Invalid("the lane pools are shared before they are reserved".into());
    let typed = Arc::try_unwrap(shared.pool)
        .map_err(|_| solely())?
        .with_reserved_lanes(&reserved);
    let queues = Arc::try_unwrap(shared.queues)
        .map_err(|_| solely())?
        .with_reserved_lanes(&reserved);
    Ok(LaneSet {
        typed: Arc::new(typed),
        queues: Arc::new(queues),
    })
}

/// Builds the lattice, the lakes, the lanes and the pools of a startup, on the calling thread.
fn build(settings: &Settings, library_path: &Path, close_timeout: Option<Duration>) -> Result<Arc<Core>, EngineError> {
    let started = Instant::now();
    // Before anything is opened: the rings the lanes are created with depend on it.
    let sequences = ringed_sequences(&settings.batching)?;
    let library = CudaLibrary::load(library_path)?;
    let runtime = |what: &str, error: &dyn fmt::Display| EngineError::Runtime(format!("{what}: {error}"));

    let mut config = lattice_config(settings);
    if let Some(timeout) = close_timeout {
        config.shutdown_timeout = timeout;
    }
    let cancel_bound = config.shutdown_timeout;
    let lattice = ControlPlaneLattice::new(config)
        .map_err(|error| runtime("the lattice would not start", &format!("{error:?}")))?;

    let sinks = NonZeroUsize::new(settings.host.task_parallelism as usize).unwrap_or(NonZeroUsize::MIN);
    let idle_release = settings.host.idle_release;
    let lake = || Lake::new(Arc::clone(&lattice), sinks, idle_release);
    let (stages, host, control) = (lake(), lake(), lake());

    let io = AsyncIo::open().map_err(|error| EngineError::io("the asynchronous file I/O would not open", error))?;

    // The device RustEE's lanes and kernels are on: the one the native library bound.
    let gpu = open_device(&library)?;
    let loading = Instant::now();
    let sources = kernel_sources(&library, settings).map_err(|error| runtime("the kernels would not load", &error))?;
    let kernels = Arc::new(load(&gpu, &sources).map_err(|error| runtime("the kernels would not compile", &error))?);
    log::info!(
        "the kernels compiled and loaded in {:.2} s",
        loading.elapsed().as_secs_f64()
    );

    // The stage graphs' lanes first, so that their ids are the plan's (`0..`), then one transfer lane per upload chain,
    // the most the settings allow; the startup uses as many as its ring needs.
    let machine = startup::read_machine();
    let compute = Plan::lane_width();
    let chains = startup::transfer_lanes(settings, MAX_LANES.saturating_sub(compute + 1));
    let ring = stage_ring(quanta_for(sequences));
    let set =
        open_lane_set(&gpu, compute, ring, chains).map_err(|error| runtime("the lanes would not open", &error))?;
    let lanes = set.queues;
    let lane_table = LaneTable::new(set.typed, Arc::clone(&kernels), Arc::clone(&library));
    let completions = Completions::new(&lanes);
    lattice
        .add_upstream(Arc::clone(&io) as Arc<dyn LatticeSource>)
        .map_err(|error| runtime("the lattice would not take the file I/O", &format!("{error:?}")))?;
    lattice
        .add_upstream(Arc::clone(&completions) as Arc<dyn LatticeSource>)
        .map_err(|error| runtime("the lattice would not take the completions", &format!("{error:?}")))?;
    let pool_bytes = startup::pinned_pool_bytes(settings, &machine);
    let pool = PinnedPool::new(&library, pool_bytes);
    let teardown = Teardown::with_cancel_bound(
        Arc::clone(&lattice),
        vec![Arc::clone(&stages), Arc::clone(&host), Arc::clone(&control)],
        cancel_bound,
    );
    Ok(Arc::new(Core {
        settings: settings.clone(),
        library,
        lattice,
        stages,
        host,
        control,
        io,
        completions,
        gpu,
        lanes,
        lane_table,
        stage_ring: ring,
        compute,
        pool,
        released: Arc::new(ReleasedBuffers::new()),
        host_pinned_bytes: startup::host_pinned_bytes(settings, &machine),
        pool_bytes,
        started,
        teardown,
    }))
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::rc::Rc;
    use std::thread;

    use super::*;

    #[test]
    fn the_stage_ring_holds_the_quanta_in_flight_in_a_power_of_two() {
        let bound = Plan::stage_bound();
        let one = stage_ring(quanta_for(1));
        assert!(one.is_power_of_two() && one >= bound * STAGE_RING_QUANTA);
        for sequences in [1, 2, 3, 8, 100] {
            let ring = stage_ring(quanta_for(sequences));
            assert!(ring.is_power_of_two(), "{sequences}");
            assert!(ring >= bound * STAGE_RING_QUANTA * sequences, "{sequences}");
            assert!(
                ring / 2 < bound * STAGE_RING_QUANTA * sequences,
                "{sequences}: the smallest power of two"
            );
        }
        assert_eq!(
            stage_ring(usize::MAX),
            MAX_STAGE_RING,
            "a count too large for the ring is bounded, not wrapped"
        );
    }

    #[test]
    fn the_rings_are_sized_for_the_stated_count_and_for_one_when_it_is_automatic() {
        let mut settings = BatchingSettings::default();
        assert_eq!(ringed_sequences(&settings).unwrap(), 1);
        settings.max_sequences = 0;
        assert_eq!(ringed_sequences(&settings).unwrap(), 1);
        settings.max_sequences = 4;
        assert_eq!(ringed_sequences(&settings).unwrap(), 4);
    }

    #[test]
    fn a_count_beyond_the_rows_of_a_pass_is_refused_before_any_ring_is_sized() {
        let mut settings = BatchingSettings::default();
        settings.max_sequences = settings.prefill_chunk_tokens;
        assert_eq!(ringed_sequences(&settings).unwrap(), 512);
        settings.max_sequences = 100_000;
        let error = ringed_sequences(&settings).unwrap_err().to_string();
        assert!(error.contains("batching.max-sequences"), "{error}");
        settings.max_sequences = 5;
        settings.max_tokens_per_pass = Auto::Value(4);
        assert!(ringed_sequences(&settings).is_err());
    }

    /// A stand-in for the runtime that counts its drops.
    struct Counted(Rc<Cell<usize>>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.set(self.0.get() + 1);
        }
    }

    #[test]
    #[allow(
        clippy::disallowed_methods,
        reason = "the test stands for a close that outlasts its window"
    )]
    fn a_close_that_spends_the_window_skips_the_free_drops_nothing_and_says_what_is_left() {
        let freed = Cell::new(false);
        let drops = Rc::new(Cell::new(0));
        let window = Duration::from_millis(20);
        let begun = Instant::now();
        let result = stop_within(
            window,
            Counted(Rc::clone(&drops)),
            |_| thread::sleep(Duration::from_millis(40)),
            |_| {
                freed.set(true);
                Ok(())
            },
            |_| "7 bytes stay".to_owned(),
        );
        assert!(begun.elapsed() < Duration::from_secs(5));
        assert!(!freed.get());
        assert_eq!(drops.get(), 0, "the runtime is left undropped");
        match result {
            Err(EngineError::Unfinished { window: reported, left }) => {
                assert_eq!(reported, window);
                assert_eq!(left, "7 bytes stay");
            }
            other => panic!("{other:?}"),
        }
    }

    #[test]
    fn a_close_within_the_window_goes_on_to_free() {
        let freed = Cell::new(false);
        let drops = Rc::new(Cell::new(0));
        let result = stop_within(
            Duration::from_secs(60),
            Counted(Rc::clone(&drops)),
            |_| {},
            |kept| {
                freed.set(true);
                drop(kept);
                Ok(())
            },
            |_| unreachable!("nothing is left"),
        );
        assert!(result.is_ok());
        assert!(freed.get());
        assert_eq!(drops.get(), 1);
    }
}
