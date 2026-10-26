//! The engine runtime: the lattice, the lakes, the lanes and the model on the device.
//!
//! [`EngineRuntime::start`] builds the lattice and what it runs on, on the calling thread, and hands the rest of the
//! startup to frames: the artifact's index is read by `AsyncIo`, the plan is made, the arena and the sequence state
//! are allocated, the weights are uploaded along their chains, and the modules are warmed. The latch is set when the
//! model is ready, or with the error that stopped it. The calling thread waits for the latch and for nothing else.

use std::fmt;
use std::mem::ManuallyDrop;
use std::num::{NonZeroU64, NonZeroUsize};
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::config::{FragmentConfig, LatticeConfig};
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_core::impls::{BaseCloneableObject, DefaultExecutor};
use euhedral_graph::{
    AsyncIo, Completions, DeviceQueue, Lake, LaneId, LanePool, LatchSetter, OwnedFile, RawFile, UnitGuard,
};
use euhedral_hardware_utils::util::CpuSet;
use euhedral_inference_cuda::{CudaLane, CudaLibrary, PinnedPool, RetireList};
use euhedral_inference_settings::{Auto, Settings};

use crate::error::EngineError;
use crate::generation::{AdmissionError, Generation, GenerationRequest, LogitsSink};
use crate::model::{ModelConfig, Weights};
use crate::plan::Plan;
use crate::residency::planner::ResidencyPlan;
use crate::stages::LaneTable;
use crate::startup;
use crate::state::SequenceState;
use crate::warmup::Warmed;
use crate::workspace::Workspace;

/// The events a transfer lane's ring holds: its markers are found by the completion word, not by waits on events, so
/// the ring only has to exist.
const TRANSFER_RING: usize = 2;

/// The events a stage lane's ring holds: the most markers of the lane that stay in flight, until their last consumer
/// has waited for them. A quantum records one marker per stage it runs, at most a few thousand, and a marker is waited
/// for by the quantum itself and by the one admitted behind it. The startup refuses a plan whose quanta would not fit
/// in `STAGE_RING_QUANTA` of them ([`Core::stage_lanes`]).
pub(crate) const STAGE_RING: usize = 8192;

/// The quanta the ring of a stage lane must hold the markers of: the one running, the one admitted behind it, and the
/// one the owner is preparing.
pub(crate) const STAGE_RING_QUANTA: usize = 3;

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
    pub(crate) lanes: Arc<LanePool>,
    /// The lanes the stage graphs run on, by lane id: the plan's compute lanes (`0..` [`Plan::lane_width`]), the lane
    /// the recording of graphs captures on, and the owner's lane (reservations and input records).
    pub(crate) stage_lanes: Vec<Arc<CudaLane>>,
    /// Every lane of the pool in lane id order: the stage lanes, then the transfer lanes.
    pub(crate) lane_table: LaneTable,
    /// The transfer lanes, one per upload chain, with the ids after the stage lanes; the first also zeroes the
    /// sequence state.
    pub(crate) transfer: Vec<Arc<CudaLane>>,
    pub(crate) pool: Arc<PinnedPool>,
    /// Buffers waiting to be freed in stream order: the pages and page tables of the caches, once their last quantum
    /// is gone. Whoever polls the lanes reaps it.
    pub(crate) retire: Arc<RetireList>,
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
    /// The lane recordings of graphs capture on.
    pub(crate) fn capture_lane(&self) -> LaneId {
        LaneId((self.stage_lanes.len() - 2) as u16)
    }

    /// The owner's lane: the sequence's reservations and its quanta's input records.
    pub(crate) fn owner_lane(&self) -> &Arc<CudaLane> {
        &self.stage_lanes[self.stage_lanes.len() - 1]
    }

    /// The lanes the plan's graphs run on.
    pub(crate) fn compute_lanes(&self) -> usize {
        self.stage_lanes.len() - 2
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
}

impl Teardown {
    /// The teardown of `lattice` and the `lakes` that feed it.
    pub fn new(lattice: Arc<ControlPlaneLattice>, lakes: Vec<Arc<Lake>>) -> Arc<Teardown> {
        Arc::new(Teardown {
            lakes,
            lattice,
            device_bytes: AtomicU64::new(0),
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
        if let Some(teardown) = &teardown {
            teardown.close();
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

/// What the engine keeps on the device. Freed only by [`RuntimeOwner::shutdown`], because freeing waits. The sequence
/// is the generation's.
struct DeviceState {
    weights: Weights,
    workspace: Arc<Workspace>,
}

/// A started engine: the model on the device and the machinery that runs it.
pub struct EngineRuntime {
    core: Arc<Core>,
    plan: ResidencyPlan,
    config: Arc<ModelConfig>,
    // Never dropped by `Drop`: freeing waits for the device, so only `release` (the owner's shutdown) frees it.
    device: ManuallyDrop<DeviceState>,
    // The sequence and the graphs that run it. Closed by `release`.
    generation: Generation,
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
    /// runs as frames on the lattice built here. `ready` is set with the runtime when the model is on the device and
    /// the modules are warm, or with the error that stopped the startup, before anything was allocated on the device
    /// when the artifact or the plan is what failed.
    ///
    /// Call it from the main thread, then wait on the latch. Keep the returned [`RuntimeOwner`] until the engine is
    /// done (see its contract): it closes the lattice and frees the device memory, on the calling thread.
    pub fn start(
        settings: &Settings,
        library: &Path,
        ready: LatchSetter<Result<Arc<EngineRuntime>, EngineError>>,
    ) -> RuntimeOwner {
        match build(settings, library) {
            Ok(core) => {
                let owner = RuntimeOwner::over(Arc::clone(&core.teardown));
                startup::run(core, ready);
                owner
            }
            Err(error) => {
                ready.set(Err(error));
                RuntimeOwner { teardown: None }
            }
        }
    }

    pub(crate) fn assemble(
        core: Arc<Core>,
        plan: ResidencyPlan,
        config: Arc<ModelConfig>,
        weights: Weights,
        workspace: Arc<Workspace>,
        generation: Generation,
        report: StartupReport,
    ) -> EngineRuntime {
        EngineRuntime {
            core,
            plan,
            config,
            device: ManuallyDrop::new(DeviceState { weights, workspace }),
            generation,
            report,
        }
    }

    /// The residency plan the model was loaded by.
    pub fn plan(&self) -> &ResidencyPlan {
        &self.plan
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

    /// Reads the device state of the one sequence the engine holds, when no generation is running on it: `None` while
    /// one is.
    pub fn inspect_sequence<R>(&self, read: impl FnOnce(&SequenceState) -> R) -> Option<R> {
        self.generation.inspect(read)
    }

    /// Starts generating from `request`'s prompt on the engine's sequence, and returns at once: the request runs as
    /// frames on the engine's lattice, and its listener is told of it there. The sequence takes one request at a time;
    /// it starts each from nothing, whatever the one before left in it.
    pub fn generate(&self, request: GenerationRequest) -> Result<(), AdmissionError> {
        self.generation.generate(request)
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
        self.generation.teacher_forced(prefix, forced, sink)
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

    /// The buffers the sequence's caches freed in stream order when their last holder let go; the generation source
    /// reaps it (`RetireList::reap`) when it polls the lanes.
    pub fn retire_list(&self) -> &Arc<RetireList> {
        &self.core.retire
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
        let Self { device, generation, .. } = self;
        let DeviceState { weights, workspace } = ManuallyDrop::into_inner(device);
        // The sequence goes first, by `close`, which refuses while a quantum is in flight. The lattice is closed by
        // now, so a quantum still counted will never be dropped: the caches are left allocated (the process is
        // ending) rather than freed under it.
        generation.close();
        drop(workspace);
        drop(weights);
    }
}

/// The fragment configuration of the engine's lattice: RustEE's default, with the most frames a worker takes at once
/// from `host.max-batch-size` when that is stated. A size of 1 or 2 makes the workers poll their caches one frame at
/// a time (`FragmentConfig::polls`), so that other workers can steal what waits behind a heavy frame.
pub fn fragment_config(settings: &Settings) -> FragmentConfig {
    let fragment = FragmentConfig::default();
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

/// Builds the lattice, the lakes, the lanes and the pools of a startup, on the calling thread.
fn build(settings: &Settings, library_path: &Path) -> Result<Arc<Core>, EngineError> {
    let started = Instant::now();
    let library = CudaLibrary::load(library_path)?;
    let runtime = |what: &str, error: &dyn fmt::Display| EngineError::Runtime(format!("{what}: {error}"));

    let config = lattice_config(settings);
    let lattice = ControlPlaneLattice::new(config)
        .map_err(|error| runtime("the lattice would not start", &format!("{error:?}")))?;

    let sinks = NonZeroUsize::new(settings.host.task_parallelism as usize).unwrap_or(NonZeroUsize::MIN);
    let idle_release = settings.host.idle_release;
    let lake = || Lake::new(Arc::clone(&lattice), sinks, idle_release);
    let (stages, host, control) = (lake(), lake(), lake());

    let io = AsyncIo::open().map_err(|error| EngineError::io("the asynchronous file I/O would not open", error))?;

    // The stage graphs' lanes first, so that their ids are the plan's (`0..`), then one transfer lane per upload chain,
    // the most the settings allow; the startup uses as many as its ring needs.
    let machine = startup::read_machine();
    let chains = startup::transfer_lanes(settings, library.max_lanes().saturating_sub(Plan::lane_width() + 2));
    let compute = Plan::lane_width();
    let mut stage_lanes = Vec::with_capacity(compute + 2);
    for lane in 0..compute + 2 {
        stage_lanes.push(CudaLane::new(&library, LaneId(lane as u16), STAGE_RING)?);
    }
    let mut transfer = Vec::with_capacity(chains);
    for lane in 0..chains {
        transfer.push(CudaLane::new(
            &library,
            LaneId((compute + 2 + lane) as u16),
            TRANSFER_RING,
        )?);
    }
    let every: Vec<Arc<CudaLane>> = stage_lanes.iter().chain(&transfer).cloned().collect();
    let lane_table = LaneTable::new(every.clone())?;
    let queues: Vec<Arc<dyn DeviceQueue>> = every
        .iter()
        .map(|lane| Arc::clone(lane) as Arc<dyn DeviceQueue>)
        .collect();
    let lanes = Arc::new(LanePool::new(queues));
    let completions = Completions::new(Arc::clone(&lanes));
    lattice
        .add_upstream(Arc::clone(&io) as Arc<dyn LatticeSource>)
        .map_err(|error| runtime("the lattice would not take the file I/O", &format!("{error:?}")))?;
    lattice
        .add_upstream(Arc::clone(&completions) as Arc<dyn LatticeSource>)
        .map_err(|error| runtime("the lattice would not take the completions", &format!("{error:?}")))?;
    let pool_bytes = startup::pinned_pool_bytes(settings, &machine);
    let pool = PinnedPool::new(&library, pool_bytes);
    let teardown = Teardown::new(
        Arc::clone(&lattice),
        vec![Arc::clone(&stages), Arc::clone(&host), Arc::clone(&control)],
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
        lanes,
        stage_lanes,
        lane_table,
        transfer,
        pool,
        retire: Arc::new(RetireList::new()),
        host_pinned_bytes: startup::host_pinned_bytes(settings, &machine),
        pool_bytes,
        started,
        teardown,
    }))
}
