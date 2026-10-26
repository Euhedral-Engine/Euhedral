//! The engine runtime: the lattice, the lakes, the lanes and the model on the device.
//!
//! [`EngineRuntime::start`] builds the lattice and what it runs on, on the calling thread, and hands the rest of the
//! startup to frames: the artifact's index is read by `AsyncIo`, the plan is made, the arena and the sequence state
//! are allocated, the weights are uploaded along their chains, and the modules are warmed. The latch is set when the
//! model is ready, or with the error that stopped it. The calling thread waits for the latch and for nothing else.

use std::fmt;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use euhedral_core::config::LatticeConfig;
use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_core::impls::DefaultExecutor;
use euhedral_graph::{
    AsyncIo, Completions, DeviceQueue, Lake, LaneId, LanePool, LatchSetter, OwnedFile, RawFile, UnitGuard,
};
use euhedral_hardware_utils::util::CpuSet;
use euhedral_inference_cuda::{CudaLane, CudaLibrary, PinnedPool};
use euhedral_inference_settings::{Auto, Settings};

use crate::error::EngineError;
use crate::model::{ModelConfig, Weights};
use crate::residency::planner::ResidencyPlan;
use crate::startup;
use crate::state::SequenceState;
use crate::workspace::Workspace;

/// The events a transfer lane's ring holds: its markers are found by the completion word, not by waits on events, so
/// the ring only has to exist.
const TRANSFER_RING: usize = 2;

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
    /// Work that has to block: the waiting allocators and frees of the device, and the kernel compiler. It runs on a
    /// small lattice of its own, so that no worker of the main lattice waits.
    pub(crate) blocking: Arc<Lake>,
    /// The small lattice behind `blocking`: one CPU of the main lattice's.
    pub(crate) blocking_lattice: Arc<ControlPlaneLattice>,
    pub(crate) io: Arc<AsyncIo>,
    pub(crate) completions: Arc<Completions>,
    pub(crate) lanes: Arc<LanePool>,
    /// The transfer lanes (lane ids `0..`), one per upload chain; the first also zeroes the sequence state.
    pub(crate) transfer: Vec<Arc<CudaLane>>,
    pub(crate) pool: Arc<PinnedPool>,
    /// Pinned host bytes the load may use.
    pub(crate) host_pinned_bytes: u64,
    pub(crate) started: Instant,
    /// Whether the startup delivered a runtime, which then closes the lattice itself.
    pub(crate) delivered: AtomicBool,
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
    _blocking: UnitGuard,
}

impl Core {
    /// Admits a unit on every lake.
    pub(crate) fn admit(&self) -> Result<Units, EngineError> {
        let closed = |error| EngineError::Runtime(format!("a lake would not take work: {error}"));
        Ok(Units {
            _stages: self.stages.admit().map_err(closed)?,
            _host: self.host.admit().map_err(closed)?,
            _control: self.control.admit().map_err(closed)?,
            _blocking: self.blocking.admit().map_err(closed)?,
        })
    }

    /// Closes the lakes and the lattice. Waits for the lattice's workers to stop: shutdown only.
    pub(crate) fn close(&self) {
        self.control.complete_gracefully();
        self.host.complete_gracefully();
        self.stages.complete_gracefully();
        self.blocking.complete_gracefully();
        self.lattice.close();
        self.blocking_lattice.close();
    }
}

/// The numbers of one startup.
#[derive(Clone, Copy, Debug, PartialEq)]
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
    /// Kernel modules compiled and loaded before the model became ready.
    pub warmed: usize,
}

/// Keeps a startup's lattice alive on the calling thread. A lattice stops when its last owner drops it, which waits
/// for its workers; if the last owner were a worker's frame, that wait could never end. Hold the guard until the
/// startup's latch has fired. When the startup failed, dropping the guard stops the lattice, on the dropping thread;
/// when it delivered a runtime, the runtime stops the lattice when it drops.
pub struct StartupGuard {
    _core: Option<Arc<Core>>,
}

impl Drop for StartupGuard {
    fn drop(&mut self) {
        if let Some(core) = &self._core
            && !core.delivered.load(Ordering::Acquire)
        {
            core.close();
        }
    }
}

impl fmt::Debug for StartupGuard {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("StartupGuard").finish_non_exhaustive()
    }
}

/// A started engine: the model on the device and the machinery that runs it.
pub struct EngineRuntime {
    core: Arc<Core>,
    plan: ResidencyPlan,
    config: ModelConfig,
    // The device memory is freed after the lattice has stopped (see `Drop`): the fields below it address no frame.
    weights: Weights,
    workspace: Workspace,
    sequence: SequenceState,
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
    /// Call it from the main thread, then wait on the latch. Keep the returned [`StartupGuard`] until the latch has
    /// fired: it keeps the lattice's teardown, if the startup fails, on the calling thread.
    pub fn start(
        settings: &Settings,
        library: &Path,
        ready: LatchSetter<Result<Arc<EngineRuntime>, EngineError>>,
    ) -> StartupGuard {
        match build(settings, library) {
            Ok(core) => {
                let guard = StartupGuard {
                    _core: Some(Arc::clone(&core)),
                };
                startup::run(core, ready);
                guard
            }
            Err(error) => {
                ready.set(Err(error));
                StartupGuard { _core: None }
            }
        }
    }

    pub(crate) fn assemble(
        core: Arc<Core>,
        plan: ResidencyPlan,
        config: ModelConfig,
        weights: Weights,
        workspace: Workspace,
        sequence: SequenceState,
        report: StartupReport,
    ) -> EngineRuntime {
        EngineRuntime {
            core,
            plan,
            config,
            weights,
            workspace,
            sequence,
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
        &self.weights
    }

    /// The workspace, sized for a prefill chunk.
    pub fn workspace(&self) -> &Workspace {
        &self.workspace
    }

    /// The device state of the one sequence the pilot holds.
    pub fn sequence(&self) -> &SequenceState {
        &self.sequence
    }

    /// The device state of the sequence, to reserve and commit its cache.
    pub fn sequence_mut(&mut self) -> &mut SequenceState {
        &mut self.sequence
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

impl Drop for EngineRuntime {
    fn drop(&mut self) {
        // The lattice stops first: no frame runs while the device memory goes. The main thread waits here, which is
        // shutdown's to do.
        self.core.close();
    }
}

/// Builds the lattice, the lakes, the lanes and the pools of a startup, on the calling thread.
fn build(settings: &Settings, library_path: &Path) -> Result<Arc<Core>, EngineError> {
    let started = Instant::now();
    let library = CudaLibrary::load(library_path)?;
    let runtime = |what: &str, error: &dyn fmt::Display| EngineError::Runtime(format!("{what}: {error}"));

    let mut config = LatticeConfig::with_executor(DefaultExecutor::new());
    if let Auto::Value(cpus) = &settings.host.cpus {
        config.allowed_cpus = cpus.ids().map(|cpu| cpu as usize).collect::<CpuSet>();
    }
    let lattice = ControlPlaneLattice::new(config)
        .map_err(|error| runtime("the lattice would not start", &format!("{error:?}")))?;

    // The blocking work gets one CPU of its own lattice. The main lattice keeps all of its CPUs.
    let mut blocking_config = LatticeConfig::with_executor(DefaultExecutor::new());
    blocking_config.name = "EuhedralBlocking".to_string();
    blocking_config.shard_name = "EuhedralBlockingShard".to_string();
    blocking_config.allowed_cpus = lattice.config().allowed_cpus.first().map_or_else(
        || lattice.config().allowed_cpus.clone(),
        |cpu| CpuSet::range(cpu, cpu + 1),
    );
    let blocking_lattice = ControlPlaneLattice::new(blocking_config)
        .map_err(|error| runtime("the blocking lattice would not start", &format!("{error:?}")))?;
    let blocking = Lake::new(
        Arc::clone(&blocking_lattice),
        NonZeroUsize::MIN,
        settings.host.idle_release,
    );

    let sinks = NonZeroUsize::new(settings.host.task_parallelism as usize).unwrap_or(NonZeroUsize::MIN);
    let idle_release = settings.host.idle_release;
    let lake = || Lake::new(Arc::clone(&lattice), sinks, idle_release);
    let (stages, host, control) = (lake(), lake(), lake());

    let io = AsyncIo::open().map_err(|error| EngineError::io("the asynchronous file I/O would not open", error))?;

    // One transfer lane per upload chain, the most the settings allow; the startup uses as many as its ring needs.
    let chains = startup::transfer_lanes(settings);
    let mut transfer = Vec::with_capacity(chains);
    for lane in 0..chains {
        transfer.push(CudaLane::new(&library, LaneId(lane as u16), TRANSFER_RING)?);
    }
    let queues: Vec<Arc<dyn DeviceQueue>> = transfer
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
    let pool = PinnedPool::new(&library, startup::pool_limit(settings));
    Ok(Arc::new(Core {
        settings: settings.clone(),
        library,
        lattice,
        stages,
        host,
        control,
        blocking,
        blocking_lattice,
        io,
        completions,
        lanes,
        transfer,
        pool,
        host_pinned_bytes: startup::host_pinned_bytes(settings),
        started,
        delivered: AtomicBool::new(false),
    }))
}
