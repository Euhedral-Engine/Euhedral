//! The startup, as a chain of frames.
//!
//! Each step is a frame that owns the [`Startup`] and what the steps before it made, does one thing, and offers the
//! next frame to a lake: the control lake for the steps that only follow one another, the host lake for parsing and
//! planning, the stages lake for work that touches the device. Reads go through `AsyncIo` and device work through a
//! transfer lane; a step that has to wait for either registers a continuation and returns. Nothing waits.
//!
//! The steps, in order:
//! 1. open the artifact, read its header and footer, then its metadata and tensor table;
//! 2. parse them, check the inventory against the model, read and check the trellis segment companions;
//! 3. plan residency against the device and the host: everything up to here fails before anything is allocated on the
//!    device;
//! 4. allocate the weight arena, the workspace and the sequence state (zeroed on a transfer lane);
//! 5. upload the weights ([`crate::residency::loader`]) and check each piece on the way;
//! 6. assemble the tensors by layer and role, warm the kernel modules, and set the latch.

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use euhedral_core::frames::Frame;
use euhedral_graph::{IoBuf, IoDone, Join, Lake, LatchSetter, OpenSpec, OwnedFile};
use euhedral_inference_cuda::{DeviceBuffer, Submission, SubmitMode};
use euhedral_inference_edrl::{
    HEADER_BYTES, Header, Metadata, ReadPlan, Region, TensorTable, TokenizerFooter, TrellisCompanions, WeightFormat,
    WeightLayout, content_end,
};
use euhedral_inference_settings::Settings;

use crate::error::EngineError;
use crate::frames::{Kind, step};
use crate::model::{
    LayerIndex, MTP_ATTENTION_PACK, ModelConfig, Role, TrellisParts, WeightRef, Weights, companion_name,
    trellis_logical, validate_inventory,
};
use crate::residency::budget::{self, HostBudget};
use crate::residency::loader::{self, LoadOutcome, Ring};
use crate::residency::planner::{Placement, PlanInputs, ResidencyPlan, plan_with};
use crate::runtime::{Core, EngineRuntime, StartupReport, Units, raw_file};
use crate::state::SequenceState;
use crate::warmup;
use crate::workspace::Workspace;

/// The most transfer lanes (and so upload chains) a startup makes; a library holds 64 lanes in all, and the rest
/// belong to the stage graphs.
const MAX_TRANSFER_LANES: usize = 32;

/// The transfer lanes a startup makes: `host.task-parallelism`, the workers that take host work at once.
pub(crate) fn transfer_lanes(settings: &Settings) -> usize {
    (settings.host.task_parallelism as usize).clamp(2, MAX_TRANSFER_LANES)
}

/// The size the pinned pool is trimmed to once the weights are on the device: `memory.pinned-buffer-pool` when stated;
/// otherwise the pool keeps what it has (no trim).
pub(crate) fn pool_limit(settings: &Settings) -> u64 {
    settings
        .memory
        .pinned_buffer_pool
        .value()
        .map_or(u64::MAX, |size| size.bytes())
}

/// The pinned host memory the load may use: `memory.host-weights-limit`, else the host budget (`memory.host-limit`
/// as stated, else what the machine can spare). Read once, on the calling thread, before the lattice exists.
pub(crate) fn host_pinned_bytes(settings: &Settings) -> u64 {
    if let Some(limit) = settings.memory.host_weights_limit.value() {
        return limit.bytes();
    }
    let stated = settings.memory.host_limit.value().map(|size| size.bytes());
    HostBudget::of(stated, machine_available_bytes(), 0).pinned_bytes()
}

/// The memory the machine and the process's cgroups can spare now (`HostBudget.availableBytes`): `MemAvailable`, or
/// the least headroom of the process's cgroup and its ancestors when that is less.
#[cfg(target_os = "linux")]
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: a few small reads of /proc and /sys/fs/cgroup, once, on the calling thread before the lattice exists; they hold no device and never wait on one"
)]
fn machine_available_bytes() -> Option<u64> {
    use std::fs;
    use std::path::Path;

    let meminfo = fs::read_to_string("/proc/meminfo")
        .ok()
        .and_then(|text| budget::mem_available(&text));
    let root = Path::new("/sys/fs/cgroup");
    let own = fs::read_to_string("/proc/self/cgroup").ok();
    let mut least: Option<u64> = None;
    if let Some(path) = own.as_deref().and_then(budget::cgroup_path) {
        let mut group = root.join(path.trim_start_matches('/'));
        while group.starts_with(root) {
            let headroom = fs::read_to_string(group.join("memory.max"))
                .ok()
                .zip(fs::read_to_string(group.join("memory.current")).ok())
                .and_then(|(max, current)| budget::cgroup_headroom(&max, &current));
            least = match (least, headroom) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            if group == root || !group.pop() {
                break;
            }
        }
    }
    budget::available_bytes(meminfo, least)
}

/// Other platforms report no figure, which the budget reads as no limit.
#[cfg(not(target_os = "linux"))]
fn machine_available_bytes() -> Option<u64> {
    None
}

/// The startup in flight: the shared parts, the latch to set, and the units that keep the lakes open.
struct Startup {
    core: Arc<Core>,
    ready: Option<LatchSetter<Result<Arc<EngineRuntime>, EngineError>>>,
    _units: Units,
}

impl Startup {
    /// Ends the startup with `error`.
    fn fail(mut self, error: EngineError) {
        if let Some(ready) = self.ready.take() {
            ready.set(Err(error));
        }
    }

    /// Ends the startup with the runtime.
    fn succeed(mut self, runtime: EngineRuntime) {
        self.core.delivered.store(true, Ordering::Release);
        if let Some(ready) = self.ready.take() {
            ready.set(Ok(Arc::new(runtime)));
        }
    }

    /// Offers `work` as the next frame of the startup to the lake `pick` chooses. A refused frame is dropped, which
    /// ends the startup (see the `Drop`).
    fn then(self, pick: fn(&Core) -> &Arc<Lake>, work: impl FnOnce(Startup) + Send + 'static) {
        let core = Arc::clone(&self.core);
        let _ = pick(&core).offer(step(Kind::Startup, 0, move || work(self)));
    }
}

impl Drop for Startup {
    fn drop(&mut self) {
        // A frame dropped before it ran (a closed lake, a lattice that is shutting down) must not leave the main
        // thread waiting on the latch.
        if let Some(ready) = self.ready.take() {
            ready.set(Err(EngineError::Runtime(
                "the startup was abandoned: a frame was dropped before it ran".to_string(),
            )));
        }
    }
}

fn control(core: &Core) -> &Arc<Lake> {
    &core.control
}

fn host(core: &Core) -> &Arc<Lake> {
    &core.host
}

/// The lake of the startup's device work. Its waiting calls (the allocators and frees of the device, pinned
/// allocations, the kernel compiler) are startup's and shutdown's alone, each under a scoped allow, and run in the
/// frames of this lake, one at a time, before any request exists.
fn device(core: &Core) -> &Arc<Lake> {
    &core.stages
}

/// Begins the startup: the rest runs as frames.
pub(crate) fn run(core: Arc<Core>, ready: LatchSetter<Result<Arc<EngineRuntime>, EngineError>>) {
    let units = match core.admit() {
        Ok(units) => units,
        Err(error) => return ready.set(Err(error)),
    };
    let startup = Startup {
        core,
        ready: Some(ready),
        _units: units,
    };
    startup.then(control, open);
}

/// What the steps after the index need of the artifact.
struct Artifact {
    file: Arc<OwnedFile>,
    table: TensorTable,
    config: ModelConfig,
}

/// Reads `range` of `file`; `next` gets the bytes (all of them) or the failure, in a frame of its own.
fn read_range(
    startup: Startup,
    file: &Arc<OwnedFile>,
    range: Range<u64>,
    what: &'static str,
    next: impl FnOnce(Startup, Vec<u8>) + Send + 'static,
) {
    let core = Arc::clone(&startup.core);
    let length = (range.end - range.start) as usize;
    let path = core.settings.model.artifact.clone();
    let then = Box::new(move |done: IoDone| -> Box<dyn Frame> {
        step(Kind::Startup, 0, move || match done.result {
            Ok(read) if read == length => {
                let mut bytes = done.buf.bytes;
                bytes.truncate(read);
                next(startup, bytes);
            }
            Ok(read) => startup.fail(EngineError::io(
                format!("reading the {what} of {}", path.display()),
                io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("{read} of {length} bytes arrived"),
                ),
            )),
            Err(error) => startup.fail(EngineError::io(
                format!("reading the {what} of {}", path.display()),
                error,
            )),
        })
    });
    let buf = IoBuf {
        bytes: vec![0; length],
        filled: 0,
    };
    // A submission that fails drops `then`, and the startup with it (see the `Drop`).
    let _ = core.io.read(raw_file(file), range.start, buf, then);
}

/// Step 1: opens the artifact.
fn open(startup: Startup) {
    let core = Arc::clone(&startup.core);
    let path = core.settings.model.artifact.clone();
    let then = Box::new(move |opened: io::Result<OwnedFile>| -> Box<dyn Frame> {
        step(Kind::Startup, 0, move || match opened {
            Ok(file) => indexed(startup, file),
            Err(error) => {
                let what = format!(
                    "opening the artifact {}",
                    startup.core.settings.model.artifact.display()
                );
                startup.fail(EngineError::io(what, error));
            }
        })
    });
    let _ = core.io.open_path(&path, OpenSpec::default(), then);
}

/// The length of `file`, and the file back.
#[cfg(unix)]
fn file_len(file: OwnedFile) -> (io::Result<u64>, OwnedFile) {
    let file = File::from(file);
    (file.metadata().map(|metadata| metadata.len()), OwnedFile::from(file))
}

/// The length of `file`, and the file back.
#[cfg(windows)]
fn file_len(file: OwnedFile) -> (io::Result<u64>, OwnedFile) {
    let file = File::from(file);
    (file.metadata().map(|metadata| metadata.len()), OwnedFile::from(file))
}

/// Reads the header and the footer, then the metadata and the table they locate.
fn indexed(startup: Startup, file: OwnedFile) {
    let path = startup.core.settings.model.artifact.clone();
    let (length, file) = file_len(file);
    let length = match length {
        Ok(length) => length,
        Err(error) => return startup.fail(EngineError::io(format!("measuring {}", path.display()), error)),
    };
    let file = Arc::new(file);
    let probe = match ReadPlan::probe(length) {
        Ok(probe) => probe,
        Err(error) => return startup.fail(EngineError::artifact(&path, error)),
    };
    let (Some(header_range), Some(footer_range)) = (probe.range(Region::Header), probe.range(Region::TokenizerFooter))
    else {
        return startup.fail(EngineError::Runtime("the read plan names no header".to_string()));
    };
    let handle = Arc::clone(&file);
    read_range(startup, &file, header_range, "header", move |startup, header| {
        let file = Arc::clone(&handle);
        read_range(startup, &handle, footer_range, "footer", move |startup, footer| {
            let path = startup.core.settings.model.artifact.clone();
            let parsed = (|| {
                let header_bytes: [u8; HEADER_BYTES] = header.as_slice().try_into().map_err(|_| {
                    EngineError::Runtime("the header read returned the wrong number of bytes".to_string())
                })?;
                let footer_bytes: [u8; 24] = footer.as_slice().try_into().map_err(|_| {
                    EngineError::Runtime("the footer read returned the wrong number of bytes".to_string())
                })?;
                let header = Header::parse(&header_bytes).map_err(|error| EngineError::artifact(&path, error))?;
                let footer =
                    TokenizerFooter::parse(&footer_bytes).map_err(|error| EngineError::artifact(&path, error))?;
                let end = content_end(length, footer.as_ref()).map_err(|error| EngineError::artifact(&path, error))?;
                let plan = ReadPlan::new(&header, length, footer.as_ref())
                    .map_err(|error| EngineError::artifact(&path, error))?;
                let (Some(metadata), Some(table)) = (plan.range(Region::Metadata), plan.range(Region::Table)) else {
                    return Err(EngineError::Runtime("the read plan names no metadata".to_string()));
                };
                Ok((header, end, metadata, table))
            })();
            let (header, end, metadata_range, table_range) = match parsed {
                Ok(parsed) => parsed,
                Err(error) => return startup.fail(error),
            };
            let handle = Arc::clone(&file);
            read_range(startup, &file, metadata_range, "metadata", move |startup, metadata| {
                let file = Arc::clone(&handle);
                read_range(startup, &handle, table_range, "tensor table", move |startup, table| {
                    startup.then(host, move |startup| parse(startup, file, header, end, metadata, table));
                });
            });
        });
    });
}

/// Step 2: parses the index, checks it against the model, and reads the trellis segment companions.
fn parse(startup: Startup, file: Arc<OwnedFile>, header: Header, end: u64, metadata: Vec<u8>, table: Vec<u8>) {
    let path = startup.core.settings.model.artifact.clone();
    let parsed = (|| {
        let metadata = Metadata::parse(&metadata).map_err(|error| EngineError::artifact(&path, error))?;
        let table = TensorTable::parse(&table, &header, end).map_err(|error| EngineError::artifact(&path, error))?;
        let config = ModelConfig::from(&metadata);
        config.check_supported()?;
        validate_inventory(&config, &table)?;
        Ok::<_, EngineError>((table, config))
    })();
    let (table, config) = match parsed {
        Ok(parsed) => parsed,
        Err(error) => return startup.fail(error),
    };
    // Every trellis object's `.segments`: a few bytes each, read and checked before the device is touched.
    let objects: Vec<(String, u64, u64, Range<u64>)> = table
        .entries()
        .iter()
        .filter(|entry| entry.format() == WeightFormat::Trellis && entry.layout() == WeightLayout::TrellisTile16V1)
        .filter_map(|entry| {
            let (n, _) = trellis_logical(entry.shape())?;
            let suh = table.get(&companion_name(entry.name(), ".suh"))?;
            let segments = table.get(&companion_name(entry.name(), ".segments"))?;
            Some((entry.name().to_string(), n, *suh.shape().first()?, segments.range()))
        })
        .collect();
    let reader = Arc::clone(&file);
    let artifact = Artifact { file, table, config };
    if objects.is_empty() {
        return startup.then(host, move |startup| plan(startup, artifact, HashMap::new()));
    }
    let core = Arc::clone(&startup.core);
    let names: Vec<(String, u64, u64)> = objects
        .iter()
        .map(|(name, n, rows, _)| (name.clone(), *n, *rows))
        .collect();
    let join = Join::new(objects.len(), move |parts: Vec<Option<io::Result<Vec<u8>>>>| {
        let checked = check_segments(&names, parts);
        startup.then(host, move |startup| match checked {
            Ok(ends) => plan(startup, artifact, ends),
            Err(error) => startup.fail(error),
        });
    });
    for (part, (_, _, _, range)) in objects.into_iter().enumerate() {
        let length = (range.end - range.start) as usize;
        let arriving = Arc::clone(&join);
        let then = Box::new(move |done: IoDone| -> Box<dyn Frame> {
            step(Kind::Segments, part, move || {
                let bytes = done.result.map(|read| {
                    let mut bytes = done.buf.bytes;
                    bytes.truncate(read);
                    bytes
                });
                arriving.arrive(part, bytes);
            })
        });
        let buf = IoBuf {
            bytes: vec![0; length],
            filled: 0,
        };
        if core.io.read(raw_file(&reader), range.start, buf, then).is_err() {
            join.abandon(part);
        }
    }
}

/// The segment ends of every trellis object, parsed and checked against the object: strictly increasing multiples of
/// 128, at most 8, the last equal to n, one per `suh` row; the packed MTP projection must also have n/2 as a boundary
/// (`CompactWeightLoader.validateTrellisSegments`).
fn check_segments(
    objects: &[(String, u64, u64)],
    parts: Vec<Option<io::Result<Vec<u8>>>>,
) -> Result<HashMap<String, Arc<[u32]>>, EngineError> {
    let mut ends = HashMap::with_capacity(objects.len());
    for ((name, n, suh_rows), part) in objects.iter().zip(parts) {
        let segments = companion_name(name, ".segments");
        let bytes = match part {
            Some(Ok(bytes)) => bytes,
            Some(Err(source)) => return Err(EngineError::io(format!("reading '{segments}'"), source)),
            None => return Err(EngineError::Runtime(format!("the read of '{segments}' did not start"))),
        };
        let invalid = |reason: String| EngineError::Payload {
            name: segments.clone(),
            reason: format!("invalid trellis segments: {reason}"),
        };
        let parsed = TrellisCompanions::parse(&bytes).map_err(|error| invalid(error.to_string()))?;
        TrellisCompanions::check_rows(&parsed, *n, *suh_rows).map_err(|error| invalid(error.to_string()))?;
        if name == MTP_ATTENTION_PACK && !parsed.iter().any(|end| u64::from(*end) == n / 2) {
            return Err(invalid(format!("n/2={} is not a segment boundary", n / 2)));
        }
        ends.insert(name.clone(), Arc::from(parsed));
    }
    Ok(ends)
}

/// Step 3: plans residency against the device and the host. Failing here has allocated nothing on the device.
fn plan(startup: Startup, artifact: Artifact, ends: HashMap<String, Arc<[u32]>>) {
    let core = Arc::clone(&startup.core);
    let planned = core
        .library
        .free_bytes()
        .map_err(EngineError::from)
        .and_then(|device_free| {
            plan_with(
                &artifact.config,
                &artifact.table,
                &core.settings,
                PlanInputs {
                    device_free,
                    multiprocessors: Some(core.library.device().sm_count),
                    host_pinned_bytes: core.host_pinned_bytes,
                },
            )
            .map_err(EngineError::from)
        });
    match planned {
        Ok(plan) => startup.then(device, move |startup| allocate(startup, artifact, plan, ends)),
        Err(error) => startup.fail(error),
    }
}

/// Step 4: allocates the weight arena, the workspace and the sequence state, and zeroes the state on the first
/// transfer lane. The load begins when the zeroing has run.
fn allocate(startup: Startup, artifact: Artifact, plan: ResidencyPlan, ends: HashMap<String, Arc<[u32]>>) {
    let core = Arc::clone(&startup.core);
    let allocated = (|| {
        let largest = plan.device.iter().map(|placement| placement.bytes).max().unwrap_or(0);
        let ring = Ring::sized(&core.settings, largest, core.host_pinned_bytes, core.transfer.len());
        let arena = DeviceBuffer::alloc(&core.library, plan.arena_bytes as usize)?;
        prime_pool(&core, &ring)?;
        // The workspace's linear scratch: the larger of the trellis routes' and the NVFP4 native route's at the
        // chunk's rows; the second region is the tensor-core route alone.
        let chunk = plan.rows.rows;
        let nvfp4_native = plan
            .device
            .iter()
            .filter(|placement| placement.format == WeightFormat::Nvfp4)
            .filter_map(|placement| match placement.shape.as_slice() {
                [out, input] => Some(core.library.nvfp4_native_scratch_bytes(
                    chunk.min(u64::from(u32::MAX)) as u32,
                    *input as u32,
                    *out as u32,
                )),
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let regions = [plan.scratch_bytes.max(nvfp4_native), plan.decode_scratch_bytes];
        let workspace = Workspace::allocate(&core.library, &artifact.config, plan.rows, &regions)?;
        let lane = &core.transfer[0];
        let submission = Submission::begin(lane, SubmitMode::default())?;
        let sequence = SequenceState::allocate(&core.library, &artifact.config, &plan, &submission)?;
        let marker = submission.finish_signalled()?;
        Ok::<_, EngineError>((ring, arena, workspace, sequence, marker))
    })();
    match allocated {
        Ok((ring, arena, workspace, sequence, marker)) => {
            // The state is zeroed when the lane reaches the marker; the load starts then.
            let frame = step(Kind::Startup, 0, move || {
                load(startup, artifact, plan, ends, ring, arena, workspace, sequence)
            });
            core.completions.after_marker(marker, frame);
        }
        Err(error) => startup.fail(error),
    }
}

/// Step 5: uploads the weights.
#[allow(
    clippy::too_many_arguments,
    reason = "the startup's state, moved from one frame to the next"
)]
fn load(
    startup: Startup,
    artifact: Artifact,
    plan: ResidencyPlan,
    ends: HashMap<String, Arc<[u32]>>,
    ring: Ring,
    arena: DeviceBuffer,
    workspace: Workspace,
    sequence: SequenceState,
) {
    let core = Arc::clone(&startup.core);
    let pieces = loader::pieces(&plan, ring.piece_bytes);
    let file = Arc::clone(&artifact.file);
    let names: Arc<[String]> = plan.device.iter().map(|placement| placement.name.clone()).collect();
    let chains = ring.depth;
    let done = move |outcome: LoadOutcome| {
        // The last frame of the load lets go: the next step is a frame of its own.
        startup.then(host, move |startup| {
            assemble(startup, artifact, plan, ends, ring, workspace, sequence, outcome)
        });
    };
    loader::start(&core, file, names, pieces, chains, arena, done);
}

/// Step 6: assembles the tensors by layer and role, warms the kernel modules and sets the latch.
#[allow(
    clippy::too_many_arguments,
    reason = "the startup's state, moved from one frame to the next"
)]
fn assemble(
    startup: Startup,
    artifact: Artifact,
    plan: ResidencyPlan,
    ends: HashMap<String, Arc<[u32]>>,
    ring: Ring,
    workspace: Workspace,
    sequence: SequenceState,
    outcome: LoadOutcome,
) {
    let LoadOutcome { arena, result } = outcome;
    let stats = match result {
        Ok(stats) => stats,
        // Freeing the arena waits for the device: it is a frame of the stages lake.
        Err(error) => {
            return startup.then(device, move |startup| {
                drop(arena);
                drop((workspace, sequence));
                startup.fail(error);
            });
        }
    };
    let core = Arc::clone(&startup.core);
    let layers = artifact.config.num_hidden_layers;
    let weights = tensors(&plan, &arena, &ends, layers);
    let objects = plan.device.len();
    let weights = Weights::new(arena, weights);
    let allocated = plan.arena_bytes + workspace.retained_bytes() + sequence.device_bytes();
    let begun = core.started;
    // The pool keeps what the load staged, up to `memory.pinned-buffer-pool` when that is stated; trimming frees
    // pinned memory, which waits for the device, so it is a frame of its own, after the load.
    startup.then(device, move |startup| {
        trim_pool(&startup.core);
        let warmed = warmup::run(&startup.core, &artifact.config, &weights, &workspace);
        let core = Arc::clone(&startup.core);
        let in_use = core
            .library
            .free_bytes()
            .map(|free| core.library.device().total_bytes.saturating_sub(free))
            .unwrap_or(0);
        let report = StartupReport {
            ready: begun.elapsed(),
            load: stats.elapsed,
            uploaded_bytes: stats.bytes,
            pieces: stats.pieces,
            chains: ring.depth,
            piece_bytes: ring.piece_bytes,
            objects,
            allocated_bytes: allocated,
            device_bytes_in_use: in_use,
            warmed,
        };
        let runtime = EngineRuntime::assemble(core, plan, artifact.config, weights, workspace, sequence, report);
        startup.succeed(runtime);
    });
}

/// Allocates the pinned staging the upload will take, before it starts: `depth` buffers of every size class up to a
/// piece, and two per chain of the piece's. The allocation is the slow part of a pinned buffer and may wait for the
/// driver, so it is done here, in the startup's allocation frame, and the chains find the pool stocked.
fn prime_pool(core: &Core, ring: &Ring) -> Result<(), EngineError> {
    let top = ring.piece_bytes.ilog2();
    for class in 12..=top {
        let count = if class == top { 2 * ring.depth } else { ring.depth };
        let loans = (0..count)
            .map(|_| core.pool.acquire(1 << class))
            .collect::<Result<Vec<_>, _>>()?;
        drop(loans);
    }
    Ok(())
}

#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, freeing pinned memory waits for the device: the startup trims its staging once, when the weights are on the device and no copy is in flight, "
)]
fn trim_pool(core: &Core) {
    core.pool.trim();
}

/// The tensors of the plan by layer and role, each addressing its place in `arena`.
fn tensors(
    plan: &ResidencyPlan,
    arena: &DeviceBuffer,
    ends: &HashMap<String, Arc<[u32]>>,
    layers: usize,
) -> HashMap<(LayerIndex, Role), WeightRef> {
    let base = arena.ptr();
    let by_name: HashMap<&str, &Placement> = plan
        .device
        .iter()
        .map(|placement| (placement.name.as_str(), placement))
        .collect();
    let mut tensors = HashMap::new();
    for placement in &plan.device {
        let Some(key) = placement.key(layers) else { continue };
        let trellis = (placement.format == WeightFormat::Trellis).then(|| {
            let part = |suffix: &str| by_name.get(companion_name(&placement.name, suffix).as_str()).copied();
            let (suh, svh, segments) = (part(".suh")?, part(".svh")?, part(".segments")?);
            Some(TrellisParts {
                suh: base.offset(suh.offset),
                svh: base.offset(svh.offset),
                segments: base.offset(segments.offset),
                segment_ends: ends.get(&placement.name)?.clone(),
                suh_rows: *suh.shape.first()?,
            })
        });
        tensors.insert(
            key,
            WeightRef {
                name: placement.name.clone(),
                shape: placement.shape.clone(),
                dtype: placement.dtype,
                format: placement.format,
                layout: placement.layout,
                ptr: base.offset(placement.offset),
                bytes: placement.bytes,
                trellis: trellis.flatten(),
            },
        );
    }
    tensors
}
