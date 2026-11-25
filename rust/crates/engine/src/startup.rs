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
use std::iter;
use std::mem;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use euhedral_core::frames::Frame;
use euhedral_graph::Lake;
use euhedral_inference_cuda::{DeviceBuffer, DevicePtr, HostWeightsArena, PooledPinned};
use euhedral_inference_edrl::{
    HEADER_BYTES, Header, Metadata, ReadPlan, Region, TensorTable, TokenizerFooter, TrellisCompanions, WeightFormat,
    WeightLayout, content_end,
};
use euhedral_inference_settings::Settings;
use euhedral_io::{IoBuf, IoDone, OpenSpec, OwnedFile};
use euhedral_memory::Claim;

use crate::composite::{Composed, Composite, compose, composed_depth, refuse_replaced_on_disk};
use crate::error::{EngineError, PlanError};
use crate::frames::{Kind, step};
use crate::generation::Generations;
use crate::handoff::Handoff;
use crate::model::{
    LayerIndex, MTP_ATTENTION_PACK, ModelConfig, Residence, Role, TrellisParts, WeightRef, Weights, companion_name,
    trellis_logical, validate_inventory,
};
use crate::residency::accounting::{Consumer, PlanReservation};
use crate::residency::budget::{self, HostBudget};
use crate::residency::footprint;
use crate::residency::loader::{self, LoadOutcome, Ring};
use crate::residency::offload::{self, Offload};
use crate::residency::planner::{LocationBytes, Placement, PlanInputs, ResidencyPlan, Tier, plan_with};
use crate::runtime::{Core, EngineRuntime, ReadyHandoff, StartupReport, Units, free_arrival, raw_file};
use crate::runtime::{lane_sizing, resolve_sequences};
use crate::state::SequenceState;
use crate::warmup;
use crate::workspace::{Workspace, input_records_for};

/// The transfer lanes a startup makes: `host.task-parallelism`, the workers that take host work at once, at least 2
/// and at most half of the lanes the library's limit (`max_lanes`) leaves after the `compute` stage-graph lanes and one
/// more, which leaves the other half to the stage graphs.
pub(crate) fn transfer_lanes(settings: &Settings, max_lanes: usize, compute: usize) -> usize {
    let library_lanes = max_lanes.saturating_sub(compute + 1);
    (settings.host.task_parallelism as usize).clamp(2, (library_lanes / 2).max(2))
}

/// The pinned buffers the pool keeps for reuse, and the budget of the upload's staging ring: `memory.pinned-buffer-pool`
/// when stated, else 1/256 of the machine's memory, at least 64 MiB (`CudaGpuMemory.defaultPinnedPoolBytes`, the
/// default `docs/OPERATIONS.md` documents).
pub(crate) fn pinned_pool_bytes(settings: &Settings, machine: &Machine) -> u64 {
    settings
        .memory
        .pinned_buffer_pool
        .value()
        .map_or_else(|| budget::default_pinned_pool_bytes(machine.total), |size| size.bytes())
}

/// What the machine reports of its memory, read once before the lattice exists.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct Machine {
    /// What it can spare now (`MemAvailable`, or the least cgroup headroom when that is less).
    pub(crate) available: Option<u64>,
    /// What it has (`MemTotal`).
    pub(crate) total: Option<u64>,
}

/// The pinned host memory the load may use: `memory.host-weights-limit`, else the host budget (`memory.host-limit`
/// as stated, else what the machine can spare less `memory.host-margin`). `None` when the machine does not report what
/// it can spare and neither limit is set: nothing may spill to the host then. The runtime reserve is 0: the margin
/// covers it. Read once, on the calling thread, before the lattice exists.
pub(crate) fn host_pinned_bytes(settings: &Settings, machine: &Machine) -> Option<u64> {
    if let Some(limit) = settings.memory.host_weights_limit.value() {
        return Some(limit.bytes());
    }
    let stated = settings.memory.host_limit.value().map(|size| size.bytes());
    HostBudget::of(stated, machine.available, settings.memory.host_margin, 0).pinned_bytes()
}

/// Reads the machine's memory: what it can spare now (`MemAvailable`, or the least headroom of the process's cgroup and its
/// ancestors when that is less; headroom counts the cgroup's reclaimable page cache as free) and what it has.
#[cfg(target_os = "linux")]
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: a few small reads of /proc and /sys/fs/cgroup, once, on the calling thread before the lattice exists; they hold no device and never wait on one"
)]
pub(crate) fn read_machine() -> Machine {
    use std::fs;
    use std::path::Path;

    let meminfo = fs::read_to_string("/proc/meminfo").ok();
    let total = meminfo.as_deref().and_then(budget::mem_total);
    let meminfo = meminfo.as_deref().and_then(budget::mem_available);
    let root = Path::new("/sys/fs/cgroup");
    let own = fs::read_to_string("/proc/self/cgroup").ok();
    let mut least: Option<u64> = None;
    if let Some(path) = own.as_deref().and_then(budget::cgroup_path) {
        let mut group = root.join(path.trim_start_matches('/'));
        while group.starts_with(root) {
            let stat_path = group.join("memory.stat");
            let headroom = fs::read_to_string(group.join("memory.max"))
                .ok()
                .zip(fs::read_to_string(group.join("memory.current")).ok())
                .and_then(|(max, current)| {
                    // Limited groups only: an unlimited one needs no stat and no warning.
                    let limited = max.trim() != "max" && !max.trim().is_empty();
                    let stat = limited.then(|| fs::read_to_string(&stat_path)).and_then(|read| {
                        read.map_err(|error| {
                            log::warn!(
                                "{} could not be read ({error}): the page cache counts as used memory",
                                stat_path.display()
                            );
                        })
                        .ok()
                    });
                    budget::cgroup_headroom(&max, &current, stat.as_deref())
                });
            least = match (least, headroom) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            if group == root || !group.pop() {
                break;
            }
        }
    }
    Machine {
        available: budget::available_bytes(meminfo, least),
        total,
    }
}

/// Other platforms report no figure, which the budget reads as unknown: nothing spills to the host unless a limit is set.
#[cfg(not(target_os = "linux"))]
pub(crate) fn read_machine() -> Machine {
    Machine::default()
}

/// The startup in flight: the shared parts, the hand-off to send, and the units that keep the lakes open.
struct Startup {
    core: Arc<Core>,
    ready: Option<ReadyHandoff>,
    /// The plan's admitted reservation, once it is made: the allocations take their claims from it, and the engine
    /// keeps what is left of it.
    reservation: Option<PlanReservation>,
    /// The device's free bytes when the plan was made.
    free_at_plan: u64,
    /// The objects taken from other artifacts; empty for a model loaded from its artifact alone.
    composite: Composite,
    _units: Units,
}

impl Startup {
    /// The reservation the allocations take their claims from.
    fn reservation(&mut self) -> Result<&mut PlanReservation, EngineError> {
        self.reservation
            .as_mut()
            .ok_or_else(|| EngineError::Runtime("the plan was not admitted before its allocations".to_string()))
    }
}

impl Startup {
    /// Ends the startup with `error`.
    fn fail(mut self, error: EngineError) {
        if let Some(ready) = self.ready.take() {
            ready.send(Err(error));
        }
    }

    /// Ends the startup with the runtime.
    fn succeed(mut self, runtime: EngineRuntime) {
        self.core
            .teardown
            .device_bytes
            .store(runtime.report().allocated_bytes, Ordering::Release);
        if let Some(ready) = self.ready.take() {
            // A start abandoned by now has no taker: the arrival frees the runtime itself.
            ready.deliver(Ok(Arc::new(runtime)), free_arrival);
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
        // thread waiting on the hand-off.
        if let Some(ready) = self.ready.take() {
            ready.send(Err(EngineError::Runtime(
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
pub(crate) fn run(core: Arc<Core>, ready: ReadyHandoff, composite: Composite) {
    let units = match core.admit() {
        Ok(units) => units,
        Err(error) => return ready.send(Err(error)),
    };
    let startup = Startup {
        core,
        ready: Some(ready),
        reservation: None,
        free_at_plan: 0,
        composite,
        _units: units,
    };
    startup.then(control, open);
}

/// What the steps after the index need of the artifact.
struct Artifact {
    /// The files payloads are read from: the artifact first, then the source of each replacement of a composite.
    files: Vec<Arc<OwnedFile>>,
    /// The file (an index into `files`) of each entry a replacement supplies; every other entry is in the first.
    source_of: HashMap<String, usize>,
    table: TensorTable,
    config: ModelConfig,
}

/// Reads `range` of `file`; `next` gets the bytes (all of them) or the failure, in a frame of its own.
fn read_range(
    startup: Startup,
    file: &Arc<OwnedFile>,
    path: &Path,
    range: Range<u64>,
    what: &'static str,
    next: impl FnOnce(Startup, Vec<u8>) + Send + 'static,
) {
    let core = Arc::clone(&startup.core);
    let length = (range.end - range.start) as usize;
    let path = path.to_path_buf();
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

/// An artifact file opened, with the regions its index is read from.
struct Indexed {
    path: PathBuf,
    file: Arc<OwnedFile>,
    header: Header,
    end: u64,
    metadata: Vec<u8>,
    table: Vec<u8>,
}

/// Step 1: opens the artifact, then the source of each replacement of the composite, one after the other.
fn open(startup: Startup) {
    let path = startup.core.settings.model.artifact.clone();
    open_index(startup, path, true, |startup, base| sources(startup, base, Vec::new()));
}

/// Opens the next source of the composite, or goes on to parse when each is open.
fn sources(startup: Startup, base: Indexed, mut opened: Vec<Indexed>) {
    let Some(replacement) = startup.composite.replacements.get(opened.len()) else {
        return startup.then(host, move |startup| parse(startup, base, opened));
    };
    let path = replacement.source.clone();
    open_index(startup, path, false, move |startup, part| {
        opened.push(part);
        sources(startup, base, opened);
    });
}

/// Opens `path`, reads its header and footer, then the metadata (only when `with_metadata`: a replacement's is never
/// used) and the table they locate; `next` gets them.
fn open_index(
    startup: Startup,
    path: PathBuf,
    with_metadata: bool,
    next: impl FnOnce(Startup, Indexed) + Send + 'static,
) {
    let core = Arc::clone(&startup.core);
    let named = path.clone();
    let then = Box::new(move |opened: io::Result<OwnedFile>| -> Box<dyn Frame> {
        step(Kind::Startup, 0, move || match opened {
            Ok(file) => indexed(startup, path, file, with_metadata, next),
            Err(error) => {
                let what = format!("opening the artifact {}", path.display());
                startup.fail(EngineError::io(what, error));
            }
        })
    });
    let _ = core.io.open_path(&named, OpenSpec::default(), then);
}

/// The length of `file`, and the file back.
#[cfg(unix)]
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: one fstat at startup, before any frame runs"
)]
fn file_len(file: OwnedFile) -> (io::Result<u64>, OwnedFile) {
    let file = File::from(file);
    (file.metadata().map(|metadata| metadata.len()), OwnedFile::from(file))
}

/// The length of `file`, and the file back.
#[cfg(windows)]
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: one fstat at startup, before any frame runs"
)]
fn file_len(file: OwnedFile) -> (io::Result<u64>, OwnedFile) {
    let file = File::from(file);
    (file.metadata().map(|metadata| metadata.len()), OwnedFile::from(file))
}

/// Reads the header and the footer, then the metadata and the table they locate.
fn indexed(
    startup: Startup,
    path: PathBuf,
    file: OwnedFile,
    with_metadata: bool,
    next: impl FnOnce(Startup, Indexed) + Send + 'static,
) {
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
    let named = path.clone();
    read_range(
        startup,
        &file,
        &named,
        header_range,
        "header",
        move |startup, header| {
            let file = Arc::clone(&handle);
            let named = path.clone();
            read_range(
                startup,
                &handle,
                &named,
                footer_range,
                "footer",
                move |startup, footer| {
                    let parsed = (|| {
                        let header_bytes: [u8; HEADER_BYTES] = header.as_slice().try_into().map_err(|_| {
                            EngineError::Runtime("the header read returned the wrong number of bytes".to_string())
                        })?;
                        let footer_bytes: [u8; 24] = footer.as_slice().try_into().map_err(|_| {
                            EngineError::Runtime("the footer read returned the wrong number of bytes".to_string())
                        })?;
                        let header =
                            Header::parse(&header_bytes).map_err(|error| EngineError::artifact(&path, error))?;
                        let footer = TokenizerFooter::parse(&footer_bytes)
                            .map_err(|error| EngineError::artifact(&path, error))?;
                        let end = content_end(length, footer.as_ref())
                            .map_err(|error| EngineError::artifact(&path, error))?;
                        let plan = ReadPlan::new(&header, length, footer.as_ref())
                            .map_err(|error| EngineError::artifact(&path, error))?;
                        let (Some(metadata), Some(table)) = (plan.range(Region::Metadata), plan.range(Region::Table))
                        else {
                            return Err(EngineError::Runtime("the read plan names no metadata".to_string()));
                        };
                        Ok((header, end, metadata, table))
                    })();
                    let (header, end, metadata_range, table_range) = match parsed {
                        Ok(parsed) => parsed,
                        Err(error) => return startup.fail(error),
                    };
                    let index = Indexed {
                        path,
                        file,
                        header,
                        end,
                        metadata: Vec::new(),
                        table: Vec::new(),
                    };
                    if !with_metadata {
                        return read_table(startup, index, table_range, next);
                    }
                    let handle = Arc::clone(&index.file);
                    let named = index.path.clone();
                    read_range(
                        startup,
                        &handle,
                        &named,
                        metadata_range,
                        "metadata",
                        move |startup, metadata| {
                            read_table(startup, Indexed { metadata, ..index }, table_range, next);
                        },
                    );
                },
            );
        },
    );
}

/// Reads the tensor table of `index`, then hands it on.
fn read_table(
    startup: Startup,
    index: Indexed,
    range: Range<u64>,
    next: impl FnOnce(Startup, Indexed) + Send + 'static,
) {
    let handle = Arc::clone(&index.file);
    let named = index.path.clone();
    read_range(
        startup,
        &handle,
        &named,
        range,
        "tensor table",
        move |startup, table| {
            next(startup, Indexed { table, ..index });
        },
    );
}

/// Step 2: parses the index, checks it against the model, and reads the trellis segment companions.
fn parse(startup: Startup, base: Indexed, parts: Vec<Indexed>) {
    let path = base.path.clone();
    let parsed = (|| {
        let metadata = Metadata::parse(&base.metadata).map_err(|error| EngineError::artifact(&path, error))?;
        let mut table = TensorTable::parse(&base.table, &base.header, base.end)
            .map_err(|error| EngineError::artifact(&path, error))?;
        let mut source_of = HashMap::new();
        if startup.composite.has_objects() {
            let tables = parts
                .iter()
                .map(|part| {
                    TensorTable::parse(&part.table, &part.header, part.end)
                        .map_err(|error| EngineError::artifact(&part.path, error))
                })
                .collect::<Result<Vec<_>, _>>()?;
            Composed { table, source_of } = compose(&table, &startup.composite, &tables)?;
        }
        let mut config = ModelConfig::from(&metadata);
        config.draft_depth = composed_depth(config.draft_depth, &table, &startup.composite)?;
        config.check_supported()?;
        validate_inventory(&config, &table)?;
        Ok::<_, EngineError>((table, config, source_of))
    })();
    let (table, config, source_of) = match parsed {
        Ok(parsed) => parsed,
        Err(error) => return startup.fail(error),
    };
    let files: Vec<Arc<OwnedFile>> = iter::once(base.file)
        .chain(parts.into_iter().map(|part| part.file))
        .collect();
    // Every trellis object's `.segments`: a few bytes each, read and checked before the device is touched.
    let objects: Vec<(String, u64, u64, Range<u64>, usize)> = table
        .entries()
        .iter()
        .filter(|entry| entry.format() == WeightFormat::Trellis && entry.layout() == WeightLayout::TrellisTile16V1)
        .filter_map(|entry| {
            let (n, _) = trellis_logical(entry.shape())?;
            let suh = table.get(&companion_name(entry.name(), ".suh"))?;
            let segments = table.get(&companion_name(entry.name(), ".segments"))?;
            let file = source_of.get(segments.name()).copied().unwrap_or(0);
            Some((
                entry.name().to_string(),
                n,
                *suh.shape().first()?,
                segments.range(),
                file,
            ))
        })
        .collect();
    let readers = files.clone();
    let artifact = Artifact {
        files,
        source_of,
        table,
        config,
    };
    if objects.is_empty() {
        return startup.then(host, move |startup| plan(startup, artifact, HashMap::new()));
    }
    let core = Arc::clone(&startup.core);
    let names: Vec<(String, u64, u64)> = objects
        .iter()
        .map(|(name, n, rows, _, _)| (name.clone(), *n, *rows))
        .collect();
    let reads = SegmentReads::new(objects.len(), move |parts| {
        let checked = check_segments(&names, parts);
        startup.then(host, move |startup| match checked {
            Ok(ends) => plan(startup, artifact, ends),
            Err(error) => startup.fail(error),
        });
    });
    for (part, (_, _, _, range, file)) in objects.into_iter().enumerate() {
        let length = (range.end - range.start) as usize;
        let arrival = Arrival::new(&reads, part);
        let then = Box::new(move |done: IoDone| -> Box<dyn Frame> {
            step(Kind::Segments, part, move || {
                arrival.arrive(done.result.map(|read| {
                    let mut bytes = done.buf.bytes;
                    bytes.truncate(read);
                    bytes
                }));
            })
        });
        let buf = IoBuf {
            bytes: vec![0; length],
            filled: 0,
        };
        // A refused read drops `then` and its arrival without running them, which counts the part as arrived.
        let _ = core.io.read(raw_file(&readers[file]), range.start, buf, then);
    }
}

/// What the continuation of the segment reads gets: each part's bytes or failure, `None` for a part that never ran.
type SegmentParts = Vec<Option<io::Result<Vec<u8>>>>;

/// The fan-in of the segment reads: a count of the parts still to arrive and one slot per part. The part that brings
/// the count to zero runs the continuation, once.
struct SegmentReads {
    pending: AtomicU32,
    parts: Box<[Handoff<io::Result<Vec<u8>>>]>,
    then: Handoff<Box<dyn FnOnce(SegmentParts) + Send>>,
}

impl SegmentReads {
    /// Reads of `count` parts, which `then` follows. Store the count before any arrival is published.
    fn new(count: usize, then: impl FnOnce(SegmentParts) + Send + 'static) -> Arc<SegmentReads> {
        let continuation = Handoff::empty();
        continuation.send(Box::new(then) as Box<dyn FnOnce(SegmentParts) + Send>);
        Arc::new(SegmentReads {
            pending: AtomicU32::new(u32::try_from(count).unwrap_or(u32::MAX)),
            parts: (0..count).map(|_| Handoff::empty()).collect(),
            then: continuation,
        })
    }
}

/// One part's way into [`SegmentReads`]. It counts as arrived when dropped, with its bytes if the part ran and with
/// none if the read was refused or its frame was dropped unrun.
struct Arrival {
    reads: Arc<SegmentReads>,
    part: usize,
    bytes: Option<io::Result<Vec<u8>>>,
}

impl Arrival {
    fn new(reads: &Arc<SegmentReads>, part: usize) -> Arrival {
        Arrival {
            reads: Arc::clone(reads),
            part,
            bytes: None,
        }
    }

    /// Records the part's result; the part counts when the arrival is dropped, at the end of its frame.
    fn arrive(mut self, bytes: io::Result<Vec<u8>>) {
        self.bytes = Some(bytes);
    }
}

impl Drop for Arrival {
    fn drop(&mut self) {
        if let Some(bytes) = self.bytes.take() {
            self.reads.parts[self.part].send(bytes);
        }
        if self.reads.pending.fetch_sub(1, Ordering::AcqRel) == 1
            && let Some(then) = self.reads.then.take()
        {
            then(self.reads.parts.iter().map(Handoff::take).collect());
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

/// Step 3a: resolves the sequence count from the free device memory, which is read after the kernels are loaded and
/// before the lanes exist. Allocates nothing on the device. The lanes open next, in a device frame.
fn plan(startup: Startup, artifact: Artifact, ends: HashMap<String, Arc<[u32]>>) {
    let core = Arc::clone(&startup.core);
    let first = device_inputs(&core).and_then(|inputs| {
        let first = resolve_sequences(&artifact.config, &artifact.table, &core.settings, inputs)?;
        Ok(first)
    });
    match first {
        Ok(first) => startup.then(device, move |startup| open_lanes(startup, artifact, ends, first)),
        Err(error) => startup.fail(error),
    }
}

/// What the plan reads of the device now: the free memory (a failed read is an error, not zero).
fn device_inputs(core: &Core) -> Result<PlanInputs, EngineError> {
    let device_free = core.library.free_bytes().map_err(EngineError::from)?;
    Ok(PlanInputs {
        device_free,
        multiprocessors: Some(core.library.device().sm_count),
        exact_numerics: core.library.exact_numerics(),
        host_pinned_bytes: core.host_pinned_bytes,
        sequences: 1,
    })
}

/// Step 3b, in a frame of the device lake: opens the lanes with rings for the first count, whose events take device
/// memory, then reads the free memory again, which is the figure the plan is made from, and resolves an automatic count
/// again ([`lane_sizing`]).
fn open_lanes(startup: Startup, artifact: Artifact, ends: HashMap<String, Arc<[u32]>>, first: usize) {
    let core = Arc::clone(&startup.core);
    let sized = core.open_lanes(first).and_then(|()| {
        let inputs = device_inputs(&core)?;
        let sizing = lane_sizing(core.settings.batching.max_sequences, first, || {
            resolve_sequences(&artifact.config, &artifact.table, &core.settings, inputs)
        })?;
        Ok((inputs, sizing))
    });
    match sized {
        Ok((inputs, sizing)) => startup.then(host, move |startup| {
            plan_residency(startup, artifact, ends, inputs, sizing.sequences)
        }),
        Err(error) => startup.fail(error),
    }
}

/// Step 3c: plans residency against the device (as the lanes left it) and the host. Allocates nothing on the device.
fn plan_residency(
    mut startup: Startup,
    artifact: Artifact,
    ends: HashMap<String, Arc<[u32]>>,
    inputs: PlanInputs,
    sequences: usize,
) {
    let core = Arc::clone(&startup.core);
    // The figure the plan is made from: the free memory after the lanes opened.
    startup.free_at_plan = inputs.device_free;
    let planned = plan_with(
        &artifact.config,
        &artifact.table,
        &core.settings,
        PlanInputs { sequences, ..inputs },
    )
    .map_err(EngineError::from)
    .and_then(|plan| refuse_offload(&plan.offload).map(|()| plan).map_err(EngineError::from))
    .and_then(|plan| refuse_replaced_on_disk(&plan.device, &artifact.source_of).map(|()| plan))
    // The plan is admitted by the account: its allocations take their claims from the reservation.
    .and_then(|plan| {
        let reservation = core.account.admit(&plan)?;
        let admitted = reservation.admitted();
        log::info!(
            "memory admitted: {} bytes on the device, {} bytes of pinned host memory",
            admitted.device,
            admitted.host
        );
        Ok((plan, reservation))
    });
    match planned {
        Ok((plan, reservation)) => {
            startup.reservation = Some(reservation);
            startup.then(device, move |startup| allocate(startup, artifact, plan, ends))
        }
        Err(error) => startup.fail(error),
    }
}

/// Refuses a plan that holds weights this build cannot run from where they are. Objects in pinned host memory are
/// loaded into the host arena and run: the embedding and the head read in place, the others staged through the device
/// ring the plan carries. Objects read from the artifact file would be left unloaded, and so would staged ones in a
/// plan with no ring. The refusal names the pinned host bytes the plan needs.
fn refuse_offload(offload: &Offload) -> Result<(), PlanError> {
    let staged = offload.host.iter().any(|name| !offload::mapped(name));
    let in_reach = offload.disk.is_empty() && (!staged || (offload.staging.is_some() && offload.host_arena_bytes > 0));
    if in_reach {
        return Ok(());
    }
    Err(PlanError::WeightsOutsideDevice {
        weights: offload.host.len() + offload.disk.len(),
        host_weights: offload.host.len(),
        disk_weights: offload.disk.len(),
        arena_bytes: offload.host_arena_bytes,
        window_bytes: offload.window_bytes(),
    })
}

/// Step 4: allocates the weight arena, the workspace and the sequence state, and zeroes the state on the first
/// transfer lane. The load begins when the zeroing has run.
fn allocate(mut startup: Startup, artifact: Artifact, plan: ResidencyPlan, ends: HashMap<String, Arc<[u32]>>) {
    let core = Arc::clone(&startup.core);
    let allocated = (|| {
        let reservation = startup.reservation()?;
        let ring = Ring::sized(
            core.settings.host.task_parallelism,
            core.pool_bytes,
            core.host_pinned_bytes.unwrap_or(u64::MAX),
            core.transfer_lanes(),
        );
        // Every allocation takes its claim before it allocates: one larger than the plan fails here, naming its consumer.
        let weights_claim = reservation.claim(Consumer::Weights, plan.arena_bytes)?;
        let arena = DeviceBuffer::alloc(&core.library, plan.arena_bytes as usize)?;
        // The ring the staged weights are copied into: claimed before it is allocated, like every other allocation.
        let staging = match plan.offload.staging {
            Some(ring) => {
                let bytes = ring.bytes();
                let claim = reservation.claim(Consumer::StagingRing, bytes)?;
                let size = usize::try_from(bytes).map_err(|_| {
                    EngineError::Runtime(format!("the staging ring of {bytes} bytes does not fit memory"))
                })?;
                Some((DeviceBuffer::alloc(&core.library, size)?, claim))
            }
            None => None,
        };
        // The host arena is one allocation made before any read, so the huge pages are likelier; the objects the plan
        // holds on the host are read into it where kernels read them.
        let host = match plan.offload.host_arena_bytes {
            0 => None,
            bytes => {
                let claim = reservation.claim(Consumer::HostWeights, bytes)?;
                let usize_bytes = usize::try_from(bytes).map_err(|_| {
                    EngineError::Runtime(format!("the host arena of {bytes} bytes does not fit memory"))
                })?;
                Some((Arc::new(HostWeightsArena::alloc(&core.library, usize_bytes)?), claim))
            }
        };
        // Every pinned allocation of the startup happens here: the ring of the upload, and the staging the cache's
        // page tables will take. Once the chains run, the pool allocates nothing.
        let loans = loader::acquire_loans(&core, &ring)?;
        // The workspace's linear scratch is the plan's: the larger of the trellis routes' and the NVFP4 native route's at
        // the chunk's rows; the second region is the tensor-core route alone.
        let regions = [plan.scratch_bytes, plan.decode_scratch_bytes];
        // A drafting sequence admits the quanta of a step together, each reading a record of its own; each sequence has
        // records of its own.
        let records = input_records_for(plan.draft_depth, plan.sequences);
        // The first scratch region is the plan's scratch; the workspace's slots and the decode scratch are the runtime
        // reserve's, which the plan sized for them.
        let in_reserve =
            Workspace::allocation_bytes(&artifact.config, plan.rows, &regions, records) - plan.scratch_bytes;
        let scratch_claim = reservation.claim(Consumer::Scratch, plan.scratch_bytes)?;
        let reserve_claim = reservation.claim(Consumer::RuntimeReserve, in_reserve)?;
        let mut workspace =
            Workspace::allocate_with_records(&core.library, &artifact.config, plan.rows, &regions, records)?;
        workspace.hold(scratch_claim);
        workspace.hold(reserve_claim);
        let (sequences, marker) = core.lane_table().submit_ops(core.transfer_lane(0), false, |kernels| {
            // The first sequence's state and the MTP layer's cache at the longest context, which its pages grow to.
            let first = footprint::first_sequence_bytes(&plan, &artifact.config)
                + footprint::mtp_cache_bytes_of(&plan, &artifact.config);
            let mut sequences = Vec::with_capacity(plan.sequences);
            for index in 0..plan.sequences {
                let mut claims = vec![reservation.claim(Consumer::Cache, plan.kv_device_bytes)?];
                // The first sequence's state is the runtime reserve's; each one beyond it has its own share.
                claims.push(match index {
                    0 => reservation.claim(Consumer::RuntimeReserve, first)?,
                    _ => reservation.claim(Consumer::ExtraSequences, plan.extra_sequence_bytes)?,
                });
                if plan.kv_host_bytes > 0 {
                    claims.push(reservation.claim(Consumer::HostPages, plan.kv_host_bytes / plan.sequences as u64)?);
                }
                let mut state =
                    SequenceState::allocate(&core.library, &core.released, &artifact.config, &plan, kernels)?;
                for claim in claims {
                    state = state.with_claim(claim);
                }
                sequences.push(state);
            }
            // The sequences have one shape: the staging of all of them is taken together, once.
            if let Some(first) = sequences.first() {
                first.prime_staging(&core.pool, sequences.len())?;
            }
            Ok::<_, EngineError>(sequences)
        })?;
        Ok::<_, EngineError>((
            ring,
            arena,
            host,
            staging,
            weights_claim,
            loans,
            workspace,
            sequences,
            marker,
        ))
    })();
    match allocated {
        Ok((ring, arena, host, staging, weights_claim, loans, workspace, sequences, marker)) => {
            // The state is zeroed when the lane reaches the marker; the load starts then.
            let frame = step(Kind::Startup, 0, move || {
                load(
                    startup,
                    artifact,
                    plan,
                    ends,
                    ring,
                    arena,
                    host,
                    staging,
                    weights_claim,
                    loans,
                    workspace,
                    sequences,
                )
            });
            core.completions().after_marker(marker, frame);
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
    pinned: Option<(Arc<HostWeightsArena>, Claim)>,
    staging: Option<(DeviceBuffer, Claim)>,
    weights_claim: Claim,
    loans: Vec<PooledPinned>,
    workspace: Workspace,
    sequences: Vec<SequenceState>,
) {
    let core = Arc::clone(&startup.core);
    let pieces = loader::pieces(&plan, ring.piece_bytes);
    let files: Arc<[Arc<OwnedFile>]> = artifact.files.iter().cloned().collect();
    let names: Arc<[String]> = plan.device.iter().map(|placement| placement.name.clone()).collect();
    // A model of one file has no table of files: every piece reads from the first.
    let file_of: Arc<[usize]> = if artifact.source_of.is_empty() {
        Arc::from([])
    } else {
        names
            .iter()
            .map(|name| artifact.source_of.get(name).copied().unwrap_or(0))
            .collect()
    };
    let chains = ring.depth;
    let (host_arena, host_claim) = match pinned {
        Some((arena, claim)) => (Some(arena), Some(claim)),
        None => (None, None),
    };
    let done = move |outcome: LoadOutcome| {
        // The last frame of the load lets go: the next step is a frame of its own.
        startup.then(host, move |startup| {
            assemble(
                startup,
                artifact,
                plan,
                ends,
                ring,
                (weights_claim, host_claim),
                staging,
                workspace,
                sequences,
                outcome,
            )
        });
    };
    loader::start(
        &core, files, file_of, names, pieces, chains, arena, host_arena, loans, done,
    );
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
    claims: (Claim, Option<Claim>),
    staging: Option<(DeviceBuffer, Claim)>,
    workspace: Workspace,
    sequences: Vec<SequenceState>,
    outcome: LoadOutcome,
) {
    let LoadOutcome { arena, host, result } = outcome;
    let stats = match result {
        Ok(stats) => stats,
        // Freeing the arena waits for the device: it is a frame of the stages lake.
        Err(error) => {
            return startup.then(device, move |startup| {
                drop((arena, host, staging));
                drop((workspace, sequences, claims));
                startup.fail(error);
            });
        }
    };
    let core = Arc::clone(&startup.core);
    let layers = artifact.config.num_hidden_layers;
    let weights = match tensors(&plan, &arena, host.as_ref(), &ends, layers) {
        Ok(weights) => weights,
        Err(error) => {
            return startup.then(device, move |startup| {
                drop((arena, host, staging));
                drop((workspace, sequences, claims));
                startup.fail(error);
            });
        }
    };
    let objects = plan.on_device().count();
    let host_backed_bytes = host_backed_bytes(&plan);
    let disk_backed_bytes = disk_backed_bytes(&plan);
    let staging_bytes = plan.offload.staging_bytes();
    let staging_geometry = plan.offload.staging;
    let mapped_objects = mapped_objects(&plan);
    let (weights_claim, host_claim) = claims;
    let mut weights = Weights::new(arena, host, weights).with_claim(weights_claim);
    if let Some(claim) = host_claim {
        weights = weights.with_claim(claim);
    }
    if let Some((buffer, claim)) = staging {
        weights = weights.with_staging(buffer).with_claim(claim);
    }
    let admitted = startup
        .reservation
        .as_ref()
        .map_or(LocationBytes { device: 0, host: 0 }, PlanReservation::admitted);
    let allocated = plan.arena_bytes
        + staging_bytes
        + workspace.retained_bytes()
        + sequences.iter().map(SequenceState::device_bytes).sum::<u64>();
    let begun = core.started;
    // The pool keeps what the load staged, up to `memory.pinned-buffer-pool` when that is stated; trimming frees
    // pinned memory, which waits for the device, so it is a frame of its own, after the load.
    startup.then(device, move |startup| {
        trim_pool(&startup.core);
        // The draft head's token ids come back to the host once, in this frame (the pool may allocate here): a drafted
        // token is a row of the head, and the verifier's tokens are the model's.
        let ids = match draft_ids_loan(&startup.core, &plan, &weights) {
            Ok(ids) => ids,
            Err(error) => {
                drop((weights, workspace, sequences));
                return startup.fail(error);
            }
        };
        let (warmed, marker) = warmup::run(&startup.core, &artifact.config, &weights, &workspace);
        let core = Arc::clone(&startup.core);
        let free_at_plan = startup.free_at_plan;
        let (completions, stages) = (Arc::clone(core.completions()), Arc::clone(&core.stages));
        // The report and the latch wait for the lane to reach the warm-up's last launch: a completion, not a wait.
        let ready = step(Kind::Startup, 0, move || {
            let in_use = core
                .library
                .free_bytes()
                .map(|free| core.library.device().total_bytes.saturating_sub(free))
                .unwrap_or(0);
            if host_backed_bytes > 0 {
                log::info!(
                    "weights in pinned host memory: {host_backed_bytes} bytes, {} read in place: {}",
                    mapped_objects.len(),
                    mapped_objects.join(", ")
                );
            }
            if let Some(ring) = staging_geometry {
                log::info!(
                    "staging ring: {} slots of {} bytes on the device, {staging_bytes} bytes",
                    ring.slots,
                    ring.slot_bytes
                );
            }
            let report = StartupReport {
                ready: begun.elapsed(),
                load: stats.elapsed,
                uploaded_bytes: stats.bytes,
                pieces: stats.pieces,
                chains: ring.depth,
                piece_bytes: ring.piece_bytes,
                objects,
                host_backed_bytes,
                disk_backed_bytes,
                staging_bytes,
                mapped_objects,
                allocated_bytes: allocated,
                admitted,
                device_bytes_in_use: in_use,
                free_at_plan,
                pinned_staging_bytes: core.pool.allocated_bytes(),
                pinned_growth_during_load: stats.pinned_growth,
                warmed,
            };
            let config = Arc::new(artifact.config);
            let workspace = Arc::new(workspace);
            let finish = Finish {
                startup,
                plan,
                config,
                weights,
                workspace,
                sequences,
                report,
            };
            match ids {
                None => finish.run(None),
                Some(ids) => ids.copy(&core, finish),
            }
        });
        match marker {
            Some(marker) => completions.after_marker(marker, ready),
            None => {
                let _ = stages.offer(ready);
            }
        }
    });
}

/// The last step of the startup: builds the generation over the model and sets the latch.
struct Finish {
    startup: Startup,
    plan: ResidencyPlan,
    config: Arc<ModelConfig>,
    weights: Weights,
    workspace: Arc<Workspace>,
    sequences: Vec<SequenceState>,
    report: StartupReport,
}

impl Finish {
    /// Takes what the generations need of the reservation: the prefix cache's claim, if the plan grants it memory.
    fn claim_prefix(&mut self) -> Result<(PlanReservation, Option<Claim>), EngineError> {
        let mut reservation = self.startup.reservation.take().ok_or_else(|| {
            EngineError::Runtime("the plan was not admitted before the engine was assembled".to_string())
        })?;
        let claim = match self.plan.prefix_cache_bytes {
            0 => None,
            bytes => Some(reservation.claim(Consumer::PrefixCache, bytes)?),
        };
        Ok((reservation, claim))
    }

    fn run(mut self, draft_ids: Option<Arc<[u32]>>) {
        let (reservation, prefix_claim) = match self.claim_prefix() {
            Ok(taken) => taken,
            Err(error) => return self.fail(error),
        };
        let Finish {
            startup,
            plan,
            config,
            weights,
            workspace,
            sequences,
            report,
        } = self;
        let core = Arc::clone(&startup.core);
        // Every sequence gets an owner, and waits idle in the pool for a request.
        match Generations::new(
            &core,
            &config,
            &plan,
            &weights,
            &workspace,
            sequences,
            draft_ids,
            prefix_claim,
        ) {
            Ok(generations) => {
                let runtime =
                    EngineRuntime::assemble(core, plan, config, weights, workspace, generations, reservation, report);
                startup.succeed(runtime);
            }
            // The plan is refused: what is on the device is freed in a frame of the stages lake, which may wait.
            Err(refused) => {
                let (error, sequences) = *refused;
                let stages = Arc::clone(&core.stages);
                let _ = stages.offer(step(Kind::Startup, 0, move || {
                    drop((weights, workspace, sequences));
                    startup.fail(error);
                }));
            }
        }
    }

    /// Ends the startup with `error`, freeing what is on the device in a frame of the stages lake, which may wait.
    fn fail(self, error: EngineError) {
        let Finish {
            startup,
            weights,
            workspace,
            sequences,
            ..
        } = self;
        let stages = Arc::clone(&startup.core.stages);
        let _ = stages.offer(step(Kind::Startup, 0, move || {
            drop((weights, workspace, sequences));
            startup.fail(error);
        }));
    }
}

/// A pinned buffer that will hold the draft head's token ids, and where they are on the device.
struct DraftIds {
    loan: PooledPinned,
    source: DevicePtr,
    bytes: usize,
}

impl DraftIds {
    /// Copies the ids to the host on the owner's lane and finishes the startup when the lane has reached the copy.
    fn copy(self, core: &Arc<Core>, finish: Finish) {
        let DraftIds {
            mut loan,
            source,
            bytes,
        } = self;
        let queued = core.lane_table().submit_ops(core.owner_lane(), false, |kernels| {
            // SAFETY: the loan is held by the frame that follows the copy, which runs once the lane has reached the
            // marker; nothing reads or writes the bytes before.
            unsafe { kernels.copy_device_to_readback(loan.buffer_mut(), 0, source, bytes) }
        });
        match queued {
            Ok(((), marker)) => {
                let frame = step(Kind::Startup, 0, move || {
                    let ids: Arc<[u32]> = loan.buffer().as_slice()[..bytes]
                        .as_chunks::<4>()
                        .0
                        .iter()
                        .map(|word| u32::from_ne_bytes(*word))
                        .collect();
                    finish.run(Some(ids));
                });
                core.completions().after_marker(marker, frame);
            }
            Err(error) => {
                // The copy may have been queued before the call failed: the lane may still write the buffer, so it is
                // not given back to the pool.
                log::error!("the draft head's token ids could not be copied to the host: {error}");
                mem::forget(loan);
                finish.fail(EngineError::from(error));
            }
        }
    }
}

/// The pinned buffer for the draft head's token ids, when the plan drafts. A plan that drafts and cannot have them is
/// an error: speculation is not silently left off.
fn draft_ids_loan(core: &Core, plan: &ResidencyPlan, weights: &Weights) -> Result<Option<DraftIds>, EngineError> {
    if plan.draft_depth == 0 {
        return Ok(None);
    }
    let ids = weights
        .get(LayerIndex::GLOBAL, Role::DraftTokenIds)
        .ok_or_else(|| EngineError::Runtime("the artifact drafts and has no draft head token ids".to_string()))?;
    let bytes = ids
        .shape
        .first()
        .and_then(|rows| usize::try_from(*rows).ok())
        .and_then(|rows| rows.checked_mul(4))
        .ok_or_else(|| EngineError::Runtime(format!("the draft head's token ids have the shape {:?}", ids.shape)))?;
    let loan = core.pool.acquire(bytes)?;
    Ok(Some(DraftIds {
        loan,
        source: ids.ptr,
        bytes,
    }))
}

#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, freeing pinned memory waits for the device: the startup trims its staging once, when the weights are on the device and no copy is in flight, "
)]
fn trim_pool(core: &Core) {
    core.pool.trim();
}

/// The payload bytes of the objects the plan holds in pinned host memory.
fn host_backed_bytes(plan: &ResidencyPlan) -> u64 {
    plan.device
        .iter()
        .filter(|placement| matches!(placement.tier, Tier::Host { .. }))
        .map(|placement| placement.bytes)
        .sum()
}

/// The payload bytes of the objects the plan leaves in the artifact file.
fn disk_backed_bytes(plan: &ResidencyPlan) -> u64 {
    plan.device
        .iter()
        .filter(|placement| matches!(placement.tier, Tier::Disk))
        .map(|placement| placement.bytes)
        .sum()
}

/// The names of the objects the plan holds in pinned host memory and kernels read in place.
fn mapped_objects(plan: &ResidencyPlan) -> Vec<String> {
    plan.device
        .iter()
        .filter(|placement| matches!(placement.tier, Tier::Host { .. }) && offload::mapped(&placement.name))
        .map(|placement| placement.name.clone())
        .collect()
}

/// The tensors of the plan by layer and role, each addressing its place: in `arena` for an object on the device, in
/// `host` for one the plan holds on the host (a kernel reads it through the device address of the host bytes).
fn tensors(
    plan: &ResidencyPlan,
    arena: &DeviceBuffer,
    host: Option<&Arc<HostWeightsArena>>,
    ends: &HashMap<String, Arc<[u32]>>,
    layers: usize,
) -> Result<HashMap<(LayerIndex, Role), WeightRef>, EngineError> {
    let base = arena.ptr();
    let host_base = match host {
        // SAFETY: taking the address is harmless; the weights keep the arena alive for as long as any kernel reads it,
        // and the load has ended, so no read writes the bytes a kernel will read.
        Some(host) => Some(unsafe { host.device_address() }?),
        None => None,
    };
    let by_name: HashMap<&str, &Placement> = plan
        .device
        .iter()
        .map(|placement| (placement.name.as_str(), placement))
        .collect();
    let mut tensors = HashMap::new();
    for placement in &plan.device {
        let Some(key) = placement.key(layers) else { continue };
        let (ptr, residence) = match placement.tier {
            Tier::Device => (base.offset(placement.offset), Residence::Device),
            Tier::Host { offset } => {
                let host_base = host_base.ok_or_else(|| {
                    EngineError::Runtime(format!(
                        "'{}' is held on the host and the plan has no host arena",
                        placement.name
                    ))
                })?;
                let residence = if offload::mapped(&placement.name) {
                    Residence::Mapped
                } else {
                    Residence::Host { offset }
                };
                (host_base.offset(offset), residence)
            }
            Tier::Disk => (
                DevicePtr::NULL,
                Residence::Disk {
                    file_offset: placement.file_offset,
                },
            ),
        };
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
                ptr,
                bytes: placement.bytes,
                residence,
                trellis: trellis.flatten(),
            },
        );
    }
    Ok(tensors)
}

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    reason = "the tests start threads to race senders, and join them"
)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::thread;
    use std::time::Duration;

    use euhedral_inference_settings::ModelSettings;

    use super::*;
    use crate::residency::offload::StagingGeometry;

    #[test]
    fn the_transfer_lane_cap_follows_the_librarys_lane_limit() {
        let mut settings = Settings::with_model(ModelSettings::default());
        settings.host.task_parallelism = 1000;
        assert_eq!(transfer_lanes(&settings, 64, 4), (64 - 5) / 2);
        assert_eq!(transfer_lanes(&settings, 32, 4), (32 - 5) / 2);
        assert_eq!(transfer_lanes(&settings, 8, 4), 2);
    }

    #[test]
    fn a_plan_with_weights_outside_the_device_is_refused_naming_the_host_bytes() {
        use crate::residency::offload::WindowGeometry;
        assert_eq!(refuse_offload(&Offload::default()), Ok(()));
        let offload = Offload {
            host: vec!["text/token_embedding".to_string()],
            disk: vec!["text/layers/0/mlp/down".to_string()],
            staging: None,
            window: Some(WindowGeometry {
                slots: 8,
                slot_bytes: 1 << 20,
            }),
            host_arena_bytes: 3 << 20,
        };
        let error = refuse_offload(&offload).unwrap_err();
        assert!(matches!(error, PlanError::WeightsOutsideDevice { arena_bytes, .. } if arena_bytes == 3 << 20));
        let refusal = error.to_string();
        assert!(
            refusal.contains(&format!("needs {} bytes of pinned host memory", 11 << 20)),
            "{refusal}"
        );
        assert!(
            refusal.contains("1 in host memory, 1 in the artifact file"),
            "{refusal}"
        );
    }

    #[test]
    fn objects_read_in_place_from_host_memory_are_not_refused_and_staged_or_disk_ones_are() {
        let host = |names: &[&str]| Offload {
            host: names.iter().map(ToString::to_string).collect(),
            host_arena_bytes: 4096,
            ..Offload::default()
        };
        assert_eq!(refuse_offload(&host(&["text/token_embedding"])), Ok(()));
        assert_eq!(
            refuse_offload(&host(&["text/token_embedding", "text/output_head"])),
            Ok(())
        );
        // A staged object needs the ring the plan carries, and the arena it is read from.
        assert!(refuse_offload(&host(&["text/token_embedding", "text/layers/0/mlp/down"])).is_err());
        let mut staged = host(&["text/token_embedding", "text/layers/0/mlp/down"]);
        staged.staging = Some(StagingGeometry {
            slots: 4,
            slot_bytes: 4096,
        });
        assert_eq!(refuse_offload(&staged), Ok(()));
        staged.host_arena_bytes = 0;
        assert!(refuse_offload(&staged).is_err());
        let mut with_disk = host(&["text/token_embedding"]);
        with_disk.disk = vec!["text/layers/0/mlp/down".to_string()];
        assert!(refuse_offload(&with_disk).is_err());
    }

    #[test]
    fn host_pinned_bytes_reads_the_limits_and_the_margin_and_is_unknown_without_a_figure() {
        use euhedral_inference_settings::{Auto, ByteSize, ModelSettings};
        const GIB: u64 = 1 << 30;
        let mut settings = Settings::with_model(ModelSettings::default());
        let unknown = Machine::default();
        assert_eq!(host_pinned_bytes(&settings, &unknown), None);
        settings.memory.host_limit = Auto::Value(ByteSize::gib(3));
        assert_eq!(host_pinned_bytes(&settings, &unknown), Some(3 * GIB));
        settings.memory.host_limit = Auto::Auto;
        settings.memory.host_weights_limit = Auto::Value(ByteSize::gib(5));
        assert_eq!(
            host_pinned_bytes(&settings, &unknown),
            Some(5 * GIB),
            "no figure needed"
        );
        settings.memory.host_limit = Auto::Value(ByteSize::gib(3));
        let machine = Machine {
            available: Some(100 * GIB),
            total: None,
        };
        assert_eq!(
            host_pinned_bytes(&settings, &machine),
            Some(5 * GIB),
            "the weights limit wins"
        );
        settings.memory.host_weights_limit = Auto::Auto;
        settings.memory.host_limit = Auto::Auto;
        assert_eq!(host_pinned_bytes(&settings, &machine), Some(90 * GIB));
        settings.memory.host_margin = Auto::Value(ByteSize::gib(1));
        assert_eq!(
            host_pinned_bytes(&settings, &machine),
            Some(99 * GIB),
            "the margin is passed through"
        );
    }

    /// A fan-in of `count` parts whose continuation hands over what it was given and counts its runs.
    fn fan_in(count: usize) -> (Arc<SegmentReads>, Arc<Handoff<SegmentParts>>, Arc<AtomicUsize>) {
        let seen = Handoff::new();
        let runs = Arc::new(AtomicUsize::new(0));
        let reads = SegmentReads::new(count, {
            let (seen, runs) = (Arc::clone(&seen), Arc::clone(&runs));
            move |parts| {
                runs.fetch_add(1, Ordering::SeqCst);
                seen.send(parts);
            }
        });
        (reads, seen, runs)
    }

    #[test]
    fn the_continuation_runs_once_when_every_part_arrived() {
        let (reads, seen, runs) = fan_in(3);
        for part in 0..3 {
            assert_eq!(runs.load(Ordering::SeqCst), 0, "not before the last part");
            Arrival::new(&reads, part).arrive(Ok(vec![part as u8]));
        }
        let parts = seen.wait(Duration::from_secs(5)).expect("the continuation ran");
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        for (part, bytes) in parts.into_iter().enumerate() {
            assert_eq!(bytes.unwrap().unwrap(), vec![part as u8]);
        }
    }

    #[test]
    fn a_part_dropped_unrun_still_counts_and_reaches_the_continuation_empty() {
        let (reads, seen, runs) = fan_in(2);
        Arrival::new(&reads, 0).arrive(Ok(vec![7]));
        drop(Arrival::new(&reads, 1));
        let parts = seen.wait(Duration::from_secs(5)).expect("the continuation ran");
        assert_eq!(runs.load(Ordering::SeqCst), 1);
        assert!(matches!(&parts[0], Some(Ok(bytes)) if bytes == &[7]));
        assert!(parts[1].is_none(), "a part that never ran is told apart");
    }

    #[test]
    fn parts_arriving_from_many_threads_run_the_continuation_once() {
        const PARTS: usize = 8;
        for _ in 0..200 {
            let (reads, seen, runs) = fan_in(PARTS);
            let arrivals: Vec<Arrival> = (0..PARTS).map(|part| Arrival::new(&reads, part)).collect();
            let threads: Vec<_> = arrivals
                .into_iter()
                .enumerate()
                .map(|(part, arrival)| thread::spawn(move || arrival.arrive(Ok(vec![part as u8]))))
                .collect();
            for thread in threads {
                thread.join().unwrap();
            }
            let parts = seen.wait(Duration::from_secs(5)).expect("the continuation ran");
            assert_eq!(runs.load(Ordering::SeqCst), 1);
            assert!(
                parts.iter().all(Option::is_some),
                "every part's bytes arrived before the continuation"
            );
        }
    }
}
