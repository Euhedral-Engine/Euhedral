//! The startup of a Flash-Next engine, as a chain of frames over the runtime's lakes.
//!
//! The dense startup ([`crate::startup`]) is one chain too; this one differs where the model does. The steps, in order:
//! 1. open the artifact and read its header and footer, then its metadata, tables and tokenizer;
//! 2. parse them and check the inventory against the configuration (nothing is allocated yet);
//! 3. open the lanes, then plan the placement of every object against the device and the host, and admit the plan;
//! 4. allocate the weight arena, the host arena, the expert cache's memory and its pinned staging, and the region the
//!    shared expert's widened down projections go to;
//! 5. upload the fixed objects ([`crate::residency::loader`]), then widen the shared down projections on a lane;
//! 6. build the n-gram tables over the file and the expert cache over the lattice, and deliver the [`FlashModel`].
//!
//! The workspace, the sequence state and the graphs are the session's: they are made over the model this delivers.
//! A plan this upload cannot serve (staged or cached fixed objects, pinned n-gram tables, experts all held in RAM) is
//! refused before the device is touched, naming the object.

use std::fmt;
use std::io;
use std::mem::{self, ManuallyDrop};
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use euhedral_core::frames::Frame;
use euhedral_graph::Lake;
use euhedral_inference_cuda::{DeviceBuffer, DevicePtr, HostWeightsArena, PinnedBuffer, PooledPinned};
use euhedral_inference_edrl::{
    DETECT_BYTES, FLASH_HEADER_BYTES, FOOTER_BYTES, FlashArtifact, FlashConfig, FlashHeader, ReadPlan, TokenizerFooter,
    TokenizerSection, content_end, shard_name,
};
use euhedral_inference_settings::Settings;
use euhedral_io::{OpenSpec, OwnedFile};
use euhedral_memory::{Claim, SlotArena, TierPool};

use super::load::{FixedLayout, FixedWeights, PadJob, layer_prefix, pad_launches, padding};
use super::ple::{FileShard, NgramRows, ShardBytes};
use super::sequence::Op;
use super::start::{family_of, one_sequence};
use super::{FlashError, Weight};
use crate::error::EngineError;
use crate::frames::{Kind, step};
use crate::handoff::Handoff;
use crate::model::flash_config::FlashModelConfig;
use crate::residency::accounting::{Consumer, PlanReservation};
use crate::residency::disk::{ArtifactFile, DiskReadPiece};
use crate::residency::expert_cache::{
    CacheConfig, CacheError, CopyLane, DeviceSlots, ExpertCache, Frames, RuntimeLanes,
};
use crate::residency::experts::{
    ExpertError, ExpertLayout, ExpertStore, FilePartPort, FileRecordSource, HostArena, PartPort, RamTier,
};
use crate::residency::flash::{ExpertStoreMode, FlashMode, FlashPlan, FlashPlanInputs, NgramMode, plan_flash};
use crate::residency::loader::{self, LoadOutcome, Ring};
use crate::runtime::{Core, RuntimeOwner, Units, build};
use crate::startup::{self, control, device, host as host_lake, read_region};

/// What a Flash-Next start delivers: the model on the device, or why it did not start.
pub type FlashReady = Arc<Handoff<Result<FlashModel, EngineError>>>;

/// A chain of reads: each range is read in turn, and `then` gets every one's bytes in order, or the first failure.
type Reads = Box<dyn FnOnce(Result<Vec<Vec<u8>>, EngineError>) + Send>;

/// The widened copies of the shared expert's down projections, one per layer that has one.
#[derive(Clone, Debug)]
pub struct PaddedDown {
    jobs: Vec<PadJob>,
    /// Where the widened copies are; `None` when there are none.
    base: Option<DevicePtr>,
}

impl PaddedDown {
    /// The widened down projection of `layer`; `None` for a layer that has none (the width needed no widening, or the
    /// layer's tensor is not a plain NVFP4 one).
    pub fn weight(&self, layer: u32) -> Option<Weight> {
        let base = self.base?;
        self.jobs.iter().find(|job| job.layer == layer).map(|job| Weight {
            address: base.offset(job.target_offset),
            bytes: job.target_bytes,
        })
    }
}

/// What the model holds on the device and the host, freed together by [`FlashModel::shutdown`]. Fields drop in
/// declaration order: the cache, whose slots and staging the others back, goes first.
struct Held {
    cache: Option<ExpertCache>,
    expert_memory: DeviceBuffer,
    pad: Option<DeviceBuffer>,
    _arena: DeviceBuffer,
    _host: Option<Arc<HostWeightsArena>>,
    /// The arenas' bytes in the memory account, released when they are freed.
    claims: Vec<Claim>,
}

/// A Flash-Next model on the device: its fixed weights by address, its expert cache, its n-gram tables, and the
/// tokenizer it was started with. Built by [`start`]; freed only by [`FlashModel::shutdown`], because freeing waits
/// for the device.
pub struct FlashModel {
    pub artifact: Arc<FlashArtifact>,
    pub dimensions: FlashModelConfig,
    pub plan: FlashPlan,
    pub weights: FixedWeights,
    pub padded_down: PaddedDown,
    /// The n-gram table of each layer that has one, by layer.
    pub tables: Vec<(u32, Arc<NgramRows>)>,
    core: Arc<Core>,
    held: ManuallyDrop<Held>,
    reservation: Option<PlanReservation>,
}

impl fmt::Debug for FlashModel {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("FlashModel").finish_non_exhaustive()
    }
}

impl FlashModel {
    /// The routed-expert cache.
    pub fn cache(&self) -> Option<&ExpertCache> {
        self.held.cache.as_ref()
    }

    /// The device address of the cache's slots: its memory, `plan.expert_cache.bytes()` long.
    pub fn expert_memory(&self) -> DevicePtr {
        self.held.expert_memory.ptr()
    }

    /// Stops the engine: ends the cache's loads, closes the lakes and the lattice with `owner`, then frees the device
    /// memory. Call it from the thread that started the engine, once no frame is running.
    pub fn shutdown(self, owner: RuntimeOwner) {
        let bound = self.core.teardown.cancel_bound;
        let ended = self.held.cache.as_ref().is_none_or(|cache| wait_closed(cache, bound));
        let allocated = self.core.teardown.device_bytes.load(Ordering::Acquire);
        let teardown = Arc::clone(&self.core.teardown);
        owner.shutdown_with(move || {
            if ended {
                self.release();
            } else {
                log::error!("the expert cache did not end its loads: the device memory stays allocated until exit");
                mem::forget(self);
            }
        });
        if !ended {
            // Nothing was freed: the bytes still count as allocated.
            teardown.device_bytes.store(allocated, Ordering::Release);
        }
    }

    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, shutdown's: freeing device memory waits for the device, on the thread that owns the engine"
    )]
    fn release(self) {
        let Self {
            held,
            reservation,
            core,
            ..
        } = self;
        drop(ManuallyDrop::into_inner(held));
        drop(reservation);
        let report = core.account.close();
        core.teardown.leaked_bytes.store(report.bytes(), Ordering::Release);
        if report.is_clean() {
            log::info!("memory account closed: no claim is still held");
        } else {
            log::error!("memory account closed with claims still held: {report}");
        }
    }
}

#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, shutdown's: the owner of the engine waits, before the lattice closes, for the cache's owner to end \
              its loads"
)]
fn wait_closed(cache: &ExpertCache, bound: Duration) -> bool {
    cache.close().wait(bound).is_ok()
}

/// Starts the Flash-Next engine of `settings` on the CUDA library at `library`, and returns at once: the rest runs as
/// frames. The returned hand-off gets the model when it is on the device, or the error that stopped the start (before
/// anything was allocated, when the artifact or the plan is what failed). Keep the owner until the engine is done:
/// [`FlashModel::shutdown`] takes it.
pub fn start(settings: &Settings, library: &Path, close_timeout: Option<Duration>) -> (RuntimeOwner, FlashReady) {
    let ready: FlashReady = Handoff::new();
    match build(settings, library, close_timeout) {
        Ok(core) => {
            let owner = RuntimeOwner::over(Arc::clone(&core.teardown));
            begin(core, Arc::clone(&ready));
            (owner, ready)
        }
        Err(error) => {
            ready.send(Err(error));
            (RuntimeOwner::idle(), ready)
        }
    }
}

/// The start in flight: the shared parts, the hand-off to send, and the units that keep the lakes open.
struct Boot {
    core: Arc<Core>,
    ready: Option<FlashReady>,
    reservation: Option<PlanReservation>,
    _units: Units,
}

impl Boot {
    fn fail(mut self, error: EngineError) {
        if let Some(ready) = self.ready.take() {
            ready.send(Err(error));
        }
    }

    /// Offers `work` as the next frame to the lake `pick` chooses. A refused frame is dropped, which ends the start.
    fn then(self, pick: fn(&Core) -> &Arc<Lake>, work: impl FnOnce(Boot) + Send + 'static) {
        let core = Arc::clone(&self.core);
        let _ = pick(&core).offer(step(Kind::Startup, 0, move || work(self)));
    }

    fn reservation(&mut self) -> Result<&mut PlanReservation, EngineError> {
        self.reservation
            .as_mut()
            .ok_or_else(|| EngineError::Runtime("the plan was not admitted before its allocations".to_string()))
    }
}

impl Drop for Boot {
    fn drop(&mut self) {
        // A frame dropped before it ran must not leave the main thread waiting on the hand-off.
        if let Some(ready) = self.ready.take() {
            ready.send(Err(EngineError::Runtime(
                "the startup was abandoned: a frame was dropped before it ran".to_string(),
            )));
        }
    }
}

fn begin(core: Arc<Core>, ready: FlashReady) {
    let units = match core.admit() {
        Ok(units) => units,
        Err(error) => return ready.send(Err(error)),
    };
    let boot = Boot {
        core,
        ready: Some(ready),
        reservation: None,
        _units: units,
    };
    boot.then(control, open);
}

/// Reads each of `ranges` of `file` in turn, then calls `then` with the bytes of all of them.
fn read_all(
    core: &Arc<Core>,
    file: Arc<OwnedFile>,
    path: PathBuf,
    mut ranges: Vec<(Range<u64>, &'static str)>,
    mut read: Vec<Vec<u8>>,
    then: Reads,
) {
    if ranges.is_empty() {
        return then(Ok(read));
    }
    let (range, what) = ranges.remove(0);
    let next = Arc::clone(core);
    let (handle, named) = (Arc::clone(&file), path.clone());
    read_region(core, &file, &path, range, what, move |bytes| match bytes {
        Ok(bytes) => {
            read.push(bytes);
            read_all(&next, handle, named, ranges, read, then);
        }
        Err(error) => then(Err(error)),
    });
}

/// What the steps after the index need of the artifact file.
struct Opened {
    file: Arc<OwnedFile>,
    length: u64,
    artifact: Arc<FlashArtifact>,
}

/// Step 1: opens the artifact.
fn open(boot: Boot) {
    let core = Arc::clone(&boot.core);
    let path = core.settings.model.artifact.clone();
    let named = path.clone();
    let then = Box::new(move |opened: io::Result<OwnedFile>| -> Box<dyn Frame> {
        step(Kind::Startup, 0, move || match opened {
            Ok(file) => index(boot, path, file),
            Err(error) => {
                let what = format!("opening the artifact {}", path.display());
                boot.fail(EngineError::io(what, error));
            }
        })
    });
    let _ = core.io.open_path(&named, OpenSpec::default(), then);
}

/// Reads the header and the footer, then the metadata, tables and tokenizer they locate, and parses them.
fn index(boot: Boot, path: PathBuf, file: OwnedFile) {
    let (length, file) = startup::file_len(file);
    let length = match length {
        Ok(length) => length,
        Err(error) => return boot.fail(EngineError::io(format!("measuring {}", path.display()), error)),
    };
    let file = Arc::new(file);
    let core = Arc::clone(&boot.core);
    let probe = match ReadPlan::probe_flash(length) {
        Ok(probe) => probe,
        Err(error) => return boot.fail(EngineError::artifact(&path, error)),
    };
    let ranges = probe
        .reads()
        .iter()
        .zip(["header", "footer"])
        .map(|(planned, what)| (planned.range.clone(), what))
        .collect();
    let (named, handle) = (path.clone(), Arc::clone(&file));
    let heads: Reads = Box::new(move |read| match read {
        Ok(read) => locate(boot, named, handle, length, read),
        Err(error) => boot.fail(error),
    });
    read_all(&core, file, path, ranges, Vec::new(), heads);
}

/// The header and the footer are in `read`: checks the family, then reads what they locate.
fn locate(boot: Boot, path: PathBuf, file: Arc<OwnedFile>, length: u64, read: Vec<Vec<u8>>) {
    let located = (|| {
        let [header_bytes, footer_bytes] = read.as_slice() else {
            return Err(EngineError::Runtime(
                "the header and footer reads came back short".to_string(),
            ));
        };
        let head: [u8; DETECT_BYTES] = header_bytes
            .get(..DETECT_BYTES)
            .and_then(|head| head.try_into().ok())
            .ok_or_else(|| EngineError::Runtime("the header read returned too few bytes".to_string()))?;
        family_of(&head)?;
        let header_bytes: [u8; FLASH_HEADER_BYTES] = header_bytes
            .as_slice()
            .try_into()
            .map_err(|_| EngineError::Runtime("the header read returned the wrong number of bytes".to_string()))?;
        let footer_bytes: [u8; FOOTER_BYTES] = footer_bytes
            .as_slice()
            .try_into()
            .map_err(|_| EngineError::Runtime("the footer read returned the wrong number of bytes".to_string()))?;
        let failed = |error| EngineError::artifact(&path, error);
        let header = FlashHeader::parse(&header_bytes).map_err(failed)?;
        let footer = TokenizerFooter::parse(&footer_bytes).map_err(failed)?;
        let plan = ReadPlan::for_flash(&header, length, footer.as_ref()).map_err(failed)?;
        content_end(length, footer.as_ref()).map_err(failed)?;
        let ranges: Vec<(Range<u64>, &'static str)> = plan
            .reads()
            .iter()
            .zip(["metadata", "tables", "tokenizer"])
            .map(|(planned, what)| (planned.range.clone(), what))
            .collect();
        Ok((header, footer, ranges))
    })();
    let (header, footer, ranges) = match located {
        Ok(located) => located,
        Err(error) => return boot.fail(error),
    };
    let core = Arc::clone(&boot.core);
    let (named, handle) = (path.clone(), Arc::clone(&file));
    let regions: Reads = Box::new(move |read| match read {
        Ok(read) => {
            let opened = parse(&named, header, footer, read).map(|artifact| Opened {
                file: handle,
                length,
                artifact: Arc::new(artifact),
            });
            match opened {
                Ok(opened) => boot.then(device, move |boot| lanes(boot, opened)),
                Err(error) => boot.fail(error),
            }
        }
        Err(error) => boot.fail(error),
    });
    read_all(&core, file, path, ranges, Vec::new(), regions);
}

/// Step 2: the artifact from the regions read: metadata, tables and (when the file has one) the tokenizer.
fn parse(
    path: &Path,
    header: FlashHeader,
    footer: Option<TokenizerFooter>,
    mut read: Vec<Vec<u8>>,
) -> Result<FlashArtifact, EngineError> {
    let failed = |error| EngineError::artifact(path, error);
    let body = (read.len() == 3).then(|| read.pop()).flatten();
    let [metadata, tables] = read.as_slice() else {
        return Err(EngineError::Runtime(
            "the metadata and table reads came back short".to_string(),
        ));
    };
    let tokenizer = match (footer, body) {
        (Some(footer), Some(body)) => Some(TokenizerSection::parse(body, &footer).map_err(failed)?),
        _ => None,
    };
    FlashArtifact::parse(header, metadata, tables, tokenizer).map_err(failed)
}

/// The refusal of a plan the upload cannot serve, naming what asks for it.
fn refuse_unserved(plan: &FlashPlan) -> Result<(), EngineError> {
    if plan.expert_store == ExpertStoreMode::RamResident {
        return Err(EngineError::Runtime(
            "the plan holds every expert record in memory, which this start does not preload: raise \
             memory.host-limit or lower memory.host-margin so the plan caches the records"
                .to_string(),
        ));
    }
    if plan.ngram == NgramMode::PinnedArena {
        return Err(EngineError::Runtime(
            "the plan pins the n-gram tables in host memory, which this start does not load: they are read from the \
             artifact file"
                .to_string(),
        ));
    }
    Ok(())
}

/// What the plan and the allocations are made from.
struct Planned {
    opened: Opened,
    dimensions: FlashModelConfig,
    plan: FlashPlan,
    layout: FixedLayout,
    experts: Arc<ExpertLayout>,
    piece: DiskReadPiece,
}

/// Step 3a, in a frame of the device lake: refuses what the start cannot serve, opens the lanes (their rings take device
/// memory), then reads the device's free memory, which is the figure the plan is made from.
fn lanes(boot: Boot, opened: Opened) {
    let core = Arc::clone(&boot.core);
    let free = (|| {
        one_sequence(&core.settings.batching)?;
        let compute = core.settings.memory.weight_streaming.gpu_streams;
        if compute == 0 {
            return Err(EngineError::Runtime(
                "memory.weight-streaming.gpu-streams is 0: a step needs at least 1 stream".to_string(),
            ));
        }
        core.open_lanes(1, compute as usize)?;
        Ok(startup::device_inputs(&core)?.device_free)
    })();
    match free {
        Ok(free) => boot.then(host_lake, move |boot| plan(boot, opened, free)),
        Err(error) => boot.fail(error),
    }
}

/// Step 3b: plans the placement against the device (`device_free`, as the lanes left it) and the host, checks everything
/// the later steps rely on, and admits the plan. Allocates nothing on the device.
fn plan(mut boot: Boot, opened: Opened, device_free: u64) {
    let core = Arc::clone(&boot.core);
    let planned = (|| {
        let dimensions = FlashModelConfig::from_flash(opened.artifact.config())?;
        let tables = opened.artifact.tables();
        let plan = plan_flash(
            &dimensions,
            tables.tensors(),
            tables.banks(),
            &core.settings.memory,
            FlashPlanInputs {
                device_free,
                host: core.host_budget,
                max_context_tokens: core.settings.context.max_tokens as usize,
                mode: FlashMode::TEXT,
            },
        )?;
        refuse_unserved(&plan)?;
        let layout = FixedLayout::of(&plan, tables.tensors()).map_err(flash_failed)?;
        let experts = Arc::new(ExpertLayout::new(tables.banks(), opened.length).map_err(expert_failed)?);
        let streaming = &core.settings.memory.weight_streaming;
        let piece = DiskReadPiece::from_setting(core.settings.memory.disk_read_piece).map_err(EngineError::Runtime)?;
        check_cache(&plan, &experts, streaming.read_split, streaming.concurrent_reads)?;
        check_ngram(opened.artifact.config(), &layout)?;
        let reservation = core.account.admit_flash(&plan)?;
        let admitted = reservation.admitted();
        log::info!(
            "memory admitted: {} bytes on the device, {} bytes of pinned host memory",
            admitted.device,
            admitted.host
        );
        Ok((dimensions, plan, layout, experts, piece, reservation))
    })();
    match planned {
        Ok((dimensions, plan, layout, experts, piece, reservation)) => {
            boot.reservation = Some(reservation);
            let planned = Planned {
                opened,
                dimensions,
                plan,
                layout,
                experts,
                piece,
            };
            boot.then(device, move |boot| allocate(boot, planned));
        }
        Err(error) => boot.fail(error),
    }
}

/// Refuses an expert cache the plan sized for records the layout does not have, and a read setting of zero: the store
/// takes its slots' sizes from the layout, so a plan that differs would load records into slots too small for them.
fn check_cache(
    plan: &FlashPlan,
    experts: &ExpertLayout,
    read_split: u32,
    concurrent_reads: u32,
) -> Result<(), EngineError> {
    let (device, staging) = (plan.expert_cache.slot_bytes, plan.expert_staging.slot_bytes);
    if device != experts.max_record_bytes() {
        return Err(EngineError::Runtime(format!(
            "the plan's expert cache slots hold {device} bytes and the largest expert record is {} bytes",
            experts.max_record_bytes()
        )));
    }
    if staging != experts.slot_bytes() as u64 {
        return Err(EngineError::Runtime(format!(
            "the plan's expert staging slots hold {staging} bytes and a staged record needs {} bytes",
            experts.slot_bytes()
        )));
    }
    if read_split == 0 {
        return Err(EngineError::Runtime(
            "memory.weight-streaming.read-split is 0: a record is read in at least 1 part".to_string(),
        ));
    }
    if concurrent_reads == 0 {
        return Err(EngineError::Runtime(
            "memory.weight-streaming.concurrent-reads is 0: the cache needs at least 1 read in flight".to_string(),
        ));
    }
    Ok(())
}

/// Refuses an artifact whose n-gram shards are not all placed: the tables are built over them after the upload.
fn check_ngram(config: &FlashConfig, layout: &FixedLayout) -> Result<(), EngineError> {
    for layer in 0..config.text.num_layers {
        if !config.has_ngram(layer as usize) {
            continue;
        }
        for shard in 0..config.ngram.split_parts {
            let name = ngram_shard(layer, shard);
            if layout.home(&name).is_none() {
                return Err(EngineError::Runtime(format!("the artifact has no n-gram shard {name}")));
            }
        }
    }
    Ok(())
}

/// The name of shard `shard` of layer `layer`'s n-gram table.
fn ngram_shard(layer: u32, shard: u32) -> String {
    format!("{}ple/ngram/{}", layer_prefix(layer), shard_name(shard))
}

fn flash_failed(error: FlashError) -> EngineError {
    EngineError::Runtime(error.to_string())
}

/// What the allocations made, moved from the allocation to the load and on.
struct Memory {
    ring: Ring,
    loans: Vec<PooledPinned>,
    arena: DeviceBuffer,
    host: Option<Arc<HostWeightsArena>>,
    expert_memory: DeviceBuffer,
    /// The pinned memory expert records pass through on their way to the device, and its claim.
    staging: Arc<HostArena>,
    staging_claim: Claim,
    pool: Arc<TierPool>,
    claims: Vec<Claim>,
}

/// Step 4: allocates, each after its claim: a claim larger than the plan's fails here, naming its consumer.
fn allocate(mut boot: Boot, planned: Planned) {
    let core = Arc::clone(&boot.core);
    let allocated = (|| {
        let reservation = boot.reservation()?;
        let Planned { plan, layout, .. } = &planned;
        let ring = Ring::sized(
            core.settings.host.task_parallelism,
            core.pool_bytes,
            core.host_pinned_bytes.unwrap_or(u64::MAX),
            core.transfer_lanes(),
        );
        let mut claims = vec![reservation.claim(Consumer::Weights, layout.device_bytes())?];
        let arena = DeviceBuffer::alloc(&core.library, sized(layout.device_bytes(), "the weight arena")?)?;
        let host = match layout.host_bytes() {
            0 => None,
            bytes => {
                claims.push(reservation.claim(Consumer::HostWeights, bytes)?);
                Some(Arc::new(HostWeightsArena::alloc(
                    &core.library,
                    sized(bytes, "the host arena")?,
                )?))
            }
        };
        // Every pinned allocation of the startup happens here: the ring of the upload and the experts' staging.
        let loans = loader::acquire_loans(&core, &ring)?;
        let staging_plan = plan.expert_staging;
        let staging_claim = reservation.claim(Consumer::ExpertStaging, staging_plan.bytes())?;
        let slot_bytes = sized(staging_plan.slot_bytes, "an expert staging slot")?;
        let pinned = PinnedBuffer::alloc(&core.library, sized(staging_plan.bytes(), "the expert staging")?)?;
        let staging = HostArena::over(staging_plan.slots, slot_bytes, pinned).map_err(expert_failed)?;
        let cache = plan.expert_cache;
        let expert_memory = DeviceBuffer::alloc(&core.library, sized(cache.bytes(), "the expert cache")?)?;
        let count = u32::try_from(cache.slot_count)
            .map_err(|_| EngineError::Runtime(format!("{} expert cache slots do not fit 32 bits", cache.slot_count)))?;
        let pool = core
            .account
            .expert_pool(reservation, expert_memory.ptr().get(), cache.slot_bytes, count)?;
        Ok::<_, EngineError>(Memory {
            ring,
            loans,
            arena,
            host,
            expert_memory,
            staging,
            staging_claim,
            pool,
            claims,
        })
    })();
    match allocated {
        Ok(memory) => load(boot, planned, memory),
        Err(error) => boot.fail(error),
    }
}

fn sized(bytes: u64, what: &str) -> Result<usize, EngineError> {
    usize::try_from(bytes).map_err(|_| EngineError::Runtime(format!("{what} of {bytes} bytes does not fit memory")))
}

/// Step 5: uploads the fixed objects.
fn load(boot: Boot, planned: Planned, memory: Memory) {
    let core = Arc::clone(&boot.core);
    let Memory {
        ring,
        loans,
        arena,
        host,
        expert_memory,
        staging,
        staging_claim,
        pool,
        claims,
    } = memory;
    let pieces = planned.layout.pieces(ring.piece_bytes);
    let files: Arc<[Arc<OwnedFile>]> = Arc::from([Arc::clone(&planned.opened.file)]);
    let names: Arc<[String]> = planned
        .layout
        .placements()
        .iter()
        .map(|placement| placement.name.clone())
        .collect();
    // One file: no table of files, every piece reads from the first.
    let file_of: Arc<[usize]> = Arc::from([]);
    let chains = ring.depth;
    let done = move |outcome: LoadOutcome| {
        // The last frame of the load lets go: the next step is a frame of its own.
        boot.then(host_lake, move |boot| {
            let kept = Kept {
                expert_memory,
                staging,
                staging_claim,
                pool,
                claims,
            };
            assemble(boot, planned, kept, outcome);
        });
    };
    loader::start(&core, files, file_of, names, pieces, chains, arena, host, loans, done);
}

/// What the load leaves in place.
struct Kept {
    expert_memory: DeviceBuffer,
    staging: Arc<HostArena>,
    staging_claim: Claim,
    pool: Arc<TierPool>,
    claims: Vec<Claim>,
}

/// Step 6: resolves the fixed weights by address.
fn assemble(boot: Boot, planned: Planned, kept: Kept, outcome: LoadOutcome) {
    let LoadOutcome { arena, host, result } = outcome;
    if let Err(error) = result {
        // Freeing the arena waits for the device: it is a frame of the stages lake.
        return boot.then(device, move |boot| {
            drop((arena, host, kept));
            boot.fail(error);
        });
    }
    let resolved = (|| {
        let host_base = match &host {
            // SAFETY: the arena is alive, and the model holds it until it is freed.
            Some(host) => Some(unsafe { host.device_address() }?),
            None => None,
        };
        planned.layout.resolve(arena.ptr(), host_base).map_err(flash_failed)
    })();
    match resolved {
        Ok(weights) => {
            let held = Held {
                cache: None,
                expert_memory: kept.expert_memory,
                pad: None,
                _arena: arena,
                _host: host,
                claims: kept.claims,
            };
            let rest = Rest {
                staging: kept.staging,
                staging_claim: kept.staging_claim,
                pool: kept.pool,
            };
            boot.then(device, move |boot| pad(boot, planned, weights, held, rest));
        }
        Err(error) => boot.then(device, move |boot| {
            drop((arena, host, kept));
            boot.fail(error);
        }),
    }
}

/// What the cache is still to be built from.
struct Rest {
    staging: Arc<HostArena>,
    staging_claim: Claim,
    pool: Arc<TierPool>,
}

/// The multiple the shared down projection's width is widened to.
const PAD_MULTIPLE: u32 = 256;

/// Step 7: widens the shared down projections into a region of their own, and goes on when the lane has done it.
fn pad(mut boot: Boot, planned: Planned, weights: FixedWeights, mut held: Held, rest: Rest) {
    let core = Arc::clone(&boot.core);
    let widened = (|| {
        let config = planned.opened.artifact.config();
        let (hidden, layers) = (config.text.hidden_size, config.text.num_layers);
        let width = config.moe.shared_expert_intermediate_size;
        let padded = width.next_multiple_of(PAD_MULTIPLE);
        let (jobs, bytes) = padding(&weights, layers, hidden, width, padded);
        if jobs.is_empty() {
            return Ok((PaddedDown { jobs, base: None }, None));
        }
        let claim = boot.reservation()?.claim(Consumer::Scratch, bytes)?;
        let region = DeviceBuffer::alloc(&core.library, sized(bytes, "the widened down projections")?)?;
        let base = region.ptr();
        let launches = pad_launches(&jobs, base, hidden, width, padded).map_err(flash_failed)?;
        let (_, marker) = core
            .lane_table()
            .submit_ops(core.transfer_lane(0), false, |kernels| {
                for launch in launches {
                    Op::Launch(launch).queue(kernels).map_err(|error| error.to_string())?;
                }
                Ok::<_, String>(())
            })
            .map_err(|error| EngineError::Runtime(format!("widening the shared down projections: {error}")))?;
        held.claims.push(claim);
        Ok::<_, EngineError>((PaddedDown { jobs, base: Some(base) }, Some((region, marker))))
    })();
    match widened {
        Ok((padded_down, None)) => build_cache(boot, planned, weights, padded_down, held, rest),
        Ok((padded_down, Some((region, marker)))) => {
            held.pad = Some(region);
            // The frame may run on any lake: the cache is built, and memory freed on failure, in the device lake's.
            let frame = step(Kind::Startup, 0, move || {
                boot.then(device, move |boot| {
                    build_cache(boot, planned, weights, padded_down, held, rest)
                });
            });
            core.completions().after_marker(marker, frame);
        }
        Err(error) => {
            drop(held);
            boot.fail(error);
        }
    }
}

/// Step 8: builds the experts' cache and the n-gram tables over the artifact file, and delivers the model.
fn build_cache(
    mut boot: Boot,
    planned: Planned,
    weights: FixedWeights,
    padded_down: PaddedDown,
    mut held: Held,
    rest: Rest,
) {
    let core = Arc::clone(&boot.core);
    let built = (|| {
        let plan = &planned.plan;
        let path = &core.settings.model.artifact;
        let layout = Arc::clone(&planned.experts);
        let file =
            ArtifactFile::open(path).map_err(|error| EngineError::io(format!("opening {}", path.display()), error))?;
        let piece = planned.piece;
        let port: Arc<dyn PartPort> = Arc::new(FilePartPort::new(Arc::clone(&core.io), Arc::new(file), piece));
        let split = core.settings.memory.weight_streaming.read_split as usize;
        let source = FileRecordSource::new(Arc::clone(&layout), port, split).map_err(expert_failed)?;
        let (memory, shards) = match plan.expert_store {
            ExpertStoreMode::RamCached => {
                let (memory, shards) = RamTier::new(Arc::clone(&layout), plan.host.expert_ram_slots, 1)
                    .map_err(expert_failed)?
                    .into_parts();
                (Some(memory), shards)
            }
            _ => (None, Vec::new()),
        };
        let store = Arc::new(ExpertStore::new(source, memory, Arc::clone(&rest.staging)));
        let slots = DeviceSlots::lease(&rest.pool, core.account.holders(), plan.expert_cache.slot_count)
            .map_err(cache_failed)?;
        let free_list = SlotArena::new(
            0,
            plan.expert_staging.slot_bytes,
            plan.expert_staging.slots as u32,
            rest.staging_claim,
        );
        let lanes = Arc::new(RuntimeLanes::new(Arc::clone(&core), 0).map_err(EngineError::Runtime)?);
        let config = CacheConfig {
            store,
            tier: shards,
            device: slots,
            staging_slots: free_list,
            max_reads: core.settings.memory.weight_streaming.concurrent_reads as usize,
            frames: Arc::clone(&lanes) as Arc<dyn Frames>,
            copier: lanes as Arc<dyn CopyLane>,
        };
        let cache = ExpertCache::on_lattice(config, &core.lattice).map_err(cache_failed)?;
        let tables = ngram_tables(&planned, &weights)?;
        Ok::<_, EngineError>((cache, tables))
    })();
    match built {
        Ok((cache, tables)) => {
            held.cache = Some(cache);
            let allocated = planned.layout.device_bytes() + planned.plan.expert_cache.bytes();
            core.teardown.device_bytes.store(allocated, Ordering::Release);
            let Planned {
                opened,
                dimensions,
                plan,
                ..
            } = planned;
            let reservation = boot.reservation.take();
            let model = FlashModel {
                artifact: opened.artifact,
                dimensions,
                plan,
                weights,
                padded_down,
                tables,
                core: Arc::clone(&core),
                held: ManuallyDrop::new(held),
                reservation,
            };
            if let Some(ready) = boot.ready.take() {
                // A start abandoned by now has no taker: the model is released here, on the lake's worker.
                ready.deliver(Ok(model), release_abandoned);
            }
        }
        Err(error) => {
            drop(held);
            boot.fail(error);
        }
    }
}

/// An arrival nobody took: frees the model it carries.
fn release_abandoned(arrival: Result<FlashModel, EngineError>) {
    if let Ok(model) = arrival {
        model.release();
    }
}

fn expert_failed(error: ExpertError) -> EngineError {
    EngineError::Runtime(format!("the expert store: {error}"))
}

fn cache_failed(error: CacheError) -> EngineError {
    EngineError::Runtime(format!("the expert cache: {error}"))
}

/// The n-gram table of each layer that has one, over the artifact file read in place.
fn ngram_tables(planned: &Planned, weights: &FixedWeights) -> Result<Vec<(u32, Arc<NgramRows>)>, EngineError> {
    let config = planned.opened.artifact.config();
    let (heads, parts) = (config.ngram.heads(), config.ngram.split_parts);
    let shard_rows = config.ngram.shard_rows.unsigned_abs();
    let row_width = u32::try_from(u64::from(config.text.hidden_size) / heads.max(1))
        .map_err(|_| EngineError::Runtime("the n-gram row width does not fit 32 bits".to_string()))?;
    let mut out = Vec::new();
    for layer in 0..config.text.num_layers {
        if !config.has_ngram(layer as usize) {
            continue;
        }
        let mut shards: Vec<Arc<dyn ShardBytes>> = Vec::with_capacity(parts as usize);
        for shard in 0..parts {
            let name = ngram_shard(layer, shard);
            let (offset, bytes) = weights.file_range(&name).map_err(flash_failed)?;
            let part = FileShard::over(&planned.opened.file, offset, bytes)
                .map_err(|error| EngineError::io(format!("sharing the artifact for {name}"), error))?;
            shards.push(Arc::new(part));
        }
        let total = shard_rows * u64::from(parts);
        let rows = NgramRows::new(shards, shard_rows, total, row_width).map_err(flash_failed)?;
        out.push((layer, Arc::new(rows)));
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
