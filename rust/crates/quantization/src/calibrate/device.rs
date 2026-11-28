//! The calibration's calls on the card: the layer pipeline over the hidden states, the GPTQ lanes, the resume file and the manifest.
//!
//! One layer is a chain of continuations, each started by the frame that ends the one before it:
//! 1. upload the staged layer on the main lane;
//! 2. the observe pass: the pipeline runs the layer with an observer that adds each distinct input's products into a
//!    Hessian, and the outputs are discarded;
//! 3. symmetrize the Hessians and allocate each linear's decoded weight, on the main lane;
//! 4. round every linear on the GPTQ lanes ([`super::rounding`]), which write their states;
//! 5. swap the decoded weights in, free the Hessians, on the main lane;
//! 6. the write pass: the pipeline runs the rounded layer and its outputs replace the states;
//! 7. free the layer, and record the layer in the manifest ([`super::manifest`]).
//!
//! The pipeline's lanes and the GPTQ lanes are two pools of one device, and the device cannot order work of one
//! against the other. Every hand-off between them is therefore a frame the completions of the lanes that ended the
//! work emit: the work is over on the card, and its result is visible on the host, before the next step queues.

use std::cell::RefCell;
use std::fmt::{Debug, Display, Formatter, Result as FmtResult};
use std::iter::once;
use std::mem::forget;
use std::path::{Path, PathBuf};
use std::slice;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_gpu::cuda::{CudaGpu, CudaModule, CudaQueue, CudaStream};
use euhedral_gpu::{Gpu, GpuError, Lane, Stream, StreamOptions, open_shared_lanes};
use euhedral_graph::{Completions, DeviceQueue, Lake, LaneId, Marker};
use euhedral_inference_cuda::launch::Modes;
use euhedral_inference_cuda::launch::load::{KernelSources, KernelTable, load};
use euhedral_inference_cuda::launch::ops::RusteeOps;
use euhedral_inference_cuda::math::{LaneHandles, MathLibraries};
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, KernelOps, LaneOps, PinnedBuffer, PinnedPool, PooledPinned};
use euhedral_inference_engine::Handoff;
use euhedral_io::AsyncIo;
use euhedral_memory::{LocationId, LocationKind, LocationSpec, OwnerSpec, Planner, Priority, Scope};

use super::draft_ids::check_draft_ids;
use super::driver::{Once, Then, end, single};
use super::embedding;
use super::files::{Region, resume, save, write_checkpoint};
use super::heads::{run as run_heads, run_trellis};
use super::hessians::HessianSet;
use super::manifest::{HEADS_DIRECTORY, Ledger, MANIFEST_FILE, Manifest, RESUME_FILE};
use super::plan::{Plan, round_order};
use super::rounding::{Ended, HEAD_ROWS, Linear, RoundSettings, Services, prewarm, round_all, rounded};
use super::{CalibrateError, Stages};
use crate::checkpoint::CheckpointError;
use crate::checkpoint::objects::{MTP_OBJECTS, Qwen38Objects};
use crate::forward::pipeline::device::{ForwardPipeline, HostStates, Parts, PipelineObserver, Request};
use crate::forward::{ForwardError, ForwardKernels, LaneContext, LayerForward, Observation};
use crate::forward::{LayerWeights, StagedLayer, Target, stage};
use crate::gptq::format::Nvfp4Search;
use crate::gptq::state::QuantState;
use crate::gptq::{GptqKernels, GptqLane, GptqSettings, ObjectShape, lane_staging};
use crate::kernels::{Module, ModuleOptions, Modules};
use crate::memory::LaneMemory;
use crate::recipe::Recipe;
use crate::resume::ResumeFile;
use crate::step::{Kind, idle, step};

mod heads;
mod staging;
mod trellis;

use super::head_state::HeadState;
use heads::{Call, DRAFT_HEAD, DeviceHeads, INPUT_PROJECTION, OUTPUT_HEAD, head_groups, read_tensors};
use staging::HeadStaging;
use trellis::{Opening, Trellis, TrellisDeviceHeads, round_layer};

pub use trellis::TrellisRun;

/// The checkpoint tensor the tokens are looked up in.
const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
const BF16_BYTES: usize = 2;
/// Bytes of the hidden states one read or write of the resume file moves.
const STATES_CHUNK: usize = 256 << 20;
/// FP32 bytes of a Hessian element.
const FP32_BYTES: usize = 4;

/// The default dampening.
pub const DEFAULT_DAMP: f32 = 0.01;
/// The default sequences at once.
pub const DEFAULT_BATCH: usize = 8;
/// Rounding lanes.
pub const ROUNDING_LANES: usize = 3;
/// Device bytes the planner leaves for what it does not account: the layer, its Hessians, its decoded weights and the
/// libraries' workspaces.
pub const DEVICE_RESERVE: u64 = 8 << 30;
/// Pinned bytes the layers are read into.
pub const STAGING_BYTES: u64 = 4 << 30;
/// The fewest pinned bytes a trellis run's lanes stage through when the budget is not set; the GPTQ lane's warm set
/// when that is more.
pub const TRELLIS_PINNED_BYTES: u64 = 1 << 30;
/// Bytes of one upload copy.
pub const UPLOAD_CHUNK_BYTES: usize = 64 << 20;

/// What a calibration runs on and with.
#[derive(Clone, Debug)]
pub struct Settings {
    /// Where the objects' states and the resume file go.
    pub out: PathBuf,
    pub recipe: Recipe,
    pub search: Nvfp4Search,
    pub damp: f32,
    pub act_order: bool,
    /// Sequences run at once.
    pub batch: usize,
    /// Tokens of a sequence.
    pub length: usize,
    /// The calibration tokens, `sequences * length` of them.
    pub tokens: Vec<i32>,
    /// Linears rounded at once.
    pub lanes: usize,
    pub staging_bytes: u64,
    /// Pinned bytes the rounding lanes stage through, at least the staging the GPTQ lanes hold at most at once; none
    /// takes exactly that in a layer run, and at least [`TRELLIS_PINNED_BYTES`] in a trellis run.
    pub lane_pinned_bytes: Option<u64>,
    /// Device bytes left for what the planner does not account; none leaves a third of what is free, at most
    /// [`DEVICE_RESERVE`].
    pub device_reserve: Option<u64>,
    pub chunk_bytes: usize,
    /// Whether a run of the whole model leaves out the MTP block: the draft head is then rounded under the output
    /// head's Hessian.
    pub skip_mtp: bool,
    /// The output head's rows the draft head keeps, in the order it stores them: [`Calibration::open`] checks them
    /// against the output head's rows, and a run needs at least one.
    pub draft_ids: Vec<i32>,
    /// The manifest the run starts from (a resumed directory's, or a new one): its key names the run in the resume
    /// file, and it is all the run resumes from.
    pub manifest: Manifest,
    /// What a trellis run is, when it is one: the layers are then rounded in the trellis format and written as an exl3
    /// checkpoint, and the GPTQ lanes only round the heads.
    pub trellis: Option<TrellisRun>,
}

impl Settings {
    /// The settings of a run over `tokens` (`length` per sequence), everything else at its default.
    pub fn new(out: PathBuf, recipe: Recipe, manifest: Manifest, tokens: Vec<i32>, length: usize) -> Settings {
        Settings {
            out,
            recipe,
            search: Nvfp4Search::DEFAULT,
            damp: DEFAULT_DAMP,
            act_order: false,
            batch: DEFAULT_BATCH,
            length,
            tokens,
            lanes: ROUNDING_LANES,
            staging_bytes: STAGING_BYTES,
            lane_pinned_bytes: None,
            device_reserve: None,
            chunk_bytes: UPLOAD_CHUNK_BYTES,
            skip_mtp: false,
            draft_ids: Vec::new(),
            manifest,
            trellis: None,
        }
    }

    /// The sequences the tokens make.
    pub fn sequences(&self) -> usize {
        self.tokens.len() / self.length.max(1)
    }
}

/// The hidden states in pinned host memory: the pipeline's input and output, and what the resume file holds.
struct States {
    buffer: PinnedBuffer,
    bytes: usize,
}

// SAFETY: the buffer is plain memory; the calls of a calibration touch it one after another (each starts in the frame
// that ends the one before), never two at once.
unsafe impl Send for States {}
// SAFETY: as for `Send`.
unsafe impl Sync for States {}

impl States {
    fn pointer(&self) -> *mut u8 {
        self.buffer.as_ptr().cast_mut()
    }
}

/// What the calls share.
struct Inner {
    settings: Settings,
    objects: Arc<Qwen38Objects>,
    library: Arc<CudaLibrary>,
    table: Arc<KernelTable>,
    kernels: Arc<ForwardKernels>,
    main: Arc<CudaQueue>,
    home: Marker,
    completions: Arc<Completions>,
    pipeline: ForwardPipeline,
    staging: Arc<PinnedPool>,
    /// The buffer the embedding read, the MTP block and the heads read their tensors into.
    head_staging: HeadStaging<PinnedBuffer>,
    states: States,
    batch_states: DeviceBuffer,
    /// The main lane's second set of math handles: the pipeline holds the first.
    handles: Handoff<LaneHandles>,
    lanes: Handoff<Vec<GptqLane>>,
    services: Arc<Services>,
    resume: Handoff<ResumeFile>,
    /// The manifest of the run, which each layer updates.
    ledger: Arc<Ledger>,
    hidden: usize,
    /// Tokens of a sequence in the pass that runs: the run's, and the MTP layer's one fewer.
    pass_length: Arc<AtomicUsize>,
    /// What the heads keep on the card between their calls.
    heads: Handoff<HeadState<DeviceBuffer>>,
    /// The trellis lanes and writer, when the run is a trellis run.
    trellis: Option<Trellis>,
}

/// The calibration on the card; its calls are [`Stages`].
pub struct Calibration {
    inner: Arc<Inner>,
}

impl Debug for Calibration {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Calibration").finish_non_exhaustive()
    }
}

/// The math handles of `stream`'s lane, made on it.
fn lane_handles(math: &Arc<MathLibraries>, stream: &mut CudaStream, what: &str) -> Result<LaneHandles, CalibrateError> {
    let mut handles = None;
    Stream::submit(stream, |_| {
        handles = Some(math.lane_handles());
        Ok(())
    })
    .map_err(|error| CalibrateError::failed(&format!("the {what} lane's handles"), error))?;
    handles
        .ok_or_else(|| CalibrateError::Failed(format!("the {what} lane made no handles")))?
        .map_err(|error| CalibrateError::failed(&format!("the {what} lane's handles"), format!("{error:?}")))
}

fn gpu_error(error: impl Display) -> GpuError {
    GpuError::Invalid(error.to_string())
}

/// The shapes of a checkpoint's objects that [`lane_groups`] reads.
trait ObjectShapes {
    /// The text layers.
    fn layers(&self) -> usize;
    /// The hidden width.
    fn hidden(&self) -> u64;
    /// The objects of text layer `layer`.
    fn layer_objects(&self, layer: usize) -> Result<Vec<String>, CalibrateError>;
    /// The rows and columns of object `name`.
    fn rows_columns(&self, name: &str) -> Result<(u64, u64), CalibrateError>;
}

impl ObjectShapes for Qwen38Objects {
    fn layers(&self) -> usize {
        self.config.full_attention.len()
    }

    fn hidden(&self) -> u64 {
        self.config.hidden
    }

    fn layer_objects(&self, layer: usize) -> Result<Vec<String>, CalibrateError> {
        Qwen38Objects::layer_objects(self, layer).map_err(|error| CalibrateError::failed("the layer's objects", error))
    }

    fn rows_columns(&self, name: &str) -> Result<(u64, u64), CalibrateError> {
        let source = self.object(name).map_err(|error| CalibrateError::failed(name, error))?;
        Ok((source.rows, source.columns))
    }
}

/// What the GPTQ lanes of a run round, one group after another, each with the format `recipe` gives it: unless the run
/// is a `trellis` one, each text layer's objects and, unless `skip_mtp`, the MTP layer's and its input projection
/// alone; in every run the output head alone, and the draft head alone, of `draft_rows` rows. Fails when the
/// checkpoint or the recipe does not know an object.
fn lane_groups(
    objects: &impl ObjectShapes,
    recipe: &Recipe,
    trellis: bool,
    skip_mtp: bool,
    draft_rows: u64,
) -> Result<Vec<Vec<ObjectShape>>, CalibrateError> {
    let shape = |name: &str, rows: u64, k: u64| {
        let format = recipe
            .format(name)
            .map_err(|error| CalibrateError::failed(name, error))?;
        Ok(ObjectShape {
            rows,
            k,
            format: rounded(format),
        })
    };
    let source = |name: &str| {
        let (rows, columns) = objects.rows_columns(name)?;
        shape(name, rows, columns)
    };
    let hidden = objects.hidden();
    let mut groups = Vec::new();
    if !trellis {
        for layer in 0..objects.layers() {
            let names = objects.layer_objects(layer)?;
            groups.push(names.iter().map(|name| source(name)).collect::<Result<_, _>>()?);
        }
        if !skip_mtp {
            let names = MTP_OBJECTS.iter().filter(|name| name.starts_with("mtp/layer/"));
            groups.push(names.map(|name| source(name)).collect::<Result<_, _>>()?);
            groups.push(vec![shape(INPUT_PROJECTION, hidden, 2 * hidden)?]);
        }
    }
    groups.push(vec![source(OUTPUT_HEAD)?]);
    groups.push(vec![shape(DRAFT_HEAD, draft_rows, hidden)?]);
    Ok(groups)
}

/// The pinned bytes the rounding lanes stage through, given the `needed` bytes of the GPTQ lanes' warm set: `set` when
/// given; else `needed`, or in a `trellis` run, whose trellis lanes stage through the same pool, at least
/// [`TRELLIS_PINNED_BYTES`]. Refuses a `set` budget below `needed`, naming both.
fn lane_pinned_budget(set: Option<u64>, needed: u64, trellis: bool) -> Result<u64, CalibrateError> {
    match set {
        Some(set) if set < needed => Err(CalibrateError::Invalid(format!(
            "the lanes' pinned budget of {set} bytes is less than the {needed} bytes of staging the GPTQ lanes hold at \
             once"
        ))),
        Some(set) => Ok(set),
        None if trellis => Ok(needed.max(TRELLIS_PINNED_BYTES)),
        None => Ok(needed),
    }
}

impl Calibration {
    /// Opens the device and everything a run uses on it. `objects` is the checkpoint, already opened. The returned
    /// completions are the sources of the frames that continue the chains: the caller adds them to its lattice
    /// before it drives the run.
    pub fn open(
        settings: Settings,
        objects: Arc<Qwen38Objects>,
        library: &Path,
        math_directory: &Path,
        io: Arc<AsyncIo>,
        host: Arc<Lake>,
    ) -> Result<(Calibration, Vec<Arc<Completions>>), CalibrateError> {
        let head_rows = objects
            .object(OUTPUT_HEAD)
            .map_err(|error| CalibrateError::failed(OUTPUT_HEAD, error))?
            .rows;
        check_draft_ids(&settings.draft_ids, head_rows).map_err(CalibrateError::Invalid)?;
        // A trellis run's GPTQ lanes round the heads only, one at a time.
        let gptq_lanes = if settings.trellis.is_some() {
            1
        } else {
            settings.lanes.max(1)
        };
        let slice_rows = HEAD_ROWS;
        let groups = lane_groups(
            objects.as_ref(),
            &settings.recipe,
            settings.trellis.is_some(),
            settings.skip_mtp,
            settings.draft_ids.len() as u64,
        )?;
        let warm = lane_staging(&groups, gptq_lanes, slice_rows as u64)
            .map_err(|error| CalibrateError::failed("the staging the GPTQ lanes hold", error))?;
        let lane_pinned_bytes =
            lane_pinned_budget(settings.lane_pinned_bytes, warm.bytes(), settings.trellis.is_some())?;
        let library = CudaLibrary::load(library).map_err(|error| CalibrateError::failed("the CUDA library", error))?;
        let math =
            MathLibraries::load(math_directory).map_err(|error| CalibrateError::failed("the math libraries", error))?;
        let gpu = CudaGpu::open(0).map_err(|error| CalibrateError::failed("the device", error))?;
        let sources =
            KernelSources::beside(&library).map_err(|error| CalibrateError::failed("the kernel sources", error))?;
        let table = Arc::new(load(&gpu, &sources).map_err(|error| CalibrateError::failed("the kernels", error))?);
        let options = ModuleOptions::beside(&library).map_err(|error| CalibrateError::failed("the compiler", error))?;
        let mut modules =
            Modules::<CudaModule>::open(&gpu, options).map_err(|error| CalibrateError::failed("the modules", error))?;
        let gptq_kernels =
            GptqKernels::load(&mut modules).map_err(|error| CalibrateError::failed("the GPTQ kernels", error))?;
        let gptq = modules
            .load(Module::Gptq)
            .map_err(|error| CalibrateError::failed("the GPTQ module", error))?;
        let rows = modules
            .load(Module::Rows)
            .map_err(|error| CalibrateError::failed("the rows module", error))?;
        let kernels = Arc::new(
            ForwardKernels::new(&gptq, &rows).map_err(|error| CalibrateError::failed("the forward kernels", error))?,
        );
        let config = &objects.config;
        let ledger = Arc::new(Ledger::new(settings.manifest.clone()));
        let (hidden, length, batch) = (config.hidden as usize, settings.length, settings.batch);
        let max_rows = u32::try_from(batch * length)
            .map_err(|_| CalibrateError::Invalid(format!("{batch} sequences of {length} tokens are too many rows")))?;

        let names: Vec<String> = once("main".to_owned())
            .chain((0..batch).map(|slot| format!("slot{slot}")))
            .collect();
        let named: Vec<(&str, StreamOptions)> = names
            .iter()
            .map(|name| (name.as_str(), StreamOptions::default()))
            .collect();
        let shared =
            open_shared_lanes(&gpu, &named).map_err(|error| CalibrateError::failed("the pipeline lanes", error))?;
        let main = Arc::clone(shared.pool.lane(LaneId(0)));
        let slots: Vec<Arc<CudaQueue>> = (1..=batch)
            .map(|slot| Arc::clone(shared.pool.lane(LaneId(slot as u16))))
            .collect();
        let completions = Completions::new(&shared.queues);
        let staging = PinnedPool::new(&library, settings.staging_bytes);
        let home = Marker::new(DeviceQueue::id(&*main), Lane::lane(&*main), 0);

        let made = main.submit_marked(|queue| {
            let mut ops = RusteeOps::new(queue, &table, Modes::default()).on_lane(&library, home);
            let forward = LayerForward::new(&mut ops, &library, &staging, config, max_rows).map_err(gpu_error)?;
            let pipeline_handles = math.lane_handles().map_err(|error| gpu_error(format!("{error:?}")))?;
            let symmetric = math.lane_handles().map_err(|error| gpu_error(format!("{error:?}")))?;
            Ok((forward, pipeline_handles, symmetric))
        });
        let ((forward, pipeline_handles, symmetric), _) =
            made.map_err(|error| CalibrateError::failed("the pipeline workspace", error))?;
        let pipeline = ForwardPipeline::new(Parts {
            forward,
            library: Arc::clone(&library),
            table: Arc::clone(&table),
            kernels: Arc::clone(&kernels),
            handles: pipeline_handles,
            main: Arc::clone(&main),
            slots,
            completions: Arc::clone(&completions),
            lake: Arc::clone(&host),
        })
        .map_err(|error| CalibrateError::failed("the pipeline", error))?;

        let states_bytes = settings.tokens.len() * hidden * BF16_BYTES;
        let states = States {
            buffer: PinnedBuffer::alloc(&library, states_bytes)
                .map_err(|error| CalibrateError::failed("the hidden states", error))?,
            bytes: states_bytes,
        };
        let head_staging = HeadStaging::open(
            &library,
            &head_groups(
                settings.trellis.is_some(),
                |call| call.names(&objects),
                |name| objects.store().tensor(name).ok().map(|tensor| tensor.bytes as usize),
            ),
        )?;
        let batch_states = DeviceBuffer::alloc(&library, batch * length * hidden * BF16_BYTES)
            .map_err(|error| CalibrateError::failed("the batch states", error))?;

        let lane_names: Vec<String> = (0..gptq_lanes).map(|lane| format!("gptq{lane}")).collect();
        let lane_refs: Vec<&str> = lane_names.iter().map(String::as_str).collect();
        let (pool, streams) = gpu
            .open_lanes(&lane_refs, StreamOptions::default())
            .map_err(|error| CalibrateError::failed("the GPTQ lanes", error))?;
        let gptq_completions = Completions::new(&pool);
        let lane_pinned = PinnedPool::new(&library, lane_pinned_bytes);
        let gptq_settings = GptqSettings {
            search: settings.search,
            ..GptqSettings::default()
        };
        let mut lanes = Vec::new();
        for mut stream in streams {
            let handles = lane_handles(&math, &mut stream, "GPTQ")?;
            let memory = LaneMemory::new(Arc::clone(&library), Arc::clone(&lane_pinned));
            let lane = GptqLane::new(
                stream,
                handles,
                Arc::clone(&gptq_kernels),
                memory,
                Arc::clone(&gptq_completions),
                gptq_settings,
            )
            .map_err(|error| CalibrateError::failed("a GPTQ lane", error))?;
            lanes.push(lane);
        }

        let free = library
            .free_bytes()
            .map_err(|error| CalibrateError::failed("the free device memory", error))?;
        let reserve = settings
            .device_reserve
            .unwrap_or_else(|| (free / 3).min(DEVICE_RESERVE));
        let capacity = free.saturating_sub(reserve);
        if capacity == 0 {
            return Err(CalibrateError::Failed(format!(
                "the device has {free} bytes free, no more than the {reserve} the run leaves for the layer"
            )));
        }
        let planner = Planner::new([
            LocationSpec {
                kind: LocationKind::Device { ordinal: 0 },
                capacity,
                reserve: 0,
            },
            LocationSpec {
                kind: LocationKind::HostPinned,
                capacity: lane_pinned_bytes,
                reserve: 0,
            },
        ]);
        let owner = planner.open_owner(OwnerSpec::new("calibrate", Priority(0), Scope::Process));
        // A trellis run's heads write their states beside, not among, the checkpoint's files.
        let heads_out = match settings.trellis {
            Some(_) => settings.out.join(HEADS_DIRECTORY),
            None => settings.out.clone(),
        };
        let trellis = match &settings.trellis {
            Some(run) => Some(trellis::open(
                Opening {
                    gpu: &gpu,
                    math: &math,
                    modules: &mut modules,
                    library: &library,
                    objects: &objects,
                    lane_pinned: &lane_pinned,
                    pinned: lane_pinned_bytes,
                    capacity,
                    settings: &settings,
                    io: &io,
                    host: &host,
                },
                run,
            )?),
            None => None,
        };
        let services = Arc::new(Services {
            planner,
            owner,
            device: LocationId(0),
            pinned: LocationId(1),
            settings: RoundSettings {
                recipe: settings.recipe.clone(),
                damp: settings.damp,
                act_order: settings.act_order,
                slice_rows,
            },
            out: heads_out,
            io,
            host,
        });
        prewarm(&services, &lanes, &warm)?;
        // The device and its lane pools stay open for the lanes' life: the queues hold them weakly.
        forget((gpu, shared, pool));
        let (trellis, trellis_completions) = match trellis {
            Some((trellis, completions)) => (Some(trellis), Some(completions)),
            None => (None, None),
        };
        let inner = Arc::new(Inner {
            settings,
            objects,
            library,
            table,
            kernels,
            main,
            home,
            completions: Arc::clone(&completions),
            pipeline,
            staging,
            head_staging,
            states,
            batch_states,
            handles: Handoff::empty(),
            lanes: Handoff::empty(),
            services,
            resume: Handoff::empty(),
            ledger,
            hidden,
            pass_length: Arc::new(AtomicUsize::new(length)),
            heads: Handoff::empty(),
            trellis,
        });
        inner.heads.send(HeadState::default());
        inner.handles.send(symmetric);
        inner.lanes.send(lanes);
        let mut sources = vec![completions, gptq_completions];
        sources.extend(trellis_completions);
        Ok((Calibration { inner }, sources))
    }
}

impl Inner {
    fn host_states(&self) -> HostStates {
        // SAFETY: the states are live as long as the pipeline, and no call but the one running touches them.
        unsafe { HostStates::new(self.states.pointer(), self.states.bytes) }
    }

    /// The hidden states as the resume file's region.
    fn region(&self) -> Region {
        // SAFETY: the states are the run's own, and the calls that read or write them run one after another.
        unsafe { Region::new(self.states.pointer(), self.states.bytes) }
    }

    fn row_bytes(&self) -> usize {
        self.hidden * BF16_BYTES
    }

    fn request(&self, layer: Arc<LayerWeights>, observer: Option<PipelineObserver>, write: bool) -> Request {
        Request {
            layer,
            states: self.host_states(),
            batch_states: self.batch_states.ptr(),
            sequences: self.settings.sequences() as u32,
            length: self.pass_length.load(Ordering::Acquire) as u32,
            batch: self.settings.batch as u32,
            observer,
            write,
        }
    }

    /// A handle on the resume file.
    fn resume_file(&self) -> Result<ResumeFile, CalibrateError> {
        let file = self
            .resume
            .take()
            .ok_or_else(|| CalibrateError::Failed("the resume file is not open".to_owned()))?;
        self.resume.send(file.clone());
        Ok(file)
    }
}

/// The Hessians the observer has made, which go back to the layer's chain when the observer is dropped: the pipeline
/// drops it after the pass, before it calls its `done`.
struct Observed {
    set: Option<HessianSet<DeviceBuffer>>,
    back: Arc<Handoff<HessianSet<DeviceBuffer>>>,
}

impl Drop for Observed {
    fn drop(&mut self) {
        if let Some(set) = self.set.take() {
            self.back.send(set);
        }
    }
}

fn observer(mut observed: Observed) -> PipelineObserver {
    Box::new(move |observation: &Observation<'_>, context| {
        let Some(set) = observed.set.as_mut() else {
            return Err(ForwardError::Invalid("the observer has no Hessians".to_owned()));
        };
        let Observation {
            object,
            input,
            rows,
            columns,
        } = *observation;
        let context = RefCell::new(context);
        set.observe(
            object,
            (input.get(), rows, columns),
            |columns| {
                let bytes = FP32_BYTES * columns as usize * columns as usize;
                let mut context = context.borrow_mut();
                let buffer = context
                    .ops()
                    .alloc_async(bytes)
                    .map_err(|error| CalibrateError::failed("a Hessian", error))?;
                context
                    .ops()
                    .zero_device_memory(buffer.ptr(), bytes as u64)
                    .map_err(|error| CalibrateError::failed("a Hessian", error))?;
                Ok(buffer)
            },
            |hessian| {
                context
                    .borrow_mut()
                    .accumulate_hessian(hessian.ptr(), input, rows, columns)
                    .map_err(|error| CalibrateError::failed(object, error))
            },
        )
        .map_err(|error| ForwardError::Invalid(error.to_string()))
    })
}

impl Stages for Calibration {
    type Staged = StagedLayer;

    fn begin(&self, plan: &Plan, then: Then<usize>) {
        begin(Arc::clone(&self.inner), plan.layers, single(then));
    }

    fn stage(&self, layer: usize, then: Then<StagedLayer>) {
        stage_target(&self.inner, Target::Layer(layer), then);
    }

    /// The layer, then the manifest with the layer counted: its files are all written by then.
    fn layer(&self, layer: usize, staged: StagedLayer, then: Then<()>) {
        let inner = Arc::clone(&self.inner);
        let recorded: Then<()> = Box::new(move |result| {
            Arc::clone(&inner.ledger).record(layer, result, save_manifest(&inner), then);
        });
        upload(Arc::clone(&self.inner), Target::Layer(layer), staged, single(recorded));
    }

    fn checkpoint(&self, done: usize, then: Then<()>) {
        checkpoint(&self.inner, done, then);
    }

    /// The heads over the states the layers left: see [`heads`].
    fn finish(&self, then: Then<()>) {
        if self.inner.trellis.is_some() {
            run_trellis(Arc::new(TrellisDeviceHeads::new(Arc::clone(&self.inner))), then);
        } else {
            run_heads(
                Arc::new(DeviceHeads::new(Arc::clone(&self.inner))),
                self.inner.settings.skip_mtp,
                then,
            );
        }
    }
}

/// Reads `target` from the checkpoint into pinned host memory.
fn stage_target(inner: &Arc<Inner>, target: Target, then: Then<StagedLayer>) {
    let cell = single(then);
    let ready = Arc::clone(&cell);
    let staging = Arc::clone(&inner.staging);
    let started = stage(
        &inner.objects,
        target,
        move |bytes| staging.acquire(bytes),
        Arc::new(move |staged: Result<StagedLayer<PooledPinned>, CheckpointError>| {
            end(
                &ready,
                staged.map_err(|error| CalibrateError::failed("reading the layer", error)),
            );
        }),
    );
    if let Err(error) = started {
        end(&cell, Err(CalibrateError::failed("reading the layer", error)));
    }
}

/// What `target` is called in a failure.
fn describe(target: Target) -> String {
    match target {
        Target::Layer(layer) => format!("layer {layer}"),
        Target::Mtp => "the MTP layer".to_owned(),
    }
}

// ---- begin: resume or embed ----

fn begin(inner: Arc<Inner>, layers: usize, then: Once<usize>) {
    let path = inner.settings.out.join(RESUME_FILE);
    let (io, host) = (Arc::clone(&inner.services.io), Arc::clone(&inner.services.host));
    let failed = Arc::clone(&then);
    let opening = Arc::clone(&inner);
    let opened = ResumeFile::open(
        &io,
        &host,
        &path,
        Box::new(move |opened| {
            match opened {
                Err(error) => end(&then, Err(CalibrateError::failed("the resume file", error))),
                Ok(file) => {
                    opening.resume.send(file.clone());
                    resume_or_embed(opening, &file, layers, then);
                }
            }
            idle()
        }),
    );
    if let Err(error) = opened {
        end(&failed, Err(CalibrateError::failed("the resume file", error)));
    }
}

/// Resumes from what the manifest lists, or embeds the tokens when that is layer 0.
fn resume_or_embed(inner: Arc<Inner>, file: &ResumeFile, layers: usize, then: Once<usize>) {
    let Some(manifest) = inner.ledger.manifest() else {
        return end(&then, Err(CalibrateError::Failed("the manifest is in use".to_owned())));
    };
    let region = inner.region();
    resume(
        file,
        manifest,
        layers,
        region,
        STATES_CHUNK,
        Box::new(move |at| match at {
            Ok(0) => embed(inner, then),
            done => end(&then, done),
        }),
    );
}

/// Looks the tokens up in the embedding table: each token's row is copied into the states.
fn embed(inner: Arc<Inner>, then: Once<usize>) {
    let store = Arc::clone(inner.objects.store());
    let tensor = match store.tensor(EMBEDDING) {
        Ok(tensor) => tensor.clone(),
        Err(error) => return end(&then, Err(CalibrateError::failed("the embedding table", error))),
    };
    let hidden = inner.hidden as u64;
    let checked = embedding::vocabulary(&tensor.dtype, &tensor.shape, hidden)
        .and_then(|vocabulary| embedding::check_tokens(&inner.settings.tokens, vocabulary));
    if let Err(error) = checked {
        return end(&then, Err(error));
    }
    // The heads' own buffer, allocated at open: a pooled loan of this size would be rounded up to a power of two and
    // allocated in this frame.
    let names = match Call::Embedding.names(&inner.objects) {
        Ok(names) => names,
        Err(error) => return end(&then, Err(error)),
    };
    let reading = Arc::clone(&inner);
    read_tensors(
        &reading,
        names,
        Box::new(move |read| {
            let outcome = read.and_then(|memory| {
                let table = memory
                    .bytes(0)
                    .ok_or_else(|| CalibrateError::Failed("the embedding table was not read".to_owned()))?;
                // SAFETY: the states are the run's own and nothing touches them before the run starts.
                let states = unsafe { slice::from_raw_parts_mut(inner.states.pointer(), inner.states.bytes) };
                embedding::copy_rows(&inner.settings.tokens, table, inner.row_bytes(), states)
            });
            end(&then, outcome.map(|()| 0));
        }),
    );
}

// ---- checkpoint ----

/// The states after `done` layers into the resume file, then the manifest listing it.
fn checkpoint(inner: &Arc<Inner>, done: usize, then: Then<()>) {
    let file = match inner.resume_file() {
        Ok(file) => file,
        Err(error) => return then(Err(error)),
    };
    let Some(manifest) = inner.ledger.manifest() else {
        return then(Err(CalibrateError::Failed("the manifest is in use".to_owned())));
    };
    let listing = Arc::clone(inner);
    let listed: Then<()> = Box::new(move |result| {
        Arc::clone(&listing.ledger).checkpointed(result, save_manifest(&listing), then);
    });
    write_checkpoint(&file, &manifest.key, inner.region(), STATES_CHUNK, done, listed);
}

/// The save of the manifest's bytes over its file, for the ledger.
fn save_manifest(inner: &Arc<Inner>) -> impl FnOnce(Vec<u8>, Then<()>) + use<> {
    let (io, host) = (Arc::clone(&inner.services.io), Arc::clone(&inner.services.host));
    let path = inner.settings.out.join(MANIFEST_FILE);
    move |bytes, saved| {
        save(
            &io,
            &host,
            path,
            bytes,
            Box::new(move |result| {
                saved(result);
                idle()
            }),
        );
    }
}

// ---- a layer ----

/// Registers `work` to run in the frame that follows `marker` on the pipeline's lanes.
fn after(inner: &Arc<Inner>, marker: Marker, work: impl FnOnce() + Send + 'static) {
    inner.completions.after_marker(marker, step(Kind::Resolve, work));
}

/// The chain of one layer's calls for `target`, from its read to the end of its write pass; `then` gets its end. The
/// manifest is the caller's to update.
fn run_target(inner: &Arc<Inner>, target: Target, then: Then<()>) {
    let cell = single(then);
    let failed = Arc::clone(&cell);
    let next = Arc::clone(inner);
    stage_target(
        inner,
        target,
        Box::new(move |staged| match staged {
            Ok(staged) => upload(next, target, staged, cell),
            Err(error) => end(&failed, Err(error)),
        }),
    );
}

fn upload(inner: Arc<Inner>, target: Target, staged: StagedLayer, then: Once<()>) {
    let submitted = inner.main.submit_marked(|queue| {
        let mut ops = RusteeOps::new(queue, &inner.table, Modes::default()).on_lane(&inner.library, inner.home);
        // SAFETY: `staged` is kept until the marker of this submission is reached, which is after the copies.
        unsafe { LayerWeights::upload(&mut ops, &staged, inner.settings.chunk_bytes) }.map_err(gpu_error)
    });
    match submitted {
        Err(error) => {
            // The copies queued before the failure may still read the pinned memory: it is never returned.
            forget(staged);
            end(
                &then,
                Err(CalibrateError::failed(
                    &format!("uploading {}", describe(target)),
                    error,
                )),
            );
        }
        Ok((weights, marker)) => {
            let next = Arc::clone(&inner);
            after(&inner, marker, move || {
                drop(staged);
                observe(next, target, weights, then);
            });
        }
    }
}

/// The observe pass: the layer runs over every sequence and the Hessians fill; its outputs are discarded.
fn observe(inner: Arc<Inner>, target: Target, weights: LayerWeights, then: Once<()>) {
    let layer = Arc::new(weights);
    let back = Arc::new(Handoff::empty());
    let seen = Observed {
        set: Some(HessianSet::new()),
        back: Arc::clone(&back),
    };
    let request = inner.request(Arc::clone(&layer), Some(observer(seen)), false);
    let (next, ended) = (Arc::clone(&inner), Arc::clone(&then));
    let started = inner.pipeline.run(
        request,
        Box::new(move |result| {
            let outcome = result
                .map_err(|error| CalibrateError::failed("the observe pass", error))
                .and_then(|()| {
                    let weights = Arc::try_unwrap(layer)
                        .map_err(|_| CalibrateError::Failed("the layer is still shared after its pass".to_owned()))?;
                    let set = back
                        .take()
                        .ok_or_else(|| CalibrateError::Failed("the pass left no Hessians".to_owned()))?;
                    Ok((weights, set))
                });
            match outcome {
                Ok((weights, set)) => complete(next, target, weights, set, ended),
                Err(error) => end(&ended, Err(error)),
            }
        }),
    );
    if let Err(error) = started {
        end(&then, Err(CalibrateError::failed("the observe pass", error)));
    }
}

/// Completes the Hessians and makes each linear's decoded weight, on the main lane, then rounds.
fn complete(inner: Arc<Inner>, target: Target, weights: LayerWeights, set: HessianSet<DeviceBuffer>, then: Once<()>) {
    if let (Some(_), Target::Layer(index)) = (&inner.trellis, target) {
        return round_layer(inner, index, weights, set, then);
    }
    let shapes = weights.shapes().to_vec();
    let order = round_order(&shapes);
    let submitted = inner.main.submit_marked(|queue| {
        let ops = RusteeOps::new(queue, &inner.table, Modes::default()).on_lane(&inner.library, inner.home);
        let mut handles = inner
            .handles
            .take()
            .ok_or_else(|| gpu_error("the main lane's handles are in use"))?;
        let made = (|| -> Result<Vec<DeviceBuffer>, ForwardError> {
            let mut context = LaneContext::new(ops, &mut handles, &inner.kernels);
            for matrix in set.matrices() {
                let k = (matrix.bytes() / FP32_BYTES).isqrt();
                context.symmetrize(matrix.ptr(), k as u32)?;
            }
            shapes
                .iter()
                .map(|(_, (rows, columns))| {
                    Ok(context
                        .ops()
                        .alloc_async(BF16_BYTES * (*rows as usize) * (*columns as usize))?)
                })
                .collect()
        })();
        inner.handles.send(handles);
        made.map_err(gpu_error)
    });
    let (decoded, marker) = match submitted {
        Ok(made) => made,
        Err(error) => return end(&then, Err(CalibrateError::failed("completing the Hessians", error))),
    };
    let next = Arc::clone(&inner);
    after(&inner, marker, move || round(next, weights, set, decoded, order, then));
}

fn round(
    inner: Arc<Inner>,
    weights: LayerWeights,
    set: HessianSet<DeviceBuffer>,
    decoded: Vec<DeviceBuffer>,
    order: Vec<usize>,
    then: Once<()>,
) {
    let built: Result<Vec<Linear>, CalibrateError> = weights
        .shapes()
        .iter()
        .zip(decoded)
        .map(|((name, (rows, columns)), buffer)| {
            let weight = weights
                .object(name)
                .ok_or_else(|| CalibrateError::Failed(format!("the layer has no weight {name}")))?;
            let hessian = set
                .matrix(name)
                .ok_or_else(|| CalibrateError::Failed(format!("{name} was not observed")))?;
            let held = Handoff::empty();
            held.send(buffer);
            Ok(Linear {
                name: name.clone(),
                rows: *rows,
                columns: *columns as usize,
                weight: weight.get(),
                hessian: hessian.addr(),
                decoded: held,
            })
        })
        .collect();
    let (linears, lanes) = match (built, inner.lanes.take()) {
        (Ok(linears), Some(lanes)) => (linears, lanes),
        (Err(error), _) => return end(&then, Err(error)),
        (_, None) => {
            return end(
                &then,
                Err(CalibrateError::Failed("the GPTQ lanes are in use".to_owned())),
            );
        }
    };
    let next = Arc::clone(&inner);
    // SAFETY: the weights and the Hessians move into the continuation and stay until the rounding has ended.
    unsafe {
        round_all(
            Arc::clone(&inner.services),
            lanes,
            linears,
            order,
            Box::new(move |ended| match ended {
                Ok(ended) => swap(next, weights, set, ended, then),
                Err(error) => end(&then, Err(error)),
            }),
        );
    }
}

/// Swaps the decoded weights in and frees the Hessians, on the main lane, then runs the write pass.
fn swap(inner: Arc<Inner>, mut weights: LayerWeights, set: HessianSet<DeviceBuffer>, ended: Ended, then: Once<()>) {
    let Ended { lanes, linears } = ended;
    inner.lanes.send(lanes);
    if !inner
        .ledger
        .list(linears.iter().map(|linear| QuantState::file_name(&linear.name)))
    {
        return end(
            &then,
            Err(CalibrateError::Failed(
                "the manifest was in use when the layer's files were listed".to_owned(),
            )),
        );
    }
    let submitted = inner.main.submit_marked(|queue| {
        let mut ops = RusteeOps::new(queue, &inner.table, Modes::default()).on_lane(&inner.library, inner.home);
        for linear in linears {
            let decoded = linear
                .decoded
                .take()
                .ok_or_else(|| gpu_error(format!("{} was not rounded", linear.name)))?;
            weights.replace(&mut ops, &linear.name, decoded).map_err(gpu_error)?;
        }
        for matrix in set.into_matrices() {
            ops.free_async(matrix).map_err(gpu_error)?;
        }
        Ok(())
    });
    match submitted {
        Err(error) => end(
            &then,
            Err(CalibrateError::failed("swapping in the rounded weights", error)),
        ),
        Ok(((), marker)) => {
            let next = Arc::clone(&inner);
            after(&inner, marker, move || write_pass(next, weights, then));
        }
    }
}

/// The pass over the rounded layer: its outputs replace the states.
fn write_pass(inner: Arc<Inner>, weights: LayerWeights, then: Once<()>) {
    let layer = Arc::new(weights);
    let request = inner.request(Arc::clone(&layer), None, true);
    let (next, ended) = (Arc::clone(&inner), Arc::clone(&then));
    let started = inner.pipeline.run(
        request,
        Box::new(move |result| {
            let outcome = result
                .map_err(|error| CalibrateError::failed("the write pass", error))
                .and_then(|()| {
                    Arc::try_unwrap(layer)
                        .map_err(|_| CalibrateError::Failed("the layer is still shared after its pass".to_owned()))
                });
            match outcome {
                Ok(weights) => release(next, weights, ended),
                Err(error) => end(&ended, Err(error)),
            }
        }),
    );
    if let Err(error) = started {
        end(&then, Err(CalibrateError::failed("the write pass", error)));
    }
}

fn release(inner: Arc<Inner>, weights: LayerWeights, then: Once<()>) {
    let submitted = inner.main.submit_marked(|queue| {
        let mut ops = RusteeOps::new(queue, &inner.table, Modes::default()).on_lane(&inner.library, inner.home);
        weights.release(&mut ops).map_err(gpu_error)
    });
    match submitted {
        Err(error) => end(&then, Err(CalibrateError::failed("freeing the layer", error))),
        Ok(((), marker)) => after(&inner, marker, move || end(&then, Ok(()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    use crate::recipe::Quantization;

    /// A checkpoint of two text layers, the second of full attention, where each object has its own shape.
    struct Checkpoint {
        shapes: HashMap<String, (u64, u64)>,
    }

    const HIDDEN: u64 = 512;

    fn layer_names(layer: usize) -> Vec<String> {
        let roles = if layer == 1 {
            [
                "attention/query_key",
                "attention/gate_value",
                "attention/output",
                "mlp/gate_up",
                "mlp/down",
            ]
        } else {
            ["gdn/query_key", "gdn/value_z", "gdn/output", "mlp/gate_up", "mlp/down"]
        };
        roles.iter().map(|role| format!("text/layers/{layer}/{role}")).collect()
    }

    fn checkpoint() -> Checkpoint {
        let mut names = layer_names(0);
        names.extend(layer_names(1));
        names.extend(MTP_OBJECTS.iter().map(|name| name.to_string()));
        names.push(OUTPUT_HEAD.to_string());
        let shapes = names
            .into_iter()
            .zip(1..)
            .map(|(name, index)| (name, (64 * index, 128 * index)))
            .collect();
        Checkpoint { shapes }
    }

    impl ObjectShapes for Checkpoint {
        fn layers(&self) -> usize {
            2
        }

        fn hidden(&self) -> u64 {
            HIDDEN
        }

        fn layer_objects(&self, layer: usize) -> Result<Vec<String>, CalibrateError> {
            Ok(layer_names(layer))
        }

        fn rows_columns(&self, name: &str) -> Result<(u64, u64), CalibrateError> {
            self.shapes
                .get(name)
                .copied()
                .ok_or_else(|| CalibrateError::Invalid(format!("unknown object {name}")))
        }
    }

    /// The shapes of groups of (name, rows, columns), each in the format `recipe` gives its name.
    fn expected(recipe: &Recipe, groups: &[Vec<(String, u64, u64)>]) -> Vec<Vec<ObjectShape>> {
        let shape = |(name, rows, k): &(String, u64, u64)| ObjectShape {
            rows: *rows,
            k: *k,
            format: rounded(recipe.format(name).unwrap()),
        };
        groups.iter().map(|group| group.iter().map(shape).collect()).collect()
    }

    /// The group of `names`, in their checkpoint shapes.
    fn group(checkpoint: &Checkpoint, names: &[String]) -> Vec<(String, u64, u64)> {
        names
            .iter()
            .map(|name| {
                let (rows, k) = checkpoint.shapes[name];
                (name.clone(), rows, k)
            })
            .collect()
    }

    #[test]
    fn a_runs_lanes_round_its_layers_mtp_block_and_heads_in_their_shapes() {
        let checkpoint = checkpoint();
        let layers = [group(&checkpoint, &layer_names(0)), group(&checkpoint, &layer_names(1))];
        let mtp_layer = [
            "mtp/layer/attention/query_key_gate_value",
            "mtp/layer/attention/output",
            "mtp/layer/mlp/gate_up",
            "mtp/layer/mlp/down",
        ]
        .map(String::from);
        let mtp = [
            group(&checkpoint, &mtp_layer),
            vec![(INPUT_PROJECTION.to_string(), HIDDEN, 2 * HIDDEN)],
        ];
        let heads = [
            group(&checkpoint, &[OUTPUT_HEAD.to_string()]),
            vec![(DRAFT_HEAD.to_string(), 7, HIDDEN)],
        ];

        let recipe = Recipe::new(Quantization::Nvfp4, false, None).unwrap();
        let whole = [&layers[..], &mtp[..], &heads[..]].concat();
        assert_eq!(
            lane_groups(&checkpoint, &recipe, false, false, 7).unwrap(),
            expected(&recipe, &whole)
        );
        let no_mtp = [&layers[..], &heads[..]].concat();
        assert_eq!(
            lane_groups(&checkpoint, &recipe, false, true, 7).unwrap(),
            expected(&recipe, &no_mtp)
        );
        let trellis = Recipe::new(Quantization::Trellis, false, None).unwrap();
        assert_eq!(
            lane_groups(&checkpoint, &trellis, true, false, 7).unwrap(),
            expected(&trellis, &heads)
        );
    }

    #[test]
    fn an_object_the_checkpoint_does_not_know_fails_the_groups() {
        let mut checkpoint = checkpoint();
        checkpoint.shapes.remove("text/layers/1/mlp/down");
        let recipe = Recipe::new(Quantization::Nvfp4, false, None).unwrap();
        assert!(lane_groups(&checkpoint, &recipe, false, false, 7).is_err());
        assert!(lane_groups(&checkpoint, &recipe, true, false, 7).is_ok());
    }

    #[test]
    fn an_unset_pinned_budget_is_the_warm_set_and_a_trellis_run_keeps_its_floor() {
        assert!(matches!(lane_pinned_budget(None, 5 << 20, false), Ok(bytes) if bytes == 5 << 20));
        assert!(matches!(
            lane_pinned_budget(None, 5 << 20, true),
            Ok(TRELLIS_PINNED_BYTES)
        ));
        let large = TRELLIS_PINNED_BYTES + 1;
        assert!(matches!(lane_pinned_budget(None, large, true), Ok(bytes) if bytes == large));
    }

    #[test]
    fn a_set_pinned_budget_below_the_warm_set_is_refused_naming_both() {
        assert!(matches!(lane_pinned_budget(Some(8192), 8192, false), Ok(8192)));
        assert!(matches!(lane_pinned_budget(Some(1 << 30), 8192, true), Ok(bytes) if bytes == 1 << 30));
        match lane_pinned_budget(Some(8191), 8192, false) {
            Err(CalibrateError::Invalid(text)) => assert!(text.contains("8191") && text.contains("8192"), "{text}"),
            other => panic!("not refused: {other:?}"),
        }
    }
}
