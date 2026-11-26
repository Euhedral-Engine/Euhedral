//! The calibration's calls on the card: the layer pipeline over the hidden states, the GPTQ lanes, and the resume file.
//!
//! One layer is a chain of continuations, each started by the frame that ends the one before it:
//! 1. upload the staged layer on the main lane;
//! 2. the observe pass: the pipeline runs the layer with an observer that adds each distinct input's products into a
//!    Hessian, and the outputs are discarded;
//! 3. symmetrize the Hessians and allocate each linear's decoded weight, on the main lane;
//! 4. round every linear on the GPTQ lanes ([`super::rounding`]), which write their states;
//! 5. swap the decoded weights in, free the Hessians, on the main lane;
//! 6. the write pass: the pipeline runs the rounded layer and its outputs replace the states;
//! 7. free the layer.
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
use std::sync::{Arc, OnceLock};

use euhedral_gpu::cuda::{CudaGpu, CudaModule, CudaQueue};
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

use super::driver::Then;
use super::hessians::HessianSet;
use super::plan::{Plan, resume_at, resume_header, round_order};
use super::rounding::{Ended, HEAD_ROWS, Linear, RoundSettings, Services, round_all};
use super::{CalibrateError, Stages};
use crate::checkpoint::CheckpointError;
use crate::checkpoint::objects::Qwen38Objects;
use crate::forward::pipeline::device::{ForwardPipeline, HostStates, Parts, PipelineObserver, Request};
use crate::forward::{ForwardError, ForwardKernels, LaneContext, LayerForward, Observation};
use crate::forward::{LayerWeights, StagedLayer, Target, stage};
use crate::gptq::format::Nvfp4Search;
use crate::gptq::{GptqKernels, GptqLane, GptqSettings};
use crate::kernels::{Module, ModuleOptions, Modules};
use crate::memory::LaneMemory;
use crate::recipe::Recipe;
use crate::resume::{HEADER, ResumeError, ResumeFile};
use crate::step::{Kind, idle, step};

/// The checkpoint tensor the tokens are looked up in.
const EMBEDDING: &str = "model.language_model.embed_tokens.weight";
/// The resume file's name in the output directory.
pub const RESUME_FILE: &str = "hidden.bin";
const BF16_BYTES: usize = 2;
/// Bytes of the hidden states one read or write of the resume file moves.
const STATES_CHUNK: usize = 256 << 20;
/// FP32 bytes of a Hessian element.
const FP32_BYTES: usize = 4;

/// Java's default dampening.
pub const DEFAULT_DAMP: f32 = 0.01;
/// Java's default sequences at once.
pub const DEFAULT_BATCH: usize = 8;
/// Rounding lanes: Java's `ROUNDING_FRAMES`.
pub const ROUNDING_LANES: usize = 3;
/// Device bytes the planner leaves for what it does not account: the layer, its Hessians, its decoded weights and the
/// libraries' workspaces.
pub const DEVICE_RESERVE: u64 = 8 << 30;
/// Pinned bytes the layers are read into.
pub const STAGING_BYTES: u64 = 4 << 30;
/// Pinned bytes the GPTQ lanes stage through.
pub const LANE_PINNED_BYTES: u64 = 1 << 30;
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
    /// What names this run in the resume file.
    pub key: String,
    /// Linears rounded at once.
    pub lanes: usize,
    pub staging_bytes: u64,
    pub lane_pinned_bytes: u64,
    pub device_reserve: u64,
    pub chunk_bytes: usize,
}

impl Settings {
    /// The settings of a run over `tokens` (`length` per sequence), everything else at its default.
    pub fn new(out: PathBuf, recipe: Recipe, key: String, tokens: Vec<i32>, length: usize) -> Settings {
        Settings {
            out,
            recipe,
            search: Nvfp4Search::DEFAULT,
            damp: DEFAULT_DAMP,
            act_order: false,
            batch: DEFAULT_BATCH,
            length,
            tokens,
            key,
            lanes: ROUNDING_LANES,
            staging_bytes: STAGING_BYTES,
            lane_pinned_bytes: LANE_PINNED_BYTES,
            device_reserve: DEVICE_RESERVE,
            chunk_bytes: UPLOAD_CHUNK_BYTES,
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
    states: States,
    batch_states: DeviceBuffer,
    /// The main lane's second set of math handles: the pipeline holds the first.
    handles: Handoff<LaneHandles>,
    lanes: Handoff<Vec<GptqLane>>,
    services: Arc<Services>,
    resume: OnceLock<ResumeFile>,
    hidden: usize,
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

fn gpu_error(error: impl Display) -> GpuError {
    GpuError::Invalid(error.to_string())
}

/// A continuation taken once, from whichever call ends the chain.
type Once<T> = Arc<Handoff<Then<T>>>;

fn single<T>(then: Then<T>) -> Once<T> {
    let cell = Arc::new(Handoff::empty());
    cell.send(then);
    cell
}

fn end<T>(cell: &Once<T>, result: Result<T, CalibrateError>) {
    if let Some(then) = cell.take() {
        then(result);
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
        let batch_states = DeviceBuffer::alloc(&library, batch * length * hidden * BF16_BYTES)
            .map_err(|error| CalibrateError::failed("the batch states", error))?;

        let lane_names: Vec<String> = (0..settings.lanes.max(1)).map(|lane| format!("gptq{lane}")).collect();
        let lane_refs: Vec<&str> = lane_names.iter().map(String::as_str).collect();
        let (pool, streams) = gpu
            .open_lanes(&lane_refs, StreamOptions::default())
            .map_err(|error| CalibrateError::failed("the GPTQ lanes", error))?;
        let gptq_completions = Completions::new(&pool);
        let lane_pinned = PinnedPool::new(&library, settings.lane_pinned_bytes);
        let gptq_settings = GptqSettings {
            search: settings.search,
            ..GptqSettings::default()
        };
        let mut lanes = Vec::new();
        for mut stream in streams {
            let mut handles = None;
            Stream::submit(&mut stream, |_| {
                handles = Some(math.lane_handles());
                Ok(())
            })
            .map_err(|error| CalibrateError::failed("the GPTQ lane's handles", error))?;
            let handles = handles
                .ok_or_else(|| CalibrateError::Failed("the GPTQ lane made no handles".to_owned()))?
                .map_err(|error| CalibrateError::failed("the GPTQ lane's handles", format!("{error:?}")))?;
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
        let capacity = free.saturating_sub(settings.device_reserve);
        if capacity == 0 {
            return Err(CalibrateError::Failed(format!(
                "the device has {free} bytes free, no more than the {} the run leaves for the layer",
                settings.device_reserve
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
                capacity: settings.lane_pinned_bytes,
                reserve: 0,
            },
        ]);
        let owner = planner.open_owner(OwnerSpec::new("calibrate", Priority(0), Scope::Process));
        let services = Arc::new(Services {
            planner,
            owner,
            device: LocationId(0),
            pinned: LocationId(1),
            settings: RoundSettings {
                recipe: settings.recipe.clone(),
                damp: settings.damp,
                act_order: settings.act_order,
                slice_rows: HEAD_ROWS,
            },
            out: settings.out.clone(),
            io,
            host,
        });
        // The device and its lane pools stay open for the lanes' life: the queues hold them weakly.
        forget((gpu, shared, pool));
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
            states,
            batch_states,
            handles: Handoff::empty(),
            lanes: Handoff::empty(),
            services,
            resume: OnceLock::new(),
            hidden,
        });
        inner.handles.send(symmetric);
        inner.lanes.send(lanes);
        Ok((Calibration { inner }, vec![completions, gptq_completions]))
    }
}

impl Inner {
    fn host_states(&self) -> HostStates {
        // SAFETY: the states are live as long as the pipeline, and no call but the one running touches them.
        unsafe { HostStates::new(self.states.pointer(), self.states.bytes) }
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
            length: self.settings.length as u32,
            batch: self.settings.batch as u32,
            observer,
            write,
        }
    }

    fn resume_file(&self) -> Result<&ResumeFile, CalibrateError> {
        self.resume
            .get()
            .ok_or_else(|| CalibrateError::Failed("the resume file is not open".to_owned()))
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
        let cell = single(then);
        let ready = Arc::clone(&cell);
        let staging = Arc::clone(&self.inner.staging);
        let started = stage(
            &self.inner.objects,
            Target::Layer(layer),
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

    fn layer(&self, layer: usize, staged: StagedLayer, then: Then<()>) {
        upload(Arc::clone(&self.inner), layer, staged, single(then));
    }

    fn checkpoint(&self, done: usize, then: Then<()>) {
        checkpoint(Arc::clone(&self.inner), done, single(then));
    }

    fn finish(&self, then: Then<()>) {
        then(Err(CalibrateError::Invalid(
            "the output head, the draft head and the MTP block are not rounded by this build".to_owned(),
        )));
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
                    let _ = opening.resume.set(file.clone());
                    read_header(opening, file, layers, then);
                }
            }
            idle()
        }),
    );
    if let Err(error) = opened {
        end(&failed, Err(CalibrateError::failed("the resume file", error)));
    }
}

fn read_header(inner: Arc<Inner>, file: ResumeFile, layers: usize, then: Once<usize>) {
    let failed = Arc::clone(&then);
    let started = file.header(Box::new(move |header| {
        match header {
            Err(error) => end(&then, Err(CalibrateError::failed("the resume header", error))),
            Ok(header) => match resume_at(&header, &inner.settings.key, layers) {
                0 => embed(inner, then),
                done => match inner.resume_file().cloned() {
                    Ok(file) => read_states(inner, file, 0, done, then),
                    Err(error) => end(&then, Err(error)),
                },
            },
        }
        idle()
    }));
    if let Err(error) = started {
        end(&failed, Err(CalibrateError::failed("the resume header", error)));
    }
}

/// Reads the states of the resume file into the pinned memory, a chunk after another, then starts at layer `done`.
fn read_states(inner: Arc<Inner>, file: ResumeFile, at: usize, done: usize, then: Once<usize>) {
    if at >= inner.states.bytes {
        end(&then, Ok(done));
        return;
    }
    let length = STATES_CHUNK.min(inner.states.bytes - at);
    let failed = Arc::clone(&then);
    let next = file.clone();
    let started = file.read(
        HEADER + at as u64,
        length,
        Box::new(move |bytes| {
            match bytes {
                Err(error) => end(&then, Err(CalibrateError::failed("the resume states", error))),
                Ok(bytes) => {
                    // SAFETY: the range is inside the states, which nothing else touches before the run starts.
                    unsafe { slice::from_raw_parts_mut(inner.states.pointer().add(at), length) }
                        .copy_from_slice(&bytes);
                    read_states(inner, next, at + length, done, then);
                }
            }
            idle()
        }),
    );
    if let Err(error) = started {
        end(&failed, Err(CalibrateError::failed("the resume states", error)));
    }
}

/// Looks the tokens up in the embedding table: each token's row is copied into the states.
fn embed(inner: Arc<Inner>, then: Once<usize>) {
    let store = Arc::clone(inner.objects.store());
    let tensor = match store.tensor(EMBEDDING) {
        Ok(tensor) => tensor.clone(),
        Err(error) => return end(&then, Err(CalibrateError::failed("the embedding table", error))),
    };
    let row_bytes = inner.row_bytes() as u64;
    if tensor.bytes % row_bytes != 0 {
        return end(
            &then,
            Err(CalibrateError::Failed(format!(
                "the embedding table of {} bytes is not rows of {row_bytes} bytes",
                tensor.bytes
            ))),
        );
    }
    let memory = match PinnedBuffer::alloc(&inner.library, tensor.bytes as usize) {
        Ok(memory) => memory,
        Err(error) => return end(&then, Err(CalibrateError::failed("the embedding table", error))),
    };
    let failed = Arc::clone(&then);
    let started = store.read_tensor(
        EMBEDDING,
        memory,
        Arc::new(move |read: Result<PinnedBuffer, CheckpointError>| {
            let outcome = read
                .map_err(|error| CalibrateError::failed("the embedding table", error))
                .and_then(|memory| copy_rows(&inner, &memory, tensor.bytes / row_bytes));
            end(&then, outcome.map(|()| 0));
        }),
    );
    if let Err(error) = started {
        end(&failed, Err(CalibrateError::failed("the embedding table", error)));
    }
}

fn copy_rows(inner: &Inner, table: &PinnedBuffer, vocabulary: u64) -> Result<(), CalibrateError> {
    let row = inner.row_bytes();
    let (source, target) = (table.as_ptr(), inner.states.pointer());
    for (position, &token) in inner.settings.tokens.iter().enumerate() {
        let token = u64::try_from(token)
            .ok()
            .filter(|token| *token < vocabulary)
            .ok_or_else(|| {
                CalibrateError::Invalid(format!(
                    "token {token} at {position} is outside the {vocabulary} embeddings"
                ))
            })?;
        // SAFETY: the token's row is inside the table, and the position's row is inside the states; they are two
        // buffers.
        unsafe {
            target
                .add(position * row)
                .copy_from_nonoverlapping(source.add(token as usize * row), row)
        };
    }
    Ok(())
}

// ---- checkpoint ----

fn checkpoint(inner: Arc<Inner>, done: usize, then: Once<()>) {
    let file = match inner.resume_file() {
        Ok(file) => file.clone(),
        Err(error) => return end(&then, Err(error)),
    };
    // The states are replaced: a header that names no layers first, so a stop in between resumes from the start.
    let invalid = resume_header(&inner.settings.key, 0);
    let failed = Arc::clone(&then);
    let next = file.clone();
    let started = file.set_header(
        &invalid,
        Box::new(move |set| {
            match set {
                Err(error) => end(&then, Err(CalibrateError::failed("the resume header", error))),
                Ok(()) => write_states(inner, next, 0, done, then),
            }
            idle()
        }),
    );
    if let Err(error) = started {
        end(&failed, Err(CalibrateError::failed("the resume header", error)));
    }
}

fn write_states(inner: Arc<Inner>, file: ResumeFile, at: usize, done: usize, then: Once<()>) {
    let failed = Arc::clone(&then);
    if at >= inner.states.bytes {
        let valid = resume_header(&inner.settings.key, done);
        let started = file.set_header(
            &valid,
            Box::new(move |set| {
                end(
                    &then,
                    set.map_err(|error| CalibrateError::failed("the resume header", error)),
                );
                idle()
            }),
        );
        if let Err(error) = started {
            end(&failed, Err(CalibrateError::failed("the resume header", error)));
        }
        return;
    }
    let length = STATES_CHUNK.min(inner.states.bytes - at);
    // SAFETY: the range is inside the states, which the run does not touch while it checkpoints.
    let bytes = unsafe { slice::from_raw_parts(inner.states.pointer().add(at), length) }.to_vec();
    let next = file.clone();
    let started = file.write(
        HEADER + at as u64,
        bytes,
        Box::new(move |wrote: Result<u64, ResumeError>| {
            match wrote {
                Err(error) => end(&then, Err(CalibrateError::failed("the resume states", error))),
                Ok(_) => write_states(inner, next, at + length, done, then),
            }
            idle()
        }),
    );
    if let Err(error) = started {
        end(&failed, Err(CalibrateError::failed("the resume states", error)));
    }
}

// ---- a layer ----

/// Registers `work` to run in the frame that follows `marker` on the pipeline's lanes.
fn after(inner: &Arc<Inner>, marker: Marker, work: impl FnOnce() + Send + 'static) {
    inner.completions.after_marker(marker, step(Kind::Resolve, work));
}

fn upload(inner: Arc<Inner>, layer: usize, staged: StagedLayer, then: Once<()>) {
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
                Err(CalibrateError::failed(&format!("uploading layer {layer}"), error)),
            );
        }
        Ok((weights, marker)) => {
            let next = Arc::clone(&inner);
            after(&inner, marker, move || {
                drop(staged);
                observe(next, weights, then);
            });
        }
    }
}

/// The observe pass: the layer runs over every sequence and the Hessians fill; its outputs are discarded.
fn observe(inner: Arc<Inner>, weights: LayerWeights, then: Once<()>) {
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
                Ok((weights, set)) => complete(next, weights, set, ended),
                Err(error) => end(&ended, Err(error)),
            }
        }),
    );
    if let Err(error) = started {
        end(&then, Err(CalibrateError::failed("the observe pass", error)));
    }
}

/// Completes the Hessians and makes each linear's decoded weight, on the main lane, then rounds.
fn complete(inner: Arc<Inner>, weights: LayerWeights, set: HessianSet<DeviceBuffer>, then: Once<()>) {
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
