//! Batched verification on a GPU: one quantum that verifies the rows of several sequences, each against its own state,
//! compared with each sequence verified alone. The tests are `#[ignore]`d: they need the card, so run them through
//! `system-run` with a shared grant.
//!
//! The model is the first four layers of the RCO artifact (three GDN layers and one full-attention layer), its MTP
//! layer, heads and embedding, loaded object by object from the artifact named by `euhedral.qwen.rco-artifact`; the
//! library comes from `euhedral.cuda.library`. A test skips with a message when either is absent. The test thread is
//! the owner of every sequence: it reserves the cache rows, uploads the input record, starts a quantum, waits for its
//! retirement, reads the results and commits.

#![allow(dead_code, reason = "the rig of the capture tests, of which these tests use a part")]
#![allow(clippy::print_stderr, reason = "a test reports what it skipped")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: the test thread waits on hand-offs and copies device memory to check results"
)]
#![allow(clippy::print_stdout, reason = "a test reports what it measured")]

use crate::linear_gpu;
use crate::typed;

use typed::Typed;

use crate::common;

use std::collections::HashMap;
use std::fmt::Display;
use std::mem;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock, PoisonError, Weak};
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_gpu::Ran;
use euhedral_gpu::cuda::CudaQueue;
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, GraphRuntime, Lake, LanePool, Marker, QuantumOutcome, RetryCounts, UnitGuard,
};
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, LaneOps, PinnedPool, SubmitMode};
use euhedral_inference_engine::Handoff;
use euhedral_inference_engine::formats::{Nvfp4Linear, TrellisLinear};
use euhedral_inference_engine::generation::{Accepted, accept};
use euhedral_inference_engine::lattice_config;
use euhedral_inference_engine::model::{
    LayerIndex, LayerKind, MTP_ATTENTION_PACK, ModelConfig, Residence, Role, WeightRef, object_key,
};
use euhedral_inference_engine::plan::{Formats, Plan, WeightLookup};
use euhedral_inference_engine::quantum::Executor;
use euhedral_inference_engine::residency::planner::{ResidencyPlan, nvfp4, plan as residency_plan};
use euhedral_inference_engine::stages::attention::KvSource;
use euhedral_inference_engine::stages::sampling::SelectionPool;
use euhedral_inference_engine::stages::sampling::token_of;
use euhedral_inference_engine::stages::{
    DraftSeed, DraftTokenWrite, InputRecordRef, LaneTable, Outcome as Conclusion, Part, PlaceProbe, Quantum,
    QuantumError, QuantumKind, SequenceRefs,
};
use euhedral_inference_engine::state::attention::AttentionScratch;
use euhedral_inference_engine::state::gdn::GdnStates;
use euhedral_inference_engine::state::kv::{KvCache, KvRetained};
use euhedral_inference_engine::state::released::ReleasedBuffers;
use euhedral_inference_engine::state::{SequenceDevice, SequenceId, SequenceState};
use euhedral_inference_engine::workspace::{BufferSlot, Workspace, WorkspaceRows};
use euhedral_inference_settings::{KvFormat, ModelSettings, Settings, TrellisKernelSettings};
use linear_gpu::Artifact;

/// Layers of the reduced model: the artifact's layers 0 to 3.
const LAYERS: usize = 4;
/// The most steps a world runs.
const MAX_STEPS: usize = 512;
/// Logits rows the test workspace holds: three sequences of eight rows.
const BATCH_LOGITS_ROWS: u64 = 24;

fn library() -> Option<Arc<CudaLibrary>> {
    let path = test_support::require_property("euhedral.cuda.library")?;
    Some(CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads"))
}

fn rco_config() -> ModelConfig {
    common::config()
}

/// The reduced model's dimensions: the RCO model's first four layers.
fn reduced_config() -> Arc<ModelConfig> {
    let mut config = rco_config();
    config.layer_kinds.truncate(LAYERS);
    config.num_hidden_layers = LAYERS;
    assert_eq!(config.layer_kinds[3], LayerKind::FullAttention);
    Arc::new(config)
}

struct Described(HashMap<(LayerIndex, Role), WeightRef>);

impl WeightLookup for Described {
    fn weight(&self, layer: LayerIndex, role: Role) -> Option<&WeightRef> {
        self.0.get(&(layer, role))
    }
}

/// The reduced model's weights on the device, with the buffers they address.
struct Model {
    weights: Described,
    _buffers: Vec<DeviceBuffer>,
    _trellis: Vec<linear_gpu::Trellis>,
}

fn upload(library: &Arc<CudaLibrary>, bytes: &[u8]) -> DeviceBuffer {
    let buffer = DeviceBuffer::alloc(library, bytes.len().max(256)).unwrap();
    buffer.copy_from_host(0, bytes).unwrap();
    buffer
}

/// Loads the objects of the first `LAYERS` layers, the globals and the MTP layer from the artifact.
fn load_model(artifact: &mut Artifact, library: &Arc<CudaLibrary>) -> Model {
    let names: Vec<String> = artifact
        .table()
        .entries()
        .iter()
        .map(|entry| entry.name().to_string())
        .collect();
    let mut weights = HashMap::new();
    let mut buffers = Vec::new();
    let mut trellis = Vec::new();
    for name in names {
        if let Some(rest) = name.strip_prefix("text/layers/") {
            let layer: usize = rest.split('/').next().unwrap().parse().unwrap();
            if layer >= LAYERS {
                continue;
            }
        }
        if name == MTP_ATTENTION_PACK {
            let entry = artifact.table().get(&name).unwrap();
            let (shape, dtype, format, layout) =
                (entry.shape().to_vec(), entry.dtype(), entry.format(), entry.layout());
            let bytes = artifact.bytes(&name);
            let (rows, k) = (shape[0], shape[1]);
            let half = rows / 2;
            let code = nvfp4::row_code_bytes(k);
            let row_scale = nvfp4::row_scale_bytes(k, layout).unwrap();
            let trailer = nvfp4::trailer_bytes(layout).unwrap() as usize;
            for (index, half_name) in ["mtp/layer/attention/query_key", "mtp/layer/attention/gate_value"]
                .iter()
                .enumerate()
            {
                let first = index as u64 * half;
                let mut out = vec![0u8; nvfp4::byte_size(half, k, layout).unwrap() as usize];
                let (from, len) = ((first * code) as usize, (half * code) as usize);
                out[..len].copy_from_slice(&bytes[from..from + len]);
                let from = (nvfp4::scale_offset(rows, k) + first * row_scale) as usize;
                let (to, len) = (nvfp4::scale_offset(half, k) as usize, (half * row_scale) as usize);
                out[to..to + len].copy_from_slice(&bytes[from..from + len]);
                let from = nvfp4::trailer_offset(rows, k, layout).unwrap() as usize;
                let to = nvfp4::trailer_offset(half, k, layout).unwrap() as usize;
                out[to..to + trailer].copy_from_slice(&bytes[from..from + trailer]);
                let buffer = upload(library, &out);
                let key = object_key(half_name, LAYERS).unwrap();
                weights.insert(
                    key,
                    WeightRef {
                        name: (*half_name).to_string(),
                        shape: vec![half, k],
                        dtype,
                        format,
                        layout,
                        ptr: buffer.ptr(),
                        bytes: out.len() as u64,
                        residence: Residence::Device,
                        trellis: None,
                    },
                );
                buffers.push(buffer);
            }
            continue;
        }
        let Some(key) = object_key(&name, LAYERS) else { continue };
        let entry = artifact.table().get(&name).unwrap();
        if entry.format() == euhedral_inference_edrl::WeightFormat::Trellis {
            let object = artifact.trellis(library, &name);
            weights.insert(key, object.weight.clone());
            trellis.push(object);
        } else {
            let (buffer, weight) = artifact.plain(library, &name);
            weights.insert(key, weight);
            buffers.push(buffer);
        }
    }
    library.synchronize().unwrap();
    Model {
        weights: Described(weights),
        _buffers: buffers,
        _trellis: trellis,
    }
}

/// The reduced model, loaded for the tests running now: its weights are read-only, so the worlds that exist together
/// read the same buffers (each test still owns its library, lanes, lattice and workspace). The last world to end frees
/// them, so the engines of the other tests find the card empty.
fn shared_model() -> Option<Arc<Model>> {
    static SHARED: Mutex<Weak<Model>> = Mutex::new(Weak::new());
    let mut shared = SHARED.lock().unwrap_or_else(PoisonError::into_inner);
    if let Some(model) = shared.upgrade() {
        return Some(model);
    }
    let library = library()?;
    let mut artifact = Artifact::open()?;
    let model = Arc::new(load_model(&mut artifact, &library));
    *shared = Arc::downgrade(&model);
    Some(model)
}

/// A lattice, the lanes and the plan: what a test's quanta run on.
struct World {
    library: Arc<CudaLibrary>,
    config: Arc<ModelConfig>,
    residency: ResidencyPlan,
    plan: Arc<Plan>,
    /// The RustEE lanes: the stages' own, then the owner's for the reservations and the input records.
    typed: Typed,
    table: LaneTable,
    pool: Arc<LanePool>,
    runtime: GraphRuntime,
    workspace: Arc<Workspace>,
    pinned: Arc<PinnedPool>,
    released: Arc<ReleasedBuffers>,
    lattice: Arc<ControlPlaneLattice>,
    lake: Arc<Lake>,
    unit: Option<UnitGuard>,
    /// The weights the plan addresses: the last world to end frees them.
    _model: Arc<Model>,
}

impl World {
    fn open() -> Option<World> {
        let model = shared_model()?;
        let library = library()?;
        let config = reduced_config();
        let residency = residency_plan(&rco_config(), &common::rco_table(), &common::settings(8192), 15 << 30).unwrap();
        let formats = Formats::new(
            Arc::new(TrellisLinear::from_settings(
                &TrellisKernelSettings::default(),
                library.device().sm_count,
            )),
            Arc::new(Nvfp4Linear::new(&library)),
        );
        let plan = Arc::new(Plan::build(&config, &model.weights, &residency, &formats, None, None).unwrap());
        let width = plan.fan_out_width();
        // The graphs' lanes, then the owner's.
        let typed = Typed::plain(&library, width + 1, 8192);
        let table = typed.table.clone();
        let pool = Arc::clone(&typed.queues);
        let settings = Settings::with_model(ModelSettings::default());
        let lattice = ControlPlaneLattice::new(lattice_config(&settings)).unwrap();
        let lake = Lake::new(
            Arc::clone(&lattice),
            NonZeroUsize::new(2).unwrap(),
            Duration::from_secs(1),
        );
        let completions = Completions::new(&pool);
        lattice
            .add_upstream(Arc::clone(&completions) as Arc<dyn LatticeSource>)
            .unwrap();
        let runtime = GraphRuntime {
            stages: Arc::clone(&lake) as Arc<dyn FrameSink>,
            completions,
            lanes: Arc::clone(&pool),
            retries: Arc::new(RetryCounts::default()),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        };
        let nvfp4_native = residency
            .device
            .iter()
            .filter(|placement| placement.format == euhedral_inference_edrl::WeightFormat::Nvfp4)
            .filter_map(|placement| match placement.shape.as_slice() {
                [out, input] => {
                    Some(library.nvfp4_native_scratch_bytes(residency.rows.rows as u32, *input as u32, *out as u32))
                }
                _ => None,
            })
            .max()
            .unwrap_or(0);
        let regions = [
            residency.scratch_bytes.max(nvfp4_native),
            residency.decode_scratch_bytes,
        ];
        // A batch scores the rows of all its sequences at once: room for three verifications of the widest width.
        let rows = WorkspaceRows {
            logits_rows: residency.rows.logits_rows.max(BATCH_LOGITS_ROWS),
            ..residency.rows
        };
        let workspace = Arc::new(Workspace::allocate(&library, &config, rows, &regions).unwrap());
        let unit = lake.admit().unwrap();
        Some(World {
            _model: model,
            pinned: PinnedPool::new(&library, 64 << 20),
            released: Arc::new(ReleasedBuffers::new()),
            library,
            config,
            residency,
            plan,
            typed,
            table,
            pool,
            runtime,
            workspace,
            lattice,
            lake,
            unit: Some(unit),
        })
    }

    fn close(mut self) {
        drop(self.unit.take());
        self.lake.complete_gracefully();
        self.lattice.close();
        self.typed.sync();
    }

    /// The lane of the sequence's own submissions.
    fn owner(&self) -> &Arc<CudaQueue> {
        self.typed.lane(self.plan.fan_out_width() as u16)
    }

    /// Queues `work` as one submission of the owner's lane and returns its marker and what it made; nothing waits.
    fn submit_owner<T, E: Display>(&self, work: impl FnOnce(&mut dyn LaneOps) -> Result<T, E>) -> (T, Marker) {
        self.table.submit_ops(self.owner(), false, work).unwrap()
    }

    /// Runs `work` as a submission on the owner's lane, and waits for it.
    fn on_owner<T, E: Display>(&self, work: impl FnOnce(&mut dyn LaneOps) -> Result<T, E>) -> T {
        let (made, _) = self.submit_owner(work);
        self.typed.sync();
        made
    }

    /// Runs `call`, which uses the library's synchronous calls, as a submission of the owner's lane.
    fn on_owner_host<T>(&self, call: impl FnOnce() -> T) -> T {
        let (made, _) = self.table.submit_host(self.owner(), call).unwrap();
        self.typed.sync();
        made
    }

    /// `bytes` bytes of `buffer` from `offset`.
    fn download(&self, buffer: &DeviceBuffer, offset: usize, bytes: usize) -> Vec<u8> {
        let mut host = vec![0u8; bytes];
        self.on_owner_host(|| buffer.copy_to_host(offset, &mut host)).unwrap();
        host
    }

    fn sequence(&self) -> SequenceState {
        let config = &self.config;
        let gdn = self.on_owner(|lane| {
            let verified = (1 + self.residency.draft_depth) as u64;
            GdnStates::allocate(&self.library, config, verified, lane)
        });
        let kv = |layers: usize| {
            KvCache::new(
                &self.library,
                &self.released,
                layers,
                config.kv_width() as u64,
                KvFormat::Nvfp4,
                4096,
                0,
            )
            .unwrap()
        };
        let scratch = AttentionScratch::allocate(&self.library, config, 8, true, self.residency.rows.rows).unwrap();
        SequenceState::from_parts(
            kv(config.attention_layers()),
            Some(kv(1)),
            Arc::new(SequenceDevice { gdn, scratch }),
            SelectionPool::new(&self.library, 4, 8).unwrap(),
        )
    }
}

/// A retired quantum and how it ended.
type Retired = (Quantum, QuantumOutcome);

/// The longest a device test waits for one retirement; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);

/// Where the retirements of one executor are handed to the test thread: step `i` waits on join `i`.
struct Steps {
    joins: Vec<OnceLock<Arc<Handoff<Retired>>>>,
    retired: AtomicUsize,
}

impl Steps {
    fn new() -> Arc<Steps> {
        Arc::new(Steps {
            joins: (0..MAX_STEPS).map(|_| OnceLock::new()).collect(),
            retired: AtomicUsize::new(0),
        })
    }

    /// Arms step `index` and returns what waits for its retirement.
    fn arm(&self, index: usize) -> Arc<Handoff<Retired>> {
        let waiting = Handoff::new();
        assert!(self.joins[index].set(Arc::clone(&waiting)).is_ok());
        waiting
    }
}

/// What one step left behind.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    keys: Vec<u64>,
    logits: Vec<u8>,
    /// The rows the step committed.
    committed: usize,
}

/// A step of a sequence.
#[derive(Clone, Copy, Debug)]
enum Step {
    Prefill(usize),
    Decode,
    /// A verification of `rows` rows of which `commit` are accepted.
    Verify {
        rows: usize,
        commit: usize,
    },
    Draft,
    /// A verification of `rows` rows (the drafts are the tokens after the first) that commits the rows greedy
    /// acceptance resolves.
    Greedy(usize),
}

fn token(position: u64) -> u32 {
    ((position * 7919 + 13) % 200_000) as u32
}

/// A quantum built for a step, with what the owner keeps until it retires.
struct Built {
    quantum: Quantum,
    step: Step,
    start: u64,
    drafting: bool,
    logits_rows: usize,
    retained: KvRetained,
    tokens: Vec<u32>,
}

/// One sequence on a world: its state and the executor, with the graph cache, that runs it.
struct Run {
    sequence: SequenceState,
    executor: Executor,
    steps: Arc<Steps>,
    started: usize,
    base_next: u64,
    mtp_next: u64,
    /// Rows each GDN layer replays first, from a partial acceptance.
    replay: Vec<u32>,
    ids: DeviceBuffer,
    /// Quanta that ran as one launch of a recorded graph: decode, verification, draft.
    replayed: [usize; 3],
    /// Quanta that recorded a graph as they ran.
    recorded: usize,
    /// Quanta that had a capture key.
    keyed: usize,
    /// Whether decode and verification quanta launch their kernels programmatically (the setting `quantum.mode.pdl`).
    pdl: bool,
    /// The token ids of the next quantum, instead of the document's.
    tokens: Option<Vec<u32>>,
    /// One past the highest position a quantum of the run wrote.
    high: u64,
}

impl Run {
    /// A run whose cache keeps `captures` graphs for each view; with 0 every quantum runs stage by stage.
    fn new(world: &World, captures: usize) -> Run {
        let steps = Steps::new();
        let again = Arc::clone(&steps);
        let executor = Executor::new(
            Arc::clone(&world.plan),
            &world.runtime,
            world.plan.fan_out_width(),
            Arc::clone(&world.typed.gpu),
            captures,
            move |quantum, outcome| {
                let index = again.retired.fetch_add(1, Ordering::AcqRel);
                again.joins[index].get().expect("armed").send((quantum, outcome));
            },
        )
        .unwrap();
        let ids_bytes: Vec<u8> = (0..131_072u32).flat_map(|row| (row * 2).to_ne_bytes()).collect();
        let ids = {
            let buffer = DeviceBuffer::alloc(&world.library, ids_bytes.len()).unwrap();
            world.on_owner_host(|| buffer.copy_from_host(0, &ids_bytes)).unwrap();
            buffer
        };
        Run {
            sequence: world.sequence(),
            executor,
            steps,
            started: 0,
            base_next: 0,
            mtp_next: 0,
            replay: Vec::new(),
            ids,
            replayed: [0; 3],
            recorded: 0,
            keyed: 0,
            pdl: false,
            tokens: None,
            high: 0,
        }
    }

    /// Moves the base cache to `position` without computing anything: the rows are reserved and committed, their
    /// contents are whatever the device memory held.
    fn skip_to(&mut self, world: &World, position: u64) {
        let mut retained = KvRetained::default();
        let cache = &mut self.sequence.kv;
        let start = cache.submitted();
        world.on_owner(|lane| cache.reserve(lane, &world.pinned, start, position - start, &mut retained));
        cache.appended(position - start).unwrap();
        cache.commit(position - start).unwrap();
        // The reservation was this cache's call for the next quantum's references: claim it.
        drop(self.sequence.refs(KvSource::Base).unwrap());
        self.base_next = position;
        drop(retained);
    }

    /// Prefills the base cache up to `position` in quanta of at most 256 rows.
    fn prefill_to(&mut self, world: &World, position: u64) {
        while self.base_next < position {
            let rows = (position - self.base_next).min(256) as usize;
            self.step(world, Step::Prefill(rows));
        }
    }

    /// Reserves, uploads the input record and builds the quantum of `step`; nothing runs yet.
    fn build(&mut self, world: &World, step: Step) -> Built {
        let config = &world.config;
        let (kind, rows) = match step {
            Step::Prefill(rows) => (QuantumKind::Prefill, rows),
            Step::Decode => (QuantumKind::Decode, 1),
            Step::Verify { rows, .. } | Step::Greedy(rows) => (QuantumKind::Verify, rows),
            Step::Draft => (QuantumKind::Draft, 1),
        };
        let drafting = kind == QuantumKind::Draft;
        let start = if drafting { self.mtp_next } else { self.base_next };
        // The owner reserves the rows the quantum appends, and records the marker of that work.
        let mut retained = KvRetained::default();
        let marker = {
            let cache = if drafting {
                self.sequence.mtp_kv.as_mut().unwrap()
            } else {
                &mut self.sequence.kv
            };
            let ((), marker) =
                world.submit_owner(|lane| cache.reserve(lane, &world.pinned, start, rows as u64, &mut retained));
            cache.appended(rows as u64).unwrap();
            marker
        };
        let replay = if matches!(kind, QuantumKind::Verify | QuantumKind::Decode) {
            mem::take(&mut self.replay)
        } else {
            Vec::new()
        };
        let source = if drafting { KvSource::Mtp } else { KvSource::Base };
        let refs: SequenceRefs = self
            .sequence
            .refs(source)
            .unwrap()
            .ordered_after(marker, &world.pool)
            .with_replay_rows(replay);
        let mode = SubmitMode {
            pdl: self.pdl && matches!(kind, QuantumKind::Decode | QuantumKind::Verify) && start < 1024,
            row_exact: kind == QuantumKind::Verify,
        };
        let tokens: Vec<u32> = self
            .tokens
            .take()
            .unwrap_or_else(|| (0..rows as u64).map(|row| token(start + row)).collect());
        self.high = self.high.max(start + rows as u64);
        let mut quantum = Quantum::new(
            kind,
            rows,
            InputRecordRef::new(start),
            Arc::clone(&world.workspace),
            refs,
            mode,
            Arc::clone(config),
            world.table.clone(),
        )
        .unwrap()
        .with_tokens(tokens.clone())
        .unwrap();
        let logits_rows = if kind == QuantumKind::Prefill {
            0
        } else {
            quantum.logits_rows()
        };
        quantum = quantum.with_logits_rows(logits_rows).unwrap();
        if logits_rows > 0 {
            quantum = quantum.with_selection_lease(self.sequence.selections.acquire().unwrap());
        }
        let draft = self.sequence.device.scratch.draft().unwrap();
        if kind == QuantumKind::Prefill {
            quantum = quantum.seeding_draft().unwrap();
        }
        if drafting {
            // A recursive draft seeds from the same rows every time, so its key repeats.
            quantum = quantum
                .with_draft_seed(DraftSeed::Address(draft.seed_rows.ptr()))
                .unwrap()
                .with_draft_token_write(DraftTokenWrite {
                    slot: 1,
                    ids: self.ids.ptr(),
                })
                .unwrap();
        }
        // The input record's upload, ahead of the quantum's first stage.
        let ((), prepared) = world.submit_owner(|lane| quantum.prepare(&world.pinned, lane));
        quantum.prepared_after(prepared);
        Built {
            quantum,
            step,
            start,
            drafting,
            logits_rows,
            retained,
            tokens,
        }
    }

    /// Runs one step to its retirement and reads what it left.
    fn step(&mut self, world: &World, step: Step) -> Outcome {
        let built = self.build(world, step);
        self.run(world, built)
    }

    fn run(&mut self, world: &World, built: Built) -> Outcome {
        let Built {
            quantum,
            step,
            start,
            drafting,
            logits_rows,
            retained,
            tokens,
        } = built;
        let config = &world.config;
        self.keyed += usize::from(quantum.capture_key().is_some());
        let waiting = self.steps.arm(self.started);
        self.started += 1;
        let started = self
            .executor
            .start(quantum, Arc::new(AtomicBool::new(false)), &[])
            .unwrap();
        // Nothing is chained to this quantum: its reference is not held, or it would never retire.
        drop(started);
        let (quantum, outcome) = waiting.wait(WAIT).expect("the quantum retires");
        assert_eq!(outcome, QuantumOutcome::Completed, "{step:?} at {start}");
        let kind_index = match step {
            Step::Decode => Some(0),
            Step::Verify { .. } | Step::Greedy(_) => Some(1),
            Step::Draft => Some(2),
            Step::Prefill(_) => None,
        };
        let ran = quantum.graph_run();
        if let (Some(Ran::Replayed), Some(index)) = (ran, kind_index) {
            self.replayed[index] += 1;
        }
        self.recorded += usize::from(matches!(ran, Some(Ran::Recorded | Ran::Rerecorded)));

        let keys: Vec<u64> = (0..logits_rows)
            .map(|row| quantum.selection().unwrap().key(row).unwrap())
            .collect();
        let width = if drafting { 131_072 } else { config.vocab_size };
        let logits = if logits_rows == 0 {
            Vec::new()
        } else {
            let row_bytes = config.vocab_size * 2;
            let mut out = Vec::new();
            for row in 0..logits_rows {
                out.extend(world.download(world.workspace.buffer(BufferSlot::Logits), row * row_bytes, width * 2));
            }
            out
        };
        // The device work retired: the frontiers move.
        let committed = match step {
            Step::Prefill(rows) => {
                self.sequence.kv.commit(rows as u64).unwrap();
                self.base_next += rows as u64;
                rows
            }
            Step::Decode => {
                self.sequence.kv.commit(1).unwrap();
                self.base_next += 1;
                1
            }
            Step::Verify { rows, commit } => {
                self.committed_verify(config, rows, commit);
                commit
            }
            Step::Greedy(rows) => {
                let commit = greedy(&tokens, &keys).count;
                self.committed_verify(config, rows, commit);
                commit
            }
            Step::Draft => {
                self.sequence.mtp_kv.as_mut().unwrap().commit(1).unwrap();
                self.mtp_next += 1;
                1
            }
        };
        drop(quantum);
        drop(retained);
        Outcome {
            keys,
            logits,
            committed,
        }
    }

    /// Commits the first `commit` of the `rows` rows of a verification; the state ran past them, so the next quantum
    /// replays them.
    fn committed_verify(&mut self, config: &ModelConfig, rows: usize, commit: usize) {
        self.sequence.kv.commit_prefix(commit as u64).unwrap();
        self.base_next += commit as u64;
        if commit < rows {
            self.replay = config
                .layer_kinds
                .iter()
                .map(|kind| u32::try_from(commit).unwrap() * u32::from(*kind == LayerKind::GatedDeltaNet))
                .collect();
        }
    }

    /// Ends the run: the sequence closes with no quantum in flight, and its graphs go with the executor.
    fn close(self, world: &World) {
        let Run { sequence, executor, .. } = self;
        executor.clear_graphs();
        drop(executor);
        sequence.close().unwrap();
        world.typed.sync();
    }

    /// The GDN states, layer by layer.
    fn gdn_state(&self, world: &World) -> Vec<Vec<u8>> {
        (0..LAYERS)
            .filter_map(|layer| self.sequence.device.gdn.layer(layer))
            .flat_map(|state| [state.convolution(), state.recurrent()])
            .map(|buffer| world.download(buffer, 0, buffer.bytes()))
            .collect()
    }
}

/// Where the retirements of the batches are handed to the test thread.
struct Runner {
    executor: Executor,
    steps: Arc<Steps>,
    started: usize,
}

impl Runner {
    fn new(world: &World) -> Runner {
        let steps = Steps::new();
        let again = Arc::clone(&steps);
        let executor = Executor::new(
            Arc::clone(&world.plan),
            &world.runtime,
            world.plan.fan_out_width(),
            Arc::clone(&world.typed.gpu),
            0,
            move |quantum, outcome| {
                let index = again.retired.fetch_add(1, Ordering::AcqRel);
                again.joins[index].get().expect("armed").send((quantum, outcome));
            },
        )
        .unwrap();
        Runner {
            executor,
            steps,
            started: 0,
        }
    }

    /// Runs `quantum` to its retirement.
    fn run(&mut self, quantum: Quantum) -> Retired {
        let waiting = self.steps.arm(self.started);
        self.started += 1;
        let started = self
            .executor
            .start(quantum, Arc::new(AtomicBool::new(false)), &[])
            .unwrap();
        drop(started);
        waiting.wait(WAIT).expect("the batch retires")
    }
}

/// What an owner commits of a member of a batch.
#[derive(Clone, Copy, Debug)]
enum Commit {
    /// The first rows, named when the part is made.
    Rows(usize),
    /// The rows greedy acceptance resolves from the member's own selected tokens.
    Greedy,
}

/// A member of a batch: the sequence's run, how many rows it verifies and what it commits.
struct Member {
    rows: usize,
    commit: Commit,
    cancel: Option<Arc<AtomicBool>>,
}

impl Member {
    fn all(rows: usize) -> Member {
        Member {
            rows,
            commit: Commit::Rows(rows),
            cancel: None,
        }
    }

    fn committing(rows: usize, commit: usize) -> Member {
        Member {
            rows,
            commit: Commit::Rows(commit),
            cancel: None,
        }
    }
}

/// What one segment of a batch left behind.
#[derive(Debug)]
struct SegmentOutcome {
    conclusion: Conclusion,
    keys: Vec<u64>,
    logits: Vec<u8>,
    /// The rows the owner committed (0 for a segment that did not complete).
    committed: usize,
}

impl Run {
    /// Reserves the rows of a verification of `member` and makes its part of a batch as sequence `id`.
    fn part(&mut self, world: &World, id: usize, member: &Member) -> (Part, KvRetained, Vec<u32>) {
        let (start, rows) = (self.base_next, member.rows);
        let mut retained = KvRetained::default();
        let ((), marker) = world.submit_owner(|lane| {
            self.sequence
                .kv
                .reserve(lane, &world.pinned, start, rows as u64, &mut retained)
        });
        self.sequence.kv.appended(rows as u64).unwrap();
        let refs = self
            .sequence
            .refs(KvSource::Base)
            .unwrap()
            .ordered_after(marker, &world.pool)
            .with_replay_rows(mem::take(&mut self.replay));
        let tokens: Vec<u32> = self
            .tokens
            .take()
            .unwrap_or_else(|| (0..rows as u64).map(|row| token(start + row)).collect());
        self.high = self.high.max(start + rows as u64);
        let mut part = Part::new(refs, SequenceId(id), start, tokens.clone())
            .selecting_leased(self.sequence.selections.acquire().unwrap());
        if let Commit::Rows(commit) = member.commit {
            part = part.committing(commit).unwrap();
        }
        if let Some(flag) = &member.cancel {
            part = part.cancelled_by(Arc::clone(flag));
        }
        (part, retained, tokens)
    }
}

/// Runs one batched verification of `members` (one per run, in order) and commits what each completed member's owner
/// would. Returns the batch's quantum, kept to read its conclusions, and each segment's outcome.
fn batch(world: &World, runner: &mut Runner, runs: &mut [Run], members: &[Member]) -> (Quantum, Vec<SegmentOutcome>) {
    assert_eq!(runs.len(), members.len());
    let mut parts = Vec::new();
    let mut retained = Vec::new();
    let mut all_tokens = Vec::new();
    for (id, (run, member)) in runs.iter_mut().zip(members).enumerate() {
        let (part, keep, tokens) = run.part(world, id, member);
        parts.push(part);
        retained.push(keep);
        all_tokens.push(tokens);
    }
    let mode = SubmitMode {
        pdl: false,
        row_exact: true,
    };
    let mut quantum = Quantum::batch(
        parts,
        Arc::clone(&world.workspace),
        mode,
        Arc::clone(&world.config),
        world.table.clone(),
    )
    .unwrap();
    assert!(quantum.capture_key().is_none(), "a batch is never captured");
    quantum.validate().unwrap();
    let ((), prepared) = world.submit_owner(|lane| quantum.prepare(&world.pinned, lane));
    quantum.prepared_after(prepared);
    let (quantum, outcome) = runner.run(quantum);
    assert_eq!(outcome, QuantumOutcome::Completed, "the batch ran");
    let row_bytes = world.config.vocab_size * 2;
    let mut outcomes = Vec::new();
    for (index, segment) in quantum.segments().iter().enumerate() {
        let conclusion = quantum.conclusion_for(index, &outcome).unwrap();
        let keys: Vec<u64> = (0..segment.rows())
            .map(|row| segment.selection().unwrap().key(row).unwrap())
            .collect();
        let mut logits = Vec::new();
        for row in 0..segment.rows() {
            logits.extend(world.download(
                world.workspace.buffer(BufferSlot::Logits),
                (segment.row_offset() + row) * row_bytes,
                row_bytes,
            ));
        }
        let mut committed = 0;
        if conclusion == Conclusion::Completed {
            committed = match members[index].commit {
                Commit::Rows(rows) => {
                    assert_eq!(quantum.committed_rows(index).unwrap(), rows, "the part names its rows");
                    rows
                }
                Commit::Greedy => {
                    assert_eq!(quantum.committed_rows(index).unwrap(), segment.rows());
                    greedy(&all_tokens[index], &keys).count
                }
            };
            runs[index].committed_verify(&world.config, members[index].rows, committed);
        }
        outcomes.push(SegmentOutcome {
            conclusion,
            keys,
            logits,
            committed,
        });
    }
    drop(retained);
    (quantum, outcomes)
}

/// The acceptance of a verification of `tokens` (the current token, then the drafts) whose rows selected `keys`.
fn greedy(tokens: &[u32], keys: &[u64]) -> Accepted {
    accept(&tokens[1..], keys, &[], 64)
}

/// The prefix lengths of the three sequences.
const PREFIXES: [u64; 3] = [300, 411, 530];

/// A run prefilled to `prefix`.
fn prefilled(world: &World, prefix: u64) -> Run {
    let mut run = Run::new(world, 0);
    run.prefill_to(world, prefix);
    run
}

/// The KV rows of every layer of the base cache, position by position (the K row, then the V row), up to what the run
/// wrote. The pages are token-major.
fn kv_rows(world: &World, run: &Run) -> Vec<Vec<u8>> {
    let cache = &run.sequence.kv;
    let plane = cache.plane_page_bytes() as usize;
    let row = plane / 256;
    let mut rows = Vec::new();
    for layer in 0..cache.layers() {
        let pages: Vec<Vec<u8>> = cache
            .pages(layer)
            .map(|page| world.typed.download(page, 2 * plane))
            .collect();
        for position in 0..run.high as usize {
            let page = &pages[position / 256];
            let at = (position % 256) * row;
            let mut both = page[at..at + row].to_vec();
            both.extend_from_slice(&page[plane + at..plane + at + row]);
            rows.push(both);
        }
    }
    rows
}

/// Asserts that the rows, the GDN state and the KV rows of `batched` are those of `alone`.
fn assert_state_equal(world: &World, alone: &Run, batched: &Run, what: &str) {
    assert_eq!(alone.base_next, batched.base_next, "{what}: position");
    assert!(
        alone.gdn_state(world) == batched.gdn_state(world),
        "{what}: the GDN states are bit for bit equal"
    );
    assert!(
        kv_rows(world, alone) == kv_rows(world, batched),
        "{what}: the KV rows are bit for bit equal"
    );
}

fn assert_segment_equals(batched: &SegmentOutcome, alone: &Outcome, what: &str) {
    assert_eq!(batched.keys, alone.keys, "{what}: the selected tokens");
    assert!(
        batched.logits == alone.logits,
        "{what}: the logits are bit for bit equal"
    );
    assert_eq!(batched.committed, alone.committed, "{what}: the committed rows");
}

/// Verifies the rounds of `rows` (and, when given, `commits`) on three sequences alone and then batched, and checks
/// every round's rows and the final states.
fn rounds_equal(world: &World, rounds: &[[usize; 3]], commits: Option<&[[usize; 3]]>) {
    let mut alone: Vec<Run> = PREFIXES.iter().map(|prefix| prefilled(world, *prefix)).collect();
    let mut batched: Vec<Run> = PREFIXES.iter().map(|prefix| prefilled(world, *prefix)).collect();
    let mut runner = Runner::new(world);
    for (round, rows) in rounds.iter().enumerate() {
        let commit = |sequence: usize| commits.map_or(rows[sequence], |commits| commits[round][sequence]);
        let wanted: Vec<Outcome> = (0..3)
            .map(|sequence| {
                alone[sequence].step(
                    world,
                    Step::Verify {
                        rows: rows[sequence],
                        commit: commit(sequence),
                    },
                )
            })
            .collect();
        let members: Vec<Member> = (0..3)
            .map(|sequence| Member::committing(rows[sequence], commit(sequence)))
            .collect();
        let (quantum, actual) = batch(world, &mut runner, &mut batched, &members);
        let total: usize = rows.iter().sum();
        assert_eq!(quantum.rows(), total);
        for sequence in 0..3 {
            assert_eq!(quantum.segments()[sequence].rows(), rows[sequence]);
            assert_eq!(
                quantum.segments()[sequence].row_offset(),
                rows[..sequence].iter().sum::<usize>()
            );
            assert_eq!(actual[sequence].conclusion, Conclusion::Completed);
            assert_segment_equals(
                &actual[sequence],
                &wanted[sequence],
                &format!("round {round} sequence {sequence}"),
            );
            assert_eq!(
                alone[sequence].base_next, batched[sequence].base_next,
                "round {round} position"
            );
        }
        drop(quantum);
    }
    for sequence in 0..3 {
        assert_state_equal(
            world,
            &alone[sequence],
            &batched[sequence],
            &format!("sequence {sequence}"),
        );
    }
    drop(runner);
    for run in alone.into_iter().chain(batched) {
        run.close(world);
    }
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn batched_verification_equals_each_sequence_verified_alone() {
    let _light = common::light();
    let Some(world) = World::open() else { return };
    // 24 rows run on the wide verify view; 16 on it as well; 6 on the decode view.
    rounds_equal(&world, &[[8, 8, 8], [5, 8, 3], [1, 2, 3]], None);
    world.close();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn batched_segments_commit_only_their_accepted_rows() {
    let _light = common::light();
    let Some(world) = World::open() else { return };
    // The rows of each round and the rows each sequence commits; a partial acceptance is replayed by the next round.
    rounds_equal(
        &world,
        &[[8, 8, 8], [5, 8, 3], [1, 1, 1]],
        Some(&[[3, 8, 1], [5, 2, 3], [1, 1, 1]]),
    );
    world.close();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn batched_segments_resolve_their_own_greedy_acceptance() {
    let _light = common::light();
    let Some(world) = World::open() else { return };
    let mut alone: Vec<Run> = PREFIXES.iter().map(|prefix| prefilled(&world, *prefix)).collect();
    let mut batched: Vec<Run> = PREFIXES.iter().map(|prefix| prefilled(&world, *prefix)).collect();
    // The index of each sequence's wrong draft; 0 is none.
    let wrong = [2usize, 0, 1];
    let vocabulary = world.config.vocab_size as u32;
    let mut rows_of = Vec::new();
    for (sequence, prefix) in PREFIXES.iter().enumerate() {
        // The model's own next four tokens after the document's, found one row at a time on a throwaway sequence.
        let mut probe = prefilled(&world, *prefix);
        let mut tokens = vec![token(*prefix)];
        for _ in 1..5 {
            probe.tokens = Some(vec![*tokens.last().unwrap()]);
            let out = probe.step(&world, Step::Verify { rows: 1, commit: 1 });
            tokens.push(token_of(out.keys[0]).expect("the probe selected a token"));
        }
        probe.close(&world);
        if wrong[sequence] > 0 {
            tokens[wrong[sequence]] = (tokens[wrong[sequence]] + 1) % vocabulary;
        }
        rows_of.push(tokens);
    }
    let wanted: Vec<Outcome> = (0..3)
        .map(|sequence| {
            alone[sequence].tokens = Some(rows_of[sequence].clone());
            alone[sequence].step(&world, Step::Greedy(5))
        })
        .collect();
    for (sequence, run) in batched.iter_mut().enumerate() {
        run.tokens = Some(rows_of[sequence].clone());
    }
    let mut runner = Runner::new(&world);
    let members: Vec<Member> = (0..3)
        .map(|_| Member {
            rows: 5,
            commit: Commit::Greedy,
            cancel: None,
        })
        .collect();
    let (quantum, actual) = batch(&world, &mut runner, &mut batched, &members);
    let mut accepted = Vec::new();
    for sequence in 0..3 {
        let what = format!("sequence {sequence}");
        assert_segment_equals(&actual[sequence], &wanted[sequence], &what);
        accepted.push(greedy(&rows_of[sequence], &actual[sequence].keys));
        assert_eq!(
            accepted[sequence],
            greedy(&rows_of[sequence], &wanted[sequence].keys),
            "{what}: the acceptance"
        );
        assert_eq!(
            accepted[sequence].count, actual[sequence].committed,
            "{what}: the committed rows"
        );
        assert_state_equal(&world, &alone[sequence], &batched[sequence], &what);
    }
    println!(
        "accepted drafts per sequence: {:?}",
        accepted
            .iter()
            .map(|accepted| accepted.accepted_drafts)
            .collect::<Vec<_>>()
    );
    assert!(
        accepted[0].accepted_drafts != accepted[1].accepted_drafts
            || accepted[1].accepted_drafts != accepted[2].accepted_drafts,
        "the sequences must accept different numbers of drafts"
    );
    drop(quantum);
    drop(runner);
    for run in alone.into_iter().chain(batched) {
        run.close(&world);
    }
    world.close();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_cancelled_member_stops_only_its_own_commit() {
    let _light = common::light();
    let Some(world) = World::open() else { return };
    let mut alone: Vec<Run> = PREFIXES.iter().map(|prefix| prefilled(&world, *prefix)).collect();
    let mut batched: Vec<Run> = PREFIXES.iter().map(|prefix| prefilled(&world, *prefix)).collect();
    let committed_before = batched[1].sequence.kv.committed();
    let flags: Vec<Arc<AtomicBool>> = (0..3).map(|_| Arc::new(AtomicBool::new(false))).collect();
    let wanted: Vec<Outcome> = [0usize, 2]
        .iter()
        .map(|sequence| alone[*sequence].step(&world, Step::Verify { rows: 8, commit: 8 }))
        .collect();
    let members: Vec<Member> = flags
        .iter()
        .map(|flag| Member {
            rows: 8,
            commit: Commit::Rows(8),
            cancel: Some(Arc::clone(flag)),
        })
        .collect();
    // The middle member's request is cancelled while the batch is in flight or before it: it ran either way.
    flags[1].store(true, Ordering::Release);
    let mut runner = Runner::new(&world);
    let (quantum, actual) = batch(&world, &mut runner, &mut batched, &members);
    assert_eq!(
        actual
            .iter()
            .map(|outcome| outcome.conclusion.clone())
            .collect::<Vec<_>>(),
        vec![Conclusion::Completed, Conclusion::Cancelled, Conclusion::Completed]
    );
    assert!(
        !quantum.stop_requested(),
        "one cancelled member does not stop the batch"
    );
    assert_eq!(actual[1].committed, 0);
    assert_eq!(
        batched[1].sequence.kv.committed(),
        committed_before,
        "the cancelled member commits nothing"
    );
    for (member, sequence) in [0usize, 2].iter().enumerate() {
        let what = format!("sequence {sequence}");
        assert_segment_equals(&actual[*sequence], &wanted[member], &what);
        assert_state_equal(&world, &alone[*sequence], &batched[*sequence], &what);
    }
    flags[0].store(true, Ordering::Release);
    flags[2].store(true, Ordering::Release);
    assert!(quantum.stop_requested(), "every member cancelled stops the batch");
    drop(quantum);
    drop(runner);
    for run in alone.into_iter().chain(batched) {
        // The cancelled member's rows were reserved and never committed: its sequence closes as it is.
        run.close(&world);
    }
    world.close();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_one_row_decode_beside_the_row_of_a_batch() {
    let _light = common::light();
    let Some(world) = World::open() else { return };
    let mut decoded = prefilled(&world, 300);
    let mut verified = prefilled(&world, 300);
    let member = prefilled(&world, 300);
    let other = prefilled(&world, 411);
    let alone = decoded.step(&world, Step::Decode);
    let exact = verified.step(&world, Step::Verify { rows: 1, commit: 1 });
    let mut runner = Runner::new(&world);
    let mut runs = [member, other];
    let (quantum, in_batch) = batch(&world, &mut runner, &mut runs, &[Member::all(1), Member::all(3)]);
    drop(quantum);
    // Plan question 2: a one-row decode is bit-identical to its row in a row-exact batch, and to a one-row verify.
    assert!(
        alone.logits == in_batch[0].logits,
        "a one-row decode equals its row in a row-exact batch"
    );
    assert!(
        exact.logits == in_batch[0].logits,
        "a one-row verification alone equals its row in a batch"
    );
    assert!(
        alone.logits == exact.logits,
        "a one-row decode equals a one-row verification"
    );
    assert_eq!(alone.keys, in_batch[0].keys, "the selected token is the same");
    drop(runner);
    let [member, other] = runs;
    for run in [decoded, verified, member, other] {
        run.close(&world);
    }
    world.close();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_batch_refuses_what_is_for_one_sequence_and_one_sequence_under_two_ids() {
    let _light = common::light();
    let Some(world) = World::open() else { return };
    let mode = SubmitMode {
        pdl: false,
        row_exact: true,
    };
    let make = |parts: Vec<Part>| {
        Quantum::batch(
            parts,
            Arc::clone(&world.workspace),
            mode,
            Arc::clone(&world.config),
            world.table.clone(),
        )
    };
    let mut first = prefilled(&world, 300);
    let mut second = prefilled(&world, 411);
    let (a, keep_a, _) = first.part(&world, 0, &Member::all(2));
    let (b, keep_b, _) = second.part(&world, 1, &Member::all(2));
    let quantum = make(vec![a, b]).unwrap();
    // The builders and accessors of one sequence refuse a quantum of two, whatever the build.
    assert!(quantum.selection().is_none());
    assert!(quantum.row_readback().is_none());
    assert!(quantum.table_generation(KvSource::Base, 0).is_none());
    let scored = quantum.with_logits_rows(1).unwrap();
    assert!(
        scored.validate().is_err(),
        "scoring a number of rows of one sequence on a batch is refused"
    );
    let mut third = prefilled(&world, 300);
    let mut fourth = prefilled(&world, 411);
    let (e, keep_e, _) = third.part(&world, 0, &Member::all(1));
    let (f, keep_f, _) = fourth.part(&world, 1, &Member::all(1));
    let quantum = make(vec![e, f]).unwrap();
    assert!(quantum.seeding_draft().is_err());
    // One sequence under two ids shares its device buffers.
    let mut same = prefilled(&world, 300);
    let (c, keep_c, _) = same.part(&world, 5, &Member::all(2));
    same.base_next += 2;
    let (d, keep_d, _) = same.part(&world, 6, &Member::all(2));
    assert!(matches!(
        make(vec![c, d]).err(),
        Some(QuantumError::DuplicateSequence {
            first: 0,
            second: 1,
            ..
        })
    ));
    drop((scored, keep_e, keep_f, keep_c, keep_d));
    for run in [third, fourth, same] {
        run.close(&world);
    }
    drop((keep_a, keep_b));
    first.close(&world);
    second.close(&world);
    world.close();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn each_member_of_a_batch_holds_its_own_count_in_flight() {
    let _light = common::light();
    let Some(world) = World::open() else { return };
    let mode = SubmitMode {
        pdl: false,
        row_exact: true,
    };
    let mut first = prefilled(&world, 300);
    let mut second = prefilled(&world, 411);
    let (a, keep_a, _) = first.part(&world, 0, &Member::all(2));
    let (b, keep_b, _) = second.part(&world, 1, &Member::all(2));
    let mut quantum = Quantum::batch(
        vec![a, b],
        Arc::clone(&world.workspace),
        mode,
        Arc::clone(&world.config),
        world.table.clone(),
    )
    .unwrap();
    assert_eq!((first.sequence.in_flight(), second.sequence.in_flight()), (1, 1));
    // The owners of a pass take their own guards out: one member ending does not wait for another's part.
    let mut guards = quantum.take_flights();
    assert_eq!(guards.len(), 2);
    let kept_by_second = guards.pop().unwrap();
    drop(guards);
    drop(quantum);
    assert_eq!(first.sequence.in_flight(), 0, "the first member's sequence settles");
    assert_eq!(second.sequence.in_flight(), 1, "the second still holds its own");
    drop(kept_by_second);
    assert_eq!(second.sequence.in_flight(), 0);
    drop((keep_a, keep_b));
    first.close(&world);
    second.close(&world);
    world.close();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn each_part_of_a_batch_keeps_its_own_sequences_place() {
    let _light = common::light();
    let Some(world) = World::open() else { return };
    let mode = SubmitMode {
        pdl: false,
        row_exact: true,
    };
    let mut first = prefilled(&world, 300);
    let mut second = prefilled(&world, 411);
    let (orders_a, orders_b) = (PlaceProbe::new(), PlaceProbe::new());
    // The owner's path: each verification takes a place in its sequence's order, becomes a part and joins the batch.
    let built_a = first.build(&world, Step::Verify { rows: 2, commit: 2 });
    let built_b = second.build(&world, Step::Verify { rows: 2, commit: 2 });
    let cancel = || Arc::new(AtomicBool::new(false));
    let a = built_a
        .quantum
        .with_probe_place(&orders_a)
        .into_part(SequenceId(0), cancel())
        .unwrap();
    let b = built_b
        .quantum
        .with_probe_place(&orders_b)
        .into_part(SequenceId(1), cancel())
        .unwrap();
    let mut quantum = Quantum::batch(
        vec![a, b],
        Arc::clone(&world.workspace),
        mode,
        Arc::clone(&world.config),
        world.table.clone(),
    )
    .unwrap();
    // The frame that ends the batch takes every segment's place out and completes each: the second's alone, then the
    // first's.
    let mut places = quantum.take_probe_places();
    assert_eq!(places.len(), 2);
    assert!(places.pop().unwrap().complete());
    assert_eq!(
        (orders_a.concluded(), orders_b.concluded()),
        (0, 1),
        "the second part's place is the second's"
    );
    assert!(places.pop().unwrap().complete());
    assert_eq!((orders_a.concluded(), orders_b.concluded()), (1, 1));
    assert!(
        quantum.take_probe_places().into_iter().all(|place| !place.complete()),
        "the places left the batch once"
    );
    drop(quantum);
    drop((built_a.retained, built_b.retained));
    first.close(&world);
    second.close(&world);
    world.close();
}
