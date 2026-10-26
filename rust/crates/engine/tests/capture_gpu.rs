//! Captured quanta on a GPU: the plan's stage graphs run on a lattice, with and without CUDA graphs. The tests are
//! `#[ignore]`d and run locally in a memory-capped scope:
//! `systemd-run --user --scope -p MemoryMax=16G cargo test -p euhedral-inference-engine --test capture_gpu -- --ignored`.
//!
//! The model is the first four layers of the RCO artifact (three GDN layers and one full-attention layer), its MTP
//! layer, heads and embedding, loaded object by object from the artifact named by `euhedral.qwen.rco-artifact`; the
//! library comes from `euhedral.cuda.library`. A test skips with a message when either is absent. The test thread is
//! the sequence's owner: it reserves the cache rows, uploads the input record, starts a quantum, waits for its
//! retirement and reads the results.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: the test thread waits on latches and copies device memory to check results"
)]
#![allow(clippy::print_stdout, reason = "a test reports what it measured")]

#[path = "common/linear_gpu.rs"]
mod linear_gpu;

mod common;

use std::collections::{BTreeSet, HashMap};
use std::mem;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, DeviceQueue, GraphRuntime, Join, Lake, LaneId, LanePool, Latch, Marker,
    QuantumOutcome, RetryCounts, UnitGuard,
};
use euhedral_inference_cuda::{
    CudaLane, CudaLibrary, DeviceBuffer, DevicePtr, PinnedPool, RetireList, Submission, SubmitMode, ops,
};
use euhedral_inference_engine::formats::{Nvfp4Linear, TrellisLinear};
use euhedral_inference_engine::lattice_config;
use euhedral_inference_engine::model::{
    LayerIndex, LayerKind, MTP_ATTENTION_PACK, ModelConfig, Role, WeightRef, object_key,
};
use euhedral_inference_engine::plan::{Formats, Plan, View, WeightLookup};
use euhedral_inference_engine::quantum::{Admission, CaptureKey, Executor, GraphCache};
use euhedral_inference_engine::residency::planner::{ResidencyPlan, nvfp4, plan as residency_plan};
use euhedral_inference_engine::stages::attention::KvSource;
use euhedral_inference_engine::stages::sampling::SelectionPool;
use euhedral_inference_engine::stages::{
    DraftSeed, DraftTokenWrite, InputRecordRef, LaneTable, Quantum, QuantumKind, SequenceRefs,
};
use euhedral_inference_engine::state::attention::AttentionScratch;
use euhedral_inference_engine::state::gdn::GdnStates;
use euhedral_inference_engine::state::kv::{KvCache, KvRetained};
use euhedral_inference_engine::state::{ReserveEpochError, SequenceDevice, SequenceState};
use euhedral_inference_engine::workspace::{BufferSlot, Workspace, WorkspaceRows};
use euhedral_inference_settings::{KvFormat, ModelSettings, Settings, TrellisKernelSettings};
use linear_gpu::Artifact;

/// Layers of the reduced model: the artifact's layers 0 to 3.
const LAYERS: usize = 4;
/// The most steps a world runs.
const MAX_STEPS: usize = 512;

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

fn upload(library: &Arc<CudaLibrary>, lane: &CudaLane, bytes: &[u8]) -> DeviceBuffer {
    let buffer = DeviceBuffer::alloc(library, bytes.len().max(256)).unwrap();
    let submission = Submission::begin(lane, SubmitMode::default()).unwrap();
    buffer.copy_from_host(0, bytes).unwrap();
    drop(submission);
    buffer
}

/// Loads the objects of the first `LAYERS` layers, the globals and the MTP layer from the artifact.
fn load_model(artifact: &mut Artifact, library: &Arc<CudaLibrary>, lane: &CudaLane) -> Model {
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
                let buffer = upload(library, lane, &out);
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

/// The reduced model, loaded once for the tests of this binary: its weights are read-only, so every test's world
/// reads the same buffers (each test still owns its library, lanes, lattice and workspace). It is never freed.
fn shared_model() -> Option<&'static Model> {
    static SHARED: OnceLock<Option<Model>> = OnceLock::new();
    SHARED
        .get_or_init(|| {
            let library = library()?;
            let mut artifact = Artifact::open()?;
            let loader = CudaLane::new(&library, LaneId(63), 16).unwrap();
            Some(load_model(&mut artifact, &library, &loader))
        })
        .as_ref()
}

/// A lattice, the lanes and the plan: what a test's quanta run on.
struct World {
    library: Arc<CudaLibrary>,
    config: Arc<ModelConfig>,
    residency: ResidencyPlan,
    plan: Arc<Plan>,
    /// The lane of the owner's own submissions: the reservations and the input records.
    owner: Arc<CudaLane>,
    table: LaneTable,
    pool: Arc<LanePool>,
    runtime: GraphRuntime,
    workspace: Arc<Workspace>,
    pinned: Arc<PinnedPool>,
    retire: Arc<RetireList>,
    lattice: Arc<ControlPlaneLattice>,
    lake: Arc<Lake>,
    unit: Option<UnitGuard>,
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
        let plan = Arc::new(Plan::build(&config, &model.weights, &residency, &formats).unwrap());
        let width = plan.fan_out_width();
        // The graphs' lanes, then the capture lane, then the owner's.
        let lanes: Vec<Arc<CudaLane>> = (0..width + 2)
            .map(|id| CudaLane::new(&library, LaneId(id as u16), 8192).unwrap())
            .collect();
        let owner = Arc::clone(&lanes[width + 1]);
        let table = LaneTable::new(lanes.clone()).unwrap();
        let pool = Arc::new(LanePool::new(
            lanes
                .iter()
                .map(|lane| Arc::clone(lane) as Arc<dyn DeviceQueue>)
                .collect(),
        ));
        let settings = Settings::with_model(ModelSettings::default());
        let lattice = ControlPlaneLattice::new(lattice_config(&settings)).unwrap();
        let lake = Lake::new(
            Arc::clone(&lattice),
            NonZeroUsize::new(2).unwrap(),
            Duration::from_secs(1),
        );
        let completions = Completions::new(Arc::clone(&pool));
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
        let workspace = Arc::new(Workspace::allocate(&library, &config, residency.rows, &regions).unwrap());
        let unit = lake.admit().unwrap();
        Some(World {
            pinned: PinnedPool::new(&library, 64 << 20),
            retire: Arc::new(RetireList::new()),
            library,
            config,
            residency,
            plan,
            owner,
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
    }

    /// Runs `work` as a submission on the owner's lane, and waits for it (a host copy).
    fn on_owner<T>(&self, work: impl FnOnce(&Submission<'_>) -> T) -> T {
        let submission = Submission::begin(&self.owner, SubmitMode::default()).unwrap();
        let out = work(&submission);
        drop(submission);
        self.owner.synchronize().unwrap();
        out
    }

    fn download(&self, ptr: DevicePtr, bytes: usize) -> Vec<u8> {
        let mut host = vec![0u8; bytes];
        self.on_owner(|submission| ops::copy_device_to_host(submission, &mut host, ptr).unwrap());
        host
    }

    fn sequence(&self) -> SequenceState {
        let config = &self.config;
        let gdn = self.on_owner(|submission| {
            let verified = (1 + self.residency.draft_depth) as u64;
            GdnStates::allocate(&self.library, config, verified, submission).unwrap()
        });
        let kv = |layers: usize| {
            KvCache::new(
                &self.library,
                &self.retire,
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

/// Where the retirements of one executor are handed to the test thread: step `i` waits on join `i`.
struct Steps {
    joins: Vec<OnceLock<Arc<Join<Retired>>>>,
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
    fn arm(&self, index: usize) -> euhedral_graph::LatchWaiter<Retired> {
        let (done, waiting) = Latch::new();
        let join = Join::new(1, move |mut parts: Vec<Option<Retired>>| {
            done.set(parts.remove(0).unwrap())
        });
        assert!(self.joins[index].set(join).is_ok());
        waiting
    }
}

/// What one step left behind.
#[derive(Debug, PartialEq, Eq)]
struct Outcome {
    keys: Vec<u64>,
    logits: Vec<u8>,
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
}

/// One sequence on a world: its state, its graph cache and the executor that runs it.
struct Run {
    sequence: SequenceState,
    executor: Executor,
    cache: GraphCache,
    steps: Arc<Steps>,
    started: usize,
    base_next: u64,
    mtp_next: u64,
    /// Rows each GDN layer replays first, from a partial acceptance.
    replay: Vec<u32>,
    ids: DeviceBuffer,
    /// Quanta that ran as one launch of a recorded graph: decode, verification, draft.
    replayed: [usize; 3],
    /// Quanta that recorded a graph beside their stages.
    recorded: usize,
    /// Quanta that had a capture key.
    keyed: usize,
}

/// The capture lane's id: the lane after the graphs' own.
fn capture_lane(world: &World) -> LaneId {
    LaneId(world.plan.fan_out_width() as u16)
}

impl Run {
    fn new(world: &World, cache: GraphCache) -> Run {
        let steps = Steps::new();
        let again = Arc::clone(&steps);
        let executor = Executor::new(
            Arc::clone(&world.plan),
            &world.runtime,
            capture_lane(world),
            move |quantum, outcome| {
                let index = again.retired.fetch_add(1, Ordering::AcqRel);
                again.joins[index].get().expect("armed").arrive(0, (quantum, outcome));
            },
        )
        .unwrap();
        let ids_bytes: Vec<u8> = (0..131_072u32).flat_map(|row| (row * 2).to_ne_bytes()).collect();
        let ids = {
            let buffer = DeviceBuffer::alloc(&world.library, ids_bytes.len()).unwrap();
            world.on_owner(|_| buffer.copy_from_host(0, &ids_bytes).unwrap());
            buffer
        };
        Run {
            sequence: world.sequence(),
            executor,
            cache,
            steps,
            started: 0,
            base_next: 0,
            mtp_next: 0,
            replay: Vec::new(),
            ids,
            replayed: [0; 3],
            recorded: 0,
            keyed: 0,
        }
    }

    /// Moves the base cache to `position` without computing anything: the rows are reserved and committed, their
    /// contents are whatever the device memory held.
    fn skip_to(&mut self, world: &World, position: u64) {
        let mut retained = KvRetained::default();
        world.on_owner(|submission| {
            let cache = &mut self.sequence.kv;
            let start = cache.submitted();
            cache
                .reserve(submission, &world.pinned, start, position - start, &mut retained)
                .unwrap();
            cache.appended(position - start).unwrap();
            cache.commit(position - start).unwrap();
        });
        // The reservation was this cache's call for the next quantum's references: claim it.
        drop(self.sequence.refs(KvSource::Base).unwrap());
        self.base_next = position;
        drop(retained);
    }

    /// Reserves, uploads the input record and builds the quantum of `step`; nothing runs yet.
    fn build(&mut self, world: &World, step: Step) -> Built {
        let config = &world.config;
        let (kind, rows) = match step {
            Step::Prefill(rows) => (QuantumKind::Prefill, rows),
            Step::Decode => (QuantumKind::Decode, 1),
            Step::Verify { rows, .. } => (QuantumKind::Verify, rows),
            Step::Draft => (QuantumKind::Draft, 1),
        };
        let drafting = kind == QuantumKind::Draft;
        let start = if drafting { self.mtp_next } else { self.base_next };
        // The owner reserves the rows the quantum appends, and records the marker of that work.
        let mut retained = KvRetained::default();
        let marker = {
            let submission = Submission::begin(&world.owner, SubmitMode::default()).unwrap();
            let cache = if drafting {
                self.sequence.mtp_kv.as_mut().unwrap()
            } else {
                &mut self.sequence.kv
            };
            cache
                .reserve(&submission, &world.pinned, start, rows as u64, &mut retained)
                .unwrap();
            cache.appended(rows as u64).unwrap();
            submission.finish_signalled().unwrap()
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
            .ordered_after(marker, &world.pool, &world.library)
            .unwrap()
            .with_replay_rows(replay);
        let mode = SubmitMode {
            pdl: matches!(kind, QuantumKind::Decode | QuantumKind::Verify) && start < 1024,
            row_exact: kind == QuantumKind::Verify,
        };
        let tokens: Vec<u32> = (0..rows as u64).map(|row| token(start + row)).collect();
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
        .with_tokens(tokens)
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
        let prepared = {
            let submission = Submission::begin(&world.owner, SubmitMode::default()).unwrap();
            quantum.prepare(&world.pinned, &submission).unwrap();
            submission.finish_signalled().unwrap()
        };
        quantum.prepared_after(prepared);
        Built {
            quantum,
            step,
            start,
            drafting,
            logits_rows,
            retained,
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
        } = built;
        let config = &world.config;
        self.keyed += usize::from(quantum.capture_key().is_some());
        let waiting = self.steps.arm(self.started);
        self.started += 1;
        let started = self
            .executor
            .start(&mut self.cache, quantum, Arc::new(AtomicBool::new(false)), None)
            .unwrap();
        let kind_index = match step {
            Step::Decode => Some(0),
            Step::Verify { .. } => Some(1),
            Step::Draft => Some(2),
            Step::Prefill(_) => None,
        };
        if let (true, Some(index)) = (started.replayed, kind_index) {
            self.replayed[index] += 1;
        }
        self.recorded += usize::from(started.recording);
        // Nothing is chained to this quantum: its reference is not held, or it would never retire.
        drop(started);
        let (quantum, outcome) = waiting.wait();
        assert_eq!(outcome, QuantumOutcome::Completed, "{step:?} at {start}");

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
                let ptr = world
                    .workspace
                    .buffer(BufferSlot::Logits)
                    .ptr()
                    .offset((row * row_bytes) as u64);
                out.extend(world.download(ptr, width * 2));
            }
            out
        };
        // The device work retired: the frontiers move.
        match step {
            Step::Prefill(rows) => {
                self.sequence.kv.commit(rows as u64).unwrap();
                self.base_next += rows as u64;
            }
            Step::Decode => {
                self.sequence.kv.commit(1).unwrap();
                self.base_next += 1;
            }
            Step::Verify { rows, commit } => {
                self.sequence.kv.commit_prefix(commit as u64).unwrap();
                self.base_next += commit as u64;
                if commit < rows {
                    // The state ran past the accepted rows: the next quantum replays them.
                    self.replay = config
                        .layer_kinds
                        .iter()
                        .map(|kind| {
                            if *kind == LayerKind::GatedDeltaNet {
                                commit as u32
                            } else {
                                0
                            }
                        })
                        .collect();
                }
            }
            Step::Draft => {
                self.sequence.mtp_kv.as_mut().unwrap().commit(1).unwrap();
                self.mtp_next += 1;
            }
        }
        drop(quantum);
        drop(retained);
        Outcome { keys, logits }
    }

    /// Ends the run: the sequence closes with no quantum in flight, and its graphs go with the cache.
    fn close(self, world: &World) {
        let Run { sequence, cache, .. } = self;
        drop(cache);
        sequence.close().unwrap();
        world.on_owner(|_| ());
    }

    /// The GDN states, layer by layer.
    fn gdn_state(&self, world: &World) -> Vec<Vec<u8>> {
        (0..LAYERS)
            .filter_map(|layer| self.sequence.device.gdn.layer(layer))
            .flat_map(|state| [state.convolution(), state.recurrent()])
            .map(|buffer| world.download(buffer.ptr(), buffer.bytes()))
            .collect()
    }
}

/// The steps both runs take: a prompt, decode, verifications of several widths with a partial acceptance, and drafts.
fn scenario() -> Vec<Step> {
    let mut steps = vec![Step::Prefill(16)];
    steps.extend([Step::Decode; 6]);
    steps.extend([Step::Verify { rows: 3, commit: 3 }; 4]);
    steps.extend([Step::Verify { rows: 5, commit: 2 }]);
    // The next quantum replays the two accepted rows first.
    steps.extend([Step::Decode; 4]);
    steps.extend([Step::Verify { rows: 5, commit: 5 }; 3]);
    steps.extend([Step::Verify { rows: 2, commit: 2 }; 3]);
    steps.extend([Step::Draft; 6]);
    steps.extend([Step::Decode; 3]);
    steps
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn replay_equals_stage_by_stage() {
    let Some(world) = World::open() else { return };
    assert!(
        world.plan.fan_out_width() >= 2,
        "the stage-by-stage quanta run on several lanes and must retire"
    );
    let mut captured = Run::new(&world, GraphCache::with_capacity(8));
    let mut reference = Run::new(&world, GraphCache::disabled());
    for (index, step) in scenario().into_iter().enumerate() {
        let wanted = reference.step(&world, step);
        let actual = captured.step(&world, step);
        if !matches!(step, Step::Prefill(_)) {
            assert!(
                wanted.keys.iter().all(|key| *key != 0),
                "step {index} {step:?}: the reference selected nothing"
            );
        }
        assert_eq!(actual.keys, wanted.keys, "step {index} {step:?}: the selected tokens");
        assert!(
            actual.logits == wanted.logits,
            "step {index} {step:?}: the logits are bitwise equal"
        );
    }
    assert!(
        reference.gdn_state(&world) == captured.gdn_state(&world),
        "the GDN states are bitwise equal"
    );
    assert_eq!(reference.replayed, [0; 3], "the reference never replays");
    assert_eq!(reference.recorded, 0, "the reference never records");
    // A key is run stage by stage on its first sight, recorded on its second (while that quantum runs) and replayed
    // from the third. Decode: 4 at positions 16 to 21, 3 after the replayed rows at 34 to 36, 1 at the next span.
    // Verification: rows 3 at 22 to 31 twice, rows 5 and rows 2 once each (the 5-row quanta cross a span). Drafts: the first reserves the MTP cache's first page and has no key, then 3 replays after the sight and the recording.
    let [decode, verify, draft] = captured.replayed;
    assert_eq!(
        [decode, verify, draft],
        [8, 3, 3],
        "replays of decode, verification and draft quanta"
    );
    // Decode at 17 and at 62, verification of 3 rows at 25, of 5 rows at 40 (its first sight was the quantum that
    // accepted two rows) and at 50, of 2 rows at 57, and the draft at 2.
    assert_eq!(captured.recorded, 7, "graphs recorded beside their second sights");
    // The sequence has passed the first span of decode and of the 5-row verification, whose graphs were released;
    // decode, 3 and 5 and 2 rows of verification and the draft keep theirs.
    assert_eq!(captured.cache.recorded(), 5, "graphs the cache holds at the end");
    assert!(captured.cache.evictions() >= 2, "the passed geometries were released");
    println!(
        "{} quanta, {} keyed, replayed {:?} (decode, verification, draft), {} recorded",
        scenario().len(),
        captured.keyed,
        captured.replayed,
        captured.recorded
    );
    captured.close(&world);
    reference.close(&world);
    world.close();
}

/// The recorded hash of the slot a quantum of the step would be admitted to, and the hash of what its stages would
/// submit now.
fn hashes(run: &mut Run, world: &World, step: Step) -> (u64, u64) {
    let built = run.build(world, step);
    let key = built.quantum.capture_key().expect("the quantum has a key");
    let Admission::Replay(slot) = run.cache.admit(Some(key)) else {
        panic!("the key has a recorded graph")
    };
    let now = run
        .executor
        .stages_hash_for_tests(&built.quantum, world.owner.as_ref())
        .unwrap();
    // The quantum is not run: its reservation is undone.
    run.sequence.kv.discard();
    if let Some(mtp) = run.sequence.mtp_kv.as_mut() {
        mtp.discard();
    }
    (slot.recorded_hash(), now)
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn check_mode_confirms_replay_hash() {
    let Some(world) = World::open() else { return };
    let mut run = Run::new(&world, GraphCache::with_capacity(8));
    run.step(&world, Step::Prefill(16));
    for _ in 0..4 {
        run.step(&world, Step::Decode);
    }
    let (recorded, now) = hashes(&mut run, &world, Step::Decode);
    assert_ne!(recorded, 0, "the recording kept the hash of what it captured");
    assert_eq!(
        recorded, now,
        "the graph repeats what the stages would submit at the next position"
    );
    run.close(&world);
    world.close();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn check_mode_covers_a_verification_twin_the_fused_verifier_and_deep_decode() {
    let Some(world) = World::open() else { return };
    let mut run = Run::new(&world, GraphCache::with_capacity(8));
    // Past 2048 keys: decode takes the three-warp GQA kernel, 3 rows the row twin (the fused verifier needs 16384
    // keys), 5 rows the fused verifier (from 2048 keys).
    run.skip_to(&world, 2100);
    for _ in 0..4 {
        run.step(&world, Step::Decode);
    }
    for rows in [3usize, 5] {
        for _ in 0..4 {
            run.step(&world, Step::Verify { rows, commit: rows });
        }
    }
    let position = run.base_next;
    assert!(position > 2048);
    for step in [
        Step::Decode,
        Step::Verify { rows: 3, commit: 3 },
        Step::Verify { rows: 5, commit: 5 },
    ] {
        let (recorded, now) = hashes(&mut run, &world, step);
        assert_ne!(recorded, 0, "{step:?}");
        assert_eq!(recorded, now, "{step:?} at {position}: the graph repeats the stages");
    }
    // The oracle can tell: the stages of another step do not hash as the graph of this one.
    let (recorded, _) = hashes(&mut run, &world, Step::Decode);
    let (_, other) = hashes(&mut run, &world, Step::Verify { rows: 3, commit: 3 });
    assert_ne!(recorded, other);
    run.close(&world);
    world.close();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_graph_is_replayed_only_after_its_recording_finished() {
    let Some(world) = World::open() else { return };
    let mut run = Run::new(&world, GraphCache::with_capacity(8));
    run.step(&world, Step::Prefill(16));
    let first = run.build(&world, Step::Decode);
    run.run(&world, first);
    assert_eq!(
        (run.recorded, run.replayed[0]),
        (0, 0),
        "the first sight runs stage by stage"
    );
    run.step(&world, Step::Decode);
    assert_eq!(
        (run.recorded, run.replayed[0]),
        (1, 0),
        "the second records beside its stages and is not replayed"
    );
    // Retired quanta have their record stage done: the third replays.
    run.step(&world, Step::Decode);
    assert_eq!((run.recorded, run.replayed[0]), (1, 1));
    run.close(&world);
    world.close();
}

/// The device memory a recorded graph holds: the free device memory before and after the quantum that records it, for
/// twenty geometries of decode and of a 3-row verification (the model of this test has four layers; the 64-layer
/// model's graphs have about 16 times the nodes). `cudaMemGetInfo` counts in steps of 2 MiB, so the mean over many
/// graphs is the figure. Run alone: any other process's allocations are in the difference.
#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact; run alone"]
fn a_recorded_graph_holds_device_memory_that_is_measured() {
    const GEOMETRIES: u64 = 20;
    let Some(world) = World::open() else { return };
    let free = || world.library.free_bytes().unwrap() as i64;
    let mut report = Vec::new();
    for (name, step) in [
        ("decode", Step::Decode),
        ("verification of 3 rows", Step::Verify { rows: 3, commit: 3 }),
    ] {
        let mut run = Run::new(&world, GraphCache::with_capacity(8));
        run.step(&world, Step::Prefill(16));
        // Warm: the kernels this view launches are loaded by the first quanta.
        for _ in 0..3 {
            run.step(&world, step);
        }
        let mut held = 0i64;
        for geometry in 1..=GEOMETRIES {
            // A position inside a span of 48 keys and off every page boundary (a quantum that grows the cache is not recorded).
            let mut position = 48 * geometry + 60;
            if position % 256 > 249 {
                position += 8;
            }
            run.skip_to(&world, position);
            run.step(&world, step);
            let before = free();
            run.step(&world, step);
            held += before - free();
            assert!(
                run.recorded as u64 > geometry,
                "{name}: the second quantum of a geometry records"
            );
        }
        report.push(format!(
            "{name}: {} KiB a graph on average over {GEOMETRIES} graphs",
            held / 1024 / GEOMETRIES as i64
        ));
        run.close(&world);
    }
    println!(
        "device memory a recorded graph holds ({} stages in the 4-layer model's decode view): {}",
        world.plan.stages(View::Decode).map_or(0, <[_]>::len),
        report.join("; ")
    );
    world.close();
}

/// A light world: a lane, a workspace and a sequence, no weights and no lattice.
struct Light {
    library: Arc<CudaLibrary>,
    lane: Arc<CudaLane>,
    pool: LanePool,
    table: LaneTable,
    pinned: Arc<PinnedPool>,
    retire: Arc<RetireList>,
    config: Arc<ModelConfig>,
    workspace: Arc<Workspace>,
}

fn light() -> Option<Light> {
    let library = library()?;
    let config = reduced_config();
    let lane = CudaLane::new(&library, LaneId(0), 4).unwrap();
    let workspace = Arc::new(
        Workspace::allocate(
            &library,
            &config,
            WorkspaceRows {
                rows: 8,
                logits_rows: 8,
            },
            &[],
        )
        .unwrap(),
    );
    Some(Light {
        table: LaneTable::new(vec![Arc::clone(&lane)]).unwrap(),
        pool: LanePool::new(vec![Arc::clone(&lane) as Arc<dyn DeviceQueue>]),
        pinned: PinnedPool::new(&library, 16 << 20),
        retire: Arc::new(RetireList::new()),
        library,
        lane,
        config,
        workspace,
    })
}

impl Light {
    fn sequence(&self, kv_tokens: u64, layers: usize) -> SequenceState {
        let submission = Submission::begin(&self.lane, SubmitMode::default()).unwrap();
        let gdn = GdnStates::allocate(&self.library, &self.config, 8, &submission).unwrap();
        submission.finish_signalled().unwrap();
        self.lane.synchronize().unwrap();
        let kv = KvCache::new(
            &self.library,
            &self.retire,
            layers,
            self.config.kv_width() as u64,
            KvFormat::Nvfp4,
            kv_tokens,
            0,
        )
        .unwrap();
        let scratch = AttentionScratch::allocate(&self.library, &self.config, 8, true, 16).unwrap();
        SequenceState::from_parts(
            kv,
            None,
            Arc::new(SequenceDevice { gdn, scratch }),
            SelectionPool::new(&self.library, 2, 8).unwrap(),
        )
    }

    /// The quantum of `kind` over `rows` rows at `start`, after the one reservation of its cache.
    fn quantum(
        &self,
        sequence: &mut SequenceState,
        kind: QuantumKind,
        rows: usize,
        start: u64,
        row_exact_mode: bool,
    ) -> Quantum {
        let mut retained = KvRetained::default();
        let submission = Submission::begin(&self.lane, SubmitMode::default()).unwrap();
        sequence
            .kv
            .reserve(&submission, &self.pinned, start, rows as u64, &mut retained)
            .unwrap();
        sequence.kv.appended(rows as u64).unwrap();
        drop(submission);
        self.lane.synchronize().unwrap();
        let quantum = Quantum::new(
            kind,
            rows,
            InputRecordRef::new(start),
            Arc::clone(&self.workspace),
            sequence.refs(KvSource::Base).unwrap(),
            SubmitMode {
                pdl: false,
                row_exact: row_exact_mode,
            },
            Arc::clone(&self.config),
            self.table.clone(),
        )
        .unwrap();
        quantum.with_selection_lease(sequence.selections.acquire().unwrap())
    }
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn page_boundary_quantum_not_captured() {
    let Some(light) = light() else { return };
    let mut sequence = light.sequence(1024, 1);
    let mut keys = Vec::new();
    for start in 0..260u64 {
        let quantum = light.quantum(&mut sequence, QuantumKind::Decode, 1, start, false);
        keys.push(quantum.capture_key());
        sequence.kv.commit(1).unwrap();
    }
    assert!(keys[0].is_none(), "the first reservation allocates the first page");
    assert!(
        keys[1..256].iter().all(Option::is_some),
        "positions 1 to 255 fit the reserved page"
    );
    assert!(
        keys[256].is_none(),
        "position 256 needs a new page and a longer page table"
    );
    assert!(
        keys[257..].iter().all(Option::is_some),
        "the next quantum is captured again"
    );
    assert_ne!(
        keys[255].as_ref().unwrap().buffer_generations,
        keys[257].as_ref().unwrap().buffer_generations,
        "the grown page table is another allocation, so the graph of the first page is not reused"
    );
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn the_capture_key_of_a_quantum_changes_every_48_positions_below_2048_and_is_constant_from_there() {
    let Some(light) = light() else { return };
    let mut sequence = light.sequence(4096, 1);
    // One reservation of the first 4000 rows, then quanta over them, one reservation each (as the owner makes).
    let mut keys = Vec::new();
    for start in 0..2200u64 {
        let quantum = light.quantum(&mut sequence, QuantumKind::Decode, 1, start, false);
        keys.push(quantum.capture_key());
        sequence.kv.commit(1).unwrap();
    }
    // Leave out the quanta at a page boundary or table growth: they have no key.
    let keyed: Vec<(u64, CaptureKey)> = keys
        .into_iter()
        .enumerate()
        .filter_map(|(start, key)| Some((start as u64, key?)))
        .collect();
    let geometry = |start: u64| {
        keyed
            .iter()
            .find(|(at, _)| *at == start)
            .map(|(_, key)| key.split_geometry)
    };
    // Below 2048 keys the split counts follow the position in spans of 48.
    for start in 1..2046u64 {
        if let (Some(a), Some(b)) = (geometry(start), geometry(start + 1)) {
            assert_eq!(a != b, (start + 1) % 48 == 0, "positions {start} and {}", start + 1);
        }
    }
    // From 2048 keys the geometry is one value.
    let beyond: BTreeSet<u64> = keyed
        .iter()
        .filter(|(start, _)| *start >= 2047)
        .map(|(_, key)| key.split_geometry)
        .collect();
    assert_eq!(beyond.len(), 1, "{beyond:?}");
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn the_row_exact_mode_is_part_of_the_capture_key() {
    let Some(light) = light() else { return };
    let mut sequence = light.sequence(1024, 1);
    let first = light.quantum(&mut sequence, QuantumKind::Decode, 1, 0, false);
    assert!(first.capture_key().is_none(), "the first reservation allocates");
    drop(first);
    sequence.kv.commit(1).unwrap();
    let decode = light.quantum(&mut sequence, QuantumKind::Decode, 1, 1, false);
    let decode = decode
        .capture_key()
        .expect("a decode quantum inside the page has a key");
    sequence.kv.commit(1).unwrap();
    let verify = light.quantum(&mut sequence, QuantumKind::Verify, 3, 2, true);
    let verify = verify.capture_key().expect("a verification inside the page has a key");
    assert!(!decode.row_exact && verify.row_exact);
    // The same key but for the mode is another graph: the linears take other routes in a row-exact launch.
    let mut other = verify.clone();
    other.row_exact = false;
    assert_ne!(verify, other);
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_selection_is_held_by_its_quantum_until_the_quantum_is_dropped() {
    let Some(light) = light() else { return };
    let pool = SelectionPool::new(&light.library, 2, 8).unwrap();
    let first = pool.acquire().unwrap();
    let second = pool.acquire().unwrap();
    assert!(!Arc::ptr_eq(first.selection(), second.selection()));
    assert!(pool.acquire().is_err(), "every selection is held: the admission waits");
    assert_eq!(pool.leased(), 2);
    let kept = Arc::clone(first.selection());
    drop(first);
    let again = pool.acquire().unwrap();
    assert!(
        Arc::ptr_eq(again.selection(), &kept),
        "the released selection is the one given again"
    );
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_reservation_marker_is_kept_in_the_ring_and_dropped_when_reached() {
    let Some(light) = light() else { return };
    let mut sequence = light.sequence(1024, 1);
    let mut retained = KvRetained::default();
    let submission = Submission::begin(&light.lane, SubmitMode::default()).unwrap();
    sequence
        .kv
        .reserve(&submission, &light.pinned, 0, 8, &mut retained)
        .unwrap();
    let reserved = submission.finish().unwrap();
    // The ring holds 4 events: the reservation's marker is stale after 4 more.
    for _ in 0..3 {
        Submission::begin(&light.lane, SubmitMode::default())
            .unwrap()
            .finish()
            .unwrap();
    }
    let view = sequence.kv.view();
    let kept = view.clone().ordered_after(reserved, &light.pool, &light.library);
    assert_eq!(kept.unwrap().ready(), Some(reserved), "within the ring");
    for _ in 0..2 {
        Submission::begin(&light.lane, SubmitMode::default())
            .unwrap()
            .finish()
            .unwrap();
    }
    assert!(!light.library.marker_waitable(reserved));
    let refused = view.clone().ordered_after(reserved, &light.pool, &light.library);
    assert!(
        refused.is_err(),
        "older than the ring and not reached: the owner holds the quantum back"
    );
    let marker: Marker = Submission::begin(&light.lane, SubmitMode::default())
        .unwrap()
        .finish_signalled()
        .unwrap();
    light.lane.synchronize().unwrap();
    assert!(light.pool.reached(marker));
    let done = view.ordered_after(reserved, &light.pool, &light.library).unwrap();
    assert_eq!(done.ready(), None, "reached work needs no ordering");
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_cache_reserved_twice_or_not_at_all_for_a_quantum_is_refused_its_references() {
    let Some(light) = light() else { return };
    let mut sequence = light.sequence(1024, 1);
    assert_eq!(
        sequence.refs(KvSource::Base).unwrap_err(),
        ReserveEpochError::Calls {
            cache: KvSource::Base,
            calls: 0
        },
        "no reservation yet"
    );
    let mut retained = KvRetained::default();
    let reserve = |sequence: &mut SequenceState, retained: &mut KvRetained, start: u64| {
        let submission = Submission::begin(&light.lane, SubmitMode::default()).unwrap();
        sequence
            .kv
            .reserve(&submission, &light.pinned, start, 1, retained)
            .unwrap();
        sequence.kv.appended(1).unwrap();
        drop(submission);
        light.lane.synchronize().unwrap();
        sequence.kv.commit(1).unwrap();
    };
    reserve(&mut sequence, &mut retained, 0);
    assert!(sequence.refs(KvSource::Base).is_ok(), "one reservation, one quantum");
    reserve(&mut sequence, &mut retained, 1);
    reserve(&mut sequence, &mut retained, 2);
    assert!(
        matches!(
            sequence.refs(KvSource::Base),
            Err(ReserveEpochError::Calls { calls: 2, .. })
        ),
        "two reservations for one quantum: the view of the first says nothing about the second"
    );
    assert!(sequence.refs(KvSource::Mtp).is_err(), "this sequence has no MTP cache");
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_sequence_closes_only_when_no_quantum_is_in_flight() {
    let Some(light) = light() else { return };
    let mut sequence = light.sequence(1024, 1);
    let mut retained = KvRetained::default();
    let submission = Submission::begin(&light.lane, SubmitMode::default()).unwrap();
    sequence
        .kv
        .reserve(&submission, &light.pinned, 0, 1, &mut retained)
        .unwrap();
    drop(submission);
    light.lane.synchronize().unwrap();
    let refs = sequence.refs(KvSource::Base).unwrap();
    assert_eq!(sequence.in_flight(), 1);
    let busy = sequence.close().unwrap_err();
    assert_eq!(busy.in_flight, 1);
    drop(refs);
    assert!(
        busy.state.close().is_ok(),
        "once the quantum is dropped the caches are freed"
    );
}

/// How long one page of growth takes at 131072 tokens with no view held and with one, and the median over `ROUNDS`
/// pairs: the cost of the copy of the page store (a handle per page of every layer) that growing after a view was
/// taken makes.
#[test]
#[ignore = "needs a GPU and the native library"]
fn growing_a_cache_after_a_view_was_taken_costs_one_handle_copy_per_page() {
    const ROUNDS: usize = 6;
    let Some(light) = light() else { return };
    // 16 attention layers and 131072 tokens: 512 pages a layer.
    let layers = 16;
    let mut kv = KvCache::new(
        &light.library,
        &light.retire,
        layers,
        light.config.kv_width() as u64,
        KvFormat::Nvfp4,
        131_072,
        0,
    )
    .unwrap();
    let mut retained = KvRetained::default();
    let submission = Submission::begin(&light.lane, SubmitMode::default()).unwrap();
    // Reserve all but the pages the rounds grow into.
    let almost = 131_072 - (2 * ROUNDS as u64 + 2) * 256;
    kv.reserve(&submission, &light.pinned, 0, almost, &mut retained)
        .unwrap();
    kv.appended(almost).unwrap();
    drop(submission);
    light.lane.synchronize().unwrap();
    let grow = |kv: &mut KvCache, retained: &mut KvRetained, held: bool| -> Duration {
        let view = held.then(|| kv.view());
        let start = kv.submitted();
        let submission = Submission::begin(&light.lane, SubmitMode::default()).unwrap();
        let begun = Instant::now();
        kv.reserve(&submission, &light.pinned, start, 256, retained).unwrap();
        let took = begun.elapsed();
        kv.appended(256).unwrap();
        drop(submission);
        light.lane.synchronize().unwrap();
        drop(view);
        took
    };
    let (mut without, mut with) = (Vec::new(), Vec::new());
    for _ in 0..ROUNDS {
        without.push(grow(&mut kv, &mut retained, false));
        with.push(grow(&mut kv, &mut retained, true));
    }
    without.sort();
    with.sort();
    let median = |times: &[Duration]| times[times.len() / 2];
    let handles = (layers * (kv.device_pages())) as f64;
    println!(
        "growth of one page at 131072 tokens ({layers} layers, {} handles): median {:?} with no view held, {:?} with a view          held; the copy of the page store costs {:?} ({:.1} ns a handle)",
        handles,
        median(&without),
        median(&with),
        median(&with).saturating_sub(median(&without)),
        median(&with).saturating_sub(median(&without)).as_nanos() as f64 / handles,
    );
}
