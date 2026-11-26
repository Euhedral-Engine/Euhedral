//! Quanta that end before their stages do, on a GPU: a quantum that is killed or fails part-way must still retire.
//! The tests are `#[ignore]`d and run locally in a memory-capped scope:
//! `systemd-run --user --scope -p MemoryMax=16G cargo test -p euhedral-inference-engine --test device -- --ignored retire::`.
//!
//! The shapes are synthetic: stages that queue nothing but their markers, on lanes of their own, in graphs the test
//! builds, and the plan's own graphs over weights with no memory behind them (a quantum that is killed never launches
//! them). No artifact is loaded. The library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY`
//! in the environment, in `~/.config/euhedral/test.properties`, or `gradle.properties`); a test skips with a message when it is absent. Every wait for a quantum
//! to retire has a deadline, so a quantum that never retires fails its test and does not hang the suite.

#![allow(clippy::print_stderr, reason = "a test reports what it skipped")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: the test thread waits for retirements against a deadline and synchronizes lanes"
)]

use crate::common;
use crate::typed;

use std::collections::HashMap;
use std::fmt::Display;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::{FrameSink, LatticeSource};
use typed::Typed;

use euhedral_gpu::cuda::CudaKernel;
use euhedral_gpu::cuda::CudaModule;
use euhedral_gpu::cuda::CudaQueue;
use euhedral_gpu::{Args, CompileOptions, GraphStats, LaneEnqueue, Launch, Module};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, DeviceQueue, EdgeKind, ExternalEdge, GraphRuntime, Lake, LaneId, Marker,
    QuantumOutcome, RetryCounts, Stage, StageCx, StageError, StageGraph, StageGraphSpec, StageId, StageSpec, Submitted,
    UnitGuard,
};
use euhedral_inference_cuda::{CudaError, CudaLibrary, DevicePtr, KernelOps, LaneOps, PinnedPool, SubmitMode};
use euhedral_inference_engine::formats::{Nvfp4Linear, TrellisLinear};
use euhedral_inference_engine::kernels::kernel_sources;
use euhedral_inference_engine::lattice_config;
use euhedral_inference_engine::model::{LayerIndex, LayerKind, ModelConfig, Residence, Role, WeightRef, object_key};
use euhedral_inference_engine::plan::{Formats, Plan, WeightLookup};
use euhedral_inference_engine::quantum::Executor;
use euhedral_inference_engine::residency::planner::plan as residency_plan;
use euhedral_inference_engine::stages::attention::KvSource;
use euhedral_inference_engine::stages::sampling::SelectionPool;
use euhedral_inference_engine::stages::{
    InputRecordRef, LaneTable, Placed, Quantum, QuantumKind, StageLaunch, submit_on,
};
use euhedral_inference_engine::state::attention::AttentionScratch;
use euhedral_inference_engine::state::gdn::GdnStates;
use euhedral_inference_engine::state::kv::{KvCache, KvRetained};
use euhedral_inference_engine::state::released::ReleasedBuffers;
use euhedral_inference_engine::state::{SequenceDevice, SequenceState};
use euhedral_inference_engine::workspace::{Workspace, WorkspaceRows};
use euhedral_inference_settings::{KvFormat, ModelSettings, Settings, TrellisKernelSettings};

/// How long a quantum may take to retire. The work is a few empty submissions; a quantum that has not retired by now
/// never will.
const DEADLINE: Duration = Duration::from_secs(20);

/// Layers of the reduced model: the artifact's layers 0 to 3.
const LAYERS: usize = 4;

fn library() -> Option<Arc<CudaLibrary>> {
    let path = test_support::require_property("euhedral.cuda.library")?;
    Some(CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads"))
}

/// The reduced model's dimensions: the RCO model's first four layers.
fn reduced_config() -> Arc<ModelConfig> {
    let mut config = common::config();
    config.layer_kinds.truncate(LAYERS);
    config.num_hidden_layers = LAYERS;
    assert_eq!(config.layer_kinds[3], LayerKind::FullAttention);
    Arc::new(config)
}

// How the quanta ended.

const PENDING: u8 = 0;
const COMPLETED: u8 = 1;
const FAILED: u8 = 2;
const KILLED: u8 = 3;

/// The outcome of each quantum, by the start position of its rows (the tests give each quantum its own).
struct Retirements(Vec<AtomicU8>);

impl Retirements {
    fn new() -> Arc<Self> {
        Arc::new(Self((0..64).map(|_| AtomicU8::new(PENDING)).collect()))
    }

    fn retire(&self, quantum: &Quantum, outcome: &QuantumOutcome) {
        let code = match outcome {
            QuantumOutcome::Completed => COMPLETED,
            QuantumOutcome::Failed { .. } | QuantumOutcome::DeviceFailed { .. } => FAILED,
            QuantumOutcome::Killed => KILLED,
        };
        self.0[quantum.start_position() as usize].store(code, Ordering::Release);
    }

    /// The outcome of the quantum at `start`, once it retired; `None` when it has not within the deadline.
    fn outcome(&self, start: usize) -> Option<u8> {
        let begun = Instant::now();
        loop {
            let code = self.0[start].load(Ordering::Acquire);
            if code != PENDING {
                return Some(code);
            }
            if begun.elapsed() > DEADLINE {
                return None;
            }
            thread::sleep(Duration::from_millis(1));
        }
    }

    fn assert_outcome(&self, start: usize, expected: u8, what: &str) {
        let outcome = self.outcome(start);
        assert_eq!(
            outcome,
            Some(expected),
            "{what}: the quantum at {start} should have retired as {expected} within {DEADLINE:?} (None: it never did)"
        );
    }
}

// The world.

/// A library, lanes, a lattice and the sequence's pieces: no weights.
struct World {
    library: Arc<CudaLibrary>,
    typed: Typed,
    table: LaneTable,
    runtime: GraphRuntime,
    config: Arc<ModelConfig>,
    workspace: Arc<Workspace>,
    pinned: Arc<PinnedPool>,
    released: Arc<ReleasedBuffers>,
    lattice: Arc<ControlPlaneLattice>,
    lake: Arc<Lake>,
    unit: Option<UnitGuard>,
}

impl World {
    /// `lane_count` lanes, numbered from 0; the last is the owner's, for the sequence's own submissions.
    fn open(lane_count: usize) -> Option<World> {
        let library = library()?;
        let config = reduced_config();
        let typed = Typed::plain(&library, lane_count, 1024);
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
            lanes: pool,
            retries: Arc::new(RetryCounts::default()),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        };
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
        let unit = lake.admit().unwrap();
        Some(World {
            pinned: PinnedPool::new(&library, 16 << 20),
            released: Arc::new(ReleasedBuffers::new()),
            library,
            typed,
            table,
            runtime,
            config,
            workspace,
            lattice,
            lake,
            unit: Some(unit),
        })
    }

    /// The lane of the sequence's own submissions.
    fn owner(&self) -> &Arc<CudaQueue> {
        self.typed.lane(self.typed.table.len() as u16 - 1)
    }

    /// Queues `work` as one submission of the owner's lane and waits for it.
    fn on_owner<E: Display>(&self, work: impl FnOnce(&mut dyn LaneOps) -> Result<(), E>) {
        self.table.submit_ops(self.owner(), false, work).unwrap();
        self.typed.sync();
    }

    fn close(mut self) {
        drop(self.unit.take());
        self.lake.complete_gracefully();
        self.lattice.close();
        self.typed.sync();
    }

    fn sequence(&self, kv_tokens: u64) -> SequenceState {
        let mut gdn = None;
        self.on_owner(|lane| {
            gdn = Some(GdnStates::allocate(&self.library, &self.config, 8, lane)?);
            Ok::<(), CudaError>(())
        });
        let gdn = gdn.unwrap();
        let kv = KvCache::new(
            &self.library,
            &self.released,
            1,
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
            SelectionPool::new(&self.library, 4, 8).unwrap(),
        )
    }

    /// The decode quantum at `start`, after the one reservation of its cache row.
    fn quantum(&self, sequence: &mut SequenceState, start: u64) -> Quantum {
        let mut retained = KvRetained::default();
        self.on_owner(|lane| sequence.kv.reserve(lane, &self.pinned, start, 1, &mut retained));
        sequence.kv.appended(1).unwrap();
        let quantum = Quantum::new(
            QuantumKind::Decode,
            1,
            InputRecordRef::new(start),
            Arc::clone(&self.workspace),
            sequence.refs(KvSource::Base).unwrap(),
            SubmitMode::default(),
            Arc::clone(&self.config),
            self.table.clone(),
        )
        .unwrap();
        quantum.with_selection_lease(sequence.selections.acquire().unwrap())
    }
}

// Synthetic graphs.

/// What a probe does in its launch.
#[derive(Clone)]
enum Act {
    Nothing,
    /// Sets the quantum's kill switch.
    Kill(Arc<AtomicBool>),
    /// Refuses the quantum.
    Fail,
    /// Panics.
    Panic,
}

/// A stage's launch that queues nothing: the stage submits only its marker. It counts its runs in `ran[index]`.
struct Probe {
    act: Act,
    ran: Arc<Vec<AtomicUsize>>,
    index: usize,
}

impl StageLaunch for Probe {
    fn idle(&self, _quantum: &Quantum) -> bool {
        false
    }

    fn waits(&self, _quantum: &Quantum) -> Vec<Marker> {
        Vec::new()
    }

    fn launch(&self, _quantum: &Quantum, _kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        self.ran[self.index].fetch_add(1, Ordering::AcqRel);
        match &self.act {
            Act::Nothing => Ok(()),
            Act::Kill(kill) => {
                kill.store(true, Ordering::Release);
                Ok(())
            }
            Act::Fail => Err(StageError::Fatal("the probe refuses".to_string())),
            Act::Panic => panic!("the probe panics"),
        }
    }
}

/// One stage of a chain: the lane it is pinned to, and what it does.
struct Link {
    lane: u16,
    act: Act,
}

fn link(lane: u16, act: Act) -> Link {
    Link { lane, act }
}

/// A chain of stages, each behind the one before it, and how many of them ran.
fn chain(
    world: &World,
    links: Vec<Link>,
    retirements: &Arc<Retirements>,
) -> (Arc<StageGraph<Quantum>>, Arc<Vec<AtomicUsize>>) {
    let ran: Arc<Vec<AtomicUsize>> = Arc::new((0..links.len()).map(|_| AtomicUsize::new(0)).collect());
    let mut spec = StageGraphSpec::new();
    let mut stages: Vec<Arc<dyn Stage<Quantum>>> = Vec::new();
    let mut previous: Option<StageId> = None;
    for (index, link) in links.iter().enumerate() {
        let stage = spec.stage(StageSpec {
            name: "probe",
            ordered: None,
            lanes: vec![LaneId(link.lane)],
        });
        if let Some(previous) = previous {
            spec.edge(previous, stage, EdgeKind::Submission);
        }
        previous = Some(stage);
        let launch = Arc::new(Probe {
            act: link.act.clone(),
            ran: Arc::clone(&ran),
            index,
        });
        stages.push(Arc::new(Placed::new(launch)));
    }
    let retirements = Arc::clone(retirements);
    let graph = StageGraph::new(spec, stages, world.runtime.clone(), move |quantum, outcome| {
        retirements.retire(&quantum, &outcome)
    })
    .unwrap();
    (graph, ran)
}

fn ran_stages(ran: &[AtomicUsize]) -> Vec<usize> {
    ran.iter().map(|count| count.load(Ordering::Acquire)).collect()
}

fn not_killed() -> Arc<AtomicBool> {
    Arc::new(AtomicBool::new(false))
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_quantum_killed_after_its_first_stage_retires() {
    let _light = common::light();
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let kill = not_killed();
    // The first stage submits its marker and sets the kill switch; the stages behind it never run.
    let (graph, ran) = chain(
        &world,
        vec![
            link(0, Act::Kill(Arc::clone(&kill))),
            link(0, Act::Nothing),
            link(0, Act::Nothing),
        ],
        &retirements,
    );
    let mut sequence = world.sequence(1024);
    let quantum = world.quantum(&mut sequence, 0);
    let _ = graph.start(quantum, kill, Vec::new());
    retirements.assert_outcome(0, KILLED, "killed after its first stage");
    assert_eq!(ran_stages(&ran), vec![1, 0, 0], "only the first stage ran");
    world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_quantum_killed_with_markers_on_two_lanes_retires() {
    let _light = common::light();
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let kill = not_killed();
    // Lane 0 holds the first stage's marker and lane 1 the second's when the second sets the switch.
    let (graph, ran) = chain(
        &world,
        vec![
            link(0, Act::Nothing),
            link(1, Act::Kill(Arc::clone(&kill))),
            link(0, Act::Nothing),
            link(1, Act::Nothing),
        ],
        &retirements,
    );
    let mut sequence = world.sequence(1024);
    let quantum = world.quantum(&mut sequence, 0);
    let _ = graph.start(quantum, kill, Vec::new());
    retirements.assert_outcome(0, KILLED, "killed with markers on two lanes");
    assert_eq!(ran_stages(&ran), vec![1, 1, 0, 0]);
    world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_quantum_killed_before_its_first_stage_retires() {
    let _light = common::light();
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let (graph, ran) = chain(&world, vec![link(0, Act::Nothing), link(0, Act::Nothing)], &retirements);
    let mut sequence = world.sequence(1024);
    let quantum = world.quantum(&mut sequence, 0);
    let _ = graph.start(quantum, Arc::new(AtomicBool::new(true)), Vec::new());
    retirements.assert_outcome(0, KILLED, "killed before it started");
    assert_eq!(ran_stages(&ran), vec![0, 0]);
    world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_quantum_whose_middle_stage_fails_retires() {
    let _light = common::light();
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let (graph, ran) = chain(
        &world,
        vec![link(0, Act::Nothing), link(0, Act::Fail), link(0, Act::Nothing)],
        &retirements,
    );
    let mut sequence = world.sequence(1024);
    let quantum = world.quantum(&mut sequence, 0);
    let _ = graph.start(quantum, not_killed(), Vec::new());
    retirements.assert_outcome(0, FAILED, "a middle stage failed");
    assert_eq!(
        ran_stages(&ran),
        vec![1, 1, 0],
        "the stage behind the failure did not run"
    );
    world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_quantum_whose_last_stage_fails_retires() {
    let _light = common::light();
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    // No stage is left to be abandoned: the quantum retires behind the markers of the stages that ran.
    let (graph, ran) = chain(
        &world,
        vec![
            link(0, Act::Nothing),
            link(1, Act::Nothing),
            link(0, Act::Nothing),
            link(1, Act::Fail),
        ],
        &retirements,
    );
    let mut sequence = world.sequence(1024);
    let quantum = world.quantum(&mut sequence, 0);
    let _ = graph.start(quantum, not_killed(), Vec::new());
    retirements.assert_outcome(0, FAILED, "the last stage failed");
    assert_eq!(ran_stages(&ran), vec![1, 1, 1, 1]);
    world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_quantum_whose_last_stage_panics_retires() {
    let _light = common::light();
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let (graph, _ran) = chain(
        &world,
        vec![link(0, Act::Nothing), link(0, Act::Nothing), link(0, Act::Panic)],
        &retirements,
    );
    let mut sequence = world.sequence(1024);
    let quantum = world.quantum(&mut sequence, 0);
    let _ = graph.start(quantum, not_killed(), Vec::new());
    retirements.assert_outcome(0, FAILED, "the last stage panicked");
    world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_quantum_that_runs_to_its_end_retires_completed() {
    let _light = common::light();
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let (graph, ran) = chain(
        &world,
        vec![
            link(0, Act::Nothing),
            link(1, Act::Nothing),
            link(0, Act::Nothing),
            link(1, Act::Nothing),
        ],
        &retirements,
    );
    let mut sequence = world.sequence(1024);
    let quantum = world.quantum(&mut sequence, 0);
    let _ = graph.start(quantum, not_killed(), Vec::new());
    retirements.assert_outcome(0, COMPLETED, "ran to its end");
    assert_eq!(ran_stages(&ran), vec![1, 1, 1, 1]);
    world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_killed_quantum_is_followed_by_quanta_on_the_same_sequence() {
    let _light = common::light();
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let shape = |kill: Act| {
        vec![
            link(0, Act::Nothing),
            link(1, kill),
            link(0, Act::Nothing),
            link(1, Act::Nothing),
        ]
    };
    let mut sequence = world.sequence(1024);

    // A killed quantum retires by itself, and the next one runs to its end.
    let kill = not_killed();
    let (killed, _) = chain(&world, shape(Act::Kill(Arc::clone(&kill))), &retirements);
    let _ = killed.start(world.quantum(&mut sequence, 0), kill, Vec::new());
    retirements.assert_outcome(0, KILLED, "the killed quantum");
    let (whole, ran) = chain(&world, shape(Act::Nothing), &retirements);
    let _ = whole.start(world.quantum(&mut sequence, 1), not_killed(), Vec::new());
    retirements.assert_outcome(1, COMPLETED, "the quantum after the killed one");
    assert_eq!(ran_stages(&ran), vec![1, 1, 1, 1]);

    // A quantum admitted behind a killed one before it retired: its first stage is ordered after the killed quantum's
    // markers, and both retire.
    let kill = not_killed();
    let (killed, _) = chain(&world, shape(Act::Kill(Arc::clone(&kill))), &retirements);
    let first = killed.start(world.quantum(&mut sequence, 2), kill, Vec::new());
    let (whole, ran) = chain(&world, shape(Act::Nothing), &retirements);
    let edge = ExternalEdge {
        from: first,
        from_stage: StageId(3),
        to_stage: StageId(0),
        kind: EdgeKind::Submission,
    };
    let _ = whole.start(world.quantum(&mut sequence, 3), not_killed(), vec![edge]);
    retirements.assert_outcome(2, KILLED, "the killed quantum with one behind it");
    retirements.assert_outcome(3, COMPLETED, "the quantum behind the killed one");
    assert_eq!(ran_stages(&ran), vec![1, 1, 1, 1]);
    world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn quanta_that_complete_are_killed_and_fail_on_shared_lanes_all_retire() {
    let _light = common::light();
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let mut sequence = world.sequence(1024);
    let shape = |act: Act| {
        vec![
            link(0, Act::Nothing),
            link(1, act),
            link(0, Act::Nothing),
            link(1, Act::Nothing),
            link(0, Act::Nothing),
        ]
    };
    // Three quanta at a time on two lanes, so that stages meet held lanes.
    for round in 0..10usize {
        let mut expected = Vec::new();
        for slot in 0..3usize {
            let id = round * 3 + slot;
            let kill = not_killed();
            let (act, outcome) = match slot {
                0 => (Act::Kill(Arc::clone(&kill)), KILLED),
                1 => (Act::Nothing, COMPLETED),
                _ => (Act::Fail, FAILED),
            };
            let (graph, _) = chain(&world, shape(act), &retirements);
            let _ = graph.start(world.quantum(&mut sequence, id as u64), kill, Vec::new());
            expected.push((id, outcome));
        }
        for (id, outcome) in expected {
            retirements.assert_outcome(id, outcome, &format!("round {round}"));
        }
    }
    world.close();
}

/// The spin kernel's source: one thread that waits `ns` nanoseconds of the device's global timer.
const SPIN: &str = r#"
extern "C" __global__ void spin(unsigned long long ns) {
    unsigned long long start, now;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(start));
    do {
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(now));
    } while (now - start < ns);
}
"#;

/// How long the spin kernel runs.
const SPIN_NS: u64 = 600_000_000;

/// A stage on a lane other than the home lane that queues a long kernel and then fails: the work a failed submission
/// leaves behind a marker RustEE never records. Its abandonment is a plan stage's.
struct SpinThenFail {
    kernel: CudaKernel,
    lanes: LaneTable,
    placed: Placed,
}

fn refused(why: impl Display) -> StageError {
    StageError::Fatal(why.to_string())
}

impl Stage<Quantum> for SpinThenFail {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let id = cx.lane.ok_or_else(|| refused("no lane"))?;
        let lane = self.lanes.get(id).ok_or_else(|| refused("no such lane"))?;
        let marker = submit_on(&**lane, cx.waits.iter().copied(), quantum.unmarked(), |queue| {
            // SAFETY: the kernel's one argument is the spin time, and it touches no memory.
            unsafe { queue.launch(&self.kernel, Launch::linear(1, 1), &Args::new().arg(SPIN_NS)) }.map_err(refused)?;
            Err(refused("the stage refuses after it queued the kernel"))
        })?;
        Ok(Submitted::marker(marker))
    }

    fn abandoned(&self, quantum: &Quantum, home: &(dyn DeviceQueue + 'static)) {
        self.placed.abandoned(quantum, home);
    }
}

/// A stage placed on another lane whose launch fails, run through the plan's own submission ([`Placed`]); it reports
/// the marker the quantum kept for the lane when the quantum is abandoned, before the abandonment orders anything.
struct Watch {
    placed: Placed,
    lane: LaneId,
    kept: Arc<AtomicU64>,
}

impl Stage<Quantum> for Watch {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        self.placed.run(quantum, cx)
    }

    fn abandoned(&self, quantum: &Quantum, home: &(dyn DeviceQueue + 'static)) {
        self.kept
            .store(quantum.unmarked().latest(self.lane).unwrap_or(0), Ordering::Release);
        self.placed.abandoned(quantum, home);
    }
}

/// A launch that fails on a lane other than the home lane, in a stage's own submission, leaves the marker of that
/// submission in the quantum for the abandonment to await, and the abandonment then orders the home lane behind it.
#[test]
#[ignore = "needs a GPU and the native library"]
fn a_stage_that_fails_through_its_own_submission_keeps_its_marker_for_the_abandonment() {
    let _light = common::light();
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let ran: Arc<Vec<AtomicUsize>> = Arc::new(vec![AtomicUsize::new(0)]);
    let mut spec = StageGraphSpec::new();
    let first = spec.stage(StageSpec {
        name: "home",
        ordered: None,
        lanes: vec![LaneId(0)],
    });
    let second = spec.stage(StageSpec {
        name: "fails",
        ordered: None,
        lanes: vec![LaneId(1)],
    });
    spec.edge(first, second, EdgeKind::Submission);
    let home: Arc<dyn Stage<Quantum>> = Arc::new(Placed::new(Arc::new(Probe {
        act: Act::Nothing,
        ran: Arc::clone(&ran),
        index: 0,
    })));
    let kept = Arc::new(AtomicU64::new(0));
    let fails: Arc<dyn Stage<Quantum>> = Arc::new(Watch {
        placed: Placed::new(Arc::new(Probe {
            act: Act::Fail,
            ran: Arc::clone(&ran),
            index: 0,
        })),
        lane: LaneId(1),
        kept: Arc::clone(&kept),
    });
    let log = Arc::clone(&retirements);
    let graph = StageGraph::new(
        spec,
        vec![home, fails],
        world.runtime.clone(),
        move |quantum, outcome| log.retire(&quantum, &outcome),
    )
    .unwrap();
    let mut sequence = world.sequence(1024);
    let quantum = world.quantum(&mut sequence, 0);
    let _ = graph.start(quantum, not_killed(), Vec::new());
    retirements.assert_outcome(0, FAILED, "the stage on the other lane failed");
    assert_ne!(
        kept.load(Ordering::Acquire),
        0,
        "the failed submission's marker was not kept in the quantum"
    );
    world.close();
}

/// A stage on lane 1, not the home lane 0, queues a kernel that runs for 600 ms and then fails its submission. RustEE
/// records no marker of a failed stage, so nothing but the engine's own record of the failed submission orders the
/// quantum's retirement behind the kernel: the quantum must retire only after the kernel ended.
#[test]
#[ignore = "needs a GPU and the native library"]
fn a_failed_submission_on_another_lane_retires_only_after_its_kernel_ended() {
    let _light = common::light();
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let settings = Settings::with_model(ModelSettings::default());
    let sources = kernel_sources(&world.library, &settings).unwrap();
    let options = CompileOptions {
        compiler: sources.compiler().map(Path::to_path_buf),
        ..CompileOptions::default()
    };
    let kernel = CudaModule::from_source(&world.typed.gpu, SPIN, &options)
        .unwrap()
        .kernel("spin")
        .unwrap();
    let ran: Arc<Vec<AtomicUsize>> = Arc::new(vec![AtomicUsize::new(0)]);
    let mut spec = StageGraphSpec::new();
    let first = spec.stage(StageSpec {
        name: "home",
        ordered: None,
        lanes: vec![LaneId(0)],
    });
    let second = spec.stage(StageSpec {
        name: "spin",
        ordered: None,
        lanes: vec![LaneId(1)],
    });
    spec.edge(first, second, EdgeKind::Submission);
    let home: Arc<dyn Stage<Quantum>> = Arc::new(Placed::new(Arc::new(Probe {
        act: Act::Nothing,
        ran: Arc::clone(&ran),
        index: 0,
    })));
    let spin: Arc<dyn Stage<Quantum>> = Arc::new(SpinThenFail {
        kernel,
        lanes: world.table.clone(),
        placed: Placed::new(Arc::new(Probe {
            act: Act::Nothing,
            ran: Arc::clone(&ran),
            index: 0,
        })),
    });
    let log = Arc::clone(&retirements);
    let graph = StageGraph::new(
        spec,
        vec![home, spin],
        world.runtime.clone(),
        move |quantum, outcome| log.retire(&quantum, &outcome),
    )
    .unwrap();
    let mut sequence = world.sequence(1024);
    let quantum = world.quantum(&mut sequence, 0);
    let begun = Instant::now();
    let _ = graph.start(quantum, not_killed(), Vec::new());
    retirements.assert_outcome(0, FAILED, "the stage on the other lane failed");
    let elapsed = begun.elapsed();
    assert!(
        // Timing-based on purpose: the claim is that a stage retires only after a kernel that spins for SPIN_NS; the kernel's run time is the signal.
        elapsed >= Duration::from_nanos(SPIN_NS) * 3 / 4,
        "the quantum retired after {elapsed:?}, while the kernel it queued runs for {SPIN_NS} ns"
    );
    world.close();
}
// The plan's graphs, with a captured quantum.

struct Described(HashMap<(LayerIndex, Role), WeightRef>);

impl WeightLookup for Described {
    fn weight(&self, layer: LayerIndex, role: Role) -> Option<&WeightRef> {
        self.0.get(&(layer, role))
    }
}

/// The reduced model's plan over weights with no device memory behind them: its stages are never launched here.
fn plan_without_weights(config: &Arc<ModelConfig>) -> Arc<Plan> {
    let residency = residency_plan(
        &common::config(),
        &common::rco_table(),
        &common::settings(8192),
        15 << 30,
    )
    .unwrap();
    let weights = Described(
        residency
            .device
            .iter()
            .filter_map(|placement| {
                let key = object_key(&placement.name, LAYERS)?;
                Some((
                    key,
                    WeightRef {
                        name: placement.name.clone(),
                        shape: placement.shape.clone(),
                        dtype: placement.dtype,
                        format: placement.format,
                        layout: placement.layout,
                        ptr: DevicePtr::NULL,
                        bytes: placement.bytes,
                        residence: Residence::Device,
                        trellis: None,
                    },
                ))
            })
            .collect(),
    );
    let formats = Formats::new(
        Arc::new(TrellisLinear::from_settings(&TrellisKernelSettings::default(), 70)),
        Arc::new(Nvfp4Linear::with_native_route(true, 70)),
    );
    Arc::new(Plan::build(config, &weights, &residency, &formats, None, None).unwrap())
}

/// A world with the plan's lanes and the owner's, an executor on it that captures, and a decode quantum at position 1,
/// which has a capture key.
struct Captured {
    world: World,
    executor: Executor,
    sequence: SequenceState,
    retirements: Arc<Retirements>,
    quantum: Option<Quantum>,
}

fn captured() -> Option<Captured> {
    let config = reduced_config();
    let plan = plan_without_weights(&config);
    let width = plan.fan_out_width();
    // The plan's lanes, then the owner's.
    let world = World::open(width + 1)?;
    let retirements = Retirements::new();
    let sink = Arc::clone(&retirements);
    let executor = Executor::new(
        plan,
        &world.runtime,
        width,
        Arc::clone(&world.typed.gpu),
        8,
        move |quantum, outcome| {
            sink.retire(&quantum, &outcome);
        },
    )
    .unwrap();
    let mut sequence = world.sequence(1024);
    // The first reservation allocates and so has no key; the quantum behind it fits the reserved rows and has one.
    drop(world.quantum(&mut sequence, 0));
    sequence.kv.commit(1).unwrap();
    let quantum = world.quantum(&mut sequence, 1);
    assert!(quantum.capture_key().is_some(), "the quantum has a key");
    Some(Captured {
        world,
        executor,
        sequence,
        retirements,
        quantum: Some(quantum),
    })
}

/// The plan has no weights here, so a captured quantum's body is never launched: only the paths that end before it.
#[test]
#[ignore = "needs a GPU and the native library"]
fn a_captured_quantum_killed_before_its_launch_retires() {
    // A capture fails when another thread of the process allocates or synchronizes meanwhile: nothing else runs.
    let _card = common::card();
    let Some(mut captured) = captured() else { return };
    let quantum = captured.quantum.take().unwrap();
    let started = captured
        .executor
        .start(quantum, Arc::new(AtomicBool::new(true)), &[])
        .unwrap();
    // The reference pins the quantum: it retires only once nothing depends on it.
    drop(started);
    captured
        .retirements
        .assert_outcome(1, KILLED, "a captured quantum killed before its launch");
    assert_eq!(
        captured.executor.graph_stats(),
        GraphStats::default(),
        "the killed quantum's body never reached the cache"
    );
    let _ = &captured.sequence;
    captured.world.close();
}
