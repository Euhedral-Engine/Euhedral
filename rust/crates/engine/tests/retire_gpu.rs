//! Quanta that end before their stages do, on a GPU: a quantum that is killed or fails part-way must still retire.
//! The tests are `#[ignore]`d and run locally in a memory-capped scope:
//! `systemd-run --user --scope -p MemoryMax=16G cargo test -p euhedral-inference-engine --test retire_gpu -- --ignored`.
//!
//! The shapes are synthetic: stages that queue nothing but their markers, on lanes of their own, in graphs the test
//! builds, and the plan's own graphs over weights with no memory behind them (a quantum that is killed never launches
//! them). No artifact is loaded. The library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY`
//! in the environment, or `gradle.properties`); a test skips with a message when it is absent. Every wait for a quantum
//! to retire has a deadline, so a quantum that never retires fails its test and does not hang the suite.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: the test thread waits for retirements against a deadline and synchronizes lanes"
)]

mod common;

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU8, AtomicUsize, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::Frame;
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, DeviceQueue, EdgeKind, ExternalEdge, GraphRuntime, Lake, LaneId, LanePool,
    Marker, QuantumOutcome, RetryCounts, Stage, StageError, StageGraph, StageGraphSpec, StageId, StageSpec, UnitGuard,
};
use euhedral_inference_cuda::{
    CudaLane, CudaLibrary, DeviceBuffer, DevicePtr, PinnedPool, RetireList, Submission, SubmitMode, ops,
};
use euhedral_inference_engine::formats::{Nvfp4Linear, TrellisLinear};
use euhedral_inference_engine::lattice_config;
use euhedral_inference_engine::model::{LayerIndex, LayerKind, ModelConfig, Role, WeightRef, object_key};
use euhedral_inference_engine::plan::{Formats, Plan, WeightLookup};
use euhedral_inference_engine::quantum::{Executor, GraphCache, Sighting};
use euhedral_inference_engine::residency::planner::plan as residency_plan;
use euhedral_inference_engine::stages::attention::KvSource;
use euhedral_inference_engine::stages::sampling::SelectionPool;
use euhedral_inference_engine::stages::{
    InputRecordRef, LaneEnds, LaneTable, Placed, Quantum, QuantumKind, Settled, Settler, Settling, StageLaunch,
};
use euhedral_inference_engine::state::attention::AttentionScratch;
use euhedral_inference_engine::state::gdn::GdnStates;
use euhedral_inference_engine::state::kv::{KvCache, KvRetained};
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
            QuantumOutcome::Failed { .. } => FAILED,
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
    lanes: Vec<Arc<CudaLane>>,
    table: LaneTable,
    runtime: GraphRuntime,
    config: Arc<ModelConfig>,
    workspace: Arc<Workspace>,
    pinned: Arc<PinnedPool>,
    retire: Arc<RetireList>,
    lattice: Arc<ControlPlaneLattice>,
    lake: Arc<Lake>,
    unit: Option<UnitGuard>,
}

impl World {
    /// `lane_count` lanes, numbered from 0; the last is the owner's, for the sequence's own submissions.
    fn open(lane_count: usize) -> Option<World> {
        let library = library()?;
        let config = reduced_config();
        let lanes: Vec<Arc<CudaLane>> = (0..lane_count)
            .map(|id| CudaLane::new(&library, LaneId(id as u16), 1024).unwrap())
            .collect();
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
            retire: Arc::new(RetireList::new()),
            library,
            lanes,
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
    fn owner(&self) -> &Arc<CudaLane> {
        self.lanes.last().unwrap()
    }

    fn close(mut self) {
        drop(self.unit.take());
        self.lake.complete_gracefully();
        self.lattice.close();
        for lane in &self.lanes {
            lane.synchronize().unwrap();
        }
    }

    fn sequence(&self, kv_tokens: u64) -> SequenceState {
        let submission = Submission::begin(self.owner(), SubmitMode::default()).unwrap();
        let gdn = GdnStates::allocate(&self.library, &self.config, 8, &submission).unwrap();
        submission.finish_signalled().unwrap();
        self.owner().synchronize().unwrap();
        let kv = KvCache::new(
            &self.library,
            &self.retire,
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
        let submission = Submission::begin(self.owner(), SubmitMode::default()).unwrap();
        sequence
            .kv
            .reserve(&submission, &self.pinned, start, 1, &mut retained)
            .unwrap();
        sequence.kv.appended(1).unwrap();
        drop(submission);
        self.owner().synchronize().unwrap();
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

    fn launch(&self, _quantum: &Quantum, _submission: &Submission<'_>) -> Result<(), StageError> {
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

/// One stage of a chain: the lane it is pinned to, whether the plan would have it signal, and what it does.
struct Link {
    lane: u16,
    signal: bool,
    act: Act,
}

fn link(lane: u16, signal: bool, act: Act) -> Link {
    Link { lane, signal, act }
}

/// A chain of stages, each behind the one before it, and how many of them ran.
fn chain(
    world: &World,
    links: Vec<Link>,
    retirements: &Arc<Retirements>,
) -> (Arc<StageGraph<Quantum>>, Arc<Vec<AtomicUsize>>) {
    let settler = Settler::new(&world.runtime);
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
        stages.push(Arc::new(Settling::new(
            Arc::new(Placed::new(launch, link.signal)),
            Arc::clone(&settler),
        )));
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
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let kill = not_killed();
    // The first stage submits an unsignalled marker and sets the kill switch; the stage that would have signalled behind
    // it never runs.
    let (graph, ran) = chain(
        &world,
        vec![
            link(0, false, Act::Kill(Arc::clone(&kill))),
            link(0, false, Act::Nothing),
            link(0, true, Act::Nothing),
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
fn a_quantum_killed_with_unsignalled_markers_on_two_lanes_retires() {
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let kill = not_killed();
    // Lane 0 holds the first stage's marker and lane 1 the second's, both unsignalled when the second sets the switch.
    let (graph, ran) = chain(
        &world,
        vec![
            link(0, false, Act::Nothing),
            link(1, false, Act::Kill(Arc::clone(&kill))),
            link(0, true, Act::Nothing),
            link(1, true, Act::Nothing),
        ],
        &retirements,
    );
    let mut sequence = world.sequence(1024);
    let quantum = world.quantum(&mut sequence, 0);
    let _ = graph.start(quantum, kill, Vec::new());
    retirements.assert_outcome(0, KILLED, "killed with two lanes unsignalled");
    assert_eq!(ran_stages(&ran), vec![1, 1, 0, 0]);
    world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_quantum_killed_before_its_first_stage_retires() {
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let (graph, ran) = chain(
        &world,
        vec![link(0, false, Act::Nothing), link(0, true, Act::Nothing)],
        &retirements,
    );
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
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let (graph, ran) = chain(
        &world,
        vec![
            link(0, false, Act::Nothing),
            link(0, false, Act::Fail),
            link(0, true, Act::Nothing),
        ],
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
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    // No stage is left to be abandoned: the failing stage settles the lane its earlier, unsignalled stage left.
    let (graph, ran) = chain(
        &world,
        vec![
            link(0, false, Act::Nothing),
            link(1, false, Act::Nothing),
            link(0, true, Act::Nothing),
            link(1, true, Act::Fail),
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
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let (graph, _ran) = chain(
        &world,
        vec![
            link(0, false, Act::Nothing),
            link(0, false, Act::Nothing),
            link(0, true, Act::Panic),
        ],
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
fn a_quantum_that_runs_to_its_end_settles_nothing_and_retires_completed() {
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let (graph, ran) = chain(
        &world,
        vec![
            link(0, false, Act::Nothing),
            link(1, false, Act::Nothing),
            link(0, true, Act::Nothing),
            link(1, true, Act::Nothing),
        ],
        &retirements,
    );
    let mut sequence = world.sequence(1024);
    let quantum = world.quantum(&mut sequence, 0);
    let ends = Arc::clone(quantum.lane_ends());
    let _ = graph.start(quantum, not_killed(), Vec::new());
    retirements.assert_outcome(0, COMPLETED, "ran to its end");
    assert_eq!(ran_stages(&ran), vec![1, 1, 1, 1]);
    assert!(
        ends.unsignalled_lanes().is_empty(),
        "every lane ended with a signalled marker of the plan's own"
    );
    world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_killed_quantum_is_followed_by_quanta_on_the_same_sequence() {
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let shape = |kill: Act| {
        vec![
            link(0, false, Act::Nothing),
            link(1, false, kill),
            link(0, true, Act::Nothing),
            link(1, true, Act::Nothing),
        ]
    };
    let mut sequence = world.sequence(1024);

    // A killed quantum retires by itself, with nothing behind it to signal for it; the next one runs to its end.
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
fn settling_signals_a_lane_whose_last_marker_is_unsignalled_and_never_waits_for_a_held_lane() {
    let Some(world) = World::open(1) else { return };
    let lane = &world.lanes[0];
    let ends = LaneEnds::new(1);
    let submission = Submission::begin(lane, SubmitMode::default()).unwrap();
    let marker = submission.finish().unwrap();
    ends.record(marker, false);
    lane.synchronize().unwrap();
    assert!(
        lane.completed() < marker.value,
        "the lane's completion word does not reach an unsignalled marker, however long it waits"
    );
    assert_eq!(ends.unsignalled_lanes(), vec![LaneId(0)]);

    let held = Submission::begin(lane, SubmitMode::default()).unwrap();
    assert_eq!(
        ends.settle(lane).unwrap(),
        Settled::Busy,
        "a held lane is not waited for"
    );
    drop(held);
    assert_eq!(ends.settle(lane).unwrap(), Settled::Signalled);
    let begun = Instant::now();
    while lane.completed() < marker.value {
        assert!(
            begun.elapsed() < DEADLINE,
            "the settling marker never reached the completion word"
        );
        thread::sleep(Duration::from_millis(1));
    }
    assert!(ends.unsignalled_lanes().is_empty());
    assert_eq!(
        ends.settle(lane).unwrap(),
        Settled::Clean,
        "a settled lane is left alone"
    );
    world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn quanta_that_complete_are_killed_and_fail_on_shared_lanes_all_retire() {
    let Some(world) = World::open(3) else { return };
    let retirements = Retirements::new();
    let mut sequence = world.sequence(1024);
    let shape = |act: Act| {
        vec![
            link(0, false, Act::Nothing),
            link(1, false, act),
            link(0, false, Act::Nothing),
            link(1, true, Act::Nothing),
            link(0, true, Act::Nothing),
        ]
    };
    // Three quanta at a time on two lanes, so that stages and settlings meet held lanes.
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

/// Settles a quantum whose lane 0 ends unsignalled while another submission holds the lane, driving the completions by
/// hand on this thread (they are attached to no lattice): the frame finds the lane held, publishes itself again, and
/// goes out again only once a lane has progressed or `backstop` has passed.
fn settle_behind_a_held_lane(backstop: Duration) {
    let Some(world) = World::open(2) else { return };
    let lane = &world.lanes[0];
    let pool = Arc::clone(&world.runtime.lanes);
    let completions = Completions::new(Arc::clone(&pool));
    let runtime = GraphRuntime {
        stages: Arc::clone(&world.runtime.stages),
        completions: Arc::clone(&completions),
        lanes: pool,
        retries: Arc::new(RetryCounts::default()),
        retry_backstop: backstop,
    };
    let frames_out = |completions: &Completions| {
        let mut frames: Vec<Box<dyn Frame>> = Vec::new();
        completions.pull(&mut |frame| frames.push(frame), &mut |_| false, u64::MAX);
        frames
    };
    // The quantum's last marker on lane 0 is unsignalled.
    let mut sequence = world.sequence(1024);
    let quantum = world.quantum(&mut sequence, 0);
    let submission = Submission::begin(lane, SubmitMode::default()).unwrap();
    let unsignalled = submission.finish().unwrap();
    quantum.lane_ends().record(unsignalled, false);
    let stage = Settling::new(
        Arc::new(Placed::new(
            Arc::new(Probe {
                act: Act::Nothing,
                ran: Arc::new(vec![AtomicUsize::new(0)]),
                index: 0,
            }),
            true,
        )),
        Settler::new(&runtime),
    );

    let held = Submission::begin(lane, SubmitMode::default()).unwrap();
    stage.abandoned(&quantum);
    assert_eq!(completions.pending(), 1, "the abandoned stage publishes one settling");
    let mut first = frames_out(&completions);
    assert_eq!(first.len(), 1, "the settling is ready at once");
    let mut frame = first.pop().unwrap();
    frame.execute().unwrap();
    Frame::do_finally(frame);
    assert_eq!(
        completions.pending(),
        1,
        "the lane was held, so the settling is published again"
    );
    assert!(quantum.lane_ends().unsignalled(LaneId(0)), "nothing was signalled yet");
    if backstop > Duration::from_secs(60) {
        assert!(
            frames_out(&completions).is_empty(),
            "no lane has progressed and the backstop is far off, so the settling waits"
        );
    }

    // The holder lets go, and the lane progresses: the settling goes out again and signals.
    let released = held.finish_signalled().unwrap();
    let begun = Instant::now();
    let mut again = Vec::new();
    while again.is_empty() {
        assert!(begun.elapsed() < DEADLINE, "the settling was never published again");
        again = frames_out(&completions);
        thread::sleep(Duration::from_millis(1));
    }
    if backstop > Duration::from_secs(60) {
        assert!(
            lane.completed() >= released.value,
            "the settling was published again by the lane's progress"
        );
    }
    let mut frame = again.pop().unwrap();
    frame.execute().unwrap();
    Frame::do_finally(frame);
    assert_eq!(completions.pending(), 0, "the settling is done");
    assert!(quantum.lane_ends().unsignalled_lanes().is_empty());
    let begun = Instant::now();
    while lane.completed() < unsignalled.value {
        assert!(
            begun.elapsed() < DEADLINE,
            "the lane never reached the quantum's marker"
        );
        thread::sleep(Duration::from_millis(1));
    }
    world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_settling_that_finds_its_lane_held_is_published_again_behind_the_lane_s_progress() {
    settle_behind_a_held_lane(Duration::from_secs(3600));
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_settling_that_finds_its_lane_held_is_published_again_at_the_backstop() {
    settle_behind_a_held_lane(Duration::ZERO);
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
                        trellis: None,
                    },
                ))
            })
            .collect(),
    );
    let formats = Formats::new(
        Arc::new(TrellisLinear::from_settings(&TrellisKernelSettings::default(), 70)),
        Arc::new(Nvfp4Linear::with_native_route(true)),
    );
    Arc::new(Plan::build(config, &weights, &residency, &formats).unwrap())
}

/// A captured graph for the quantum's key, recorded from a copy of eight bytes, and the buffers it copies.
fn record_a_graph(
    world: &World,
    cache: &mut GraphCache,
    quantum: &Quantum,
    capture: &CudaLane,
) -> (DeviceBuffer, DeviceBuffer) {
    let (from, to) = (
        DeviceBuffer::alloc(&world.library, 256).unwrap(),
        DeviceBuffer::alloc(&world.library, 256).unwrap(),
    );
    let key = quantum.capture_key().expect("the quantum has a key");
    let mut recorded = false;
    for _ in 0..2 {
        let sighting = cache
            .get_or_record(key.clone(), capture, SubmitMode::default(), |submission| {
                ops::copy_device_to_device(submission, to.ptr(), from.ptr(), 8)
            })
            .unwrap();
        recorded = matches!(sighting, Sighting::Graph(_));
    }
    assert!(recorded, "the second sight of a key records its graph");
    (from, to)
}

/// A world with the plan's lanes, the capture lane, and the owner's, an executor on it, and a cache holding the
/// graph of a decode quantum at position 1.
struct Captured {
    world: World,
    executor: Executor,
    cache: GraphCache,
    sequence: SequenceState,
    retirements: Arc<Retirements>,
    _buffers: (DeviceBuffer, DeviceBuffer),
    quantum: Option<Quantum>,
}

fn captured() -> Option<Captured> {
    let config = reduced_config();
    let plan = plan_without_weights(&config);
    let width = plan.fan_out_width();
    // The plan's lanes, then the capture lane, then the owner's.
    let world = World::open(width + 2)?;
    let retirements = Retirements::new();
    let sink = Arc::clone(&retirements);
    let executor = Executor::new(plan, &world.runtime, LaneId(width as u16), move |quantum, outcome| {
        sink.retire(&quantum, &outcome);
    })
    .unwrap();
    let mut sequence = world.sequence(1024);
    // The first reservation allocates and so has no key; the quantum behind it fits the reserved rows and has one.
    drop(world.quantum(&mut sequence, 0));
    sequence.kv.commit(1).unwrap();
    let quantum = world.quantum(&mut sequence, 1);
    let mut cache = GraphCache::with_capacity(8);
    let buffers = record_a_graph(&world, &mut cache, &quantum, &world.lanes[width]);
    Some(Captured {
        world,
        executor,
        cache,
        sequence,
        retirements,
        _buffers: buffers,
        quantum: Some(quantum),
    })
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_captured_quantum_killed_before_its_launch_retires() {
    let Some(mut captured) = captured() else { return };
    let quantum = captured.quantum.take().unwrap();
    let started = captured
        .executor
        .start(&mut captured.cache, quantum, Arc::new(AtomicBool::new(true)), None)
        .unwrap();
    assert!(started.replayed, "the quantum replays the recorded graph");
    // The reference pins the quantum: it retires only once nothing depends on it.
    drop(started);
    captured
        .retirements
        .assert_outcome(1, KILLED, "a captured quantum killed before its launch");
    let _ = &captured.sequence;
    captured.world.close();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_captured_quantum_that_is_not_killed_launches_its_graph_and_retires_completed() {
    let Some(mut captured) = captured() else { return };
    let quantum = captured.quantum.take().unwrap();
    let started = captured
        .executor
        .start(&mut captured.cache, quantum, not_killed(), None)
        .unwrap();
    assert!(started.replayed, "the quantum replays the recorded graph");
    // The reference pins the quantum: it retires only once nothing depends on it.
    drop(started);
    captured.retirements.assert_outcome(1, COMPLETED, "a captured quantum");
    captured.world.close();
}
