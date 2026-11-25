//! The edges that keep a quantum's first weight copies behind the previous quantum's last readers of the staging
//! slots, on a staged plan's own graph run on the fake backend (no GPU, no wall clock), and the marker rings that hold
//! a staged view's stages. Every wait is bounded, so a mutant that hangs fails fast.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "a test file: a failed unwrap is the failure, and the test locks its records and waits, bounded"
)]

use std::any::Any;
use std::collections::{BTreeSet, HashMap};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_gpu::fake::{FakeGpu, FakeKernel, FakeModule, FakeQueue};
use euhedral_gpu::{Args, CompileOptions, LaneEnqueue, Launch, Module};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, EdgeKind, ExternalEdge, Gate, GateRef, GraphRuntime, Lake, LaneId,
    QuantumOutcome, QuantumRef, Stage, StageCx, StageError, StageGraph, StageId, Submitted, UnitGuard,
};
use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_engine::formats::{Nvfp4Linear, TrellisLinear};
use euhedral_inference_engine::generation::{check_marker_rings, markers_in_flight, staging_ring};
use euhedral_inference_engine::lattice_config;
use euhedral_inference_engine::model::{LayerIndex, ModelConfig, Residence, Role, WeightRef, object_key};
use euhedral_inference_engine::plan::{Formats, Plan, PlanBuildError, StageKind, StagingRing, View, WeightLookup};
use euhedral_inference_engine::quantum::{Dependency, Predecessor, external_edges};
use euhedral_inference_engine::residency::offload::StagingGeometry;
use euhedral_inference_engine::residency::planner::{ResidencyPlan, plan as residency};
use euhedral_inference_engine::runtime::{open_lane_set, stage_ring_for};
use euhedral_inference_engine::stages::{Unmarked, submit_on};
use euhedral_inference_settings::{ModelSettings, Settings, TrellisKernelSettings};

mod common;

use common::{config, rco_table, settings};

const BOUND: Duration = Duration::from_secs(20);
const SLOTS: usize = 3;

type Logs = (
    Arc<Mutex<Vec<(u32, usize)>>>,
    Arc<Mutex<Vec<(u32, usize)>>>,
    Arc<Mutex<HashMap<(u32, usize), Gate>>>,
);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

struct Described(HashMap<(LayerIndex, Role), WeightRef>);

impl WeightLookup for Described {
    fn weight(&self, layer: LayerIndex, role: Role) -> Option<&WeightRef> {
        self.0.get(&(layer, role))
    }
}

struct Fixture {
    config: Arc<ModelConfig>,
    resident: ResidencyPlan,
    weights: Described,
    formats: Formats,
}

fn fixture() -> Fixture {
    let config = Arc::new(config());
    let resident = residency(&config, &rco_table(), &settings(8192), 15 << 30).unwrap();
    let weights = Described(
        resident
            .device
            .iter()
            .filter_map(|placement| {
                let key = object_key(&placement.name, config.num_hidden_layers)?;
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
    Fixture {
        config,
        resident,
        weights,
        formats,
    }
}

impl Fixture {
    /// The names of the projections staged: every mlp down and gdn output.
    fn selection(&self) -> BTreeSet<String> {
        self.resident
            .device
            .iter()
            .filter(|placement| {
                placement.name.starts_with("text/layers/")
                    && (placement.name.ends_with("/mlp/down") || placement.name.ends_with("/gdn/output"))
            })
            .map(|placement| placement.name.clone())
            .collect()
    }

    fn staged(&self, names: &BTreeSet<String>, lane: LaneId) -> Plan {
        self.try_staged(names, lane).unwrap()
    }

    fn try_staged(&self, names: &BTreeSet<String>, lane: LaneId) -> Result<Plan, PlanBuildError> {
        let mut weights = self.weights.0.clone();
        for (index, weight) in weights.values_mut().enumerate() {
            if names.contains(&weight.name) {
                weight.residence = Residence::Host {
                    offset: 4096 * index as u64,
                };
            }
        }
        let slot_bytes = self
            .resident
            .device
            .iter()
            .filter(|placement| names.contains(&placement.name))
            .map(|placement| placement.bytes)
            .max()
            .unwrap();
        let ring = StagingRing {
            base: DevicePtr::NULL.offset(1 << 32),
            slots: SLOTS,
            slot_bytes,
        };
        Plan::build(
            &self.config,
            &Described(weights),
            &self.resident,
            &self.formats,
            Some(&ring),
            Some(lane),
        )
    }

    fn unstaged(&self) -> Plan {
        Plan::build(&self.config, &self.weights, &self.resident, &self.formats, None, None).unwrap()
    }
}

/// What a quantum of `plan`'s `view` offers the quantum behind it, as the executor's `Started` does.
fn offered(plan: &Plan, view: View) -> (StageId, StageId, Vec<Option<StageId>>) {
    let units = plan.unit_stages(view).unwrap().len();
    (
        StageId(0),
        StageId((units - 1) as u16),
        plan.slot_readers(view).to_vec(),
    )
}

fn predecessor(offer: &(StageId, StageId, Vec<Option<StageId>>)) -> Predecessor<'_> {
    Predecessor {
        root: offer.0,
        sink: offer.1,
        slot_readers: &offer.2,
    }
}

#[test]
fn each_slots_first_transfer_waits_for_the_predecessors_last_reader_of_that_slot() {
    let fixture = fixture();
    let plan = fixture.staged(&fixture.selection(), LaneId(9));
    for (previous, next) in [
        (View::Decode, View::Decode),
        (View::Verify { rows: 2 }, View::Verify { rows: 2 }),
        (View::PrefillChunk, View::Decode),
        (View::Decode, View::PrefillChunk),
    ] {
        let offer = offered(&plan, previous);
        let first = plan.first_transfers(next);
        assert_eq!(first.len(), SLOTS, "{next:?}");
        let kinds = plan.kinds(next).expect("the view is planned");
        assert!(
            !matches!(kinds[0], StageKind::WeightTransfer),
            "{next:?}: the first stage is not a transfer"
        );
        assert!(first.iter().all(|&(_, unit)| unit != StageId(0)), "{next:?}");
        let edges = external_edges(first, &[predecessor(&offer)]);
        assert_eq!(
            edges[0],
            Dependency {
                predecessor: 0,
                from_stage: offer.1,
                to_stage: StageId(0)
            },
            "the first stage waits for the predecessor's sink"
        );
        assert_eq!(edges.len(), 1 + SLOTS, "{previous:?} then {next:?}");
        for &(slot, transfer) in first {
            let reader = offer.2[slot].expect("the predecessor reads every slot");
            assert!(
                edges.contains(&Dependency {
                    predecessor: 0,
                    from_stage: reader,
                    to_stage: transfer
                }),
                "slot {slot} of {next:?} after {previous:?}: an edge from its last reader {reader:?}"
            );
            assert_ne!(
                reader, offer.1,
                "the reader is a unit before the sink: the copies overlap the tail"
            );
        }
    }
}

#[test]
fn an_mtp_draft_predecessor_gives_its_root() {
    let fixture = fixture();
    let plan = fixture.staged(&fixture.selection(), LaneId(9));
    let draft = offered(&plan, View::Draft { rows: 1 });
    assert!(draft.2.is_empty(), "a draft reads no staging slot");
    let first = plan.first_transfers(View::Verify { rows: 2 });
    let edges = external_edges(first, &[predecessor(&draft)]);
    for &(_, transfer) in first {
        assert!(
            edges.contains(&Dependency {
                predecessor: 0,
                from_stage: draft.0,
                to_stage: transfer
            }),
            "the transfer {transfer:?} is ordered behind the draft's root"
        );
    }
    assert_eq!(edges.len(), 1 + first.len());
}

/// Two sequences share the ring: a quantum behind both has edges from each, and from each one's own last reader.
#[test]
fn a_quantum_behind_two_predecessors_waits_for_each_ones_last_reader() {
    let fixture = fixture();
    let plan = fixture.staged(&fixture.selection(), LaneId(9));
    let one = offered(&plan, View::Decode);
    let mut two = offered(&plan, View::Verify { rows: 2 });
    two.0 = StageId(one.0.0 + 7);
    two.1 = StageId(one.1.0 + 5);
    two.2 = one
        .2
        .iter()
        .map(|reader| reader.map(|stage| StageId(stage.0 + 1)))
        .collect();
    assert_ne!(one.2, two.2, "the predecessors' readers differ");
    for next in [View::Decode, View::Verify { rows: 2 }] {
        let first = plan.first_transfers(next);
        let edges = external_edges(first, &[predecessor(&one), predecessor(&two)]);
        assert_eq!(edges.len(), 2 * (1 + first.len()), "{next:?}");
        for (index, offer) in [&one, &two].into_iter().enumerate() {
            assert!(
                edges.contains(&Dependency {
                    predecessor: index,
                    from_stage: offer.1,
                    to_stage: StageId(0)
                }),
                "{next:?}: the sink of predecessor {index}"
            );
            for &(slot, transfer) in first {
                assert!(
                    edges.contains(&Dependency {
                        predecessor: index,
                        from_stage: offer.2[slot].expect("every slot is read"),
                        to_stage: transfer
                    }),
                    "{next:?}: slot {slot} waits for predecessor {index}'s last reader"
                );
            }
        }
    }
}

#[test]
fn with_no_ring_the_edge_set_is_todays() {
    let fixture = fixture();
    let plan = fixture.unstaged();
    let (one, two) = (offered(&plan, View::Decode), offered(&plan, View::Verify { rows: 2 }));
    assert!(one.2.is_empty() && two.2.is_empty());
    for view in [View::Decode, View::PrefillChunk, View::Verify { rows: 2 }] {
        assert!(plan.first_transfers(view).is_empty());
        let edges = external_edges(plan.first_transfers(view), &[predecessor(&one), predecessor(&two)]);
        let today: Vec<Dependency> = [&one, &two]
            .iter()
            .enumerate()
            .map(|(predecessor, offer)| Dependency {
                predecessor,
                from_stage: offer.1,
                to_stage: StageId(0),
            })
            .collect();
        assert_eq!(
            edges.to_vec(),
            today,
            "{view:?}: each predecessor's sink to the first stage, and no more"
        );
    }
}

#[test]
fn the_marker_rings_hold_a_staged_views_stages() {
    let fixture = fixture();
    let (staged, unstaged) = (fixture.staged(&fixture.selection(), LaneId(9)), fixture.unstaged());
    let total: usize = staged.stages(View::Decode).unwrap().len();
    for sequences in [1, 2, 4] {
        let compute = markers_in_flight(&staged, sequences, false);
        assert_eq!(
            compute,
            markers_in_flight(&unstaged, sequences, false),
            "the stage lanes' markers are the unstaged view's: the transfers are not on them"
        );
        let transfers = markers_in_flight(&staged, sequences, true);
        assert!(
            transfers > 0 && transfers <= compute,
            "{transfers} transfers, {compute} compute"
        );
        assert_eq!(markers_in_flight(&unstaged, sequences, true), 0);
    }
    let compute_stages = unstaged.stages(View::Decode).unwrap().len();
    assert!(
        total > compute_stages,
        "the staged view has {total} stages, more than the {compute_stages} the rings are sized from"
    );
}

/// With every stageable projection staged, the markers fit the rings the engine sizes for the sequences, and a ring
/// one event short of them is refused with the lane named.
#[test]
fn the_marker_check_holds_every_staged_projection() {
    let fixture = fixture();
    let mut all: BTreeSet<String> = fixture
        .resident
        .device
        .iter()
        .filter(|placement| placement.name.starts_with("text/layers/"))
        .map(|placement| placement.name.clone())
        .collect();
    // Every object the plan can stage: those it names as not stageable (norms, convolutions) stay on the device.
    let plan = loop {
        match fixture.try_staged(&all, LaneId(9)) {
            Ok(plan) => break plan,
            Err(PlanBuildError::NotStageable { name }) => assert!(all.remove(&name), "{name}"),
            Err(error) => panic!("{error}"),
        }
    };
    assert!(
        all.len() > fixture.selection().len(),
        "more is staged than the usual selection"
    );
    for sequences in [1, 2, 4] {
        let ring = stage_ring_for(sequences);
        let (compute, transfers) = (
            markers_in_flight(&plan, sequences, false),
            markers_in_flight(&plan, sequences, true),
        );
        assert!(transfers > 0, "{sequences} sequences");
        assert_eq!(
            check_marker_rings(&plan, sequences, ring),
            Ok(()),
            "{sequences} sequences"
        );
        // The transfers are fewer than the compute stages, so the stage lanes' ring bounds the staging lane's too: the
        // staging check cannot trip before the stage check, and a ring one short of the compute markers is refused.
        assert!(transfers < compute, "{transfers} transfers, {compute} compute markers");
        let error = check_marker_rings(&plan, sequences, compute - 1).unwrap_err();
        assert!(error.contains("on a stage lane"), "{error}");
    }
}

#[test]
fn the_staging_ring_is_refused_without_the_buffer_or_the_host_arena() {
    let geometry = Some(StagingGeometry {
        slots: 4,
        slot_bytes: 1 << 20,
    });
    let base = Some(DevicePtr::NULL.offset(1 << 32));
    assert_eq!(staging_ring(None, None, false), Ok(None));
    let ring = staging_ring(geometry, base, true).unwrap().unwrap();
    assert_eq!((ring.slots, ring.slot_bytes), (4, 1 << 20));
    assert!(staging_ring(geometry, base, false).unwrap_err().contains("host arena"));
    assert!(staging_ring(geometry, None, true).is_err());
}

// The staged plan's own graph on the fake backend.

struct Q(u32);

/// One unit of the graph: logs that it ran, after a submission on its lane. A unit named in `held` is held at its gate
/// for the quantum until the test releases it.
struct Unit {
    index: usize,
    kernel: FakeKernel,
    word: u64,
    log: Arc<Mutex<Vec<(u32, usize)>>>,
    held: Arc<Mutex<Vec<(u32, usize)>>>,
    gates: Arc<Mutex<HashMap<(u32, usize), Gate>>>,
}

impl Stage<Q, FakeQueue> for Unit {
    fn run(&self, quantum: &Q, cx: &StageCx<'_, FakeQueue>) -> Result<Submitted, StageError> {
        let lane = cx.lane_handle().expect("a stage with a lane");
        let word = self.word;
        let made = submit_on(&**lane, cx.waits.iter().copied(), &Unmarked::new(16), |queue| {
            // SAFETY: a live fake allocation.
            unsafe {
                queue.launch(
                    &self.kernel,
                    Launch::linear(1, 1),
                    &Args::new().device(word).device(word).device(word).arg(2u32),
                )
            }
            .map_err(|error| StageError::Fatal(error.to_string()))
        });
        if made.is_ok() {
            lock(&self.log).push((quantum.0, self.index));
        }
        made.map(Submitted::marker)
    }

    fn pass_gate(&self, quantum: &Q, gate: GateRef<'_>) -> bool {
        if lock(&self.held).contains(&(quantum.0, self.index)) {
            lock(&self.gates).insert((quantum.0, self.index), gate.hold());
            return false;
        }
        true
    }
}

/// What the rig handed to the lake and what came back: every frame is counted in and counted out when it finishes.
#[derive(Default)]
struct Traffic {
    offered: AtomicUsize,
    finished: AtomicUsize,
}

impl Traffic {
    /// Whether every frame handed to the lake has finished.
    fn balanced(&self) -> bool {
        self.finished.load(Ordering::SeqCst) == self.offered.load(Ordering::SeqCst)
    }
}

/// A frame that counts its own end and is otherwise the frame it wraps.
struct Counted {
    inner: Box<dyn Frame>,
    traffic: Arc<Traffic>,
}

impl Frame for Counted {
    fn header(&self) -> &FrameHeader {
        self.inner.header()
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        self.inner.header_mut()
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        self.inner.execute()
    }

    fn is_alive(&self) -> bool {
        self.inner.is_alive()
    }

    fn kill(&self) {
        self.inner.kill();
    }

    fn do_finally(self: Box<Self>) {
        let Counted { inner, traffic } = *self;
        inner.do_finally();
        traffic.finished.fetch_add(1, Ordering::SeqCst);
    }

    fn do_finally_with_error(self: Box<Self>, error: FrameError) {
        let Counted { inner, traffic } = *self;
        inner.do_finally_with_error(error);
        traffic.finished.fetch_add(1, Ordering::SeqCst);
    }
}

/// The lake, counting what is offered to it.
struct Counting {
    lake: Arc<Lake>,
    traffic: Arc<Traffic>,
}

impl FrameSink for Counting {
    fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        self.traffic.offered.fetch_add(1, Ordering::SeqCst);
        let counted = Box::new(Counted {
            inner: frame,
            traffic: Arc::clone(&self.traffic),
        });
        self.lake.offer(counted).inspect_err(|_| {
            // Refused: the frame goes back to its owner, who does not run it.
            self.traffic.offered.fetch_sub(1, Ordering::SeqCst);
        })
    }
}

struct Rig {
    traffic: Arc<Traffic>,
    gpu: Arc<FakeGpu>,
    graph: Arc<StageGraph<Q, FakeQueue>>,
    lattice: Arc<ControlPlaneLattice>,
    lake: Arc<Lake>,
    unit: Option<UnitGuard>,
    completions: Arc<Completions>,
    log: Arc<Mutex<Vec<(u32, usize)>>>,
    held: Arc<Mutex<Vec<(u32, usize)>>>,
    gates: Arc<Mutex<HashMap<(u32, usize), Gate>>>,
    retired: Arc<Mutex<Vec<(u32, QuantumOutcome)>>>,
    view: View,
    plan: Arc<Plan>,
    /// The completions hold the lanes weakly: the rig keeps them alive.
    _lanes: Box<dyn Any>,
}

fn rig(view: View) -> Rig {
    let fixture = fixture();
    let compute = Plan::lane_width();
    // Two transfer lanes: the last is the staging lane.
    let gpu = FakeGpu::new();
    gpu.define("combine", |memory, _launch, args| {
        let at = args.read::<u64>(0).unwrap();
        let x = u32::from_le_bytes(memory.bytes_mut(at, 4)?.try_into().unwrap());
        memory.bytes_mut(at, 4)?.copy_from_slice(&(x + 1).to_le_bytes());
        Ok(())
    });
    let kernel = FakeModule::from_source(&gpu, "combine", &CompileOptions::default())
        .unwrap()
        .kernel("combine")
        .unwrap();
    let lanes = open_lane_set(&gpu, compute, 4096, 2).unwrap();
    let staging_lane = LaneId((compute + 2) as u16);
    let plan = Arc::new(fixture.staged(&fixture.selection(), staging_lane));
    let completions = Completions::new(&lanes.queues);
    let keep: Box<dyn Any> = Box::new((Arc::clone(&lanes.queues), Arc::clone(&lanes.typed)));
    let lattice = ControlPlaneLattice::new(lattice_config(&Settings::with_model(ModelSettings::default()))).unwrap();
    let lake = Lake::new(
        Arc::clone(&lattice),
        NonZeroUsize::new(2).unwrap(),
        Duration::from_secs(1),
    );
    lattice
        .add_upstream(Arc::clone(&completions) as Arc<dyn LatticeSource>)
        .unwrap();
    let unit = lake.admit().unwrap();
    let units = plan.unit_stages(view).unwrap().len();
    let word = gpu.alloc(4);
    let (log, held, gates): Logs = Default::default();
    let stages: Vec<Arc<dyn Stage<Q, FakeQueue>>> = (0..units)
        .map(|index| {
            Arc::new(Unit {
                index,
                kernel: kernel.clone(),
                word,
                log: Arc::clone(&log),
                held: Arc::clone(&held),
                gates: Arc::clone(&gates),
            }) as Arc<dyn Stage<Q, FakeQueue>>
        })
        .collect();
    let retired: Arc<Mutex<Vec<(u32, QuantumOutcome)>>> = Arc::default();
    let traffic = Arc::new(Traffic::default());
    let record = Arc::clone(&retired);
    let graph = StageGraph::new(
        plan.spec_homed(view, 0, compute).unwrap().unwrap(),
        stages,
        GraphRuntime {
            stages: Arc::new(Counting {
                lake: Arc::clone(&lake),
                traffic: Arc::clone(&traffic),
            }),
            completions: Arc::clone(&completions),
            lanes: Arc::clone(&lanes.typed),
            retries: Arc::default(),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        },
        move |quantum: Q, outcome| lock(&record).push((quantum.0, outcome)),
    )
    .unwrap();
    Rig {
        traffic,
        gpu,
        graph,
        lattice,
        lake,
        unit: Some(unit),
        completions,
        log,
        held,
        gates,
        retired,
        view,
        plan,
        _lanes: keep,
    }
}

impl Rig {
    /// Starts quantum `id` behind `after` (each with the quantum it names), with the edges the executor gives it.
    fn start(&self, id: u32, kill: &Arc<AtomicBool>, after: &[(&QuantumRef, &Offer)]) -> QuantumRef {
        let predecessors: Vec<Predecessor<'_>> = after.iter().map(|(_, offer)| predecessor(offer)).collect();
        let external = external_edges(self.plan.first_transfers(self.view), &predecessors)
            .into_iter()
            .map(|edge| ExternalEdge {
                from: after[edge.predecessor].0.clone(),
                from_stage: edge.from_stage,
                to_stage: edge.to_stage,
                kind: EdgeKind::Submission,
            });
        self.graph.start(Q(id), Arc::clone(kill), external)
    }

    fn offer(&self) -> Offer {
        offered(&self.plan, self.view)
    }

    /// Runs the device until everything the rig handed to the lake has finished and no completion is in flight: what
    /// can run has run. Bounded: a frame that never finishes fails the test, naming what is left.
    fn settle(&self) {
        let deadline = Instant::now() + BOUND;
        loop {
            let ran = self.gpu.run_until_idle();
            if ran == 0 && self.completions.pending() == 0 && self.traffic.balanced() {
                return;
            }
            assert!(
                Instant::now() < deadline,
                "the frames never finished: {} offered, {} finished, {} completions pending",
                self.traffic.offered.load(Ordering::SeqCst),
                self.traffic.finished.load(Ordering::SeqCst),
                self.completions.pending()
            );
            thread::yield_now();
        }
    }

    fn settle_until(&self, done: impl Fn() -> bool) {
        let deadline = Instant::now() + BOUND;
        while !done() {
            let (ran, last) = {
                let log = lock(&self.log);
                (log.len(), log.last().copied())
            };
            let tail: Vec<String> = {
                let ops = self.gpu.log();
                ops.iter().rev().take(4).map(|op| format!("{op:?}")).collect()
            };
            assert!(
                Instant::now() < deadline,
                "pending {} idle {} tail {tail:?}; the quanta did not finish in time: {} units ran of {} per quantum, {:?} retired, {} held, last {:?}",
                self.completions.pending(),
                self.gpu.run_until_idle(),
                ran,
                self.plan.unit_stages(self.view).unwrap().len(),
                lock(&self.retired).iter().map(|(id, _)| *id).collect::<Vec<_>>(),
                lock(&self.gates).len(),
                last
            );
            self.settle();
        }
    }

    fn ran(&self, quantum: u32, unit: usize) -> bool {
        lock(&self.log).contains(&(quantum, unit))
    }

    fn release(&self, quantum: u32, unit: usize) {
        lock(&self.held).retain(|held| *held != (quantum, unit));
        lock(&self.gates)
            .remove(&(quantum, unit))
            .expect("the unit was held")
            .release();
    }
}

type Offer = (StageId, StageId, Vec<Option<StageId>>);

/// The unit of the predecessor whose slot is read last, and the other first transfers of the next quantum.
#[test]
fn a_copy_overlaps_the_tail_of_the_previous_quantum_and_never_its_reader() {
    let rig = rig(View::Decode);
    let offer = rig.offer();
    let (late_slot, late_reader) = offer
        .2
        .iter()
        .enumerate()
        .filter_map(|(slot, reader)| reader.map(|reader| (slot, usize::from(reader.0))))
        .max_by_key(|&(_, reader)| reader)
        .unwrap();
    let transfers: Vec<(usize, usize)> = rig
        .plan
        .first_transfers(rig.view)
        .iter()
        .map(|&(slot, unit)| (slot, usize::from(unit.0)))
        .collect();
    lock(&rig.held).push((1, late_reader));
    let first = rig.start(1, &Arc::default(), &[]);
    rig.settle();
    assert!(rig.ran(1, 0), "the first quantum started");
    assert!(!rig.ran(1, late_reader), "its last reader is held");
    let second = rig.start(2, &Arc::default(), &[(&first, &offer)]);
    drop(first);
    rig.settle_until(|| {
        transfers
            .iter()
            .filter(|&&(slot, _)| slot != late_slot)
            .all(|&(_, unit)| rig.ran(2, unit))
    });
    rig.settle();
    for &(slot, unit) in &transfers {
        if slot == late_slot {
            assert!(
                !rig.ran(2, unit),
                "the copy into slot {slot} ran while the previous quantum's last reader of it is held"
            );
        } else {
            assert!(
                rig.ran(2, unit),
                "the copy into slot {slot} waits only for its own reader, which has run: it overlaps the tail"
            );
        }
    }
    assert!(!rig.ran(2, 0), "the first stage waits for the previous quantum's sink");
    rig.release(1, late_reader);
    drop(second);
    rig.settle_until(|| lock(&rig.retired).len() == 2);
    assert!(
        lock(&rig.retired)
            .iter()
            .all(|(_, outcome)| *outcome == QuantumOutcome::Completed)
    );
}

/// A killed quantum's stages never run, and a later quantum's slot edges still arrive: its copies run and it retires.
#[test]
fn a_killed_predecessors_slot_edges_still_arrive() {
    let rig = rig(View::Decode);
    let offer = rig.offer();
    let reader = offer
        .2
        .iter()
        .flatten()
        .map(|reader| usize::from(reader.0))
        .min()
        .unwrap();
    lock(&rig.held).push((1, reader));
    let kill = Arc::new(AtomicBool::new(false));
    let first = rig.start(1, &kill, &[]);
    rig.settle();
    let second = rig.start(2, &Arc::default(), &[(&first, &offer)]);
    drop(first);
    rig.settle();
    for &(_, unit) in rig.plan.first_transfers(rig.view) {
        assert!(!rig.ran(2, usize::from(unit.0)), "the copies wait for the held reader");
    }
    kill.store(true, Ordering::Release);
    rig.release(1, reader);
    drop(second);
    rig.settle_until(|| lock(&rig.retired).len() == 2);
    let retired = lock(&rig.retired);
    let outcome = |id: u32| {
        retired
            .iter()
            .find(|(quantum, _)| *quantum == id)
            .map(|(_, outcome)| outcome.clone())
    };
    assert_eq!(
        outcome(2),
        Some(QuantumOutcome::Completed),
        "the next quantum ran to its end"
    );
    assert_ne!(
        outcome(1),
        Some(QuantumOutcome::Completed),
        "the killed quantum did not"
    );
    assert!(rig.ran(2, 0));
    assert!(!rig.ran(1, reader));
}

#[test]
fn control_one_quantum_alone_retires() {
    let rig = rig(View::Decode);
    drop(rig.start(1, &Arc::default(), &[]));
    rig.settle_until(|| lock(&rig.retired).len() == 1);
}

#[test]
fn control_a_second_quantum_after_a_finished_one_retires() {
    let rig = rig(View::Decode);
    let offer = rig.offer();
    let first = rig.start(1, &Arc::default(), &[]);
    rig.settle();
    let second = rig.start(2, &Arc::default(), &[(&first, &offer)]);
    drop((first, second));
    rig.settle_until(|| lock(&rig.retired).len() == 2);
}

#[test]
fn control_a_second_quantum_with_no_edges_retires() {
    let rig = rig(View::Decode);
    let first = rig.start(1, &Arc::default(), &[]);
    let second = rig.start(2, &Arc::default(), &[]);
    drop((first, second));
    rig.settle_until(|| lock(&rig.retired).len() == 2);
}

impl Drop for Rig {
    fn drop(&mut self) {
        self.unit.take();
        self.lake.complete_gracefully();
        self.lattice.close();
    }
}
