//! The engine's lane submission on RustEE's shared lanes, on the fake backend: no GPU and no wall clock. A stage's wait
//! and its marked submission (`submit_on`, which `run_stage` is, and what it keeps of a failed submission), the lanes a startup opens, and a graph of two branches
//! run by several workers. Every wait is bounded, so a mutant that hangs fails fast.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "a test file: a failed unwrap is the failure, and the test locks its records and waits, bounded"
)]

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::thread;
use std::time::{Duration, Instant};

use euhedral_core::frames::{Frame, FrameError};
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_gpu::fake::{FakeGpu, FakeKernel, FakeModule, FakeOp, FakeQueue};
use euhedral_gpu::{
    Args, CompileOptions, Enqueue, GraphHold, LaneEnqueue, Launch, Module, Ran, SharedGraphCache, StreamOptions,
    open_shared_lanes,
};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, DeviceQueue, EdgeKind, GraphRuntime, LaneId, QuantumOutcome, Stage, StageCx,
    StageError, StageGraph, StageGraphSpec, StageSpec, Submitted,
};
use euhedral_inference_engine::quantum::home::{home_of, lane_for};
use euhedral_inference_engine::quantum::whole::{Launches, Site, run_cached};
use euhedral_inference_engine::runtime::open_lane_set;
use euhedral_inference_engine::stages::{Unmarked, submit_on};

/// The longest any wait in this file takes.
const BOUND: Duration = Duration::from_secs(20);

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(PoisonError::into_inner)
}

/// A device whose kernels combine words: `combine(a, b, out, op)` writes `out = a + b` for op 0, `a * 2 + b` for op 1,
/// and `out = a + 1` (b ignored) for op 2.
fn gpu() -> Arc<FakeGpu> {
    let gpu = FakeGpu::new();
    gpu.define("combine", |memory, _launch, args| {
        let (a, b, out) = (
            args.read::<u64>(0).unwrap(),
            args.read::<u64>(1).unwrap(),
            args.read::<u64>(2).unwrap(),
        );
        let op = args.read::<u32>(3).unwrap();
        let x = u32::from_le_bytes(memory.bytes_mut(a, 4)?.try_into().unwrap());
        let y = u32::from_le_bytes(memory.bytes_mut(b, 4)?.try_into().unwrap());
        let value = match op {
            0 => x + y,
            1 => x * 2 + y,
            _ => x + 1,
        };
        memory.bytes_mut(out, 4)?.copy_from_slice(&value.to_le_bytes());
        Ok(())
    });
    gpu
}

fn kernel(gpu: &Arc<FakeGpu>) -> FakeKernel {
    FakeModule::from_source(gpu, "combine", &CompileOptions::default())
        .unwrap()
        .kernel("combine")
        .unwrap()
}

fn word(gpu: &FakeGpu, at: u64) -> u32 {
    u32::from_le_bytes(gpu.read(at, 4).try_into().unwrap())
}

fn combine<E: LaneEnqueue<Kernel = FakeKernel>>(
    enqueue: &mut E,
    kernel: &FakeKernel,
    words: [u64; 3],
    op: u32,
) -> Result<(), StageError> {
    // SAFETY: live fake allocations.
    unsafe {
        enqueue.launch(
            kernel,
            Launch::linear(1, 1),
            &Args::new().device(words[0]).device(words[1]).device(words[2]).arg(op),
        )
    }
    .map_err(|error| StageError::Fatal(error.to_string()))
}

struct Sink(Producer<Box<dyn Frame>, Multi>);

impl FrameSink for Sink {
    fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        self.0.offer(frame)
    }
}

fn run_frame(mut frame: Box<dyn Frame>) {
    if !frame.is_alive() {
        frame.do_finally();
        return;
    }
    match frame.execute() {
        Ok(()) | Err(FrameError::Cancelled) => frame.do_finally(),
        Err(error) => frame.do_finally_with_error(error),
    }
}

/// A quantum: the words its stages combine. `base` is `base`, `base + 4`, ... in the device's memory: the input, the
/// two branches' results and the join's.
struct Batch {
    words: [u64; 4],
}

/// One stage: combines two words into a third on its lane, after the markers the graph says to wait for.
struct Combine {
    kernel: FakeKernel,
    inputs: (usize, usize),
    out: usize,
    op: u32,
    /// How many times the stage was told the lane was busy.
    retries: Arc<AtomicUsize>,
    /// How many stages have submitted their work.
    submitted: Arc<AtomicUsize>,
}

/// The record a stage that has no quantum to keep it in gives `submit_on`.
fn nothing() -> Unmarked {
    Unmarked::new(8)
}

impl Stage<Batch, FakeQueue> for Combine {
    fn run(&self, batch: &Batch, cx: &StageCx<'_, FakeQueue>) -> Result<Submitted, StageError> {
        let lane = cx.lane_handle().expect("a stage with a lane");
        let words = [
            batch.words[self.inputs.0],
            batch.words[self.inputs.1],
            batch.words[self.out],
        ];
        let made = submit_on(&**lane, cx.waits.iter().copied(), &nothing(), |queue| {
            combine(queue, &self.kernel, words, self.op)
        });
        if matches!(made, Err(StageError::Retry)) {
            self.retries.fetch_add(1, Ordering::SeqCst);
        }
        if made.is_ok() {
            self.submitted.fetch_add(1, Ordering::SeqCst);
        }
        made.map(Submitted::marker)
    }
}

/// The graph of a quantum with two branches: `root` (lane 0) writes `in + 1`; `left` (lane 0) and `right` (lane 1)
/// read it; `join` (lane 0) combines them.
struct Rig {
    gpu: Arc<FakeGpu>,
    graph: Arc<StageGraph<Batch, FakeQueue>>,
    frames: Consumer<Box<dyn Frame>, Multi>,
    completions: Arc<Completions>,
    retired: Arc<Mutex<Vec<QuantumOutcome>>>,
    submitted: Arc<AtomicUsize>,
}

fn rig(markers: u32) -> Rig {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let specs = [("a", markers), ("b", markers)].map(|(name, markers)| {
        (
            name,
            StreamOptions {
                markers,
                ..StreamOptions::default()
            },
        )
    });
    let lanes = open_shared_lanes(&gpu, &specs).unwrap();
    let completions = Completions::new(&lanes.queues);
    let queue = MpmcQueue::<Box<dyn Frame>>::bounded(1024);
    let frames: Consumer<Box<dyn Frame>, Multi> = queue.consumer();
    let mut spec = StageGraphSpec::new();
    let mut stage = |name: &'static str, lane: u16| {
        spec.stage(StageSpec {
            name,
            ordered: None,
            lanes: vec![LaneId(lane)],
        })
    };
    let (root, left, right, join) = (stage("root", 0), stage("left", 0), stage("right", 1), stage("join", 0));
    spec.edge(root, left, EdgeKind::Submission);
    spec.edge(root, right, EdgeKind::Submission);
    spec.edge(left, join, EdgeKind::Submission);
    spec.edge(right, join, EdgeKind::Submission);
    let retries = Arc::new(AtomicUsize::new(0));
    let submitted = Arc::new(AtomicUsize::new(0));
    let combine_stage = |inputs: (usize, usize), out: usize, op: u32| {
        Arc::new(Combine {
            kernel: kernel.clone(),
            inputs,
            out,
            op,
            retries: Arc::clone(&retries),
            submitted: Arc::clone(&submitted),
        }) as Arc<dyn Stage<Batch, FakeQueue>>
    };
    let retired = Arc::new(Mutex::new(Vec::new()));
    let log = Arc::clone(&retired);
    let graph = StageGraph::new(
        spec,
        vec![
            // Word 0 holds the input: root writes word 0 + 1 into word 1 ...
            combine_stage((0, 0), 1, 2),
            // ... left writes word 1 * 2 + word 0 into word 2 ... (op 1: a * 2 + b)
            combine_stage((1, 0), 2, 1),
            // ... right writes word 1 + word 0 into word 3 ...
            combine_stage((1, 0), 3, 0),
            // ... and join writes word 2 + word 3 into word 3.
            combine_stage((2, 3), 3, 0),
        ],
        GraphRuntime {
            stages: Arc::new(Sink(queue.producer())),
            completions: Arc::clone(&completions),
            lanes: Arc::clone(&lanes.pool),
            retries: Arc::default(),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        },
        move |_batch: Batch, outcome| lock(&log).push(outcome),
    )
    .unwrap();
    Rig {
        gpu,
        graph,
        frames,
        completions,
        retired,
        submitted,
    }
}

impl Rig {
    /// Starts `quanta` quanta whose input words are 1, 2, ... and runs them to retirement with `workers` threads
    /// polling the frames, while this thread runs the device and the completions. Returns what each quantum's join
    /// wrote.
    fn run(&self, quanta: usize, workers: usize) -> Vec<u32> {
        let batches: Vec<[u64; 4]> = (0..quanta)
            .map(|quantum| {
                let base = self.gpu.alloc(16);
                self.gpu.write(base, &(quantum as u32 + 1).to_le_bytes());
                [base, base + 4, base + 8, base + 12]
            })
            .collect();
        let stop = Arc::new(AtomicBool::new(false));
        let handles: Vec<_> = (0..workers)
            .map(|_| {
                let (frames, stop) = (self.frames.clone(), Arc::clone(&stop));
                thread::spawn(move || {
                    while !stop.load(Ordering::Acquire) {
                        match frames.poll() {
                            Some(frame) => run_frame(frame),
                            None => thread::yield_now(),
                        }
                    }
                })
            })
            .collect();
        for words in &batches {
            let _ = self
                .graph
                .start(Batch { words: *words }, Arc::new(AtomicBool::new(false)), None);
        }
        let deadline = Instant::now() + BOUND;
        while lock(&self.retired).len() < quanta {
            assert!(Instant::now() < deadline, "the quanta did not retire in time");
            // The device runs only once every stage has submitted: every wait is then enqueued (a marker that has
            // run already is not waited for), and what the log shows of the waits does not depend on timing.
            if self.submitted.load(Ordering::SeqCst) < 4 * quanta {
                thread::yield_now();
                continue;
            }
            self.gpu.run_until_idle();
            self.completions.reap();
            let mut frame = None;
            self.completions.pull(&mut |next| frame = Some(next), &mut |_| false, 1);
            if let Some(frame) = frame {
                run_frame(frame);
            }
            thread::yield_now();
        }
        stop.store(true, Ordering::Release);
        for handle in handles {
            handle.join().unwrap();
        }
        assert!(
            lock(&self.retired)
                .iter()
                .all(|outcome| *outcome == QuantumOutcome::Completed)
        );
        batches.iter().map(|words| word(&self.gpu, words[3])).collect()
    }
}

/// The two-branch quantum on four workers retires and equals what one worker makes: the join reads what both
/// branches wrote, whichever lane they ran on.
///
/// The fake runs the operations of all lanes in the order they were submitted, and a stage is submitted after its
/// predecessors, so the data alone would come out right with no wait at all. The device's log does not: the join, on
/// lane 0, must have waited for the marker of `right`, on lane 1, and `right` for the root's marker on lane 0, and
/// each marker waited for must have been made before the wait was reached. A join that skips its wait leaves no
/// `Await` on lane 0.
#[test]
fn a_two_branch_quantum_on_four_workers_equals_one_worker() {
    let quanta = 24;
    let single = rig(256).run(quanta, 1);
    let four_workers = rig(256);
    let four = four_workers.run(quanta, 4);
    // For input x: root writes x + 1 ... (op 2 of word 0): left = 2 * (x + 1) + x, right = (x + 1) + x, join = both.
    let expected: Vec<u32> = (1..=quanta as u32).map(|x| (2 * (x + 1) + x) + ((x + 1) + x)).collect();
    assert_eq!(single, expected, "one worker");
    assert_eq!(four, expected, "four workers");

    let log = four_workers.gpu.log();
    let waits = |lane: u16| -> Vec<u64> {
        log.iter()
            .filter_map(|op| match op {
                FakeOp::Await { lane: on, value } if *on == lane => Some(*value),
                _ => None,
            })
            .collect()
    };
    // Lane 1 holds `right`'s markers only (one a quantum), so every one of them is waited for by a join on lane 0,
    // after it was made; and `right`'s own wait is for a marker of lane 0.
    let marks: Vec<u64> = log
        .iter()
        .filter_map(|op| match op {
            FakeOp::Mark { lane: 1, value } => Some(*value),
            _ => None,
        })
        .collect();
    assert_eq!(marks.len(), quanta, "one marker of `right` a quantum");
    let join_waits = waits(0);
    for value in marks {
        let made = log.iter().position(|op| *op == FakeOp::Mark { lane: 1, value });
        let reached = log.iter().position(|op| *op == FakeOp::Await { lane: 0, value });
        assert!(
            made.is_some_and(|made| reached.is_some_and(|reached| made < reached)),
            "no join waited for marker {value} of lane 1, after it was made ({} waits on lane 0)",
            join_waits.len()
        );
    }
    assert!(!waits(1).is_empty(), "the right branches wait for the roots");
}

/// A lane whose ring has no free marker answers `Retry`, and nothing was submitted: the launch did not run, and once the
/// lane has completed work the same call goes through.
#[test]
fn a_busy_lane_makes_the_stage_retry_and_submits_nothing() {
    let gpu = gpu();
    let lanes = open_shared_lanes(
        &gpu,
        &[(
            "a",
            StreamOptions {
                markers: 2,
                ..StreamOptions::default()
            },
        )],
    )
    .unwrap();
    let lane = Arc::clone(lanes.pool.lane(LaneId(0)));
    // Two markers fill the ring: the device has run neither.
    let first = submit_on(&*lane, [], &nothing(), |_| Ok(())).unwrap();
    submit_on(&*lane, [], &nothing(), |_| Ok(())).unwrap();
    let launched = AtomicUsize::new(0);
    let busy = submit_on(&*lane, [], &nothing(), |_| {
        launched.fetch_add(1, Ordering::SeqCst);
        Ok(())
    });
    assert!(matches!(busy, Err(StageError::Retry)), "{busy:?}");
    assert_eq!(launched.load(Ordering::SeqCst), 0, "the launch did not run");
    // The device runs what the two markers queued and nothing else: the refused call enqueued neither work nor a mark.
    assert_eq!(gpu.run_until_idle(), 2, "nothing was submitted");
    // A marker of the lane's own is never waited for, and once the device has run the work, the stage goes through.
    let marker = submit_on(&*lane, [first], &nothing(), |_| {
        launched.fetch_add(1, Ordering::SeqCst);
        Ok(())
    })
    .unwrap();
    assert_eq!(launched.load(Ordering::SeqCst), 1);
    assert!(marker.value > first.value);
}

/// A launch that fails still publishes its marker, keeps it in the quantum's record, and the stage's error comes back:
/// the kernels it queued before failing are behind that marker.
#[test]
fn a_refused_launch_publishes_its_marker_keeps_it_and_returns_the_stages_error() {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let words = [gpu.alloc(4), gpu.alloc(4), gpu.alloc(4)];
    let lanes = open_shared_lanes(&gpu, &[("a", StreamOptions::default())]).unwrap();
    let lane = Arc::clone(lanes.pool.lane(LaneId(0)));
    let unmarked = Unmarked::new(1);
    assert_eq!(unmarked.latest(LaneId(0)), None);
    let refused = submit_on(&*lane, [], &unmarked, |queue| {
        combine(queue, &kernel, words, 2)?;
        Err(StageError::Corrupt("a bad packet".into()))
    });
    assert!(matches!(refused, Err(StageError::Corrupt(_))), "{refused:?}");
    let kept = unmarked
        .latest(LaneId(0))
        .expect("the failed submission's marker is kept");
    // The marker is on the device queue after the kernel, so a lane ordered behind it is ordered behind the kernel.
    gpu.run_until_idle();
    let log = gpu.log();
    let kernel_at = log.iter().position(|op| matches!(op, FakeOp::Launch { .. })).unwrap();
    let mark_at = log
        .iter()
        .position(|op| *op == FakeOp::Mark { lane: 0, value: kept })
        .expect("the marker was made");
    assert!(kernel_at < mark_at, "{log:?}");
    // A succeeding submission keeps nothing.
    let clean = Unmarked::new(1);
    submit_on(&*lane, [], &clean, |_| Ok(())).unwrap();
    assert_eq!(clean.latest(LaneId(0)), None);
}

/// A launch that panics after it queued work publishes its marker and keeps it, and the panic goes on.
#[test]
fn a_panicking_launch_publishes_its_marker_keeps_it_and_resumes_the_panic() {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let words = [gpu.alloc(4), gpu.alloc(4), gpu.alloc(4)];
    let lanes = open_shared_lanes(&gpu, &[("a", StreamOptions::default())]).unwrap();
    let lane = Arc::clone(lanes.pool.lane(LaneId(0)));
    let unmarked = Unmarked::new(1);
    let caught = catch_unwind(AssertUnwindSafe(|| {
        let _ = submit_on(&*lane, [], &unmarked, |queue| {
            combine(queue, &kernel, words, 2)?;
            panic!("the launch panicked");
        });
    }));
    assert!(caught.is_err(), "the panic is resumed");
    let kept = unmarked.latest(LaneId(0)).expect("the marker is kept");
    gpu.run_until_idle();
    assert!(gpu.log().contains(&FakeOp::Mark { lane: 0, value: kept }));
    // The lane is whole: the next submission goes through, behind the kept marker.
    let next = submit_on(&*lane, [], &Unmarked::new(1), |_| Ok(())).unwrap();
    assert!(next.value > kept);
}

/// A failed submission on a lane other than the home lane: abandonment orders the home lane behind its marker, on the
/// device, after the kernels the failed submission queued, and nothing is left kept.
#[test]
fn abandonment_orders_the_home_lane_behind_the_marker_of_a_failed_submission_on_another_lane() {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let words = [gpu.alloc(4), gpu.alloc(4), gpu.alloc(4)];
    let lanes = open_shared_lanes(
        &gpu,
        &[("a", StreamOptions::default()), ("b", StreamOptions::default())],
    )
    .unwrap();
    let home = Arc::clone(lanes.pool.lane(LaneId(0)));
    let other = Arc::clone(lanes.pool.lane(LaneId(1)));
    let unmarked = Unmarked::new(2);
    let failed = submit_on(&*other, [], &unmarked, |queue| {
        combine(queue, &kernel, words, 2)?;
        Err(StageError::Fatal("the second stage refused".into()))
    });
    assert!(matches!(failed, Err(StageError::Fatal(_))), "{failed:?}");
    let kept = unmarked.latest(LaneId(1)).unwrap();
    let queue_of = |lane: LaneId| Some(lanes.pool.lane(lane).id());
    unmarked.await_on(&*home, queue_of);
    assert_eq!(unmarked.latest(LaneId(1)), None, "an awaited marker is not kept");
    gpu.run_until_idle();
    let log = gpu.log();
    let mark_at = log
        .iter()
        .position(|op| *op == FakeOp::Mark { lane: 1, value: kept })
        .unwrap();
    let await_at = log
        .iter()
        .position(|op| *op == FakeOp::Await { lane: 0, value: kept })
        .expect("the home lane waits for the marker of the failed submission");
    assert!(mark_at < await_at, "{log:?}");
    // A second abandonment orders nothing more.
    unmarked.await_on(&*home, queue_of);
    gpu.run_until_idle();
    let awaits = gpu
        .log()
        .iter()
        .filter(|op| matches!(op, FakeOp::Await { lane: 0, .. }))
        .count();
    assert_eq!(awaits, 1);
}

/// Failures on several lanes: each lane's latest marker is kept and awaited; a failure on the home lane needs no wait.
#[test]
fn abandonment_awaits_the_latest_failed_marker_of_every_other_lane() {
    let gpu = gpu();
    let lanes = open_shared_lanes(
        &gpu,
        &[
            ("a", StreamOptions::default()),
            ("b", StreamOptions::default()),
            ("c", StreamOptions::default()),
        ],
    )
    .unwrap();
    let lane = |index: u16| Arc::clone(lanes.pool.lane(LaneId(index)));
    let unmarked = Unmarked::new(3);
    let fail = |lane: &Arc<_>| {
        let _ = submit_on(&**lane, [], &unmarked, |_| Err(StageError::Fatal("refused".into())));
    };
    let (home, one, two) = (lane(0), lane(1), lane(2));
    fail(&home);
    fail(&one);
    fail(&one);
    fail(&two);
    let (latest_one, latest_two) = (unmarked.latest(LaneId(1)).unwrap(), unmarked.latest(LaneId(2)).unwrap());
    unmarked.await_on(&*home, |lane| Some(lanes.pool.lane(lane).id()));
    gpu.run_until_idle();
    let awaited: Vec<FakeOp> = gpu
        .log()
        .into_iter()
        .filter(|op| matches!(op, FakeOp::Await { .. }))
        .collect();
    assert_eq!(
        awaited,
        vec![
            FakeOp::Await {
                lane: 0,
                value: latest_one
            },
            FakeOp::Await {
                lane: 0,
                value: latest_two
            },
        ]
    );
    assert_eq!(
        unmarked.latest(LaneId(0)),
        None,
        "the home lane's own work needs no wait"
    );
}

/// The owner's lane and the transfer lanes are never the home of a run, in either pool.
#[test]
fn the_owner_and_transfer_lanes_are_never_a_runs_home() {
    let gpu = FakeGpu::new();
    let (compute, transfers) = (3, 4);
    let set = open_lane_set(&gpu, compute, 8, transfers).unwrap();
    assert_eq!(set.typed.len(), compute + 1 + transfers);
    assert_eq!(set.queues.len(), compute + 1 + transfers);
    let mut seen = [false; 3];
    for _ in 0..200 {
        for home in [set.typed.next_home(), set.queues.next_home()] {
            assert!(usize::from(home.0) < compute, "lane {} is a home", home.0);
            seen[usize::from(home.0)] = true;
        }
    }
    assert_eq!(seen, [true; 3], "every compute lane is a home in turn");
}

/// Stages run as one submission of a lane end in one marker, and the launches before it queue in order.
#[test]
fn several_launches_in_one_submission_end_in_one_marker() {
    let gpu = gpu();
    let kernel = kernel(&gpu);
    let words = [gpu.alloc(4), gpu.alloc(4), gpu.alloc(4)];
    let lanes = open_shared_lanes(&gpu, &[("a", StreamOptions::default())]).unwrap();
    let lane = Arc::clone(lanes.pool.lane(LaneId(0)));
    let first = submit_on(&*lane, [], &nothing(), |_| Ok(())).unwrap();
    let marker = submit_on(&*lane, [], &nothing(), |queue| {
        // out = a + 1 three times over the same word: x = 0 -> 1 -> 2 -> 3 only if run in order.
        combine(queue, &kernel, [words[2], words[1], words[2]], 2)?;
        combine(queue, &kernel, [words[2], words[1], words[2]], 2)?;
        combine(queue, &kernel, [words[2], words[1], words[2]], 2)
    })
    .unwrap();
    assert_eq!(marker.value, first.value + 1, "one marker for the three launches");
    assert_eq!(gpu.run_until_idle(), 5, "two markers and three kernels");
    assert_eq!(word(&gpu, words[2]), 3);
}

/// A fixed start time for the quantum with `id`, so the tests do not depend on a clock.
fn clock_of(id: u64) -> u64 {
    id.wrapping_mul(1_000_003).wrapping_add(17)
}

/// The home of a quantum is taken from its id and start time: quanta spread over more than one home (and over all
/// of them, in proportion), and the same id and time have the same home.
#[test]
fn quanta_with_different_ids_have_different_homes() {
    for stage_lanes in [2usize, 3, 4, 7] {
        let mut counts = vec![0usize; stage_lanes];
        for id in 0..4000u64 {
            let home = home_of(id, clock_of(id), stage_lanes);
            assert!(home < stage_lanes, "home {home} of {stage_lanes} lanes");
            assert_eq!(
                home,
                home_of(id, clock_of(id), stage_lanes),
                "a quantum's home is a function of its id and time"
            );
            counts[home] += 1;
        }
        let even = 4000 / stage_lanes;
        for (home, count) in counts.iter().enumerate() {
            assert!(
                count.abs_diff(even) * 5 < even,
                "home {home} of {stage_lanes} has {count} of 4000 quanta"
            );
        }
    }
    assert_eq!(home_of(12345, 99, 1), 0, "one stage lane is every quantum's home");
}

/// A quantum's stages run on `(home + offset) mod stage_lanes`: a chain (one offset) keeps its lane, the branches of a
/// fan-out (different offsets, no more than the stage lanes) are on different lanes whatever the home, and no lane
/// outside the stage lanes (the owner's, the transfer lanes) is ever chosen.
#[test]
fn a_chain_keeps_its_lane_branches_stay_distinct_and_reserved_lanes_are_never_chosen() {
    let (compute, transfers) = (4, 3);
    for id in 0..200u64 {
        let home = home_of(id, clock_of(id), compute);
        // A chain of three stages shares offset 1: the same lane each time.
        let chain: Vec<usize> = [1usize, 1, 1]
            .iter()
            .map(|&offset| lane_for(home, offset, compute))
            .collect();
        assert!(
            chain.windows(2).all(|pair| pair[0] == pair[1]),
            "the chain moved: {chain:?}"
        );
        // A fan-out as wide as the stage lanes: every branch on its own lane, the root on the home.
        let branches: Vec<usize> = (0..compute).map(|offset| lane_for(home, offset, compute)).collect();
        assert_eq!(branches[0], home);
        let mut sorted = branches.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), compute, "branches shared a lane: {branches:?}");
        // The lanes after the stage lanes (capture, owner, transfers) are never among them.
        assert!(
            branches.iter().all(|&lane| lane < compute),
            "a reserved lane was chosen: {branches:?}"
        );
    }
    // RustEE's own rotation (it homes each run's retirement) is limited to the same lanes.
    let gpu = FakeGpu::new();
    let set = open_lane_set(&gpu, compute, 8, transfers).unwrap();
    assert_eq!(set.typed.len(), compute + 1 + transfers);
}

/// A quantum's launches as a graph body: one kernel writing `words[2] = words[0] + 1`, then, when `refuse`, a refusal.
struct Words {
    kernel: FakeKernel,
    words: [u64; 3],
    refuse: bool,
}

impl Launches<FakeGpu> for Words {
    fn launch<E: Enqueue<Kernel = FakeKernel>>(&mut self, enq: &mut E) -> Result<(), StageError> {
        combine(enq, &self.kernel, self.words, 2)?;
        if self.refuse {
            return Err(StageError::Fatal("the second stage refused".into()));
        }
        Ok(())
    }
}

/// A quantum run as a graph body is one marked submission on its lane, whether the cache ran it directly, recorded it
/// or replayed it, and every run computes from the words as they are then.
#[test]
fn a_quantum_run_as_a_graph_body_is_one_marked_submission_each_time() {
    let gpu = gpu();
    let words = [gpu.alloc(4), gpu.alloc(4), gpu.alloc(4)];
    let lanes = open_shared_lanes(&gpu, &[("a", StreamOptions::default())]).unwrap();
    let lane = Arc::clone(lanes.pool.lane(LaneId(0)));
    let cache = SharedGraphCache::<FakeGpu, u128>::with_capacity(4);
    cache.set_replay_as_recorded(true);
    let mut launches = Words {
        kernel: kernel(&gpu),
        words,
        refuse: false,
    };
    let mut ran = Vec::new();
    let mut last = None;
    for input in [10u32, 20, 30, 40] {
        gpu.write(words[0], &input.to_le_bytes());
        let mut hold = GraphHold::new();
        let (made, marker) = run_cached(
            &Site {
                gpu: &*gpu,
                lane: &*lane,
                waits: &[],
            },
            &cache,
            7,
            &mut launches,
            &mut hold,
        )
        .unwrap();
        assert_eq!(marker.lane, LaneId(0));
        gpu.run_until_idle();
        assert_eq!(
            word(&gpu, words[2]),
            input + 1,
            "{made:?}: the run computed from the current words"
        );
        if let Some(previous) = last.replace(marker) {
            assert!(marker.value > previous.value, "each run is a submission of its own");
        }
        ran.push(made);
    }
    assert_eq!(ran[0], Ran::Direct, "the first sight of a key runs directly");
    assert_eq!(ran[1], Ran::Recorded, "the second records");
    assert_eq!(&ran[2..], &[Ran::Replayed; 2], "then it replays");
    let stats = cache.stats();
    assert_eq!((stats.directs, stats.records, stats.replays), (1, 1, 2), "{stats:?}");
}

/// A body that fails after queueing work answers the body's own error (the lane is the home lane, which RustEE's
/// retirement covers); a lane with no free marker answers a retry and queues nothing.
#[test]
fn a_failure_in_a_graph_body_answers_its_error_and_a_busy_lane_retries() {
    let gpu = gpu();
    let words = [gpu.alloc(4), gpu.alloc(4), gpu.alloc(4)];
    let lanes = open_shared_lanes(
        &gpu,
        &[(
            "a",
            StreamOptions {
                markers: 2,
                ..StreamOptions::default()
            },
        )],
    )
    .unwrap();
    let lane = Arc::clone(lanes.pool.lane(LaneId(0)));
    let cache = SharedGraphCache::<FakeGpu, u128>::with_capacity(4);
    cache.set_replay_as_recorded(true);
    let mut launches = Words {
        kernel: kernel(&gpu),
        words,
        refuse: true,
    };
    let mut hold = GraphHold::new();
    let failed = run_cached(
        &Site {
            gpu: &*gpu,
            lane: &*lane,
            waits: &[],
        },
        &cache,
        1,
        &mut launches,
        &mut hold,
    );
    assert!(
        matches!(&failed, Err(StageError::Fatal(message)) if message == "the second stage refused"),
        "{failed:?}"
    );
    gpu.run_until_idle();

    // Both markers of the lane are held by submissions that have not run: the body is not queued.
    launches.refuse = false;
    for _ in 0..2 {
        submit_on(&*lane, [], &nothing(), |_| Ok(())).unwrap();
    }
    let mut hold = GraphHold::new();
    let retried = run_cached(
        &Site {
            gpu: &*gpu,
            lane: &*lane,
            waits: &[],
        },
        &cache,
        2,
        &mut launches,
        &mut hold,
    );
    assert!(matches!(retried, Err(StageError::Retry)), "{retried:?}");
}
