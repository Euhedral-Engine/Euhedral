//! The execution plan on the tensor table of the RCO artifact (`tests/fixtures/rco_table.tsv`): the stage lists of
//! its views, their edges, scratch regions and lanes, without a device.
//!
//! The expected orders are worked by hand from `ExecutionPlan.fullModel` rewritten by `prefillView` (the fused decode
//! and prefill views) and from `ExecutionPlan.mtpDraft` (the unfused MTP views).

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use std::collections::{BTreeSet, HashMap};
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, DeviceQueue, GraphRuntime, Join, Lake, LaneId, LanePool, Latch,
    QuantumOutcome, RetryCounts, Stage, StageCx, StageError, StageGraph, Submitted,
};
use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_engine::formats::{Nvfp4Linear, TrellisLinear};
use euhedral_inference_engine::lattice_config;
use euhedral_inference_engine::model::{LayerKind, ModelConfig, WeightRef, object_key};
use euhedral_inference_engine::plan::{Formats, Plan, StageKind, View, WeightLookup};
use euhedral_inference_engine::quantum::split_geometry;
use euhedral_inference_engine::residency::planner::{ResidencyPlan, plan as residency};
use euhedral_inference_settings::{ModelSettings, Settings, TrellisKernelSettings};

mod common;

use common::{config, rco_table, settings};

use euhedral_inference_engine::model::{LayerIndex, Role};

struct Described(HashMap<(LayerIndex, Role), WeightRef>);

impl WeightLookup for Described {
    fn weight(&self, layer: LayerIndex, role: Role) -> Option<&WeightRef> {
        self.0.get(&(layer, role))
    }
}

struct Built {
    config: Arc<ModelConfig>,
    residency: ResidencyPlan,
    plan: Plan,
}

/// The RCO artifact's objects as weights with no device memory behind them, and the plan over them (built once: the
/// plan of 64 layers and six views takes a moment in a debug build).
fn built() -> &'static Built {
    static BUILT: OnceLock<Built> = OnceLock::new();
    BUILT.get_or_init(|| {
        let config = Arc::new(config());
        let residency = residency(&config, &rco_table(), &settings(8192), 15 << 30).unwrap();
        let weights = Described(
            residency
                .device
                .iter()
                .filter_map(|placement| {
                    let key = object_key(&placement.name, 64)?;
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
        let plan = Plan::build(&config, &weights, &residency, &formats).unwrap();
        Built {
            config,
            residency,
            plan,
        }
    })
}

use StageKind::{
    Argmax, AttentionCausal, Embed, GdnConvolution, GdnGatedRmsNorm, GdnProjectControl, GdnRecurrence, KvAppend,
    Linear, MtpStem, QkNormRope, ResidualAdd, ResidualRmsNorm, RmsNormUnitOffset, SelectedToken, Swiglu,
};

/// The stages of one layer of the fused views, as `prefillView` leaves them: the layer's last stage is the residual
/// add fused with the next layer's input norm, or the residual add alone after the last layer.
fn fused_layer(gdn: bool, last: bool) -> Vec<StageKind> {
    let mut stages = if gdn {
        vec![
            GdnProjectControl,
            Linear,
            Linear,
            GdnConvolution,
            GdnRecurrence,
            GdnGatedRmsNorm,
            Linear,
        ]
    } else {
        vec![Linear, Linear, QkNormRope, KvAppend, AttentionCausal, Linear]
    };
    stages.extend([ResidualRmsNorm, Linear, Swiglu, Linear]);
    stages.push(if last { ResidualAdd } else { ResidualRmsNorm });
    stages
}

#[test]
fn decode_view_stage_order_matches_java() {
    let built = built();
    let mut expected = vec![Embed, RmsNormUnitOffset];
    for (index, kind) in built.config.layer_kinds.iter().enumerate() {
        let gdn = *kind == LayerKind::GatedDeltaNet;
        expected.extend(fused_layer(gdn, index + 1 == built.config.layer_kinds.len()));
    }
    expected.extend([RmsNormUnitOffset, Linear, Argmax]);
    assert_eq!(expected.len(), 2 + 48 * 12 + 16 * 11 + 3);
    for view in [
        View::Decode,
        View::Verify { rows: 5 },
        View::Verify { rows: 9 },
        View::PrefillChunk,
    ] {
        assert_eq!(built.plan.kinds(view).unwrap(), expected, "{view:?}");
    }
}

#[test]
fn the_dependencies_of_the_decode_view_are_javas() {
    let built = built();
    let deps = built.plan.dependencies(View::Decode).unwrap();
    // 0 embedding, 1 the first input norm, then layer 0 (GDN): the joint projection and control, the query/key and
    // value/z projections, convolution, recurrence, gated norm, output, residual and norm, gate/up, SwiGLU, down,
    // and the residual fused with layer 1's input norm.
    let layer_0: Vec<Vec<usize>> = vec![
        vec![],
        vec![0],
        vec![1],
        vec![1],
        vec![1],
        vec![3, 4],
        vec![2, 5],
        vec![6, 4],
        vec![7],
        vec![0, 8],
        vec![9],
        vec![10],
        vec![11],
        vec![9, 12],
    ];
    assert_eq!(deps[..14], layer_0[..]);
    // Layer 1 starts at 14 and reads the fused node 13.
    assert_eq!(deps[14], vec![13]);
    assert_eq!(deps[15], vec![13]);
    assert_eq!(deps[16], vec![13]);
    assert_eq!(deps[17], vec![15, 16]);
    assert_eq!(deps[21], vec![13, 20], "the mixer's residual reads the layer's input");
    // Layer 3 is the first attention layer, from 38.
    let attention = 2 + 3 * 12;
    assert_eq!(deps[attention], vec![attention - 1], "query/key");
    assert_eq!(deps[attention + 1], vec![attention - 1], "gate/value");
    assert_eq!(deps[attention + 2], vec![attention], "Q/K norm and RoPE");
    assert_eq!(deps[attention + 3], vec![attention + 2, attention + 1], "append");
    assert_eq!(
        deps[attention + 4],
        vec![attention + 2, attention + 1, attention + 3],
        "attention"
    );
    assert_eq!(deps[attention + 5], vec![attention + 4]);
    // The tail: the last residual add, the final norm, the head, the selection.
    let count = deps.len();
    assert_eq!(deps[count - 3], vec![count - 4]);
    assert_eq!(deps[count - 2], vec![count - 3]);
    assert_eq!(deps[count - 1], vec![count - 2]);
    // Every edge points back, and the dependencies are among the edges.
    let edges = built.plan.edges(View::Decode).unwrap();
    for (stage, producers) in edges.iter().enumerate() {
        assert!(producers.iter().all(|producer| *producer < stage));
        assert!(deps[stage].iter().all(|dependency| producers.contains(dependency)));
    }
}

#[test]
fn the_mtp_views_are_the_unfused_layer() {
    let built = built();
    let expected = vec![
        Embed,
        MtpStem,
        Linear,
        RmsNormUnitOffset,
        Linear,
        Linear,
        QkNormRope,
        KvAppend,
        AttentionCausal,
        Linear,
        ResidualAdd,
        RmsNormUnitOffset,
        Linear,
        Swiglu,
        Linear,
        ResidualAdd,
        RmsNormUnitOffset,
        Linear,
        Argmax,
        SelectedToken,
    ];
    for view in [
        View::Draft { rows: 1 },
        View::Draft { rows: 8 },
        View::CatchUp { rows: 9 },
    ] {
        assert_eq!(built.plan.kinds(view).unwrap(), expected, "{view:?}");
    }
    let deps = built.plan.dependencies(View::Draft { rows: 1 }).unwrap();
    let wanted: Vec<Vec<usize>> = vec![
        vec![],
        vec![0],
        vec![1],
        vec![2],
        vec![3],
        vec![3],
        vec![4],
        vec![6, 5],
        vec![6, 5, 7],
        vec![8],
        vec![2, 9],
        vec![10],
        vec![11],
        vec![12],
        vec![13],
        vec![10, 14],
        vec![15],
        vec![16],
        vec![17],
        vec![18],
    ];
    assert_eq!(deps, wanted);
    assert!(built.plan.drafts());
}

#[test]
fn the_trellis_value_projection_has_a_scratch_region_of_its_own_in_decode_type_views() {
    let built = built();
    for (view, decode_type) in [
        (View::Decode, true),
        (View::Verify { rows: 5 }, true),
        (View::Verify { rows: 9 }, true),
        (View::PrefillChunk, false),
    ] {
        let kinds = built.plan.kinds(view).unwrap();
        let regions = built.plan.scratch_regions(view).unwrap();
        // Stages 2.. of layer 0: the second linear (index 4) is the value/z projection.
        assert_eq!(kinds[4], Linear);
        assert_eq!(regions[3], Some(0), "query/key: {view:?}");
        assert_eq!(regions[4], Some(usize::from(decode_type)), "value/z: {view:?}");
        // The LM head is NVFP4: it takes scratch only in the multi-row views.
        let head = regions.len() - 2;
        let multi_row = matches!(view, View::PrefillChunk | View::Verify { rows: 9 });
        assert_eq!(regions[head].is_some(), multi_row, "head: {view:?}");
    }
    let draft = built.plan.scratch_regions(View::Draft { rows: 3 }).unwrap();
    assert!(
        draft.iter().all(Option::is_none),
        "the MTP layer is NVFP4 and the draft is not multi-row"
    );
    let catch_up = built.plan.scratch_regions(View::CatchUp { rows: 12 }).unwrap();
    assert!(catch_up.iter().any(Option::is_some));
}

#[test]
fn fan_out_width_is_plan_width() {
    let built = built();
    let plan = &built.plan;
    let widths: Vec<usize> = [
        View::Decode,
        View::Verify { rows: 9 },
        View::PrefillChunk,
        View::Draft { rows: 1 },
        View::CatchUp { rows: 9 },
    ]
    .iter()
    .map(|view| plan.width(*view).unwrap())
    .collect();
    assert_eq!(plan.fan_out_width(), *widths.iter().max().unwrap());
    assert!(plan.fan_out_width() >= 2, "a GDN layer fans out: {widths:?}");
    assert_eq!(
        plan.width(View::Decode),
        Some(3),
        "the joint projection, query/key and value/z run at once"
    );
    assert_eq!(plan.fan_out_width(), 3);
    // Every graph names lanes within the pool the width sizes: the root only the home lane, the rest any of them.
    for view in [View::Decode, View::PrefillChunk, View::Draft { rows: 2 }] {
        let spec = plan.spec(view).unwrap();
        assert!(spec.validate().is_ok());
        let stages = spec.stages();
        assert_eq!(
            stages[0].lanes,
            vec![LaneId(0)],
            "{view:?}: the root enters on the home lane"
        );
        let all: Vec<LaneId> = (0..plan.fan_out_width()).map(|lane| LaneId(lane as u16)).collect();
        assert!(stages[1..].iter().all(|stage| stage.lanes == all), "{view:?}");
    }
    // The selection is the one sink, and the embedding the one root.
    assert_eq!(plan.roots(View::Decode).unwrap(), vec![0]);
    assert_eq!(
        plan.sink(View::Decode).unwrap(),
        plan.stages(View::Decode).unwrap().len() - 1
    );
}

#[test]
fn a_verification_selects_every_row_and_a_decode_the_last() {
    // The stage kinds are alike; the selection differs, so the two views have stages of their own.
    let built = built();
    let decode = built.plan.stages(View::Decode).unwrap();
    let verify = built.plan.stages(View::Verify { rows: 5 }).unwrap();
    let last = decode.len() - 1;
    assert!(!Arc::ptr_eq(&decode[last].stage, &verify[last].stage));
    assert_eq!(built.residency.draft_depth, 4);
    assert_eq!(built.plan.verified_rows(), 5);
}

#[test]
fn capture_key_changes_every_48_positions_below_2048() {
    let group = 24 / 4;
    let mut distinct = BTreeSet::new();
    let mut changes = 0;
    let mut previous = split_geometry(0, 1, group);
    distinct.insert(previous);
    for start in 1..2047u64 {
        let geometry = split_geometry(start, 1, group);
        if geometry != previous {
            assert_eq!(start % 48, 0, "the counts change on a multiple of 48 positions");
            changes += 1;
            previous = geometry;
            distinct.insert(geometry);
        }
    }
    assert_eq!((changes, distinct.len()), (42, 43));
}

#[test]
fn capture_key_constant_from_2048() {
    let group = 24 / 4;
    let wanted = split_geometry(2047, 1, group);
    for start in 2047..4200u64 {
        assert_eq!(split_geometry(start, 1, group), wanted, "{start}");
    }
    assert_eq!(split_geometry(131_070, 1, group), wanted);
}

#[test]
fn prefill_and_exact_never_captured() {
    // Prefill and the wide views have no graph; exact numerics is checked on a device by `capture_gpu`.
    assert!(!View::PrefillChunk.capturable());
    assert!(!View::Verify { rows: 9 }.capturable());
    assert!(!View::CatchUp { rows: 9 }.capturable());
    assert!(View::Decode.capturable() && View::Verify { rows: 5 }.capturable() && View::Draft { rows: 4 }.capturable());
}

/// The stages every chain of `edges` leads to the last one from: all of them, for a funnel.
fn ancestors_of_last(edges: &[Vec<usize>]) -> Vec<bool> {
    let mut reaches = vec![false; edges.len()];
    *reaches.last_mut().unwrap() = true;
    for stage in (0..edges.len()).rev() {
        if reaches[stage] {
            for &producer in &edges[stage] {
                reaches[producer] = true;
            }
        }
    }
    reaches
}

#[test]
fn every_stage_is_an_ancestor_of_the_sink_in_every_view() {
    let plan = &built().plan;
    for view in [
        View::Decode,
        View::Verify { rows: 5 },
        View::Verify { rows: 9 },
        View::PrefillChunk,
        View::Draft { rows: 1 },
        View::CatchUp { rows: 9 },
    ] {
        let edges = plan.edges(view).unwrap();
        let sink = plan.sink(view).unwrap();
        assert_eq!(sink, edges.len() - 1, "{view:?}");
        let reaches = ancestors_of_last(edges);
        let loose: Vec<usize> = (0..edges.len()).filter(|stage| !reaches[*stage]).collect();
        assert!(
            loose.is_empty(),
            "{view:?}: stages {loose:?} are not ancestors of the sink, so its marker would not cover them"
        );
        assert_eq!(plan.roots(view).unwrap(), vec![0], "{view:?}");
    }
}

#[test]
fn the_signalled_stages_are_the_ends_of_the_chains_of_lane_continuations() {
    let plan = &built().plan;
    for view in [
        View::Decode,
        View::Verify { rows: 5 },
        View::PrefillChunk,
        View::Draft { rows: 2 },
    ] {
        let signalled = plan.signalled(view).unwrap();
        let path = plan.path_predecessors(view).unwrap();
        // A stage ends its chain exactly when no stage continues its lane.
        for (stage, signal) in signalled.iter().enumerate() {
            let continued = path.contains(&Some(stage));
            assert_eq!(*signal, !continued, "{view:?}: stage {stage}");
        }
        // The selection ends the main chain and signals; so does every branch that ends in a join.
        assert!(signalled[plan.sink(view).unwrap()]);
        let count = signalled.iter().filter(|signal| **signal).count();
        assert!(
            count * 3 < signalled.len(),
            "{view:?}: {count} of {} stages signal",
            signalled.len()
        );
    }
}

/// A stage that records the lane the graph gave it.
struct Records {
    index: usize,
    lanes: Arc<Vec<AtomicU32>>,
}

impl Stage<()> for Records {
    fn run(&self, _quantum: &(), cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        self.lanes[self.index].store(u32::from(cx.lane.map_or(u16::MAX, |lane| lane.0)), Ordering::Release);
        Ok(Submitted::done())
    }
}

#[derive(Debug)]
struct Idle;

impl DeviceQueue for Idle {
    fn completed(&self) -> u64 {
        u64::MAX
    }

    fn name(&self) -> &str {
        "idle"
    }
}

/// The lane RustEE's graph gives each stage of the plan's graph of `view`, run on a lattice of its own over lanes that
/// do nothing.
fn placed_by_rustee(plan: &Plan, view: View) -> Vec<u16> {
    let spec = plan.spec(view).unwrap();
    let count = spec.stages().len();
    let lanes = Arc::new((0..count).map(|_| AtomicU32::new(u32::MAX)).collect::<Vec<_>>());
    let stages: Vec<Arc<dyn Stage<()>>> = (0..count)
        .map(|index| {
            Arc::new(Records {
                index,
                lanes: Arc::clone(&lanes),
            }) as Arc<dyn Stage<()>>
        })
        .collect();
    let settings = Settings::with_model(ModelSettings::default());
    let lattice = ControlPlaneLattice::new(lattice_config(&settings)).unwrap();
    let lake = Lake::new(
        Arc::clone(&lattice),
        NonZeroUsize::new(2).unwrap(),
        Duration::from_secs(1),
    );
    let pool = Arc::new(LanePool::new(
        (0..plan.fan_out_width())
            .map(|_| Arc::new(Idle) as Arc<dyn DeviceQueue>)
            .collect(),
    ));
    let completions = Completions::new(Arc::clone(&pool));
    lattice
        .add_upstream(Arc::clone(&completions) as Arc<dyn LatticeSource>)
        .unwrap();
    let (done, finished) = Latch::new();
    let join = Join::new(1, move |mut parts: Vec<Option<QuantumOutcome>>| {
        done.set(parts.remove(0).unwrap())
    });
    let graph = StageGraph::new(
        spec,
        stages,
        GraphRuntime {
            stages: Arc::clone(&lake) as Arc<dyn FrameSink>,
            completions,
            lanes: pool,
            retries: Arc::new(RetryCounts::default()),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        },
        move |(), outcome| join.arrive(0, outcome),
    )
    .unwrap();
    let unit = lake.admit().unwrap();
    let _ = graph.start((), Arc::new(AtomicBool::new(false)), Vec::new());
    assert_eq!(finished.wait(), QuantumOutcome::Completed);
    drop(unit);
    lake.complete_gracefully();
    lattice.close();
    lanes.iter().map(|lane| lane.load(Ordering::Acquire) as u16).collect()
}

#[test]
fn rustee_places_the_stages_of_every_view_as_javas_path_policy_does() {
    let plan = &built().plan;
    for view in [
        View::Decode,
        View::Verify { rows: 5 },
        View::PrefillChunk,
        View::Draft { rows: 2 },
        View::CatchUp { rows: 9 },
    ] {
        let placed = placed_by_rustee(plan, view);
        let path = plan.path_predecessors(view).unwrap();
        let width = plan.fan_out_width() as u16;
        assert!(
            placed.iter().all(|lane| *lane < width),
            "{view:?}: every stage ran on a lane of the plan"
        );
        assert_eq!(placed[0], 0, "{view:?}: the root runs on the home lane");
        let mut continued = 0;
        let mut branches = 0;
        for (stage, predecessor) in path.iter().enumerate() {
            match predecessor {
                // The critical successor continues the lane of the predecessor it is critical for.
                Some(predecessor) => {
                    continued += 1;
                    assert_eq!(
                        placed[stage], placed[*predecessor],
                        "{view:?}: stage {stage} continues {predecessor}"
                    );
                }
                None if stage > 0 => branches += 1,
                None => {}
            }
        }
        assert_eq!(continued + branches + 1, placed.len(), "{view:?}");
        // The branches of the fan-outs are spread: the graph does not run on the lane of the main chain alone.
        let used: BTreeSet<u16> = placed.iter().copied().collect();
        // (The MTP layer has one fan-out of two branches, which may both take the main chain's lane.)
        assert!(
            used.len() > 1 || matches!(view, View::Draft { .. } | View::CatchUp { .. }),
            "{view:?}: the branches of a fan-out took lanes of their own: {used:?}"
        );
    }
}
