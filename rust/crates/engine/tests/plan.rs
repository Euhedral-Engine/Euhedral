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
use std::sync::{Arc, OnceLock};

use euhedral_graph::LaneId;
use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_engine::formats::{Nvfp4Linear, TrellisLinear};
use euhedral_inference_engine::model::{LayerKind, ModelConfig, WeightRef, object_key};
use euhedral_inference_engine::plan::{Formats, Plan, StageKind, View, WeightLookup};
use euhedral_inference_engine::quantum::split_geometry;
use euhedral_inference_engine::residency::planner::{ResidencyPlan, plan as residency};
use euhedral_inference_settings::TrellisKernelSettings;

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
    // Every graph names lanes within the pool the width sizes, and runs its critical chain on the home lane.
    for view in [View::Decode, View::PrefillChunk, View::Draft { rows: 2 }] {
        let lanes = plan.placement(view).unwrap();
        assert!(lanes.iter().all(|lane| usize::from(lane.0) < plan.fan_out_width()));
        assert_eq!(lanes[0], LaneId(0));
        let spec = plan.spec(view).unwrap();
        assert_eq!(spec.stages().len(), lanes.len());
        assert!(
            spec.stages()
                .iter()
                .zip(lanes)
                .all(|(stage, lane)| stage.lanes == vec![*lane])
        );
        assert!(spec.validate().is_ok());
    }
    // The branches of a layer's fan-out do not all share a lane.
    let lanes = plan.placement(View::Decode).unwrap();
    assert!(lanes[2..5].iter().any(|lane| *lane != lanes[2]), "{:?}", &lanes[..8]);
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

/// The stages with no later stage on their lane that they precede, worked out from the edges by closure.
fn lane_sinks(edges: &[Vec<usize>], lanes: &[LaneId]) -> Vec<bool> {
    let count = edges.len();
    let mut precedes = vec![vec![false; count]; count];
    for (stage, producers) in edges.iter().enumerate() {
        for &producer in producers {
            precedes[producer][stage] = true;
        }
    }
    for through in 0..count {
        let onward = precedes[through].clone();
        for row in precedes.iter_mut() {
            if row[through] {
                for (reached, via) in row.iter_mut().zip(&onward) {
                    *reached |= *via;
                }
            }
        }
    }
    (0..count)
        .map(|stage| !(0..count).any(|later| precedes[stage][later] && lanes[later] == lanes[stage]))
        .collect()
}

#[test]
fn the_signalled_stages_are_exactly_the_sinks_of_their_lanes() {
    let plan = &built().plan;
    for view in [
        View::Decode,
        View::Verify { rows: 5 },
        View::PrefillChunk,
        View::Draft { rows: 2 },
    ] {
        let signalled = plan.signalled(view).unwrap();
        let wanted = lane_sinks(plan.edges(view).unwrap(), plan.placement(view).unwrap());
        assert_eq!(signalled, wanted.as_slice(), "{view:?}");
        let count = signalled.iter().filter(|signal| **signal).count();
        let lanes = plan.placement(view).unwrap();
        for lane in 0..plan.fan_out_width() {
            let used = lanes.iter().any(|placed| usize::from(placed.0) == lane);
            let ends = signalled
                .iter()
                .zip(lanes)
                .filter(|(signal, placed)| **signal && usize::from(placed.0) == lane)
                .count();
            assert_eq!(
                ends >= 1,
                used,
                "{view:?}: lane {lane} ends in a signalled stage when it has stages"
            );
        }
        // The selection ends its lane, and almost every other stage does not: a flag write per stage is not paid.
        assert!(signalled[plan.sink(view).unwrap()]);
        assert!(count <= 8, "{view:?}: {count} of {} stages signal", signalled.len());
    }
}
