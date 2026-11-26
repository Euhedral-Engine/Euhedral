//! The execution plan on the tensor tables of the Q3 artifact and of its P2E2-compressed form
//! (`tests/fixtures/q3_table.tsv`, `tests/fixtures/q3_compressed_table.tsv`), without a device: every linear routes to
//! the grouped format, the scratch the residency plan reserves covers every launch of it, and a plan with no grouped
//! format refuses the artifact by name.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use std::collections::HashMap;
use std::sync::Arc;

use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_edrl::{WeightFormat, WeightLayout};
use euhedral_inference_engine::formats::{LinearFormat, LinearMode, Nvfp4Linear, Q3Linear, TrellisLinear};
use euhedral_inference_engine::model::{LayerIndex, Residence, Role, WeightRef, object_key};
use euhedral_inference_engine::plan::{Formats, Plan, PlanBuildError, StageKind, View, WeightLookup};
use euhedral_inference_engine::residency::planner::{ResidencyPlan, nvfp4, plan as residency};
use euhedral_inference_settings::TrellisKernelSettings;

mod common;

use common::{config, q3_table, settings};

struct Described(HashMap<(LayerIndex, Role), WeightRef>);

impl WeightLookup for Described {
    fn weight(&self, layer: LayerIndex, role: Role) -> Option<&WeightRef> {
        self.0.get(&(layer, role))
    }
}

fn weights(residency: &ResidencyPlan) -> Described {
    Described(
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
                        residence: Residence::Device,
                        trellis: None,
                    },
                ))
            })
            .collect(),
    )
}

fn formats(grouped: bool) -> Formats {
    let formats = Formats::new(
        Arc::new(TrellisLinear::from_settings(&TrellisKernelSettings::default(), 70)),
        Arc::new(Nvfp4Linear::with_native_route(true, 70)),
    );
    if grouped {
        formats.with_grouped(Arc::new(Q3Linear::with_mx_route(true, 70)))
    } else {
        formats
    }
}

fn residency_of(compressed: bool) -> ResidencyPlan {
    residency(&config(), &q3_table(compressed), &settings(8192), 15 << 30).unwrap()
}

fn plan_of(compressed: bool) {
    let config = Arc::new(config());
    let residency = residency_of(compressed);
    let weights = weights(&residency);
    let plan = Plan::build(&config, &weights, &residency, &formats(true), None, None).unwrap();
    for view in [View::Draft { rows: 3 }, View::CatchUp { rows: 12 }] {
        assert!(plan.scratch_regions(view).is_some(), "{view:?}");
    }
    let grouped: Vec<_> = residency
        .device
        .iter()
        .filter(|placement| {
            matches!(
                placement.format,
                WeightFormat::Q3G64Fp16 | WeightFormat::Q4G64Fp16 | WeightFormat::Q5G64Fp16
            )
        })
        .collect();
    assert!(
        grouped.len() > 300,
        "the artifact's linears are grouped: {}",
        grouped.len()
    );
    assert_eq!(
        grouped
            .iter()
            .any(|placement| placement.layout == WeightLayout::RowSplitP2e2V1),
        compressed
    );

    // The scratch the plan reserves covers every grouped launch up to the prefill chunk's rows.
    let q3 = Q3Linear::with_mx_route(true, 70);
    for rows in [1usize, 8, 9, 33, 128, 129, 512] {
        let mut largest = 0;
        for placement in &grouped {
            let key = object_key(&placement.name, 64);
            if key.is_none_or(|(_, role)| role == Role::Embedding) {
                continue;
            }
            let weight = WeightRef {
                name: placement.name.clone(),
                shape: placement.shape.clone(),
                dtype: placement.dtype,
                format: placement.format,
                layout: placement.layout,
                ptr: DevicePtr::NULL,
                bytes: placement.bytes,
                residence: Residence::Device,
                trellis: None,
            };
            largest = largest.max(q3.scratch_bytes(&weight, rows, LinearMode::default()) as u64);
        }
        assert!(
            residency.scratch_bytes >= largest,
            "{rows} rows: the plan reserves {} bytes and a launch takes {largest}",
            residency.scratch_bytes
        );
    }
}

#[test]
fn the_plan_of_a_q3_artifact_builds() {
    plan_of(false);
}

#[test]
fn the_plan_of_a_compressed_q3_artifact_builds() {
    plan_of(true);
}

#[test]
fn a_plan_with_no_grouped_format_names_the_tensor_it_refuses() {
    for compressed in [false, true] {
        let config = Arc::new(config());
        let residency = residency_of(compressed);
        let weights = weights(&residency);
        let error = Plan::build(&config, &weights, &residency, &formats(false), None, None).unwrap_err();
        assert!(
            matches!(&error, PlanBuildError::Format { name, .. } if name.starts_with("text/")),
            "{error}"
        );
    }
}

/// A Q3 gate and up projection and its SwiGLU are one node of every base view (the stage fuses them from sixty-four
/// rows and composes them below), and the MTP topology keeps them apart.
#[test]
fn the_gate_up_projection_of_a_q3_layer_is_one_node_with_its_swiglu() {
    for compressed in [false, true] {
        let config = Arc::new(config());
        let residency = residency_of(compressed);
        let weights = weights(&residency);
        let plan = Plan::build(&config, &weights, &residency, &formats(true), None, None).unwrap();
        for view in [View::PrefillChunk, View::Decode] {
            let kinds = plan.kinds(view).unwrap();
            let fused = kinds.iter().filter(|kind| **kind == StageKind::GateUpSwiglu).count();
            assert_eq!(fused, 64, "{view:?}, compressed {compressed}");
            assert!(!kinds.contains(&StageKind::Swiglu), "{view:?}");
        }
        let kinds = plan.kinds(View::Draft { rows: 3 }).unwrap();
        assert!(!kinds.contains(&StageKind::GateUpSwiglu));
    }
}

/// A compressed tensor of a format that has no compressed layout is refused where the plan is built.
#[test]
fn a_q4_or_q5_tensor_with_the_compressed_layout_is_refused_at_plan_build() {
    for format in [WeightFormat::Q4G64Fp16, WeightFormat::Q5G64Fp16] {
        let config = Arc::new(config());
        let residency = residency_of(false);
        let mut lookup = weights(&residency).0;
        let weight = lookup.get_mut(&(LayerIndex(0), Role::Down)).unwrap();
        weight.format = format;
        weight.layout = WeightLayout::RowSplitP2e2V1;
        let name = weight.name.clone();
        let error = Plan::build(&config, &Described(lookup), &residency, &formats(true), None, None).unwrap_err();
        assert!(
            matches!(&error, PlanBuildError::Format { name: refused, .. } if *refused == name),
            "{error}"
        );
    }
}

/// A plain NVFP4 token table of a width that is a multiple of 128 builds into the plan; one of another width, and an
/// SD4 one, are refused where the plan is built, by the table's name.
#[test]
fn an_nvfp4_embedding_is_checked_at_plan_build() {
    let config = Arc::new(config());
    let residency = residency_of(false);
    let build = |format: WeightLayout, hidden: u64| {
        let mut lookup = weights(&residency).0;
        let table = lookup.get_mut(&(LayerIndex::GLOBAL, Role::Embedding)).unwrap();
        let vocabulary = table.shape[0];
        table.format = WeightFormat::Nvfp4;
        table.layout = format;
        table.shape = vec![vocabulary, hidden];
        table.bytes = nvfp4::byte_size(vocabulary, hidden, format).unwrap();
        let name = table.name.clone();
        let built = Plan::build(&config, &Described(lookup), &residency, &formats(true), None, None);
        (name, built.map(|_| ()))
    };
    assert!(build(WeightLayout::RowSplitK128V1, 5120).1.is_ok());
    let (name, narrow) = build(WeightLayout::RowSplitK128V1, 5184);
    let error = narrow.unwrap_err().to_string();
    assert!(error.contains(&name) && error.contains("multiple of 128"), "{error}");
    let (name, sd4) = build(WeightLayout::RowSplitK128Sd4V1, 5120);
    let error = sd4.unwrap_err().to_string();
    assert!(error.contains(&name) && error.contains("SD4"), "{error}");
}
