//! The execution plan on the tensor table of the unquantized BF16 artifact (`tests/fixtures/bf16_table.tsv`), without a
//! device: every projection routes to the row-exact BF16 format, a BF16 object the route cannot take is refused where
//! the plan is built, and the residency planner places the artifact's sizes across the device, host memory and the
//! artifact file.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use std::collections::HashMap;
use std::sync::Arc;

use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_edrl::{TensorDataType, WeightFormat, WeightLayout};
use euhedral_inference_engine::formats::{
    Bf16Linear, LinearFormat, LinearMode, Nvfp4Linear, ROW_EXACT_KERNELS, TokenEmbedding, TrellisLinear,
};
use euhedral_inference_engine::model::{LayerIndex, Residence, Role, WeightRef, object_key};
use euhedral_inference_engine::plan::{Formats, Plan, PlanBuildError, StageKind, View, WeightLookup};
use euhedral_inference_engine::residency::planner::{PlanInputs, ResidencyPlan, plan as residency, plan_with};
use euhedral_inference_settings::{Auto, ByteSize, SpeculationMode, TrellisKernelSettings};

mod common;

use common::{config, settings, table_of};

const TABLE: &str = include_str!("fixtures/bf16_table.tsv");

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

fn formats() -> Formats {
    Formats::new(
        Arc::new(TrellisLinear::from_settings(&TrellisKernelSettings::default(), 70)),
        Arc::new(Nvfp4Linear::with_native_route(true, 70)),
    )
}

/// The plan over the whole artifact on a device that holds it.
fn resident() -> ResidencyPlan {
    let mut settings = settings(8192);
    settings.speculation.mode = SpeculationMode::None;
    residency(&config(), &table_of(TABLE, |_| false), &settings, 64 << 30).unwrap()
}

fn build(lookup: HashMap<(LayerIndex, Role), WeightRef>, residency: &ResidencyPlan) -> Result<Plan, PlanBuildError> {
    Plan::build(
        &Arc::new(config()),
        &Described(lookup),
        residency,
        &formats(),
        None,
        None,
    )
}

fn pointer(value: u64) -> DevicePtr {
    // SAFETY: the launch is planned, never run: no address is dereferenced.
    unsafe { DevicePtr::from_raw(value) }
}

fn matrix(n: u64, k: u64) -> WeightRef {
    WeightRef {
        name: format!("text/layers/0/weight_{n}x{k}"),
        shape: vec![n, k],
        dtype: TensorDataType::Bf16,
        format: WeightFormat::Bf16,
        layout: WeightLayout::ContiguousLeV1,
        ptr: pointer(0x1000),
        bytes: 2 * n * k,
        residence: Residence::Device,
        trellis: None,
    }
}

#[test]
fn the_plan_of_a_bf16_artifact_builds_with_every_projection_on_a_row_exact_kernel() {
    let residency = resident();
    let projections: Vec<_> = residency
        .device
        .iter()
        .filter(|placement| placement.format == WeightFormat::Bf16 && placement.shape.len() == 2)
        .collect();
    assert!(!projections.is_empty(), "the artifact has BF16 matrices");
    for placement in projections {
        let weight = matrix(placement.shape[0], placement.shape[1]);
        if Bf16Linear::check(&weight).is_err() {
            // The gated-control inputs (a few rows wide) leave through their own launch, not this format.
            continue;
        }
        for rows in [1usize, 17] {
            let launch = Bf16Linear::launch(&weight, pointer(0x2000), pointer(0x3000), rows).unwrap();
            assert!(
                ROW_EXACT_KERNELS.iter().any(|(_, name)| *name == launch.symbol),
                "{} x {rows} rows launches {}",
                placement.name,
                launch.symbol
            );
        }
    }
    let plan = build(weights(&residency).0, &residency).unwrap();
    let kinds = plan.kinds(View::Decode).unwrap();
    assert!(kinds.contains(&StageKind::Linear) && kinds.contains(&StageKind::Embed));
}

#[test]
fn a_bf16_projection_the_route_cannot_take_is_refused_at_plan_build_by_name() {
    let residency = resident();
    // 5136 inputs: not divisible by 32. 5124 outputs: not divisible by 8.
    for (n, k) in [(5120u64, 5136u64), (5124, 17_408)] {
        let mut lookup = weights(&residency).0;
        let weight = lookup.get_mut(&(LayerIndex(0), Role::Down)).unwrap();
        weight.shape = vec![n, k];
        weight.bytes = 2 * n * k;
        let name = weight.name.clone();
        let error = build(lookup, &residency).unwrap_err();
        assert!(
            matches!(&error, PlanBuildError::Geometry(why) if why.contains(&name) && why.contains("divisible")),
            "{n} x {k}: {error}"
        );
    }
}

#[test]
fn a_bf16_object_whose_bytes_are_not_its_values_is_refused() {
    let mut weight = matrix(5120, 17_408);
    weight.bytes -= 2;
    let why = Bf16Linear::check(&weight).unwrap_err();
    assert!(why.contains(&weight.name) && why.contains("bytes"), "{why}");
    let mut vector = matrix(5120, 17_408);
    vector.shape = vec![5120 * 17_408];
    assert!(Bf16Linear::check(&vector).is_err());
}

#[test]
fn a_bf16_linear_names_the_value_that_is_not_bf16_contiguous_or_sized() {
    let mut weight = matrix(5120, 17_408);
    weight.format = WeightFormat::Q3G64Fp16;
    let why = Bf16Linear::check(&weight).unwrap_err();
    assert!(why.contains(&weight.name) && why.contains("Q3G64Fp16"), "{why}");
    let mut weight = matrix(5120, 17_408);
    weight.layout = WeightLayout::RowSplitK128V1;
    let why = Bf16Linear::check(&weight).unwrap_err();
    assert!(why.contains(&weight.name) && why.contains("RowSplitK128V1"), "{why}");
    let mut weight = matrix(5120, 17_408);
    weight.dtype = TensorDataType::Fp16;
    let why = Bf16Linear::check(&weight).unwrap_err();
    assert!(why.contains(&weight.name) && why.contains("Fp16"), "{why}");
    for (n, k) in [(0u64, 17_408u64), (5120, 0)] {
        let weight = matrix(n, k);
        let why = Bf16Linear::check(&weight).unwrap_err();
        assert!(
            why.contains(&weight.name) && why.contains(&format!("{n} x {k}")),
            "{why}"
        );
    }
}

#[test]
fn a_bf16_embedding_names_the_value_that_is_not_readable() {
    let residency = resident();
    let table = weights(&residency)
        .0
        .remove(&(LayerIndex::GLOBAL, Role::Embedding))
        .unwrap();
    let mut layout = table.clone();
    layout.layout = WeightLayout::RowSplitK128V1;
    let why = TokenEmbedding::check(&layout).unwrap_err();
    assert!(why.contains(&table.name) && why.contains("cannot read"), "{why}");
    let mut dtype = table.clone();
    dtype.dtype = TensorDataType::Fp16;
    let why = TokenEmbedding::check(&dtype).unwrap_err();
    assert!(why.contains(&table.name) && why.contains("cannot read"), "{why}");
    let mut bytes = table.clone();
    bytes.bytes -= 2;
    let why = TokenEmbedding::check(&bytes).unwrap_err();
    assert!(
        why.contains(&table.name) && why.contains(&bytes.bytes.to_string()),
        "{why}"
    );
}

#[test]
fn a_bf16_embedding_is_checked_at_plan_build() {
    let residency = resident();
    let mut lookup = weights(&residency).0;
    let table = lookup.get_mut(&(LayerIndex::GLOBAL, Role::Embedding)).unwrap();
    assert!(TokenEmbedding::check(table).is_ok());
    let name = table.name.clone();
    let vocabulary = table.shape[0];
    table.shape = vec![vocabulary, 5124];
    table.bytes = 2 * vocabulary * 5124;
    let error = build(lookup, &residency).unwrap_err().to_string();
    assert!(error.contains(&name) && error.contains("multiple of 8"), "{error}");
}

/// The row-exact kernels by the shape and the rows: split over K for `n <= 1536` or `n` not a multiple of 32, and four
/// row tiles above 16 rows.
#[test]
fn the_bf16_route_is_the_row_exact_kernel_of_the_shape_and_the_rows() {
    let wide = matrix(5120, 17_408);
    let narrow = matrix(1536, 5120);
    let odd = matrix(5128, 5120);
    for (weight, rows, symbol, grid) in [
        (&wide, 1usize, "euhedral_q4_linear_tc_bf16", [160, 1, 1]),
        (&wide, 16, "euhedral_q4_linear_tc_bf16", [160, 1, 1]),
        (&wide, 17, "euhedral_q4_linear_tc_rows_bf16", [160, 1, 1]),
        (&wide, 129, "euhedral_q4_linear_tc_rows_bf16", [160, 3, 1]),
        (&narrow, 1, "euhedral_q4_linear_tc_split_bf16", [192, 1, 1]),
        (&narrow, 65, "euhedral_q4_linear_tc_rows_split_bf16", [192, 2, 1]),
        (&odd, 2, "euhedral_q4_linear_tc_split_bf16", [641, 1, 1]),
    ] {
        let launch = Bf16Linear::launch(weight, pointer(0x2000), pointer(0x3000), rows).unwrap();
        assert_eq!(
            (launch.symbol, launch.grid),
            (symbol, grid),
            "{} x {} rows",
            weight.shape[0],
            rows
        );
        assert_eq!(launch.block, [128, 1, 1]);
    }
    assert_eq!(
        Bf16Linear.scratch_bytes(
            &wide,
            512,
            LinearMode {
                row_exact: true,
                prefill: true
            }
        ),
        0
    );
}

#[test]
fn the_bf16_route_refuses_misaligned_operands_and_no_rows() {
    let weight = matrix(5120, 17_408);
    assert!(Bf16Linear::launch(&weight, pointer(0x2008), pointer(0x3000), 1).is_err());
    assert!(Bf16Linear::launch(&weight, pointer(0x2000), pointer(0x3002), 1).is_err());
    assert!(Bf16Linear::launch(&weight, pointer(0x2000), pointer(0x3000), 0).is_err());
    let mut unplaced = matrix(5120, 17_408);
    unplaced.ptr = DevicePtr::NULL;
    assert!(Bf16Linear::launch(&unplaced, pointer(0x2000), pointer(0x3000), 1).is_err());
}

/// The artifact is 52 GiB: on a device of 28 GiB with 24000 MiB of host memory for the weights, the planner places it
/// across the device, the host and the file, and holds the device within its budget.
#[test]
fn the_planner_places_the_bf16_artifact_across_device_host_and_disk() {
    let mut settings = settings(4096);
    settings.speculation.mode = SpeculationMode::None;
    settings.memory.host_limit = Auto::Value(ByteSize::mib(24_000));
    let host = 24_000u64 << 20;
    let plan = plan_with(
        &config(),
        &table_of(TABLE, |_| false),
        &settings,
        PlanInputs {
            device_free: 28 << 30,
            multiprocessors: Some(170),
            exact_numerics: false,
            host_pinned_bytes: Some(host),
            sequences: 1,
        },
    )
    .unwrap();
    assert!(!plan.offload.host.is_empty() && !plan.offload.disk.is_empty());
    assert!(plan.device_bytes <= 28 << 30);
    assert!(plan.offload.host_bytes() <= host);
    assert!(plan.offload.holds_on_host("text/token_embedding"));
    assert_eq!(plan.scratch_bytes, 0);
}
