//! The plan of resident weights is what it was before the plan could stage weights: the node lists, edges, units,
//! lanes, width and capture rule of every view, compared with a snapshot of the plan taken from the revision before.
//!
//! `tests/fixtures/resident_plan.txt` was produced by running this file's `dump` over the plan built by the tree
//! immediately before the staging ring existed (`Plan::build` with four arguments), on the tensor table of the RCO
//! artifact with every weight on the device and a 8192-token context. The ignored test `print_the_snapshot` writes the
//! snapshot to the output of the run, for a new one.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Arc;

use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_engine::formats::{Nvfp4Linear, TrellisLinear};
use euhedral_inference_engine::model::{LayerIndex, Residence, Role, WeightRef, object_key};
use euhedral_inference_engine::plan::{Formats, Plan, View, WeightLookup};
use euhedral_inference_engine::residency::planner::plan as residency;
use euhedral_inference_settings::TrellisKernelSettings;

mod common;

use common::{config, rco_table, settings};

struct Described(HashMap<(LayerIndex, Role), WeightRef>);

impl WeightLookup for Described {
    fn weight(&self, layer: LayerIndex, role: Role) -> Option<&WeightRef> {
        self.0.get(&(layer, role))
    }
}

const VIEWS: [View; 8] = [
    View::Decode,
    View::Verify { rows: 2 },
    View::Verify { rows: 9 },
    View::PrefillChunk,
    View::Draft { rows: 1 },
    View::Draft { rows: 4 },
    View::CatchUp { rows: 9 },
    View::Verify { rows: 6 },
];

fn dump(plan: &Plan) -> String {
    let mut out = String::new();
    for view in VIEWS {
        writeln!(out, "view {view:?}").unwrap();
        let Some(kinds) = plan.kinds(view) else {
            writeln!(out, "  absent").unwrap();
            continue;
        };
        writeln!(out, "  width {:?} capturable {}", plan.width(view), view.capturable()).unwrap();
        writeln!(out, "  kinds {kinds:?}").unwrap();
        writeln!(out, "  edges {:?}", plan.edges(view).unwrap()).unwrap();
        writeln!(out, "  dependencies {:?}", plan.dependencies(view).unwrap()).unwrap();
        writeln!(out, "  units {:?}", plan.units(view).unwrap()).unwrap();
        writeln!(out, "  unit_edges {:?}", plan.unit_edges(view).unwrap()).unwrap();
        writeln!(out, "  paths {:?}", plan.path_predecessors(view).unwrap()).unwrap();
        writeln!(out, "  scratch {:?}", plan.scratch_regions(view).unwrap()).unwrap();
        for home in [0, 1] {
            let spec = plan.spec_homed(view, home, 5).unwrap().unwrap();
            let lanes: Vec<_> = spec.stages().iter().map(|stage| stage.lanes.clone()).collect();
            writeln!(out, "  lanes at home {home} {lanes:?}").unwrap();
        }
    }
    out
}

fn resident_plan() -> Plan {
    let config = Arc::new(config());
    let residency = residency(&config, &rco_table(), &settings(8192), 15 << 30).unwrap();
    let weights = Described(
        residency
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
        Arc::new(Nvfp4Linear::with_native_route(true)),
    );
    Plan::build(&config, &weights, &residency, &formats, None, None).unwrap()
}

#[test]
fn the_plan_of_resident_weights_is_the_snapshot() {
    let now = dump(&resident_plan());
    let expected = include_str!("fixtures/resident_plan.txt");
    if now != expected {
        let line = now.lines().zip(expected.lines()).position(|(a, b)| a != b);
        panic!("the resident plan differs from the snapshot at line {line:?}");
    }
}

#[test]
#[ignore = "prints the snapshot a new fixture is made from"]
fn print_the_snapshot() {
    print!("{}", dump(&resident_plan()));
}
