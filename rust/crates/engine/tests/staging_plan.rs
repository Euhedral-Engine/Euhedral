//! The plan of a model whose weights are partly held in host memory: each use of a staged weight reads a slot of the
//! staging ring, a transfer on the transfer lane fills it, and the edges keep a transfer behind the slot's last reader.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use std::collections::{BTreeSet, HashMap};
use std::path::Path;
use std::sync::Arc;

use euhedral_graph::LaneId;
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, DevicePtr, HostWeightsArena};
use euhedral_inference_engine::formats::{Nvfp4Linear, TrellisLinear};
use euhedral_inference_engine::model::{LayerIndex, ModelConfig, Residence, Role, WeightRef, Weights, object_key};
use euhedral_inference_engine::plan::{
    DECODE_MAX_ROWS, Formats, Plan, PlanBuildError, StageKind, StagingRing, View, WeightLookup,
};
use euhedral_inference_engine::residency::planner::{ResidencyPlan, plan as residency};
use euhedral_inference_settings::TrellisKernelSettings;

mod common;

use common::{config, rco_table, settings};

const TRANSFER_LANE: LaneId = LaneId(7);
const STAGEABLE: [&str; 5] = [
    "/gdn/output",
    "/mlp/down",
    "/mlp/gate_up",
    "/attention/output",
    "/gdn/value_z",
];

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
    /// The weights with the named objects held in host memory; the embedding and the head are read in place.
    fn lookup(&self, names: &BTreeSet<String>) -> Described {
        let mut weights = self.weights.0.clone();
        for (index, weight) in weights.values_mut().enumerate() {
            if names.contains(&weight.name) {
                weight.residence = if weight.name == "text/token_embedding" || weight.name == "text/output_head" {
                    Residence::Mapped
                } else {
                    Residence::Host {
                        offset: 4096 * index as u64,
                    }
                };
            }
        }
        Described(weights)
    }

    /// The names of the stageable layer projections whose name ends with one of `suffixes`.
    fn candidates(&self, suffixes: &[&str]) -> Vec<String> {
        self.resident
            .device
            .iter()
            .filter(|placement| {
                placement.name.starts_with("text/layers/")
                    && suffixes.iter().any(|suffix| placement.name.ends_with(suffix))
            })
            .map(|placement| placement.name.clone())
            .collect()
    }

    fn slot_bytes(&self, names: &BTreeSet<String>) -> u64 {
        self.resident
            .device
            .iter()
            .filter(|placement| names.contains(&placement.name))
            .map(|placement| placement.bytes)
            .max()
            .unwrap_or(0)
    }

    fn ring(&self, names: &BTreeSet<String>, slots: usize) -> StagingRing {
        StagingRing {
            base: DevicePtr::NULL.offset(1 << 32),
            slots,
            slot_bytes: self.slot_bytes(names),
        }
    }

    fn build(&self, names: &BTreeSet<String>, slots: usize) -> Result<Plan, PlanBuildError> {
        let ring = self.ring(names, slots);
        Plan::build(
            &self.config,
            &self.lookup(names),
            &self.resident,
            &self.formats,
            Some(&ring),
            Some(TRANSFER_LANE),
        )
    }

    fn unstaged(&self) -> Plan {
        Plan::build(&self.config, &self.weights, &self.resident, &self.formats, None, None).unwrap()
    }
}

fn set(names: impl IntoIterator<Item = String>) -> BTreeSet<String> {
    names.into_iter().collect()
}

const BASE_VIEWS: [View; 6] = [
    View::Decode,
    View::Verify { rows: 2 },
    View::Verify { rows: 9 },
    View::PrefillChunk,
    View::Verify { rows: DECODE_MAX_ROWS },
    View::Verify {
        rows: DECODE_MAX_ROWS + 1,
    },
];

/// Whether each node of a graph is reached from each other: `reach[a]` holds the nodes before `a`.
fn ancestors(edges: &[Vec<usize>]) -> Vec<Vec<bool>> {
    let mut reach: Vec<Vec<bool>> = Vec::with_capacity(edges.len());
    for (node, before) in edges.iter().enumerate() {
        let mut row = vec![false; edges.len()];
        for &producer in before {
            assert!(producer < node, "an edge goes to a later node");
            row[producer] = true;
            for (index, reached) in reach[producer].iter().enumerate() {
                row[index] |= reached;
            }
        }
        reach.push(row);
    }
    reach
}

/// What a view's transfers must keep: each transfer follows the reader of the use `slots` before it, and each consumer
/// follows its transfer. Returns the first violation, on the node edges and on the unit edges.
fn violation(plan: &Plan, view: View, slots: usize, edges: &[Vec<usize>]) -> Option<String> {
    let kinds = plan.kinds(view).unwrap();
    let dependencies = plan.dependencies(view).unwrap();
    let transfers: Vec<usize> = (0..kinds.len())
        .filter(|&node| kinds[node] == StageKind::WeightTransfer)
        .collect();
    let consumers: Vec<usize> = transfers
        .iter()
        .map(|&transfer| {
            let found: Vec<usize> = (0..kinds.len())
                .filter(|&node| dependencies[node].contains(&transfer))
                .collect();
            assert_eq!(found.len(), 1, "one consumer for transfer {transfer}");
            found[0]
        })
        .collect();
    let node_reach = ancestors(edges);
    for (index, &transfer) in transfers.iter().enumerate() {
        if !node_reach[consumers[index]][transfer] {
            return Some(format!(
                "{view:?}: the consumer of use {index} does not follow its transfer"
            ));
        }
        if index >= slots && !node_reach[transfer][consumers[index - slots]] {
            return Some(format!(
                "{view:?}: the transfer of use {index} overtakes the reader of use {}",
                index - slots
            ));
        }
    }
    // The same, over the units the graph runs.
    let units = plan.units(view).unwrap();
    let unit_edges = plan.unit_edges(view).unwrap();
    let unit_of: Vec<usize> = {
        let mut of = vec![usize::MAX; kinds.len()];
        for (unit, nodes) in units.iter().enumerate() {
            for &node in nodes {
                of[node] = unit;
            }
        }
        of
    };
    let unit_reach = ancestors(unit_edges);
    let before = |a: usize, b: usize| {
        let (ua, ub) = (unit_of[a], unit_of[b]);
        if ua == ub { a < b } else { unit_reach[ub][ua] }
    };
    for (index, &transfer) in transfers.iter().enumerate() {
        if !before(transfer, consumers[index]) {
            return Some(format!(
                "{view:?}: unit order: the consumer of use {index} does not follow its transfer"
            ));
        }
        if index >= slots && !before(consumers[index - slots], transfer) {
            return Some(format!(
                "{view:?}: unit order: the transfer of use {index} overtakes a reader"
            ));
        }
    }
    None
}

struct Xorshift(u64);

impl Xorshift {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn below(&mut self, bound: usize) -> usize {
        (self.next() % bound as u64) as usize
    }
}

#[test]
fn a_transfer_never_overtakes_its_slots_reader() {
    let fixture = fixture();
    let all = fixture.candidates(&STAGEABLE);
    let resident = fixture.unstaged();
    let mut random = Xorshift(0x9e37_79b9_7f4a_7c15);
    for case in 0..12 {
        let density = 1 + random.below(4);
        let chosen = set(all.iter().filter(|_| random.below(5) < density).cloned());
        if chosen.is_empty() {
            continue;
        }
        let slots = [1, 2, 3, 5][random.below(4)];
        let plan = fixture.build(&chosen, slots).unwrap();
        for view in BASE_VIEWS {
            check_units(&plan, &resident, view);
            let edges = plan.edges(view).unwrap();
            assert_eq!(violation(&plan, view, slots, edges), None, "case {case}, {slots} slots");
            let transfers = plan
                .kinds(view)
                .unwrap()
                .iter()
                .filter(|kind| **kind == StageKind::WeightTransfer)
                .count();
            assert!(transfers > 0);
            // A checker that cannot see a dropped hazard checks nothing: with each transfer's edges cut, it must fail
            // as soon as a slot is reused.
            if transfers > slots {
                let kinds = plan.kinds(view).unwrap();
                let cut: Vec<Vec<usize>> = edges
                    .iter()
                    .enumerate()
                    .map(|(node, before)| {
                        if kinds[node] == StageKind::WeightTransfer {
                            Vec::new()
                        } else {
                            before.clone()
                        }
                    })
                    .collect();
                assert!(violation(&plan, view, slots, &cut).is_some(), "the cut graph passes");
            }
        }
    }
}

const MTP_VIEWS: [View; 3] = [
    View::Draft { rows: 1 },
    View::Draft { rows: 4 },
    View::CatchUp { rows: 9 },
];

fn every_view() -> Vec<View> {
    BASE_VIEWS.iter().chain(&MTP_VIEWS).copied().collect()
}

/// The lane each node of `view` runs on, with each unit's lane taken from its first node.
fn node_lanes(plan: &Plan, view: View, home: usize, stage_lanes: usize) -> Vec<LaneId> {
    let spec = plan.spec_homed(view, home, stage_lanes).unwrap().unwrap();
    let units = plan.units(view).unwrap();
    let mut lanes = vec![LaneId(u16::MAX); plan.kinds(view).unwrap().len()];
    for (unit, stage) in spec.stages().iter().enumerate() {
        assert_eq!(stage.lanes.len(), 1);
        for &node in &units[unit] {
            lanes[node] = stage.lanes[0];
        }
    }
    lanes
}

#[test]
fn an_empty_selection_changes_nothing() {
    let fixture = fixture();
    let before = fixture.unstaged();
    let empty = BTreeSet::new();
    let ring = StagingRing {
        base: DevicePtr::NULL.offset(1 << 32),
        slots: 3,
        slot_bytes: 1 << 20,
    };
    let after = Plan::build(
        &fixture.config,
        &fixture.lookup(&empty),
        &fixture.resident,
        &fixture.formats,
        Some(&ring),
        Some(TRANSFER_LANE),
    )
    .unwrap();
    // Objects that are read in place over the bus are not staged, whatever the ring.
    let mapped = fixture
        .resident
        .device
        .iter()
        .filter(|placement| placement.name == "text/token_embedding" || placement.name == "text/output_head")
        .map(|placement| placement.name.clone());
    let mapped = set(mapped);
    assert_eq!(mapped.len(), 2);
    let in_place = Plan::build(
        &fixture.config,
        &fixture.lookup(&mapped),
        &fixture.resident,
        &fixture.formats,
        Some(&ring),
        Some(TRANSFER_LANE),
    )
    .unwrap();
    let mut seen_mtp = false;
    for view in every_view() {
        for other in [&after, &in_place] {
            assert_eq!(before.kinds(view), other.kinds(view), "{view:?}");
            assert_eq!(before.edges(view), other.edges(view), "{view:?}");
            assert_eq!(before.units(view), other.units(view), "{view:?}");
            assert_eq!(before.unit_edges(view), other.unit_edges(view), "{view:?}");
            assert_eq!(before.capturable(view), other.capturable(view), "{view:?}");
            assert_eq!(
                format!("{:?}", before.spec(view)),
                format!("{:?}", other.spec(view)),
                "{view:?}"
            );
            assert!(other.slot_readers(view).is_empty() && other.first_transfers(view).is_empty());
        }
        assert_eq!(before.kinds(view), after.kinds(view), "{view:?}");
        assert_eq!(before.edges(view), after.edges(view), "{view:?}");
        assert_eq!(before.dependencies(view), after.dependencies(view), "{view:?}");
        assert_eq!(before.units(view), after.units(view), "{view:?}");
        assert_eq!(before.unit_edges(view), after.unit_edges(view), "{view:?}");
        assert_eq!(before.width(view), after.width(view), "{view:?}");
        assert_eq!(before.capturable(view), after.capturable(view), "{view:?}");
        assert_eq!(
            before.path_predecessors(view),
            after.path_predecessors(view),
            "{view:?}"
        );
        assert_eq!(before.scratch_regions(view), after.scratch_regions(view), "{view:?}");
        assert_eq!(
            format!("{:?}", before.spec_homed(view, 1, 4).map(|spec| spec.unwrap())),
            format!("{:?}", after.spec_homed(view, 1, 4).map(|spec| spec.unwrap())),
            "{view:?}"
        );
        assert!(after.slot_readers(view).is_empty() && after.first_transfers(view).is_empty());
        seen_mtp |= view.rows().is_some() && matches!(view, View::Draft { .. }) && after.stages(view).is_some();
    }
    assert!(seen_mtp, "the model drafts, so the MTP views are compared");
    assert_eq!(before.fan_out_width(), after.fan_out_width());
}

#[test]
fn an_unstaged_view_names_no_slot() {
    let fixture = fixture();
    let plan = fixture.build(&set(fixture.candidates(&["/mlp/down"])), 2).unwrap();
    for view in MTP_VIEWS {
        assert!(plan.stages(view).is_some());
        assert!(plan.slot_readers(view).is_empty(), "{view:?}");
        assert!(plan.first_transfers(view).is_empty(), "{view:?}");
    }
    let resident = fixture.unstaged();
    for view in every_view() {
        assert!(resident.slot_readers(view).is_empty() && resident.first_transfers(view).is_empty());
    }
    for view in BASE_VIEWS {
        assert_eq!(plan.slot_readers(view).len(), 2, "{view:?}: one entry for each slot");
        assert!(plan.slot_readers(view).iter().all(Option::is_some));
        assert_eq!(plan.first_transfers(view).len(), 2);
    }
}

#[test]
fn mtp_views_stage_nothing_and_stay_capturable() {
    let fixture = fixture();
    let plan = fixture.build(&set(fixture.candidates(&STAGEABLE)), 2).unwrap();
    let resident = fixture.unstaged();
    for view in MTP_VIEWS {
        let kinds = plan.kinds(view).expect("the model drafts");
        assert!(!kinds.contains(&StageKind::WeightTransfer), "{view:?}");
        assert_eq!(kinds, resident.kinds(view).unwrap(), "{view:?}");
        assert_eq!(plan.edges(view), resident.edges(view), "{view:?}");
        assert_eq!(plan.capturable(view), view.capturable(), "{view:?}");
    }
    assert!(plan.capturable(View::Draft { rows: 1 }));
    for view in BASE_VIEWS {
        assert!(
            !plan.capturable(view),
            "{view:?}: a staged view runs in the graph, not a capture"
        );
        assert!(resident.capturable(view) == view.capturable());
    }
}

#[test]
fn transfers_run_on_the_transfer_lane_only() {
    let fixture = fixture();
    let chosen = set(fixture.candidates(&STAGEABLE));
    let staged = fixture.build(&chosen, 3).unwrap();
    let resident = fixture.unstaged();
    let stage_lanes = Plan::lane_width() + 1;
    for view in BASE_VIEWS {
        let kinds = staged.kinds(view).unwrap();
        let plain = resident.kinds(view).unwrap();
        let compute: Vec<usize> = (0..kinds.len())
            .filter(|&n| kinds[n] != StageKind::WeightTransfer)
            .collect();
        assert_eq!(compute.len(), plain.len());
        assert!(
            staged.width(view).unwrap() == resident.width(view).unwrap(),
            "{view:?}: the width counts compute only"
        );
        for home in 0..stage_lanes {
            let lanes = node_lanes(&staged, view, home, stage_lanes);
            let reference = node_lanes(&resident, view, home, stage_lanes);
            for (node, kind) in kinds.iter().enumerate() {
                if *kind == StageKind::WeightTransfer {
                    assert_eq!(lanes[node], TRANSFER_LANE, "{view:?}: a transfer");
                } else {
                    assert_ne!(lanes[node], TRANSFER_LANE, "{view:?}: a compute stage");
                    assert!(usize::from(lanes[node].0) < stage_lanes);
                }
            }
            let staged_compute: Vec<LaneId> = compute.iter().map(|&node| lanes[node]).collect();
            assert_eq!(
                staged_compute, reference,
                "{view:?}, home {home}: compute stages keep their lanes"
            );
        }
    }
}

/// The shape of a staged view's units: a transfer alone, the units tiling the nodes from the compute root to the sink,
/// each refining a unit of the resident view, a consumer starting a unit and a slot's last reader ending one.
fn check_units(staged: &Plan, resident: &Plan, view: View) {
    let kinds = staged.kinds(view).unwrap();
    let units = staged.units(view).unwrap();
    let edges = staged.unit_edges(view).unwrap();
    for nodes in units {
        if nodes.iter().any(|&node| kinds[node] == StageKind::WeightTransfer) {
            assert_eq!(nodes.len(), 1, "{view:?}: a transfer shares its unit with no stage");
        }
    }
    // The units tile the nodes, the first holds the compute root and the last the sink.
    let mut flat: Vec<usize> = units.iter().flatten().copied().collect();
    flat.sort_unstable();
    assert_eq!(flat, (0..kinds.len()).collect::<Vec<_>>());
    assert_eq!(units[0][0], 0);
    assert_eq!(kinds[0], resident.kinds(view).unwrap()[0]);
    assert_eq!(units.last().unwrap().last(), Some(&(kinds.len() - 1)));
    let reach = ancestors(edges);
    let last = units.len() - 1;
    for (unit, nodes) in units.iter().enumerate() {
        if kinds[nodes[0]] == StageKind::WeightTransfer {
            assert!(reach[last][unit], "{view:?}: the sink follows every transfer");
        }
    }
    // Each staged unit of compute stages lies inside one unit of the resident view: fusing only got finer.
    let plain_units = resident.units(view).unwrap();
    let mut plain_unit_of = vec![0; kinds.len()];
    for (unit, nodes) in plain_units.iter().enumerate() {
        for &node in nodes {
            plain_unit_of[node] = unit;
        }
    }
    // The rank of a compute stage among the compute stages is its index in the resident view.
    let mut rank = vec![usize::MAX; kinds.len()];
    let mut next = 0;
    for (node, kind) in kinds.iter().enumerate() {
        if *kind != StageKind::WeightTransfer {
            rank[node] = next;
            next += 1;
        }
    }
    for nodes in units {
        if kinds[nodes[0]] == StageKind::WeightTransfer {
            continue;
        }
        let plain: BTreeSet<usize> = nodes.iter().map(|&node| plain_unit_of[rank[node]]).collect();
        assert_eq!(
            plain.len(),
            1,
            "{view:?}: a staged unit crosses a unit of the resident view"
        );
    }
    // A consumer starts a unit.
    let dependencies = staged.dependencies(view).unwrap();
    for (node, kind) in kinds.iter().enumerate() {
        if *kind != StageKind::WeightTransfer
            && dependencies[node]
                .iter()
                .any(|&d| kinds[d] == StageKind::WeightTransfer)
        {
            assert!(
                units.iter().any(|nodes| nodes[0] == node),
                "{view:?}: the consumer {node} starts a unit"
            );
        }
    }
    let kinds = staged.kinds(view).unwrap();
    let units = staged.units(view).unwrap();
    let dependencies = staged.dependencies(view).unwrap();
    let consumers: Vec<usize> = (0..kinds.len())
        .filter(|&node| {
            kinds[node] != StageKind::WeightTransfer
                && dependencies[node]
                    .iter()
                    .any(|&d| kinds[d] == StageKind::WeightTransfer)
        })
        .collect();
    for (slot, reader) in staged.slot_readers(view).iter().enumerate() {
        let Some(reader) = reader else { continue };
        let slots = staged.slot_readers(view).len();
        let last = consumers
            .iter()
            .enumerate()
            .rfind(|(index, _)| index % slots == slot)
            .unwrap()
            .1;
        let unit = &units[usize::from(reader.0)];
        assert_eq!(
            unit.last(),
            Some(last),
            "{view:?}: slot {slot}'s last reader ends its unit"
        );
    }
}

#[test]
fn a_transfer_is_a_unit_of_its_own() {
    let fixture = fixture();
    let chosen = set(fixture.candidates(&STAGEABLE));
    let staged = fixture.build(&chosen, 2).unwrap();
    let resident = fixture.unstaged();
    for view in BASE_VIEWS {
        check_units(&staged, &resident, view);
    }
}

#[test]
fn two_projections_that_run_side_by_side_share_one_slot() {
    let fixture = fixture();
    let resident = fixture.unstaged();
    let names = fixture.candidates(&["/gdn/query_key", "/gdn/value_z"]);
    for layer in ["text/layers/0/", "text/layers/1/"] {
        let chosen = set(names.iter().filter(|name| name.starts_with(layer)).cloned());
        assert_eq!(chosen.len(), 2);
        let plan = fixture.build(&chosen, 1).unwrap();
        for view in BASE_VIEWS {
            check_units(&plan, &resident, view);
            assert_eq!(
                violation(&plan, view, 1, plan.edges(view).unwrap()),
                None,
                "{layer} {view:?}"
            );
        }
    }
}

#[test]
fn more_uses_than_slots_and_a_weight_larger_than_a_slot_are_refused() {
    let fixture = fixture();
    let chosen = set(fixture.candidates(&["/mlp/down"]));
    assert!(matches!(fixture.build(&chosen, 0), Err(PlanBuildError::NoSlots { .. })));
    let mut ring = fixture.ring(&chosen, 2);
    ring.slot_bytes -= 1;
    let held = fixture.lookup(&chosen);
    let error = Plan::build(
        &fixture.config,
        &held,
        &fixture.resident,
        &fixture.formats,
        Some(&ring),
        Some(TRANSFER_LANE),
    )
    .unwrap_err();
    assert!(matches!(error, PlanBuildError::WeightTooLarge { .. }), "{error}");
    let ring = fixture.ring(&chosen, 2);
    let error = Plan::build(
        &fixture.config,
        &held,
        &fixture.resident,
        &fixture.formats,
        Some(&ring),
        None,
    )
    .unwrap_err();
    assert!(matches!(error, PlanBuildError::NoTransferLane), "{error}");
}

/// The uses of a view in order: each transfer, its consumer, and the units they are in.
fn uses(plan: &Plan, view: View) -> Vec<(usize, usize)> {
    let kinds = plan.kinds(view).unwrap();
    let dependencies = plan.dependencies(view).unwrap();
    (0..kinds.len())
        .filter(|&node| kinds[node] == StageKind::WeightTransfer)
        .map(|transfer| {
            let consumer = (0..kinds.len())
                .find(|&node| dependencies[node].contains(&transfer))
                .unwrap();
            (transfer, consumer)
        })
        .collect()
}

fn unit_of(plan: &Plan, view: View, node: usize) -> usize {
    plan.units(view)
        .unwrap()
        .iter()
        .position(|unit| unit.contains(&node))
        .unwrap()
}

#[test]
fn the_plan_names_each_slots_last_reader_and_first_transfer() {
    let fixture = fixture();
    let all = set(fixture.candidates(&STAGEABLE));
    let one = set(fixture.candidates(&["/mlp/down"]).into_iter().take(2));
    for (chosen, slots) in [(all.clone(), 3), (all, 1), (one.clone(), 2), (one, 5)] {
        let plan = fixture.build(&chosen, slots).unwrap();
        for view in BASE_VIEWS {
            let uses = uses(&plan, view);
            let readers = plan.slot_readers(view);
            assert_eq!(readers.len(), slots, "{view:?}: one entry for each slot");
            for (slot, reader) in readers.iter().enumerate() {
                let last = uses.iter().enumerate().rfind(|(index, _)| index % slots == slot);
                let expected = last.map(|(_, &(_, consumer))| unit_of(&plan, view, consumer));
                assert_eq!(reader.map(|id| usize::from(id.0)), expected, "{view:?}: slot {slot}");
            }
            let first = plan.first_transfers(view);
            assert_eq!(first.len(), slots.min(uses.len()), "{view:?}");
            for &(slot, id) in first {
                let (transfer, _) = uses[slot];
                let unit = usize::from(id.0);
                assert_eq!(plan.units(view).unwrap()[unit], vec![transfer], "{view:?}: slot {slot}");
                assert!(
                    plan.unit_edges(view).unwrap()[unit].is_empty(),
                    "{view:?}: a first transfer is a root"
                );
            }
        }
    }
}

#[test]
fn held_weights_with_no_ring_are_refused_but_objects_read_in_place_are_not() {
    let fixture = fixture();
    let chosen = set(fixture.candidates(&["/mlp/down"]));
    let held = fixture.lookup(&chosen);
    let error = Plan::build(&fixture.config, &held, &fixture.resident, &fixture.formats, None, None).unwrap_err();
    assert!(matches!(error, PlanBuildError::NoStagingRing { .. }), "{error}");
    assert!(error.to_string().starts_with("text/layers/"), "{error}");
    let mapped = set(["text/token_embedding".to_string(), "text/output_head".to_string()]);
    let held = fixture.lookup(&mapped);
    Plan::build(&fixture.config, &held, &fixture.resident, &fixture.formats, None, None).unwrap();
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn the_weights_hand_the_plan_their_host_arena_through_the_lookup_trait() {
    let Some(library) = test_support::require_property("euhedral.cuda.library") else {
        return;
    };
    let library = CudaLibrary::load(Path::new(&library)).unwrap();
    let device = DeviceBuffer::alloc(&library, 4096).unwrap();
    let arena = Arc::new(HostWeightsArena::alloc(&library, 1 << 20).unwrap());
    let held = Weights::new(device, Some(arena.clone()), HashMap::new());
    let lookup: &dyn WeightLookup = &held;
    let seen = lookup.host_arena().expect("the arena of the weights");
    assert!(Arc::ptr_eq(&seen, &arena));
    let device = DeviceBuffer::alloc(&library, 4096).unwrap();
    let none = Weights::new(device, None, HashMap::new());
    let lookup: &dyn WeightLookup = &none;
    assert!(lookup.host_arena().is_none());
}
