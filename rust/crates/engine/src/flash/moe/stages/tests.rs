use euhedral_graph::testing::{FakeDevice, StageHarness};

use super::*;
use crate::flash::moe::block::BlockPlan;
use crate::flash::moe::routing::Routing;
use crate::flash::moe::testing::{Host, Source, layer, shape};

const WEIGHT: u16 = 0x3f80;

/// The plan stage of the fixture layer, and a harness to run it in.
fn rig() -> (MoePlan, StageHarness) {
    let layer = Arc::new(layer(Arc::new(Source::default()), true));
    let (lanes, _) = FakeDevice::pool(&["lane"]);
    (MoePlan::new(layer), StageHarness::new(lanes))
}

/// A host whose router chose `ids` (two per row) for `rows` rows, over a plan that names nothing yet.
fn host(ids: &[i32], rows: usize) -> Host {
    let plan = Arc::new(BlockPlan::new(shape().layout().max_active()));
    let picks: Vec<(i32, u16)> = ids.iter().map(|id| (*id, WEIGHT)).collect();
    Host::new(plan).picking(&picks, rows)
}

fn run(stage: &MoePlan, harness: &StageHarness, host: &Host) -> Result<Submitted, StageError> {
    let run = harness.run(stage as &dyn Stage<Host>, host);
    assert!(run.contract_breach().is_none());
    run.result
}

#[test]
fn the_plan_publishes_the_blocks_experts_fills_the_description_and_queues_its_copy() {
    let (stage, harness) = rig();
    let host = host(&[6, 1, 6, 3, 1, 6], 3);
    assert!(run(&stage, &harness, &host).is_ok());
    let plan = host.plan.as_ref().unwrap();
    assert_eq!((plan.rows(), plan.pairs(), plan.active()), (3, 6, 3));
    assert_eq!([0, 1, 2].map(|i| plan.expert(i)), [Some(1), Some(3), Some(6)]);
    assert_eq!(host.log.lines(), ["queue Description after []"]);
    let layout = *shape().layout();
    // The first item names the first expert's index, and the pair table starts with the first pair's row and weight.
    let item = layout.items_offset() / 4;
    assert_eq!((host.word(item), host.word(item + 2)), (0, 2));
    let pair = layout.pairs_offset() / 4;
    assert_eq!((host.word(pair), host.word(pair + 1)), (0, u32::from(WEIGHT)));
    assert!(host.routing_is_home(), "the grouping scratch is given back");
}

#[test]
fn a_block_of_no_rows_publishes_no_expert_and_queues_nothing() {
    let (stage, harness) = rig();
    let host = host(&[], 0);
    let plan = host.plan.as_ref().unwrap();
    plan.publish(&{
        let mut routing = Routing::new(*shape().layout());
        routing.plan(3, |i| [6, 1, 6, 3, 1, 6][i], |_| WEIGHT).unwrap();
        routing
    })
    .unwrap();
    assert!(run(&stage, &harness, &host).is_ok());
    assert_eq!((plan.rows(), plan.pairs(), plan.active()), (0, 0, 0));
    assert!(host.log.lines().is_empty());
}

#[test]
fn a_router_pick_outside_the_experts_fails_the_stage_and_gives_the_scratch_back() {
    let (stage, harness) = rig();
    let host = host(&[6, 9, 6, 3, 1, 6], 3);
    let error = run(&stage, &harness, &host).unwrap_err();
    assert!(matches!(&error, StageError::Fatal(why) if why.contains("layer 3") && why.contains("expert id 9")));
    assert!(host.routing_is_home());
    assert_eq!(host.plan.as_ref().unwrap().active(), 0);
    assert!(host.log.lines().is_empty());
    // The scratch is usable again: the same stage plans a good block.
    let again = self::host(&[6, 1, 6, 3, 1, 6], 3);
    assert!(run(&stage, &harness, &again).is_ok());
}

#[test]
fn more_rows_than_the_layer_routes_fail_the_stage_before_the_scratch_is_taken() {
    let (stage, harness) = rig();
    let host = host(&[0; 10], 5);
    assert!(matches!(
        run(&stage, &harness, &host),
        Err(StageError::Fatal(why)) if why.contains("5 rows") && why.contains("layer 3")
    ));
    assert!(host.routing_is_home());
}

#[test]
fn a_retried_copy_plans_the_block_again_and_publishes_the_same_experts() {
    let (stage, harness) = rig();
    let host = host(&[6, 1, 6, 3, 1, 6], 3).answering(Err(StageError::Retry));
    assert!(matches!(run(&stage, &harness, &host), Err(StageError::Retry)));
    let plan = host.plan.as_ref().unwrap();
    let first = [0, 1, 2].map(|i| (plan.expert(i), plan.items(i)));
    assert!(host.routing_is_home());
    assert!(run(&stage, &harness, &host).is_ok());
    assert_eq!([0, 1, 2].map(|i| (plan.expert(i), plan.items(i))), first);
    assert_eq!(
        host.log.lines(),
        ["queue Description after []", "queue Description after []"]
    );
}

#[test]
fn a_plan_stage_that_finds_the_scratch_in_use_fails_the_quantum() {
    let (stage, harness) = rig();
    let host = host(&[6, 1, 6, 3, 1, 6], 3);
    let held = host.take_routing().unwrap();
    assert!(matches!(run(&stage, &harness, &host), Err(StageError::Fatal(_))));
    host.give_back(held);
}

#[test]
fn a_quantum_with_no_block_fails_the_plan_stage() {
    let (stage, harness) = rig();
    let mut host = host(&[], 0);
    host.plan = None;
    assert!(matches!(run(&stage, &harness, &host), Err(StageError::Fatal(_))));
}

#[test]
fn a_block_of_rows_within_what_the_layer_routes_is_accepted_and_one_past_it_is_refused() {
    let layer = layer(Arc::new(Source::default()), true);
    assert_eq!(check_rows(layer.max_rows(), &layer).unwrap(), 4);
    assert_eq!(check_rows(0, &layer).unwrap(), 0);
    let StageError::Fatal(why) = check_rows(5, &layer).unwrap_err() else {
        panic!("a refusal is fatal");
    };
    assert!(
        why.contains("5 rows") && why.contains("4 rows") && why.contains("layer 3"),
        "{why}"
    );
}
