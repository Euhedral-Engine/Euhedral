use std::sync::atomic::Ordering;

use euhedral_graph::testing::{FakeDevice, GateEvent, StageHarness};

use super::*;
use crate::flash::moe::testing::{Lease, Log, Source, layer, planned};

/// A quantum as the fetch stages see it.
struct Host {
    plan: Option<Arc<BlockPlan>>,
    stopped: AtomicBool,
    cancel: Option<Arc<AtomicBool>>,
}

impl MoeHost for Host {
    fn block_plan(&self) -> Result<&Arc<BlockPlan>, StageError> {
        self.plan.as_ref().ok_or_else(|| fatal("no block"))
    }

    fn stopped(&self) -> bool {
        self.stopped.load(Ordering::Relaxed)
    }

    fn cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        self.cancel.clone()
    }
}

fn host(plan: Arc<BlockPlan>) -> Host {
    Host {
        plan: Some(plan),
        stopped: AtomicBool::new(false),
        cancel: None,
    }
}

/// Experts 1, 3 and 6 are named, by three rows.
fn block() -> Arc<BlockPlan> {
    planned(3, &[6, 1, 6, 3, 1, 6])
}

struct Rig {
    source: Arc<Source>,
    harness: StageHarness,
    stages: Vec<MoeFetch>,
}

fn rig(source: Source) -> Rig {
    let source = Arc::new(source);
    let layer = Arc::new(layer(source.clone(), true));
    let (lanes, _) = FakeDevice::pool(&["lane"]);
    Rig {
        stages: (0..8).map(|index| MoeFetch::new(layer.clone(), index)).collect(),
        harness: StageHarness::new(lanes),
        source,
    }
}

fn pass(rig: &Rig, index: u32, host: &Host) -> (bool, bool) {
    let passed = rig
        .harness
        .pass_gate(&rig.stages[index as usize] as &dyn Stage<Host>, host);
    assert!(passed.contract_breach().is_none());
    (passed.open, passed.held)
}

#[test]
fn a_fetch_past_the_experts_of_the_block_completes_without_a_request() {
    let rig = rig(Source::default());
    let host = host(block());
    for index in 3..8 {
        assert_eq!(pass(&rig, index, &host), (true, false));
    }
    assert_eq!(rig.source.pending(), 0);
    assert!(rig.harness.gate_events().is_empty());
}

#[test]
fn a_fetch_of_a_stopped_quantum_asks_for_nothing() {
    let rig = rig(Source::default());
    let host = host(block());
    host.stopped.store(true, Ordering::Relaxed);
    assert_eq!(pass(&rig, 0, &host), (true, false));
    assert_eq!(rig.source.pending(), 0);
}

#[test]
fn a_fetch_asks_for_its_expert_of_the_layers_bank_and_holds_its_stage_until_it_arrives() {
    let rig = rig(Source::default());
    let host = host(block());
    assert_eq!(pass(&rig, 1, &host), (false, true));
    let request = rig.source.take(3).expect("the second expert of the block is expert 3");
    assert_eq!(request.expert.bank, 5);
    assert!(request.scan, "a block of several rows scans");
    assert!(rig.harness.gate_events().is_empty(), "the stage waits for the arrival");
    let log = Log::default();
    request.answer(Arrived::Held(Lease::boxed(0xa000, None, &log)));
    assert_eq!(rig.harness.gate_events(), [GateEvent::Released]);
    assert_eq!(
        host.plan.as_ref().unwrap().take(1).map(|lease| lease.device_address()),
        Some(0xa000)
    );
    let ran = rig.harness.run(&rig.stages[1] as &dyn Stage<Host>, &host);
    assert!(ran.result.unwrap().marker.is_none());
}

#[test]
fn a_single_row_does_not_scan() {
    let rig = rig(Source::default());
    let host = host(planned(1, &[4, 5]));
    assert_eq!(pass(&rig, 0, &host), (false, true));
    assert!(!rig.source.take(4).unwrap().scan);
}

#[test]
fn experts_that_arrive_in_any_order_each_land_in_the_slot_of_their_index() {
    let orders = [[0, 1, 2], [0, 2, 1], [1, 0, 2], [1, 2, 0], [2, 0, 1], [2, 1, 0]];
    for order in orders {
        let rig = rig(Source::default());
        let host = host(block());
        for index in 0..3 {
            assert_eq!(pass(&rig, index, &host), (false, true));
        }
        let log = Log::default();
        for index in order {
            let expert = [1, 3, 6][index as usize];
            let lease = Lease::boxed(0x1000 * (u64::from(expert) + 1), None, &log);
            rig.source.take(expert).unwrap().answer(Arrived::Held(lease));
        }
        assert_eq!(
            rig.harness.gate_events(),
            [const { GateEvent::Released }; 3],
            "{order:?}"
        );
        let plan = host.plan.as_ref().unwrap();
        let held: Vec<_> = (0..3).map(|index| plan.take(index).unwrap().device_address()).collect();
        assert_eq!(held, [0x2000, 0x4000, 0x7000], "{order:?}");
    }
}

#[test]
fn a_failed_arrival_fails_the_step_naming_the_layer_and_the_expert() {
    let rig = rig(Source::default());
    let host = host(block());
    assert_eq!(pass(&rig, 2, &host), (false, true));
    rig.source
        .take(6)
        .unwrap()
        .answer(Arrived::Failed("the record is gone".to_owned()));
    assert!(matches!(
        rig.harness.gate_events().as_slice(),
        [GateEvent::Failed(StageError::Fatal(why))] if why == "layer 3 expert 6: the record is gone"
    ));
    assert!(!host.plan.as_ref().unwrap().is_held(2));
}

#[test]
fn a_refused_request_fails_the_step_naming_the_layer_and_the_expert() {
    let rig = rig(Source::refusing("no such bank"));
    let host = host(block());
    assert_eq!(pass(&rig, 0, &host), (false, true));
    assert!(matches!(
        rig.harness.gate_events().as_slice(),
        [GateEvent::Failed(StageError::Fatal(why))] if why == "layer 3 expert 1: no such bank"
    ));
}

#[test]
fn a_cancelled_arrival_releases_the_stage_without_a_lease() {
    let rig = rig(Source::default());
    let host = host(block());
    assert_eq!(pass(&rig, 0, &host), (false, true));
    rig.source.take(1).unwrap().answer(Arrived::Cancelled);
    assert_eq!(rig.harness.gate_events(), [GateEvent::Released]);
    assert!(!host.plan.as_ref().unwrap().is_held(0));
}

#[test]
fn a_request_carries_the_flag_of_the_quantums_segment_when_it_has_one_and_the_blocks_otherwise() {
    let rig = rig(Source::default());
    let mut host = host(block());
    let own = Arc::new(AtomicBool::new(false));
    host.cancel = Some(Arc::clone(&own));
    pass(&rig, 0, &host);
    assert!(Arc::ptr_eq(&rig.source.take(1).unwrap().cancel, &own));
    host.cancel = None;
    pass(&rig, 1, &host);
    assert!(Arc::ptr_eq(
        &rig.source.take(3).unwrap().cancel,
        host.plan.as_ref().unwrap().spare_cancel()
    ));
}

#[test]
fn a_quantum_with_no_block_fails_the_stage_at_its_gate() {
    let rig = rig(Source::default());
    let host = Host {
        plan: None,
        stopped: AtomicBool::new(false),
        cancel: None,
    };
    assert_eq!(pass(&rig, 0, &host), (false, true));
    assert!(matches!(
        rig.harness.gate_events().as_slice(),
        [GateEvent::Failed(StageError::Fatal(why))] if why.contains("no block")
    ));
}
