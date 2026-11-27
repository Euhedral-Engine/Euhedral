use std::time::Duration;

use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;
use std::sync::atomic::Ordering;

use euhedral_graph::testing::{FakeDevice, StageHarness};
use euhedral_graph::{LaneId, Marker};

use super::*;
use crate::flash::moe::testing::{Host, Lease, Log, Source, layer, planned};

fn marker(value: u64) -> Marker {
    let (_, devices) = FakeDevice::pool(&["lane"]);
    let mut marker = devices[0].submit(Duration::ZERO, &[]);
    marker.value = value;
    marker
}

#[test]
fn a_lease_is_fenced_behind_the_marker_of_the_work_that_read_it_and_then_closed() {
    let (plan, log) = (planned(3, &[6, 1, 6, 3, 1, 6]), Log::default());
    let lease = Lease::boxed(0xa000, None, &log);
    let done = settle(&plan, 0, lease, |_| Ok(Submitted::marker(marker(7))));
    assert!(done.is_ok());
    assert_eq!(log.lines(), ["fence 0xa000 behind 7", "close 0xa000"]);
    assert!(!plan.is_held(0), "a fenced lease is closed, not kept");
}

#[test]
fn a_lease_of_a_submission_that_asked_to_retry_stays_in_its_slot_unfenced() {
    let (plan, log) = (planned(3, &[6, 1, 6, 3, 1, 6]), Log::default());
    let lease = Lease::boxed(0xa000, None, &log);
    let done = settle(&plan, 0, lease, |_| Err(StageError::Retry));
    assert!(matches!(done, Err(StageError::Retry)));
    assert!(log.lines().is_empty());
    assert_eq!(plan.take(0).map(|lease| lease.device_address()), Some(0xa000));
}

#[test]
fn a_lease_of_a_failed_submission_stays_in_its_slot_until_the_quantum_retires() {
    let (plan, log) = (planned(3, &[6, 1, 6, 3, 1, 6]), Log::default());
    let lease = Lease::boxed(0xa000, None, &log);
    let done = settle(&plan, 0, lease, |_| Err(fatal("the launch failed")));
    assert!(matches!(done, Err(StageError::Fatal(_))));
    assert!(log.lines().is_empty(), "work that reads the slot may still be queued");
    assert!(plan.is_held(0));
    drop(plan);
    assert_eq!(log.lines(), ["close 0xa000"]);
}

#[test]
fn the_submission_sees_the_lease_it_must_wait_for() {
    let (plan, log) = (planned(3, &[6, 1, 6, 3, 1, 6]), Log::default());
    let lease = Lease::boxed(0xa000, Some(marker(3)), &log);
    let mut seen = None;
    let _ = settle(&plan, 0, lease, |lease| {
        seen = Some((lease.device_address(), lease.ready().map(|ready| ready.value)));
        Ok(Submitted::done())
    });
    assert_eq!(seen, Some((0xa000, Some(3))));
}

/// The expert stage of slot `index`, and a harness to run it in.
fn rig(index: u32) -> (MoeExpert, StageHarness) {
    let layer = Arc::new(layer(Arc::new(Source::default()), true));
    let (lanes, _) = FakeDevice::pool(&["lane"]);
    (MoeExpert::new(layer, index), StageHarness::new(lanes))
}

/// Experts 1, 3 and 6 are named, by three rows.
fn block() -> Arc<BlockPlan> {
    planned(3, &[6, 1, 6, 3, 1, 6])
}

#[test]
fn a_slot_past_the_blocks_experts_does_no_work_and_closes_a_lease_it_was_handed() {
    let (stage, harness) = rig(3);
    let (host, log) = (Host::new(block()), Log::default());
    host.plan.as_ref().unwrap().hold(3, Lease::boxed(0xa000, None, &log));
    let run = harness.run(&stage as &dyn Stage<Host>, &host);
    assert!(run.result.is_ok() && run.contract_breach().is_none());
    assert!(
        host.log.lines().is_empty(),
        "no work is queued for a slot the block does not use"
    );
    assert_eq!(log.lines(), ["close 0xa000"]);
}

#[test]
fn a_stopped_quantum_does_no_work_and_closes_the_lease() {
    let (stage, harness) = rig(0);
    let (host, log) = (Host::new(block()), Log::default());
    host.stopped.store(true, Ordering::Relaxed);
    host.plan.as_ref().unwrap().hold(0, Lease::boxed(0xa000, None, &log));
    assert!(harness.run(&stage as &dyn Stage<Host>, &host).result.is_ok());
    assert!(host.log.lines().is_empty());
    assert_eq!(log.lines(), ["close 0xa000"]);
}

#[test]
fn a_named_expert_that_is_not_held_fails_the_stage() {
    let (stage, harness) = rig(1);
    let host = Host::new(block());
    let run = harness.run(&stage as &dyn Stage<Host>, &host);
    assert!(matches!(run.result, Err(StageError::Fatal(why)) if why.contains("slot 1") && why.contains("not held")));
    assert!(host.log.lines().is_empty());
}

#[test]
fn a_held_expert_writes_its_slot_words_and_queues_its_work_after_the_lease_is_ready() {
    let (stage, harness) = rig(2);
    let (host, log) = (Host::new(block()), Log::default());
    host.plan
        .as_ref()
        .unwrap()
        .hold(2, Lease::boxed(0x7_0000_a000, Some(marker(3)), &log));
    let run = harness.run(&stage as &dyn Stage<Host>, &host);
    assert!(run.result.is_ok());
    // Slot 2 is words 4 and 5: the address, low then high.
    assert_eq!((host.word(4), host.word(5)), (0xa000, 7));
    assert_eq!(
        host.log.lines(),
        ["queue Expert { index: 2, first_item: 2, item_count: 1, pairs: 6 } after [3]"]
    );
    assert_eq!(
        log.lines(),
        ["close 0x70000a000"],
        "an unmarked submission leaves nothing to fence"
    );
}

#[test]
fn a_lease_is_fenced_behind_the_marker_the_expert_work_was_queued_with() {
    let (stage, harness) = rig(0);
    let (host, log) = (Host::new(block()), Log::default());
    host.plan.as_ref().unwrap().hold(0, Lease::boxed(0xa000, None, &log));
    let host = host.answering(Ok(Submitted::marker(marker(9))));
    assert!(harness.run(&stage as &dyn Stage<Host>, &host).result.is_ok());
    assert_eq!(log.lines(), ["fence 0xa000 behind 9", "close 0xa000"]);
}

#[test]
fn a_quantum_abandoned_after_a_partly_queued_expert_keeps_the_lease_until_it_retires() {
    let (stage, harness) = rig(0);
    let (host, log) = (Host::new(block()), Log::default());
    let plan = host.plan.clone().unwrap();
    plan.hold(0, Lease::boxed(0xa000, None, &log));
    let host = host.answering(Err(fatal("the second kernel did not launch")));
    assert!(matches!(
        harness.run(&stage as &dyn Stage<Host>, &host).result,
        Err(StageError::Fatal(_))
    ));
    assert!(host.log.lines().iter().all(|line| line != "await unmarked"));
    harness.abandoned(&stage as &dyn Stage<Host>, &host, LaneId(0));
    assert_eq!(
        host.log.lines().last().map(String::as_str),
        Some("await unmarked"),
        "home retires behind the work that failed on another lane"
    );
    assert!(log.lines().is_empty(), "queued work may still read the slot");
    assert!(plan.is_held(0));
    <MoeExpert as Stage<Host>>::retired(&stage, &host, false);
    assert_eq!(log.lines(), ["close 0xa000"]);
    assert!(!plan.is_held(0));
}

#[test]
fn a_retried_expert_keeps_its_lease_for_the_next_run_and_gives_it_up_when_abandoned() {
    let (stage, harness) = rig(0);
    let (host, log) = (Host::new(block()), Log::default());
    let plan = host.plan.clone().unwrap();
    plan.hold(0, Lease::boxed(0xa000, None, &log));
    let host = host.answering(Err(StageError::Retry));
    assert!(matches!(
        harness.run(&stage as &dyn Stage<Host>, &host).result,
        Err(StageError::Retry)
    ));
    assert!(plan.is_held(0) && log.lines().is_empty());
    assert!(
        harness.run(&stage as &dyn Stage<Host>, &host).result.is_ok(),
        "the retry finds its lease"
    );
    plan.hold(0, Lease::boxed(0xb000, None, &log));
    harness.abandoned(&stage as &dyn Stage<Host>, &host, LaneId(0));
    assert_eq!(log.lines().last().map(String::as_str), Some("close 0xb000"));
    assert!(!plan.is_held(0));
}

#[test]
fn a_stage_that_never_ran_releases_what_it_was_given_when_abandoned_and_when_retired() {
    let (stage, harness) = rig(0);
    let (host, log) = (Host::new(block()), Log::default());
    let plan = host.plan.clone().unwrap();
    plan.hold(0, Lease::boxed(0xa000, None, &log));
    harness.abandoned(&stage as &dyn Stage<Host>, &host, LaneId(0));
    assert_eq!(log.lines(), ["close 0xa000"]);
    plan.hold(0, Lease::boxed(0xb000, None, &log));
    <MoeExpert as Stage<Host>>::retired(&stage, &host, true);
    assert_eq!(log.lines(), ["close 0xa000", "close 0xb000"]);
}

#[test]
fn a_quantum_with_no_block_fails_the_stage() {
    let (stage, harness) = rig(0);
    let mut host = Host::new(block());
    host.plan = None;
    assert!(matches!(
        harness.run(&stage as &dyn Stage<Host>, &host).result,
        Err(StageError::Fatal(why)) if why.contains("no block")
    ));
}

#[test]
fn a_submission_that_unwinds_leaves_its_lease_kept_for_retirement() {
    let (plan, log) = (block(), Log::default());
    let lease = Lease::boxed(0xa000, None, &log);
    let unwound = catch_unwind(AssertUnwindSafe(|| {
        let _ = settle(&plan, 0, lease, |_| panic!("the launch panicked"));
    }));
    assert!(unwound.is_err());
    assert!(log.lines().is_empty(), "the lease is not dropped unfenced");
    assert!(plan.is_held(0));
    plan.release_queued(0);
    assert_eq!(log.lines(), ["close 0xa000"]);
}

#[test]
fn a_quantum_whose_staging_does_not_fit_the_layer_fails_the_expert_stage_before_any_write() {
    let (stage, harness) = rig(0);
    let (mut host, log) = (Host::new(block()), Log::default());
    host.fits = false;
    host.plan.as_ref().unwrap().hold(0, Lease::boxed(0xa000, None, &log));
    assert!(matches!(
        harness.run(&stage as &dyn Stage<Host>, &host).result,
        Err(StageError::Fatal(why)) if why.contains("staging")
    ));
    assert_eq!(host.word(0), 0);
    assert!(host.log.lines().is_empty());
}
