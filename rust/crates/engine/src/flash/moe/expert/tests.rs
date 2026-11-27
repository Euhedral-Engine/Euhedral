use std::time::Duration;

use euhedral_graph::testing::FakeDevice;

use super::*;
use crate::flash::moe::testing::{Lease, Log, planned};

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
