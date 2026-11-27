use crate::flash::moe::testing::{Lease, Log, planned};

#[test]
fn a_plan_names_the_active_experts_in_ascending_order_with_their_items() {
    // Rows choose (6, 1), (6, 3) and (1, 6): experts 1, 3 and 6 with 2, 1 and 3 pairs, one item each.
    let plan = planned(3, &[6, 1, 6, 3, 1, 6]);
    assert_eq!((plan.rows(), plan.pairs(), plan.active()), (3, 6, 3));
    assert_eq!([0, 1, 2, 3].map(|i| plan.expert(i)), [Some(1), Some(3), Some(6), None]);
    assert_eq!(
        [0, 1, 2, 3].map(|i| plan.items(i)),
        [Some((0, 1)), Some((1, 1)), Some((2, 1)), None]
    );
}

#[test]
fn padding_pairs_name_no_expert() {
    let plan = planned(2, &[8, 8, 2, 8]);
    assert_eq!(
        (plan.pairs(), plan.active(), plan.expert(0), plan.expert(1)),
        (1, 1, Some(2), None)
    );
}

#[test]
fn a_block_of_no_rows_names_no_expert() {
    let plan = planned(3, &[6, 1, 6, 3, 1, 6]);
    plan.publish_none();
    assert_eq!(
        (plan.rows(), plan.pairs(), plan.active(), plan.expert(0)),
        (0, 0, 0, None)
    );
}

#[test]
fn a_lease_is_taken_once_from_the_slot_of_its_index() {
    let (plan, log) = (planned(3, &[6, 1, 6, 3, 1, 6]), Log::default());
    plan.hold(2, Lease::boxed(0xa000, None, &log));
    assert!([plan.is_held(0), plan.is_held(1), plan.is_held(2)] == [false, false, true]);
    assert_eq!(plan.take(2).map(|lease| lease.device_address()), Some(0xa000));
    assert!(plan.take(2).is_none());
    assert_eq!(log.lines(), ["close 0xa000"]);
}

#[test]
fn a_lease_held_for_a_slot_the_plan_does_not_have_is_closed() {
    let (plan, log) = (planned(3, &[6, 1, 6, 3, 1, 6]), Log::default());
    plan.hold(plan.width(), Lease::boxed(0xb000, None, &log));
    assert_eq!(log.lines(), ["close 0xb000"]);
}

#[test]
fn leases_left_by_a_block_are_closed_when_the_next_is_published_and_when_the_plan_drops() {
    let (plan, log) = (planned(3, &[6, 1, 6, 3, 1, 6]), Log::default());
    plan.hold(0, Lease::boxed(0xa000, None, &log));
    plan.hold(1, Lease::boxed(0xb000, None, &log));
    plan.publish_none();
    assert_eq!(log.lines(), ["close 0xa000", "close 0xb000"]);
    plan.hold(1, Lease::boxed(0xc000, None, &log));
    drop(plan);
    assert_eq!(log.lines().last().map(String::as_str), Some("close 0xc000"));
}
