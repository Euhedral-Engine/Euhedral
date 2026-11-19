//! The residency plan's fit, spill and refusal are the memory planner's decision: what the plan asks of each
//! location, where the host budget and the device budget draw the line, and that an unlimited budget is no overflow.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use euhedral_inference_engine::error::PlanError;
use euhedral_inference_engine::plan::Plan;
use euhedral_inference_engine::residency::planner::{PlanInputs, plan_with, sequences_for};
use euhedral_inference_engine::runtime::{resolve_sequences, stage_ring_for};
use euhedral_inference_settings::{Auto, ByteSize};

mod common;

use common::{config, rco_table, settings};

const GIB: u64 = 1 << 30;

fn inputs(device_free: u64, host_pinned_bytes: Option<u64>) -> PlanInputs {
    PlanInputs {
        device_free,
        multiprocessors: Some(70),
        host_pinned_bytes,
        sequences: 1,
    }
}

#[test]
fn the_location_bytes_are_the_plans_device_bytes_and_host_pages() {
    let (table, config) = (rco_table(), config());
    let mut settings = settings(131_072);
    settings.context.kv_cache_on_gpu_tokens = 32_768;
    let whole = plan_with(&config, &table, &settings, inputs(64 * GIB, Some(u64::MAX))).unwrap();
    assert_eq!(whole.location_bytes().device, whole.device_bytes);
    assert_eq!(whole.location_bytes().host, 0);
    let spilled = plan_with(
        &config,
        &table,
        &settings,
        inputs(whole.device_bytes - 1, Some(u64::MAX)),
    )
    .unwrap();
    assert!(spilled.kv_host_bytes > 0);
    assert_eq!(spilled.location_bytes().device, spilled.device_bytes);
    assert_eq!(spilled.location_bytes().host, spilled.kv_host_bytes);
}

#[test]
fn the_host_budget_draws_the_line_at_the_spilled_bytes() {
    let (table, config) = (rco_table(), config());
    let mut settings = settings(131_072);
    settings.context.kv_cache_on_gpu_tokens = 32_768;
    let whole = plan_with(&config, &table, &settings, inputs(64 * GIB, Some(u64::MAX))).unwrap();
    let short = whole.device_bytes - 1;
    let spilled = plan_with(&config, &table, &settings, inputs(short, Some(u64::MAX)))
        .unwrap()
        .kv_host_bytes;
    // Host memory that cannot hold the pages leaves them on the device, and the weights make the room instead.
    let without_pages = plan_with(&config, &table, &settings, inputs(short, Some(spilled - 1))).unwrap();
    assert_eq!(without_pages.kv_host_bytes, 0);
    assert!(!without_pages.offload.is_empty());
    assert_eq!(
        plan_with(&config, &table, &settings, inputs(short, Some(spilled)))
            .unwrap()
            .kv_host_bytes,
        spilled
    );
}

#[test]
fn unknown_budgets_do_not_overflow() {
    let (table, config) = (rco_table(), config());
    let mut settings = settings(131_072);
    settings.context.kv_cache_on_gpu_tokens = 32_768;
    settings.memory.gpu_limit = Auto::Auto;
    let plan = plan_with(&config, &table, &settings, inputs(u64::MAX, Some(u64::MAX))).unwrap();
    assert_eq!(plan.kv_host_tokens, 0);
    assert_eq!(plan.budget_bytes, u64::MAX);
    // A limit and an unlimited host with the device short: the spill is admitted, not an overflow.
    settings.memory.gpu_limit = Auto::Value(ByteSize(plan.device_bytes - 1));
    let spilled = plan_with(&config, &table, &settings, inputs(u64::MAX, Some(u64::MAX))).unwrap();
    assert!(spilled.kv_host_bytes > 0);
    // Nothing fits and the host is unlimited: the refusal reports the numbers it always did.
    settings.memory.gpu_limit = Auto::Value(ByteSize(GIB));
    let Err(PlanError::DeviceTooSmallWithOffload { available, free, .. }) =
        plan_with(&config, &table, &settings, inputs(u64::MAX, Some(u64::MAX)))
    else {
        panic!("expected a refusal");
    };
    assert_eq!((available, free), (GIB, u64::MAX));
}

#[test]
fn an_unknown_host_budget_refuses_a_spill_and_a_stated_one_allows_it() {
    let (table, config) = (rco_table(), config());
    let mut settings = settings(131_072);
    settings.context.kv_cache_on_gpu_tokens = 32_768;
    let whole = plan_with(&config, &table, &settings, inputs(64 * GIB, None)).unwrap();
    assert_eq!(
        whole.kv_host_bytes, 0,
        "a plan that needs no spill needs no host budget"
    );
    let short = whole.device_bytes - 1;
    let spilled = plan_with(&config, &table, &settings, inputs(short, Some(u64::MAX)))
        .unwrap()
        .kv_host_bytes;
    match plan_with(&config, &table, &settings, inputs(short, None)) {
        Err(error @ PlanError::HostBudgetUnknown { needed }) => {
            assert_eq!(needed, spilled);
            let text = error.to_string();
            assert!(text.contains("memory.host-limit") && text.contains("memory.host-weights-limit"));
        }
        other => panic!("expected HostBudgetUnknown, got {other:?}"),
    }
    assert_eq!(
        plan_with(&config, &table, &settings, inputs(short, Some(spilled)))
            .unwrap()
            .kv_host_bytes,
        spilled
    );
    // A device too small even with the spill is the device's refusal, not the host's.
    assert!(matches!(
        plan_with(&config, &table, &settings, inputs(GIB, None)),
        Err(PlanError::DeviceTooSmallWithOffload { .. })
    ));
}

#[test]
fn the_sequence_search_treats_an_unknown_host_budget_as_no_room_to_spill() {
    let (table, config) = (rco_table(), config());
    let settings = settings(131_072);
    let mut two = inputs(u64::MAX, Some(u64::MAX));
    two.sequences = 2;
    let whole_two = plan_with(&config, &table, &settings, two).unwrap();
    // One sequence fits whole; a second fits only with the cache spilled.
    let free = whole_two.device_bytes - 1;
    assert!(plan_with(&config, &table, &settings, inputs(free, Some(u64::MAX))).is_ok());
    let spills = plan_with(
        &config,
        &table,
        &settings,
        PlanInputs {
            sequences: 2,
            ..inputs(free, Some(u64::MAX))
        },
    );
    assert!(spills.unwrap().kv_host_bytes > 0);
    let mut probe = inputs(free, None);
    probe.sequences = 1;
    let known = sequences_for(&config, &table, &settings, inputs(free, Some(0)), 8).unwrap();
    assert_eq!(known, 1);
    assert_eq!(sequences_for(&config, &table, &settings, probe, 8), Ok(known));
}

#[test]
fn the_prefix_cache_keeps_its_size_when_the_host_budget_is_unknown() {
    let (table, config) = (rco_table(), config());
    let mut settings = settings(131_072);
    settings.prefix_cache.size = ByteSize::gib(1);
    let plan = plan_with(&config, &table, &settings, inputs(64 * GIB, None)).unwrap();
    assert_eq!(plan.prefix_cache_bytes, GIB);
}

#[test]
fn an_automatic_count_with_room_for_several_sequences_holds_them_all_in_the_lanes_rings() {
    let (table, config) = (rco_table(), config());
    let mut settings = settings(4096);
    settings.batching.max_sequences = 0;
    let sequences = resolve_sequences(&config, &table, &settings, inputs(64 * GIB, Some(u64::MAX))).unwrap();
    assert!(
        sequences > 1,
        "the plan has room for more than one sequence: {sequences}"
    );
    // The lanes open with the count the plan resolved, so their rings hold the quanta of every one of them.
    let ring = stage_ring_for(sequences);
    assert!(
        ring >= Plan::stage_bound() * sequences,
        "{sequences} sequences, ring {ring}"
    );
    assert!(ring > stage_ring_for(1), "the ring grows with the count: {ring}");
}
