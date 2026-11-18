//! The residency plan's fit, spill and refusal are the memory planner's decision: what the plan asks of each
//! location, where the host budget and the device budget draw the line, and that an unknown budget is no overflow.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use euhedral_inference_engine::error::PlanError;
use euhedral_inference_engine::residency::planner::{PlanInputs, plan_with};
use euhedral_inference_settings::{Auto, ByteSize};

mod common;

use common::{config, rco_table, settings};

const GIB: u64 = 1 << 30;

fn inputs(device_free: u64, host_pinned_bytes: u64) -> PlanInputs {
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
    let whole = plan_with(&config, &table, &settings, inputs(64 * GIB, u64::MAX)).unwrap();
    assert_eq!(whole.location_bytes().device, whole.device_bytes);
    assert_eq!(whole.location_bytes().host, 0);
    let spilled = plan_with(&config, &table, &settings, inputs(whole.device_bytes - 1, u64::MAX)).unwrap();
    assert!(spilled.kv_host_bytes > 0);
    assert_eq!(spilled.location_bytes().device, spilled.device_bytes);
    assert_eq!(spilled.location_bytes().host, spilled.kv_host_bytes);
}

#[test]
fn the_host_budget_draws_the_line_at_the_spilled_bytes() {
    let (table, config) = (rco_table(), config());
    let mut settings = settings(131_072);
    settings.context.kv_cache_on_gpu_tokens = 32_768;
    let whole = plan_with(&config, &table, &settings, inputs(64 * GIB, u64::MAX)).unwrap();
    let short = whole.device_bytes - 1;
    let spilled = plan_with(&config, &table, &settings, inputs(short, u64::MAX))
        .unwrap()
        .kv_host_bytes;
    assert!(plan_with(&config, &table, &settings, inputs(short, spilled - 1)).is_err());
    assert_eq!(
        plan_with(&config, &table, &settings, inputs(short, spilled))
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
    let plan = plan_with(&config, &table, &settings, inputs(u64::MAX, u64::MAX)).unwrap();
    assert_eq!(plan.kv_host_tokens, 0);
    assert_eq!(plan.budget_bytes, u64::MAX);
    // A limit and an unknown host with the device short: the spill is admitted, not an overflow.
    settings.memory.gpu_limit = Auto::Value(ByteSize(plan.device_bytes - 1));
    let spilled = plan_with(&config, &table, &settings, inputs(u64::MAX, u64::MAX)).unwrap();
    assert!(spilled.kv_host_bytes > 0);
    // Nothing fits and the host is unknown: the refusal reports the numbers it always did.
    settings.memory.gpu_limit = Auto::Value(ByteSize(GIB));
    let Err(PlanError::WeightsDoNotFit { available, free, .. }) =
        plan_with(&config, &table, &settings, inputs(u64::MAX, u64::MAX))
    else {
        panic!("expected a refusal");
    };
    assert_eq!((available, free), (GIB, u64::MAX));
}
