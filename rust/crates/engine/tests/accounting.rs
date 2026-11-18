//! The account of the engine's memory on the plans of the RCO artifact (`tests/fixtures/rco_table.tsv`), with fake
//! device sizes: the admitted bytes are the plan's, an allocation larger than its share is refused by name, and an
//! account whose allocations are freed ends clean.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use euhedral_inference_engine::error::PlanError;
use euhedral_inference_engine::residency::accounting::{Consumer, MemoryAccount, PlanReservation};
use euhedral_inference_engine::residency::planner::{LocationBytes, PlanInputs, ResidencyPlan, plan_with};
use euhedral_inference_settings::ByteSize;

mod common;

use common::{config, rco_table, settings};

const GIB: u64 = 1 << 30;

fn inputs(sequences: usize) -> PlanInputs {
    PlanInputs {
        device_free: 64 * GIB,
        multiprocessors: Some(70),
        host_pinned_bytes: u64::MAX,
        sequences,
    }
}

/// A plan that is resident, and one that spills its cache to the host, holds two sequences and a prefix cache.
fn plans() -> Vec<ResidencyPlan> {
    let (config, table) = (config(), rco_table());
    let resident = plan_with(&config, &table, &settings(131_072), inputs(1)).unwrap();
    let mut spilling = settings(131_072);
    spilling.context.kv_cache_on_gpu_tokens = 32_768;
    spilling.prefix_cache.size = ByteSize::gib(1);
    // A device a byte short of the whole caches of both sequences: the cache spills.
    let whole = plan_with(&config, &table, &spilling, inputs(2)).unwrap();
    assert_eq!(whole.kv_host_bytes, 0);
    let mut inputs = inputs(2);
    inputs.device_free = whole.device_bytes - 1;
    let spilled = plan_with(&config, &table, &spilling, inputs).unwrap();
    assert!(spilled.kv_host_bytes > 0, "the second plan spills");
    assert!(spilled.prefix_cache_bytes > 0, "the second plan has a prefix cache");
    assert_eq!(spilled.sequences, 2);
    vec![resident, spilled]
}

fn admit(plan: &ResidencyPlan) -> (MemoryAccount, PlanReservation) {
    let account = MemoryAccount::open(plan.budget_bytes, u64::MAX);
    let reservation = account.admit(plan).unwrap();
    (account, reservation)
}

#[test]
fn the_admitted_bytes_are_the_plans() {
    for plan in plans() {
        let consumers = plan.consumers();
        let device: u64 = consumers
            .iter()
            .filter(|asked| !asked.host)
            .map(|asked| asked.bytes)
            .sum();
        let host: u64 = consumers
            .iter()
            .filter(|asked| asked.host)
            .map(|asked| asked.bytes)
            .sum();
        assert_eq!(device, plan.device_bytes);
        assert_eq!(host, plan.kv_host_bytes + plan.prefix_cache_bytes);
        let (account, reservation) = admit(&plan);
        assert_eq!(
            reservation.admitted(),
            LocationBytes {
                host: plan.kv_host_bytes + plan.prefix_cache_bytes,
                device: plan.device_bytes,
            }
        );
        // Each consumer may take what the plan names for it, and the account holds it.
        assert_eq!(reservation.left(Consumer::Weights), plan.arena_bytes);
        assert_eq!(reservation.left(Consumer::Scratch), plan.scratch_bytes);
        assert_eq!(
            reservation.left(Consumer::Cache),
            plan.kv_device_bytes * plan.sequences as u64
        );
        assert_eq!(reservation.left(Consumer::HostPages), plan.kv_host_bytes);
        assert_eq!(reservation.left(Consumer::PrefixCache), plan.prefix_cache_bytes);
        assert_eq!(account.held(), reservation.admitted());
        drop(reservation);
        assert!(account.close().is_clean());
    }
}

#[test]
fn a_claim_beyond_the_plan_is_refused_and_named() {
    let plan = plans().remove(0);
    let (account, mut reservation) = admit(&plan);
    let refused = reservation
        .claim(Consumer::Weights, plan.arena_bytes + 1)
        .unwrap_err()
        .to_string();
    assert!(refused.contains("weights"), "{refused}");
    assert!(refused.contains(&(plan.arena_bytes + 1).to_string()), "{refused}");
    // The refusal took nothing: the whole share is still there.
    let weights = reservation.claim(Consumer::Weights, plan.arena_bytes).unwrap();
    // A share taken once cannot be taken again.
    let again = reservation.claim(Consumer::Weights, 1).unwrap_err().to_string();
    assert!(again.contains("weights"), "{again}");
    // A consumer the plan has none of (no spill here), or the bytes beyond its share, is refused by name too.
    let host = reservation.claim(Consumer::HostPages, 1).unwrap_err().to_string();
    assert!(host.contains("cache pages on the host"), "{host}");
    let prefix = reservation
        .claim(Consumer::PrefixCache, plan.prefix_cache_bytes + 1)
        .unwrap_err()
        .to_string();
    assert!(prefix.contains("prefix cache"), "{prefix}");
    drop(weights);
    drop(reservation);
    assert!(account.close().is_clean());
}

#[test]
fn a_host_that_cannot_hold_the_plan_is_not_admitted() {
    let plan = plans().remove(1);
    let account = MemoryAccount::open(plan.budget_bytes, plan.kv_host_bytes - 1);
    assert!(matches!(account.admit(&plan), Err(PlanError::NotAdmitted(_))));
}

#[test]
fn an_owner_ended_after_a_fake_plan_and_allocate_is_clean() {
    for plan in plans() {
        let (account, mut reservation) = admit(&plan);
        // The allocations of a startup, each a claim of its consumer's share.
        let mut claims = Vec::new();
        for asked in plan.consumers() {
            claims.push(reservation.claim(asked.consumer, asked.bytes).unwrap());
        }
        assert_eq!(account.held(), reservation.admitted());
        drop(claims);
        drop(reservation);
        let report = account.close();
        assert!(report.is_clean(), "{report:?}");
        assert_eq!(account.held(), LocationBytes { host: 0, device: 0 });
    }
}

#[test]
fn a_claim_still_held_at_the_end_is_a_leak_of_its_bytes() {
    let plan = plans().remove(0);
    let (account, mut reservation) = admit(&plan);
    let leaked = reservation.claim(Consumer::Cache, plan.kv_device_bytes).unwrap();
    // The weights are freed, and the rest of the reservation is the account's own: only the cache leaks.
    drop(reservation.claim(Consumer::Weights, plan.arena_bytes).unwrap());
    drop(reservation);
    let report = account.close();
    assert!(!report.is_clean());
    assert_eq!(report.bytes(), plan.kv_device_bytes);
    drop(leaked);
}
