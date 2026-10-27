//! The drift verdict: the thresholds of the relaxed-numerics drift test on synthetic steps, and the comparison of two rows.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use euhedral_inference_bench::drift::{
    BURN_IN, Candidate, PrefixSummary, StepMetrics, WINDOW, breaches, compare_step, summarize,
};

fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

fn step(hidden: f64, kl: f64, agrees: bool, lead: f64) -> StepMetrics {
    StepMetrics {
        hidden_relative: hidden,
        kl,
        top_agrees: agrees,
        flip_lead: lead,
        bitwise: false,
    }
}

fn passing() -> Vec<StepMetrics> {
    (0..512).map(|_| step(0.06, 1e-4, true, 0.0)).collect()
}

fn trips(metrics: &[StepMetrics], candidate: Candidate) -> Vec<String> {
    breaches(&[summarize(metrics)], candidate)
}

#[test]
fn drift_thresholds_trip() {
    assert!(trips(&passing(), Candidate::HostTier).is_empty());

    // Mean KL above 1e-2 trips; just under does not.
    let mut high_kl = passing();
    high_kl.iter_mut().for_each(|s| s.kl = 1.01e-2);
    assert!(
        trips(&high_kl, Candidate::HostTier)
            .iter()
            .any(|b| b.contains("mean KL"))
    );
    high_kl.iter_mut().for_each(|s| s.kl = 0.99e-2);
    assert!(trips(&high_kl, Candidate::HostTier).is_empty());

    // More than 2% of 512 steps (10.24, so 11) flipping with a lead above 0.25 trips; near-ties do not count.
    let mut flips = passing();
    flips
        .iter_mut()
        .take(10)
        .for_each(|s| (s.top_agrees, s.flip_lead) = (false, 1.0));
    assert!(trips(&flips, Candidate::HostTier).is_empty(), "10 of 512 is within 2%");
    flips[10] = step(0.06, 1e-4, false, 1.0);
    assert!(
        trips(&flips, Candidate::HostTier)
            .iter()
            .any(|b| b.contains("flip top-1"))
    );
    let mut ties = passing();
    ties.iter_mut()
        .take(100)
        .for_each(|s| (s.top_agrees, s.flip_lead) = (false, 0.25));
    assert!(
        trips(&ties, Candidate::HostTier).is_empty(),
        "a lead of 0.25 is a near-tie"
    );

    // A 32-step window averaging 12% trips, 11.9% does not (the median of one position is its peak).
    let mut peak = passing();
    peak[192..192 + WINDOW]
        .iter_mut()
        .for_each(|s| s.hidden_relative = 0.12);
    assert!(
        trips(&peak, Candidate::HostTier)
            .iter()
            .any(|b| b.contains("peak window"))
    );
    peak[192..192 + WINDOW]
        .iter_mut()
        .for_each(|s| s.hidden_relative = 0.119);
    assert!(trips(&peak, Candidate::HostTier).is_empty());

    // The error doubling after the burn-in trips the growth bound: late / early above 1.5.
    let mut growing = passing();
    let settled = BURN_IN + (512 - BURN_IN) / 2;
    growing[settled..].iter_mut().for_each(|s| s.hidden_relative = 0.11);
    growing[BURN_IN..settled]
        .iter_mut()
        .for_each(|s| s.hidden_relative = 0.06);
    let found = trips(&growing, Candidate::HostTier);
    assert!(found.iter().any(|b| b.contains("growth")), "{found:?}");
}

#[test]
fn the_hidden_bounds_are_medians_over_the_positions() {
    let good = summarize(&passing());
    let mut bad_peak = good.clone();
    bad_peak.peak = 0.5;
    let mut bad_growth = good.clone();
    bad_growth.growth = 3.0;
    // One bad position of three does not move the median; two of three do.
    assert!(breaches(&[good.clone(), good.clone(), bad_peak.clone()], Candidate::HostTier).is_empty());
    assert!(
        breaches(&[good.clone(), bad_peak.clone(), bad_peak], Candidate::HostTier)
            .iter()
            .any(|b| b.contains("peak"))
    );
    assert!(breaches(&[good.clone(), good.clone(), bad_growth.clone()], Candidate::HostTier).is_empty());
    assert!(
        breaches(&[good, bad_growth.clone(), bad_growth], Candidate::HostTier)
            .iter()
            .any(|b| b.contains("growth"))
    );
}

#[test]
fn the_exact_candidate_must_agree_bitwise() {
    let mut metrics = passing();
    metrics.iter_mut().for_each(|s| s.bitwise = true);
    assert!(trips(&metrics, Candidate::Exact).is_empty());
    metrics[300].bitwise = false;
    assert!(trips(&metrics, Candidate::Exact).iter().any(|b| b.contains("differs")));
    // The host-tier candidate is bounded, not equal.
    assert!(trips(&metrics, Candidate::HostTier).is_empty());
}

#[test]
fn two_equal_rows_compare_equal_and_a_changed_row_does_not() {
    let a = bf16(&[3.0, 1.0, 0.0, -2.0]);
    let hidden = bf16(&[1.0, -2.0, 0.5, 4.0]);
    let same = compare_step(&a, &a, &hidden, &hidden);
    assert_eq!(
        (
            same.hidden_relative,
            same.kl,
            same.top_agrees,
            same.flip_lead,
            same.bitwise
        ),
        (0.0, 0.0, true, 0.0, true)
    );

    // The second run prefers token 1: the first run's lead of its top over the second's top is 3 - 1 = 2.
    let b = bf16(&[2.0, 3.0, 0.0, -2.0]);
    let other_hidden = bf16(&[1.0, -2.0, 0.5, 3.0]);
    let changed = compare_step(&a, &b, &hidden, &other_hidden);
    assert!(!changed.top_agrees && !changed.bitwise);
    assert_eq!(changed.flip_lead, 2.0);
    assert!(changed.kl > 0.0);
    // sqrt((4-3)^2 / (1 + 4 + 0.25 + 16))
    assert!((changed.hidden_relative - (1.0f64 / 21.25).sqrt()).abs() < 1e-12);
}

#[test]
fn a_summary_counts_windows_and_agreement() {
    let mut metrics = passing();
    metrics[0] = step(0.06, 1e-4, false, 0.1);
    let summary: PrefixSummary = summarize(&metrics);
    assert_eq!(
        (summary.steps, summary.top_agreements, summary.decided_flips),
        (512, 511, 0)
    );
    assert!((summary.peak - (0.06 * 31.0 + 0.06) / 32.0).abs() < 1e-12);
}
