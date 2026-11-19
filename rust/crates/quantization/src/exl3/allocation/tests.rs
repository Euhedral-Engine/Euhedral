#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use super::*;

fn linear(name: &str, layer: usize, numel: u64, priority: i32, group: &str) -> Linear {
    Linear {
        name: name.to_owned(),
        layer,
        numel,
        priority,
        group: group.to_owned(),
    }
}

fn rates(linears: &[Linear], layers: usize, bpw: f64) -> Vec<f64> {
    allocate(linears, layers, bpw)
        .unwrap()
        .into_iter()
        .map(|(_, rate)| rate)
        .collect()
}

// A mutant that promotes past the budget, or counts the budget in whole bits instead of weights, breaks the bound.
#[test]
fn average_never_exceeds_the_target_and_rates_are_supported() {
    let linears: Vec<Linear> = (0..8)
        .flat_map(|layer| {
            [
                linear(
                    &format!("l{layer}.a"),
                    layer,
                    1000 + 37 * layer as u64,
                    1,
                    &format!("l{layer}.a"),
                ),
                linear(&format!("l{layer}.b"), layer, 3000, 0, &format!("l{layer}.b")),
            ]
        })
        .collect();
    let total: f64 = linears.iter().map(|l| l.numel as f64).sum();
    for bpw in [1.0, 2.3, 2.5, 2.75, 3.1, 3.99, 4.0, 4.6, 7.9] {
        let rates = rates(&linears, 8, bpw);
        let used: f64 = linears.iter().zip(&rates).map(|(l, r)| l.numel as f64 * r).sum();
        assert!(used <= (bpw * total).trunc(), "{bpw}: {used}");
        for rate in rates {
            let step = if rate < 4.0 { rate * 2.0 } else { rate };
            assert_eq!(step, step.round(), "{bpw}: {rate} is not a supported rate");
            assert!((1.0..=8.0).contains(&rate));
        }
    }
}

// A mutant that sorts the priorities the other way promotes the second linear.
#[test]
fn the_higher_priority_group_is_promoted_first() {
    let linears = [linear("low", 1, 100, 0, "low"), linear("high", 1, 100, 2, "high")];
    // The budget has room for one half-bit promotion of 100 weights (550 of 560), not two.
    assert_eq!(rates(&linears, 4, 2.8), [2.5, 3.0]);
}

// A mutant that orders by layer instead of by distance to the nearer end promotes layer 1.
#[test]
fn equal_priorities_promote_the_layer_nearest_an_end_first() {
    let linears = [linear("inner", 1, 100, 1, "inner"), linear("last", 3, 100, 1, "last")];
    assert_eq!(rates(&linears, 4, 2.8), [2.5, 3.0]);
}

// A mutant that promotes members one at a time lets the group split.
#[test]
fn a_group_is_promoted_together_or_not_at_all() {
    let linears = [linear("a", 0, 100, 1, "g"), linear("b", 0, 100, 1, "g")];
    // One member alone would fit (50 of 60 spare), the pair (100) does not.
    assert_eq!(rates(&linears, 1, 2.8), [2.5, 2.5]);
    assert_eq!(rates(&linears, 1, 3.0), [3.0, 3.0]);
}

// A mutant that rounds the floor instead of truncating starts above the target.
#[test]
fn the_base_rate_is_the_largest_supported_rate_not_above_the_target() {
    let linears = [linear("a", 0, 100, 1, "a")];
    assert_eq!(rates(&linears, 1, 2.99), [2.5]);
    assert_eq!(rates(&linears, 1, 4.9), [4.0]);
    assert_eq!(rates(&linears, 1, 0.3), [1.0]);
    assert_eq!(rates(&linears, 1, 12.0), [8.0]);
}

// A mutant that accepts a zero or NaN target loops or returns rates for nothing.
#[test]
fn a_target_that_is_not_positive_is_refused() {
    let linears = [linear("a", 0, 100, 1, "a")];
    for bpw in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(
            matches!(allocate(&linears, 1, bpw), Err(AllocationError::BadTarget(_))),
            "{bpw}"
        );
    }
}

// A mutant without the layer check underflows the distance to the far end.
#[test]
fn a_linear_beyond_the_stack_is_refused() {
    let linears = [linear("a", 4, 100, 1, "a")];
    assert!(matches!(
        allocate(&linears, 4, 3.0),
        Err(AllocationError::BadLayer {
            layer: 4,
            layers: 4,
            ..
        })
    ));
}
