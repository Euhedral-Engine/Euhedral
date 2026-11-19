//! The multiple-choice knapsack: one option per group, the most score under a byte budget.

use std::mem::swap;

use super::error::MixedError;

/// The most options one group may have: a choice is kept as a byte per group and budget step.
const MAX_OPTIONS: usize = 256;

/// The option of each group that maximizes the sum of `score` with the sum of `cost` at most `budget`, by dynamic
/// programming over the budget in steps of `unit` bytes: a cost counts as its nearest whole number of units (halves
/// up), the budget as its whole units. Of equal sums the lowest option of the earliest group that differs wins.
/// Time and memory are `O(groups * options * budget / unit)`.
///
/// Refuses a unit below one byte, score and cost that differ in shape, a negative cost, a count of units that does not
/// fit 31 bits, and, as [`MixedError::BudgetTooSmall`], a budget below the cheapest choices.
pub fn solve(score: &[Vec<f64>], cost: &[Vec<i64>], budget: i64, unit: i64) -> Result<Vec<usize>, MixedError> {
    if unit < 1 {
        return Err(MixedError::invalid("the unit must be at least one byte"));
    }
    if score.len() != cost.len() {
        return Err(MixedError::invalid("score and cost differ in groups"));
    }
    let groups = score.len();
    if budget < 0 {
        return Err(MixedError::BudgetTooSmall(budget));
    }
    let capacity = whole_units(budget / unit)?;
    let mut units: Vec<Vec<usize>> = Vec::with_capacity(groups);
    for i in 0..groups {
        if score[i].len() != cost[i].len() {
            return Err(MixedError::invalid(format!(
                "score and cost differ in options at group {i}"
            )));
        }
        if cost[i].len() > MAX_OPTIONS {
            return Err(MixedError::invalid(format!(
                "group {i} has more than {MAX_OPTIONS} options"
            )));
        }
        let mut row = Vec::with_capacity(cost[i].len());
        for &bytes in &cost[i] {
            if bytes < 0 {
                return Err(MixedError::invalid(format!("group {i} has a negative cost")));
            }
            row.push(whole_units(nearest(bytes as f64 / unit as f64))?);
        }
        units.push(row);
    }
    let mut choice = vec![vec![0u8; capacity + 1]; groups];
    let mut best = vec![0.0f64; capacity + 1];
    let mut next = vec![f64::NEG_INFINITY; capacity + 1];
    for i in 0..groups {
        next.fill(f64::NEG_INFINITY);
        let picked = &mut choice[i];
        for (k, &weight) in units[i].iter().enumerate() {
            let value = score[i][k];
            for b in weight..=capacity {
                let candidate = best[b - weight] + value;
                if candidate > next[b] {
                    next[b] = candidate;
                    picked[b] = k as u8;
                }
            }
        }
        swap(&mut best, &mut next);
    }
    if best[capacity] == f64::NEG_INFINITY {
        return Err(MixedError::BudgetTooSmall(budget));
    }
    let mut result = vec![0usize; groups];
    let mut remaining = capacity;
    for i in (0..groups).rev() {
        result[i] = usize::from(choice[i][remaining]);
        remaining -= units[i][result[i]];
    }
    Ok(result)
}

/// `x` rounded to the nearest whole number, halves up (`Math.round`).
fn nearest(x: f64) -> i64 {
    let floor = x.floor();
    (if x - floor >= 0.5 { floor + 1.0 } else { floor }) as i64
}

fn whole_units(count: i64) -> Result<usize, MixedError> {
    i32::try_from(count)
        .ok()
        .and_then(|count| usize::try_from(count).ok())
        .ok_or_else(|| {
            MixedError::invalid(format!(
                "{count} units of the budget do not fit 31 bits: use a larger unit"
            ))
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::java_random::Random;

    fn below(random: &mut Random, bound: i32) -> i32 {
        random.next_int_bounded(bound).unwrap()
    }

    fn best_by_brute_force(score: &[Vec<f64>], cost: &[Vec<i64>], budget: i64) -> f64 {
        let n = score.len();
        let total: usize = score.iter().map(Vec::len).product();
        let mut best = f64::NEG_INFINITY;
        for code in 0..total {
            let mut rest = code;
            let (mut bytes, mut value) = (0, 0.0);
            for i in 0..n {
                let pick = rest % score[i].len();
                rest /= score[i].len();
                bytes += cost[i][pick];
                value += score[i][pick];
            }
            if bytes <= budget && value > best {
                best = value;
            }
        }
        best
    }

    #[test]
    fn matches_brute_force_on_random_small_instances() {
        let mut random = Random::new(5);
        for trial in 0..200 {
            let n = 2 + below(&mut random, 5) as usize;
            let k = 2 + below(&mut random, 3) as usize;
            let mut score = vec![vec![0.0; k]; n];
            let mut cost = vec![vec![0i64; k]; n];
            let (mut minimum, mut maximum) = (0i64, 0i64);
            for i in 0..n {
                let (mut low, mut high) = (i64::MAX, 0i64);
                for j in 0..k {
                    score[i][j] = random.next_gaussian();
                    cost[i][j] = 1 + i64::from(below(&mut random, 20));
                    low = low.min(cost[i][j]);
                    high = high.max(cost[i][j]);
                }
                minimum += low;
                maximum += high;
            }
            let budget = minimum + i64::from(below(&mut random, (maximum - minimum + 1) as i32));
            let choice = solve(&score, &cost, budget, 1).unwrap();
            let bytes: i64 = (0..n).map(|i| cost[i][choice[i]]).sum();
            let value: f64 = (0..n).map(|i| score[i][choice[i]]).sum();
            assert!(bytes <= budget, "trial {trial}");
            assert!(
                (best_by_brute_force(&score, &cost, budget) - value).abs() < 1e-12,
                "trial {trial}"
            );
        }
    }

    #[test]
    fn costs_round_to_the_unit() {
        // 100 and 300 bytes with a unit of 100 are 1 and 3 units; a budget of 4 units takes the better of the two groups.
        let score = vec![vec![0.0, 1.0], vec![0.0, 2.0]];
        let cost = vec![vec![100, 300], vec![100, 300]];
        assert_eq!(solve(&score, &cost, 400, 100).unwrap(), [0, 1]);
        // 149 counts as 1 unit and 150 as 2: the half rounds up.
        let score = vec![vec![0.0, 1.0]];
        assert_eq!(solve(&score, &[vec![0, 149]], 100, 100).unwrap(), [1]);
        assert_eq!(solve(&score, &[vec![0, 150]], 100, 100).unwrap(), [0]);
    }

    #[test]
    fn equal_sums_take_the_lowest_option() {
        let score = vec![vec![1.0, 1.0, 1.0]];
        assert_eq!(solve(&score, &[vec![1, 1, 1]], 1, 1).unwrap(), [0]);
    }

    #[test]
    fn a_budget_below_the_cheapest_choices_is_refused() {
        let failure = solve(&[vec![0.0, 1.0]], &[vec![5, 9]], 4, 1).unwrap_err();
        assert_eq!(failure, MixedError::BudgetTooSmall(4));
        assert!(failure.to_string().contains("budget"));
        assert_eq!(
            solve(&[vec![0.0]], &[vec![0]], -1, 1).unwrap_err(),
            MixedError::BudgetTooSmall(-1)
        );
    }

    #[test]
    fn malformed_instances_are_refused() {
        assert!(solve(&[vec![0.0]], &[vec![1]], 1, 0).is_err());
        assert!(solve(&[vec![0.0]], &[], 1, 1).is_err());
        assert!(solve(&[vec![0.0, 1.0]], &[vec![1]], 1, 1).is_err());
        assert!(solve(&[vec![0.0]], &[vec![-1]], 1, 1).is_err());
        assert!(solve(&[vec![0.0]], &[vec![1]], i64::MAX, 1).is_err());
    }

    #[test]
    fn more_than_a_hundred_and_twenty_seven_options_pick_the_right_one() {
        let options = 200;
        let mut score = vec![0.0; options];
        score[190] = 1.0;
        let cost = vec![1i64; options];
        assert_eq!(solve(&[score], &[cost], 1, 1).unwrap(), [190]);
    }
}
