//! The candidates measured first: each pure option, the seeds a file names and random feasible mixes.

use super::super::allocation::Allocation;
use super::super::error::MixedError;
use super::super::feasibility::Feasibility;
use super::Search;
use super::candidate::Candidate;
use super::failure::SearchFailure;
use crate::java_random::Random;

/// Attempts at a random mix that fits before the search gives up on that draw.
const MIX_ATTEMPTS: usize = 100;
/// Floor of the log of a draw of zero.
const TINY: f64 = 1e-300;

/// A draw below `bound`; a bound that is not in `1..=i32::MAX` is refused, naming it.
pub(super) fn below(random: &mut Random, bound: usize) -> Result<usize, MixedError> {
    i32::try_from(bound)
        .ok()
        .and_then(|limit| random.next_int_bounded(limit))
        .and_then(|draw| usize::try_from(draw).ok())
        .ok_or_else(|| {
            MixedError::Invalid(format!(
                "a random draw below {bound}: the bound must be 1 to {}",
                i32::MAX
            ))
        })
}

impl Search<'_> {
    pub(super) fn seed_candidates(
        &self,
        feasibility: &dyn Feasibility,
        budget: i64,
        seed: i64,
    ) -> Result<Vec<Candidate>, SearchFailure> {
        let mut seeds: Vec<Candidate> = self
            .space
            .pure()
            .into_iter()
            .map(|pure| Candidate::given(pure, "seed:pure"))
            .collect();
        for given in &self.settings.seeds {
            seeds.push(Candidate::given(self.space.normalize(given)?, "seed:file"));
        }
        let mut random = Random::new(seed);
        for _ in 0..self.settings.random {
            if let Some(mix) = self.random_mix(&mut random, feasibility, budget)? {
                seeds.push(Candidate::given(mix, "seed:random"));
            }
        }
        Ok(seeds)
    }

    /// A random allocation within the budget that the planner accepts; none after repeated failures.
    fn random_mix(
        &self,
        random: &mut Random,
        feasibility: &dyn Feasibility,
        budget: i64,
    ) -> Result<Option<Allocation>, SearchFailure> {
        let (k, n) = (self.space.options().len(), self.space.objects().len());
        for _ in 0..MIX_ATTEMPTS {
            let weights: Vec<f64> = (0..k).map(|_| -(random.next_double() + TINY).ln()).collect();
            let sum: f64 = weights.iter().sum();
            let mut choice = Vec::with_capacity(n);
            for _ in 0..n {
                let draw = random.next_double() * sum;
                let mut at = 0.0;
                let mut picked = k - 1;
                for (j, weight) in weights.iter().enumerate() {
                    at += weight;
                    if draw < at {
                        picked = j;
                        break;
                    }
                }
                choice.push(picked);
            }
            // Cheapen random objects until the mix is within the budget; after enough draws, every object at once.
            let mut draws = 0;
            while self.space.bytes_of(&choice)? > budget {
                if draws == n.saturating_mul(64) {
                    for (i, slot) in choice.iter_mut().enumerate() {
                        *slot = self.space.cheapest(i);
                    }
                    break;
                }
                let i = below(random, n)?;
                choice[i] = self.space.cheapest(i);
                draws += 1;
            }
            let allocation = self.space.allocation(&choice);
            if feasibility.check(&allocation)?.fits {
                return Ok(Some(allocation));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_draw_with_no_valid_bound_is_refused_naming_it() {
        let mut random = Random::new(1);
        for bound in [0, usize::MAX] {
            let refusal = below(&mut random, bound).unwrap_err().to_string();
            assert!(refusal.contains(&bound.to_string()), "{refusal}");
        }
        assert!(below(&mut random, 5).unwrap() < 5);
    }
}
