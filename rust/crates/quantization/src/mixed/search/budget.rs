//! The byte budget RCO may use, found on a nested family of allocations.

use super::super::feasibility::Feasibility;
use super::Search;
use super::failure::SearchFailure;

impl Search<'_> {
    /// The choice that puts the first `dearest` objects on their dearest option and the rest on their cheapest.
    fn family_choice(&self, dearest: usize) -> Vec<usize> {
        (0..self.space.objects().len())
            .map(|i| {
                if i < dearest {
                    self.space.dearest(i)
                } else {
                    self.space.cheapest(i)
                }
            })
            .collect()
    }

    /// The bytes of projections RCO may use: the largest allocation of a nested family (the first `k` objects on their
    /// dearest option, the rest on their cheapest) the planner accepts, found by bisection on `k`, plus the device bytes
    /// the plan leaves free there (capped below the next member of the family). A proposal is checked by the planner
    /// itself and, if it fails, re-solved with a smaller budget, so the figure only has to be near.
    pub(super) fn derive(&self, feasibility: &dyn Feasibility) -> Result<i64, SearchFailure> {
        let n = self.space.objects().len();
        let fits = |dearest: usize| -> Result<_, SearchFailure> {
            Ok(feasibility.check(&self.space.allocation(&self.family_choice(dearest)))?)
        };
        if !fits(0)?.fits {
            return Err(SearchFailure::NothingFits {
                context: self.settings.context,
                free_mib: feasibility.free_bytes() >> 20,
            });
        }
        if fits(n)?.fits {
            return Ok(self.space.bytes_of(&self.family_choice(n))?);
        }
        // Invariant: `n_fits` fits and `n_fails` does not; the family is nested, so the first that fails is found.
        let (mut n_fits, mut n_fails) = (0, n);
        while n_fails - n_fits > 1 {
            let middle = n_fits + (n_fails - n_fits) / 2;
            if fits(middle)?.fits {
                n_fits = middle;
            } else {
                n_fails = middle;
            }
        }
        let headroom = (feasibility.free_bytes() - fits(n_fits)?.device_bytes).max(0);
        let largest = self.space.bytes_of(&self.family_choice(n_fits))?;
        let next = self.space.bytes_of(&self.family_choice(n_fails))?;
        Ok(largest.saturating_add(headroom).min(next))
    }
}
