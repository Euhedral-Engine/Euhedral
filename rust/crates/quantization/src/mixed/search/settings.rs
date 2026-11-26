use super::super::allocation::Allocation;
use super::super::rco;
use super::failure::SearchFailure;
use crate::mixed::VERIFY_MAX_ROWS;

/// The most candidates a round or the seeds may ask for.
pub const MOST_CANDIDATES: usize = 4096;
/// The most members of the ensemble.
pub const MOST_MEMBERS: usize = 256;
/// The most weights in the lambda list (the sweep runs their square) or budget fractions.
pub const MOST_WEIGHTS: usize = 16;
/// The most knapsack steps.
pub const MOST_KNAPSACK_STEPS: u32 = 1 << 20;
/// The most bands per family.
pub const MOST_BANDS: usize = 4096;

/// Everything a search is told besides its sources, priors, costs and the device.
///
/// `identity` is the text that tells this search from another for the resume file (the references and the cost table,
/// and the paths of the sources): the measurements depend on them. Priors and the scoring batch only steer proposals
/// and how many chunks run together, so a search may change either and resume.
#[derive(Debug, Clone)]
pub struct Settings {
    /// The target context the composite must fit with every object on the device.
    pub context: u32,
    /// Allocations to measure first (objects they do not name keep the base's option).
    pub seeds: Vec<Allocation>,
    /// Candidates per round.
    pub batch: usize,
    /// The share of a batch proposed for what the ensemble is unsure of.
    pub explore: f64,
    /// Rounds without a new front point after which the search stops.
    pub patience: u32,
    /// Feasible random mixes to measure first.
    pub random: usize,
    /// Bands per family of the fidelity model.
    pub bands: usize,
    /// The rows of the verification step whose time is on the front.
    pub verify_rows: u32,
    /// Models in the ensemble.
    pub members: usize,
    /// The fidelity model's ridge.
    pub ridge: f64,
    /// Weights of the normalized prefill and verification costs, in KL.
    pub lambdas: Vec<f64>,
    /// The budgets RCO is run at, as shares of the way from the cheapest allocation to the largest that fits.
    pub budget_fractions: Vec<f64>,
    pub rco: rco::Settings,
    /// The budget's resolution in the knapsack.
    pub knapsack_steps: u32,
    /// The rounds after which the search stops whatever the front.
    pub max_rounds: u32,
    /// The seed of every random choice.
    pub seed: i64,
    pub identity: String,
}

impl Settings {
    /// Refuses a setting outside its limit, naming it and the value.
    pub fn validate(&self) -> Result<(), SearchFailure> {
        let range = |name: &str, value: usize, most: usize| {
            if (1..=most).contains(&value) {
                Ok(())
            } else {
                Err(SearchFailure::refused(format!(
                    "{name} is {value}, not from 1 to {most}"
                )))
            }
        };
        if self.context < 1 {
            return Err(SearchFailure::refused("context is 0, not at least 1 token"));
        }
        range("batch", self.batch, MOST_CANDIDATES)?;
        if self.random > MOST_CANDIDATES {
            return Err(SearchFailure::refused(format!(
                "random is {}, not from 0 to {MOST_CANDIDATES}",
                self.random
            )));
        }
        range("members", self.members, MOST_MEMBERS)?;
        range("bands", self.bands, MOST_BANDS)?;
        if !(0.0..=1.0).contains(&self.explore) {
            return Err(SearchFailure::refused(format!(
                "explore is {}, not from 0 to 1",
                self.explore
            )));
        }
        if self.patience < 1 {
            return Err(SearchFailure::refused("patience is 0, not at least 1 round"));
        }
        if self.max_rounds < 1 {
            return Err(SearchFailure::refused("max rounds is 0, not at least 1"));
        }
        if !(2..=VERIFY_MAX_ROWS).contains(&self.verify_rows) {
            return Err(SearchFailure::refused(format!(
                "verify rows is {}, not from 2 to {VERIFY_MAX_ROWS}: the composite drafts one token fewer",
                self.verify_rows
            )));
        }
        if !self.ridge.is_finite() || self.ridge < 0.0 {
            return Err(SearchFailure::refused(format!(
                "ridge is {}, not a number of at least 0",
                self.ridge
            )));
        }
        if !(1..=MOST_KNAPSACK_STEPS).contains(&self.knapsack_steps) {
            return Err(SearchFailure::refused(format!(
                "knapsack steps is {}, not from 1 to {MOST_KNAPSACK_STEPS}",
                self.knapsack_steps
            )));
        }
        let weights = |name: &str, values: &[f64], low: f64, high: f64| {
            if values.is_empty() || values.len() > MOST_WEIGHTS {
                return Err(SearchFailure::refused(format!(
                    "{name} has {} values, not from 1 to {MOST_WEIGHTS}",
                    values.len()
                )));
            }
            match values.iter().find(|value| !(low..=high).contains(*value)) {
                Some(value) => Err(SearchFailure::refused(format!(
                    "{name} holds {value}, not a number from {low} to {high}"
                ))),
                None => Ok(()),
            }
        };
        weights("lambdas", &self.lambdas, 0.0, f64::MAX)?;
        weights("budget fractions", &self.budget_fractions, 0.0, 1.0)?;
        Ok(())
    }
}
