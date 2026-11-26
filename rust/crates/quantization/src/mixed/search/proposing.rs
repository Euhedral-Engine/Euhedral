//! A round's proposals: the ensemble fitted to what was measured, then an RCO sweep over the cost weights and budgets
//! whose results are repaired to fit the device.

use std::collections::HashSet;
use std::sync::Arc;

use super::super::allocation::Allocation;
use super::super::ensemble::Ensemble;
use super::super::error::MixedError;
use super::super::feasibility::Feasibility;
use super::super::fidelity::FidelityModel;
use super::super::knapsack;
use super::super::rco::{self, Objective as RcoObjective};
use super::super::search_state::Measured;
use super::super::space::Space;
use super::candidate::Candidate;
use super::failure::SearchFailure;
use super::ports::Fan;
use super::Search;

/// Repairs of an assignment that does not fit, each with a smaller byte limit.
const FIT_REPAIRS: usize = 12;
/// The multiplier of a round in a cell's seed.
const ROUND_STRIDE: i64 = 1_000_003;
/// The multiplier of a cell's index in its seed.
const CELL_STRIDE: i64 = 7919;
/// The multiplier of the seed in a round's ensemble.
const ENSEMBLE_STRIDE: i64 = 31;
/// A floor for the log of a probability.
const TINY: f64 = 1e-300;

/// The ensemble a round fitted and the candidates its sweep proposed, each cell's feasible results.
pub(super) struct Proposals {
    pub ensemble: Arc<Ensemble>,
    pub candidates: Vec<Candidate>,
}

/// The objective RCO minimizes: the ensemble's KL plus the weights times the normalized prefill and verification
/// costs, each the share of the way between the cheapest and the dearest allocation.
pub(super) struct Objective {
    ensemble: Arc<Ensemble>,
    space: Arc<Space>,
    lambda_prefill: f64,
    lambda_decode: f64,
    prefill_low: f64,
    prefill_range: f64,
    decode_low: f64,
    decode_range: f64,
}

/// The sum over objects of the lowest and of the highest cost of an option.
fn extent(cost: &[Vec<f64>]) -> (f64, f64) {
    let (mut low, mut high) = (0.0, 0.0);
    for row in cost {
        low += row.iter().copied().fold(f64::INFINITY, f64::min);
        high += row.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    }
    (low, high)
}

fn expected(cost: &[Vec<f64>], p: &[Vec<f64>]) -> f64 {
    p.iter()
        .zip(cost)
        .map(|(probabilities, costs)| probabilities.iter().zip(costs).map(|(p, cost)| p * cost).sum::<f64>())
        .sum()
}

impl Objective {
    pub fn new(ensemble: Arc<Ensemble>, space: Arc<Space>, lambda_prefill: f64, lambda_decode: f64) -> Objective {
        let (prefill_low, prefill_high) = extent(space.prefill());
        let (decode_low, decode_high) = extent(space.decode());
        Objective {
            ensemble,
            space,
            lambda_prefill,
            lambda_decode,
            prefill_low,
            prefill_range: prefill_high - prefill_low,
            decode_low,
            decode_range: decode_high - decode_low,
        }
    }
}

impl RcoObjective for Objective {
    fn value(&self, p: &[Vec<f64>]) -> Result<f64, MixedError> {
        let mut value = self.ensemble.mean_soft(p)?;
        if self.lambda_prefill != 0.0 && self.prefill_range > 0.0 {
            value += self.lambda_prefill * (expected(self.space.prefill(), p) - self.prefill_low) / self.prefill_range;
        }
        if self.lambda_decode != 0.0 && self.decode_range > 0.0 {
            value += self.lambda_decode * (expected(self.space.decode(), p) - self.decode_low) / self.decode_range;
        }
        Ok(value)
    }

    fn gradient(&self, p: &[Vec<f64>]) -> Result<Vec<Vec<f64>>, MixedError> {
        let mut gradient = self.ensemble.gradient(p)?;
        for (i, row) in gradient.iter_mut().enumerate() {
            for (k, slot) in row.iter_mut().enumerate() {
                if self.lambda_prefill != 0.0 && self.prefill_range > 0.0 {
                    *slot += self.lambda_prefill * self.space.prefill()[i][k] / self.prefill_range;
                }
                if self.lambda_decode != 0.0 && self.decode_range > 0.0 {
                    *slot += self.lambda_decode * self.space.decode()[i][k] / self.decode_range;
                }
            }
        }
        Ok(gradient)
    }
}

/// One RCO run of the sweep, owning what it needs so it can run on any worker.
pub(super) struct Cell {
    pub space: Arc<Space>,
    pub feasibility: Arc<dyn Feasibility>,
    pub ensemble: Arc<Ensemble>,
    pub rco: rco::Settings,
    pub knapsack_steps: u32,
    /// The bytes of projections the search derived for the device: the budget fractions are shares of the way to it.
    pub ceiling: i64,
    pub lambda_prefill: f64,
    pub lambda_decode: f64,
    pub lambda_index: usize,
    pub fraction: f64,
    pub restart: usize,
    pub seed: i64,
}

impl Cell {
    /// The cell's run, its assignment made to fit the device; none when it cannot be made to fit.
    pub fn run(self) -> Result<Option<Candidate>, MixedError> {
        let minimum = self.space.minimum_bytes()?;
        let budget = minimum.saturating_add((self.fraction * (self.ceiling - minimum).max(0) as f64) as i64);
        let unit = (budget / i64::from(self.knapsack_steps)).max(1);
        let objective = Objective::new(Arc::clone(&self.ensemble), Arc::clone(&self.space), self.lambda_prefill, self.lambda_decode);
        let outcome = rco::run(&objective, self.space.bytes(), budget, unit, self.rco, self.seed, self.restart)?;
        let mut choice = outcome.choice;
        let mut check = self.feasibility.check(&self.space.allocation(&choice))?;
        let log_p: Vec<Vec<f64>> = outcome
            .p
            .iter()
            .map(|row| row.iter().map(|p| p.max(TINY).ln()).collect())
            .collect();
        let mut limit = self.space.bytes_of(&choice)?.min(budget);
        for _ in 0..FIT_REPAIRS {
            if check.fits {
                break;
            }
            limit -= (check.device_bytes - self.feasibility.free_bytes()).max(unit);
            if limit < minimum {
                return Ok(None);
            }
            choice = knapsack::solve(&log_p, self.space.bytes(), limit, unit)?;
            check = self.feasibility.check(&self.space.allocation(&choice))?;
        }
        if !check.fits {
            return Ok(None);
        }
        let allocation = self.space.allocation(&choice);
        let mut hard = vec![vec![0.0; self.space.options().len()]; choice.len()];
        for (row, picked) in hard.iter_mut().zip(&choice) {
            row[*picked] = 1.0;
        }
        Ok(Some(Candidate {
            origin: format!(
                "rco:prefill={},decode={},budget={}",
                self.lambda_prefill, self.lambda_decode, self.fraction
            ),
            predicted: self.ensemble.mean(&allocation)?,
            spread: self.ensemble.spread(&allocation)?,
            value: objective.value(&hard)?,
            cell: Some(self.lambda_index),
            allocation,
        }))
    }
}

impl Search<'_> {
    /// The sweep's lambda cells: every pair (prefill, decode) of the lambda list.
    pub(super) fn lambda_cells(&self) -> usize {
        self.settings.lambdas.len() * self.settings.lambdas.len()
    }

    /// The ensemble fitted to `data` for `round`: each member's fit is a task of its own, joined in member order.
    fn fit<F: Fan>(&self, fan: &F, data: Arc<Vec<Measured>>, seed: i64, round: u32) -> Result<Ensemble, SearchFailure> {
        let fits = Ensemble::member_fits(
            data.len(),
            self.settings.members,
            seed.wrapping_mul(ENSEMBLE_STRIDE).wrapping_add(i64::from(round)),
        )?;
        let tasks: Vec<Box<dyn FnOnce() -> Result<FidelityModel, MixedError> + Send>> = fits
            .into_iter()
            .map(|fit| {
                let (data, priors) = (Arc::clone(&data), Arc::clone(&self.priors));
                let (ridge, bands) = (self.settings.ridge, self.settings.bands);
                Box::new(move || fit.fit(&data, ridge, &priors, bands)) as Box<dyn FnOnce() -> Result<FidelityModel, MixedError> + Send>
            })
            .collect();
        let members = fan.map(tasks)?.into_iter().collect::<Result<Vec<_>, MixedError>>()?;
        Ok(Ensemble::from_members(members)?)
    }

    /// The round's proposals from `data`: the fitted ensemble, and one RCO run per (weights, budget, restart) cell,
    /// each on the fan, their results joined in cell order and de-duplicated.
    pub(super) fn propose<F: Fan>(
        &self,
        fan: &F,
        feasibility: &Arc<dyn Feasibility>,
        data: &[Measured],
        round: u32,
        budget: i64,
        seed: i64,
    ) -> Result<Proposals, SearchFailure> {
        let ensemble = Arc::new(self.fit(fan, Arc::new(data.to_vec()), seed, round)?);
        let (lambdas, fractions) = (&self.settings.lambdas, &self.settings.budget_fractions);
        let mut tasks: Vec<Box<dyn FnOnce() -> Result<Option<Candidate>, MixedError> + Send>> = Vec::new();
        for lambda_index in 0..self.lambda_cells() {
            for (fraction_index, fraction) in fractions.iter().enumerate() {
                for restart in 0..self.settings.rco.restarts {
                    let cell = Cell {
                        space: Arc::clone(&self.space),
                        feasibility: Arc::clone(feasibility),
                        ensemble: Arc::clone(&ensemble),
                        rco: self.settings.rco,
                        knapsack_steps: self.settings.knapsack_steps,
                        ceiling: budget,
                        lambda_prefill: lambdas[lambda_index / lambdas.len()],
                        lambda_decode: lambdas[lambda_index % lambdas.len()],
                        lambda_index,
                        fraction: *fraction,
                        restart,
                        seed: seed
                            .wrapping_add(ROUND_STRIDE.wrapping_mul(i64::from(round)))
                            .wrapping_add(
                                CELL_STRIDE.wrapping_mul((lambda_index * fractions.len() + fraction_index) as i64),
                            ),
                    };
                    tasks.push(Box::new(move || cell.run()));
                }
            }
        }
        let mut seen: HashSet<Allocation> = HashSet::new();
        let mut candidates = Vec::new();
        for found in fan.map(tasks)? {
            if let Some(candidate) = found? {
                if seen.insert(candidate.allocation.clone()) {
                    candidates.push(candidate);
                }
            }
        }
        Ok(Proposals { ensemble, candidates })
    }
}
