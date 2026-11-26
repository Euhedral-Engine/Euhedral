//! A round's batch: the best the ensemble predicts for each weight in turn, then the candidates and moves it is least
//! sure of. Nothing measured, nothing twice.

use std::collections::HashSet;

use super::super::allocation::Allocation;
use super::super::ensemble::Ensemble;
use super::super::error::MixedError;
use super::super::feasibility::Feasibility;
use super::super::search_state::Measured;
use super::candidate::{Candidate, front_of};
use super::proposing::Proposals;
use super::seeding::below;
use super::Search;
use crate::java_random::Random;

/// Neighbours tried by one move.
const MOVES_TRIED: usize = 24;
/// The multiplier of the seed in a round's choice.
const CHOICE_STRIDE: i64 = 17;

/// What the choice has so far, and where its exploring picks come from.
struct Choosing {
    taken: HashSet<Allocation>,
    chosen: Vec<Candidate>,
    by_spread: Vec<Candidate>,
    spread_at: usize,
    anchors: Vec<Measured>,
    random: Random,
    exploiting: usize,
    exploring: usize,
    need: usize,
}

impl Choosing {
    /// The next candidate by spread that is not taken, labelled as an exploring pick; none when there is none.
    fn by_spread(&mut self) -> Option<Candidate> {
        while self.spread_at < self.by_spread.len() && self.taken.contains(&self.by_spread[self.spread_at].allocation) {
            self.spread_at += 1;
        }
        let pick = self.by_spread.get(self.spread_at)?.relabel("explore");
        self.spread_at += 1;
        Some(pick)
    }

    fn add(&mut self, pick: Candidate) {
        self.taken.insert(pick.allocation.clone());
        self.chosen.push(pick);
    }
}

/// The front, or the feasible measured candidates if there is none yet, or all of them.
fn anchors(data: &[Measured]) -> Vec<Measured> {
    let front = front_of(data);
    if !front.is_empty() {
        return front;
    }
    let fitting: Vec<Measured> = data.iter().filter(|each| each.fits).cloned().collect();
    if fitting.is_empty() { data.to_vec() } else { fitting }
}

impl Search<'_> {
    /// The `need` candidates of round `round` from `proposals`, never one in `data`. Exploring picks are `share` of the
    /// batch, rounded half up.
    pub(super) fn choose(
        &self,
        proposals: &Proposals,
        feasibility: &dyn Feasibility,
        data: &[Measured],
        share: f64,
        need: usize,
        round: u32,
        seed: i64,
    ) -> Result<Vec<Candidate>, MixedError> {
        let mut choosing = self.exploit(proposals, data, share, need, round, seed);
        // Exploring: alternately the proposal the ensemble is least sure of and a move; then, short of candidates (a
        // small space, or most of it measured), moves around the front for the rest.
        let mut e = 0;
        while choosing.chosen.len() < choosing.exploiting + choosing.exploring {
            let mut pick = if e % 2 == 0 { choosing.by_spread() } else { None };
            if pick.is_none() {
                pick = self.step_aside(&proposals.ensemble, feasibility, &mut choosing)?;
            }
            if pick.is_none() && e % 2 == 1 {
                pick = choosing.by_spread();
            }
            let Some(pick) = pick else { break };
            choosing.add(pick);
            e += 1;
        }
        while choosing.chosen.len() < choosing.need {
            let Some(pick) = self.step_aside(&proposals.ensemble, feasibility, &mut choosing)? else {
                break;
            };
            choosing.add(pick);
        }
        Ok(choosing.chosen)
    }

    /// The exploiting picks, and the ordering the exploring ones draw from.
    fn exploit(
        &self,
        proposals: &Proposals,
        data: &[Measured],
        share: f64,
        need: usize,
        round: u32,
        seed: i64,
    ) -> Choosing {
        let mut taken: HashSet<Allocation> = data.iter().map(|each| each.allocation.clone()).collect();
        let exploring = ((share * need as f64 + 0.5).floor() as usize).min(need);
        let exploiting = need - exploring;
        let mut by_lambda: Vec<Vec<&Candidate>> = vec![Vec::new(); self.lambda_cells()];
        for candidate in &proposals.candidates {
            if let Some(cell) = candidate.cell {
                by_lambda[cell].push(candidate);
            }
        }
        for list in &mut by_lambda {
            list.sort_by(|a, b| a.value.total_cmp(&b.value));
        }
        // Round-robin over the cells, each giving its best not yet taken.
        let mut at = vec![0; by_lambda.len()];
        let mut chosen = Vec::new();
        let mut progressed = true;
        while chosen.len() < exploiting && progressed {
            progressed = false;
            for (l, list) in by_lambda.iter().enumerate() {
                if chosen.len() >= exploiting {
                    break;
                }
                while at[l] < list.len() && taken.contains(&list[at[l]].allocation) {
                    at[l] += 1;
                }
                if at[l] == list.len() {
                    continue;
                }
                let pick = list[at[l]];
                at[l] += 1;
                taken.insert(pick.allocation.clone());
                chosen.push(pick.relabel("exploit"));
                progressed = true;
            }
        }
        let mut by_spread = proposals.candidates.clone();
        by_spread.sort_by(|a, b| b.spread.total_cmp(&a.spread));
        Choosing {
            taken,
            chosen,
            by_spread,
            spread_at: 0,
            anchors: anchors(data),
            random: Random::new(seed.wrapping_mul(CHOICE_STRIDE).wrapping_add(i64::from(round))),
            exploiting,
            exploring,
            need,
        }
    }

    /// The feasible single or double move around a random anchor with the largest ensemble spread, among
    /// `MOVES_TRIED` tried (the first of equals); none if none is new.
    fn step_aside(
        &self,
        ensemble: &Ensemble,
        feasibility: &dyn Feasibility,
        choosing: &mut Choosing,
    ) -> Result<Option<Candidate>, MixedError> {
        if choosing.anchors.is_empty() {
            return Ok(None);
        }
        let (n, k) = (self.space.objects().len(), self.space.options().len());
        let mut moves = Vec::with_capacity(MOVES_TRIED);
        for _ in 0..MOVES_TRIED {
            let anchor = below(&mut choosing.random, choosing.anchors.len());
            let mut choice = self.space.choice(&choosing.anchors[anchor].allocation)?;
            let changes = 1 + below(&mut choosing.random, 2);
            for _ in 0..changes {
                let i = below(&mut choosing.random, n);
                let other = below(&mut choosing.random, k - 1);
                choice[i] = if other >= choice[i] { other + 1 } else { other };
            }
            moves.push(choice);
        }
        let mut best: Option<Candidate> = None;
        for choice in moves {
            let allocation = self.space.allocation(&choice);
            if choosing.taken.contains(&allocation) || !feasibility.check(&allocation)?.fits {
                continue;
            }
            let spread = ensemble.spread(&allocation)?;
            if best.as_ref().is_none_or(|held| spread > held.spread) {
                best = Some(Candidate {
                    predicted: ensemble.mean(&allocation)?,
                    spread,
                    value: 0.0,
                    cell: None,
                    origin: "explore:move".to_owned(),
                    allocation,
                });
            }
        }
        Ok(best)
    }
}
