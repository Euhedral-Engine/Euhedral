//! The search for the allocations on the quality and speed front of a set of calibrated sources, each candidate scored
//! against a reference.
//!
//! A round fits an [`Ensemble`] of fidelity models to what was measured, runs RCO ([`rco`]) from several starting
//! points over a grid of cost weights and budgets, and picks a batch of candidates: most of them the best the ensemble
//! predicts, the rest where it is least sure. Each candidate is assembled, checked against the residency plan at the
//! target context and scored; the ensemble is refitted for the next round. The search stops after several rounds
//! without a new point on the [`pareto`] front. The optimization is Riemannian Constrained Optimization (Helcig and
//! Alistarh, arXiv:2605.00649).
//!
//! The search is a plain sequence of steps over the narrow traits of [`ports`]: its [`Scoring`], its [`Assembly`], its
//! [`Ledger`] (the resume file, the out directory and the log) and a [`Fan`] that runs the independent host tasks (the
//! members' fits, the RCO cells) wherever the caller has workers. Nothing here opens a file, waits or owns a thread.
//!
//! Everything measured is kept in the ledger after each scored candidate, so a stopped search resumes without scoring
//! one twice. A round's batch is chosen from what earlier rounds measured only, so a search stopped in the middle of a
//! round and resumed measures the batch an unbroken search would have, and reaches the same front.
//!
//! A candidate that does not fit the device is not scored.

mod budget;
mod candidate;
mod choosing;
mod failure;
mod output;
mod ports;
mod proposing;
mod seeding;
mod settings;
#[cfg(test)]
mod tests;

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;

use sha2::{Digest, Sha256};

pub use self::failure::SearchFailure;
pub use self::ports::{Assembly, Fan, Inline, Ledger, Score, Scoring, World};
pub use self::settings::Settings;
use self::candidate::{Candidate, Prepared, front_of};
use self::output::render;
use super::cost_table::CostTable;
use super::ensemble::Ensemble;
use super::option::QuantOption;
use super::priors::Priors;
use super::search_state::{Measured, Progress};
use super::space::{Sources, Space};

/// How much worse a round's held-out error is than the last round's for the explore share to rise.
const WORSE: f64 = 1.1;
/// The explore share's cap, and how far it rises at once.
const EXPLORE_CAP: f64 = 0.6;
const EXPLORE_STEP: f64 = 0.1;

/// What a search ended with: everything measured, the front, the rounds run (not counting the seeds) and the bytes of
/// projections RCO was budgeted.
#[derive(Debug, Clone)]
pub struct Outcome {
    pub measured: Vec<Measured>,
    pub front: Vec<Measured>,
    pub rounds: u32,
    pub budget: i64,
}

/// The search over `sources`' options: what the objects can be, what each costs and what its error is expected to be.
pub struct Search<'a> {
    settings: Settings,
    sources: &'a dyn Sources,
    costs: &'a CostTable,
    space: Arc<Space>,
    priors: Arc<Priors>,
}

impl<'a> Search<'a> {
    /// A search over the options that have both a source (`options`) and a prior. Refuses settings outside their
    /// limits, a base the priors have no error for, and fewer than two options.
    pub fn new(
        settings: Settings,
        sources: &'a dyn Sources,
        options: &[QuantOption],
        priors: &Priors,
        costs: &'a CostTable,
    ) -> Result<Search<'a>, SearchFailure> {
        settings.validate()?;
        let known = priors.options();
        let mut options: Vec<QuantOption> = options.iter().copied().filter(|option| known.contains(option)).collect();
        options.sort();
        options.dedup();
        if !options.contains(&sources.base()) {
            return Err(SearchFailure::refused(format!(
                "the priors have no error for the base {} of every object",
                sources.base().key()
            )));
        }
        if options.len() < 2 {
            return Err(SearchFailure::refused(
                "a search needs two options with a source and a prior",
            ));
        }
        let space = Space::new(sources, &options, costs, settings.verify_rows)?;
        let priors = priors.restrict(space.objects(), &options)?;
        Ok(Search {
            settings,
            sources,
            costs,
            space: Arc::new(space),
            priors: Arc::new(priors),
        })
    }

    /// The identity of this search for the resume file: what the measurements depend on.
    pub fn fingerprint(&self) -> String {
        let text = format!(
            "{}\nbase={}\nbands={} verifyRows={} context={} objects={}",
            self.settings.identity,
            self.sources.base().key(),
            self.settings.bands,
            self.settings.verify_rows,
            self.settings.context,
            self.space.objects().len()
        );
        Sha256::digest(text.as_bytes()).iter().map(|byte| format!("{byte:02x}")).collect()
    }

    /// Runs the search, resuming from the ledger if it holds one.
    pub fn run<F: Fan>(&self, mut world: World<'_, F>) -> Result<Outcome, SearchFailure> {
        let feasibility = Arc::clone(&world.feasibility);
        let budget = self.derive(&*feasibility)?;
        let seed = world.ledger.seed();
        world.ledger.note(&format!(
            "search: {} objects, {} options, projections {}..{} MiB, budget {} MiB, {} MiB free, {} measured",
            self.space.objects().len(),
            self.space.options().len(),
            self.space.minimum_bytes()? >> 20,
            self.space.maximum_bytes()? >> 20,
            budget >> 20,
            feasibility.free_bytes() >> 20,
            world.ledger.measured().len()
        ));
        let mut progress = world.ledger.progress();
        if progress.explore.is_nan() {
            progress.explore = self.settings.explore;
            world.ledger.set_progress(progress)?;
        }
        if progress.round == 0 {
            if world.ledger.measured().is_empty() && !world.imported.is_empty() {
                self.import(&mut world)?;
            }
            let seeds = self.seed_candidates(&*feasibility, budget, seed)?;
            self.measure(&mut world, seeds, 0)?;
            progress.round = 1;
            world.ledger.set_progress(progress)?;
        }
        let mut rounds = 0;
        loop {
            let progress = world.ledger.progress();
            if progress.round > self.settings.max_rounds || progress.stale >= self.settings.patience {
                break;
            }
            let round = progress.round;
            // Only what earlier rounds measured: a resumed round proposes what the unbroken one would have.
            let data: Vec<Measured> = world.ledger.measured().iter().filter(|each| each.round < round).cloned().collect();
            if data.is_empty() {
                return Err(SearchFailure::NothingMeasured);
            }
            let proposals = self.propose(world.fan, &feasibility, &data, round, budget, seed)?;
            let chosen = self.choose(
                &proposals,
                &*feasibility,
                &data,
                progress.explore,
                self.settings.batch,
                round,
                seed,
            )?;
            if chosen.is_empty() {
                world.ledger.note(&format!("round {round}: no candidate that was not measured"));
            }
            self.measure(&mut world, chosen, round)?;
            self.finish_round(&mut world, round, &proposals.ensemble, budget)?;
            rounds += 1;
        }
        self.write_outputs(&mut world, budget)?;
        let measured = world.ledger.measured().to_vec();
        let front = front_of(&measured);
        Ok(Outcome {
            measured,
            front,
            rounds,
            budget,
        })
    }

    /// `candidate` assembled and checked: its bytes, the planner's verdict and the cost table's two predictions.
    fn prepare<F: Fan>(&self, world: &World<'_, F>, candidate: Candidate) -> Result<Prepared, SearchFailure> {
        let allocation = &candidate.allocation;
        let fits = world.feasibility.check(allocation)?.fits;
        let bytes = world.assembly.resident_bytes(allocation)?;
        let prefill = self.costs.prefill_micros_per_token(self.sources, allocation)?;
        let decode = self.costs.verify_micros(self.sources, allocation, self.settings.verify_rows)?;
        Ok(Prepared {
            candidate,
            bytes,
            prefill,
            decode,
            fits,
        })
    }

    /// Scores the candidates of `batch` not measured yet, each recorded as soon as it is scored. One that does not fit
    /// the device is left out.
    fn measure<F: Fan>(&self, world: &mut World<'_, F>, batch: Vec<Candidate>, round: u32) -> Result<(), SearchFailure> {
        let mut listed = HashSet::new();
        let todo: Vec<Candidate> = batch
            .into_iter()
            .filter(|candidate| !world.ledger.contains(&candidate.allocation) && listed.insert(candidate.allocation.clone()))
            .collect();
        let total = todo.len();
        for (k, candidate) in todo.into_iter().enumerate() {
            let ready = self.prepare(world, candidate)?;
            if !ready.fits {
                world.ledger.note(&format!(
                    "round {round}, {}/{total}: {} does not fit the device, not scored",
                    k + 1,
                    ready.candidate.origin
                ));
                continue;
            }
            let predicted = ready.candidate.predicted;
            let score = world.scoring.score(&ready.candidate.allocation)?;
            let measured = Measured {
                allocation: ready.candidate.allocation,
                kl: score.mean_kl,
                kl_se: score.kl_se,
                bytes: ready.bytes,
                prefill: ready.prefill,
                decode: ready.decode,
                fits: ready.fits,
                round,
                origin: ready.candidate.origin,
            };
            world.ledger.add(measured.clone())?;
            world.ledger.note(&format!(
                "round {round}, {}/{total}: {} kl {:.5} +- {:.5}, {} MiB, prefill {:.2} us/token, verify {:.1} us, predicted {:.5}",
                k + 1,
                measured.origin,
                measured.kl,
                measured.kl_se,
                measured.bytes >> 20,
                measured.prefill,
                measured.decode,
                predicted
            ));
        }
        Ok(())
    }

    /// Records the imported candidates: each one's measured KL, with size, costs and fit prepared for this search.
    fn import<F: Fan>(&self, world: &mut World<'_, F>) -> Result<(), SearchFailure> {
        let imported = std::mem::take(&mut world.imported);
        let mut ready = Vec::with_capacity(imported.len());
        for each in &imported {
            let candidate = Candidate::given(self.space.normalize(&each.allocation)?, format!("import:{}", each.origin));
            ready.push(self.prepare(world, candidate)?);
        }
        let mut fitting = 0;
        for (prepared, each) in ready.into_iter().zip(&imported) {
            let fits = prepared.fits;
            let added = world.ledger.add(Measured {
                allocation: prepared.candidate.allocation,
                kl: each.kl,
                kl_se: each.kl_se,
                bytes: prepared.bytes,
                prefill: prepared.prefill,
                decode: prepared.decode,
                fits,
                round: 0,
                origin: prepared.candidate.origin,
            })?;
            if added && fits {
                fitting += 1;
            }
        }
        world.ledger.note(&format!("imported {} measured candidates, {fitting} fit", imported.len()));
        Ok(())
    }

    /// The round's bookkeeping, written in one record: the held-out error of the ensemble that proposed it on what it
    /// proposed, the explore share, whether the front gained a point, the round counter.
    fn finish_round<F: Fan>(
        &self,
        world: &mut World<'_, F>,
        round: u32,
        proposer: &Ensemble,
        budget: i64,
    ) -> Result<(), SearchFailure> {
        let mine: Vec<Measured> = world.ledger.measured().iter().filter(|each| each.round == round).cloned().collect();
        let held_out = proposer.held_out_error(&mine)?;
        let before = world.ledger.progress();
        let mut progress = Progress {
            round: round + 1,
            ..before
        };
        if !before.held_out.is_nan() && held_out > WORSE * before.held_out {
            progress.explore = EXPLORE_CAP.min(before.explore + EXPLORE_STEP);
        }
        progress.held_out = held_out;
        let front = front_of(world.ledger.measured());
        let grew = front.iter().any(|each| each.round == round);
        progress.stale = if grew { 0 } else { before.stale.saturating_add(1) };
        world.ledger.set_progress(progress)?;
        world.ledger.note(&format!(
            "round {round}: {} measured, held-out error {held_out:.4}, explore {:.1}, front {} points{}",
            mine.len(),
            progress.explore,
            front.len(),
            if grew { "" } else { " (no new point)" }
        ));
        self.write_outputs(world, budget)
    }

    /// `front.json`, the front's allocations and the text report.
    fn write_outputs<F: Fan>(&self, world: &mut World<'_, F>, budget: i64) -> Result<(), SearchFailure> {
        let files = render(
            world.ledger.measured(),
            world.ledger.progress(),
            budget,
            world.feasibility.free_bytes(),
        )?;
        world.ledger.publish(&files)
    }
}

impl fmt::Debug for Search<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Search").field("settings", &self.settings).finish_non_exhaustive()
    }
}
