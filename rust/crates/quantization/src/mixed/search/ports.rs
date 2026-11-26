//! What a search is given to work with: its scorer, its assembly, its ledger and its host fan-out. Each is a narrow
//! trait so the search's own contract is proved with fakes; the command supplies the device, the files and the lake.

use std::fmt;
use std::sync::Arc;

use super::super::allocation::Allocation;
use super::super::feasibility::Feasibility;
use super::super::search_state::{Measured, Progress};
use super::failure::SearchFailure;
use crate::score::summary::{FIELDS, mean, standard_error};

/// The KL of one candidate against the references: the mean over every scored row and its standard error (over the
/// chunks' means when there are several chunks, as the scorer's summary does).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Score {
    pub mean_kl: f64,
    pub kl_se: f64,
}

impl Score {
    /// The pooled score of `scored`, a vector of `steps * FIELDS` values per chunk of every reference together. `None`
    /// when no row was scored or a chunk is not whole rows.
    pub fn of(scored: &[Vec<f64>]) -> Option<Score> {
        let (mut all, mut per_chunk) = (Vec::new(), Vec::with_capacity(scored.len()));
        for chunk in scored {
            if chunk.is_empty() || !chunk.len().is_multiple_of(FIELDS) {
                return None;
            }
            let kl: Vec<f64> = chunk.as_chunks::<FIELDS>().0.iter().map(|row| row[2]).collect();
            per_chunk.push(mean(&kl));
            all.extend(kl);
        }
        if all.is_empty() {
            return None;
        }
        Some(Score {
            mean_kl: mean(&all),
            kl_se: standard_error(&per_chunk, &all),
        })
    }
}

/// Scores assembled candidates, one at a time, against the reference.
///
/// A scoring that scores several at once adds a submitting method with this one as its default, and the search
/// starts the next candidate's host work while the previous ones are scored; today the search scores one and waits.
pub trait Scoring {
    /// The score of the composite `allocation` makes. A failure stops the search; what was recorded stays.
    fn score(&mut self, allocation: &Allocation) -> Result<Score, SearchFailure>;
}

/// What assembling a candidate gives the search: the bytes of the composite's tensors that stay resident.
pub trait Assembly {
    fn resident_bytes(&self, allocation: &Allocation) -> Result<i64, SearchFailure>;
}

/// The search's resume file, its out directory and its log: everything it writes and what it reads back on a resume.
pub trait Ledger {
    /// The seed of the run that started the search.
    fn seed(&self) -> i64;
    /// Every measured candidate, in the order they were measured.
    fn measured(&self) -> &[Measured];
    /// Whether `allocation` was measured.
    fn contains(&self, allocation: &Allocation) -> bool;
    /// How far the search has got.
    fn progress(&self) -> Progress;
    /// Records `candidate`, durably; `false` and nothing written when its allocation is already recorded.
    fn add(&mut self, candidate: Measured) -> Result<bool, SearchFailure>;
    /// Records `progress` in one write.
    fn set_progress(&mut self, progress: Progress) -> Result<(), SearchFailure>;
    /// Writes each file (a path below the out directory, its text), each whole or not at all.
    fn publish(&mut self, files: &[(String, String)]) -> Result<(), SearchFailure>;
    /// One line of the search's log.
    fn note(&mut self, line: &str);
}

/// Runs independent host tasks and joins them, in the order they were given, whichever order they finish in: the fits
/// of the ensemble's members and the cells of the RCO sweep. The command runs them as frames on the host lake; a
/// result must not depend on where a task ran.
pub trait Fan {
    fn map<T: Send + 'static>(&self, tasks: Vec<Box<dyn FnOnce() -> T + Send>>) -> Result<Vec<T>, SearchFailure>;
}

/// The fan that runs every task on the calling thread, in order.
#[derive(Debug, Clone, Copy, Default)]
pub struct Inline;

impl Fan for Inline {
    fn map<T: Send + 'static>(&self, tasks: Vec<Box<dyn FnOnce() -> T + Send>>) -> Result<Vec<T>, SearchFailure> {
        Ok(tasks.into_iter().map(|task| task()).collect())
    }
}

/// What the search is run with.
pub struct World<'w, F: Fan> {
    pub ledger: &'w mut dyn Ledger,
    pub assembly: &'w dyn Assembly,
    pub scoring: &'w mut dyn Scoring,
    pub feasibility: Arc<dyn Feasibility>,
    pub fan: &'w F,
    /// Candidates another search measured, recorded before a new search's seeds.
    pub imported: Vec<Measured>,
}

impl<F: Fan> fmt::Debug for World<'_, F> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("World")
            .field("imported", &self.imported.len())
            .finish_non_exhaustive()
    }
}
