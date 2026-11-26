//! One chunk of a scoring run: the slots its positions' values land in, and the verdict when the last holder lets go.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};

use euhedral_inference_engine::generation::{FinishReason, GenerationOutcome};

use super::summary::FIELDS;

/// What a chunk reports when it ends: its values, `steps * FIELDS` of them, or why not. It runs in whichever frame
/// let go last.
pub(crate) type Ended = Arc<dyn Fn(Result<Vec<f64>, String>) + Send + Sync>;

/// The shared state of one chunk. The engine's sink, the reference reads and the scorer's groups each hold it while
/// they have work for it; when the last lets go, its `Drop` reports.
pub(crate) struct ChunkScore {
    steps: usize,
    slots: Vec<AtomicU64>,
    /// Which positions stored their values: a row that never came, or a group that was dropped, leaves its step unset.
    computed: Vec<AtomicBool>,
    fault: OnceLock<String>,
    outcome: OnceLock<GenerationOutcome>,
    ended: Ended,
}

impl ChunkScore {
    /// A chunk of `steps` positions that reports to `ended`.
    pub(crate) fn new(steps: usize, ended: Ended) -> Arc<ChunkScore> {
        Arc::new(ChunkScore {
            steps,
            slots: (0..steps * FIELDS).map(|_| AtomicU64::new(0)).collect(),
            computed: (0..steps).map(|_| AtomicBool::new(false)).collect(),
            fault: OnceLock::new(),
            outcome: OnceLock::new(),
            ended,
        })
    }

    /// The positions of the chunk.
    pub(crate) fn steps(&self) -> usize {
        self.steps
    }

    /// Records the first failure; later ones are dropped.
    pub(crate) fn fail(&self, why: String) {
        let _ = self.fault.set(why);
    }

    /// Records how the engine's run ended.
    pub(crate) fn finished(&self, outcome: GenerationOutcome) {
        let _ = self.outcome.set(outcome);
    }

    /// Stores the values of `step`.
    pub(crate) fn store(&self, step: usize, values: &[f64]) {
        let (Some(slots), Some(computed)) = (
            self.slots.get(step * FIELDS..(step + 1) * FIELDS),
            self.computed.get(step),
        ) else {
            self.fail(format!("values for step {step} of a chunk of {}", self.steps));
            return;
        };
        for (slot, value) in slots.iter().zip(values) {
            slot.store(value.to_bits(), Ordering::Relaxed);
        }
        computed.store(true, Ordering::Release);
    }

    fn verdict(&self) -> Result<Vec<f64>, String> {
        if let Some(why) = self.fault.get() {
            return Err(why.clone());
        }
        let Some(outcome) = self.outcome.get() else {
            return Err("the engine ended the chunk without an outcome".to_owned());
        };
        if outcome.reason != FinishReason::Length || outcome.generated != self.steps {
            return Err(format!(
                "the run ended with {:?} after {} of {} steps",
                outcome.reason, outcome.generated, self.steps
            ));
        }
        if let Some(step) = self.computed.iter().position(|done| !done.load(Ordering::Acquire)) {
            return Err(format!(
                "step {step} was not scored: its row never arrived or its group was lost"
            ));
        }
        Ok(self
            .slots
            .iter()
            .map(|slot| f64::from_bits(slot.load(Ordering::Relaxed)))
            .collect())
    }
}

impl Drop for ChunkScore {
    fn drop(&mut self) {
        (self.ended)(self.verdict());
    }
}

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::disallowed_types,
    reason = "a test: a failed unwrap is the failure, and the reports are read after the chunk ends"
)]
mod tests {
    use std::sync::Mutex;
    use std::time::Duration;

    use super::*;

    type Reports = Arc<Mutex<Vec<Result<Vec<f64>, String>>>>;

    fn fresh(steps: usize) -> (Arc<ChunkScore>, Reports) {
        let reports: Reports = Arc::default();
        let sink = Arc::clone(&reports);
        let ended: Ended = Arc::new(move |verdict| sink.lock().unwrap().push(verdict));
        (ChunkScore::new(steps, ended), reports)
    }

    fn outcome(reason: FinishReason, generated: usize) -> GenerationOutcome {
        GenerationOutcome {
            reason,
            prompt_tokens: 0,
            generated,
            elapsed: Duration::ZERO,
            cached_prompt_tokens: 0,
        }
    }

    fn only(reports: &Reports) -> Result<Vec<f64>, String> {
        let reports = reports.lock().unwrap();
        assert_eq!(reports.len(), 1, "a chunk reports exactly once");
        reports[0].clone()
    }

    #[test]
    fn a_chunk_reports_once_when_the_last_holder_lets_go_and_not_before() {
        let (chunk, reports) = fresh(1);
        let other = Arc::clone(&chunk);
        chunk.finished(outcome(FinishReason::Length, 1));
        chunk.store(0, &[1.0; FIELDS]);
        drop(chunk);
        assert!(reports.lock().unwrap().is_empty(), "a holder is still working");
        drop(other);
        assert_eq!(only(&reports), Ok(vec![1.0; FIELDS]));
    }

    #[test]
    fn a_fault_wins_over_a_complete_run_and_the_first_fault_is_kept() {
        let (chunk, reports) = fresh(1);
        chunk.finished(outcome(FinishReason::Length, 1));
        chunk.store(0, &[1.0; FIELDS]);
        chunk.fail("first".to_owned());
        chunk.fail("second".to_owned());
        drop(chunk);
        assert_eq!(only(&reports), Err("first".to_owned()));
    }

    #[test]
    fn a_run_that_did_not_end_by_length_or_fell_short_is_not_a_result() {
        for (stopped, generated) in [(true, 2), (false, 1)] {
            let reason = if stopped {
                FinishReason::Stop
            } else {
                FinishReason::Length
            };
            let (chunk, reports) = fresh(2);
            chunk.finished(outcome(reason, generated));
            chunk.store(0, &[1.0; FIELDS]);
            chunk.store(1, &[1.0; FIELDS]);
            drop(chunk);
            assert!(only(&reports).is_err(), "{generated} steps, stopped: {stopped}");
        }
        let (chunk, reports) = fresh(1);
        chunk.store(0, &[1.0; FIELDS]);
        drop(chunk);
        assert!(only(&reports).is_err(), "no outcome from the engine");
    }

    #[test]
    fn a_step_never_scored_names_itself_and_a_step_out_of_range_is_a_fault() {
        let (chunk, reports) = fresh(2);
        chunk.finished(outcome(FinishReason::Length, 2));
        chunk.store(1, &[2.0; FIELDS]);
        drop(chunk);
        let why = only(&reports).unwrap_err();
        assert!(why.contains("step 0"), "{why}");

        let (chunk, reports) = fresh(1);
        chunk.finished(outcome(FinishReason::Length, 1));
        chunk.store(0, &[1.0; FIELDS]);
        chunk.store(5, &[1.0; FIELDS]);
        drop(chunk);
        assert!(only(&reports).is_err(), "a value for a step the chunk does not have");
    }
}
