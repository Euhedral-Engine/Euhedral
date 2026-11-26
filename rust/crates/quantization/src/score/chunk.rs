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
