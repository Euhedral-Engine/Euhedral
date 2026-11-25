//! Teacher-forced decoding: the model is fed given tokens and each step's logits row is handed to a sink.
//!
//! The prefix is prefilled in chunks, then the forced tokens are decoded one row at a time: step `k` scores the
//! position after the prefix and the first `k` forced tokens, so its logits row is the model's prediction of forced
//! token `k`. The rows are the BF16 rows of the head, copied to pinned memory after each step; the steps run one at a
//! time, since the workspace's logits are the next step's to overwrite.

use super::request::{AdmissionError, FinishReason, GenerationOutcome};
use super::source::{Admission, Flying, Mode, OVERLAP_MAX_START_POSITION, Owner, Role};
use crate::stages::QuantumKind;
use crate::stages::attention::KvSource;

/// Receives the logits rows of a teacher-forced run. Every method runs on a worker of the engine's lattice, one call
/// at a time and in order, and must not wait.
pub trait LogitsSink: Send {
    /// Whether the sink wants the final hidden state (before the final norm, BF16) of each step's last row; it is
    /// copied beside the logits row and handed to [`LogitsSink::hidden`].
    fn wants_hidden(&self) -> bool {
        false
    }

    /// The final hidden state of step `step`, `hidden_size` BF16 values in little-endian bytes, delivered just before
    /// that step's [`LogitsSink::row`] when [`LogitsSink::wants_hidden`]. Valid for the call only.
    fn hidden(&mut self, step: usize, hidden: &[u8]) {
        let _ = (step, hidden);
    }

    /// The logits row of step `step`, `vocabulary` BF16 values in little-endian bytes, which predicts `forced`, the
    /// forced token that follows. The bytes are valid for the call only.
    fn row(&mut self, step: usize, forced: u32, logits: &[u8]);

    /// The run ended: after the last row, or at the failure the outcome names.
    fn finished(&mut self, outcome: GenerationOutcome);
}

impl Owner {
    /// Why a run of `rows` rows per step cannot start on this sequence, if it cannot.
    pub(super) fn forced_rows_refusal(&self, rows: usize, sink: &dyn LogitsSink) -> Option<AdmissionError> {
        if rows <= 1 {
            return None;
        }
        if sink.wants_hidden() {
            return Some(AdmissionError::Engine(
                "a teacher-forced run that takes the hidden state decodes one row at a time".to_string(),
            ));
        }
        let held = self.sequence.row_readbacks.rows();
        (rows > held).then(|| {
            AdmissionError::Engine(format!(
                "the sequences verify at most {held} rows: {rows} were asked (batching.forced-rows sizes them)"
            ))
        })
    }

    /// Admits the verification of the next `rows` forced tokens: their logits rows go to the sink in order, and the
    /// workspace's passes batch it with the verifications of the other sequences.
    pub(super) fn admit_forced_rows(&mut self, rows: usize) -> Result<(), String> {
        let Some(run) = self.run.as_mut() else { return Ok(()) };
        let start = run.position;
        let Mode::Forced { forced, .. } = &run.mode else {
            return Err("a forced step in a request that is not teacher-forced".to_string());
        };
        let tokens = forced[run.decodes..run.decodes + rows].to_vec();
        let admission = Admission {
            role: Role::Verify,
            kind: QuantumKind::Verify,
            rows,
            start,
            source: KvSource::Base,
            pdl: start < OVERLAP_MAX_START_POSITION,
            logits: rows,
            replay: Vec::new(),
            last_chunk: false,
            more: false,
        };
        self.admit_quantum(admission, move |quantum| quantum.with_tokens(tokens))?;
        let Some(run) = self.run.as_mut() else { return Ok(()) };
        run.position += rows as u64;
        run.decodes += rows;
        Ok(())
    }

    /// The verification of forced rows retired: its rows join the cache and their logits go to the sink.
    pub(super) fn forced_rows_retired(&mut self, flying: &mut Flying) {
        let Some(lease) = flying.row.take() else {
            return self.fail("a forced verification retired with no row readback".to_string());
        };
        if let Err(error) = self.sequence.kv.commit(flying.rows as u64) {
            return self.fail(format!("committing the quantum's rows: {error}"));
        }
        let Some(run) = self.run.as_mut() else { return };
        if let Mode::Forced { forced, sink, .. } = &mut run.mode {
            for row in 0..flying.rows {
                let step = run.generated + row;
                let (Some(bytes), Some(next)) = (lease.readback().row(row), forced.get(step)) else {
                    return self.fail("a forced row is outside its readback".to_string());
                };
                sink.row(step, *next, bytes);
            }
        }
        run.generated += flying.rows;
        if run.generated >= run.wanted {
            run.stop(FinishReason::Length);
        }
    }
}
