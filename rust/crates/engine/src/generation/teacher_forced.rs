//! Teacher-forced decoding: the model is fed given tokens and each step's logits row is handed to a sink.
//!
//! The prefix is prefilled in chunks, then the forced tokens are decoded one row at a time: step `k` scores the
//! position after the prefix and the first `k` forced tokens, so its logits row is the model's prediction of forced
//! token `k`. The rows are the BF16 rows of the head, which the quantum itself copies to the row readback the run
//! holds ([`CopyRows`](crate::stages::sampling::CopyRows)): the workspace's logits belong to the next sequence's quantum
//! once this one retires, so no copy is queued after that. The steps run one at a time, since the run's readback is
//! the next step's to overwrite.

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
        let readbacks = &self.sequence.row_readbacks;
        forced_refusal(
            rows,
            sink.wants_hidden(),
            !readbacks.is_exhausted(),
            !readbacks.hidden_is_exhausted(),
            readbacks.rows(),
        )
        .map(AdmissionError::Engine)
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

    /// A forced step of one row retired: its logits row (and its hidden row) are in the readback the quantum copied them
    /// to, so they go to the sink now, and the readback goes back to the sequence.
    pub(super) fn forced_step_retired(&mut self, flying: &mut Flying) {
        let Some(lease) = flying.row.take() else {
            return self.fail("a forced step retired with no row readback".to_string());
        };
        let Some(run) = self.run.as_mut() else { return };
        let step = run.generated;
        if let Mode::Forced { forced, sink, .. } = &mut run.mode
            && let Some(next) = forced.get(step)
        {
            let readback = lease.readback();
            if sink.wants_hidden() {
                let Some(hidden) = readback.hidden() else {
                    return self.fail("a forced step's readback holds no hidden row".to_string());
                };
                sink.hidden(step, hidden);
            }
            let Some(row) = readback.row(0) else {
                return self.fail("a forced step's readback holds no logits row".to_string());
            };
            sink.row(step, *next, row);
        }
        run.generated += 1;
        if run.generated >= run.wanted {
            run.stop(FinishReason::Length);
        }
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

/// Why a teacher-forced run of `rows` rows per step cannot start, given whether a plain readback and one that holds the
/// hidden state are free to lease and how many rows the plain ones hold. A one-row step and its hidden row are copied
/// by the quantum to a readback it leases, so a sequence with none free refuses at admission, not mid-run.
fn forced_refusal(rows: usize, wants_hidden: bool, plain_free: bool, hidden_free: bool, held: usize) -> Option<String> {
    if rows <= 1 {
        let free = if wants_hidden { hidden_free } else { plain_free };
        return (!free).then(|| "the sequence has no row readback to copy the rows to".to_string());
    }
    if wants_hidden {
        return Some("a teacher-forced run that takes the hidden state decodes one row at a time".to_string());
    }
    (rows > held).then(|| {
        format!("the sequences verify at most {held} rows: {rows} were asked (batching.forced-rows sizes them)")
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_one_row_run_needs_the_readback_its_sink_takes() {
        assert_eq!(forced_refusal(1, false, true, false, 4), None);
        assert_eq!(forced_refusal(1, true, false, true, 4), None);
        assert!(forced_refusal(1, false, false, true, 4).is_some());
        assert!(forced_refusal(1, true, true, false, 4).is_some());
        assert!(
            forced_refusal(0, false, false, false, 0).is_some(),
            "a sequence with none"
        );
    }

    #[test]
    fn a_multi_row_run_is_refused_with_the_hidden_state_or_past_the_rows_held() {
        assert_eq!(forced_refusal(4, false, true, true, 4), None);
        assert!(forced_refusal(4, true, true, true, 4).unwrap().contains("hidden state"));
        let wide = forced_refusal(5, false, true, true, 4).unwrap();
        assert!(wide.contains('4') && wide.contains('5'), "{wide}");
        assert!(forced_refusal(2, false, false, false, 0).is_some());
    }
}
