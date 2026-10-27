//! Teacher-forced decoding: the model is fed given tokens and each step's logits row is handed to a sink.
//!
//! The prefix is prefilled in chunks, then the forced tokens are decoded one row at a time: step `k` scores the
//! position after the prefix and the first `k` forced tokens, so its logits row is the model's prediction of forced
//! token `k`. The rows are the BF16 rows of the head, copied to pinned memory after each step; the steps run one at a
//! time, since the workspace's logits are the next step's to overwrite.

use super::request::GenerationOutcome;

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
