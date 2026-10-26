//! The embedding lookup of a quantum's token ids (`Stages.Embed`), and the passing of tokens between the quanta of an
//! MTP step on the device.

use euhedral_graph::{Marker, Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{Submission, ops};

use super::{Launch, Quantum, eager, fatal};
use crate::formats::Q3Embedding;
use crate::model::WeightRef;
use crate::state::attention::DRAFT_TOKEN_SLOTS;
use crate::workspace::BufferSlot;

/// Looks up the quantum's token ids in the Q3 embedding table into `output`.
///
/// Rows that take their tokens from the step's tokens on the device (the quantum's
/// [`Quantum::reading_draft_tokens`]) copy them into the input record first, where the lookup reads its ids; a
/// quantum that [`Quantum::writing_current_token`] then copies its last token to slot 0 of the step's tokens, as the
/// step's current token.
#[derive(Clone, Debug)]
pub struct Embed {
    table: WeightRef,
    output: BufferSlot,
}

impl Embed {
    /// The lookup in `table` (`[vocabulary, hidden]`), writing the rows of `output`.
    pub fn new(table: WeightRef, output: BufferSlot) -> Self {
        Self { table, output }
    }
}

impl Launch for Embed {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    /// The upload of the quantum's input record ([`Quantum::prepare`]): every stage that follows reads the ids or the
    /// positions of it, through this stage's marker.
    fn waits(&self, quantum: &Quantum) -> Vec<Marker> {
        quantum.prepared.into_iter().collect()
    }

    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError> {
        if self.idle(quantum) {
            return Ok(());
        }
        if self.table.shape.get(1) != Some(&(quantum.config.hidden_size as u64)) {
            return Err(fatal(format!(
                "{} is {:?} wide but the model's hidden size is {}",
                self.table.name,
                self.table.shape.get(1),
                quantum.config.hidden_size
            )));
        }
        let rows = quantum.rows;
        let ids = quantum.slot(BufferSlot::TokenIds);
        let step_tokens = || {
            quantum
                .seq
                .device()
                .scratch
                .draft()
                .map(|draft| draft.tokens.ptr())
                .ok_or_else(|| fatal("the sequence has no step tokens"))
        };
        if let Some(slot) = quantum.draft_token_read {
            if (slot + rows) as u64 > DRAFT_TOKEN_SLOTS {
                return Err(fatal(format!(
                    "draft token slots {slot}..{} are out of {DRAFT_TOKEN_SLOTS}",
                    slot + rows
                )));
            }
            ops::copy_device_to_device(submission, ids, step_tokens()?.offset(4 * slot as u64), 4 * rows as u64)
                .map_err(fatal)?;
        }
        Q3Embedding
            .submit(submission, &self.table, ids, quantum.slot(self.output), rows)
            .map_err(fatal)?;
        if quantum.writes_current_token {
            ops::copy_device_to_device(submission, step_tokens()?, ids.offset(4 * (rows as u64 - 1)), 4)
                .map_err(fatal)?;
        }
        Ok(())
    }
}

eager!(Embed);
