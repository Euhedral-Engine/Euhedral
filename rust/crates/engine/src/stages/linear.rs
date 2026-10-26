//! The linear stage: one projection through its weight's format (`Stages.Linear`).

use std::fmt;
use std::sync::Arc;

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::Submission;

use super::{Launch, Quantum, QuantumKind, eager, fatal};
use crate::formats::{LinearFormat, LinearMode, ScratchRegion};
use crate::model::WeightRef;
use crate::workspace::BufferSlot;

/// Which rows a linear covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearRows {
    /// The quantum's rows.
    All,
    /// The quantum's logits rows: the head. With none, nothing runs.
    Logits,
}

/// `output = weight * input` over the quantum's rows, launched by the format the weight is stored in. The format
/// decides the route from the rows and the mode; the stage hands it the mode of the quantum (row-exact for a
/// verification, prefill for a prompt's chunks) and the expansion scratch it asked for.
pub struct Linear {
    format: Arc<dyn LinearFormat>,
    weight: WeightRef,
    input: BufferSlot,
    output: BufferSlot,
    rows: LinearRows,
    scratch_region: Option<usize>,
}

impl Linear {
    /// The projection of `weight` by `format`, from `input` to `output`, taking the workspace's scratch region
    /// `scratch_region` when the route needs one.
    pub fn new(
        format: Arc<dyn LinearFormat>,
        weight: WeightRef,
        input: BufferSlot,
        output: BufferSlot,
        rows: LinearRows,
        scratch_region: Option<usize>,
    ) -> Self {
        Self {
            format,
            weight,
            input,
            output,
            rows,
            scratch_region,
        }
    }
}

impl fmt::Debug for Linear {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Linear")
            .field("weight", &self.weight.name)
            .field("input", &self.input)
            .field("output", &self.output)
            .field("rows", &self.rows)
            .field("scratch_region", &self.scratch_region)
            .finish_non_exhaustive()
    }
}

impl Linear {
    /// The rows this linear covers in `quantum`.
    fn covered(&self, quantum: &Quantum) -> usize {
        match self.rows {
            LinearRows::All => quantum.rows,
            LinearRows::Logits => quantum.logits_rows,
        }
    }
}

impl Launch for Linear {
    fn idle(&self, quantum: &Quantum) -> bool {
        self.covered(quantum) == 0
    }

    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError> {
        if self.idle(quantum) {
            return Ok(());
        }
        let rows = self.covered(quantum);
        let mode = LinearMode {
            row_exact: quantum.mode.row_exact,
            prefill: quantum.kind == QuantumKind::Prefill,
        };
        let needed = self.format.scratch_bytes(&self.weight, rows, mode);
        let scratch = match self.scratch_region.and_then(|region| quantum.workspace.scratch(region)) {
            Some(buffer) if buffer.bytes() >= needed => ScratchRegion {
                ptr: buffer.ptr(),
                bytes: buffer.bytes(),
            },
            _ if needed == 0 => ScratchRegion::NONE,
            found => {
                return Err(fatal(format!(
                    "{} needs {needed} scratch bytes for {rows} rows but the region holds {}",
                    self.weight.name,
                    found.map_or(0, |buffer| buffer.bytes())
                )));
            }
        };
        self.format
            .submit(
                submission,
                &self.weight,
                quantum.slot(self.input),
                quantum.slot(self.output),
                rows,
                scratch,
                mode,
            )
            .map_err(fatal)
    }
}

eager!(Linear);
