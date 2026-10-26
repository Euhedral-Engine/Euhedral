//! The norms: the rounded residual add with the next RMSNorm in one launch, and the `(1 + w)` RMSNorm
//! (`Stages.ResidualNorm`, `Stages.RmsNorm`).

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{DevicePtr, ops};

use super::{Quantum, fatal, launch_u32, submit};
use crate::workspace::BufferSlot;

/// `hidden = residual + delta` and `normalized = rms_norm(hidden) * (1 + weight)`, for the rows of the quantum.
#[derive(Clone, Debug)]
pub struct ResidualRmsNorm {
    residual: BufferSlot,
    delta: BufferSlot,
    weight: DevicePtr,
    hidden: BufferSlot,
    normalized: BufferSlot,
}

impl ResidualRmsNorm {
    /// Adds `delta` to `residual` into `hidden` and normalizes it with the norm at `weight` into `normalized`.
    pub fn new(
        residual: BufferSlot,
        delta: BufferSlot,
        weight: DevicePtr,
        hidden: BufferSlot,
        normalized: BufferSlot,
    ) -> Self {
        Self {
            residual,
            delta,
            weight,
            hidden,
            normalized,
        }
    }
}

impl Stage<Quantum> for ResidualRmsNorm {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let rows = quantum.rows_u32()?;
        let width = launch_u32(quantum.config.hidden_size, "the hidden size")?;
        submit(quantum, cx, false, |submission| {
            ops::residual_rms_norm_bf16(
                submission,
                quantum.slot(self.residual),
                quantum.slot(self.delta),
                self.weight,
                quantum.slot(self.hidden),
                quantum.slot(self.normalized),
                rows,
                width,
                quantum.epsilon(),
            )
            .map_err(fatal)
        })
    }
}

/// Which rows a norm covers.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NormRows {
    /// The quantum's rows.
    All,
    /// The final norm, whose rows are the quantum's logits rows: the last ones of the input.
    ///
    /// It also keeps what drafting reads of it. A quantum that seeds drafting normalizes every row into the
    /// sequence's draft seed rows first (`docs/speculation/MTP_CONTRACT.md` section 2), and a draft quantum copies
    /// its last normalized row to the recursion hidden, which seeds the next recursive draft row.
    Final,
}

/// The `(1 + w)` RMSNorm of BF16 rows (`rms_norm_unit_offset_bf16`).
#[derive(Clone, Debug)]
pub struct RmsNormUnitOffset {
    input: BufferSlot,
    output: BufferSlot,
    weight: DevicePtr,
    rows: NormRows,
}

impl RmsNormUnitOffset {
    /// Normalizes `rows` of `input` with the norm at `weight` into `output`.
    pub fn new(input: BufferSlot, output: BufferSlot, weight: DevicePtr, rows: NormRows) -> Self {
        Self {
            input,
            output,
            weight,
            rows,
        }
    }
}

impl Stage<Quantum> for RmsNormUnitOffset {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let hidden = quantum.config.hidden_size;
        let width = launch_u32(hidden, "the hidden size")?;
        let row_bytes = hidden as u64 * 2;
        let covered = match self.rows {
            NormRows::All => quantum.rows,
            NormRows::Final => quantum.logits_rows,
        };
        let seeds = self.rows == NormRows::Final && quantum.seeds_draft;
        if covered == 0 && !seeds {
            return Ok(Submitted::done());
        }
        submit(quantum, cx, false, |submission| {
            let input = quantum.slot(self.input);
            if seeds {
                let draft = quantum
                    .seq
                    .state()
                    .scratch
                    .draft()
                    .ok_or_else(|| fatal("a quantum seeds drafting but the sequence has no draft buffers"))?;
                if quantum.rows as u64 > draft.seed_capacity {
                    return Err(fatal(format!(
                        "{} draft seed rows exceed the {} the sequence holds",
                        quantum.rows, draft.seed_capacity
                    )));
                }
                ops::rms_norm_unit_offset_bf16(
                    submission,
                    input,
                    self.weight,
                    draft.seed_rows.ptr(),
                    quantum.rows_u32()?,
                    width,
                    quantum.epsilon(),
                )
                .map_err(fatal)?;
            }
            if covered == 0 {
                return Ok(());
            }
            // The final norm of fewer rows than the quantum has reads the last ones.
            let first = (quantum.rows - covered) as u64;
            let output = quantum.slot(self.output);
            ops::rms_norm_unit_offset_bf16(
                submission,
                input.offset(first * row_bytes),
                self.weight,
                output,
                launch_u32(covered, "the norm rows")?,
                width,
                quantum.epsilon(),
            )
            .map_err(fatal)?;
            if self.rows == NormRows::Final
                && quantum.kind == super::QuantumKind::Draft
                && let Some(draft) = quantum.seq.state().scratch.draft()
            {
                let last = (covered as u64 - 1) * row_bytes;
                ops::copy_device_to_device(submission, draft.recursion_hidden.ptr(), output.offset(last), row_bytes)
                    .map_err(fatal)?;
            }
            Ok(())
        })
    }
}
