//! The linear stage: one projection through its weight's format (`Stages.Linear`).

use std::fmt;
use std::sync::Arc;

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::KernelOps;

use super::{Launch, Quantum, QuantumKind, eager, fatal, launch_u32};
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
    swiglu: Option<BufferSlot>,
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
            swiglu: None,
        }
    }

    /// The same projection followed by the SwiGLU of its output into `swiglu`: where the weight's format runs the
    /// pair as one launch it does, and everywhere else the stage queues the linear and then the SwiGLU, so the slots
    /// hold the same values either way.
    pub fn with_swiglu(mut self, swiglu: BufferSlot) -> Self {
        self.swiglu = Some(swiglu);
        self
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
            .field("swiglu", &self.swiglu)
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

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
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
        if let Some(swiglu) = self.swiglu
            && self
                .format
                .submit_gate_up_swiglu(
                    kernels,
                    &self.weight,
                    quantum.slot(self.input),
                    quantum.slot(swiglu),
                    rows,
                    scratch,
                    mode,
                )
                .map_err(fatal)?
        {
            return Ok(());
        }
        self.format
            .submit(
                kernels,
                &self.weight,
                quantum.slot(self.input),
                quantum.slot(self.output),
                rows,
                scratch,
                mode,
            )
            .map_err(fatal)?;
        match self.swiglu {
            Some(swiglu) => kernels
                .swiglu_bf16(
                    quantum.slot(self.output),
                    quantum.slot(swiglu),
                    quantum.rows_u32()?,
                    launch_u32(quantum.config.intermediate_size, "the intermediate size")?,
                )
                .map_err(fatal),
            None => Ok(()),
        }
    }
}

eager!(Linear);
