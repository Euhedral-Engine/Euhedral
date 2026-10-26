//! The elementwise operations: the rounded residual add and the FFN's SwiGLU (`Stages.ResidualAdd`,
//! `Stages.SwiGlu`).

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::ops;

use super::{Quantum, fatal, launch_u32, submit};
use crate::workspace::BufferSlot;

/// `output = residual + delta`, rounded to BF16, for the quantum's rows.
#[derive(Clone, Debug)]
pub struct ResidualAdd {
    residual: BufferSlot,
    delta: BufferSlot,
    output: BufferSlot,
}

impl ResidualAdd {
    pub fn new(residual: BufferSlot, delta: BufferSlot, output: BufferSlot) -> Self {
        Self {
            residual,
            delta,
            output,
        }
    }
}

impl Stage<Quantum> for ResidualAdd {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let rows = quantum.rows_u32()?;
        let width = launch_u32(quantum.config.hidden_size, "the hidden size")?;
        submit(quantum, cx, false, |submission| {
            ops::residual_add_bf16(
                submission,
                quantum.slot(self.residual),
                quantum.slot(self.delta),
                quantum.slot(self.output),
                rows,
                width,
            )
            .map_err(fatal)
        })
    }
}

/// The FFN's SwiGLU of the gate and up rows, for the quantum's rows.
#[derive(Clone, Debug)]
pub struct Swiglu {
    gate_up: BufferSlot,
    output: BufferSlot,
}

impl Swiglu {
    pub fn new(gate_up: BufferSlot, output: BufferSlot) -> Self {
        Self { gate_up, output }
    }
}

impl Stage<Quantum> for Swiglu {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let rows = quantum.rows_u32()?;
        let intermediate = launch_u32(quantum.config.intermediate_size, "the intermediate size")?;
        submit(quantum, cx, false, |submission| {
            ops::swiglu_bf16(
                submission,
                quantum.slot(self.gate_up),
                quantum.slot(self.output),
                rows,
                intermediate,
            )
            .map_err(fatal)
        })
    }
}
