//! The elementwise operations: the rounded residual add and the FFN's SwiGLU (`Stages.ResidualAdd`,
//! `Stages.SwiGlu`).

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{Submission, ops};

use super::{Launch, Quantum, eager, fatal, launch_u32};
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

impl Launch for ResidualAdd {
    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError> {
        ops::residual_add_bf16(
            submission,
            quantum.slot(self.residual),
            quantum.slot(self.delta),
            quantum.slot(self.output),
            quantum.rows_u32()?,
            launch_u32(quantum.config.hidden_size, "the hidden size")?,
        )
        .map_err(fatal)
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

impl Launch for Swiglu {
    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError> {
        ops::swiglu_bf16(
            submission,
            quantum.slot(self.gate_up),
            quantum.slot(self.output),
            quantum.rows_u32()?,
            launch_u32(quantum.config.intermediate_size, "the intermediate size")?,
        )
        .map_err(fatal)
    }
}

eager!(ResidualAdd, Swiglu);
