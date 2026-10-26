//! The errors of the engine: what stopped a startup, and why a plan was refused.

use std::io;

use euhedral_inference_cuda::CudaError;
use euhedral_inference_edrl::EdrlError;
use thiserror::Error;

/// Why a residency plan could not be made.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PlanError {
    /// The model's geometry is not one this engine runs.
    #[error("unsupported model: {0}")]
    Unsupported(String),

    /// The artifact's tensors are not the model's inventory.
    #[error("the artifact does not hold the model: {0}")]
    Inventory(String),

    /// The weights alone, beside the state every sequence needs, do not fit the device budget.
    #[error(
        "the weights need {weights_needed} bytes on the device beside {state_needed} bytes of state, but only \
         {available} bytes are available ({free} free, limited by memory.gpu-limit and memory.gpu-reserve); weights \
         cannot be backed by host memory yet"
    )]
    WeightsDoNotFit {
        weights_needed: u64,
        state_needed: u64,
        available: u64,
        free: u64,
    },

    /// The device has no budget left after the limit and the reserve.
    #[error("the device budget is empty: {free} bytes are free and memory.gpu-reserve takes {reserve}")]
    NoBudget { free: u64, reserve: u64 },

    /// A setting the plan reads is out of range.
    #[error("{0}")]
    Setting(String),
}

/// Why the engine did not start, or stopped.
#[derive(Debug, Error)]
pub enum EngineError {
    /// The model or its artifact was refused before anything was allocated on the device.
    #[error(transparent)]
    Plan(#[from] PlanError),

    /// The artifact's bytes are malformed.
    #[error("artifact {path}: {source}")]
    Artifact {
        path: String,
        #[source]
        source: EdrlError,
    },

    /// A tensor's payload failed the format's own checks.
    #[error("tensor {name}: {reason}")]
    Payload { name: String, reason: String },

    /// The CUDA library refused a call.
    #[error(transparent)]
    Cuda(#[from] CudaError),

    /// A file could not be opened or read.
    #[error("{what}: {source}")]
    Io {
        what: String,
        #[source]
        source: io::Error,
    },

    /// The runtime could not start the lattice or one of its sources.
    #[error("{0}")]
    Runtime(String),
}
