//! The errors of the engine: what stopped a startup, and why a plan was refused.

use std::io;
use std::path::Path;
use std::time::Duration;

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
         {available} bytes are available ({free} free, limited by memory.gpu-limit); weights \
         cannot be backed by host memory yet"
    )]
    WeightsDoNotFit {
        weights_needed: u64,
        state_needed: u64,
        available: u64,
        free: u64,
    },

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

    /// The engine did not finish stopping within the window it was given; `left` says what was not done.
    #[error("the engine did not finish stopping within {} ms: {left}", .window.as_millis())]
    Unfinished { window: Duration, left: String },
}

impl EngineError {
    /// Whether nothing could be loaded: the CUDA library, its runtime libraries or the driver is missing. A start
    /// that fails this way never reached the device.
    pub fn loads_nothing(&self) -> bool {
        matches!(self, Self::Cuda(error) if error.loads_nothing())
    }

    pub(crate) fn io(what: impl Into<String>, source: io::Error) -> Self {
        Self::Io {
            what: what.into(),
            source,
        }
    }

    pub(crate) fn artifact(path: &Path, source: EdrlError) -> Self {
        Self::Artifact {
            path: path.display().to_string(),
            source,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use super::*;

    #[test]
    fn a_missing_library_or_driver_loads_nothing_and_another_call_failing_does_not() {
        let library = EngineError::Cuda(CudaError::LibraryMissing(PathBuf::from("x.so")));
        let runtime = EngineError::Cuda(CudaError::RuntimeMissing {
            library: PathBuf::from("x.so"),
            missing: vec![PathBuf::from("libcudart.so")],
        });
        let call = EngineError::Cuda(CudaError::Call {
            call: "cudaMalloc",
            status: 2,
        });
        assert!(library.loads_nothing());
        assert!(runtime.loads_nothing());
        assert!(!call.loads_nothing());
        assert!(!EngineError::Runtime("x".to_owned()).loads_nothing());
    }
}
