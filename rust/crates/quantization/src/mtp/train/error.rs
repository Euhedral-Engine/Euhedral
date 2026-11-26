//! Why a training run cannot start or cannot go on.

use std::io::Error as IoError;

use euhedral_gpu::GpuError;
use euhedral_inference_cuda::CudaError;
use thiserror::Error;

use super::progress::StateError;
use super::settings::SettingsError;
use crate::assemble::AssembleError;
use crate::chain::ChainError;
use crate::checkpoint::CheckpointError;
use crate::kernels::ModuleError;
use crate::mtp::DrafterError;
use crate::resume::ResumeError;
use crate::training::TrainError;

/// A failure of the run.
#[derive(Debug, Error)]
pub enum RunError {
    #[error(transparent)]
    Settings(#[from] SettingsError),
    #[error(transparent)]
    State(#[from] StateError),
    #[error(transparent)]
    Gpu(#[from] GpuError),
    #[error(transparent)]
    Cuda(#[from] CudaError),
    #[error(transparent)]
    Module(#[from] ModuleError),
    #[error(transparent)]
    Drafter(#[from] DrafterError),
    #[error(transparent)]
    Train(#[from] TrainError),
    #[error(transparent)]
    Checkpoint(#[from] CheckpointError),
    #[error(transparent)]
    Artifact(#[from] AssembleError),
    #[error(transparent)]
    Resume(#[from] ResumeError),
    #[error("{0}")]
    Io(#[from] IoError),
    /// An input the run cannot use: the corpus, the checkpoint or the artifact is not what the drafter needs.
    #[error("{0}")]
    Input(String),
    /// A chain or a frame that broke its own rules.
    #[error("{0}")]
    Broken(String),
}

impl ChainError for RunError {
    fn broken(message: &str) -> Self {
        RunError::Broken(message.to_owned())
    }
}
