//! The errors of the engine: what stopped a startup, and why a plan was refused.

use std::io;
use std::path::Path;
use std::time::Duration;

use euhedral_inference_cuda::CudaError;
use euhedral_inference_edrl::EdrlError;
use thiserror::Error;

/// What a refusal says is too large: one context, or the contexts of several sequences.
fn what_does_not_fit(sequences: usize) -> String {
    if sequences > 1 {
        format!("{sequences} sequences")
    } else {
        "a context".to_string()
    }
}

/// What the refusal of a plan of several sequences adds: they may be what did not fit.
fn or_fewer_sequences(sequences: usize) -> &'static str {
    if sequences > 1 {
        " or fewer sequences (batching.max-sequences)"
    } else {
        ""
    }
}

/// What a device refusal adds when the host is short too: the pinned bytes the plan holds there, and the budget.
fn host_figures(host: &HostFigures) -> String {
    if host.needed() <= host.budget {
        return String::new();
    }
    format!(
        "; the host is short too: the weights outside the GPU need {} bytes of pinned host memory (host arena and \
         disk window) and the cache pages on the host {} bytes, against a host budget of {} bytes",
        host.weights, host.cache_pages, host.budget
    )
}

/// What a host refusal names beside the host budget: when cache pages were moved to the host, the setting that keeps
/// them on the device.
fn cache_pages_part(cache_pages: u64) -> String {
    if cache_pages == 0 {
        return String::new();
    }
    format!(
        " ({} MiB of it the cache pages moved to the host; raise context.kv-cache-on-gpu-tokens to keep more of them on \
         the GPU)",
        cache_pages.div_ceil(1 << 20)
    )
}

/// The pinned host bytes a refused plan needs, beside what the host allows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HostFigures {
    /// The host arena and the disk window of the weights that left the device.
    pub weights: u64,
    /// The cache pages put on the host.
    pub cache_pages: u64,
    /// The host budget the plan was made with.
    pub budget: u64,
}

impl HostFigures {
    /// All the pinned host bytes the plan needs.
    pub fn needed(&self) -> u64 {
        self.weights + self.cache_pages
    }
}

/// Why a residency plan could not be made.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum PlanError {
    /// The model's geometry is not one this engine runs.
    #[error("unsupported model: {0}")]
    Unsupported(String),

    /// The artifact's tensors are not the model's inventory.
    #[error("the artifact does not hold the model: {0}")]
    Inventory(String),

    /// The model does not fit the device even with the weights it can offload in host memory or in the artifact file:
    /// what stays on the device (the weights that cannot leave, the staging ring, the cache and the working memory)
    /// is more than the device has.
    #[error(
        "{} of {context_tokens} tokens does not fit in the {} MiB free on this GPU, even with the weights in host \
         memory (the plan needs {weights_needed} bytes of weights and {state_needed} bytes of state on the device, \
         and {available} bytes are available: {free} free, limited by memory.gpu-limit); set a smaller max \
         context.max-tokens{}{}",
        what_does_not_fit(*.sequences),
        available / (1 << 20),
        or_fewer_sequences(*.sequences),
        host_figures(host)
    )]
    DeviceTooSmallWithOffload {
        /// The device bytes of the weights that stay and of the staging ring, at the offload that came closest.
        weights_needed: u64,
        /// The device bytes of the cache, the runtime's reserve, the scratch and the state of every sequence.
        state_needed: u64,
        available: u64,
        free: u64,
        sequences: usize,
        context_tokens: u64,
        /// The host the plan used, for the message when the host is short too.
        host: HostFigures,
    },

    /// The weights the device cannot hold and the cache pages moved to the host need more pinned host memory than the
    /// host budget allows. `needed` counts the host arena, the disk window and the cache pages on the host, of which
    /// `cache_pages` are the cache pages.
    #[error(
        "the weights the GPU cannot hold{} need {} MiB of host memory{} (the token embedding, the LM head, the \
         host-backed weights and the disk window), more than the {} MiB host budget; set memory.host-weights-limit or \
         memory.host-limit higher, or a smaller context.max-tokens or fewer sequences (batching.max-sequences)",
        if *.cache_pages > 0 { " and the attention cache pages moved to the host" } else { "" },
        needed.div_ceil(1 << 20),
        cache_pages_part(*.cache_pages),
        budget / (1 << 20)
    )]
    HostTooSmall { needed: u64, cache_pages: u64, budget: u64 },

    /// The plan puts weights outside the device, and this build cannot run weights held in host memory or in the file.
    #[error(
        "the model does not fit the GPU: {weights} weights ({host_weights} in host memory, {disk_weights} in the \
         artifact file) would be held outside it, which needs {} bytes of pinned host memory ({arena_bytes} for the \
         host arena and {window_bytes} for the disk window); this build does not run host or disk weights yet",
        arena_bytes + window_bytes
    )]
    WeightsOutsideDevice {
        weights: usize,
        host_weights: usize,
        disk_weights: usize,
        arena_bytes: u64,
        window_bytes: u64,
    },

    /// The plan needs pinned host memory (cache pages or weights) and the machine does not report what it can spare.
    #[error(
        "the plan needs {needed} bytes of pinned host memory (cache pages and weights the device cannot hold), and \
         this machine does not report how much host memory it can spare; set memory.host-limit or \
         memory.host-weights-limit to allow it, or lower context.max-tokens"
    )]
    HostBudgetUnknown { needed: u64 },

    /// A setting the plan reads is out of range.
    #[error("{0}")]
    Setting(String),

    /// The process's memory planner did not admit a plan the residency plan had made: the two disagree.
    #[error("the memory planner did not admit the plan: {0}")]
    NotAdmitted(String),
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
