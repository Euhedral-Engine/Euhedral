//! The error every fallible call in this crate returns.

use std::path::PathBuf;

use euhedral_graph::LaneId;
use thiserror::Error;

/// A failed call into the CUDA library, or a refused request that never reached it.
#[derive(Debug, Error)]
pub enum CudaError {
    /// The library file is not there.
    #[error("the CUDA library {} does not exist", .0.display())]
    LibraryMissing(PathBuf),

    /// The runtime libraries that belong beside the CUDA library are not there.
    #[error(
        "the CUDA runtime libraries {missing:?} are missing; they belong beside {}, as the native files lay it out",
        .library.display()
    )]
    RuntimeMissing { library: PathBuf, missing: Vec<PathBuf> },

    /// The loader refused the library or a library it links.
    #[error(
        "cannot load {} or a library it links: the CUDA 13 runtime and NVRTC belong beside it and the NVIDIA driver \
         must be installed ({source})",
        .path.display()
    )]
    Load { path: PathBuf, source: libloading::Error },

    /// The kernel header directory the library compiles against was refused.
    #[error("the CUDA include directory {} is invalid", .0.display())]
    IncludeDirectory(PathBuf),

    /// A C entry point returned a nonzero status.
    #[error("{call} failed: {}", describe(*.status))]
    Call { call: &'static str, status: i32 },

    /// An allocation returned a null address.
    #[error("{call} returned a null address for {bytes} bytes")]
    NullAllocation { call: &'static str, bytes: u64 },

    /// A request that would be wrong while the lane is captured into a graph.
    #[error("{0} is not allowed while the lane is being captured into a graph")]
    Capturing(&'static str),

    /// A lane id of 64 or more; a `LanePool` holds at most 64 lanes.
    #[error("lane {} is out of range: a lane pool holds at most 64 lanes", .0.0)]
    LaneOutOfRange(LaneId),

    /// A second lane was created with an id this library already holds.
    #[error("lane {} already exists in this library", .0.0)]
    LaneInUse(LaneId),

    /// A marker names a lane this library has not created.
    #[error("no lane {} exists in this library", .0.0)]
    UnknownLane(LaneId),

    /// An argument the call cannot take.
    #[error("{0}")]
    Argument(String),
}

/// The meaning of a status: the library's own codes, else a CUDA runtime or driver error number.
fn describe(status: i32) -> String {
    match status {
        -1 => "invalid argument".to_string(),
        -2 => "size overflow".to_string(),
        -3 => "format mismatch".to_string(),
        -4 => "kernel unavailable".to_string(),
        -5 => "route unavailable".to_string(),
        other => format!("CUDA error {other}"),
    }
}

/// `Ok` for status 0, else the call's failure.
pub(crate) fn check(call: &'static str, status: i32) -> Result<(), CudaError> {
    if status == 0 {
        Ok(())
    } else {
        Err(CudaError::Call { call, status })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_failed_call_names_the_call_and_the_status() {
        let message = check("euhedral_cuda_free", -1).unwrap_err().to_string();
        assert!(message.contains("euhedral_cuda_free"), "{message}");
        assert!(message.contains("invalid argument"), "{message}");
        assert!(
            check("euhedral_cuda_free", 2)
                .unwrap_err()
                .to_string()
                .contains("CUDA error 2")
        );
        assert!(check("euhedral_cuda_free", 0).is_ok());
    }
}
