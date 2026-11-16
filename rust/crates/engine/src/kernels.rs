//! Where the engine's kernels come from when they run on RustEE: the sources beside the native library, compiled with
//! the settings' compile stack.

use euhedral_inference_cuda::CudaLibrary;
use euhedral_inference_cuda::launch::load::{KernelSources, LoadError};
use euhedral_inference_settings::Settings;

/// The kernel sources installed beside `library`, to be compiled on a thread with `kernels.compile.stack` bytes of
/// stack. The caller passes them to [`load`](euhedral_inference_cuda::launch::load::load) once, at startup.
///
/// A stack the platform's address space cannot hold is asked for as the largest `usize`, and the thread that compiles
/// reports it cannot start ([`LoadError::CompileThread`]).
pub fn kernel_sources(library: &CudaLibrary, settings: &Settings) -> Result<KernelSources, LoadError> {
    let stack = usize::try_from(settings.kernels.compile.stack.bytes()).unwrap_or(usize::MAX);
    Ok(KernelSources::beside(library)?.with_compile_stack(stack))
}
