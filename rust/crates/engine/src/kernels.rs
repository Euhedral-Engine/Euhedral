//! Where the engine's kernels come from when they run on RustEE: the sources beside the native library, compiled with
//! the settings' compile stack.

use euhedral_inference_cuda::CudaLibrary;
use euhedral_inference_cuda::launch::Dispatch;
use euhedral_inference_cuda::launch::load::{KernelSources, LoadError};
use euhedral_inference_settings::{Auto, KernelSettings, Settings};

/// The kernel sources installed beside `library`, to be compiled on a thread with `kernels.compile.stack` bytes of
/// stack. The caller passes them to [`load`](euhedral_inference_cuda::launch::load::load) once, at startup.
///
/// A stack the platform's address space cannot hold is asked for as the largest `usize`, and the thread that compiles
/// reports it cannot start ([`LoadError::CompileThread`]).
pub fn kernel_sources(library: &CudaLibrary, settings: &Settings) -> Result<KernelSources, LoadError> {
    let stack = usize::try_from(settings.kernels.compile.stack.bytes()).unwrap_or(usize::MAX);
    let dispatch = dispatch(library.device().sm_count, &settings.kernels);
    Ok(KernelSources::beside(library)?
        .with_compile_stack(stack)
        .with_dispatch(dispatch))
}

/// The launch choices of a device of `sm_count` streaming multiprocessors under `settings`: what `auto` stands for is
/// the value measured on the reference card, and the skinny NVFP4 grid follows the device.
pub fn dispatch(sm_count: u32, settings: &KernelSettings) -> Dispatch {
    let mut dispatch = Dispatch::measured(sm_count);
    if let Auto::Value(slices) = settings.rms_norm_decode_slices {
        dispatch.rms_norm_decode_slices = slices;
    }
    if let Auto::Value(table) = settings.fused_verify_from {
        dispatch.fused_verify_from = table.0;
    }
    dispatch
}

#[cfg(test)]
mod tests {
    use euhedral_inference_settings::FusedVerifyFrom;

    use super::*;

    #[test]
    fn auto_is_the_measured_value_and_a_setting_replaces_it() {
        let auto = dispatch(70, &KernelSettings::default());
        assert_eq!(auto, Dispatch::measured(70));
        assert_eq!(auto.skinny_target_ctas(), 140);
        let settings = KernelSettings {
            rms_norm_decode_slices: Auto::Value(12),
            fused_verify_from: Auto::Value(FusedVerifyFrom([1; 9])),
            ..KernelSettings::default()
        };
        let set = dispatch(128, &settings);
        assert_eq!(set.rms_norm_decode_slices, 12);
        assert_eq!(set.fused_verify_from, [1; 9]);
        assert_eq!(set.skinny_target_ctas(), 256);
    }
}
