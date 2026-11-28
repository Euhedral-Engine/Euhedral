//! Device test of the module loader: every quantization module compiles with the bundled NVRTC and every listed
//! kernel resolves. `#[ignore]`d like the crate's other device tests:
//! `cargo test -p euhedral-inference-quantization --test kernels_gpu -- --ignored --nocapture`.
//!
//! The engine library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY`); the CUDA headers
//! and the bundled NVRTC are beside it. Without the property the test fails: a device test that skips is no check.
//! The builtins library of NVRTC must be on `LD_LIBRARY_PATH`.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "the compile times are the test's output")]

use std::path::PathBuf;
use std::time::Instant;

use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_inference_cuda::CudaLibrary;
use euhedral_inference_quantization::kernels::{Module, ModuleOptions, Modules};

#[test]
#[ignore = "needs a GPU and the bundled NVRTC"]
fn every_module_compiles_and_every_listed_kernel_resolves() {
    let path = test_support::test_setting("euhedral.cuda.library")
        .expect("set euhedral.cuda.library (EUHEDRAL_TEST_CUDA_LIBRARY) to the engine library");
    let library = CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads");
    let options = ModuleOptions::beside(&library).expect("the product directory");
    assert!(options.compiler.is_some(), "no bundled NVRTC beside the library");
    let gpu = CudaGpu::open(0).expect("the device opens through RustEE");
    let mut modules = Modules::<CudaModule>::open(&gpu, options).expect("the compiler gives machine code");
    for module in Module::ALL {
        let started = Instant::now();
        let compiled = modules.load(module).unwrap_or_else(|error| panic!("{error}"));
        let resolved = compiled.kernels().count();
        assert_eq!(resolved, module.kernels().len());
        println!(
            "compiled {} in {:.0} ms: {resolved} kernels resolved",
            module.file(),
            started.elapsed().as_secs_f64() * 1000.0
        );
    }
}
