//! Device test of the module loader: every quantization module compiles with NVRTC and every listed kernel resolves.
//! `#[ignore]`d like the crate's other device tests:
//! `cargo test -p euhedral-inference-quantization --test kernels_gpu -- --ignored --nocapture`.
//!
//! The CUDA header directory comes from the `euhedral.cuda.headers` property (`EUHEDRAL_TEST_CUDA_HEADERS`); the
//! test skips with a message without it. NVRTC is found as the backend finds it (`EUHEDRAL_NVRTC`, with its builtins
//! library on `LD_LIBRARY_PATH`).
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "the compile times are the test's output")]
#![allow(
    clippy::disallowed_methods,
    reason = "a test: NVRTC needs a deep stack, and the test waits for the compile by nature"
)]

use std::path::PathBuf;
use std::thread::Builder;
use std::time::Instant;

use euhedral_gpu::cuda::{CudaGpu, CudaModule};
use euhedral_inference_quantization::kernels::{Module, ModuleCache, ModuleOptions};

const COMPILE_STACK_BYTES: usize = 256 << 20;

#[test]
#[ignore = "needs a GPU and NVRTC"]
fn every_module_compiles_and_every_listed_kernel_resolves() {
    let Some(headers) = test_support::require_property("euhedral.cuda.headers") else {
        return;
    };
    let options = ModuleOptions::new(PathBuf::from(headers));
    let worker = Builder::new()
        .name("kernel-compile".to_owned())
        .stack_size(COMPILE_STACK_BYTES)
        .spawn(move || {
            let gpu = CudaGpu::open(0).expect("the device opens through RustEE");
            let mut cache = ModuleCache::<CudaModule>::new(gpu, options);
            for module in Module::ALL {
                let started = Instant::now();
                let compiled = cache.load(module).unwrap_or_else(|error| panic!("{error}"));
                let resolved = compiled.kernels().count();
                assert_eq!(resolved, module.kernels().len());
                println!(
                    "compiled {} in {:.0} ms: {resolved} kernels resolved",
                    module.file(),
                    started.elapsed().as_secs_f64() * 1000.0
                );
            }
        })
        .unwrap();
    worker.join().unwrap();
}
