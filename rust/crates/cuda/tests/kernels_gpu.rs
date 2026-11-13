//! Device tests of the kernel loader and the planned-launch queue. `#[ignore]`d like the crate's other device tests:
//! `cargo test -p euhedral-inference-cuda --test kernels_gpu -- --ignored --test-threads=1 --nocapture`.
//!
//! The library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY`); a test skips with a
//! message when it is absent, or when the bundled NVRTC is not beside it (the compiler `load`
//! requires; its directory is on `LD_LIBRARY_PATH` for the builtins library).
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "the measurements are the test's output")]

mod common;

use std::fs::create_dir_all;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use euhedral_gpu::cuda::CudaGpu;
use euhedral_inference_cuda::CudaLibrary;
use euhedral_inference_cuda::launch::catalog::{Arguments, KERNELS, Launch, MAX_ARGUMENTS, MAX_TENSOR_MAPS};
use euhedral_inference_cuda::launch::load::{KernelSources, Slot, Unavailable, load};
use euhedral_inference_cuda::{DevicePtr, kernel, ops};

use common::twin::{Case, Twin, Verdict};

fn library() -> Option<Arc<CudaLibrary>> {
    let path = test_support::require_property("euhedral.cuda.library")?;
    Some(CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads"))
}

/// Whether the bundled compiler `load` requires is beside the library; a test that compiles skips without it.
fn compiler_available(library: &CudaLibrary) -> bool {
    KernelSources::beside(library)
        .expect("the product directory")
        .compiler()
        .is_some()
}

fn millis(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1000.0
}

#[test]
#[ignore = "needs a GPU, the native library and NVRTC"]
fn every_catalog_kernel_resolves_with_the_parameters_a_plan_passes() {
    let Some(library) = library() else { return };
    if !compiler_available(&library) {
        println!("skipped: no bundled NVRTC beside the library");
        return;
    }
    let gpu = CudaGpu::open(0).expect("the device opens through RustEE");
    let cache = test_support::gradle_property("euhedral.kernel.cache").map(|directory| {
        let cache = PathBuf::from(directory).join("kernel-module-cache");
        create_dir_all(&cache).expect("the cache directory");
        cache
    });

    // Cold: no cache.
    let sources = KernelSources::beside(&library).expect("the product directory");
    let total = Instant::now();
    let table = load(&gpu, &sources).expect("every module compiles and loads");
    let cold = total.elapsed();
    println!(
        "modules compiled and loaded without a cache: {:.0} ms in total, on the compile thread (the largest module included)",
        millis(cold)
    );
    for timing in table.timings() {
        println!("  {:<28} {:>8.1} ms", timing.module.root(), millis(timing.elapsed));
    }

    // With the module cache RustEE provides: a first load fills it, the second reads it.
    if let Some(cache) = cache {
        let cached = sources.clone().with_cache_dir(cache);
        let fill = Instant::now();
        load(&gpu, &cached).expect("the cache fills");
        let fill = fill.elapsed();
        let warm = Instant::now();
        load(&gpu, &cached).expect("the cache is read");
        println!(
            "with the module cache: first {:.0} ms (fills it), second {:.0} ms",
            millis(fill),
            millis(warm.elapsed())
        );
    } else {
        println!("with the module cache: skipped, set euhedral.kernel.cache to a directory for it");
    }

    // Every catalog kernel resolves (the architecture-specific modules on a compute capability 12.x device only, as in
    // the native loader), opted in to the dynamic shared memory its plans need, and its declared parameters are what a
    // plan can pass.
    let mut unavailable = Vec::new();
    let mut unexpected = Vec::new();
    let mut ready = 0;
    let mut failures = Vec::new();
    for (id, slot) in table.iter() {
        match slot {
            Slot::Unavailable(reason @ Unavailable::Compile { .. }) => {
                failures.push(format!("{}: {reason}", id.name()));
            }
            Slot::Unavailable(reason) => {
                unavailable.push(format!("{}: {reason}", id.name()));
                if !(matches!(reason, Unavailable::ArchitectureSpecificTarget)
                    && id.spec().architecture_specific()
                    && gpu.arch().0 != 12)
                {
                    unexpected.push(format!("{}: {reason}", id.name()));
                }
            }
            Slot::Ready(resolved) => {
                ready += 1;
                let needed = id.spec().shared_opt_in_bytes;
                match resolved.kernel().max_dynamic_shared_bytes() {
                    Some(limit) if limit >= needed => {}
                    other => failures.push(format!(
                        "{}: the plans need {needed} bytes of dynamic shared memory, the kernel's limit is {other:?}",
                        id.name()
                    )),
                }
                let Some(sizes) = resolved.parameter_sizes() else {
                    failures.push(format!("{}: the driver reports no parameter sizes", id.name()));
                    continue;
                };
                // A plan's arguments are its tensor maps (128 bytes, in front), then words of 4 or 8 bytes, at most
                // MAX_ARGUMENTS of them.
                let maps = if id.spec().tensor_map_arguments() {
                    MAX_TENSOR_MAPS
                } else {
                    0
                };
                if sizes.len() > maps + MAX_ARGUMENTS {
                    failures.push(format!(
                        "{}: {} parameters, a plan passes at most {}",
                        id.name(),
                        sizes.len(),
                        maps + MAX_ARGUMENTS
                    ));
                }
                if sizes.len() < maps || sizes[..maps].iter().any(|&size| size != 128) {
                    failures.push(format!(
                        "{}: the leading {maps} parameters are not tensor maps: {sizes:?}",
                        id.name()
                    ));
                    continue;
                }
                if let Some(size) = sizes[maps..].iter().find(|&&size| size != 4 && size != 8) {
                    failures.push(format!(
                        "{}: a parameter of {size} bytes, a plan passes 4 or 8",
                        id.name()
                    ));
                }
            }
        }
    }
    println!("{ready} of {} catalog kernels resolved", KERNELS.len());
    println!("unavailable ({}):", unavailable.len());
    for line in &unavailable {
        println!("  {line}");
    }
    assert!(failures.is_empty(), "parameter mismatches:\n{}", failures.join("\n"));
    assert!(
        unexpected.is_empty(),
        "kernels that should resolve and do not:\n{}",
        unexpected.join("\n")
    );
    assert_eq!(ready + unavailable.len(), KERNELS.len());
    // On this machine's class of device (compute capability 12.x) nothing is unavailable.
    if gpu.arch().0 == 12 {
        assert_eq!(ready, KERNELS.len());
    }
}

/// The harness itself, end to end on one real kernel: the C wrapper and the planned launch agree bit for bit, and a
/// plan that differs is told apart.
#[test]
#[ignore = "needs a GPU, the native library and NVRTC"]
fn the_twin_harness_compares_a_wrapper_with_its_plan_bit_for_bit() {
    let Some(library) = library() else { return };
    if !compiler_available(&library) {
        println!("skipped: no bundled NVRTC beside the library");
        return;
    }
    let mut twin = Twin::open(&library).expect("both sides open on one device");
    let count: u32 = 1000;
    // bf16 values from a fixed pattern (any bit pattern is a value or a NaN, and both sides compute it the same way).
    let pattern = |seed: u32| -> Vec<u8> {
        (0..count)
            .flat_map(|index| (0x3c00u16 + (index.wrapping_mul(seed) % 0x300) as u16).to_le_bytes())
            .collect()
    };
    let bytes = count as usize * 2;
    let planned = |pointers: &[DevicePtr], delta_index: usize| {
        vec![Launch::new(
            kernel!("euhedral_residual_add_bf16"),
            [count.div_ceil(256), 1, 1],
            [256, 1, 1],
            0,
            Arguments::new()
                .pointer(pointers[0])
                .pointer(pointers[delta_index])
                .pointer(pointers[2])
                .u32(count),
        )]
    };
    let case = |delta_index: usize| Case {
        name: "residual add",
        buffers: vec![pattern(7), pattern(13), vec![0; bytes]],
        prepare: None,
        native: Box::new(move |submission, pointers| {
            ops::residual_add_bf16(submission, pointers[0], pointers[1], pointers[2], 1, count)
        }),
        plan: Box::new(move |pointers| planned(pointers, delta_index)),
        outputs: vec![2],
        row_exact: false,
        tables: Vec::new(),
        scratch: Vec::new(),
    };
    let verdict = twin.run(&case(1)).expect("the case runs");
    println!("residual add: {verdict:?}");
    assert_eq!(verdict, Verdict::Identical { bytes });
    // The plan adds the wrong operand: the harness must report a difference, not equality.
    let verdict = twin.run(&case(0)).expect("the case runs");
    assert!(matches!(verdict, Verdict::Differs { buffer: 2, .. }), "{verdict:?}");
    assert_eq!(case(1).name, "residual add");
}
