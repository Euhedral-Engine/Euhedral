//! Temporary: removed together with the native launch path, once the kernels run through RustEE.
//!
//! The NVFP4 linear launches (decode rows, grouped twins, the scalar reference, and the native route's quantize,
//! skinny and finish kernels) through the C wrappers and through their plans on a RustEE lane, compared byte for byte:
//! `cargo test -p euhedral-inference-cuda --test twin_linear -- --ignored --test-threads=1 --nocapture`.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "the tallies are the test's output")]

mod common;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use euhedral_inference_cuda::launch::load::KernelSources;
use euhedral_inference_cuda::launch::nvfp4::{
    Nvfp4Layout, Nvfp4Operands, nvfp4_byte_size, nvfp4_native_scratch_bytes, plan_linear, plan_native,
};
use euhedral_inference_cuda::{CudaLibrary, DevicePtr, ops};

use common::twin::{Case, Random, Tally, Twin, watchdog};

const ROWS: [u32; 9] = [1, 2, 8, 9, 32, 64, 65, 128, 512];
/// (inputs, outputs, layouts): 512 inputs one skinny split; 2048 several; 40 outputs the reference; 5376 the wide
/// decode tile and ten splits; 1024 by 256 the tile route above 64 rows.
const SHAPES: [(u32, u32, bool); 5] = [
    (512, 128, true),
    (2048, 128, true),
    (256, 40, false),
    (5376, 64, false),
    (1024, 256, false),
];

fn library() -> Option<Arc<CudaLibrary>> {
    let path = test_support::require_property("euhedral.cuda.library")?;
    Some(CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads"))
}

/// A weight of `out_features` rows: random codes, small E4M3 scales, a global scale of one.
fn weights(random: &mut Random, in_features: u32, out_features: u32, layout: Nvfp4Layout) -> Vec<u8> {
    let total = nvfp4_byte_size(out_features, in_features, layout) as usize;
    let k = (in_features as usize).div_ceil(128) * 128;
    let codes = (out_features as usize * k / 2).div_ceil(256) * 256;
    let mut bytes = random.bytes(total);
    for byte in &mut bytes[codes..] {
        // Scales and the sd4 table: E4M3 values of moderate size (and indices of any value).
        *byte = 0x30 + (*byte & 0x1f);
    }
    bytes[total - 4..].copy_from_slice(&1.0f32.to_le_bytes());
    bytes
}

fn operands(p: &[DevicePtr], rows: u32, in_features: u32, out_features: u32, weight_bytes: usize) -> Nvfp4Operands {
    Nvfp4Operands {
        input: p[0],
        weights: p[1],
        output: p[2],
        rows,
        in_features,
        out_features,
        weights_byte_size: weight_bytes as u64,
    }
}

#[test]
#[ignore = "needs a GPU, the native library and NVRTC"]
fn nvfp4_linear_plans_match_the_native_wrappers() {
    let Some(library) = library() else { return };
    if KernelSources::beside(&library)
        .expect("the product directory")
        .compiler()
        .is_none()
    {
        println!("skipped: no bundled NVRTC beside the library");
        return;
    }
    let done = watchdog(360);
    let mut twin = Twin::open(&library).expect("both sides open on one device");
    let mut random = Random(0x11ea);
    let mut tally = Tally::default();
    let probe = DevicePtr::NULL.offset(0x1000);
    for exact in [false, true] {
        library.exact_numerics_for_tests(exact);
        for (in_features, out_features, both_layouts) in SHAPES {
            let layouts: &[Nvfp4Layout] = if both_layouts {
                &[Nvfp4Layout::Plain, Nvfp4Layout::Sd4]
            } else {
                &[Nvfp4Layout::Plain]
            };
            for &layout in layouts {
                let weight = weights(&mut random, in_features, out_features, layout);
                for rows in ROWS {
                    let shape =
                        format!("{layout:?} rows {rows} in {in_features} out {out_features} exact_numerics {exact}");
                    let scratch_bytes =
                        nvfp4_native_scratch_bytes(u64::from(rows), u64::from(in_features), u64::from(out_features));
                    let make = |random: &mut Random, scratch: usize| {
                        vec![
                            random.bf16(rows as usize * in_features as usize),
                            weight.clone(),
                            vec![0; rows as usize * out_features as usize * 2],
                            vec![0; scratch],
                        ]
                    };
                    let weight_bytes = weight.len();

                    if plan_linear(
                        &operands(&[probe; 3], rows, in_features, out_features, weight_bytes),
                        exact,
                    )
                    .is_err()
                    {
                        tally.refused("linear");
                    } else {
                        let case = Case {
                            name: &shape,
                            buffers: make(&mut random, 256),
                            prepare: None,
                            native: Box::new(move |submission, p| {
                                ops::linear_nvfp4_bf16(
                                    submission,
                                    p[0],
                                    p[1],
                                    p[2],
                                    rows,
                                    in_features,
                                    out_features,
                                    weight_bytes as u64,
                                )
                            }),
                            plan: Box::new(move |p| {
                                plan_linear(&operands(p, rows, in_features, out_features, weight_bytes), exact)
                                    .unwrap()
                                    .launches()
                                    .to_vec()
                            }),
                            outputs: vec![2],
                            row_exact: false,
                            tables: Vec::new(),
                            scratch: Vec::new(),
                        };
                        let verdict = twin.run(&case).expect("the case runs");
                        tally.record("linear", &shape, verdict);
                    }

                    if plan_native(
                        &operands(&[probe; 3], rows, in_features, out_features, weight_bytes),
                        probe,
                        scratch_bytes,
                        exact,
                    )
                    .is_err()
                    {
                        tally.refused("native");
                    } else {
                        let case = Case {
                            name: &shape,
                            buffers: make(&mut random, scratch_bytes as usize),
                            prepare: None,
                            native: Box::new(move |submission, p| {
                                ops::linear_nvfp4_native_bf16(
                                    submission,
                                    p[0],
                                    p[1],
                                    p[2],
                                    p[3],
                                    rows,
                                    in_features,
                                    out_features,
                                    weight_bytes as u64,
                                    scratch_bytes,
                                )
                            }),
                            plan: Box::new(move |p| {
                                plan_native(
                                    &operands(p, rows, in_features, out_features, weight_bytes),
                                    p[3],
                                    scratch_bytes,
                                    exact,
                                )
                                .unwrap()
                                .launches()
                                .to_vec()
                            }),
                            outputs: vec![2, 3],
                            row_exact: false,
                            tables: Vec::new(),
                            scratch: Vec::new(),
                        };
                        let verdict = twin.run(&case).expect("the case runs");
                        tally.record("native", &shape, verdict);
                    }
                }
            }
        }
    }
    library.exact_numerics_for_tests(false);
    done.store(true, Ordering::Relaxed);
    tally.finish();
}
