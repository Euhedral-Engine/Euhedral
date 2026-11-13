// Temporary: removed together with the native launch path, once the kernels run through RustEE.
//! Device test: every elementwise, norm and sampling planner launches, through RustEE, what the native wrapper
//! launches, bit for bit. `#[ignore]`d like the crate's other device tests:
//! `cargo test -p euhedral-inference-cuda --test twin_elementwise -- --ignored --test-threads=1 --nocapture`.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "the counts are the test's output")]

mod common;

use std::path::PathBuf;
use std::sync::Arc;

use euhedral_inference_cuda::launch::catalog::{Launch, lookup};
use euhedral_inference_cuda::launch::elementwise::{
    ArgmaxShape, ResidualAddShape, ResidualRmsNormShape, RmsNormShape, SelectedTokenShape, SwigluShape, plan_argmax,
    plan_argmax_logprob, plan_residual_add, plan_residual_rms_norm, plan_rms_norm, plan_rms_norm_unit_offset,
    plan_selected_token, plan_swiglu,
};
use euhedral_inference_cuda::launch::load::{KernelSources, Slot, Unavailable};
use euhedral_inference_cuda::launch::{Modes, PlanError};
use euhedral_inference_cuda::{CudaLibrary, DevicePtr, ops};

use common::twin::{Case, Rng, Tally, Twin};

const ROWS: [u32; 17] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 32, 33, 48, 64, 65, 128, 512];
/// Multiples of the kernels' vector widths (8) and not, small and large, either side of 8192.
const WIDTHS: [u32; 11] = [1, 7, 8, 16, 24, 100, 128, 1000, 4096, 8192, 8200];
const EPSILON: f32 = 1e-6;

fn library() -> Option<Arc<CudaLibrary>> {
    let path = test_support::require_property("euhedral.cuda.library")?;
    Some(CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads"))
}

fn plan(planned: impl FnOnce(&mut dyn FnMut(Launch)) -> Result<(), PlanError>) -> Vec<Launch> {
    let mut launches = Vec::new();
    planned(&mut |launch| launches.push(launch)).expect("the shape plans");
    launches
}

fn modes_sweep() -> [Modes; 4] {
    [(false, false), (true, false), (false, true), (true, true)].map(|(row_exact, exact_numerics)| Modes {
        row_exact,
        exact_numerics,
    })
}

#[test]
#[ignore = "needs a GPU, the native library and NVRTC"]
fn elementwise_outputs_equal_the_native_library_bit_for_bit() {
    let Some(library) = library() else { return };
    if KernelSources::beside(&library)
        .expect("the product directory")
        .compiler()
        .is_none()
    {
        println!("skipped: no bundled NVRTC beside the library");
        return;
    }
    let mut twin = Twin::open(&library).expect("both sides open on one device");
    for name in [
        "euhedral_residual_add_bf16",
        "euhedral_swiglu_bf16",
        "euhedral_residual_rms_norm_bf16",
        "euhedral_residual_rms_norm_row_bf16",
        "euhedral_rms_norm_bf16",
        "euhedral_argmax_bf16",
        "euhedral_argmax_logprob_bf16",
        "euhedral_selected_token",
    ] {
        let id = lookup(name).expect("a catalog kernel");
        if let Slot::Unavailable(reason) = twin.table().get(id) {
            assert!(
                matches!(reason, Unavailable::ArchitectureSpecificTarget),
                "{name}: {reason}"
            );
        }
    }
    let mut tally = Tally::default();
    let mut rng = Rng(0x00e1_e3e1_7415_e000);
    let mut seen = 0u32;

    // Residual add and swiglu take no modes.
    for rows in ROWS {
        for width in WIDTHS {
            let count = rows as usize * width as usize;
            if count > 1 << 21 {
                continue;
            }
            let shape = format!("rows={rows} width={width}");
            let buffers = vec![rng.bf16_bytes(count), rng.bf16_bytes(count), vec![0; count * 2]];
            let case = Case {
                row_exact: false,
                tables: Vec::new(),
                scratch: Vec::new(),
                name: "residual add",
                buffers,
                prepare: None,
                native: Box::new(move |submission, p| {
                    ops::residual_add_bf16(submission, p[0], p[1], p[2], rows, width)
                }),
                plan: Box::new(move |p| {
                    plan(|mut sink| {
                        plan_residual_add(
                            &ResidualAddShape {
                                residual: p[0],
                                delta: p[1],
                                output: p[2],
                                rows,
                                width,
                            },
                            &mut sink,
                        )
                    })
                }),
                outputs: vec![2],
            };
            tally.run(&mut twin, "residual_add", &shape, &case);

            let buffers = vec![rng.bf16_bytes(count * 2), vec![0; count * 2]];
            let case = Case {
                row_exact: false,
                tables: Vec::new(),
                scratch: Vec::new(),
                name: "swiglu",
                buffers,
                prepare: None,
                native: Box::new(move |submission, p| ops::swiglu_bf16(submission, p[0], p[1], rows, width)),
                plan: Box::new(move |p| {
                    plan(|mut sink| {
                        plan_swiglu(
                            &SwigluShape {
                                gate_up: p[0],
                                output: p[1],
                                rows,
                                intermediate_size: width,
                            },
                            &mut sink,
                        )
                    })
                }),
                outputs: vec![1],
            };
            tally.run(&mut twin, "swiglu", &shape, &case);
        }
    }

    // Norms: the modes matter to the residual norm; the plain norms take none, but run in every mode so the exact
    // numerics switch is shown to leave them alone.
    for modes in modes_sweep() {
        library.exact_numerics_for_tests(modes.exact_numerics);
        for rows in ROWS {
            for width in WIDTHS {
                let count = rows as usize * width as usize;
                if count > 1 << 21 {
                    continue;
                }
                let shape = format!(
                    "rows={rows} width={width} row_exact={} exact_numerics={}",
                    modes.row_exact, modes.exact_numerics
                );
                let buffers = vec![
                    rng.bf16_bytes(count),
                    rng.bf16_bytes(count),
                    rng.bf16_bytes(width as usize),
                    vec![0; count * 2],
                    vec![0; count * 2],
                ];
                let case = Case {
                    row_exact: false,
                    tables: Vec::new(),
                    scratch: Vec::new(),
                    name: "residual rms norm",
                    buffers,
                    prepare: None,
                    native: Box::new(move |submission, p| {
                        ops::select_row_exact(submission, modes.row_exact);
                        ops::residual_rms_norm_bf16(submission, p[0], p[1], p[2], p[3], p[4], rows, width, EPSILON)
                    }),
                    plan: Box::new(move |p| {
                        plan(|mut sink| {
                            plan_residual_rms_norm(
                                &ResidualRmsNormShape {
                                    residual: p[0],
                                    delta: p[1],
                                    weight: p[2],
                                    hidden: p[3],
                                    normalized: p[4],
                                    rows,
                                    width,
                                    epsilon: EPSILON,
                                },
                                modes,
                                &mut sink,
                            )
                        })
                    }),
                    outputs: vec![3, 4],
                };
                tally.run(&mut twin, "residual_rms_norm", &shape, &case);
                seen += 1;

                if modes.row_exact {
                    continue;
                }
                for unit_offset in [false, true] {
                    let buffers = vec![
                        rng.bf16_bytes(count),
                        rng.bf16_bytes(width as usize),
                        vec![0; count * 2],
                    ];
                    let case = Case {
                        row_exact: false,
                        tables: Vec::new(),
                        scratch: Vec::new(),
                        name: "rms norm",
                        buffers,
                        prepare: None,
                        native: Box::new(move |submission, p| {
                            if unit_offset {
                                ops::rms_norm_unit_offset_bf16(submission, p[0], p[1], p[2], rows, width, EPSILON)
                            } else {
                                ops::rms_norm_bf16(submission, p[0], p[1], p[2], rows, width, EPSILON)
                            }
                        }),
                        plan: Box::new(move |p| {
                            let norm = RmsNormShape {
                                input: p[0],
                                weight: p[1],
                                output: p[2],
                                rows,
                                width,
                                epsilon: EPSILON,
                            };
                            plan(|mut sink| {
                                if unit_offset {
                                    plan_rms_norm_unit_offset(&norm, &mut sink)
                                } else {
                                    plan_rms_norm(&norm, &mut sink)
                                }
                            })
                        }),
                        outputs: vec![2],
                    };
                    let planner = if unit_offset {
                        "rms_norm_unit_offset"
                    } else {
                        "rms_norm"
                    };
                    tally.run(&mut twin, planner, &shape, &case);
                }
            }
        }
    }
    library.exact_numerics_for_tests(false);
    assert!(seen > 0);

    // Sampling: logits with many ties (two values) and with a spread; the key, then the token it chose.
    let counts = [1u32, 2, 31, 32, 33, 1000, 1024, 1025, 4096, 32768, 151_936];
    for count in counts {
        for tied in [false, true] {
            let logits = if tied {
                (0..count)
                    .flat_map(|_| [0x80u8, if rng.next() & 1 == 0 { 0x3f } else { 0x40 }])
                    .collect()
            } else {
                rng.bf16_bytes(count as usize)
            };
            let shape = format!("count={count} tied={tied}");
            for with_logprob in [false, true] {
                let case = Case {
                    row_exact: false,
                    tables: Vec::new(),
                    scratch: Vec::new(),
                    name: "argmax",
                    buffers: vec![logits.clone(), vec![0; 16]],
                    prepare: None,
                    native: Box::new(move |submission, p| {
                        if with_logprob {
                            ops::argmax_logprob_bf16(submission, p[0], count, p[1])
                        } else {
                            ops::argmax_bf16(submission, p[0], count, p[1])
                        }
                    }),
                    plan: Box::new(move |p| {
                        let argmax = ArgmaxShape {
                            logits: p[0],
                            count,
                            result: p[1],
                        };
                        plan(|mut sink| {
                            if with_logprob {
                                plan_argmax_logprob(&argmax, &mut sink)
                            } else {
                                plan_argmax(&argmax, &mut sink)
                            }
                        })
                    }),
                    outputs: vec![1],
                };
                let planner = if with_logprob { "argmax_logprob" } else { "argmax" };
                tally.run(&mut twin, planner, &shape, &case);
            }
            for with_ids in [false, true] {
                let ids: Vec<u8> = (0..count)
                    .flat_map(|_| ((rng.next() >> 40) as i32).to_le_bytes())
                    .collect();
                let case = Case {
                    row_exact: false,
                    tables: Vec::new(),
                    scratch: Vec::new(),
                    name: "selected token",
                    buffers: vec![logits.clone(), vec![0; 16], ids, vec![0xa5; 4]],
                    prepare: Some(Box::new(move |submission, p| {
                        ops::argmax_bf16(submission, p[0], count, p[1])
                    })),
                    native: Box::new(move |submission, p| {
                        let ids = if with_ids { p[2] } else { DevicePtr::NULL };
                        ops::selected_token(submission, p[1], ids, p[3])
                    }),
                    plan: Box::new(move |p| {
                        let ids = if with_ids { p[2] } else { DevicePtr::NULL };
                        plan(|mut sink| {
                            plan_selected_token(
                                &SelectedTokenShape {
                                    key: p[1],
                                    ids,
                                    token: p[3],
                                },
                                &mut sink,
                            )
                        })
                    }),
                    outputs: vec![3],
                };
                tally.run(&mut twin, "selected_token", &format!("{shape} ids={with_ids}"), &case);
            }
        }
    }
    tally.finish();
}
