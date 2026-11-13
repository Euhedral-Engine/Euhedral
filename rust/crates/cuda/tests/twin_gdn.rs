// Temporary: removed together with the native launch path, once the kernels run through RustEE.
//! Device test: every Gated DeltaNet planner launches, through RustEE, what the native wrapper launches, bit for
//! bit, including a stage that turns row-exact on for its replay and back off. `#[ignore]`d like the crate's other
//! device tests:
//! `cargo test -p euhedral-inference-cuda --test twin_gdn -- --ignored --test-threads=1 --nocapture`.
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
use euhedral_inference_cuda::launch::gdn::{
    ConvolutionShape, GatedRmsNormShape, ProjectControlShape, RecurrenceShape, plan_convolution, plan_gated_rms_norm,
    plan_project_control, plan_recurrence,
};
use euhedral_inference_cuda::launch::load::{KernelSources, Slot, Unavailable};
use euhedral_inference_cuda::launch::{Modes, PlanError};
use euhedral_inference_cuda::{CudaError, CudaLibrary, DevicePtr, Submission, ops};

use common::twin::{Case, Rng, Tally, Twin};

const ROWS: [u32; 17] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 16, 32, 33, 48, 64, 65, 128, 512];
const EPSILON: f32 = 1e-6;
const HEAD_DIM: u32 = 128;
/// 1 / sqrt(128), the recurrence's output scale.
const OUTPUT_SCALE: f32 = 0.088_388_35;

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

/// The shapes of the stage's launches, by the buffers' addresses.
#[derive(Clone, Copy)]
struct Heads {
    key: u32,
    value: u32,
}

impl Heads {
    fn query_key_width(self) -> u32 {
        2 * self.key * HEAD_DIM
    }
    fn value_width(self) -> u32 {
        self.value * HEAD_DIM
    }
    fn channels(self) -> u32 {
        self.query_key_width() + self.value_width()
    }
}

fn convolution(p: [DevicePtr; 5], rows: u32, widths: (u32, u32), kernel_size: u32) -> Vec<Launch> {
    plan(|mut sink| {
        plan_convolution(
            &ConvolutionShape {
                query_key: p[0],
                value_z: p[1],
                weights: p[2],
                state: p[3],
                output: p[4],
                rows,
                query_key_width: widths.0,
                value_width: widths.1,
                convolution_width: widths.0 + widths.1,
                kernel_size,
            },
            &mut sink,
        )
    })
}

fn recurrence(p: [DevicePtr; 5], rows: u32, heads: Heads, modes: Modes) -> Vec<Launch> {
    plan(|mut sink| {
        plan_recurrence(
            &RecurrenceShape {
                convolved: p[0],
                alpha: p[1],
                beta: p[2],
                recurrent_state: p[3],
                output: p[4],
                rows,
                key_heads: heads.key,
                value_heads: heads.value,
                key_head_dim: HEAD_DIM,
                value_head_dim: HEAD_DIM,
                output_scale: OUTPUT_SCALE,
            },
            modes,
            &mut sink,
        )
    })
}

fn gated_norm(p: [DevicePtr; 4], rows: u32, value_heads: u32) -> Vec<Launch> {
    plan(|mut sink| {
        plan_gated_rms_norm(
            &GatedRmsNormShape {
                recurrent: p[0],
                value_z: p[1],
                norm_weight: p[2],
                output: p[3],
                rows,
                value_heads,
                head_dim: HEAD_DIM,
                epsilon: EPSILON,
            },
            &mut sink,
        )
    })
}

#[test]
#[ignore = "needs a GPU, the native library and NVRTC"]
fn gdn_outputs_equal_the_native_library_bit_for_bit() {
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
        "euhedral_gdn_project_control_fp32",
        "euhedral_gdn_project_control_8x4_fp32",
        "euhedral_gdn_convolution_bf16",
        "euhedral_gdn_recurrence_bf16",
        "euhedral_gdn_recurrence_c8_bf16",
        "euhedral_gdn_gated_rms_norm_bf16",
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
    let mut rng = Rng(0x0dd5_7a6e_0001_0001);

    // Project control: below and from 48 rows, heads that divide by 4 and not, widths either side of the batch.
    for rows in ROWS {
        for width in [100u32, 128, 1300, 2048] {
            for heads in [1u32, 4, 6, 16] {
                let buffers = vec![
                    rng.bf16_bytes((rows * width) as usize),
                    rng.bf16_bytes((heads * width) as usize),
                    rng.bf16_bytes((heads * width) as usize),
                    rng.f32_bytes(heads as usize, -2.0, 1.0),
                    rng.f32_bytes(heads as usize, -2.0, 2.0),
                    vec![0; (rows * heads) as usize * 4],
                    vec![0; (rows * heads) as usize * 4],
                ];
                let case = Case {
                    name: "project control",
                    buffers,
                    prepare: None,
                    native: Box::new(move |submission, p| {
                        ops::gdn_project_control_fp32(
                            submission, p[0], p[1], p[2], p[3], p[4], p[5], p[6], rows, width, heads,
                        )
                    }),
                    plan: Box::new(move |p| {
                        plan(|mut sink| {
                            plan_project_control(
                                &ProjectControlShape {
                                    input: p[0],
                                    a_weight: p[1],
                                    b_weight: p[2],
                                    a_log: p[3],
                                    dt_bias: p[4],
                                    alpha: p[5],
                                    beta: p[6],
                                    rows,
                                    width,
                                    heads,
                                },
                                &mut sink,
                            )
                        })
                    }),
                    outputs: vec![5, 6],
                };
                tally.run(
                    &mut twin,
                    "project_control",
                    &format!("rows={rows} width={width} heads={heads}"),
                    &case,
                );
            }
        }
    }

    // Convolution: channel counts that fill the 128-channel blocks and not, every window size class.
    for rows in ROWS {
        for widths in [(128u32, 128u32), (100, 60), (64, 200)] {
            for kernel_size in [2u32, 3, 4, 5, 32] {
                let channels = widths.0 + widths.1;
                let history = (kernel_size - 1) as usize;
                let buffers = vec![
                    rng.bf16_bytes((rows * widths.0) as usize),
                    rng.bf16_bytes((rows * widths.1 * 2) as usize),
                    rng.bf16_bytes(kernel_size as usize * channels as usize),
                    rng.bf16_bytes(channels as usize * history),
                    vec![0; (rows * channels) as usize * 2],
                ];
                let case = Case {
                    name: "convolution",
                    buffers,
                    prepare: None,
                    native: Box::new(move |submission, p| {
                        ops::gdn_convolution_bf16(
                            submission,
                            p[0],
                            p[1],
                            p[2],
                            p[3],
                            p[4],
                            rows,
                            widths.0,
                            widths.1,
                            channels,
                            kernel_size,
                        )
                    }),
                    plan: Box::new(move |p| convolution([p[0], p[1], p[2], p[3], p[4]], rows, widths, kernel_size)),
                    outputs: vec![3, 4],
                };
                let shape = format!("rows={rows} widths={widths:?} kernel={kernel_size}");
                tally.run(&mut twin, "convolution", &shape, &case);
            }
        }
    }

    // Recurrence, in every mode.
    let head_sets = [(1u32, 1u32), (2, 2), (2, 4), (4, 8), (16, 48)];
    for modes in modes_sweep() {
        library.exact_numerics_for_tests(modes.exact_numerics);
        for rows in ROWS {
            for (key, value) in head_sets {
                let heads = Heads { key, value };
                let buffers = recurrence_buffers(&mut rng, rows, heads);
                let case = Case {
                    name: "recurrence",
                    buffers,
                    prepare: None,
                    native: Box::new(move |submission, p| {
                        ops::select_row_exact(submission, modes.row_exact);
                        native_recurrence(submission, [p[0], p[1], p[2], p[3], p[4]], rows, heads)
                    }),
                    plan: Box::new(move |p| recurrence([p[0], p[1], p[2], p[3], p[4]], rows, heads, modes)),
                    outputs: vec![3, 4],
                };
                let shape = format!(
                    "rows={rows} heads={key}/{value} row_exact={} exact_numerics={}",
                    modes.row_exact, modes.exact_numerics
                );
                tally.run(&mut twin, "recurrence", &shape, &case);
            }
        }
    }
    library.exact_numerics_for_tests(false);

    // Gated norm.
    for rows in ROWS {
        for value_heads in [1u32, 3, 32] {
            let width = (value_heads * HEAD_DIM) as usize;
            let buffers = vec![
                rng.bf16_bytes(rows as usize * width),
                rng.bf16_bytes(rows as usize * width * 2),
                rng.bf16_bytes(HEAD_DIM as usize),
                vec![0; rows as usize * width * 2],
            ];
            let case = Case {
                name: "gated rms norm",
                buffers,
                prepare: None,
                native: Box::new(move |submission, p| {
                    ops::gdn_gated_rms_norm_bf16(
                        submission,
                        p[0],
                        p[1],
                        p[2],
                        p[3],
                        rows,
                        value_heads,
                        HEAD_DIM,
                        EPSILON,
                    )
                }),
                plan: Box::new(move |p| gated_norm([p[0], p[1], p[2], p[3]], rows, value_heads)),
                outputs: vec![3],
            };
            tally.run(
                &mut twin,
                "gated_rms_norm",
                &format!("rows={rows} value_heads={value_heads}"),
                &case,
            );
        }
    }

    // The speculative stage: the quantum's mode, a replay of the same rows row-exact from a copy of the same state,
    // then the quantum's mode again for the norm.
    let heads = Heads { key: 2, value: 4 };
    let (widths, kernel_size) = ((heads.query_key_width(), heads.value_width()), 4u32);
    let channels = heads.channels();
    for modes in modes_sweep() {
        library.exact_numerics_for_tests(modes.exact_numerics);
        for rows in ROWS {
            let history = (kernel_size - 1) as usize;
            let recurrent = recurrence_buffers(&mut rng, rows, heads);
            let value_z = rng.bf16_bytes(rows as usize * widths.1 as usize * 2);
            let conv_state = rng.bf16_bytes(channels as usize * history);
            let out_bytes = rows as usize * channels as usize * 2;
            let recurrent_out = rows as usize * widths.1 as usize * 2;
            // 0 query/key, 1 value/z, 2 weights, 3 and 4 convolution state, 5 and 6 convolved,
            // 7 alpha, 8 beta, 9 and 10 recurrent state, 11 and 12 recurrent out, 13 norm weight, 14 norm out.
            let buffers = vec![
                rng.bf16_bytes(rows as usize * widths.0 as usize),
                value_z,
                rng.bf16_bytes(kernel_size as usize * channels as usize),
                conv_state.clone(),
                conv_state,
                vec![0; out_bytes],
                vec![0; out_bytes],
                recurrent[1].clone(),
                recurrent[2].clone(),
                recurrent[3].clone(),
                recurrent[3].clone(),
                vec![0; recurrent_out],
                vec![0; recurrent_out],
                rng.bf16_bytes(HEAD_DIM as usize),
                vec![0; recurrent_out],
            ];
            let replay = Modes {
                row_exact: true,
                ..modes
            };
            let case = Case {
                name: "stage",
                buffers,
                prepare: None,
                native: Box::new(move |submission, p| {
                    ops::select_row_exact(submission, modes.row_exact);
                    ops::gdn_convolution_bf16(
                        submission,
                        p[0],
                        p[1],
                        p[2],
                        p[3],
                        p[5],
                        rows,
                        widths.0,
                        widths.1,
                        channels,
                        kernel_size,
                    )?;
                    native_recurrence(submission, [p[5], p[7], p[8], p[9], p[11]], rows, heads)?;
                    ops::select_row_exact(submission, true);
                    ops::gdn_convolution_bf16(
                        submission,
                        p[0],
                        p[1],
                        p[2],
                        p[4],
                        p[6],
                        rows,
                        widths.0,
                        widths.1,
                        channels,
                        kernel_size,
                    )?;
                    native_recurrence(submission, [p[6], p[7], p[8], p[10], p[12]], rows, heads)?;
                    ops::select_row_exact(submission, modes.row_exact);
                    ops::gdn_gated_rms_norm_bf16(
                        submission,
                        p[12],
                        p[1],
                        p[13],
                        p[14],
                        rows,
                        heads.value,
                        HEAD_DIM,
                        EPSILON,
                    )
                }),
                plan: Box::new(move |p| {
                    let mut launches = convolution([p[0], p[1], p[2], p[3], p[5]], rows, widths, kernel_size);
                    launches.extend(recurrence([p[5], p[7], p[8], p[9], p[11]], rows, heads, modes));
                    launches.extend(convolution([p[0], p[1], p[2], p[4], p[6]], rows, widths, kernel_size));
                    launches.extend(recurrence([p[6], p[7], p[8], p[10], p[12]], rows, heads, replay));
                    launches.extend(gated_norm([p[12], p[1], p[13], p[14]], rows, heads.value));
                    launches
                }),
                outputs: vec![3, 4, 5, 6, 9, 10, 11, 12, 14],
            };
            let shape = format!(
                "rows={rows} quantum row_exact={} exact_numerics={}",
                modes.row_exact, modes.exact_numerics
            );
            tally.run(&mut twin, "replay_stage", &shape, &case);
        }
    }
    library.exact_numerics_for_tests(false);
    tally.finish();
}

/// The recurrence's buffers: convolved rows, alpha, beta, state, output.
fn recurrence_buffers(rng: &mut Rng, rows: u32, heads: Heads) -> Vec<Vec<u8>> {
    let gates = (rows * heads.value) as usize;
    vec![
        rng.bf16_bytes(rows as usize * heads.channels() as usize),
        rng.f32_bytes(gates, 0.05, 0.95),
        rng.f32_bytes(gates, 0.05, 0.95),
        rng.f32_bytes((heads.value * HEAD_DIM * HEAD_DIM) as usize, -0.5, 0.5),
        vec![0; rows as usize * heads.value_width() as usize * 2],
    ]
}

fn native_recurrence(submission: &Submission<'_>, p: [DevicePtr; 5], rows: u32, heads: Heads) -> Result<(), CudaError> {
    ops::gdn_recurrence_bf16(
        submission,
        p[0],
        p[1],
        p[2],
        p[3],
        p[4],
        rows,
        heads.key,
        heads.value,
        HEAD_DIM,
        HEAD_DIM,
        OUTPUT_SCALE,
    )
}
