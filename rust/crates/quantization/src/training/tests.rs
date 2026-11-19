//! Host tests of the planners, and a fake device that checks what reaches a lane.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: the fake kernels record their launches behind a lock"
)]

use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use euhedral_gpu::fake::{FakeGpu, FakeModule};
use euhedral_gpu::{Args, GpuError, Lane, Launch, StreamOptions, open_shared_lanes};
use euhedral_graph::LaneId;
use euhedral_inference_cuda::math::Operation;

use super::*;
use crate::kernels::{Module, ModuleOptions, compile};

const BASE: u64 = 0x1000;

fn heads() -> Heads {
    Heads {
        rows: 3,
        q_heads: 4,
        kv_heads: 2,
        head: 64,
        rotary: 16,
        position0: 1000,
        theta: 1e7,
    }
}

#[test]
fn an_elementwise_launch_covers_every_value_and_no_more_blocks() {
    let blocks = |count| plan_copy(BASE, BASE, count).unwrap()[0].launch.grid.x;
    assert_eq!(blocks(1), 1);
    assert_eq!(blocks(256), 1);
    assert_eq!(blocks(257), 2);
    assert_eq!(blocks(100), 1);
    let swiglu = plan_swiglu_forward(BASE, BASE, 4, 40).unwrap();
    assert_eq!((swiglu[0].launch.grid.x, swiglu[0].launch.block.x), (1, 256));
}

#[test]
fn a_row_wise_launch_has_a_block_per_row_or_per_head() {
    let norm = plan_rms_norm_forward(BASE, BASE, BASE, BASE, 5, 96, 1e-6, 1.0).unwrap();
    assert_eq!((norm[0].launch.grid.x, norm[0].launch.block.x), (5, 256));
    let rope = plan_qk_norm_rope_forward(BASE, BASE, BASE, BASE, BASE, &heads(), 1e-6).unwrap();
    assert_eq!((rope[0].launch.grid.x, rope[0].launch.block.x), (3 * 6, 64));
    let back = plan_qk_norm_rope_backward(BASE, BASE, BASE, BASE, BASE, BASE, None, None, &heads()).unwrap();
    assert_eq!((back[0].launch.grid.x, back[0].launch.block.x), (3 * 6, 64));
}

#[test]
fn a_zero_count_or_shape_is_refused_rather_than_launched_empty() {
    assert!(matches!(plan_copy(BASE, BASE, 0), Err(TrainError::OutOfRange { .. })));
    assert!(matches!(
        plan_swiglu_forward(BASE, BASE, 0, 4),
        Err(TrainError::OutOfRange { .. })
    ));
    assert!(matches!(
        plan_rms_norm_forward(BASE, BASE, BASE, BASE, 2, 0, 1e-6, 0.0),
        Err(TrainError::OutOfRange { .. })
    ));
    assert!(matches!(
        plan_rms_norm_forward(BASE, BASE, BASE, BASE, 0, 8, 1e-6, 0.0),
        Err(TrainError::OutOfRange { .. })
    ));
    assert!(matches!(
        plan_adam(BASE, BASE, BASE, BASE, 8, &adam(), 0),
        Err(TrainError::OutOfRange { name: "step", .. })
    ));
}

#[test]
fn a_null_address_is_refused_and_names_the_operand() {
    assert!(matches!(plan_zero(0, 4), Err(TrainError::NullAddress("x"))));
    assert!(matches!(
        plan_gate_backward(BASE, BASE, BASE, 0, BASE, 2, 3, 3),
        Err(TrainError::NullAddress("d_context"))
    ));
    assert!(matches!(
        plan_linear(BASE, 0, BASE, 2, 3, 4, false),
        Err(TrainError::NullAddress("w"))
    ));
}

#[test]
fn a_head_the_block_cannot_hold_or_a_rotation_wider_than_it_is_refused() {
    let plan = |shape: Heads| plan_qk_norm_rope_forward(BASE, BASE, BASE, BASE, BASE, &shape, 1e-6);
    for head in [16, 48, 544, 1024] {
        let shape = Heads {
            head,
            rotary: 16,
            ..heads()
        };
        assert!(matches!(plan(shape), Err(TrainError::HeadDimension(_))), "{head}");
    }
    assert!(
        plan(Heads {
            head: 512,
            rotary: 512,
            ..heads()
        })
        .is_ok()
    );
    assert!(matches!(
        plan(Heads { rotary: 15, ..heads() }),
        Err(TrainError::Rotary { .. })
    ));
    assert!(matches!(
        plan(Heads { rotary: 66, ..heads() }),
        Err(TrainError::Rotary { .. })
    ));
    assert!(plan(Heads { rotary: 0, ..heads() }).is_ok());
}

#[test]
fn a_gate_row_shorter_than_the_width_used_is_refused() {
    assert!(matches!(
        plan_gate_forward(BASE, BASE, BASE, 4, 12, 11),
        Err(TrainError::ShortLeading { .. })
    ));
    assert!(plan_gate_forward(BASE, BASE, BASE, 4, 12, 12).is_ok());
}

#[test]
fn an_absent_weight_gradient_is_a_null_pointer_in_the_kernel() {
    let with = plan_rms_norm_backward(1, 2, 3, 4, 5, Some(6), 2, 8, 1.0).unwrap();
    let without = plan_rms_norm_backward(1, 2, 3, 4, 5, None, 2, 8, 1.0).unwrap();
    assert_eq!(with[0].args.read::<u64>(5), Some(6));
    assert_eq!(without[0].args.read::<u64>(5), Some(0));
    let rope = plan_qk_norm_rope_backward(1, 2, 3, 4, 5, 6, Some(7), None, &heads()).unwrap();
    assert_eq!(rope[0].args.read::<u64>(6), Some(7));
    assert_eq!(rope[0].args.read::<u64>(7), Some(0));
}

fn adam() -> AdamSettings {
    AdamSettings {
        rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        decay: 0.01,
    }
}

#[test]
fn arguments_follow_each_kernels_parameter_order() {
    let rope = plan_qk_norm_rope_forward(1, 2, 3, 4, 5, &heads(), 1e-6).unwrap();
    let args = &rope[0].args;
    assert_eq!(
        (0..5).map(|i| args.read::<u64>(i).unwrap()).collect::<Vec<_>>(),
        [1, 2, 3, 4, 5]
    );
    assert_eq!([5, 6, 7, 8].map(|i| args.read::<i32>(i).unwrap()), [4, 2, 64, 16]);
    assert_eq!(args.read::<i64>(9), Some(1000));
    assert_eq!(args.read::<f64>(10), Some(1e7));
    assert_eq!(args.read::<f32>(11), Some(1e-6));

    let step = plan_adam(1, 2, 3, 4, 100, &adam(), 7).unwrap();
    let args = &step[0].args;
    assert_eq!(args.read::<i64>(4), Some(100));
    assert_eq!(args.read::<f32>(5), Some(1e-3));
    assert_eq!(args.read::<f32>(9), Some(0.01));
    assert_eq!(args.read::<i32>(10), Some(7));

    let fill = plan_fill(1, 9, 2.5).unwrap();
    assert_eq!(fill[0].args.read::<i64>(1), Some(9));
    assert_eq!(fill[0].args.read::<f32>(2), Some(2.5));
}

/// What a product asks of cuBLAS: operations, the column-major shape, and which operand is first.
fn shape(gemm: &Gemm) -> (Operation, Operation, i32, i32, i32) {
    let p = gemm.product();
    (p.a_operation, p.b_operation, p.m, p.n, p.k)
}

#[test]
fn a_row_major_product_is_the_transposed_product_with_its_operands_exchanged() {
    let (rows, inputs, outputs) = (5, 24, 16);
    let linear = plan_linear(10, 20, 30, rows, inputs, outputs, false).unwrap();
    assert_eq!(shape(&linear), (Operation::Transpose, Operation::None, 16, 5, 24));
    assert_eq!(linear.operands(), [(20, 24), (10, 24), (30, 16)]);
    assert_eq!(linear.product().beta, 0.0);
    assert_eq!(
        plan_linear(10, 20, 30, rows, inputs, outputs, true)
            .unwrap()
            .product()
            .beta,
        1.0
    );

    let input = plan_linear_input_gradient(10, 20, 30, rows, inputs, outputs).unwrap();
    assert_eq!(shape(&input), (Operation::None, Operation::None, 24, 5, 16));
    assert_eq!(input.operands(), [(20, 24), (10, 16), (30, 24)]);
    assert_eq!(input.product().beta, 1.0);

    let weight = plan_linear_weight_gradient(10, 20, 30, rows, inputs, outputs).unwrap();
    assert_eq!(shape(&weight), (Operation::None, Operation::Transpose, 24, 16, 5));
    assert_eq!(weight.operands(), [(20, 24), (10, 16), (30, 24)]);
    assert_eq!(weight.product().beta, 0.0);

    let matrices = |a, b| Matrices {
        a: (10, a),
        b: (20, b),
        c: (30, 10),
    };
    let plain = plan_product(&matrices(8, 10), 6, 10, 8, 0.5, 0.0).unwrap();
    assert_eq!(shape(&plain), (Operation::None, Operation::None, 10, 6, 8));
    assert_eq!(plain.operands(), [(20, 10), (10, 8), (30, 10)]);
    let transposed = plan_product_transposed(&matrices(8, 8), 6, 10, 8, 0.5, 0.0).unwrap();
    assert_eq!(shape(&transposed), (Operation::Transpose, Operation::None, 10, 6, 8));
    let left = plan_product_left_transposed(&matrices(6, 10), 6, 10, 8, 1.0, 1.0).unwrap();
    assert_eq!(shape(&left), (Operation::None, Operation::Transpose, 10, 6, 8));
    assert_eq!((left.product().alpha, left.product().beta), (1.0, 1.0));
}

#[test]
fn a_leading_dimension_shorter_than_its_row_is_refused() {
    let matrices = |a, b, c| Matrices {
        a: (10, a),
        b: (20, b),
        c: (30, c),
    };
    assert!(matches!(
        plan_product(&matrices(7, 10, 10), 6, 10, 8, 1.0, 0.0),
        Err(TrainError::ShortLeading { name: "a", .. })
    ));
    assert!(matches!(
        plan_product(&matrices(8, 9, 10), 6, 10, 8, 1.0, 0.0),
        Err(TrainError::ShortLeading { name: "b", .. })
    ));
    assert!(matches!(
        plan_product_transposed(&matrices(8, 7, 10), 6, 10, 8, 1.0, 0.0),
        Err(TrainError::ShortLeading { name: "b", .. })
    ));
    assert!(matches!(
        plan_product_left_transposed(&matrices(5, 10, 10), 6, 10, 8, 1.0, 0.0),
        Err(TrainError::ShortLeading { name: "a", .. })
    ));
    assert!(plan_product(&matrices(11, 12, 13), 6, 10, 8, 1.0, 0.0).is_ok());
    assert!(matches!(
        plan_product(&matrices(8, 10, 10), 0, 10, 8, 1.0, 0.0),
        Err(TrainError::OutOfRange { .. })
    ));
}

/// A fake device whose training kernels record their launches.
struct Rig {
    gpu: Arc<FakeGpu>,
    launched: Arc<Mutex<Vec<(String, Args)>>>,
}

fn rig() -> Rig {
    let gpu = FakeGpu::new();
    let launched = Arc::new(Mutex::new(Vec::new()));
    for name in Module::Train.kernels() {
        let launched = Arc::clone(&launched);
        let name = (*name).to_owned();
        gpu.define(&name.clone(), move |_, _, args| {
            launched.lock().unwrap().push((name.clone(), args.clone()));
            Ok(())
        });
    }
    Rig { gpu, launched }
}

fn run(rig: &Rig, steps: &[Step]) -> Result<(), GpuError> {
    let options = ModuleOptions::new(PathBuf::from("/cuda/include"));
    let compiled = compile::<FakeModule>(&rig.gpu, &options, Module::Train).unwrap();
    let kernels = TrainKernels::new(&compiled);
    let lanes = open_shared_lanes(&rig.gpu, &[("test", StreamOptions::default())]).unwrap();
    let lane = Arc::clone(lanes.pool.lane(LaneId(0)));
    lane.submit(|queue| {
        // SAFETY: the fake kernels read no memory.
        unsafe { kernels.enqueue(queue, steps) }.map_err(|error| match error {
            TrainError::Gpu(text) => GpuError::Invalid(text),
            other => GpuError::Invalid(other.to_string()),
        })
    })?;
    while rig.gpu.step() {}
    Ok(())
}

#[test]
fn planned_steps_reach_the_lane_in_order() {
    let rig = rig();
    let mut steps = plan_zero(BASE, 10).unwrap();
    steps.extend(plan_swiglu_forward(BASE, BASE, 2, 3).unwrap());
    steps.extend(plan_adam(BASE, BASE, BASE, BASE, 6, &adam(), 1).unwrap());
    run(&rig, &steps).unwrap();
    let launched = rig.launched.lock().unwrap();
    let names: Vec<&str> = launched.iter().map(|(name, _)| name.as_str()).collect();
    assert_eq!(names, ["fill", "swiglu_forward", "adam"]);
    assert_eq!(launched[0].1.read::<i64>(1), Some(10));
}

#[test]
fn a_step_for_a_kernel_the_table_lacks_is_refused_before_any_launch() {
    let rig = rig();
    let mut steps = plan_zero(BASE, 10).unwrap();
    steps.push(Step {
        kernel: "no_such_kernel",
        launch: Launch::linear(1, 32),
        args: Args::new(),
    });
    assert!(run(&rig, &steps).is_err());
    assert!(rig.launched.lock().unwrap().is_empty());
}

#[test]
fn every_listed_kernel_has_a_planner_that_names_it() {
    let launched = [
        plan_adam(1, 1, 1, 1, 1, &adam(), 1),
        plan_bf16_to_float(1, 1, 1),
        plan_copy(1, 1, 1),
        plan_fill(1, 1, 0.0),
        plan_gate_forward(1, 1, 1, 1, 1, 1),
        plan_gate_backward(1, 1, 1, 1, 1, 1, 1, 1),
        plan_qk_norm_rope_forward(1, 1, 1, 1, 1, &heads(), 1e-6),
        plan_qk_norm_rope_backward(1, 1, 1, 1, 1, 1, None, None, &heads()),
        plan_rms_norm_forward(1, 1, 1, 1, 1, 1, 1e-6, 0.0),
        plan_rms_norm_backward(1, 1, 1, 1, 1, None, 1, 1, 0.0),
        plan_swiglu_forward(1, 1, 1, 1),
        plan_swiglu_backward(1, 1, 1, 1, 1),
    ]
    .map(|plan| plan.unwrap()[0].kernel);
    let mut listed = Module::Train.kernels().to_vec();
    let mut named = launched.to_vec();
    listed.sort_unstable();
    named.sort_unstable();
    assert_eq!(named, listed);
}
