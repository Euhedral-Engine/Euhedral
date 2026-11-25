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

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use euhedral_gpu::fake::{FakeGpu, FakeModule};
use euhedral_gpu::{Args, Lane, Launch, StreamOptions, open_shared_lanes};
use euhedral_graph::LaneId;
use euhedral_inference_cuda::math::Operation;

use super::*;
use crate::kernels::{Module, ModuleOptions, compile};
use crate::trellis::plan_fill_floats;

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
    let blocks = |count| plan_copy(BASE, BASE, count).unwrap()[0].launch().grid.x;
    assert_eq!(blocks(1), 1);
    assert_eq!(blocks(256), 1);
    assert_eq!(blocks(257), 2);
    assert_eq!(blocks(100), 1);
    let swiglu = plan_swiglu_forward(BASE, BASE, 4, 40).unwrap();
    assert_eq!((swiglu[0].launch().grid.x, swiglu[0].launch().block.x), (1, 256));
}

#[test]
fn a_row_wise_launch_has_a_block_per_row_or_per_head() {
    let norm = plan_rms_norm_forward(BASE, BASE, BASE, BASE, 5, 96, 1e-6, 1.0).unwrap();
    assert_eq!((norm[0].launch().grid.x, norm[0].launch().block.x), (5, 256));
    let rope = plan_qk_norm_rope_forward(BASE, BASE, BASE, BASE, BASE, &heads(), 1e-6).unwrap();
    assert_eq!((rope[0].launch().grid.x, rope[0].launch().block.x), (3 * 6, 64));
    let back = plan_qk_norm_rope_backward(BASE, BASE, BASE, BASE, BASE, BASE, None, None, &heads()).unwrap();
    assert_eq!((back[0].launch().grid.x, back[0].launch().block.x), (3 * 6, 64));
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
fn query_and_key_heads_must_both_be_present() {
    let plan = |shape: Heads| plan_qk_norm_rope_forward(BASE, BASE, BASE, BASE, BASE, &shape, 1e-6);
    for (q_heads, kv_heads, name) in [(0, 2, "q_heads"), (4, 0, "kv_heads")] {
        let shape = Heads {
            q_heads,
            kv_heads,
            ..heads()
        };
        assert!(
            matches!(plan(shape), Err(TrainError::OutOfRange { name: refused, value: 0 }) if refused == name),
            "{name}"
        );
    }
}

#[test]
fn a_refused_size_is_reported_as_it_was_asked_for() {
    let count = 1u64 << 50;
    assert!(matches!(
        plan_copy(BASE, BASE, count),
        Err(TrainError::OutOfRange { name: "count", value }) if value == count as i64
    ));
    assert!(matches!(
        plan_copy(BASE, BASE, 0),
        Err(TrainError::OutOfRange {
            name: "count",
            value: 0
        })
    ));
    let rows = 1u32 << 30;
    let shape = Heads {
        rows,
        q_heads: 4,
        kv_heads: 2,
        head: 32,
        ..heads()
    };
    assert!(matches!(
        plan_qk_norm_rope_forward(BASE, BASE, BASE, BASE, BASE, &shape, 1e-6),
        Err(TrainError::OutOfRange { name: "rows", value }) if value == i64::from(rows) * 6
    ));
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
    assert_eq!(with[0].args().read::<u64>(5), Some(6));
    assert_eq!(without[0].args().read::<u64>(5), Some(0));
    let rope = plan_qk_norm_rope_backward(1, 2, 3, 4, 5, 6, Some(7), None, &heads()).unwrap();
    assert_eq!(rope[0].args().read::<u64>(6), Some(7));
    assert_eq!(rope[0].args().read::<u64>(7), Some(0));
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

/// A scalar parameter's value as the test planned it, to be found in the slot the source gives that name.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Scalar {
    Int(i32),
    Long(i64),
    Float(f32),
    Double(f64),
}

/// A planned kernel launch and the values its scalar parameters were given, by parameter name.
type Case<'a> = (Vec<Step>, Vec<(&'a str, Scalar)>);

/// A parameter of a kernel as `train.cu` declares it.
#[derive(Debug)]
struct Parameter {
    name: String,
    pointer: bool,
    bytes: usize,
    floating: bool,
}

/// The parameter lists of the module's `extern "C" __global__` kernels, read from the source the crate loads.
fn declared(source: &str) -> HashMap<String, Vec<Parameter>> {
    const MARK: &str = "extern \"C\" __global__ void ";
    let mut kernels = HashMap::new();
    for part in source.split(MARK).skip(1) {
        let (name, rest) = part.split_once('(').unwrap();
        let (list, _) = rest.split_once(')').unwrap();
        let parameters = list
            .split(',')
            .map(|text| {
                let text = text.trim();
                let cut = text.rfind(|c: char| !(c.is_alphanumeric() || c == '_')).unwrap();
                let (kind, name) = text.split_at(cut + 1);
                let pointer = kind.contains('*');
                let words: Vec<&str> = kind.trim_end_matches(['*', ' ']).split_whitespace().collect();
                let bytes = match (pointer, words.last().copied(), words.contains(&"long")) {
                    (true, ..) | (_, Some("double"), _) | (_, _, true) => 8,
                    (_, Some("int" | "float"), _) => 4,
                    other => panic!("a parameter type this test does not know: {other:?} in {text}"),
                };
                Parameter {
                    name: name.to_owned(),
                    pointer,
                    bytes,
                    floating: matches!(words.last().copied(), Some("float" | "double")) && !pointer,
                }
            })
            .collect();
        kernels.insert(name.trim().to_owned(), parameters);
    }
    kernels
}

fn read_scalar(args: &Args, index: usize, parameter: &Parameter) -> Scalar {
    let value = match (parameter.floating, parameter.bytes) {
        (true, 4) => args.read::<f32>(index).map(Scalar::Float),
        (true, _) => args.read::<f64>(index).map(Scalar::Double),
        (false, 4) => args.read::<i32>(index).map(Scalar::Int),
        (false, _) => args.read::<i64>(index).map(Scalar::Long),
    };
    value.unwrap()
}

#[test]
fn arguments_match_the_parameter_lists_the_loaded_source_declares() {
    let declared = declared(Module::Train.source());
    let settings = AdamSettings {
        rate: 0.0123,
        beta1: 0.91,
        beta2: 0.9991,
        eps: 1.5e-8,
        decay: 0.0107,
    };
    let shape = heads();
    // Every call gives each scalar its own value, and the expectation names the parameter that value is for.
    let cases: Vec<Case> = vec![
        (
            plan_adam(1, 2, 3, 4, 100, &settings, 7).unwrap(),
            vec![
                ("count", Scalar::Long(100)),
                ("lr", Scalar::Float(0.0123)),
                ("beta1", Scalar::Float(0.91)),
                ("beta2", Scalar::Float(0.9991)),
                ("eps", Scalar::Float(1.5e-8)),
                ("decay", Scalar::Float(0.0107)),
                ("t", Scalar::Int(7)),
            ],
        ),
        (plan_bf16_to_float(1, 33, 2).unwrap(), vec![("count", Scalar::Long(33))]),
        (plan_copy(1, 2, 34).unwrap(), vec![("count", Scalar::Long(34))]),
        (
            plan_fill(1, 9, 2.5).unwrap(),
            vec![("count", Scalar::Long(9)), ("value", Scalar::Float(2.5))],
        ),
        (
            plan_gate_forward(1, 2, 3, 4, 5, 6).unwrap(),
            vec![
                ("rows", Scalar::Int(4)),
                ("width", Scalar::Int(5)),
                ("rowWidth", Scalar::Int(6)),
            ],
        ),
        (
            plan_gate_backward(1, 2, 3, 4, 5, 4, 5, 6).unwrap(),
            vec![
                ("rows", Scalar::Int(4)),
                ("width", Scalar::Int(5)),
                ("rowWidth", Scalar::Int(6)),
            ],
        ),
        (
            plan_qk_norm_rope_forward(1, 2, 3, 4, 5, &shape, 1.5e-6).unwrap(),
            vec![
                ("qHeads", Scalar::Int(4)),
                ("kvHeads", Scalar::Int(2)),
                ("d", Scalar::Int(64)),
                ("rotary", Scalar::Int(16)),
                ("position0", Scalar::Long(1000)),
                ("theta", Scalar::Double(1e7)),
                ("eps", Scalar::Float(1.5e-6)),
            ],
        ),
        (
            plan_qk_norm_rope_backward(1, 2, 3, 4, 5, 6, Some(7), Some(8), &shape).unwrap(),
            vec![
                ("qHeads", Scalar::Int(4)),
                ("kvHeads", Scalar::Int(2)),
                ("d", Scalar::Int(64)),
                ("rotary", Scalar::Int(16)),
                ("position0", Scalar::Long(1000)),
                ("theta", Scalar::Double(1e7)),
            ],
        ),
        (
            plan_rms_norm_forward(1, 2, 3, 4, 2, 8, 1.5e-6, 0.25).unwrap(),
            vec![
                ("width", Scalar::Int(8)),
                ("eps", Scalar::Float(1.5e-6)),
                ("offset", Scalar::Float(0.25)),
            ],
        ),
        (
            plan_rms_norm_backward(1, 2, 3, 4, 5, Some(6), 2, 8, 0.25).unwrap(),
            vec![("width", Scalar::Int(8)), ("offset", Scalar::Float(0.25))],
        ),
        (
            plan_swiglu_forward(1, 2, 3, 5).unwrap(),
            vec![("rows", Scalar::Int(3)), ("width", Scalar::Int(5))],
        ),
        (
            plan_swiglu_backward(1, 2, 3, 3, 5).unwrap(),
            vec![("rows", Scalar::Int(3)), ("width", Scalar::Int(5))],
        ),
    ];
    let mut seen = Vec::new();
    for (steps, expected) in &cases {
        let step = &steps[0];
        let parameters = &declared[step.kernel()];
        let args = step.args();
        assert_eq!(args.len(), parameters.len(), "{}: argument count", step.kernel());
        let mut next_pointer = 1;
        for (index, parameter) in parameters.iter().enumerate() {
            let at = format!("{} argument {index} ({})", step.kernel(), parameter.name);
            assert_eq!(args.sizes().nth(index), Some(parameter.bytes), "{at}: width");
            if parameter.pointer {
                // The pointers were given 1, 2, 3, ... in the order of the kernel's pointer parameters.
                assert_eq!(args.read::<u64>(index), Some(next_pointer), "{at}: address");
                next_pointer += 1;
            } else if let Some((_, value)) = expected.iter().find(|(name, _)| *name == parameter.name) {
                assert_eq!(read_scalar(args, index, parameter), *value, "{at}");
            } else {
                panic!("{at}: a scalar parameter the case does not give a value");
            }
        }
        seen.push(step.kernel());
    }
    seen.sort_unstable();
    let mut listed = Module::Train.kernels().to_vec();
    listed.sort_unstable();
    assert_eq!(seen, listed, "every kernel is covered");
}

/// Column-major single-precision product `C = alpha op(A) op(B) + beta C` over host memory keyed by the operands'
/// addresses: what cuBLAS computes for the call `Gemm` describes.
fn host_sgemm(gemm: &Gemm, memory: &mut HashMap<u64, Vec<f32>>) {
    let p = gemm.product();
    let [(a, lda), (b, ldb), (c, ldc)] = gemm.operands();
    let (m, n, k) = (p.m as usize, p.n as usize, p.k as usize);
    let (lda, ldb, ldc) = (lda as usize, ldb as usize, ldc as usize);
    let (a_memory, b_memory) = (memory[&a].clone(), memory[&b].clone());
    let c_memory = memory.get_mut(&c).unwrap();
    for j in 0..n {
        for i in 0..m {
            let mut sum = 0.0f32;
            for l in 0..k {
                let left = match p.a_operation {
                    Operation::None => a_memory[l * lda + i],
                    Operation::Transpose => a_memory[i * lda + l],
                };
                let right = match p.b_operation {
                    Operation::None => b_memory[j * ldb + l],
                    Operation::Transpose => b_memory[l * ldb + j],
                };
                sum += left * right;
            }
            let at = j * ldc + i;
            c_memory[at] = p.alpha * sum + p.beta * c_memory[at];
        }
    }
}

/// Small whole numbers, so every sum and product is exact in single precision.
fn whole(len: usize, seed: usize) -> Vec<f32> {
    (0..len).map(|i| ((i * 7 + seed * 3) % 11) as f32 - 5.0).collect()
}

/// A row-major `[rows, columns]` matrix read from `memory` with row stride `leading`.
fn rows_of(memory: &[f32], leading: usize, rows: usize, columns: usize) -> Vec<Vec<f32>> {
    (0..rows)
        .map(|r| (0..columns).map(|c| memory[r * leading + c]).collect())
        .collect()
}

fn row_major_product(left: &[Vec<f32>], right: &[Vec<f32>]) -> Vec<Vec<f32>> {
    (0..left.len())
        .map(|i| {
            (0..right[0].len())
                .map(|j| (0..right.len()).map(|l| left[i][l] * right[l][j]).sum())
                .collect()
        })
        .collect()
}

fn flip(matrix: &[Vec<f32>]) -> Vec<Vec<f32>> {
    (0..matrix[0].len())
        .map(|c| matrix.iter().map(|row| row[c]).collect())
        .collect()
}

#[test]
fn every_row_major_product_equals_the_host_matrix_product() {
    // The linear products take packed matrices; the general products below are padded after each row, so a leading
    // dimension that is mistaken for a width is seen.
    let (rows, inputs, outputs) = (3usize, 4usize, 5usize);
    let (x_ld, w_ld, y_ld) = (inputs, inputs, outputs);
    let x_memory = whole(rows * x_ld, 1);
    let w_memory = whole(outputs * w_ld, 2);
    let y_memory = whole(rows * y_ld, 3);
    let (x, w) = (
        rows_of(&x_memory, x_ld, rows, inputs),
        rows_of(&w_memory, w_ld, outputs, inputs),
    );
    let memory = || HashMap::from([(10, x_memory.clone()), (20, w_memory.clone()), (30, y_memory.clone())]);
    let check = |gemm: &Gemm, expected: &[Vec<f32>], leading: usize, what: &str| {
        let mut host = memory();
        host_sgemm(gemm, &mut host);
        let got = rows_of(&host[&30], leading, expected.len(), expected[0].len());
        assert_eq!(got, expected, "{what}");
        // Nothing past the rows' columns is written.
        let before = &y_memory;
        for r in 0..expected.len() {
            let columns = expected[0].len();
            assert_eq!(
                &host[&30][r * leading + columns..(r + 1) * leading],
                &before[r * leading + columns..(r + 1) * leading],
                "{what}: padding"
            );
        }
    };
    let y_rows = rows_of(&y_memory, y_ld, rows, outputs);

    // y = x w^T, overwriting, then accumulating.
    let forward = row_major_product(&x, &flip(&w));
    check(
        &plan_linear(10, 20, 30, rows as u32, inputs as u32, outputs as u32, false).unwrap(),
        &forward,
        y_ld,
        "linear",
    );
    let plus: Vec<Vec<f32>> = forward
        .iter()
        .zip(&y_rows)
        .map(|(f, y)| f.iter().zip(y).map(|(f, y)| f + y).collect())
        .collect();
    check(
        &plan_linear(10, 20, 30, rows as u32, inputs as u32, outputs as u32, true).unwrap(),
        &plus,
        y_ld,
        "linear accumulating",
    );

    // dx [rows, in] += dy [rows, out] w [out, in]: the operands change roles.
    let dy_ld = outputs;
    let dy_memory = whole(rows * dy_ld, 4);
    let dy = rows_of(&dy_memory, dy_ld, rows, outputs);
    let dx_memory = whole(rows * x_ld, 5);
    let dx_rows = rows_of(&dx_memory, x_ld, rows, inputs);
    let mut host = HashMap::from([(10, dy_memory), (20, w_memory.clone()), (30, dx_memory)]);
    host_sgemm(
        &plan_linear_input_gradient(10, 20, 30, rows as u32, inputs as u32, outputs as u32).unwrap(),
        &mut host,
    );
    let expected: Vec<Vec<f32>> = row_major_product(&dy, &w)
        .iter()
        .zip(&dx_rows)
        .map(|(p, d)| p.iter().zip(d).map(|(p, d)| p + d).collect())
        .collect();
    assert_eq!(rows_of(&host[&30], x_ld, rows, inputs), expected, "input gradient");

    // c [m, n] = alpha a b + beta c in the three layouts.
    let (m, n, k) = (3usize, 5usize, 4usize);
    let matrices = |a, b, c| Matrices {
        a: (10, a),
        b: (20, b),
        c: (30, c),
    };
    let (c_ld, a_ld, b_ld) = (n + 3, k + 1, n + 2);
    let a_memory = whole(m * a_ld, 6);
    let b_memory = whole(k * b_ld, 7);
    let c_memory = whole(m * c_ld, 8);
    let (a, b, c) = (
        rows_of(&a_memory, a_ld, m, k),
        rows_of(&b_memory, b_ld, k, n),
        rows_of(&c_memory, c_ld, m, n),
    );
    let scaled_sum = |product: Vec<Vec<f32>>| -> Vec<Vec<f32>> {
        product
            .iter()
            .zip(&c)
            .map(|(p, c)| p.iter().zip(c).map(|(p, c)| 2.0 * p + 0.5 * c).collect())
            .collect()
    };
    let mut host = HashMap::from([(10, a_memory.clone()), (20, b_memory.clone()), (30, c_memory.clone())]);
    host_sgemm(
        &plan_product(
            &matrices(a_ld as u32, b_ld as u32, c_ld as u32),
            m as u32,
            n as u32,
            k as u32,
            2.0,
            0.5,
        )
        .unwrap(),
        &mut host,
    );
    assert_eq!(
        rows_of(&host[&30], c_ld, m, n),
        scaled_sum(row_major_product(&a, &b)),
        "product"
    );

    // b stored [n, k].
    let bt_ld = k + 2;
    let bt_memory = whole(n * bt_ld, 9);
    let bt = rows_of(&bt_memory, bt_ld, n, k);
    let mut host = HashMap::from([(10, a_memory), (20, bt_memory), (30, c_memory.clone())]);
    host_sgemm(
        &plan_product_transposed(
            &matrices(a_ld as u32, bt_ld as u32, c_ld as u32),
            m as u32,
            n as u32,
            k as u32,
            2.0,
            0.5,
        )
        .unwrap(),
        &mut host,
    );
    assert_eq!(
        rows_of(&host[&30], c_ld, m, n),
        scaled_sum(row_major_product(&a, &flip(&bt))),
        "right operand transposed"
    );

    // a stored [k, m].
    let at_ld = m + 2;
    let at_memory = whole(k * at_ld, 10);
    let at = rows_of(&at_memory, at_ld, k, m);
    let mut host = HashMap::from([(10, at_memory), (20, b_memory), (30, c_memory)]);
    host_sgemm(
        &plan_product_left_transposed(
            &matrices(at_ld as u32, b_ld as u32, c_ld as u32),
            m as u32,
            n as u32,
            k as u32,
            2.0,
            0.5,
        )
        .unwrap(),
        &mut host,
    );
    assert_eq!(
        rows_of(&host[&30], c_ld, m, n),
        scaled_sum(row_major_product(&flip(&at), &b)),
        "left operand transposed"
    );
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
    for (m, n, k, name) in [(0, 10, 8, "m"), (6, 0, 8, "n"), (6, 10, 0, "k")] {
        assert!(
            matches!(
                plan_product(&matrices(8, 10, 10), m, n, k, 1.0, 0.0),
                Err(TrainError::OutOfRange { name: refused, value: 0 }) if refused == name
            ),
            "{name}"
        );
    }
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

fn run(rig: &Rig, steps: &[Step]) -> Result<(), TrainError> {
    let options = ModuleOptions::new(PathBuf::from("/cuda/include"));
    let compiled = compile::<FakeModule>(&rig.gpu, &options, Module::Train).unwrap();
    let kernels = TrainKernels::new(&compiled);
    let lanes = open_shared_lanes(&rig.gpu, &[("test", StreamOptions::default())]).unwrap();
    let lane = Arc::clone(lanes.pool.lane(LaneId(0)));
    let mut refused = None;
    lane.submit(|queue| {
        // SAFETY: the fake kernels read no memory.
        refused = unsafe { kernels.enqueue(queue, steps) }.err();
        Ok(())
    })?;
    while rig.gpu.step() {}
    refused.map_or(Ok(()), Err)
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
fn a_step_of_another_module_is_refused_before_any_launch() {
    let rig = rig();
    let mut steps = plan_zero(BASE, 10).unwrap();
    steps.extend(plan_fill_floats(BASE, 4, 0.0).unwrap());
    assert!(matches!(
        run(&rig, &steps),
        Err(TrainError::Step(StepError::WrongModule { .. }))
    ));
    assert!(rig.launched.lock().unwrap().is_empty());
}

#[test]
fn a_step_needs_a_kernel_its_module_lists() {
    let launch = Launch::linear(1, 32);
    assert!(Step::new(Module::Train, "no_such_kernel", launch, Args::new()).is_none());
    assert!(Step::new(Module::Trellis, "fill", launch, Args::new()).is_none());
    assert!(Step::new(Module::Train, "fill", launch, Args::new()).is_some());
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
    .map(|plan| plan.unwrap()[0].kernel());
    let mut listed = Module::Train.kernels().to_vec();
    let mut named = launched.to_vec();
    listed.sort_unstable();
    named.sort_unstable();
    assert_eq!(named, listed);
}
