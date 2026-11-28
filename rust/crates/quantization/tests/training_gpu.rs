//! Device tests of the training operations. `#[ignore]`d like the crate's other device tests:
//! `cargo test -p euhedral-inference-quantization --test training_gpu -- --ignored --nocapture`.
//!
//! The engine library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY`); the bundled
//! NVRTC and the CUDA headers are beside it. The math libraries come from `euhedral.quantization.cuda-libraries`.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "a test: waits for the lane to retire the work, and copies synchronously to check results"
)]

mod support;

use std::ops::Deref;
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_gpu::cuda::CudaModule;
use euhedral_gpu::{GpuError, LaneStream};
use euhedral_inference_cuda::DeviceBuffer;
use euhedral_inference_cuda::math::MathLibraries;
use euhedral_inference_quantization::kernels::Module;
use euhedral_inference_quantization::training::{
    AdamSettings, Gemm, Heads, Matrices, Step, TrainKernels, plan_adam, plan_bf16_to_float, plan_copy, plan_fill,
    plan_gate_backward, plan_gate_forward, plan_linear, plan_linear_input_gradient, plan_product,
    plan_product_left_transposed, plan_product_transposed, plan_qk_norm_rope_backward, plan_qk_norm_rope_forward,
    plan_rms_norm_backward, plan_rms_norm_forward, plan_swiglu_backward, plan_swiglu_forward, plan_zero,
};
use support::{Device, device, gaussians};

/// One piece of work on the lane, in order.
enum Work {
    Steps(Vec<Step>),
    Product(Gemm),
}

struct Rig {
    device: Device,
    math: Arc<MathLibraries>,
    kernels: TrainKernels<CudaModule>,
}

impl Deref for Rig {
    type Target = Device;

    fn deref(&self) -> &Device {
        &self.device
    }
}

fn rig() -> Rig {
    let directory = test_support::test_setting("euhedral.quantization.cuda-libraries")
        .expect("set euhedral.quantization.cuda-libraries (EUHEDRAL_TEST_QUANTIZATION_CUDA_LIBRARIES)");
    let math = MathLibraries::load(&PathBuf::from(directory)).expect("the math libraries load");
    let (device, compiled) = device(Module::Train, "train");
    let kernels = TrainKernels::new(&compiled);
    Rig { device, math, kernels }
}

impl Rig {
    /// Runs `work` in order on one submission and returns when the lane has retired it.
    fn run(&self, work: &[Work]) {
        let mut handles = self.math.lane_handles().expect("handles");
        self.submit_and_wait(|queue| {
            for item in work {
                match item {
                    // SAFETY: the steps name live buffers of the test, which outlive the wait.
                    Work::Steps(steps) => unsafe { self.kernels.enqueue(queue, steps) }
                        .map_err(|error| GpuError::Invalid(error.to_string()))?,
                    Work::Product(gemm) => {
                        let stream = queue.raw_stream().expect("the lane lends a stream");
                        // SAFETY: as above, on the lane's own stream and handles.
                        unsafe { gemm.enqueue(&mut handles, &stream) }
                            .map_err(|error| GpuError::Invalid(error.to_string()))?;
                    }
                }
            }
            Ok(())
        });
        handles.close().expect("the handles close");
    }

    fn upload(&self, values: &[f32]) -> DeviceBuffer {
        let buffer = DeviceBuffer::alloc(&self.library, (4 * values.len()).max(1)).expect("device allocation");
        buffer
            .copy_from_host(0, &values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .expect("upload");
        buffer
    }

    fn zeros(&self, count: usize) -> DeviceBuffer {
        self.upload(&vec![0.0; count])
    }

    fn download(&self, buffer: &DeviceBuffer, count: usize) -> Vec<f64> {
        let mut bytes = vec![0u8; 4 * count];
        buffer.copy_to_host(0, &mut bytes).expect("download");
        let (words, _) = bytes.as_chunks::<4>();
        words.iter().map(|word| f64::from(f32::from_le_bytes(*word))).collect()
    }
}

fn scaled(count: usize, seed: u64, scale: f32) -> Vec<f32> {
    gaussians(count, seed).into_iter().map(|v| v * scale).collect()
}

fn wide(values: &[f32]) -> Vec<f64> {
    values.iter().copied().map(f64::from).collect()
}

fn narrow(values: &[f64]) -> Vec<f32> {
    values.iter().map(|v| *v as f32).collect()
}

/// The largest difference relative to the largest magnitude of `expected` (and at least 1e-6).
fn assert_close(expected: &[f64], actual: &[f64], tolerance: f64, what: &str) {
    assert_eq!(expected.len(), actual.len(), "{what}: length");
    let scale = expected.iter().fold(1e-6f64, |m, v| m.max(v.abs()));
    let (mut worst, mut at) = (0.0f64, 0);
    for (index, (e, a)) in expected.iter().zip(actual).enumerate() {
        let error = (e - a).abs() / scale;
        if error > worst || error.is_nan() {
            (worst, at) = (error, index);
        }
    }
    assert!(worst <= tolerance, "{what}: relative error {worst} at {at}");
}

/// The gradient of `sum(r * f(x))` by central differences.
fn numeric_gradient(f: &dyn Fn(&[f64]) -> Vec<f64>, x: &[f64], r: &[f64]) -> Vec<f64> {
    let h = 1e-5;
    let value = |point: &[f64]| f(point).iter().zip(r).map(|(y, r)| y * r).sum::<f64>();
    (0..x.len())
        .map(|i| {
            let mut point = x.to_vec();
            point[i] = x[i] + h;
            let up = value(&point);
            point[i] = x[i] - h;
            (up - value(&point)) / (2.0 * h)
        })
        .collect()
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// Row-major `c [m, n] = a [m, k] b [k, n]`.
fn matmul(a: &[f64], b: &[f64], m: usize, n: usize, k: usize) -> Vec<f64> {
    let mut c = vec![0.0; m * n];
    for i in 0..m {
        for p in 0..k {
            for j in 0..n {
                c[i * n + j] += a[i * k + p] * b[p * n + j];
            }
        }
    }
    c
}

fn transpose(a: &[f64], rows: usize, cols: usize) -> Vec<f64> {
    let mut t = vec![0.0; a.len()];
    for i in 0..rows {
        for j in 0..cols {
            t[j * rows + i] = a[i * cols + j];
        }
    }
    t
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn rms_norm_forward_and_backward_match_the_host_for_both_offsets() {
    let rig = rig();
    let (rows, width) = (5usize, 96usize);
    for offset in [1.0f32, 0.0] {
        let x = scaled(rows * width, 1, 1.0);
        let w = scaled(width, 2, 0.3);
        let dy = scaled(rows * width, 3, 1.0);
        let reference = |x: &[f64], w: &[f64]| -> Vec<f64> {
            let mut y = Vec::new();
            for row in x.chunks(width) {
                let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / width as f64 + 1e-6).sqrt();
                y.extend(row.iter().zip(w).map(|(v, w)| v * inv * (f64::from(offset) + w)));
            }
            y
        };
        let (dx_buffer, dw_buffer) = (rig.upload(&vec![0.5; rows * width]), rig.zeros(width));
        let (x_buffer, w_buffer, dy_buffer) = (rig.upload(&x), rig.upload(&w), rig.upload(&dy));
        let (y, inverse) = (rig.zeros(rows * width), rig.zeros(rows));
        let mut steps = plan_rms_norm_forward(
            x_buffer.addr(),
            w_buffer.addr(),
            y.addr(),
            inverse.addr(),
            rows as u32,
            width as u32,
            1e-6,
            offset,
        )
        .unwrap();
        steps.extend(
            plan_rms_norm_backward(
                x_buffer.addr(),
                w_buffer.addr(),
                inverse.addr(),
                dy_buffer.addr(),
                dx_buffer.addr(),
                Some(dw_buffer.addr()),
                rows as u32,
                width as u32,
                offset,
            )
            .unwrap(),
        );
        rig.run(&[Work::Steps(steps)]);
        let (xd, wd, r) = (wide(&x), wide(&w), wide(&dy));
        assert_close(&reference(&xd, &wd), &rig.download(&y, rows * width), 1e-5, "forward");
        let by_x = numeric_gradient(&|v| reference(v, &wd), &xd, &r);
        let expected: Vec<f64> = by_x.iter().map(|g| g + 0.5).collect();
        assert_close(
            &expected,
            &rig.download(&dx_buffer, rows * width),
            1e-3,
            "dx adds to what is there",
        );
        let by_w = numeric_gradient(&|v| reference(&xd, v), &wd, &r);
        assert_close(&by_w, &rig.download(&dw_buffer, width), 1e-3, "dw");
    }
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn rms_norm_epsilon_matters_for_a_row_whose_mean_square_is_near_it() {
    let rig = rig();
    let (rows, width, eps) = (3usize, 64usize, 1e-6f32);
    // Row one is scaled so that its mean square, about 1e-6, is comparable to the epsilon.
    let mut x = scaled(rows * width, 7, 1.0);
    x[width..2 * width].iter_mut().for_each(|v| *v *= 1e-3);
    let w = scaled(width, 8, 0.3);
    let (xb, wb) = (rig.upload(&x), rig.upload(&w));
    let (y, inverse) = (rig.zeros(rows * width), rig.zeros(rows));
    let steps = plan_rms_norm_forward(
        xb.addr(),
        wb.addr(),
        y.addr(),
        inverse.addr(),
        rows as u32,
        width as u32,
        eps,
        1.0,
    )
    .unwrap();
    rig.run(&[Work::Steps(steps)]);
    let (xd, wd) = (wide(&x), wide(&w));
    let inverses: Vec<f64> = xd
        .chunks(width)
        .map(|row| 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / width as f64 + f64::from(eps)).sqrt())
        .collect();
    let expected: Vec<f64> = xd
        .chunks(width)
        .zip(&inverses)
        .flat_map(|(row, inv)| row.iter().zip(&wd).map(move |(v, w)| v * inv * (1.0 + w)))
        .collect();
    assert_close(&inverses, &rig.download(&inverse, rows), 1e-5, "inverse");
    assert_close(&expected, &rig.download(&y, rows * width), 1e-5, "forward");
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn rms_norm_backward_without_a_weight_gradient_leaves_dx_correct() {
    let rig = rig();
    let (rows, width) = (3usize, 64usize);
    let (x, w, dy) = (
        scaled(rows * width, 4, 1.0),
        scaled(width, 5, 0.3),
        scaled(rows * width, 6, 1.0),
    );
    let (xb, wb, dyb) = (rig.upload(&x), rig.upload(&w), rig.upload(&dy));
    let (y, inverse, dx) = (rig.zeros(rows * width), rig.zeros(rows), rig.zeros(rows * width));
    let mut steps = plan_rms_norm_forward(
        xb.addr(),
        wb.addr(),
        y.addr(),
        inverse.addr(),
        rows as u32,
        width as u32,
        1e-6,
        1.0,
    )
    .unwrap();
    steps.extend(
        plan_rms_norm_backward(
            xb.addr(),
            wb.addr(),
            inverse.addr(),
            dyb.addr(),
            dx.addr(),
            None,
            rows as u32,
            width as u32,
            1.0,
        )
        .unwrap(),
    );
    rig.run(&[Work::Steps(steps)]);
    let wd = wide(&w);
    let reference = |x: &[f64]| -> Vec<f64> {
        let mut y = Vec::new();
        for row in x.chunks(width) {
            let inv = 1.0 / (row.iter().map(|v| v * v).sum::<f64>() / width as f64 + 1e-6).sqrt();
            y.extend(row.iter().zip(&wd).map(|(v, w)| v * inv * (1.0 + w)));
        }
        y
    };
    let expected = numeric_gradient(&reference, &wide(&x), &wide(&dy));
    assert_close(&expected, &rig.download(&dx, rows * width), 1e-3, "dx");
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn swiglu_forward_and_backward_match_the_host() {
    let rig = rig();
    let (rows, width) = (4usize, 40usize);
    let gate_up = scaled(rows * 2 * width, 7, 1.0);
    let dh = scaled(rows * width, 8, 1.0);
    let reference = |g: &[f64]| -> Vec<f64> {
        let mut h = Vec::new();
        for row in g.chunks(2 * width) {
            h.extend((0..width).map(|j| silu(row[j]) * row[width + j]));
        }
        h
    };
    let (gb, dhb) = (rig.upload(&gate_up), rig.upload(&dh));
    let (h, dg) = (rig.zeros(rows * width), rig.zeros(rows * 2 * width));
    let mut steps = plan_swiglu_forward(gb.addr(), h.addr(), rows as u32, width as u32).unwrap();
    steps.extend(plan_swiglu_backward(gb.addr(), dhb.addr(), dg.addr(), rows as u32, width as u32).unwrap());
    rig.run(&[Work::Steps(steps)]);
    let gd = wide(&gate_up);
    assert_close(&reference(&gd), &rig.download(&h, rows * width), 1e-5, "forward");
    let expected = numeric_gradient(&reference, &gd, &wide(&dh));
    assert_close(&expected, &rig.download(&dg, rows * 2 * width), 1e-3, "backward");
}

/// The host norm and half-split rotation of `qk` rows, with `w = [q_norm; k_norm]`.
fn qk_reference(shape: &Heads, qk: &[f64], norms: &[f64]) -> Vec<f64> {
    let (d, rotary) = (shape.head as usize, shape.rotary as usize);
    let heads = (shape.q_heads + shape.kv_heads) as usize;
    let mut out = vec![0.0; qk.len()];
    for row in 0..shape.rows as usize {
        for head in 0..heads {
            let base = (row * heads + head) * d;
            let x = &qk[base..base + d];
            let w = if head < shape.q_heads as usize {
                &norms[..d]
            } else {
                &norms[d..]
            };
            let inv = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / d as f64 + 1e-6).sqrt();
            let n: Vec<f64> = x.iter().zip(w).map(|(v, w)| v * inv * (1.0 + w)).collect();
            out[base..base + d].copy_from_slice(&n);
            let half = rotary / 2;
            for f in 0..half {
                let angle = (shape.position0 + row as i64) as f64 * shape.theta.powf(-(2.0 * f as f64) / rotary as f64);
                let (c, s) = (angle.cos(), angle.sin());
                out[base + f] = n[f] * c - n[f + half] * s;
                out[base + f + half] = n[f + half] * c + n[f] * s;
            }
        }
    }
    out
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn qk_norm_rope_forward_and_backward_match_the_host() {
    let rig = rig();
    let shape = Heads {
        rows: 3,
        q_heads: 4,
        kv_heads: 2,
        head: 64,
        rotary: 16,
        position0: 1000,
        theta: 1e7,
    };
    let count = 3 * 6 * 64;
    let qk = scaled(count, 9, 1.0);
    let norms = scaled(128, 10, 0.3);
    let d_out = scaled(count, 11, 1.0);
    let (qkb, qn, kn) = (rig.upload(&qk), rig.upload(&norms[..64]), rig.upload(&norms[64..]));
    let (dob, out, inverse) = (rig.upload(&d_out), rig.zeros(count), rig.zeros(3 * 6));
    let (dqk, dqn, dkn) = (rig.zeros(count), rig.zeros(64), rig.zeros(64));
    let mut steps = plan_qk_norm_rope_forward(
        qkb.addr(),
        qn.addr(),
        kn.addr(),
        out.addr(),
        inverse.addr(),
        &shape,
        1e-6,
    )
    .unwrap();
    steps.extend(
        plan_qk_norm_rope_backward(
            qkb.addr(),
            qn.addr(),
            kn.addr(),
            inverse.addr(),
            dob.addr(),
            dqk.addr(),
            Some(dqn.addr()),
            Some(dkn.addr()),
            &shape,
        )
        .unwrap(),
    );
    rig.run(&[Work::Steps(steps)]);
    let (qkd, nd, r) = (wide(&qk), wide(&norms), wide(&d_out));
    assert_close(
        &qk_reference(&shape, &qkd, &nd),
        &rig.download(&out, count),
        1e-5,
        "forward",
    );
    let by_x = numeric_gradient(&|v| qk_reference(&shape, v, &nd), &qkd, &r);
    assert_close(&by_x, &rig.download(&dqk, count), 1e-3, "dqk");
    let by_norm = numeric_gradient(&|v| qk_reference(&shape, &qkd, v), &nd, &r);
    assert_close(&by_norm[..64], &rig.download(&dqn, 64), 1e-3, "dq_norm");
    assert_close(&by_norm[64..], &rig.download(&dkn, 64), 1e-3, "dk_norm");
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn gate_forward_and_backward_match_the_host_on_a_wider_gate_row() {
    let rig = rig();
    let (rows, width, row_width) = (4usize, 12usize, 20usize);
    let context = scaled(rows * width, 12, 1.0);
    let gate = scaled(rows * row_width, 13, 1.0);
    let d_out = scaled(rows * width, 14, 1.0);
    let reference = |c: &[f64], g: &[f64]| -> Vec<f64> {
        (0..rows * width)
            .map(|i| c[i] * sigmoid(g[(i / width) * row_width + i % width]))
            .collect()
    };
    let (cb, gb, db) = (rig.upload(&context), rig.upload(&gate), rig.upload(&d_out));
    let (out, dc, dg) = (
        rig.zeros(rows * width),
        rig.zeros(rows * width),
        rig.zeros(rows * row_width),
    );
    let mut steps = plan_gate_forward(
        cb.addr(),
        gb.addr(),
        out.addr(),
        rows as u32,
        width as u32,
        row_width as u32,
    )
    .unwrap();
    steps.extend(
        plan_gate_backward(
            cb.addr(),
            gb.addr(),
            db.addr(),
            dc.addr(),
            dg.addr(),
            rows as u32,
            width as u32,
            row_width as u32,
        )
        .unwrap(),
    );
    rig.run(&[Work::Steps(steps)]);
    let (cd, gd, r) = (wide(&context), wide(&gate), wide(&d_out));
    assert_close(&reference(&cd, &gd), &rig.download(&out, rows * width), 1e-5, "forward");
    assert_close(
        &numeric_gradient(&|v| reference(v, &gd), &cd, &r),
        &rig.download(&dc, rows * width),
        1e-3,
        "dcontext",
    );
    let by_gate = numeric_gradient(&|v| reference(&cd, v), &gd, &r);
    assert_close(
        &by_gate,
        &rig.download(&dg, rows * row_width),
        1e-3,
        "dgate, columns past the width stay zero",
    );
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn linear_and_its_input_gradient_match_the_host_products() {
    let rig = rig();
    let (rows, inputs, outputs) = (5usize, 24usize, 16usize);
    let x = scaled(rows * inputs, 15, 1.0);
    let w = scaled(outputs * inputs, 16, 0.3);
    let dy = scaled(rows * outputs, 17, 1.0);
    let (xb, wb, dyb) = (rig.upload(&x), rig.upload(&w), rig.upload(&dy));
    let (y, y_sum, dx) = (
        rig.zeros(rows * outputs),
        rig.upload(&vec![1.0; rows * outputs]),
        rig.upload(&vec![0.25; rows * inputs]),
    );
    let (r, i, o) = (rows as u32, inputs as u32, outputs as u32);
    rig.run(&[
        Work::Product(plan_linear(xb.addr(), wb.addr(), y.addr(), r, i, o, false).unwrap()),
        Work::Product(plan_linear(xb.addr(), wb.addr(), y_sum.addr(), r, i, o, true).unwrap()),
        Work::Product(plan_linear_input_gradient(dyb.addr(), wb.addr(), dx.addr(), r, i, o).unwrap()),
    ]);
    let (xd, wd, dyd) = (wide(&x), wide(&w), wide(&dy));
    let forward = matmul(&xd, &transpose(&wd, outputs, inputs), rows, outputs, inputs);
    assert_close(&forward, &rig.download(&y, rows * outputs), 1e-5, "forward overwrites");
    let accumulated: Vec<f64> = forward.iter().map(|v| v + 1.0).collect();
    assert_close(
        &accumulated,
        &rig.download(&y_sum, rows * outputs),
        1e-5,
        "forward accumulates",
    );
    let input_gradient: Vec<f64> = matmul(&dyd, &wd, rows, inputs, outputs)
        .iter()
        .map(|v| v + 0.25)
        .collect();
    assert_close(&input_gradient, &rig.download(&dx, rows * inputs), 1e-5, "dx adds");
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn the_three_row_major_products_match_the_host_with_alpha_and_beta() {
    let rig = rig();
    let (m, n, k) = (6usize, 10usize, 8usize);
    let a = scaled(m * k, 18, 1.0);
    let b = scaled(k * n, 19, 1.0);
    let (ad, bd) = (wide(&a), wide(&b));
    let (ab, bb) = (rig.upload(&a), rig.upload(&b));
    // The transposed forms read the same values through the other layout.
    let (a_stored_k_by_m, b_stored_n_by_k) = (transpose(&ad, m, k), transpose(&bd, k, n));
    let (atb, btb) = (
        rig.upload(&narrow(&a_stored_k_by_m)),
        rig.upload(&narrow(&b_stored_n_by_k)),
    );
    let start = vec![0.5; m * n];
    let (plain, transposed, left, left_sum) = (
        rig.upload(&narrow(&start)),
        rig.upload(&narrow(&start)),
        rig.upload(&narrow(&start)),
        rig.upload(&narrow(&start)),
    );
    let (mm, nn, kk) = (m as u32, n as u32, k as u32);
    let at = |pointer: &DeviceBuffer, leading: u32| (pointer.addr(), leading);
    rig.run(&[
        Work::Product(
            plan_product(
                &Matrices {
                    a: at(&ab, kk),
                    b: at(&bb, nn),
                    c: at(&plain, nn),
                },
                mm,
                nn,
                kk,
                0.5,
                0.0,
            )
            .unwrap(),
        ),
        Work::Product(
            plan_product_transposed(
                &Matrices {
                    a: at(&ab, kk),
                    b: at(&btb, kk),
                    c: at(&transposed, nn),
                },
                mm,
                nn,
                kk,
                0.5,
                1.0,
            )
            .unwrap(),
        ),
        Work::Product(
            plan_product_left_transposed(
                &Matrices {
                    a: at(&atb, mm),
                    b: at(&bb, nn),
                    c: at(&left, nn),
                },
                mm,
                nn,
                kk,
                1.0,
                0.0,
            )
            .unwrap(),
        ),
        Work::Product(
            plan_product_left_transposed(
                &Matrices {
                    a: at(&atb, mm),
                    b: at(&bb, nn),
                    c: at(&left_sum, nn),
                },
                mm,
                nn,
                kk,
                1.0,
                1.0,
            )
            .unwrap(),
        ),
    ]);
    let product = matmul(&ad, &bd, m, n, k);
    let half: Vec<f64> = product.iter().map(|v| 0.5 * v).collect();
    assert_close(&half, &rig.download(&plain, m * n), 1e-5, "alpha, beta 0 overwrites");
    let half_plus: Vec<f64> = half.iter().map(|v| v + 0.5).collect();
    assert_close(
        &half_plus,
        &rig.download(&transposed, m * n),
        1e-5,
        "b transposed, beta 1",
    );
    assert_close(&product, &rig.download(&left, m * n), 1e-5, "a transposed");
    let plus: Vec<f64> = product.iter().map(|v| v + 0.5).collect();
    assert_close(&plus, &rig.download(&left_sum, m * n), 1e-5, "a transposed, beta 1");
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn adam_follows_the_bias_corrected_update_and_zeroes_the_gradient() {
    let rig = rig();
    let count = 100usize;
    let settings = AdamSettings {
        rate: 1e-3,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        decay: 0.01,
    };
    let (p0, g0, g1) = (scaled(count, 20, 1.0), scaled(count, 21, 1.0), scaled(count, 22, 1.0));
    let (p, g, m, v) = (rig.upload(&p0), rig.upload(&g0), rig.zeros(count), rig.zeros(count));
    let steps = plan_adam(p.addr(), g.addr(), m.addr(), v.addr(), count as u64, &settings, 1).unwrap();
    rig.run(&[Work::Steps(steps)]);
    assert_eq!(rig.download(&g, count), vec![0.0; count], "the gradient is zeroed");
    g.copy_from_host(0, &g1.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
        .unwrap();
    let steps = plan_adam(p.addr(), g.addr(), m.addr(), v.addr(), count as u64, &settings, 2).unwrap();
    rig.run(&[Work::Steps(steps)]);

    // The hyperparameters as the kernel sees them: single precision.
    let (b1, b2) = (f64::from(settings.beta1), f64::from(settings.beta2));
    let (mut pe, mut me, mut ve) = (wide(&p0), vec![0.0; count], vec![0.0; count]);
    for (step, gradient) in [(1, wide(&g0)), (2, wide(&g1))] {
        for i in 0..count {
            me[i] = b1 * me[i] + (1.0 - b1) * gradient[i];
            ve[i] = b2 * ve[i] + (1.0 - b2) * gradient[i] * gradient[i];
            let (m_hat, v_hat) = (me[i] / (1.0 - b1.powi(step)), ve[i] / (1.0 - b2.powi(step)));
            pe[i] -= 1e-3 * (m_hat / (v_hat.sqrt() + 1e-8) + 0.01 * pe[i]);
        }
    }
    assert_close(&pe, &rig.download(&p, count), 1e-5, "parameters");
    assert_close(&me, &rig.download(&m, count), 1e-5, "first moment");
    assert_close(&ve, &rig.download(&v, count), 1e-5, "second moment");
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn adam_epsilon_and_weight_decay_show_in_the_change_of_the_parameters() {
    let rig = rig();
    let count = 100usize;
    // Gradients about as small as the epsilon, and a decay large enough to be seen at the tolerance.
    let settings = AdamSettings {
        rate: 0.1,
        beta1: 0.9,
        beta2: 0.999,
        eps: 1e-8,
        decay: 0.5,
    };
    let p0 = scaled(count, 24, 0.05);
    let (g0, g1) = (scaled(count, 25, 1e-8), scaled(count, 26, 1e-8));
    let (p, g, m, v) = (rig.upload(&p0), rig.upload(&g0), rig.zeros(count), rig.zeros(count));
    for (step, gradient) in [(1, &g0), (2, &g1)] {
        g.copy_from_host(0, &gradient.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .unwrap();
        let steps = plan_adam(p.addr(), g.addr(), m.addr(), v.addr(), count as u64, &settings, step).unwrap();
        rig.run(&[Work::Steps(steps)]);
    }
    let (b1, b2) = (f64::from(settings.beta1), f64::from(settings.beta2));
    let (rate, eps, decay) = (
        f64::from(settings.rate),
        f64::from(settings.eps),
        f64::from(settings.decay),
    );
    let (mut pe, mut me, mut ve) = (wide(&p0), vec![0.0; count], vec![0.0; count]);
    for (step, gradient) in [(1, wide(&g0)), (2, wide(&g1))] {
        for i in 0..count {
            me[i] = b1 * me[i] + (1.0 - b1) * gradient[i];
            ve[i] = b2 * ve[i] + (1.0 - b2) * gradient[i] * gradient[i];
            let (m_hat, v_hat) = (me[i] / (1.0 - b1.powi(step)), ve[i] / (1.0 - b2.powi(step)));
            pe[i] -= rate * (m_hat / (v_hat.sqrt() + eps) + decay * pe[i]);
        }
    }
    let change = |values: &[f64]| -> Vec<f64> { values.iter().zip(&p0).map(|(v, p)| v - f64::from(*p)).collect() };
    assert_close(
        &change(&pe),
        &change(&rig.download(&p, count)),
        1e-5,
        "change of the parameters",
    );
}

#[test]
#[ignore = "needs a GPU, the bundled NVRTC and the math libraries"]
fn widening_copying_and_filling_move_exactly_the_values_asked() {
    let rig = rig();
    let count = 300usize;
    let bf16: Vec<u16> = (0..count)
        .map(|i| ((i as u32 * 0x3d1) & 0xffff) as u16 | 0x0001)
        .collect();
    let source = DeviceBuffer::alloc(&rig.library, 2 * count).unwrap();
    source
        .copy_from_host(0, &bf16.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
        .unwrap();
    let floats = scaled(count, 23, 1.0);
    let (widened, from, copied, filled) = (
        rig.zeros(count),
        rig.upload(&floats),
        rig.zeros(count + 1),
        rig.upload(&vec![1.0; count + 1]),
    );
    let mut steps = plan_bf16_to_float(source.addr(), count as u64, widened.addr()).unwrap();
    steps.extend(plan_copy(from.addr(), copied.addr(), count as u64).unwrap());
    steps.extend(plan_fill(filled.addr(), count as u64, 2.5).unwrap());
    rig.run(&[Work::Steps(steps)]);
    let expected: Vec<f64> = bf16
        .iter()
        .map(|v| f64::from(f32::from_bits(u32::from(*v) << 16)))
        .collect();
    let got = rig.download(&widened, count);
    for (e, g) in expected.iter().zip(&got) {
        assert!(
            e.to_bits() == g.to_bits() || (e.is_nan() && g.is_nan()),
            "{e} against {g}"
        );
    }
    let copied = rig.download(&copied, count + 1);
    assert_eq!(&copied[..count], &wide(&floats)[..], "the copy");
    assert_eq!(copied[count], 0.0, "no value past the count is written");
    let filled_values = rig.download(&filled, count + 1);
    assert!(filled_values[..count].iter().all(|v| *v == 2.5));
    assert_eq!(filled_values[count], 1.0, "no value past the count is written");
    let zeroing = plan_zero(filled.addr(), count as u64 + 1).unwrap();
    rig.run(&[Work::Steps(zeroing)]);
    assert!(rig.download(&filled, count + 1).iter().all(|v| *v == 0.0));
}
