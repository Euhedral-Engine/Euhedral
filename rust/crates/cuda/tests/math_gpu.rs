//! Device tests of the cuBLAS and cuSOLVER calls on a lane's lent stream. `#[ignore]`d like the crate's other device
//! tests: `cargo test -p euhedral-inference-cuda --test math_gpu -- --ignored --nocapture`.
//!
//! The CUDA library comes from `euhedral.cuda.library` (`EUHEDRAL_TEST_CUDA_LIBRARY`) and the directory of the math
//! libraries from `euhedral.quantization.cuda-libraries` (`EUHEDRAL_TEST_QUANTIZATION_CUDA_LIBRARIES`); a test skips
//! with a message when either is absent.
#![cfg(all(target_os = "linux", target_arch = "x86_64"))]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "a test: waits for the queue to retire the work, and copies synchronously to check results"
)]

use std::path::PathBuf;
use std::sync::Arc;
use std::thread::sleep;
use std::time::{Duration, Instant};

use euhedral_gpu::cuda::{CudaGpu, CudaQueue};
use euhedral_gpu::{Lane, LaneStream, SharedLanes, StreamOptions, open_shared_lanes};
use euhedral_graph::{LaneId, Marker};
use euhedral_inference_cuda::math::{
    Fill, GemmOutput, LaneHandles, MathLibraries, MathMode, Matrix, Operation, Product, check_info,
};
use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer};

/// The library, the math libraries and one lane, or `None` (with a skip line) when a setting is absent.
struct Rig {
    library: Arc<CudaLibrary>,
    math: Arc<MathLibraries>,
    lanes: SharedLanes<CudaQueue>,
    lane: Arc<CudaQueue>,
    // The device stays open for the lanes.
    _gpu: Arc<CudaGpu>,
}

fn rig() -> Option<Rig> {
    let library = test_support::require_property("euhedral.cuda.library")?;
    let directory = test_support::require_property("euhedral.quantization.cuda-libraries")?;
    let library = CudaLibrary::load(&PathBuf::from(library)).expect("the CUDA library loads");
    let math = MathLibraries::load(&PathBuf::from(directory)).expect("the math libraries load");
    let gpu = CudaGpu::open(0).expect("the device opens");
    let lanes = open_shared_lanes(&gpu, &[("math", StreamOptions::default())]).expect("a lane opens");
    let lane = lanes.pool.lane(LaneId(0)).clone();
    Some(Rig {
        library,
        math,
        lanes,
        lane,
        _gpu: gpu,
    })
}

impl Rig {
    /// Waits until the queue has retired `marker`: the work before it is complete and its results are readable.
    fn retired(&self, marker: Marker) {
        let start = Instant::now();
        while !self.lanes.queues.reached(marker) {
            assert!(start.elapsed() < Duration::from_secs(30), "the lane did not retire");
            sleep(Duration::from_micros(100));
        }
    }

    fn buffer(&self, bytes: usize) -> DeviceBuffer {
        DeviceBuffer::alloc(&self.library, bytes).expect("device allocation")
    }

    fn upload_f32(&self, values: &[f32]) -> DeviceBuffer {
        let buffer = self.buffer(values.len() * 4);
        buffer
            .copy_from_host(0, &values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .expect("upload");
        buffer
    }

    fn upload_i32(&self, value: i32) -> DeviceBuffer {
        let buffer = self.buffer(4);
        buffer.copy_from_host(0, &value.to_le_bytes()).expect("upload");
        buffer
    }

    fn upload_bf16(&self, values: &[u16]) -> DeviceBuffer {
        let buffer = self.buffer(values.len() * 2);
        buffer
            .copy_from_host(0, &values.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>())
            .expect("upload");
        buffer
    }
}

fn download_f32(buffer: &DeviceBuffer, count: usize) -> Vec<f32> {
    let mut bytes = vec![0u8; count * 4];
    buffer.copy_to_host(0, &mut bytes).expect("download");
    let (words, _) = bytes.as_chunks::<4>();
    words.iter().map(|word| f32::from_le_bytes(*word)).collect()
}

fn download_i32(buffer: &DeviceBuffer) -> i32 {
    let mut bytes = [0u8; 4];
    buffer.copy_to_host(0, &mut bytes).expect("download");
    i32::from_le_bytes(bytes)
}

/// BF16 from FP32, rounding to nearest even.
fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    let rounded = bits.wrapping_add(0x7fff + ((bits >> 16) & 1));
    (rounded >> 16) as u16
}

fn from_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

/// What an info buffer holds before a call writes it: a call that does not write it shows.
const SENTINEL: i32 = 777;

/// Deterministic values in [-1, 1).
fn values(count: usize, seed: u64) -> Vec<f32> {
    let mut state = seed;
    (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((state >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

/// `op(A) op(B)` in f64 and the sum of the absolute terms per element (the bound's scale), column-major `m` by `n`.
fn reference(product: &Product, a: &[f32], lda: usize, b: &[f32], ldb: usize) -> (Vec<f64>, Vec<f64>) {
    let (m, n, k) = (product.m as usize, product.n as usize, product.k as usize);
    let at = |i: usize, p: usize| match product.a_operation {
        Operation::None => a[i + p * lda],
        Operation::Transpose => a[p + i * lda],
    };
    let bt = |p: usize, j: usize| match product.b_operation {
        Operation::None => b[p + j * ldb],
        Operation::Transpose => b[j + p * ldb],
    };
    let mut exact = vec![0.0; m * n];
    let mut scale = vec![0.0; m * n];
    for j in 0..n {
        for i in 0..m {
            for p in 0..k {
                let term = f64::from(at(i, p)) * f64::from(bt(p, j));
                exact[i + j * m] += term;
                scale[i + j * m] += term.abs();
            }
        }
    }
    (exact, scale)
}

/// The FP32 accumulation bound: `k` additions, each off by at most 2^-24 of the running magnitude, doubled for slack.
fn accumulation_bound(k: usize, scale: f64) -> f64 {
    2.0 * k as f64 * f64::powi(2.0, -24) * scale
}

#[test]
#[ignore = "needs a GPU, the native library and the math libraries"]
fn gemm_matches_a_host_product_within_the_fp32_accumulation_bound() {
    let Some(rig) = rig() else { return };
    let (m, n, k) = (33usize, 17usize, 129usize);
    // A is stored k by m and used transposed, so both operations are exercised; B is k by n.
    let product = Product {
        a_operation: Operation::Transpose,
        b_operation: Operation::None,
        m: m as i32,
        n: n as i32,
        k: k as i32,
        alpha: 1.0,
        beta: 0.0,
    };
    let a_bits: Vec<u16> = values(k * m, 1).into_iter().map(to_bf16).collect();
    let b_bits: Vec<u16> = values(k * n, 2).into_iter().map(to_bf16).collect();
    let a: Vec<f32> = a_bits.iter().copied().map(from_bf16).collect();
    let b: Vec<f32> = b_bits.iter().copied().map(from_bf16).collect();
    let (exact, scale) = reference(&product, &a, k, &b, k);

    let a_device = rig.upload_bf16(&a_bits);
    let b_device = rig.upload_bf16(&b_bits);
    let c32 = rig.buffer(m * n * 4);
    let c16 = rig.buffer(m * n * 2);
    // A product scaled by alpha = 0.5 added to a filled C with beta = 1.
    let scaled = Product {
        alpha: 0.5,
        beta: 1.0,
        ..product
    };
    let before = values(m * n, 9);
    let c_scaled = rig.upload_f32(&before);
    let operands = |c: &DeviceBuffer| {
        (
            Matrix {
                pointer: a_device.ptr(),
                leading: k as i32,
            },
            Matrix {
                pointer: b_device.ptr(),
                leading: k as i32,
            },
            Matrix {
                pointer: c.ptr(),
                leading: m as i32,
            },
        )
    };
    let mut handles: Option<LaneHandles> = None;
    let ((), marker) = rig
        .lane
        .submit_marked(|queue| {
            let stream = queue.raw_stream().expect("the lane lends a stream");
            let handles = match &mut handles {
                Some(handles) => handles,
                empty => empty.insert(rig.math.lane_handles().expect("handles")),
            };
            let (a, b, c) = operands(&c32);
            // SAFETY: the operands are live device allocations of the sizes the product implies; they are kept until
            // the test reads the results after the retirement.
            unsafe { handles.gemm_bf16(&stream, &product, a, b, c, GemmOutput::Fp32) }.expect("gemm");
            let (a, b, c) = operands(&c16);
            // SAFETY: as above.
            unsafe { handles.gemm_bf16(&stream, &product, a, b, c, GemmOutput::Bf16) }.expect("gemm");
            let (a, b, c) = operands(&c_scaled);
            // SAFETY: as above.
            unsafe { handles.gemm_bf16(&stream, &scaled, a, b, c, GemmOutput::Fp32) }.expect("gemm");
            Ok(())
        })
        .expect("submit");
    rig.retired(marker);

    let got = download_f32(&c32, m * n);
    for index in 0..m * n {
        let error = (f64::from(got[index]) - exact[index]).abs();
        assert!(
            error <= accumulation_bound(k, scale[index]),
            "element {index}: {} against {}, error {error}",
            got[index],
            exact[index]
        );
    }
    // alpha and beta: 0.5 * product + C, the half of the accumulation bound and one rounding of the sum.
    let got = download_f32(&c_scaled, m * n);
    for index in 0..m * n {
        let expected = 0.5 * exact[index] + f64::from(before[index]);
        let bound = 0.5 * accumulation_bound(k, scale[index]) + expected.abs() * f64::powi(2.0, -23);
        assert!(
            (f64::from(got[index]) - expected).abs() <= bound,
            "scaled element {index}: {} against {expected}",
            got[index]
        );
    }
    handles.take().expect("handles").close().expect("the handles close");
    let mut bytes = vec![0u8; m * n * 2];
    c16.copy_to_host(0, &mut bytes).unwrap();
    for index in 0..m * n {
        let value = f64::from(from_bf16(u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]])));
        // One rounding to BF16 (a relative 2^-8) on top of the accumulation.
        let bound = accumulation_bound(k, scale[index]) + exact[index].abs() * f64::powi(2.0, -8);
        assert!(
            (value - exact[index]).abs() <= bound,
            "bf16 element {index}: {value} against {}",
            exact[index]
        );
    }
}

#[test]
#[ignore = "needs a GPU, the native library and the math libraries"]
fn sgemm_matches_a_host_product_within_the_fp32_accumulation_bound() {
    let Some(rig) = rig() else { return };
    let (m, n, k) = (20usize, 24usize, 96usize);
    let product = Product {
        a_operation: Operation::None,
        b_operation: Operation::Transpose,
        m: m as i32,
        n: n as i32,
        k: k as i32,
        alpha: 1.0,
        beta: 0.0,
    };
    // A is m by k, B is stored n by k and used transposed.
    let a = values(m * k, 3);
    let b = values(n * k, 4);
    let (exact, scale) = reference(&product, &a, m, &b, n);
    let (a_device, b_device) = (rig.upload_f32(&a), rig.upload_f32(&b));
    let c = rig.buffer(m * n * 4);
    let mut handles: Option<LaneHandles> = None;
    let ((), marker) = rig
        .lane
        .submit_marked(|queue| {
            let stream = queue.raw_stream().expect("the lane lends a stream");
            let handles = handles.insert(rig.math.lane_handles().expect("handles"));
            handles.set_math_mode(MathMode::Default).expect("the math mode");
            // SAFETY: the operands are live device allocations of the sizes the product implies and stay so until the
            // results are read after the retirement.
            unsafe {
                handles.gemm_fp32(
                    &stream,
                    &product,
                    Matrix {
                        pointer: a_device.ptr(),
                        leading: m as i32,
                    },
                    Matrix {
                        pointer: b_device.ptr(),
                        leading: n as i32,
                    },
                    Matrix {
                        pointer: c.ptr(),
                        leading: m as i32,
                    },
                )
            }
            .expect("sgemm");
            Ok(())
        })
        .expect("submit");
    rig.retired(marker);
    let got = download_f32(&c, m * n);
    for index in 0..m * n {
        let error = (f64::from(got[index]) - exact[index]).abs();
        assert!(
            error <= accumulation_bound(k, scale[index]),
            "element {index}: {} against {}, error {error}",
            got[index],
            exact[index]
        );
    }
    handles.take().expect("handles").close().expect("the handles close");
}

/// The symmetric positive definite matrix the solver tests factor: tridiagonal, 4 on the diagonal, 1 beside it.
fn spd(n: usize) -> Vec<f32> {
    let mut a = vec![0.0f32; n * n];
    for i in 0..n {
        a[i + i * n] = 4.0;
        if i + 1 < n {
            a[i + 1 + i * n] = 1.0;
            a[i + (i + 1) * n] = 1.0;
        }
    }
    a
}

/// The inverse of the matrix by Gauss-Jordan in f64, column-major.
fn invert(a: &[f32], n: usize) -> Vec<f64> {
    let mut work: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| f64::from(a[i + j * n]))
                .chain((0..n).map(|j| if i == j { 1.0 } else { 0.0 }))
                .collect()
        })
        .collect();
    for pivot in 0..n {
        let scale = work[pivot][pivot];
        work[pivot].iter_mut().for_each(|value| *value /= scale);
        for row in 0..n {
            if row != pivot {
                let factor = work[row][pivot];
                let source = work[pivot].clone();
                for (target, value) in work[row].iter_mut().zip(source) {
                    *target -= factor * value;
                }
            }
        }
    }
    let mut inverse = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inverse[i + j * n] = work[i][n + j];
        }
    }
    inverse
}

#[test]
#[ignore = "needs a GPU, the native library and the math libraries"]
fn cholesky_with_the_lower_triangle_agrees_with_the_host_and_reports_info() {
    cholesky_round_trip(Fill::Lower);
}

#[test]
#[ignore = "needs a GPU, the native library and the math libraries"]
fn cholesky_with_the_upper_triangle_agrees_with_the_host_and_reports_info() {
    cholesky_round_trip(Fill::Upper);
}

/// Factors, inverts and solves with `fill`, and checks each answer and each info word against the host.
fn cholesky_round_trip(fill: Fill) {
    let Some(rig) = rig() else { return };
    let n = 6usize;
    let columns = 2usize;
    let a = spd(n);
    let inverse = invert(&a, n);
    // B = A X for a known X, so the solve returns X.
    let x: Vec<f32> = (0..n * columns).map(|index| 1.0 + index as f32).collect();
    let mut b = vec![0.0f32; n * columns];
    for column in 0..columns {
        for row in 0..n {
            b[row + column * n] = (0..n).map(|p| a[row + p * n] * x[p + column * n]).sum();
        }
    }
    // A matrix that is not positive definite: the factorisation reports the failing minor in `info`.
    let mut indefinite = spd(n);
    indefinite[2 + 2 * n] = -1.0;

    let for_inverse = rig.upload_f32(&a);
    let for_solve = rig.upload_f32(&a);
    let rhs = rig.upload_f32(&b);
    let bad = rig.upload_f32(&indefinite);
    let infos: Vec<DeviceBuffer> = (0..5).map(|_| rig.upload_i32(SENTINEL)).collect();
    let mut handles: Option<LaneHandles> = None;

    // The workspace size is known on the host when the query returns; the buffer is allocated after it.
    let (floats, sized) = rig
        .lane
        .submit_marked(|queue| {
            let stream = queue.raw_stream().expect("the lane lends a stream");
            let handles = handles.insert(rig.math.lane_handles().expect("handles"));
            // SAFETY: the matrix is a live device allocation of n * n floats; the query does not read it.
            let potrf = unsafe { handles.solver_workspace(&stream, false, fill, n as i32, for_solve.ptr()) };
            // SAFETY: as above.
            let potri = unsafe { handles.solver_workspace(&stream, true, fill, n as i32, for_solve.ptr()) };
            Ok(potrf.expect("potrf size").max(potri.expect("potri size")))
        })
        .expect("submit");
    rig.retired(sized);
    let workspace = rig.buffer(floats as usize * 4 + 4);

    let ((), marker) = rig
        .lane
        .submit_marked(|queue| {
            let stream = queue.raw_stream().expect("the lane lends a stream");
            let handles = handles.as_mut().expect("handles");
            let (n, w) = (n as i32, floats);
            // SAFETY: every buffer is a live device allocation of the size the call names (matrices n * n floats,
            // the right-hand side n * columns, the workspace at least the queried size, each info an int). They are kept
            // until the results are read after the retirement; the calls run in order on the lent stream, so the
            // workspace is not used by two calls at once.
            unsafe {
                let space = workspace.ptr();
                handles
                    .cholesky(&stream, false, fill, n, for_inverse.ptr(), space, w, infos[0].ptr())
                    .expect("potrf");
                handles
                    .cholesky(&stream, true, fill, n, for_inverse.ptr(), space, w, infos[1].ptr())
                    .expect("potri");
                handles
                    .cholesky(&stream, false, fill, n, for_solve.ptr(), space, w, infos[2].ptr())
                    .expect("potrf");
                handles
                    .solve_cholesky(
                        &stream,
                        fill,
                        n,
                        columns as i32,
                        for_solve.ptr(),
                        rhs.ptr(),
                        infos[3].ptr(),
                    )
                    .expect("potrs");
                handles
                    .cholesky(&stream, false, fill, n, bad.ptr(), space, w, infos[4].ptr())
                    .expect("potrf on an indefinite matrix is accepted");
            }
            Ok(())
        })
        .expect("submit");
    rig.retired(marker);

    for (index, info) in infos.iter().enumerate().take(4) {
        check_info("cusolver", download_i32(info)).unwrap_or_else(|error| panic!("info {index}: {error}"));
    }
    let failed = download_i32(&infos[4]);
    assert_eq!(
        failed, 3,
        "the third leading minor of the indefinite matrix is not positive"
    );
    assert!(check_info("cusolverDnSpotrf", failed).is_err());

    // The inverse is in the triangle `fill` names.
    let got = download_f32(&for_inverse, n * n);
    for j in 0..n {
        let rows = if fill == Fill::Lower { j..n } else { 0..j + 1 };
        for i in rows {
            let (value, expected) = (f64::from(got[i + j * n]), inverse[i + j * n]);
            assert!(
                (value - expected).abs() < 1e-5,
                "inverse ({i}, {j}): {value} against {expected}"
            );
        }
    }
    // The solve wrote X over B.
    let solved = download_f32(&rhs, n * columns);
    for (index, (value, expected)) in solved.iter().zip(&x).enumerate() {
        assert!(
            (value - expected).abs() < 1e-4,
            "solution {index}: {value} against {expected}"
        );
    }
    handles.take().expect("handles").close().expect("the handles close");
}
