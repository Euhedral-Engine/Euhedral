//! The FP32 products. The kernels and the library's matrices are row-major, cuBLAS is column-major, so each product
//! is the transposed product with its operands exchanged: `C^T = op(B)^T op(A)^T`.

use std::ffi::c_void;

use euhedral_gpu::RawStream;
use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_cuda::math::{LaneHandles, Matrix, Operation, Product};

use super::TrainError;
use super::plan::{address, dimension};

/// One FP32 product of row-major matrices, as the call the lane's cuBLAS handle makes. Built only by the planners.
#[derive(Clone, Copy, Debug)]
pub struct Gemm {
    product: Product,
    a: (u64, i32),
    b: (u64, i32),
    c: (u64, i32),
}

impl Gemm {
    /// The cuBLAS product: operations, shape and scales, in cuBLAS's column-major terms.
    pub fn product(&self) -> &Product {
        &self.product
    }

    /// The device addresses of the three operands as cuBLAS takes them, and their leading dimensions.
    pub fn operands(&self) -> [(u64, i32); 3] {
        [self.a, self.b, self.c]
    }

    /// Enqueues the product on `stream`, the stream the lane lends for this submission, with the lane's `handles`.
    ///
    /// # Safety
    /// The stream must be the lane's own, the three operands live device allocations of the sizes the shape and the
    /// leading dimensions imply, untouched by other work until the lane retires this one; `handles` must belong to
    /// that lane.
    pub unsafe fn enqueue(
        &self,
        handles: &mut LaneHandles,
        stream: &RawStream<'_, *mut c_void>,
    ) -> Result<(), TrainError> {
        let matrix = |(pointer, leading): (u64, i32)| Matrix {
            // SAFETY: the caller vouches for the addresses.
            pointer: unsafe { DevicePtr::from_raw(pointer) },
            leading,
        };
        // SAFETY: the caller's contract.
        unsafe { handles.gemm_fp32(stream, &self.product, matrix(self.a), matrix(self.b), matrix(self.c)) }
            .map_err(|error| TrainError::Gpu(error.to_string()))
    }
}

/// A row-major operand of `rows` by `columns` at `pointer` with leading dimension `leading`.
fn operand(name: &'static str, pointer: u64, leading: u32, columns: u32) -> Result<(u64, i32), TrainError> {
    let pointer = address(name, pointer)?;
    let signed = dimension(name, leading)?;
    if leading < columns {
        return Err(TrainError::ShortLeading {
            name,
            leading: signed,
            row: i32::try_from(columns).unwrap_or(i32::MAX),
        });
    }
    Ok((pointer, signed))
}

#[allow(clippy::too_many_arguments, reason = "a product's own parameters")]
fn product(
    operations: (Operation, Operation),
    shape: (u32, u32, u32),
    scales: (f32, f32),
    a: (u64, i32),
    b: (u64, i32),
    c: (u64, i32),
) -> Result<Gemm, TrainError> {
    let (m, n, k) = shape;
    Ok(Gemm {
        product: Product {
            a_operation: operations.0,
            b_operation: operations.1,
            m: dimension("m", m)?,
            n: dimension("n", n)?,
            k: dimension("k", k)?,
            alpha: scales.0,
            beta: scales.1,
        },
        a,
        b,
        c,
    })
}

/// `y [rows, out] = x [rows, in] w^T` (plus `y` when `accumulate`), with `w` `[out, in]`.
pub fn plan_linear(
    x: u64,
    w: u64,
    y: u64,
    rows: u32,
    inputs: u32,
    outputs: u32,
    accumulate: bool,
) -> Result<Gemm, TrainError> {
    product(
        (Operation::Transpose, Operation::None),
        (outputs, rows, inputs),
        (1.0, if accumulate { 1.0 } else { 0.0 }),
        operand("w", w, inputs, inputs)?,
        operand("x", x, inputs, inputs)?,
        operand("y", y, outputs, outputs)?,
    )
}

/// `dx [rows, in] += dy [rows, out] w`.
pub fn plan_linear_input_gradient(
    dy: u64,
    w: u64,
    dx: u64,
    rows: u32,
    inputs: u32,
    outputs: u32,
) -> Result<Gemm, TrainError> {
    product(
        (Operation::None, Operation::None),
        (inputs, rows, outputs),
        (1.0, 1.0),
        operand("w", w, inputs, inputs)?,
        operand("dy", dy, outputs, outputs)?,
        operand("dx", dx, inputs, inputs)?,
    )
}

/// `dw [out, in] = dy^T x`, overwriting `dw`.
pub fn plan_linear_weight_gradient(
    dy: u64,
    x: u64,
    dw: u64,
    rows: u32,
    inputs: u32,
    outputs: u32,
) -> Result<Gemm, TrainError> {
    product(
        (Operation::None, Operation::Transpose),
        (inputs, outputs, rows),
        (1.0, 0.0),
        operand("x", x, inputs, inputs)?,
        operand("dy", dy, outputs, outputs)?,
        operand("dw", dw, inputs, inputs)?,
    )
}

/// The operands of a product of `a` and `b` into `c`, each with its leading dimension.
#[derive(Clone, Copy, Debug)]
pub struct Matrices {
    /// The left operand.
    pub a: (u64, u32),
    /// The right operand.
    pub b: (u64, u32),
    /// The result.
    pub c: (u64, u32),
}

/// `c [m, n] = alpha a [m, k] b [k, n] + beta c`.
pub fn plan_product(matrices: &Matrices, m: u32, n: u32, k: u32, alpha: f32, beta: f32) -> Result<Gemm, TrainError> {
    product(
        (Operation::None, Operation::None),
        (n, m, k),
        (alpha, beta),
        operand("b", matrices.b.0, matrices.b.1, n)?,
        operand("a", matrices.a.0, matrices.a.1, k)?,
        operand("c", matrices.c.0, matrices.c.1, n)?,
    )
}

/// `c [m, n] = alpha a [m, k] b^T + beta c` with `b` stored `[n, k]`.
pub fn plan_product_transposed(
    matrices: &Matrices,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
) -> Result<Gemm, TrainError> {
    product(
        (Operation::Transpose, Operation::None),
        (n, m, k),
        (alpha, beta),
        operand("b", matrices.b.0, matrices.b.1, k)?,
        operand("a", matrices.a.0, matrices.a.1, k)?,
        operand("c", matrices.c.0, matrices.c.1, n)?,
    )
}

/// `c [m, n] = alpha a^T b [k, n] + beta c` with `a` stored `[k, m]`.
pub fn plan_product_left_transposed(
    matrices: &Matrices,
    m: u32,
    n: u32,
    k: u32,
    alpha: f32,
    beta: f32,
) -> Result<Gemm, TrainError> {
    product(
        (Operation::None, Operation::Transpose),
        (n, m, k),
        (alpha, beta),
        operand("b", matrices.b.0, matrices.b.1, n)?,
        operand("a", matrices.a.0, matrices.a.1, m)?,
        operand("c", matrices.c.0, matrices.c.1, n)?,
    )
}
