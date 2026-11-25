//! The lane's operations a layer needs beyond the engine's kernels: cuBLAS products, the Hessian accumulation and the
//! row gather.

use std::ffi::c_void;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use euhedral_gpu::cuda::{CudaKernel, CudaModule};
use euhedral_gpu::{Args, Dim3, LaneEnqueue, LaneStream, Launch};
use euhedral_inference_cuda::launch::ops::RusteeOps;
use euhedral_inference_cuda::math::{GemmOutput, LaneHandles, Matrix, Operation, Product};
use euhedral_inference_cuda::{CudaError, DevicePtr};

use super::ForwardError;
use crate::kernels::Compiled;

/// The width of a column block of the Hessian: the product of two blocks is one cuBLAS call.
const HESSIAN_BLOCK: u32 = 1024;
const MIRROR_BLOCK: u32 = 256;
const GATHER_BLOCK: u32 = 256;

/// The two kernels this module launches itself, resolved from their modules.
#[derive(Clone, Debug)]
pub struct ForwardKernels {
    mirror_upper: CudaKernel,
    gather_rows: CudaKernel,
}

impl ForwardKernels {
    /// The kernels from the compiled GPTQ module (`mirror_upper`) and rows module (`gather_rows`).
    pub fn new(gptq: &Compiled<CudaModule>, rows: &Compiled<CudaModule>) -> Result<ForwardKernels, ForwardError> {
        let find = |compiled: &Compiled<CudaModule>, name: &str| {
            compiled
                .kernel(name)
                .cloned()
                .ok_or_else(|| ForwardError::Invalid(format!("the module has no kernel {name}")))
        };
        Ok(ForwardKernels {
            mirror_upper: find(gptq, "mirror_upper")?,
            gather_rows: find(rows, "gather_rows")?,
        })
    }
}

/// The launch of `mirror_upper` over a `k` by `k` column-major matrix at `hessian`: its lower triangle from its upper.
pub fn plan_mirror_upper(hessian: DevicePtr, k: u32) -> Result<(Launch, Args), ForwardError> {
    let Ok(columns) = i32::try_from(k) else {
        return Err(ForwardError::Invalid(format!(
            "{k} columns do not fit the kernel's index"
        )));
    };
    if hessian.is_null() || k == 0 {
        return Err(ForwardError::Invalid(format!(
            "a mirror of {k} columns at address {}",
            hessian.get()
        )));
    }
    let count = u64::from(k) * u64::from(k);
    let blocks = u32::try_from(count.div_ceil(u64::from(MIRROR_BLOCK)))
        .map_err(|_| ForwardError::Invalid(format!("{k} columns make a grid too large")))?;
    Ok((
        Launch {
            grid: Dim3::x(blocks),
            block: Dim3::x(MIRROR_BLOCK),
            shared_bytes: 0,
            programmatic: false,
        },
        Args::new().device(hessian.get()).arg(columns),
    ))
}

/// The launch of `gather_rows`: `out[i, :] = table[ids[i], :]` for `count` BF16 rows of `width` elements, `width` a
/// multiple of 8.
pub fn plan_gather_rows(
    table: DevicePtr,
    ids: DevicePtr,
    count: u32,
    width: u32,
    out: DevicePtr,
) -> Result<(Launch, Args), ForwardError> {
    let Ok(row_width) = i32::try_from(width) else {
        return Err(ForwardError::Invalid(format!(
            "a width of {width} does not fit the kernel's index"
        )));
    };
    if table.is_null() || ids.is_null() || out.is_null() || count == 0 || width == 0 || !width.is_multiple_of(8) {
        return Err(ForwardError::Invalid(format!(
            "a gather of {count} rows of {width} values (a multiple of 8) from {} with ids {} to {}",
            table.get(),
            ids.get(),
            out.get()
        )));
    }
    let threads = u64::from(count) * u64::from(width / 8);
    let blocks = u32::try_from(threads.div_ceil(u64::from(GATHER_BLOCK)))
        .map_err(|_| ForwardError::Invalid(format!("{count} rows of {width} make a grid too large")))?;
    Ok((
        Launch {
            grid: Dim3::x(blocks),
            block: Dim3::x(GATHER_BLOCK),
            shared_bytes: 0,
            programmatic: false,
        },
        Args::new()
            .device(table.get())
            .device(ids.get())
            .arg(i64::from(count))
            .arg(row_width)
            .device(out.get()),
    ))
}

/// The lane's operations for one submission: the engine's kernel operations, the lane's cuBLAS handles and the
/// kernels of [`ForwardKernels`]. Everything is queued on the lane's stream, in the order of the calls.
pub struct LaneContext<'a, Q> {
    ops: RusteeOps<'a, Q>,
    handles: &'a mut LaneHandles,
    kernels: &'a ForwardKernels,
}

impl<'a, Q> LaneContext<'a, Q>
where
    Q: LaneEnqueue<Kernel = CudaKernel> + LaneStream<Raw = *mut c_void>,
{
    /// The operations of `ops`, with `handles` (this lane's) for the products.
    pub fn new(ops: RusteeOps<'a, Q>, handles: &'a mut LaneHandles, kernels: &'a ForwardKernels) -> Self {
        Self { ops, handles, kernels }
    }

    /// The engine's kernel and allocation operations on this lane.
    pub fn ops(&mut self) -> &mut RusteeOps<'a, Q> {
        &mut self.ops
    }

    /// `output [rows, n] = input [rows, k] * weights [n, k]^T`, BF16 with FP32 accumulation.
    pub fn linear(
        &mut self,
        input: DevicePtr,
        weights: DevicePtr,
        output: DevicePtr,
        rows: u32,
        k: u32,
        n: u32,
    ) -> Result<(), ForwardError> {
        let product = Product {
            a_operation: Operation::Transpose,
            b_operation: Operation::None,
            m: signed(n)?,
            n: signed(rows)?,
            k: signed(k)?,
            alpha: 1.0,
            beta: 0.0,
        };
        let (a, b, c) = (
            Matrix {
                pointer: weights,
                leading: signed(k)?,
            },
            Matrix {
                pointer: input,
                leading: signed(k)?,
            },
            Matrix {
                pointer: output,
                leading: signed(n)?,
            },
        );
        self.product(&product, a, b, c, GemmOutput::Bf16)
    }

    /// Adds `x^T x` to the upper blocks of the column-major FP32 Hessian `hessian [k, k]`, for `x` BF16 `[rows, k]`.
    /// Only blocks on or above the diagonal are computed; [`LaneContext::symmetrize`] completes the matrix once the
    /// accumulation is done.
    pub fn accumulate_hessian(
        &mut self,
        hessian: DevicePtr,
        x: DevicePtr,
        rows: u32,
        k: u32,
    ) -> Result<(), ForwardError> {
        let leading = signed(k)?;
        let mut i = 0;
        while i < k {
            let height = HESSIAN_BLOCK.min(k - i);
            let mut j = i;
            while j < k {
                let width = HESSIAN_BLOCK.min(k - j);
                let product = Product {
                    a_operation: Operation::None,
                    b_operation: Operation::Transpose,
                    m: signed(height)?,
                    n: signed(width)?,
                    k: signed(rows)?,
                    alpha: 1.0,
                    beta: 1.0,
                };
                let a = Matrix {
                    pointer: x.offset(2 * u64::from(i)),
                    leading,
                };
                let b = Matrix {
                    pointer: x.offset(2 * u64::from(j)),
                    leading,
                };
                let c = Matrix {
                    pointer: hessian.offset(4 * (u64::from(j) * u64::from(k) + u64::from(i))),
                    leading,
                };
                self.product(&product, a, b, c, GemmOutput::Fp32)?;
                j += HESSIAN_BLOCK;
            }
            i += HESSIAN_BLOCK;
        }
        Ok(())
    }

    /// Completes a Hessian accumulated by [`LaneContext::accumulate_hessian`]: its lower triangle from its upper one.
    pub fn symmetrize(&mut self, hessian: DevicePtr, k: u32) -> Result<(), ForwardError> {
        let (launch, args) = plan_mirror_upper(hessian, k)?;
        // SAFETY: `hessian` is a `k` by `k` FP32 device matrix the caller keeps until the lane retires this launch;
        // the arguments are in the kernel's parameter order.
        unsafe { self.ops.launch_kernel(&self.kernels.mirror_upper, launch, &args) }?;
        Ok(())
    }

    /// Gathers `count` BF16 rows of `width` elements from `table` by the `i32` `ids` into `out`.
    pub fn gather_rows(
        &mut self,
        table: DevicePtr,
        ids: DevicePtr,
        count: u32,
        width: u32,
        out: DevicePtr,
    ) -> Result<(), ForwardError> {
        let (launch, args) = plan_gather_rows(table, ids, count, width, out)?;
        // SAFETY: the caller keeps the table, the ids and the output live, with the sizes the arguments imply, until
        // the lane retires the launch; the arguments are in the kernel's parameter order.
        unsafe { self.ops.launch_kernel(&self.kernels.gather_rows, launch, &args) }?;
        Ok(())
    }

    fn product(
        &mut self,
        product: &Product,
        a: Matrix,
        b: Matrix,
        c: Matrix,
        output: GemmOutput,
    ) -> Result<(), ForwardError> {
        let handles = &mut *self.handles;
        // SAFETY: the operands are device allocations of the sizes the product implies, which the caller keeps live and
        // untouched by other work until the lane retires the product; the stream is this lane's, lent for this
        // submission.
        let done: Result<(), CudaError> = self
            .ops
            .with_raw_stream(|stream| unsafe { handles.gemm_bf16(stream, product, a, b, c, output) })
            .and_then(|result| result);
        Ok(done?)
    }
}

fn signed(value: u32) -> Result<i32, ForwardError> {
    i32::try_from(value).map_err(|_| ForwardError::Invalid(format!("{value} does not fit a signed 32-bit extent")))
}

impl<Q> Debug for LaneContext<'_, Q> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("LaneContext").finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const ADDRESS: u64 = 0x1000;

    fn pointer(address: u64) -> DevicePtr {
        // SAFETY: the planners only carry the address into the arguments; nothing dereferences it.
        unsafe { DevicePtr::from_raw(address) }
    }

    #[test]
    fn a_mirror_covers_every_element_of_the_matrix_with_one_thread() {
        let (launch, _) = plan_mirror_upper(pointer(ADDRESS), 1100).unwrap();
        assert!(u64::from(launch.grid.x) * u64::from(launch.block.x) >= 1100 * 1100);
        assert!(u64::from(launch.grid.x - 1) * u64::from(launch.block.x) < 1100 * 1100);
    }

    #[test]
    fn a_mirror_with_no_matrix_or_no_columns_is_refused() {
        assert!(plan_mirror_upper(pointer(0), 8).is_err());
        assert!(plan_mirror_upper(pointer(ADDRESS), 0).is_err());
        assert!(plan_mirror_upper(pointer(ADDRESS), u32::MAX).is_err());
    }

    #[test]
    fn a_gather_gives_one_thread_to_each_eight_values_and_refuses_a_ragged_width() {
        let (launch, _) = plan_gather_rows(pointer(ADDRESS), pointer(ADDRESS), 5, 24, pointer(ADDRESS)).unwrap();
        assert!(u64::from(launch.grid.x) * u64::from(launch.block.x) >= 15);
        assert!(plan_gather_rows(pointer(ADDRESS), pointer(ADDRESS), 5, 20, pointer(ADDRESS)).is_err());
        assert!(plan_gather_rows(pointer(ADDRESS), pointer(ADDRESS), 0, 24, pointer(ADDRESS)).is_err());
        assert!(plan_gather_rows(pointer(ADDRESS), pointer(0), 5, 24, pointer(ADDRESS)).is_err());
    }
}
