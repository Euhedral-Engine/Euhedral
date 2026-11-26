//! The draft head's two BF16 products: the logits of a chunk of rows and the gradient they send back into the drafter's
//! output. The head is the artifact's, frozen, so it has no gradient of its own.

use std::ffi::c_void;

use euhedral_gpu::RawStream;
use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_cuda::math::{GemmOutput, LaneHandles, Matrix, Operation, Product};

use crate::training::{TrainError, address, dimension};

/// One product with BF16 operands and an FP32 result, as the call the lane's cuBLAS handle makes. Built only by the
/// planners.
#[derive(Clone, Copy, Debug)]
pub struct HeadProduct {
    product: Product,
    a: (u64, i32),
    b: (u64, i32),
    c: (u64, i32),
}

impl HeadProduct {
    /// The cuBLAS product: operations, shape and scales, in cuBLAS's column-major terms.
    pub fn product(&self) -> &Product {
        &self.product
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
        unsafe {
            handles.gemm_bf16(
                stream,
                &self.product,
                matrix(self.a),
                matrix(self.b),
                matrix(self.c),
                GemmOutput::Fp32,
            )
        }
        .map_err(|error| TrainError::Gpu(error.to_string()))
    }
}

/// The draft head's logits `[rows, draft_rows]` (FP32) of `rows` BF16 hidden rows `[rows, hidden]` at `input`, against
/// the head `[draft_rows, hidden]` (BF16).
pub fn plan_head_logits(
    head: u64,
    input: u64,
    logits: u64,
    rows: u32,
    hidden: u32,
    draft_rows: u32,
) -> Result<HeadProduct, TrainError> {
    Ok(HeadProduct {
        product: Product {
            a_operation: Operation::Transpose,
            b_operation: Operation::None,
            m: dimension("draft_rows", draft_rows)?,
            n: dimension("rows", rows)?,
            k: dimension("hidden", hidden)?,
            alpha: 1.0,
            beta: 0.0,
        },
        a: (address("head", head)?, dimension("hidden", hidden)?),
        b: (address("input", input)?, dimension("hidden", hidden)?),
        c: (address("logits", logits)?, dimension("draft_rows", draft_rows)?),
    })
}

/// Adds the head's gradient into `gradient` `[rows, hidden]` (FP32): the BF16 logits' gradient `[rows, draft_rows]` at
/// `logits` times the head.
pub fn plan_head_gradient(
    head: u64,
    logits: u64,
    gradient: u64,
    rows: u32,
    hidden: u32,
    draft_rows: u32,
) -> Result<HeadProduct, TrainError> {
    Ok(HeadProduct {
        product: Product {
            a_operation: Operation::None,
            b_operation: Operation::None,
            m: dimension("hidden", hidden)?,
            n: dimension("rows", rows)?,
            k: dimension("draft_rows", draft_rows)?,
            alpha: 1.0,
            beta: 1.0,
        },
        a: (address("head", head)?, dimension("hidden", hidden)?),
        b: (address("logits", logits)?, dimension("draft_rows", draft_rows)?),
        c: (address("gradient", gradient)?, dimension("hidden", hidden)?),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_logits_product_is_the_transposed_head_times_the_rows() {
        let product = plan_head_logits(0x100, 0x200, 0x300, 7, 64, 32).unwrap();
        let shape = product.product();
        assert_eq!((shape.m, shape.n, shape.k), (32, 7, 64));
        assert_eq!((shape.alpha, shape.beta), (1.0, 0.0));
        assert_eq!(product.a, (0x100, 64));
        assert_eq!(product.b, (0x200, 64));
        assert_eq!(product.c, (0x300, 32));
    }

    #[test]
    fn the_gradient_product_adds_into_the_output_gradient() {
        let product = plan_head_gradient(0x100, 0x200, 0x300, 7, 64, 32).unwrap();
        let shape = product.product();
        assert_eq!((shape.m, shape.n, shape.k), (64, 7, 32));
        assert_eq!((shape.alpha, shape.beta), (1.0, 1.0));
        assert_eq!(product.b, (0x200, 32));
        assert_eq!(product.c, (0x300, 64));
    }

    #[test]
    fn a_null_address_or_an_empty_chunk_is_refused() {
        assert!(plan_head_logits(0, 1, 1, 7, 64, 32).is_err());
        assert!(plan_head_logits(1, 1, 1, 0, 64, 32).is_err());
        assert!(plan_head_gradient(1, 1, 0, 7, 64, 32).is_err());
        assert!(plan_head_gradient(1, 1, 1, 7, 64, 0).is_err());
    }
}
