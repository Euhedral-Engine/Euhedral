//! Unquantized BF16 linears (`Bf16Format.linear`, `BF16_PROJECTION`): the projection runs on the row-exact BF16 GEMMs,
//! whose rows do not depend on how many launch together, so a verification and a prefill chunk compute each row as a
//! one-row launch does. The route takes no scratch.
//!
//! The Gated DeltaNet A and B projections are BF16 in every artifact and leave through the gated control launch
//! (`euhedral_gdn_control_fp32`, the stage's own), not through this format.

use euhedral_inference_cuda::{CudaError, DevicePtr, KernelOps};
use euhedral_inference_edrl::{TensorDataType, WeightFormat, WeightLayout};

use crate::formats::row_exact::{argument, row_exact_bf16};
use crate::formats::shapes::Launch;
use crate::formats::{LinearFormat, LinearMode, ScratchRegion};
use crate::model::WeightRef;

/// The BF16 projections.
#[derive(Clone, Copy, Debug, Default)]
pub struct Bf16Linear;

impl Bf16Linear {
    /// Whether the route can take `weight`, decided from its description alone: a contiguous BF16 matrix of `n` outputs
    /// by `k` inputs with `k` divisible by 32 and `n` by 8, holding exactly `2 n k` bytes.
    pub fn check(weight: &WeightRef) -> Result<(), String> {
        let name = &weight.name;
        if weight.format != WeightFormat::Bf16 || weight.layout != WeightLayout::ContiguousLeV1 {
            return Err(format!(
                "{name} is {:?}/{:?}, not contiguous BF16",
                weight.format, weight.layout
            ));
        }
        if weight.dtype != TensorDataType::Bf16 {
            return Err(format!("{name} holds {:?} values, not BF16", weight.dtype));
        }
        let [n, k] = weight.shape.as_slice() else {
            return Err(format!("{name} has shape {:?}, not [outputs, inputs]", weight.shape));
        };
        if *k == 0 || *n == 0 || !k.is_multiple_of(32) || !n.is_multiple_of(8) {
            return Err(format!(
                "{name} is {n} x {k}: a row-exact BF16 linear needs K divisible by 32 and N by 8"
            ));
        }
        if n.checked_mul(*k).and_then(|values| values.checked_mul(2)) != Some(weight.bytes) {
            return Err(format!(
                "{name} holds {} bytes, not those of a BF16 matrix of {n} x {k}",
                weight.bytes
            ));
        }
        Ok(())
    }

    /// The launch of `rows` rows at `x` over `weight`, the result at `y`.
    pub fn launch(weight: &WeightRef, x: DevicePtr, y: DevicePtr, rows: usize) -> Result<Launch, CudaError> {
        Self::check(weight).map_err(argument)?;
        let mut planned = None;
        row_exact_bf16(
            x,
            weight.ptr,
            y,
            rows as u64,
            weight.shape[1],
            weight.shape[0],
            &mut |launch| {
                planned = Some(launch);
                Ok(())
            },
        )?;
        planned.ok_or_else(|| argument(format!("{} planned no launch", weight.name)))
    }
}

impl LinearFormat for Bf16Linear {
    fn submit(
        &self,
        kernels: &mut dyn KernelOps,
        weight: &WeightRef,
        x: DevicePtr,
        y: DevicePtr,
        rows: usize,
        _scratch: ScratchRegion,
        _mode: LinearMode,
    ) -> Result<(), CudaError> {
        Self::launch(weight, x, y, rows)?.queue(kernels)
    }

    fn scratch_bytes(&self, _weight: &WeightRef, _rows: usize, _mode: LinearMode) -> usize {
        0
    }
}
