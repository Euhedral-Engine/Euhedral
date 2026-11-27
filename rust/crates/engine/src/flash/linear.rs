//! The NVFP4 linears of the stages with Java's route (`CudaGpuMemory.linearNvfp4Bf16`): 1 to 8 rows on the decode
//! kernels, 9 rows and more on the native route, which quantizes the activations into an expansion scratch, and the
//! decode or reference kernels where the native route declines. The route is a function of the rows, the shape, the
//! weight layout, the numerics mode and the device, so a stage settles it when it builds its operations.

use euhedral_inference_cuda::launch::nvfp4::{
    NATIVE_MIN_ROWS, Nvfp4Route, SKINNY_MAX_ROWS, nvfp4_native_scratch_bytes, route, skinny_target_ctas,
};
use euhedral_inference_cuda::{CudaLibrary, DevicePtr};

use super::sequence::Op;
use super::{FlashError, Weight};

/// An NVFP4 weight of a stage: its tensor and whether its scale table is the SD4 layout.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4Weight {
    pub weight: Weight,
    pub sd4: bool,
}

/// The expansion scratch the native route quantizes the activations into.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Expansion {
    pub address: DevicePtr,
    pub bytes: u64,
}

/// What the device offers the NVFP4 linears.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Nvfp4Device {
    native_available: bool,
    /// The grid the skinny kernels split over K to fill: two blocks per multiprocessor.
    target_ctas: u64,
}

impl Nvfp4Device {
    /// The linears of `library`'s device.
    pub fn new(library: &CudaLibrary) -> Self {
        Self::with_native_route(library.nvfp4_native_available(), library.device().sm_count)
    }

    /// A device of `sm_count` multiprocessors that has the native route or does not.
    pub fn with_native_route(native_available: bool, sm_count: u32) -> Self {
        Self {
            native_available,
            target_ctas: skinny_target_ctas(sm_count),
        }
    }

    /// The expansion scratch a linear of `rows` rows over `in_features` by `out_features` needs: enough for either
    /// kernel of the native route, none below the rows it starts at.
    pub fn expansion_bytes(&self, rows: u32, in_features: u32, out_features: u32) -> u64 {
        if !self.native_available || rows < NATIVE_MIN_ROWS {
            return 0;
        }
        let (inputs, outputs, rows) = (u64::from(in_features), u64::from(out_features), u64::from(rows));
        nvfp4_native_scratch_bytes(rows, inputs, outputs, self.target_ctas).max(nvfp4_native_scratch_bytes(
            rows.min(u64::from(SKINNY_MAX_ROWS)),
            inputs,
            outputs,
            self.target_ctas,
        ))
    }

    /// The linear `output = input * weight^T` of `rows` rows, on the route Java takes for it. A lane's row-exact or
    /// exact numerics are settled when the operation is queued ([`Op::queue`]), which skips the native route there.
    #[allow(
        clippy::too_many_arguments,
        reason = "the operands of one linear, as the kernels take them"
    )]
    pub fn linear(
        &self,
        input: DevicePtr,
        weight: Nvfp4Weight,
        output: DevicePtr,
        rows: u32,
        in_features: u32,
        out_features: u32,
        expansion: Expansion,
    ) -> Result<Op<'static>, FlashError> {
        let routed = route(
            rows,
            in_features,
            out_features,
            weight.sd4,
            false,
            false,
            self.native_available,
        );
        let native = if matches!(routed, Nvfp4Route::Native { .. }) {
            let needed = self.expansion_bytes(rows, in_features, out_features);
            if expansion.bytes < needed {
                return Err(FlashError::refused(
                    "nvfp4 linear",
                    format!(
                        "{rows} rows over {in_features} by {out_features} need {needed} bytes of expansion scratch, the region holds {}",
                        expansion.bytes
                    ),
                ));
            }
            Some(expansion)
        } else {
            None
        };
        Ok(Op::Nvfp4Linear {
            input,
            weights: weight.weight.address,
            output,
            rows,
            in_features,
            out_features,
            weights_bytes: weight.weight.bytes,
            native,
        })
    }
}

#[cfg(test)]
mod tests;
