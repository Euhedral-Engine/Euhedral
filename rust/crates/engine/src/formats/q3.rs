//! Grouped linears: Q3, Q4 and Q5 values in 64-value groups with FP16 scales, and Q3 entropy coded as P2E2 (the routes
//! of `GroupedLinearFormat` and `P2e2Format`).
//!
//! The route is a function of the rows, the shape and the layout:
//!
//! - **9 rows and more** on a device with block-scaled FP8 tensor cores (and not for a verification, which computes
//!   each row as a one-row launch does): the FP8 route, which quantizes the activations into the scratch. It declines
//!   shapes it has no kernel for and runs under no exact numerics, and the launch then takes the route below.
//! - **Anything else**: the decode kernels (up to 8 rows contiguous, up to 32 on the token-row twins) or, for a shape
//!   or an alignment they do not take and under exact numerics, the scalar reference.
//! - **A P2E2 tensor** of up to 8 rows runs on its fused decode kernels where they take the launch; every other launch
//!   expands the tensor into the scratch first and runs the row-split route on the expansion, so the outputs are the
//!   same bits in every case.
//! - **A Q3 gate/up projection of 64 rows and more** on the FP8 route runs with the SwiGLU after it as one paired
//!   launch that never splits K (`GATE_UP_MIN_ROWS`, the rows from which the plan fuses them): the unpaired
//!   route splits K by the tile count against the device's multiprocessors, which sums in another order.

#[cfg(test)]
use euhedral_inference_cuda::launch::q3::q3_decode_shape;
use euhedral_inference_cuda::launch::q3::{
    Bits, DECODE_MAX_ROWS, MX_MIN_ROWS, P2E2_EXPAND_LIMIT, linear_scratch_bytes, mx_scratch_bytes, p2e2_chunk_offsets,
    p2e2_chunk_rows, q3_byte_size,
};
use euhedral_inference_cuda::sys::EUHEDRAL_CUDA_ROUTE_UNAVAILABLE;
use euhedral_inference_cuda::{CudaError, CudaLibrary, DevicePtr, KernelOps};
use euhedral_inference_edrl::{WeightFormat, WeightLayout};

use crate::formats::{LinearFormat, LinearMode, ScratchRegion};
use crate::model::WeightRef;

/// Rows from which a Q3 gate/up projection and its SwiGLU run as one launch.
pub const GATE_UP_MIN_ROWS: usize = 64;

/// The Q3, Q4 and Q5 linears of a device.
#[derive(Debug)]
pub struct Q3Linear {
    mx_available: bool,
    sm_count: u32,
    exact_numerics: bool,
}

impl Q3Linear {
    /// Linears on `library`'s device: the FP8 route runs where the device has it.
    pub fn new(library: &CudaLibrary) -> Self {
        let linears = Self::with_mx_route(library.q3_mx_available(), library.device().sm_count);
        if library.exact_numerics() {
            linears.with_exact_numerics()
        } else {
            linears
        }
    }

    /// Linears on a device of `sm_count` streaming multiprocessors that has the FP8 route or does not (for tests).
    pub fn with_mx_route(mx_available: bool, sm_count: u32) -> Self {
        Self {
            mx_available,
            sm_count,
            exact_numerics: false,
        }
    }

    /// The same linears for a library that runs the exact kernels, which keep every tensor on the scalar reference
    /// and expand a P2E2 tensor whatever its rows: the scratch is sized for that.
    pub fn with_exact_numerics(mut self) -> Self {
        self.exact_numerics = true;
        self
    }

    /// The bytes of FP8 activations a launch of `rows` rows over `shape` takes where the FP8 route runs it: none where it
    /// does not (too few rows, no device support, a verification, a width it has no kernel for).
    /// `split_rows` names the whole tensor's weight rows for a run in output-row chunks (0 for a tensor run whole).
    /// `tiled_outputs`: whether the outputs must be whole tiles too (a P2E2 tensor reads its expansion, which has
    /// none of this to say about the compressed one).
    fn mx_reserve(&self, rows: u32, shape: Shape, row_exact: bool, tiled_outputs: bool, split_rows: u32) -> u64 {
        let Shape { out, input } = shape;
        if rows >= MX_MIN_ROWS
            && self.mx_available
            && !row_exact
            && input.is_multiple_of(128)
            && (!tiled_outputs || out.is_multiple_of(128))
        {
            mx_scratch_bytes(rows, input, out, self.sm_count, split_rows)
        } else {
            0
        }
    }

    /// The scratch bytes a launch of exactly `rows` rows over a tensor of `shape` asks of its region when it goes
    /// the route the host expects: [`LinearFormat::scratch_bytes`] covers it in every mode.
    #[cfg(test)]
    fn demand(&self, p2e2: bool, shape: Shape, rows: u32, row_exact: bool) -> u64 {
        if !p2e2 {
            return self.mx_reserve(rows, shape, row_exact, true, 0);
        }
        if rows <= DECODE_MAX_ROWS && !self.exact_numerics && q3_decode_shape(shape.input, shape.out) {
            return 0;
        }
        q3_byte_size(shape.out, shape.input).next_multiple_of(256) + self.mx_reserve(rows, shape, row_exact, false, 0)
    }

    /// Expands the P2E2 tensor `weight` into `scratch`, with room behind it for `reserve` bytes of activations, and
    /// returns the expansion's size and the room behind it.
    fn expand(
        kernels: &mut dyn KernelOps,
        weight: &WeightRef,
        shape: Shape,
        rows: u32,
        scratch: ScratchRegion,
        reserve: u64,
    ) -> Result<(u64, ScratchRegion), CudaError> {
        let Shape { out, input } = shape;
        let expanded = q3_byte_size(out, input);
        let offset = expanded.next_multiple_of(256);
        if (scratch.bytes as u64) < offset + reserve {
            return Err(CudaError::Argument(format!(
                "a launch of {rows} rows over {} needs {} bytes of scratch and has {}",
                weight.name,
                offset + reserve,
                scratch.bytes
            )));
        }
        kernels.q3_p2e2_expand(weight.ptr, weight.bytes, out, input, 0, out, scratch.ptr, expanded)?;
        Ok((
            expanded,
            ScratchRegion {
                ptr: scratch.ptr.offset(offset),
                bytes: reserve as usize,
            },
        ))
    }

    /// Runs a P2E2 tensor too large to expand whole in chunks of output rows: each chunk is expanded into the scratch,
    /// run there, and its outputs copied into place. A Q3 linear computes an output from its own weight row alone, and
    /// the FP8 route's split of K follows the whole tensor, so the outputs equal those of the whole tensor.
    #[allow(clippy::too_many_arguments, reason = "one launch's operands")]
    fn run_p2e2_in_chunks(
        &self,
        kernels: &mut dyn KernelOps,
        weight: &WeightRef,
        shape: Shape,
        x: DevicePtr,
        y: DevicePtr,
        rows: u32,
        scratch: ScratchRegion,
        row_exact: bool,
    ) -> Result<(), CudaError> {
        let Shape { out, input } = shape;
        let chunk = p2e2_chunk_rows(rows, input, out, Some(self.sm_count));
        let mut first = 0;
        while first < out {
            let count = chunk.min(out - first);
            let part = Shape { out: count, input };
            let expanded = q3_byte_size(count, input);
            let (output_offset, activation_offset) = p2e2_chunk_offsets(rows, input, count);
            let reserve = self.mx_reserve(rows, part, row_exact, false, out);
            if (scratch.bytes as u64) < activation_offset + reserve {
                return Err(CudaError::Argument(format!(
                    "a launch of {rows} rows over {}, too large to expand whole ({P2E2_EXPAND_LIMIT} bytes), needs {} bytes of scratch for a chunk of {count} rows and has {}",
                    weight.name,
                    activation_offset + reserve,
                    scratch.bytes
                )));
            }
            kernels.q3_p2e2_expand(
                weight.ptr,
                weight.bytes,
                out,
                input,
                first,
                count,
                scratch.ptr,
                expanded,
            )?;
            let result = scratch.ptr.offset(output_offset);
            self.run_row_split(
                kernels,
                Bits::Q3,
                part,
                x,
                scratch.ptr,
                expanded,
                result,
                rows,
                ScratchRegion {
                    ptr: scratch.ptr.offset(activation_offset),
                    bytes: reserve as usize,
                },
                row_exact,
                out,
            )?;
            let value_bytes = 2;
            kernels.copy_device_to_device_2d(
                y.offset(u64::from(first) * value_bytes),
                u64::from(out) * value_bytes,
                result,
                u64::from(count) * value_bytes,
                u64::from(count) * value_bytes,
                u64::from(rows),
            )?;
            first += count;
        }
        Ok(())
    }

    /// Runs a row-split tensor of `bits` at `weights`: the FP8 route where it runs and `activations` holds its scratch,
    /// else the decode kernels.
    #[allow(clippy::too_many_arguments, reason = "one launch's operands")]
    fn run_row_split(
        &self,
        kernels: &mut dyn KernelOps,
        bits: Bits,
        shape: Shape,
        x: DevicePtr,
        weights: DevicePtr,
        weights_bytes: u64,
        y: DevicePtr,
        rows: u32,
        activations: ScratchRegion,
        row_exact: bool,
        split_rows: u32,
    ) -> Result<(), CudaError> {
        let Shape { out, input } = shape;
        let needed = self.mx_reserve(rows, shape, row_exact, true, split_rows);
        if needed > 0 {
            if (activations.bytes as u64) < needed {
                return Err(CudaError::Argument(format!(
                    "a {}-bit linear of {rows} rows needs {needed} bytes of scratch and has {}",
                    bits.number(),
                    activations.bytes
                )));
            }
            let mx = kernels.linear_q3_mx_bf16(
                bits.number(),
                x,
                weights,
                y,
                activations.ptr,
                rows,
                input,
                out,
                weights_bytes,
                activations.bytes as u64,
                split_rows,
            );
            match mx {
                // The FP8 route declines under exact numerics and for alignments it has no kernel for.
                Err(CudaError::Call {
                    status: EUHEDRAL_CUDA_ROUTE_UNAVAILABLE,
                    ..
                }) => {}
                other => return other,
            }
        }
        match bits {
            Bits::Q3 => kernels.linear_q3_bf16(x, weights, y, rows, input, out, weights_bytes),
            Bits::Q4 | Bits::Q5 => {
                kernels.linear_quantized_bf16(x, weights, y, rows, input, out, weights_bytes, bits.number())
            }
        }
    }
}

impl LinearFormat for Q3Linear {
    fn submit(
        &self,
        kernels: &mut dyn KernelOps,
        weight: &WeightRef,
        x: DevicePtr,
        y: DevicePtr,
        rows: usize,
        scratch: ScratchRegion,
        mode: LinearMode,
    ) -> Result<(), CudaError> {
        let (bits, shape, p2e2) = describe(weight)?;
        let rows = u32::try_from(rows)
            .ok()
            .filter(|rows| *rows > 0)
            .ok_or_else(|| CudaError::Argument(format!("a grouped linear takes at least one row, not {rows}")))?;
        // A verification may be selected on the lane as well as asked for by the stage.
        let row_exact = mode.row_exact || kernels.row_exact();
        if !p2e2 {
            return self.run_row_split(
                kernels,
                bits,
                shape,
                x,
                weight.ptr,
                weight.bytes,
                y,
                rows,
                scratch,
                row_exact,
                0,
            );
        }
        let Shape { out, input } = shape;
        if rows <= DECODE_MAX_ROWS {
            let decode = kernels.linear_q3_p2e2_decode_bf16(x, weight.ptr, y, rows, input, out, weight.bytes);
            match decode {
                Err(CudaError::Call {
                    status: EUHEDRAL_CUDA_ROUTE_UNAVAILABLE,
                    ..
                }) => {}
                other => return other,
            }
        }
        // Every other launch expands the tensor into the scratch and runs the row-split route on the expansion, with
        // the activations of the FP8 route behind it.
        if q3_byte_size(out, input) > P2E2_EXPAND_LIMIT {
            return self.run_p2e2_in_chunks(kernels, weight, shape, x, y, rows, scratch, row_exact);
        }
        let reserve = self.mx_reserve(rows, shape, row_exact, false, 0);
        let (expanded, activations) = Self::expand(kernels, weight, shape, rows, scratch, reserve)?;
        self.run_row_split(
            kernels,
            Bits::Q3,
            shape,
            x,
            scratch.ptr,
            expanded,
            y,
            rows,
            activations,
            row_exact,
            0,
        )
    }

    fn fuses_gate_up_swiglu(&self, weight: &WeightRef, rows: usize, mode: LinearMode) -> bool {
        let Ok((Bits::Q3, Shape { out, input }, p2e2)) = describe(weight) else {
            return false;
        };
        rows >= GATE_UP_MIN_ROWS
            && self.mx_available
            && !mode.row_exact
            && !self.exact_numerics
            && input.is_multiple_of(128)
            && out.is_multiple_of(128)
            && (!p2e2 || q3_byte_size(out, input) <= P2E2_EXPAND_LIMIT)
    }

    fn submit_gate_up_swiglu(
        &self,
        kernels: &mut dyn KernelOps,
        weight: &WeightRef,
        x: DevicePtr,
        y: DevicePtr,
        rows: usize,
        scratch: ScratchRegion,
        mode: LinearMode,
    ) -> Result<bool, CudaError> {
        if !self.fuses_gate_up_swiglu(weight, rows, mode) || kernels.row_exact() || kernels.exact_numerics() {
            return Ok(false);
        }
        let (_, shape, p2e2) = describe(weight)?;
        let Shape { out, input } = shape;
        let rows = u32::try_from(rows)
            .map_err(|_| CudaError::Argument(format!("{rows} rows are more than a launch takes")))?;
        // The paired kernel never splits K: the activations alone are its scratch.
        let reserve = mx_scratch_bytes(rows, input, 0, self.sm_count, 0);
        let (weights, weights_bytes, activations) = if p2e2 {
            let (expanded, activations) = Self::expand(kernels, weight, shape, rows, scratch, reserve)?;
            (scratch.ptr, expanded, activations)
        } else {
            if (scratch.bytes as u64) < reserve {
                return Err(CudaError::Argument(format!(
                    "a gate/up launch of {rows} rows over {} needs {reserve} bytes of scratch and has {}",
                    weight.name, scratch.bytes
                )));
            }
            (weight.ptr, weight.bytes, scratch)
        };
        let fused = kernels.q3_mx_gate_up_swiglu_bf16(
            x,
            weights,
            y,
            activations.ptr,
            rows,
            input,
            out,
            weights_bytes,
            activations.bytes as u64,
        );
        match fused {
            // The library declines what the host cannot see (alignment); the caller composes the pair.
            Err(CudaError::Call {
                status: EUHEDRAL_CUDA_ROUTE_UNAVAILABLE,
                ..
            }) => Ok(false),
            other => other.map(|()| true),
        }
    }

    fn scratch_bytes(&self, weight: &WeightRef, rows: usize, _mode: LinearMode) -> usize {
        let Ok((_, Shape { out, input }, p2e2)) = describe(weight) else {
            return 0;
        };
        let Ok(rows) = u32::try_from(rows) else {
            return usize::MAX;
        };
        let bytes = linear_scratch_bytes(
            p2e2,
            rows,
            input,
            out,
            self.mx_available,
            Some(self.sm_count),
            self.exact_numerics,
        );
        usize::try_from(bytes).unwrap_or(usize::MAX)
    }
}

/// `out_features` and `in_features` of a grouped tensor.
#[derive(Clone, Copy, Debug)]
struct Shape {
    out: u32,
    input: u32,
}

/// The bits, shape and compression of a grouped linear, or the refusal of a weight that is none.
fn describe(weight: &WeightRef) -> Result<(Bits, Shape, bool), CudaError> {
    let refuse = || CudaError::Argument(format!("{} is not a two-dimensional Q3, Q4 or Q5 object", weight.name));
    let bits = match weight.format {
        WeightFormat::Q3G64Fp16 => Bits::Q3,
        WeightFormat::Q4G64Fp16 => Bits::Q4,
        WeightFormat::Q5G64Fp16 => Bits::Q5,
        _ => return Err(refuse()),
    };
    let p2e2 = match (bits, weight.layout) {
        (_, WeightLayout::RowSplitK128V1) => false,
        (Bits::Q3, WeightLayout::RowSplitP2e2V1) => true,
        _ => return Err(refuse()),
    };
    match weight.shape.as_slice() {
        [out, input] => Ok((
            bits,
            Shape {
                out: u32::try_from(*out).map_err(|_| refuse())?,
                input: u32::try_from(*input).map_err(|_| refuse())?,
            },
            p2e2,
        )),
        _ => Err(refuse()),
    }
}

#[cfg(test)]
mod tests {
    use euhedral_inference_edrl::TensorDataType;

    use super::*;
    use crate::model::Residence;

    fn weight(p2e2: bool, out: u64, input: u64) -> WeightRef {
        let layout = if p2e2 {
            WeightLayout::RowSplitP2e2V1
        } else {
            WeightLayout::RowSplitK128V1
        };
        WeightRef {
            name: "text/layers/0/mlp/gate_up".to_string(),
            shape: vec![out, input],
            dtype: TensorDataType::Uint8,
            format: WeightFormat::Q3G64Fp16,
            layout,
            ptr: DevicePtr::NULL,
            bytes: q3_byte_size(out as u32, input as u32),
            residence: Residence::Device,
            trellis: None,
        }
    }

    #[test]
    fn the_scratch_of_a_format_covers_every_launch_in_every_mode() {
        let shapes = [
            (5120u64, 5120u64),
            (34816, 5120),
            (5120, 17408),
            (5000, 5120),
            (4096, 3072),
            (1024, 1000),
        ];
        for exact in [false, true] {
            for mx in [false, true] {
                let mut linears = Q3Linear::with_mx_route(mx, 70);
                if exact {
                    linears = linears.with_exact_numerics();
                }
                for p2e2 in [false, true] {
                    for (out, input) in shapes {
                        let weight = weight(p2e2, out, input);
                        let shape = Shape {
                            out: out as u32,
                            input: input as u32,
                        };
                        for rows in (1..=40).chain([63, 64, 65, 127, 128, 129, 200, 256, 257, 512]) {
                            for row_exact in [false, true] {
                                let demanded = linears.demand(p2e2, shape, rows, row_exact);
                                let mode = LinearMode {
                                    row_exact,
                                    prefill: true,
                                };
                                let held = linears.scratch_bytes(&weight, rows as usize, mode) as u64;
                                assert!(
                                    held >= demanded,
                                    "exact {exact} mx {mx} p2e2 {p2e2} {out}x{input} rows {rows} row-exact {row_exact}: \
                                     {held} bytes held, {demanded} demanded"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn a_compressed_tensor_is_expanded_wherever_its_fused_decode_does_not_run() {
        let weight = weight(true, 5120, 5120);
        let expanded = q3_byte_size(5120, 5120).next_multiple_of(256) as usize;
        let mode = LinearMode::default();
        let plain = Q3Linear::with_mx_route(true, 70);
        assert_eq!(plain.scratch_bytes(&weight, 1, mode), 0);
        assert_eq!(plain.scratch_bytes(&weight, 8, mode), 0);
        assert!(plain.scratch_bytes(&weight, 9, mode) >= expanded);
        // The exact kernels expand it whatever the rows.
        let exact = Q3Linear::with_mx_route(true, 70).with_exact_numerics();
        assert_eq!(exact.scratch_bytes(&weight, 1, mode), expanded);
        assert_eq!(exact.scratch_bytes(&weight, 8, mode), expanded);
        // So does a shape the decode kernels do not take.
        let odd = self::weight(true, 5000, 5120);
        assert!(plain.scratch_bytes(&odd, 1, mode) > 0);
    }

    #[test]
    fn a_gate_up_projection_fuses_with_its_swiglu_from_sixty_four_rows() {
        let mode = LinearMode::default();
        for p2e2 in [false, true] {
            let gate_up = weight(p2e2, 34816, 5120);
            let linears = Q3Linear::with_mx_route(true, 170);
            assert!(!linears.fuses_gate_up_swiglu(&gate_up, 63, mode), "p2e2 {p2e2}");
            assert!(linears.fuses_gate_up_swiglu(&gate_up, 64, mode), "p2e2 {p2e2}");
            assert!(linears.fuses_gate_up_swiglu(&gate_up, 512, mode), "p2e2 {p2e2}");
            // Not on a device without the FP8 route, not for a verification, not under exact numerics.
            assert!(!Q3Linear::with_mx_route(false, 170).fuses_gate_up_swiglu(&gate_up, 64, mode));
            let verification = LinearMode {
                row_exact: true,
                prefill: false,
            };
            assert!(!linears.fuses_gate_up_swiglu(&gate_up, 64, verification));
            assert!(
                !Q3Linear::with_mx_route(true, 170)
                    .with_exact_numerics()
                    .fuses_gate_up_swiglu(&gate_up, 64, mode)
            );
            // Not for a shape the paired kernel has no tiles for.
            assert!(!linears.fuses_gate_up_swiglu(&weight(p2e2, 34800, 5120), 64, mode));
            // The scratch of the plain linear holds the paired launch's.
            let paired = mx_scratch_bytes(64, 5120, 0, 170, 0) as usize;
            assert!(linears.scratch_bytes(&gate_up, 64, mode) >= paired);
        }
        // Q4 and Q5 tensors are never paired.
        let mut q4 = weight(false, 34816, 5120);
        q4.format = WeightFormat::Q4G64Fp16;
        assert!(!Q3Linear::with_mx_route(true, 170).fuses_gate_up_swiglu(&q4, 64, mode));
    }
}
