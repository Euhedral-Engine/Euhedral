//! The elementwise, norm and sampling launches: the host dispatch of `native/src/host/qwen_layer_ops.c` and
//! `native/src/host/rms_norm_bf16.c`, as plain functions from a shape to the launches the C wrapper would issue.
//!
//! Each function pushes its [`Launch`]es into the caller's sink and allocates nothing. Where the C wrapper returns an
//! error status the function returns the matching [`PlanError`]; launches already pushed stay pushed. The thread-local
//! modes of the C library are explicit values in [`Modes`].

use crate::DevicePtr;
use crate::kernel;
use crate::launch::catalog::{Arguments, KernelId, Launch};
use crate::launch::{Modes, PlanError};

const RESIDUAL_ADD: KernelId = kernel!("euhedral_residual_add_bf16");
const SWIGLU: KernelId = kernel!("euhedral_swiglu_bf16");
const RESIDUAL_RMS_NORM: KernelId = kernel!("euhedral_residual_rms_norm_bf16");
const RESIDUAL_RMS_NORM_ROW: KernelId = kernel!("euhedral_residual_rms_norm_row_bf16");
const RMS_NORM: KernelId = kernel!("euhedral_rms_norm_bf16");
const ARGMAX: KernelId = kernel!("euhedral_argmax_bf16");
const ARGMAX_LOGPROB: KernelId = kernel!("euhedral_argmax_logprob_bf16");
const SELECTED_TOKEN: KernelId = kernel!("euhedral_selected_token");

/// The shape of `euhedral_cuda_residual_add_bf16` (`qwen_layer_ops.c:432`).
#[derive(Clone, Copy, Debug)]
pub struct ResidualAddShape {
    pub residual: DevicePtr,
    pub delta: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub width: u32,
}

/// `euhedral_cuda_residual_add_bf16`, `qwen_layer_ops.c:432-451`: one launch.
pub fn plan_residual_add(shape: &ResidualAddShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.residual.is_null() || s.delta.is_null() || s.output.is_null() || s.rows == 0 || s.width == 0 {
        return Err(PlanError::InvalidArgument);
    }
    let count = u64::from(s.rows) * u64::from(s.width);
    if count > u64::from(u32::MAX) {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(s.residual)
        .pointer(s.delta)
        .pointer(s.output)
        .u32(count as u32);
    // qwen_layer_ops.c:449.
    sink(Launch::new(
        RESIDUAL_ADD,
        [count.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        args,
    ));
    Ok(())
}

/// The shape of `euhedral_cuda_swiglu_bf16` (`qwen_layer_ops.c:487`).
#[derive(Clone, Copy, Debug)]
pub struct SwigluShape {
    pub gate_up: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub intermediate_size: u32,
}

/// `euhedral_cuda_swiglu_bf16`, `qwen_layer_ops.c:487-501`: one launch.
pub fn plan_swiglu(shape: &SwigluShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.gate_up.is_null() || s.output.is_null() || s.rows == 0 || s.intermediate_size == 0 {
        return Err(PlanError::InvalidArgument);
    }
    let count = u64::from(s.rows) * u64::from(s.intermediate_size);
    if count > u64::from(u32::MAX) {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(s.gate_up)
        .pointer(s.output)
        .u32(s.rows)
        .u32(s.intermediate_size);
    // qwen_layer_ops.c:499.
    sink(Launch::new(
        SWIGLU,
        [count.div_ceil(256) as u32, 1, 1],
        [256, 1, 1],
        0,
        args,
    ));
    Ok(())
}

/// The shape of `euhedral_cuda_residual_rms_norm_bf16` (`qwen_layer_ops.c:371`).
#[derive(Clone, Copy, Debug)]
pub struct ResidualRmsNormShape {
    pub residual: DevicePtr,
    pub delta: DevicePtr,
    pub weight: DevicePtr,
    pub hidden: DevicePtr,
    pub normalized: DevicePtr,
    pub rows: u32,
    pub width: u32,
    pub epsilon: f32,
}

/// `euhedral_cuda_residual_rms_norm_bf16`, `qwen_layer_ops.c:371-410`.
///
/// One row, or row-exact rows, on 16-byte aligned pointers with a width of a multiple of 8 up to 8192 and relaxed
/// numerics run the row kernel (one CTA per row); other row-exact rows run one launch per row; the rest run the
/// 128-thread kernel over all rows.
pub fn plan_residual_rms_norm(
    shape: &ResidualRmsNormShape,
    modes: Modes,
    sink: &mut impl FnMut(Launch),
) -> Result<(), PlanError> {
    let s = shape;
    // qwen_layer_ops.c:374.
    let row_exact = s.rows > 1 && modes.row_exact;
    // qwen_layer_ops.c:375-377.
    let row_kernel = !s.residual.is_null()
        && !s.delta.is_null()
        && !s.weight.is_null()
        && !s.hidden.is_null()
        && !s.normalized.is_null()
        && s.width.is_multiple_of(8)
        && s.width <= 8192
        && !modes.exact_numerics
        && (s.residual.get() | s.delta.get() | s.weight.get() | s.hidden.get() | s.normalized.get()) & 15 == 0;
    // qwen_layer_ops.c:378-388: the recursion runs before the checks below, and each row call runs them.
    if row_exact
        && !row_kernel
        && !s.residual.is_null()
        && !s.delta.is_null()
        && !s.hidden.is_null()
        && !s.normalized.is_null()
    {
        return plan_rows(s, modes, sink);
    }
    // qwen_layer_ops.c:389-392.
    if s.residual.is_null()
        || s.delta.is_null()
        || s.weight.is_null()
        || s.hidden.is_null()
        || s.normalized.is_null()
        || s.rows == 0
        || s.width == 0
        || !s.epsilon.is_finite()
        || s.epsilon < 0.0
    {
        return Err(PlanError::InvalidArgument);
    }
    // qwen_layer_ops.c:393.
    if s.rows > i32::MAX as u32 || u64::from(s.rows) * u64::from(s.width) > u64::from(u32::MAX) {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(s.residual)
        .pointer(s.delta)
        .pointer(s.weight)
        .pointer(s.hidden)
        .pointer(s.normalized)
        .u32(s.rows)
        .u32(s.width)
        .f32(s.epsilon);
    // qwen_layer_ops.c:399-400.
    if (s.rows == 1 || row_exact) && row_kernel {
        let block = (s.width / 8).div_ceil(32) * 32;
        sink(Launch::new(
            RESIDUAL_RMS_NORM_ROW,
            [s.rows, 1, 1],
            [block, 1, 1],
            0,
            args,
        ));
        return Ok(());
    }
    // qwen_layer_ops.c:401-409.
    if row_exact {
        return plan_rows(s, modes, sink);
    }
    // qwen_layer_ops.c:410.
    sink(Launch::new(RESIDUAL_RMS_NORM, [s.rows, 1, 1], [128, 1, 1], 0, args));
    Ok(())
}

/// The per-row recursion (`qwen_layer_ops.c:380-388`, `:400-408`): each row is a call of its own on the row's slice.
fn plan_rows(s: &ResidualRmsNormShape, modes: Modes, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let stride = u64::from(s.width) * 2;
    for row in 0..s.rows {
        let step = u64::from(row) * stride;
        let one = ResidualRmsNormShape {
            residual: s.residual.offset(step),
            delta: s.delta.offset(step),
            hidden: s.hidden.offset(step),
            normalized: s.normalized.offset(step),
            rows: 1,
            ..*s
        };
        plan_residual_rms_norm(&one, modes, sink)?;
    }
    Ok(())
}

/// The shape of `euhedral_cuda_rms_norm_bf16` and `…_unit_offset_bf16` (`rms_norm_bf16.c:57,62`).
#[derive(Clone, Copy, Debug)]
pub struct RmsNormShape {
    pub input: DevicePtr,
    pub weight: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub width: u32,
    pub epsilon: f32,
}

/// `euhedral_cuda_rms_norm_bf16`, `rms_norm_bf16.c:57`: the weight is used as it is (offset 0).
pub fn plan_rms_norm(shape: &RmsNormShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    plan_rms_norm_with_offset(shape, 0.0, sink)
}

/// `euhedral_cuda_rms_norm_unit_offset_bf16`, `rms_norm_bf16.c:62`: the weight plus one (offset 1).
pub fn plan_rms_norm_unit_offset(shape: &RmsNormShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    plan_rms_norm_with_offset(shape, 1.0, sink)
}

/// `rms_norm_bf16_with_offset`, `rms_norm_bf16.c:32-55`: one launch.
fn plan_rms_norm_with_offset(
    shape: &RmsNormShape,
    weight_offset: f32,
    sink: &mut impl FnMut(Launch),
) -> Result<(), PlanError> {
    let s = shape;
    if s.input.is_null()
        || s.weight.is_null()
        || s.output.is_null()
        || s.rows == 0
        || s.width == 0
        || !s.epsilon.is_finite()
        || s.epsilon < 0.0
    {
        return Err(PlanError::InvalidArgument);
    }
    let args = Arguments::new()
        .pointer(s.input)
        .pointer(s.weight)
        .pointer(s.output)
        .u32(s.rows)
        .u32(s.width)
        .f32(s.epsilon)
        .f32(weight_offset);
    // rms_norm_bf16.c:46-47: a single row (decode) spreads its output pass over 40 slices.
    let slices = if s.rows == 1 { 40 } else { 1 };
    sink(Launch::new(RMS_NORM, [s.rows, slices, 1], [128, 1, 1], 0, args));
    Ok(())
}

/// The shape of `euhedral_cuda_argmax_bf16` and `…_argmax_logprob_bf16` (`qwen_layer_ops.c:450,462`).
#[derive(Clone, Copy, Debug)]
pub struct ArgmaxShape {
    pub logits: DevicePtr,
    pub count: u32,
    pub result: DevicePtr,
}

/// `euhedral_cuda_argmax_bf16`, `qwen_layer_ops.c:450-460`: one launch.
pub fn plan_argmax(shape: &ArgmaxShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    plan_argmax_with(ARGMAX, shape, sink)
}

/// `euhedral_cuda_argmax_logprob_bf16`, `qwen_layer_ops.c:462-472`: one launch.
pub fn plan_argmax_logprob(shape: &ArgmaxShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    plan_argmax_with(ARGMAX_LOGPROB, shape, sink)
}

fn plan_argmax_with(kernel: KernelId, shape: &ArgmaxShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    if shape.logits.is_null() || shape.result.is_null() || shape.count == 0 {
        return Err(PlanError::InvalidArgument);
    }
    let args = Arguments::new()
        .pointer(shape.logits)
        .u32(shape.count)
        .pointer(shape.result);
    sink(Launch::new(kernel, [1, 1, 1], [1024, 1, 1], 0, args));
    Ok(())
}

/// The shape of `euhedral_cuda_selected_token` (`qwen_layer_ops.c:474`).
#[derive(Clone, Copy, Debug)]
pub struct SelectedTokenShape {
    pub key: DevicePtr,
    /// May be null (the C wrapper does not check it).
    pub ids: DevicePtr,
    pub token: DevicePtr,
}

/// `euhedral_cuda_selected_token`, `qwen_layer_ops.c:474-485`: one launch of one thread.
pub fn plan_selected_token(shape: &SelectedTokenShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    if shape.key.is_null() || shape.token.is_null() {
        return Err(PlanError::InvalidArgument);
    }
    let args = Arguments::new()
        .pointer(shape.key)
        .pointer(shape.ids)
        .pointer(shape.token);
    sink(Launch::new(SELECTED_TOKEN, [1, 1, 1], [1, 1, 1], 0, args));
    Ok(())
}

/// The check of `euhedral_cuda_zero_device_memory` (`qwen_layer_ops.c:504-507`). The zeroing is a memset of
/// `byte_size` bytes at `address`, not a kernel launch, so it plans no [`Launch`].
pub fn check_zero_device_memory(address: DevicePtr, byte_size: u64) -> Result<(), PlanError> {
    if address.is_null() || byte_size == 0 {
        return Err(PlanError::InvalidArgument);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptr(address: u64) -> DevicePtr {
        // SAFETY: the planner never dereferences a pointer.
        unsafe { DevicePtr::from_raw(address) }
    }

    fn collect(
        plan: impl FnOnce(&mut dyn FnMut(Launch)) -> Result<(), PlanError>,
    ) -> (Vec<Launch>, Result<(), PlanError>) {
        let mut launches = Vec::new();
        let result = plan(&mut |launch| launches.push(launch));
        (launches, result)
    }

    const RELAXED: Modes = Modes {
        row_exact: false,
        exact_numerics: false,
    };
    const ROW_EXACT: Modes = Modes {
        row_exact: true,
        exact_numerics: false,
    };
    const EXACT: Modes = Modes {
        row_exact: false,
        exact_numerics: true,
    };

    fn add(rows: u32, width: u32) -> ResidualAddShape {
        ResidualAddShape {
            residual: ptr(0x1000),
            delta: ptr(0x2000),
            output: ptr(0x3000),
            rows,
            width,
        }
    }

    fn plan_add(shape: &ResidualAddShape) -> (Vec<Launch>, Result<(), PlanError>) {
        collect(|sink| plan_residual_add(shape, &mut |l| sink(l)))
    }

    #[test]
    fn residual_add_grid_rounds_up_by_256() {
        // qwen_layer_ops.c:449: (count + 255) / 256 blocks of 256.
        for (rows, width, grid) in [(1, 256, 1), (1, 257, 2), (3, 1000, 12), (1, 1, 1), (2, 128, 1)] {
            let (launches, result) = plan_add(&add(rows, width));
            assert_eq!(result, Ok(()));
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].kernel, kernel!("euhedral_residual_add_bf16"));
            assert_eq!(launches[0].grid, [grid, 1, 1], "rows {rows} width {width}");
            assert_eq!(launches[0].block, [256, 1, 1]);
            assert_eq!(launches[0].shared_bytes, 0);
            assert!(launches[0].programmatic);
        }
    }

    #[test]
    fn residual_add_arguments_and_limits() {
        let (launches, _) = plan_add(&add(3, 1000));
        // qwen_layer_ops.c:447-448: three pointers and the element count as u32.
        assert_eq!(launches[0].args.words(), &[0x1000, 0x2000, 0x3000, 3000]);
        assert_eq!(launches[0].args.sizes(), &[8, 8, 8, 4]);
        // 65536 x 65535 = 2^32 - 65536 elements fit; the block count is (2^32 - 65536 + 255) / 256 = 16776960.
        let (launches, result) = plan_add(&add(65536, 65535));
        assert_eq!(result, Ok(()));
        assert_eq!(launches[0].grid, [16_776_960, 1, 1]);
        assert_eq!(launches[0].args.words()[3], 4_294_901_760);
        // 65536 x 65536 = 2^32 does not.
        let (launches, result) = plan_add(&add(65536, 65536));
        assert_eq!((launches.len(), result), (0, Err(PlanError::SizeOverflow)));
        for bad in [
            ResidualAddShape {
                residual: DevicePtr::NULL,
                ..add(1, 1)
            },
            ResidualAddShape {
                delta: DevicePtr::NULL,
                ..add(1, 1)
            },
            ResidualAddShape {
                output: DevicePtr::NULL,
                ..add(1, 1)
            },
            add(0, 1),
            add(1, 0),
        ] {
            assert_eq!(plan_add(&bad), (vec![], Err(PlanError::InvalidArgument)));
        }
    }

    #[test]
    fn swiglu_grid_arguments_and_errors() {
        let shape = SwigluShape {
            gate_up: ptr(0x1000),
            output: ptr(0x2000),
            rows: 5,
            intermediate_size: 3072,
        };
        let (launches, result) = collect(|sink| plan_swiglu(&shape, &mut |l| sink(l)));
        assert_eq!(result, Ok(()));
        // qwen_layer_ops.c:499: 15360 elements are 60 blocks of 256; arguments are rows and width, not the count.
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].kernel, kernel!("euhedral_swiglu_bf16"));
        assert_eq!(launches[0].grid, [60, 1, 1]);
        assert_eq!(launches[0].block, [256, 1, 1]);
        assert_eq!(launches[0].args.words(), &[0x1000, 0x2000, 5, 3072]);
        assert_eq!(launches[0].args.sizes(), &[8, 8, 4, 4]);
        let odd = SwigluShape {
            rows: 1,
            intermediate_size: 257,
            ..shape
        };
        let (launches, _) = collect(|sink| plan_swiglu(&odd, &mut |l| sink(l)));
        assert_eq!(launches[0].grid, [2, 1, 1]);
        let big = SwigluShape {
            rows: 65536,
            intermediate_size: 65536,
            ..shape
        };
        assert_eq!(
            collect(|sink| plan_swiglu(&big, &mut |l| sink(l))),
            (vec![], Err(PlanError::SizeOverflow))
        );
        for bad in [
            SwigluShape {
                gate_up: DevicePtr::NULL,
                ..shape
            },
            SwigluShape {
                output: DevicePtr::NULL,
                ..shape
            },
            SwigluShape { rows: 0, ..shape },
            SwigluShape {
                intermediate_size: 0,
                ..shape
            },
        ] {
            assert_eq!(
                collect(|sink| plan_swiglu(&bad, &mut |l| sink(l))),
                (vec![], Err(PlanError::InvalidArgument))
            );
        }
    }

    fn norm(rows: u32, width: u32) -> ResidualRmsNormShape {
        ResidualRmsNormShape {
            residual: ptr(0x1000),
            delta: ptr(0x2000),
            weight: ptr(0x3000),
            hidden: ptr(0x4000),
            normalized: ptr(0x5000),
            rows,
            width,
            epsilon: 1e-6,
        }
    }

    fn plan_norm(shape: &ResidualRmsNormShape, modes: Modes) -> (Vec<Launch>, Result<(), PlanError>) {
        collect(|sink| plan_residual_rms_norm(shape, modes, &mut |l| sink(l)))
    }

    const ROW: KernelId = kernel!("euhedral_residual_rms_norm_row_bf16");
    const WIDE: KernelId = kernel!("euhedral_residual_rms_norm_bf16");

    #[test]
    fn residual_rms_norm_one_row_takes_the_row_kernel_with_a_block_of_whole_warps() {
        // qwen_layer_ops.c:399-400: block (width / 8 + 31) / 32 * 32.
        for (width, block) in [(8, 32), (256, 32), (264, 64), (5120, 640), (8192, 1024), (2048, 256)] {
            let (launches, result) = plan_norm(&norm(1, width), RELAXED);
            assert_eq!(result, Ok(()));
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].kernel, ROW);
            assert_eq!(launches[0].grid, [1, 1, 1]);
            assert_eq!(launches[0].block, [block, 1, 1], "width {width}");
            assert!(launches[0].programmatic);
        }
        let (launches, _) = plan_norm(&norm(1, 5120), RELAXED);
        // qwen_layer_ops.c:394: five pointers, rows, width and epsilon's bits.
        assert_eq!(
            launches[0].args.words(),
            &[
                0x1000,
                0x2000,
                0x3000,
                0x4000,
                0x5000,
                1,
                5120,
                u64::from(1e-6f32.to_bits())
            ]
        );
        assert_eq!(launches[0].args.sizes(), &[8, 8, 8, 8, 8, 4, 4, 4]);
        assert_eq!(launches[0].args.f32_at(7), Some(1e-6));
    }

    #[test]
    fn residual_rms_norm_row_kernel_conditions_each_flipped_alone() {
        // qwen_layer_ops.c:375-377: width % 8, width <= 8192, every pointer 16-byte aligned, not exact.
        let wide_width = |width| plan_norm(&norm(1, width), RELAXED).0[0].kernel;
        assert_eq!(wide_width(8192), ROW);
        assert_eq!(wide_width(8200), WIDE);
        assert_eq!(wide_width(8184), ROW);
        assert_eq!(wide_width(12), WIDE);
        assert_eq!(wide_width(1), WIDE);
        assert_eq!(plan_norm(&norm(1, 4096), EXACT).0[0].kernel, WIDE);
        for which in 0..5 {
            let mut shape = norm(1, 4096);
            let slot = [
                &mut shape.residual,
                &mut shape.delta,
                &mut shape.weight,
                &mut shape.hidden,
                &mut shape.normalized,
            ]
            .into_iter()
            .nth(which)
            .unwrap();
            *slot = ptr(slot.get() + 8);
            assert_eq!(plan_norm(&shape, RELAXED).0[0].kernel, WIDE, "pointer {which}");
        }
        // The wide kernel keeps one CTA row per launch and a 128-thread block.
        let (launches, _) = plan_norm(&norm(1, 12), RELAXED);
        assert_eq!((launches[0].grid, launches[0].block), ([1, 1, 1], [128, 1, 1]));
    }

    #[test]
    fn residual_rms_norm_many_rows_relaxed_is_one_wide_launch() {
        // qwen_layer_ops.c:410: rows blocks of 128, whatever the width.
        let (launches, result) = plan_norm(&norm(7, 4096), RELAXED);
        assert_eq!(result, Ok(()));
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].kernel, WIDE);
        assert_eq!((launches[0].grid, launches[0].block), ([7, 1, 1], [128, 1, 1]));
        // Exact numerics with many rows: the same wide launch.
        let (launches, _) = plan_norm(&norm(7, 4096), EXACT);
        assert_eq!(launches[0].kernel, WIDE);
        // Exact numerics and row-exact together: the row kernel is off, so one launch per row.
        let both = Modes {
            row_exact: true,
            exact_numerics: true,
        };
        assert_eq!(plan_norm(&norm(3, 4096), both).0.len(), 3);
    }

    #[test]
    fn residual_rms_norm_row_exact_with_the_row_kernel_is_one_launch_of_rows_blocks() {
        // qwen_layer_ops.c:399: (rows == 1 || row_exact) && row_kernel.
        let (launches, result) = plan_norm(&norm(9, 5120), ROW_EXACT);
        assert_eq!(result, Ok(()));
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].kernel, ROW);
        assert_eq!((launches[0].grid, launches[0].block), ([9, 1, 1], [640, 1, 1]));
        assert_eq!(launches[0].args.words()[5], 9);
    }

    #[test]
    fn residual_rms_norm_row_exact_without_the_row_kernel_recurses_per_row() {
        // qwen_layer_ops.c:378-388, :400-408: one one-row call per row, each slice width * 2 bytes on.
        for rows in [1u32, 2, 3, 1000] {
            // width 8200 is too wide for the row kernel; one row is not a recursion (row-exact needs rows > 1).
            let (launches, result) = plan_norm(&norm(rows, 8200), ROW_EXACT);
            assert_eq!(result, Ok(()));
            assert_eq!(launches.len(), rows as usize);
            for (row, launch) in launches.iter().enumerate() {
                let step = row as u64 * 16400;
                assert_eq!(launch.kernel, WIDE);
                assert_eq!((launch.grid, launch.block), ([1, 1, 1], [128, 1, 1]));
                // weight is shared; the rest move; rows is 1, width and epsilon are kept.
                assert_eq!(
                    launch.args.words()[..7],
                    [
                        0x1000 + step,
                        0x2000 + step,
                        0x3000,
                        0x4000 + step,
                        0x5000 + step,
                        1,
                        8200
                    ]
                );
                assert_eq!(launch.args.f32_at(7), Some(1e-6));
            }
        }
    }

    #[test]
    fn residual_rms_norm_recursion_checks_each_call() {
        // A null weight still recurses (qwen_layer_ops.c:378 does not test it); the first row's call refuses it.
        let no_weight = ResidualRmsNormShape {
            weight: DevicePtr::NULL,
            ..norm(3, 8200)
        };
        assert_eq!(
            plan_norm(&no_weight, ROW_EXACT),
            (vec![], Err(PlanError::InvalidArgument))
        );
        let nan = ResidualRmsNormShape {
            epsilon: f32::NAN,
            ..norm(3, 8200)
        };
        assert_eq!(plan_norm(&nan, ROW_EXACT), (vec![], Err(PlanError::InvalidArgument)));
        // A null hidden pointer skips the recursion and fails the plain check.
        let no_hidden = ResidualRmsNormShape {
            hidden: DevicePtr::NULL,
            ..norm(3, 8200)
        };
        assert_eq!(
            plan_norm(&no_hidden, ROW_EXACT),
            (vec![], Err(PlanError::InvalidArgument))
        );
    }

    #[test]
    fn residual_rms_norm_validation_and_size_limits() {
        for bad in [
            ResidualRmsNormShape {
                residual: DevicePtr::NULL,
                ..norm(2, 64)
            },
            ResidualRmsNormShape {
                delta: DevicePtr::NULL,
                ..norm(2, 64)
            },
            ResidualRmsNormShape {
                weight: DevicePtr::NULL,
                ..norm(2, 64)
            },
            ResidualRmsNormShape {
                hidden: DevicePtr::NULL,
                ..norm(2, 64)
            },
            ResidualRmsNormShape {
                normalized: DevicePtr::NULL,
                ..norm(2, 64)
            },
            norm(0, 64),
            norm(2, 0),
            ResidualRmsNormShape {
                epsilon: f32::INFINITY,
                ..norm(2, 64)
            },
            ResidualRmsNormShape {
                epsilon: -1e-9,
                ..norm(2, 64)
            },
        ] {
            assert_eq!(plan_norm(&bad, RELAXED), (vec![], Err(PlanError::InvalidArgument)));
        }
        // Epsilon 0 is allowed (`epsilon < 0.0` is the test).
        assert_eq!(
            plan_norm(
                &ResidualRmsNormShape {
                    epsilon: 0.0,
                    ..norm(2, 64)
                },
                RELAXED
            )
            .1,
            Ok(())
        );
        // qwen_layer_ops.c:393: more rows than i32::MAX, or rows * width past u32::MAX.
        assert_eq!(
            plan_norm(&norm(1 << 31, 1), RELAXED),
            (vec![], Err(PlanError::SizeOverflow))
        );
        assert_eq!(
            plan_norm(&norm(65536, 65536), RELAXED),
            (vec![], Err(PlanError::SizeOverflow))
        );
        let (launches, result) = plan_norm(&norm(i32::MAX as u32, 1), RELAXED);
        assert_eq!(result, Ok(()));
        assert_eq!(launches[0].grid, [i32::MAX as u32, 1, 1]);
    }

    fn rms(rows: u32) -> RmsNormShape {
        RmsNormShape {
            input: ptr(0x1000),
            weight: ptr(0x2000),
            output: ptr(0x3000),
            rows,
            width: 128,
            epsilon: 1e-6,
        }
    }

    #[test]
    fn rms_norm_slices_one_row_over_forty() {
        // rms_norm_bf16.c:46-47.
        for (rows, grid) in [(1, [1, 40, 1]), (2, [2, 1, 1]), (3, [3, 1, 1]), (100, [100, 1, 1])] {
            let (launches, result) = collect(|sink| plan_rms_norm(&rms(rows), &mut |l| sink(l)));
            assert_eq!(result, Ok(()));
            assert_eq!(launches.len(), 1);
            assert_eq!(launches[0].kernel, kernel!("euhedral_rms_norm_bf16"));
            assert_eq!((launches[0].grid, launches[0].block), (grid, [128, 1, 1]));
            assert_eq!(launches[0].shared_bytes, 0);
        }
    }

    #[test]
    fn rms_norm_offset_is_a_float_word_of_zero_or_one() {
        let (plain, _) = collect(|sink| plan_rms_norm(&rms(2), &mut |l| sink(l)));
        let (unit, _) = collect(|sink| plan_rms_norm_unit_offset(&rms(2), &mut |l| sink(l)));
        // rms_norm_bf16.c:44: input, weight, output, rows, width, epsilon, weight offset.
        assert_eq!(
            plain[0].args.words()[..6],
            [0x1000, 0x2000, 0x3000, 2, 128, u64::from(1e-6f32.to_bits())]
        );
        assert_eq!(plain[0].args.sizes(), &[8, 8, 8, 4, 4, 4, 4]);
        assert_eq!(plain[0].args.words()[6], 0);
        assert_eq!(unit[0].args.words()[6], 0x3f80_0000);
        assert_eq!(unit[0].args.f32_at(6), Some(1.0));
        assert_eq!(unit[0].args.words()[..6], plain[0].args.words()[..6]);
    }

    #[test]
    fn rms_norm_validation() {
        for bad in [
            RmsNormShape {
                input: DevicePtr::NULL,
                ..rms(1)
            },
            RmsNormShape {
                weight: DevicePtr::NULL,
                ..rms(1)
            },
            RmsNormShape {
                output: DevicePtr::NULL,
                ..rms(1)
            },
            rms(0),
            RmsNormShape { width: 0, ..rms(1) },
            RmsNormShape {
                epsilon: f32::NAN,
                ..rms(1)
            },
            RmsNormShape {
                epsilon: -1.0,
                ..rms(1)
            },
        ] {
            assert_eq!(
                collect(|sink| plan_rms_norm_unit_offset(&bad, &mut |l| sink(l))),
                (vec![], Err(PlanError::InvalidArgument))
            );
        }
    }

    fn check_argmax(name: &str, plan: impl Fn(&ArgmaxShape, &mut Vec<Launch>) -> Result<(), PlanError>) {
        let shape = ArgmaxShape {
            logits: ptr(0x1000),
            count: 248_320,
            result: ptr(0x2000),
        };
        let mut launches = Vec::new();
        assert_eq!(plan(&shape, &mut launches), Ok(()));
        assert_eq!(launches.len(), 1);
        assert_eq!(launches[0].kernel.name(), name);
        // qwen_layer_ops.c:458: one block of 1024; logits, count, result.
        assert_eq!((launches[0].grid, launches[0].block), ([1, 1, 1], [1024, 1, 1]));
        assert_eq!(launches[0].args.words(), &[0x1000, 248_320, 0x2000]);
        assert_eq!(launches[0].args.sizes(), &[8, 4, 8]);
        for bad in [
            ArgmaxShape {
                logits: DevicePtr::NULL,
                ..shape
            },
            ArgmaxShape {
                result: DevicePtr::NULL,
                ..shape
            },
            ArgmaxShape { count: 0, ..shape },
        ] {
            let mut none = Vec::new();
            assert_eq!(plan(&bad, &mut none), Err(PlanError::InvalidArgument));
            assert!(none.is_empty());
        }
    }

    #[test]
    fn argmax_and_selected_token() {
        check_argmax("euhedral_argmax_bf16", |s, v| plan_argmax(s, &mut |l| v.push(l)));
        check_argmax("euhedral_argmax_logprob_bf16", |s, v| {
            plan_argmax_logprob(s, &mut |l| v.push(l))
        });
        let token = SelectedTokenShape {
            key: ptr(0x1000),
            ids: DevicePtr::NULL,
            token: ptr(0x3000),
        };
        let (launches, result) = collect(|sink| plan_selected_token(&token, &mut |l| sink(l)));
        assert_eq!(result, Ok(()));
        // qwen_layer_ops.c:483: one thread; the ids pointer may be null.
        assert_eq!((launches[0].grid, launches[0].block), ([1, 1, 1], [1, 1, 1]));
        assert_eq!(launches[0].args.words(), &[0x1000, 0, 0x3000]);
        assert_eq!(launches[0].args.sizes(), &[8, 8, 8]);
        for bad in [
            SelectedTokenShape {
                key: DevicePtr::NULL,
                ..token
            },
            SelectedTokenShape {
                token: DevicePtr::NULL,
                ..token
            },
        ] {
            assert_eq!(
                collect(|sink| plan_selected_token(&bad, &mut |l| sink(l))),
                (vec![], Err(PlanError::InvalidArgument))
            );
        }
    }

    #[test]
    fn zero_device_memory_checks() {
        // qwen_layer_ops.c:504-505.
        assert_eq!(check_zero_device_memory(ptr(0x1000), 1), Ok(()));
        assert_eq!(
            check_zero_device_memory(DevicePtr::NULL, 1),
            Err(PlanError::InvalidArgument)
        );
        assert_eq!(
            check_zero_device_memory(ptr(0x1000), 0),
            Err(PlanError::InvalidArgument)
        );
    }
}
