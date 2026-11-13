//! The Gated DeltaNet launches: the host dispatch of `native/src/host/qwen_layer_ops.c`'s GDN wrappers, as plain
//! functions from a shape to the launches the wrapper would issue.
//!
//! Each function pushes its [`Launch`]es into the caller's sink and allocates nothing. Where the C wrapper returns an
//! error status the function returns the matching [`PlanError`]. The thread-local modes of the C library are explicit
//! values in [`Modes`]. Where C computes in `u32` the arithmetic wraps here too.

use crate::DevicePtr;
use crate::kernel;
use crate::launch::catalog::{Arguments, KernelId, Launch};
use crate::launch::{Modes, PlanError};

const PROJECT_CONTROL: KernelId = kernel!("euhedral_gdn_project_control_fp32");
const PROJECT_CONTROL_TILED: KernelId = kernel!("euhedral_gdn_project_control_8x4_fp32");
const CONVOLUTION: KernelId = kernel!("euhedral_gdn_convolution_bf16");
const RECURRENCE: KernelId = kernel!("euhedral_gdn_recurrence_bf16");
const RECURRENCE_C8: KernelId = kernel!("euhedral_gdn_recurrence_c8_bf16");
const GATED_RMS_NORM: KernelId = kernel!("euhedral_gdn_gated_rms_norm_bf16");

/// The fewest rows at which the 8-row by 4-head kernel runs (`qwen_layer_ops.c:431`, machine-measured: 9.8 against
/// 19.6 us at 8 rows, 23.2 against 21.2 at 48).
const TILED_FROM_ROWS: u32 = 48;

/// The shape of `euhedral_cuda_gdn_project_control_fp32` (`qwen_layer_ops.c:412`).
#[derive(Clone, Copy, Debug)]
pub struct ProjectControlShape {
    pub input: DevicePtr,
    pub a_weight: DevicePtr,
    pub b_weight: DevicePtr,
    pub a_log: DevicePtr,
    pub dt_bias: DevicePtr,
    pub alpha: DevicePtr,
    pub beta: DevicePtr,
    pub rows: u32,
    pub width: u32,
    pub heads: u32,
}

/// `euhedral_cuda_gdn_project_control_fp32`, `qwen_layer_ops.c:412-434`: one launch, the tiled kernel from 48 rows when
/// the heads divide by 4.
pub fn plan_project_control(shape: &ProjectControlShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.input.is_null()
        || s.a_weight.is_null()
        || s.b_weight.is_null()
        || s.a_log.is_null()
        || s.dt_bias.is_null()
        || s.alpha.is_null()
        || s.beta.is_null()
        || s.rows == 0
        || s.width == 0
        || s.heads == 0
    {
        return Err(PlanError::InvalidArgument);
    }
    let count = u64::from(s.rows) * u64::from(s.heads);
    if count > i32::MAX as u64 {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(s.input)
        .pointer(s.a_weight)
        .pointer(s.b_weight)
        .pointer(s.a_log)
        .pointer(s.dt_bias)
        .pointer(s.alpha)
        .pointer(s.beta)
        .u32(s.rows)
        .u32(s.width)
        .u32(s.heads);
    // qwen_layer_ops.c:431-433.
    if s.rows >= TILED_FROM_ROWS && s.heads.is_multiple_of(4) {
        let blocks = s.rows.wrapping_add(7) / 8 * (s.heads / 4);
        sink(Launch::new(PROJECT_CONTROL_TILED, [blocks, 1, 1], [128, 1, 1], 0, args));
    } else {
        sink(Launch::new(PROJECT_CONTROL, [count as u32, 1, 1], [128, 1, 1], 0, args));
    }
    Ok(())
}

/// The shape of `euhedral_cuda_gdn_convolution_bf16` (`qwen_layer_ops.c:290`).
#[derive(Clone, Copy, Debug)]
pub struct ConvolutionShape {
    pub query_key: DevicePtr,
    pub value_z: DevicePtr,
    pub weights: DevicePtr,
    pub state: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub query_key_width: u32,
    pub value_width: u32,
    pub convolution_width: u32,
    pub kernel_size: u32,
}

/// `euhedral_cuda_gdn_convolution_bf16`, `qwen_layer_ops.c:290-312`: one launch of (channels / 128, rows / 32) blocks.
pub fn plan_convolution(shape: &ConvolutionShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    // qwen_layer_ops.c:293-296: the sum of the two widths is a u32 sum and wraps.
    if s.query_key.is_null()
        || s.value_z.is_null()
        || s.weights.is_null()
        || s.state.is_null()
        || s.output.is_null()
        || s.rows == 0
        || s.query_key_width == 0
        || s.value_width == 0
        || s.convolution_width != s.query_key_width.wrapping_add(s.value_width)
        || s.kernel_size < 2
        || s.kernel_size > 32
    {
        return Err(PlanError::InvalidArgument);
    }
    let args = Arguments::new()
        .pointer(s.query_key)
        .pointer(s.value_z)
        .pointer(s.weights)
        .pointer(s.state)
        .pointer(s.output)
        .u32(s.rows)
        .u32(s.query_key_width)
        .u32(s.value_width)
        .u32(s.convolution_width)
        .u32(s.kernel_size);
    // qwen_layer_ops.c:310: 32-row blocks (QWEN_GDN_CONV_ROWS), 128 channels per block; u32 arithmetic.
    let grid = [
        s.convolution_width.wrapping_add(127) / 128,
        s.rows.wrapping_add(31) / 32,
        1,
    ];
    sink(Launch::new(CONVOLUTION, grid, [128, 1, 1], 0, args));
    Ok(())
}

/// The shape of `euhedral_cuda_gdn_recurrence_bf16` (`qwen_layer_ops.c:314`).
#[derive(Clone, Copy, Debug)]
pub struct RecurrenceShape {
    pub convolved: DevicePtr,
    pub alpha: DevicePtr,
    pub beta: DevicePtr,
    pub recurrent_state: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub key_heads: u32,
    pub value_heads: u32,
    pub key_head_dim: u32,
    pub value_head_dim: u32,
    pub output_scale: f32,
}

/// `euhedral_cuda_gdn_recurrence_bf16`, `qwen_layer_ops.c:314-348`: one launch of `value_heads * 128 / 8` blocks of 32
/// threads. The column-owned kernel (c8) runs for two or more rows, not row-exact, not exact, when the convolved and
/// state pointers are 16-byte aligned (`:343-345`); the warp kernel runs otherwise.
pub fn plan_recurrence(shape: &RecurrenceShape, modes: Modes, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.convolved.is_null()
        || s.alpha.is_null()
        || s.beta.is_null()
        || s.recurrent_state.is_null()
        || s.output.is_null()
        || s.rows == 0
        || s.key_heads == 0
        || s.value_heads == 0
        || !s.value_heads.is_multiple_of(s.key_heads)
        || s.key_head_dim != 128
        || s.value_head_dim != 128
        || !s.output_scale.is_finite()
        || s.output_scale <= 0.0
    {
        return Err(PlanError::InvalidArgument);
    }
    // qwen_layer_ops.c:321.
    let rows_count = u64::from(s.value_heads) * u64::from(s.value_head_dim);
    if rows_count > u64::from(u32::MAX) {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(s.convolved)
        .pointer(s.alpha)
        .pointer(s.beta)
        .pointer(s.recurrent_state)
        .pointer(s.output)
        .u32(s.rows)
        .u32(s.key_heads)
        .u32(s.value_heads)
        .u32(s.key_head_dim)
        .u32(s.value_head_dim)
        .f32(s.output_scale);
    // qwen_layer_ops.c:343-347.
    let aligned = (s.convolved.get() | s.recurrent_state.get()) & 15 == 0;
    let kernel = if !modes.exact_numerics && aligned && s.rows > 1 && !modes.row_exact {
        RECURRENCE_C8
    } else {
        RECURRENCE
    };
    sink(Launch::new(
        kernel,
        [(rows_count / 8) as u32, 1, 1],
        [32, 1, 1],
        0,
        args,
    ));
    Ok(())
}

/// The shape of `euhedral_cuda_gdn_gated_rms_norm_bf16` (`qwen_layer_ops.c:350`).
#[derive(Clone, Copy, Debug)]
pub struct GatedRmsNormShape {
    pub recurrent: DevicePtr,
    pub value_z: DevicePtr,
    pub norm_weight: DevicePtr,
    pub output: DevicePtr,
    pub rows: u32,
    pub value_heads: u32,
    pub head_dim: u32,
    pub epsilon: f32,
}

/// `euhedral_cuda_gdn_gated_rms_norm_bf16`, `qwen_layer_ops.c:350-369`: one launch of `rows * value_heads` blocks.
pub fn plan_gated_rms_norm(shape: &GatedRmsNormShape, sink: &mut impl FnMut(Launch)) -> Result<(), PlanError> {
    let s = shape;
    if s.recurrent.is_null()
        || s.value_z.is_null()
        || s.norm_weight.is_null()
        || s.output.is_null()
        || s.rows == 0
        || s.value_heads == 0
        || s.head_dim != 128
        || !s.epsilon.is_finite()
        || s.epsilon < 0.0
    {
        return Err(PlanError::InvalidArgument);
    }
    let count = u64::from(s.rows) * u64::from(s.value_heads);
    if count > u64::from(u32::MAX) {
        return Err(PlanError::SizeOverflow);
    }
    let args = Arguments::new()
        .pointer(s.recurrent)
        .pointer(s.value_z)
        .pointer(s.norm_weight)
        .pointer(s.output)
        .u32(s.rows)
        .u32(s.value_heads)
        .u32(s.head_dim)
        .f32(s.epsilon);
    sink(Launch::new(GATED_RMS_NORM, [count as u32, 1, 1], [128, 1, 1], 0, args));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ptr(address: u64) -> DevicePtr {
        // SAFETY: the planner never dereferences a pointer.
        unsafe { DevicePtr::from_raw(address) }
    }

    fn one(result: Result<(), PlanError>, launches: Vec<Launch>) -> Launch {
        assert_eq!(result, Ok(()));
        assert_eq!(launches.len(), 1);
        launches[0]
    }

    fn control(rows: u32, heads: u32) -> ProjectControlShape {
        ProjectControlShape {
            input: ptr(0x100),
            a_weight: ptr(0x200),
            b_weight: ptr(0x300),
            a_log: ptr(0x400),
            dt_bias: ptr(0x500),
            alpha: ptr(0x600),
            beta: ptr(0x700),
            rows,
            width: 5120,
            heads,
        }
    }

    fn plan_control(shape: &ProjectControlShape) -> (Result<(), PlanError>, Vec<Launch>) {
        let mut launches = Vec::new();
        (plan_project_control(shape, &mut |l| launches.push(l)), launches)
    }

    #[test]
    fn project_control_tiles_from_48_rows_and_four_heads() {
        // qwen_layer_ops.c:431: rows >= 48 && heads % 4 == 0.
        let (result, launches) = plan_control(&control(47, 48));
        let below = one(result, launches);
        assert_eq!(below.kernel, kernel!("euhedral_gdn_project_control_fp32"));
        // :433: rows * heads blocks of 128.
        assert_eq!((below.grid, below.block), ([47 * 48, 1, 1], [128, 1, 1]));
        assert!(below.programmatic);
        let (result, launches) = plan_control(&control(48, 48));
        let at = one(result, launches);
        assert_eq!(at.kernel, kernel!("euhedral_gdn_project_control_8x4_fp32"));
        // :432: (48 + 7) / 8 = 6 row tiles times 48 / 4 = 12 head tiles.
        assert_eq!((at.grid, at.block), ([72, 1, 1], [128, 1, 1]));
        assert!(!at.programmatic);
        let (result, launches) = plan_control(&control(49, 48));
        let above = one(result, launches);
        // (49 + 7) / 8 = 7 tiles times 12.
        assert_eq!(
            (above.kernel, above.grid),
            (kernel!("euhedral_gdn_project_control_8x4_fp32"), [84, 1, 1])
        );
        // Heads not a multiple of 4 keep the per-row kernel at any rows.
        for heads in [47, 50, 1, 2, 3] {
            let (result, launches) = plan_control(&control(512, heads));
            let launch = one(result, launches);
            assert_eq!(launch.kernel, kernel!("euhedral_gdn_project_control_fp32"));
            assert_eq!(launch.grid, [512 * heads, 1, 1]);
        }
        let (result, launches) = plan_control(&control(512, 4));
        // 512 / 8 = 64 tiles times one head tile.
        assert_eq!(one(result, launches).grid, [64, 1, 1]);
    }

    #[test]
    fn project_control_arguments_and_errors() {
        let (_, launches) = plan_control(&control(48, 48));
        // qwen_layer_ops.c:430: seven pointers, rows, width, heads.
        assert_eq!(
            launches[0].args.words(),
            &[0x100, 0x200, 0x300, 0x400, 0x500, 0x600, 0x700, 48, 5120, 48]
        );
        assert_eq!(launches[0].args.sizes(), &[8, 8, 8, 8, 8, 8, 8, 4, 4, 4]);
        // :421: rows * heads is checked against i32::MAX, which 2^31 - 1 meets and 2^31 exceeds.
        let (result, launches) = plan_control(&control(1, i32::MAX as u32));
        assert_eq!(result, Ok(()));
        assert_eq!(launches[0].kernel, kernel!("euhedral_gdn_project_control_fp32"));
        let (result, launches) = plan_control(&control(2, 1 << 30));
        assert_eq!((result, launches.len()), (Err(PlanError::SizeOverflow), 0));
        let (result, _) = plan_control(&control(1, 1 << 31));
        assert_eq!(result, Err(PlanError::SizeOverflow));
        for bad in [
            ProjectControlShape {
                input: DevicePtr::NULL,
                ..control(1, 1)
            },
            ProjectControlShape {
                a_weight: DevicePtr::NULL,
                ..control(1, 1)
            },
            ProjectControlShape {
                b_weight: DevicePtr::NULL,
                ..control(1, 1)
            },
            ProjectControlShape {
                a_log: DevicePtr::NULL,
                ..control(1, 1)
            },
            ProjectControlShape {
                dt_bias: DevicePtr::NULL,
                ..control(1, 1)
            },
            ProjectControlShape {
                alpha: DevicePtr::NULL,
                ..control(1, 1)
            },
            ProjectControlShape {
                beta: DevicePtr::NULL,
                ..control(1, 1)
            },
            control(0, 1),
            control(1, 0),
            ProjectControlShape {
                width: 0,
                ..control(1, 1)
            },
        ] {
            let (result, launches) = plan_control(&bad);
            assert_eq!((result, launches.len()), (Err(PlanError::InvalidArgument), 0));
        }
    }

    fn conv(rows: u32, qk: u32, value: u32) -> ConvolutionShape {
        ConvolutionShape {
            query_key: ptr(0x100),
            value_z: ptr(0x200),
            weights: ptr(0x300),
            state: ptr(0x400),
            output: ptr(0x500),
            rows,
            query_key_width: qk,
            value_width: value,
            convolution_width: qk + value,
            kernel_size: 4,
        }
    }

    fn plan_conv(shape: &ConvolutionShape) -> (Result<(), PlanError>, Vec<Launch>) {
        let mut launches = Vec::new();
        (plan_convolution(shape, &mut |l| launches.push(l)), launches)
    }

    #[test]
    fn convolution_grid_is_channels_by_row_blocks() {
        // qwen_layer_ops.c:310: ((channels + 127) / 128, (rows + 31) / 32) of 128 threads.
        for (rows, channels, grid) in [
            (1, 128, [1, 1]),
            (32, 129, [2, 1]),
            (33, 10240, [80, 2]),
            (64, 8192 + 2048, [80, 2]),
            (65, 2, [1, 3]),
        ] {
            let (result, launches) = plan_conv(&conv(rows, channels - 1, 1));
            let launch = one(result, launches);
            assert_eq!(launch.kernel, kernel!("euhedral_gdn_convolution_bf16"));
            assert_eq!(launch.grid, [grid[0], grid[1], 1], "rows {rows} channels {channels}");
            assert_eq!(launch.block, [128, 1, 1]);
            assert!(launch.programmatic);
        }
        let (_, launches) = plan_conv(&conv(7, 2048, 6144));
        // :309: five pointers, rows, the three widths, the kernel size.
        assert_eq!(
            launches[0].args.words(),
            &[0x100, 0x200, 0x300, 0x400, 0x500, 7, 2048, 6144, 8192, 4]
        );
        assert_eq!(launches[0].args.sizes(), &[8, 8, 8, 8, 8, 4, 4, 4, 4, 4]);
    }

    #[test]
    fn convolution_validation() {
        // The width test is a u32 sum: 2^31 + 2^31 wraps to 0, which the stated width 0 matches.
        let wrapped = ConvolutionShape {
            query_key_width: 1 << 31,
            value_width: 1 << 31,
            convolution_width: 0,
            ..conv(1, 1, 1)
        };
        let (result, launches) = plan_conv(&wrapped);
        let launch = one(result, launches);
        // (0 + 127) / 128 blocks in x.
        assert_eq!(launch.grid, [0, 1, 1]);
        for size in [0, 1] {
            let (result, _) = plan_conv(&ConvolutionShape {
                kernel_size: size,
                ..conv(1, 1, 1)
            });
            assert_eq!(result, Err(PlanError::InvalidArgument), "kernel size {size}");
        }
        for size in [2, 31, 32] {
            assert_eq!(
                plan_conv(&ConvolutionShape {
                    kernel_size: size,
                    ..conv(1, 1, 1)
                })
                .0,
                Ok(())
            );
        }
        assert_eq!(
            plan_conv(&ConvolutionShape {
                kernel_size: 33,
                ..conv(1, 1, 1)
            })
            .0,
            Err(PlanError::InvalidArgument)
        );
        for bad in [
            ConvolutionShape {
                convolution_width: 3,
                ..conv(1, 1, 1)
            },
            ConvolutionShape {
                query_key: DevicePtr::NULL,
                ..conv(1, 1, 1)
            },
            ConvolutionShape {
                value_z: DevicePtr::NULL,
                ..conv(1, 1, 1)
            },
            ConvolutionShape {
                weights: DevicePtr::NULL,
                ..conv(1, 1, 1)
            },
            ConvolutionShape {
                state: DevicePtr::NULL,
                ..conv(1, 1, 1)
            },
            ConvolutionShape {
                output: DevicePtr::NULL,
                ..conv(1, 1, 1)
            },
            conv(0, 1, 1),
            ConvolutionShape {
                query_key_width: 0,
                convolution_width: 1,
                ..conv(1, 1, 1)
            },
            ConvolutionShape {
                value_width: 0,
                convolution_width: 1,
                ..conv(1, 1, 1)
            },
        ] {
            let (result, launches) = plan_conv(&bad);
            assert_eq!((result, launches.len()), (Err(PlanError::InvalidArgument), 0));
        }
    }

    fn recurrence(rows: u32) -> RecurrenceShape {
        RecurrenceShape {
            convolved: ptr(0x1000),
            alpha: ptr(0x2008),
            beta: ptr(0x3008),
            recurrent_state: ptr(0x4000),
            output: ptr(0x5008),
            rows,
            key_heads: 16,
            value_heads: 48,
            key_head_dim: 128,
            value_head_dim: 128,
            output_scale: 0.088_388_35,
        }
    }

    fn plan_rec(
        shape: &RecurrenceShape,
        row_exact: bool,
        exact_numerics: bool,
    ) -> (Result<(), PlanError>, Vec<Launch>) {
        let mut launches = Vec::new();
        let modes = Modes {
            row_exact,
            exact_numerics,
        };
        (plan_recurrence(shape, modes, &mut |l| launches.push(l)), launches)
    }

    const C8: KernelId = kernel!("euhedral_gdn_recurrence_c8_bf16");
    const WARP: KernelId = kernel!("euhedral_gdn_recurrence_bf16");

    #[test]
    fn recurrence_c8_conditions_each_flipped_alone() {
        let kernel_of = |shape: &RecurrenceShape, row_exact, exact| {
            let (result, launches) = plan_rec(shape, row_exact, exact);
            one(result, launches).kernel
        };
        // qwen_layer_ops.c:343-344: not exact, aligned, rows > 1, not row-exact: all four give c8.
        assert_eq!(kernel_of(&recurrence(2), false, false), C8);
        assert_eq!(kernel_of(&recurrence(1000), false, false), C8);
        // Each flipped alone gives the warp kernel.
        assert_eq!(kernel_of(&recurrence(1), false, false), WARP);
        assert_eq!(kernel_of(&recurrence(2), true, false), WARP);
        assert_eq!(kernel_of(&recurrence(2), false, true), WARP);
        let shape = RecurrenceShape {
            convolved: ptr(0x1008),
            ..recurrence(2)
        };
        assert_eq!(kernel_of(&shape, false, false), WARP);
        let shape = RecurrenceShape {
            recurrent_state: ptr(0x4004),
            ..recurrence(2)
        };
        assert_eq!(kernel_of(&shape, false, false), WARP);
        // Only those two pointers are tested: the others may be unaligned.
        assert_eq!(recurrence(2).alpha.get() & 15, 8);
        assert_eq!(kernel_of(&recurrence(2), false, false), C8);
    }

    #[test]
    fn recurrence_grid_arguments_and_errors() {
        let (result, launches) = plan_rec(&recurrence(2), false, false);
        let launch = one(result, launches);
        // qwen_layer_ops.c:346: 48 * 128 / 8 = 768 blocks of 32 threads.
        assert_eq!((launch.grid, launch.block), ([768, 1, 1], [32, 1, 1]));
        assert!(launch.programmatic);
        // :338-339: five pointers, five u32, the output scale's bits.
        assert_eq!(
            launch.args.words()[..10],
            [0x1000, 0x2008, 0x3008, 0x4000, 0x5008, 2, 16, 48, 128, 128]
        );
        assert_eq!(launch.args.sizes(), &[8, 8, 8, 8, 8, 4, 4, 4, 4, 4, 4]);
        assert_eq!(launch.args.words()[10], u64::from(0.088_388_35f32.to_bits()));
        assert_eq!(launch.args.f32_at(10), Some(0.088_388_35));
        // The warp kernel has the same grid and arguments.
        let (_, warp) = plan_rec(&recurrence(1), false, false);
        assert_eq!(warp[0].grid, [768, 1, 1]);
        // :321: value_heads * 128 must fit in u32: 2^25 heads is 2^32, 2^25 - 8 is not.
        let heads = (1u32 << 25) - 8;
        let (result, launches) = plan_rec(
            &RecurrenceShape {
                value_heads: heads,
                key_heads: 1,
                ..recurrence(1)
            },
            false,
            false,
        );
        assert_eq!(one(result, launches).grid, [(heads as u64 * 128 / 8) as u32, 1, 1]);
        let (result, launches) = plan_rec(
            &RecurrenceShape {
                value_heads: 1 << 25,
                key_heads: 1,
                ..recurrence(1)
            },
            false,
            false,
        );
        assert_eq!((result, launches.len()), (Err(PlanError::SizeOverflow), 0));
        for bad in [
            RecurrenceShape {
                convolved: DevicePtr::NULL,
                ..recurrence(1)
            },
            RecurrenceShape {
                alpha: DevicePtr::NULL,
                ..recurrence(1)
            },
            RecurrenceShape {
                beta: DevicePtr::NULL,
                ..recurrence(1)
            },
            RecurrenceShape {
                recurrent_state: DevicePtr::NULL,
                ..recurrence(1)
            },
            RecurrenceShape {
                output: DevicePtr::NULL,
                ..recurrence(1)
            },
            recurrence(0),
            RecurrenceShape {
                key_heads: 0,
                ..recurrence(1)
            },
            RecurrenceShape {
                value_heads: 0,
                ..recurrence(1)
            },
            RecurrenceShape {
                value_heads: 50,
                ..recurrence(1)
            },
            RecurrenceShape {
                key_head_dim: 64,
                ..recurrence(1)
            },
            RecurrenceShape {
                value_head_dim: 64,
                ..recurrence(1)
            },
            RecurrenceShape {
                output_scale: 0.0,
                ..recurrence(1)
            },
            RecurrenceShape {
                output_scale: -1.0,
                ..recurrence(1)
            },
            RecurrenceShape {
                output_scale: f32::NAN,
                ..recurrence(1)
            },
            RecurrenceShape {
                output_scale: f32::INFINITY,
                ..recurrence(1)
            },
        ] {
            let (result, launches) = plan_rec(&bad, false, false);
            assert_eq!((result, launches.len()), (Err(PlanError::InvalidArgument), 0));
        }
    }

    fn gated(rows: u32, heads: u32) -> GatedRmsNormShape {
        GatedRmsNormShape {
            recurrent: ptr(0x100),
            value_z: ptr(0x200),
            norm_weight: ptr(0x300),
            output: ptr(0x400),
            rows,
            value_heads: heads,
            head_dim: 128,
            epsilon: 1e-6,
        }
    }

    fn plan_gated(shape: &GatedRmsNormShape) -> (Result<(), PlanError>, Vec<Launch>) {
        let mut launches = Vec::new();
        (plan_gated_rms_norm(shape, &mut |l| launches.push(l)), launches)
    }

    #[test]
    fn gated_rms_norm_grid_arguments_and_errors() {
        let (result, launches) = plan_gated(&gated(5, 48));
        let launch = one(result, launches);
        // qwen_layer_ops.c:368: rows * heads blocks of 128.
        assert_eq!(launch.kernel, kernel!("euhedral_gdn_gated_rms_norm_bf16"));
        assert_eq!((launch.grid, launch.block), ([240, 1, 1], [128, 1, 1]));
        // :367: four pointers, rows, heads, dim, epsilon's bits.
        assert_eq!(
            launch.args.words(),
            &[0x100, 0x200, 0x300, 0x400, 5, 48, 128, u64::from(1e-6f32.to_bits())]
        );
        assert_eq!(launch.args.sizes(), &[8, 8, 8, 8, 4, 4, 4, 4]);
        // 65536 * 65535 fits, 65536 * 65536 does not.
        let (result, launches) = plan_gated(&gated(65536, 65535));
        assert_eq!(one(result, launches).grid, [4_294_901_760, 1, 1]);
        assert_eq!(plan_gated(&gated(65536, 65536)).0, Err(PlanError::SizeOverflow));
        assert_eq!(
            plan_gated(&GatedRmsNormShape {
                epsilon: 0.0,
                ..gated(1, 1)
            })
            .0,
            Ok(())
        );
        for bad in [
            GatedRmsNormShape {
                recurrent: DevicePtr::NULL,
                ..gated(1, 1)
            },
            GatedRmsNormShape {
                value_z: DevicePtr::NULL,
                ..gated(1, 1)
            },
            GatedRmsNormShape {
                norm_weight: DevicePtr::NULL,
                ..gated(1, 1)
            },
            GatedRmsNormShape {
                output: DevicePtr::NULL,
                ..gated(1, 1)
            },
            gated(0, 1),
            gated(1, 0),
            GatedRmsNormShape {
                head_dim: 64,
                ..gated(1, 1)
            },
            GatedRmsNormShape {
                epsilon: f32::NAN,
                ..gated(1, 1)
            },
            GatedRmsNormShape {
                epsilon: -1.0,
                ..gated(1, 1)
            },
        ] {
            let (result, launches) = plan_gated(&bad);
            assert_eq!((result, launches.len()), (Err(PlanError::InvalidArgument), 0));
        }
    }
}
