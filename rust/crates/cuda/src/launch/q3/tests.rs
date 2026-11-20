//! Host tests of the grouped-linear planner. Every expected value is worked by hand from
//! `native/src/host/q3_linear_bf16.c`, `q45_linear.c` and `q3_mx.c`, not taken from the planner.

use super::*;

const INPUT: u64 = 0x1000;
const WEIGHTS: u64 = 0x2000;
const OUTPUT: u64 = 0x3000;
const SCRATCH: u64 = 0x10000;

fn at(address: u64) -> DevicePtr {
    // SAFETY: the plan only does arithmetic on the address and never dereferences it.
    unsafe { DevicePtr::from_raw(address) }
}

fn operands(bits: Bits, rows: u32, in_features: u32, out_features: u32) -> Q3Operands {
    Q3Operands {
        input: at(INPUT),
        weights: at(WEIGHTS),
        output: at(OUTPUT),
        rows,
        in_features,
        out_features,
        weights_byte_size: quantized_byte_size(bits, out_features, in_features).unwrap(),
    }
}

fn only(plan: &Q3Plan) -> &Launch {
    assert_eq!(plan.launches().len(), 1);
    &plan.launches()[0]
}

#[test]
fn byte_sizes() {
    // 16 rows of 1024: 16 groups of 64 a row. Q3 codes are 24 bytes a group: 16 * 16 * 24 = 6144 (a multiple of 256),
    // the FP16 scales 16 * 16 * 2 = 512.
    assert_eq!(q3_scale_offset(16, 1024), 6144);
    assert_eq!(q3_byte_size(16, 1024), 6656);
    // Q4 codes are 32 bytes a group: 8192; Q5 adds a fifth-bit plane of 8 bytes a group: 2048.
    assert_eq!(quantized_byte_size(Bits::Q4, 16, 1024), Some(8704));
    assert_eq!(quantized_byte_size(Bits::Q5, 16, 1024), Some(10752));
    assert_eq!(quantized_byte_size(Bits::Q3, 16, 1024), Some(6656));
    // A width that is not a multiple of 128 has no Q4 or Q5 tensor.
    assert_eq!(quantized_byte_size(Bits::Q4, 16, 1000), None);
}

#[test]
fn rows_pick_the_decode_kernel_and_its_grid() {
    let one = plan_linear_q3(&operands(Bits::Q3, 1, 1024, 32), false).unwrap();
    let launch = only(&one);
    assert_eq!(launch.kernel.name(), "euhedral_q3_decode_contiguous");
    assert_eq!(
        (launch.grid, launch.block, launch.shared_bytes),
        ([2, 1, 1], [128, 1, 1], 0)
    );
    assert_eq!(
        launch.args.words(),
        [INPUT, WEIGHTS, OUTPUT, 1, 1024, 32, q3_scale_offset(32, 1024)]
    );
    assert_eq!(launch.args.sizes(), [8, 8, 8, 4, 4, 4, 8]);
    for rows in 2..=8 {
        let plan = plan_linear_q3(&operands(Bits::Q3, rows, 1024, 32), false).unwrap();
        assert_eq!(
            only(&plan).kernel.name(),
            format!("euhedral_q3_decode_contiguous_rows{rows}")
        );
        assert_eq!(only(&plan).grid, [2, 1, 1]);
    }
    // 9 to 32 rows run on the token-row twin, four outputs a block.
    let tokens = plan_linear_q3(&operands(Bits::Q3, 9, 1024, 32), false).unwrap();
    assert_eq!(only(&tokens).kernel.name(), "euhedral_q3_decode_contiguous_tokens");
    assert_eq!(only(&tokens).grid, [8, 1, 1]);
}

#[test]
fn q4_and_q5_run_eight_outputs_a_block_and_carry_no_scale_offset() {
    for (bits, name) in [(Bits::Q4, "q4"), (Bits::Q5, "q5")] {
        let plan = plan_linear_q45(bits, &operands(bits, 2, 1024, 32), false).unwrap();
        let launch = only(&plan);
        assert_eq!(launch.kernel.name(), format!("euhedral_{name}_decode_contiguous_rows2"));
        assert_eq!(launch.grid, [4, 1, 1]);
        assert_eq!(launch.args.words(), [INPUT, WEIGHTS, OUTPUT, 2, 1024, 32]);
    }
    let wrong = Q3Operands {
        weights_byte_size: 1,
        ..operands(Bits::Q4, 2, 1024, 32)
    };
    assert_eq!(plan_linear_q45(Bits::Q4, &wrong, false), Err(PlanError::FormatMismatch));
    assert_eq!(
        plan_linear_q45(Bits::Q3, &operands(Bits::Q3, 2, 1024, 32), false),
        Err(PlanError::InvalidArgument)
    );
}

#[test]
fn the_reference_runs_under_exact_numerics_and_for_shapes_the_decode_kernels_do_not_take() {
    let exact = plan_linear_q3(&operands(Bits::Q3, 3, 1024, 32), true).unwrap();
    assert_eq!(only(&exact).kernel.name(), "euhedral_q3_reference");
    assert_eq!(only(&exact).grid, [96, 1, 1]);
    // 1024 is the narrowest width of the decode kernels; 128 falls back.
    let narrow = plan_linear_q3(&operands(Bits::Q3, 3, 128, 32), false).unwrap();
    assert_eq!(only(&narrow).kernel.name(), "euhedral_q3_reference");
    // More than 32 rows have no decode kernel.
    let many = plan_linear_q3(&operands(Bits::Q3, 33, 1024, 32), false).unwrap();
    assert_eq!(only(&many).kernel.name(), "euhedral_q3_reference");
    let q4 = plan_linear_q45(Bits::Q4, &operands(Bits::Q4, 3, 128, 32), false).unwrap();
    assert_eq!(only(&q4).kernel.name(), "euhedral_q45_reference");
    assert_eq!(only(&q4).args.words(), [INPUT, WEIGHTS, OUTPUT, 3, 128, 32, 4]);
}

#[test]
fn a_misaligned_input_takes_the_reference() {
    let mut misaligned = operands(Bits::Q3, 1, 1024, 32);
    misaligned.input = at(INPUT + 2);
    let plan = plan_linear_q3(&misaligned, false).unwrap();
    assert_eq!(only(&plan).kernel.name(), "euhedral_q3_reference");
}

#[test]
fn a_size_that_is_not_the_shapes_is_a_format_mismatch() {
    let wrong = Q3Operands {
        weights_byte_size: q3_byte_size(32, 1024) + 4,
        ..operands(Bits::Q3, 1, 1024, 32)
    };
    assert_eq!(plan_linear_q3(&wrong, false), Err(PlanError::FormatMismatch));
}

#[test]
fn the_fp8_splits_follow_the_tiles_and_the_multiprocessors() {
    // 512 rows over 5120 outputs: 4 x 40 tiles. On 70 multiprocessors one wave set is ceil(160 / 70) = 3 waves; two
    // splits make ceil(320 / 70) = 5 waves over 2 = 2.5, times 1.04 = 2.6, below 0.92 * 3 = 2.76; four make
    // ceil(640 / 70) = 10 over 4 = 2.5, times 1.12 = 2.8, which is worse.
    assert_eq!(mx_splits(512, 5120, 5120, 70, 0), 2);
    // A skinny launch of 1 x 40 tiles is under one wave: four splits make ceil(160 / 70) = 3 waves over 4 = 0.75, times
    // 1.12 = 0.84, below 0.92 * 1 (two make 2 waves over 2 = 1, times 1.04, which is not better).
    assert_eq!(mx_splits(9, 5120, 5120, 70, 0), 4);
    // A whole number of waves gains nothing from a split: 1 x 70 tiles are one wave and two splits cost 1.04.
    assert_eq!(mx_splits(9, 5120, 8960, 70, 0), 1);
    // The splits follow the whole tensor when it is run in chunks: 128 outputs of a 5120-row tensor.
    assert_eq!(mx_splits(512, 5120, 128, 70, 5120), 2);
    // 4 x 1 tiles: four splits make one wave over 4 = 0.25, times 1.12.
    assert_eq!(mx_splits(512, 5120, 128, 70, 0), 4);
}

#[test]
fn the_fp8_scratch_is_its_planes_and_its_partials() {
    // Two planes of 512 x 5120 = 2621440 bytes, the 512 x 160 scales = 81920, and two splits of 512 x 5120 floats.
    assert_eq!(
        mx_scratch_bytes(512, 5120, 5120, 70, 0),
        2 * 2_621_440 + 81_920 + 2 * 512 * 5120 * 4
    );
    // Without a weight there are no partials.
    assert_eq!(mx_scratch_bytes(512, 5120, 0, 70, 0), 2 * 2_621_440 + 81_920);
}

#[test]
fn the_largest_scratch_up_to_a_row_count_is_no_smaller_than_any_launch_of_fewer_rows() {
    for (width, outputs, sm_count) in [(5120, 5120, 70), (5120, 17_408, 70), (1024, 4096, 140), (5120, 5120, 1)] {
        for limit in [9, 64, 128, 129, 300, 512, 1000] {
            let upto = mx_scratch_bytes_upto(limit, width, outputs, Some(sm_count));
            for rows in MX_MIN_ROWS..=limit {
                assert!(
                    upto >= mx_scratch_bytes(rows, width, outputs, sm_count, 0),
                    "{rows} rows of {limit} over {width} x {outputs} on {sm_count}"
                );
            }
            // A device that is not known gets the most any device takes.
            assert!(mx_scratch_bytes_upto(limit, width, outputs, None) >= upto);
        }
    }
    assert_eq!(mx_scratch_bytes_upto(MX_MIN_ROWS - 1, 5120, 5120, Some(70)), 0);
}

#[test]
fn the_scratch_of_a_linear() {
    let upto = |p2e2, rows, mx| linear_scratch_bytes(p2e2, rows, 5120, 5120, mx, Some(70), false);
    // Up to 8 rows take no scratch, whatever the layout.
    assert_eq!(upto(false, 8, true), 0);
    assert_eq!(upto(true, 8, true), 0);
    // A plain tensor takes the FP8 route's scratch where it runs, and none where it does not or the shape is not tiled.
    assert_eq!(upto(false, 512, true), mx_scratch_bytes_upto(512, 5120, 5120, Some(70)));
    assert_eq!(upto(false, 512, false), 0);
    assert_eq!(linear_scratch_bytes(false, 512, 5120, 5000, true, Some(70), false), 0);
    // A P2E2 tensor takes its expansion, 256-aligned, with the FP8 scratch behind it.
    let expanded = q3_byte_size(5120, 5120);
    assert_eq!(upto(true, 512, false), expanded);
    assert_eq!(
        upto(true, 512, true),
        expanded + mx_scratch_bytes_upto(512, 5120, 5120, Some(70))
    );
    // Where the fused decode does not run, a P2E2 tensor of few rows is expanded as well: under exact numerics, and
    // for a shape the decode kernels do not take.
    assert_eq!(
        linear_scratch_bytes(true, 8, 5120, 5120, true, Some(70), true),
        expanded
    );
    assert_eq!(
        linear_scratch_bytes(true, 1, 5120, 5120, false, Some(70), true),
        expanded
    );
    assert_eq!(linear_scratch_bytes(false, 8, 5120, 5120, true, Some(70), true), 0);
    assert!(linear_scratch_bytes(true, 8, 5120, 5000, true, Some(70), false) > 0);
    // One too large to expand whole is run in chunks: the scratch holds one chunk's expansion, its output and its
    // activations, so it is far below the whole expansion and does not grow with the head.
    let chunked = linear_scratch_bytes(true, 512, 5120, 248_320, true, Some(70), false);
    assert!(chunked > 0 && chunked < q3_byte_size(248_320, 5120));
    assert_eq!(
        linear_scratch_bytes(true, 8, 5120, 496_640, false, Some(70), true),
        linear_scratch_bytes(true, 8, 5120, 248_320, false, Some(70), true)
    );
}

/// The chunks of a head tile its rows in whole kilorows, and a chunk with its output fits the expansion limit; a
/// chunk is never one of the widths the FFN and mixer kernels are specialised for.
#[test]
fn the_chunks_of_a_large_head_fit_the_expansion_limit_and_avoid_the_specialised_widths() {
    for rows in [1, 8, 9, 64, 512] {
        let chunk = p2e2_chunk_rows(rows, 5120);
        assert_eq!(chunk % 1024, 0);
        let (_, activations) = p2e2_chunk_offsets(rows, 5120, chunk);
        assert!(activations <= P2E2_EXPAND_LIMIT + 512, "{rows}");
    }
    // Widths of K whose unguarded chunk of one row is 5120 and 34816.
    for input in [57_344, 9344] {
        let per_unit = q3_byte_size(1024, input) + 1024 * 2;
        let unguarded = (P2E2_EXPAND_LIMIT / per_unit) * 1024;
        assert!(unguarded == 5120 || unguarded == 34816, "{input}: {unguarded}");
        assert_eq!(u64::from(p2e2_chunk_rows(1, input)), unguarded - 1024);
    }
}

fn mx(rows: u32, scratch: u64, scratch_bytes: u64) -> Result<Q3Plan, PlanError> {
    plan_mx(
        Bits::Q3,
        &operands(Bits::Q3, rows, 5120, 5120),
        at(scratch),
        scratch_bytes,
        70,
        0,
        false,
        false,
    )
}

#[test]
fn the_fp8_route_quantizes_multiplies_and_reduces_the_partials() {
    let plan = mx(512, SCRATCH, u64::MAX).unwrap();
    let [quantize, linear, reduce] = plan.launches() else {
        panic!("{:?}", plan.launches());
    };
    // 512 x 160 scale groups, 256 to a block.
    assert_eq!(quantize.kernel.name(), "euhedral_q3mx_quantize");
    assert_eq!((quantize.grid, quantize.block), ([320, 1, 1], [256, 1, 1]));
    let plane = 2_621_440;
    let (high, low, scales) = (SCRATCH, SCRATCH + plane, SCRATCH + 2 * plane);
    assert_eq!(quantize.args.words(), [INPUT, high, low, scales, 512, 5120]);
    assert_eq!(linear.kernel.name(), "euhedral_q3mx_linear_128x128");
    assert_eq!(
        (linear.grid, linear.block, linear.shared_bytes),
        ([160, 2, 1], [384, 1, 1], MX_SHARED_BYTES)
    );
    assert_eq!(MX_SHARED_BYTES, 75_264);
    assert_eq!(reduce.kernel.name(), "euhedral_q3mx_reduce");
}

#[test]
fn the_fp8_route_without_a_split_has_no_reduction() {
    let plan = plan_mx(
        Bits::Q3,
        &operands(Bits::Q3, 9, 5120, 8960),
        at(SCRATCH),
        u64::MAX,
        70,
        0,
        false,
        false,
    )
    .unwrap();
    assert_eq!(plan.launches().len(), 2);
    assert_eq!(plan.launches()[1].grid, [70, 1, 1]);
}

#[test]
fn the_paired_gate_up_never_splits_and_needs_only_the_activations() {
    // 64 rows of width 5120 against 34816 gate and up rows: the tiles are 1 x 272 and the paired kernel never splits.
    let paired = |scratch_bytes| {
        plan_mx_gate_up_swiglu(
            &operands(Bits::Q3, 64, 5120, 34816),
            at(SCRATCH),
            scratch_bytes,
            false,
            false,
        )
    };
    // Two planes of 64 x 5120 bytes and the 64 x 160 scales, each 256-aligned.
    let needed = 2 * 327_680 + 10_240;
    assert_eq!(mx_scratch_bytes(64, 5120, 0, 170, 0), needed);
    let plan = paired(needed).unwrap();
    let [quantize, gate_up] = plan.launches() else {
        panic!("{:?}", plan.launches());
    };
    assert_eq!(quantize.kernel.name(), "euhedral_q3mx_quantize");
    assert_eq!((quantize.grid, quantize.block), ([40, 1, 1], [256, 1, 1]));
    assert_eq!(gate_up.kernel.name(), "euhedral_q3mx_gate_up_swiglu_128x64");
    assert_eq!(
        (gate_up.grid, gate_up.block, gate_up.shared_bytes),
        ([272, 1, 1], [384, 1, 1], MX_SHARED_BYTES)
    );
    let (high, low, scales) = (SCRATCH, SCRATCH + 327_680, SCRATCH + 2 * 327_680);
    assert_eq!(
        gate_up.args.words(),
        [
            high,
            low,
            scales,
            WEIGHTS,
            OUTPUT,
            64,
            34816,
            5120,
            0,
            q3_scale_offset(34816, 5120),
            0
        ]
    );
    assert_eq!(paired(needed - 256), Err(PlanError::InvalidArgument));
    // It declines as the other FP8 routes do.
    let declines = |exact, row_exact| {
        plan_mx_gate_up_swiglu(
            &operands(Bits::Q3, 64, 5120, 34816),
            at(SCRATCH),
            needed,
            exact,
            row_exact,
        )
    };
    assert_eq!(declines(true, false), Err(PlanError::RouteUnavailable));
    assert_eq!(declines(false, true), Err(PlanError::RouteUnavailable));
    assert_eq!(
        plan_mx_gate_up_swiglu(&operands(Bits::Q3, 64, 5120, 34800), at(SCRATCH), needed, false, false),
        Err(PlanError::RouteUnavailable)
    );
}

#[test]
fn a_linear_of_the_scratch_covers_its_paired_gate_up() {
    for sm_count in [Some(70), Some(170), None] {
        for rows in [64, 129, 512, 1000] {
            let linear = linear_scratch_bytes(false, rows, 5120, 34816, true, sm_count, false);
            assert!(
                mx_scratch_bytes_upto(rows, 5120, 0, sm_count) <= linear,
                "{rows} {sm_count:?}"
            );
        }
    }
}

#[test]
fn the_fp8_route_declines_what_it_has_no_kernel_for_and_refuses_what_is_wrong() {
    let declined = |plan: Result<Q3Plan, PlanError>| assert_eq!(plan, Err(PlanError::RouteUnavailable));
    // A width or an output count that is not a multiple of 128.
    declined(plan_mx(
        Bits::Q3,
        &operands(Bits::Q3, 64, 5120, 5000),
        at(SCRATCH),
        u64::MAX,
        70,
        0,
        false,
        false,
    ));
    // Exact numerics and row-exact execution keep the BF16 kernels.
    declined(plan_mx(
        Bits::Q3,
        &operands(Bits::Q3, 64, 5120, 5120),
        at(SCRATCH),
        u64::MAX,
        70,
        0,
        true,
        false,
    ));
    declined(plan_mx(
        Bits::Q3,
        &operands(Bits::Q3, 64, 5120, 5120),
        at(SCRATCH),
        u64::MAX,
        70,
        0,
        false,
        true,
    ));
    // An input that is not 16-aligned.
    let mut misaligned = operands(Bits::Q3, 64, 5120, 5120);
    misaligned.input = at(INPUT + 8);
    declined(plan_mx(
        Bits::Q3,
        &misaligned,
        at(SCRATCH),
        u64::MAX,
        70,
        0,
        false,
        false,
    ));
    // A scratch that is too small, or not 256-aligned, is the caller's error.
    assert_eq!(mx(512, SCRATCH, 1000), Err(PlanError::InvalidArgument));
    assert_eq!(mx(512, SCRATCH + 16, u64::MAX), Err(PlanError::InvalidArgument));
    // A size that is not the tensor's.
    let wrong = Q3Operands {
        weights_byte_size: 1,
        ..operands(Bits::Q3, 64, 5120, 5120)
    };
    assert_eq!(
        plan_mx(Bits::Q3, &wrong, at(SCRATCH), u64::MAX, 70, 0, false, false),
        Err(PlanError::FormatMismatch)
    );
}

#[test]
fn q4_and_q5_take_their_planes_to_the_fp8_kernel() {
    for (bits, kernel, high, scales) in [
        (Bits::Q4, "euhedral_q4mx_linear_128x128", 5120 * 80 * 32, 5120 * 80 * 32),
        (
            Bits::Q5,
            "euhedral_q5mx_linear_128x128",
            5120 * 80 * 32,
            5120 * 80 * 32 + 5120 * 80 * 8,
        ),
    ] {
        let plan = plan_mx(
            bits,
            &operands(bits, 64, 5120, 5120),
            at(SCRATCH),
            u64::MAX,
            70,
            0,
            false,
            false,
        )
        .unwrap();
        let linear = &plan.launches()[1];
        assert_eq!(linear.kernel.name(), kernel);
        let words = linear.args.words();
        // The two offsets follow the output pointer, the row count, the weight rows and the width.
        assert_eq!(&words[8..10], [high, scales], "{bits:?}");
    }
}

/// The smallest valid P2E2 tensor of 16 rows of 1024: planes at 4096, 4352 and 4864, and 80 words of padding.
const P2E2_BYTES: u64 = 4864 + 320;

#[test]
fn p2e2_decode_runs_four_rows_a_launch() {
    let operands = Q3Operands {
        weights_byte_size: P2E2_BYTES,
        ..operands(Bits::Q3, 6, 1024, 16)
    };
    let plan = plan_p2e2_decode(&operands, false).unwrap();
    let [first, second] = plan.launches() else {
        panic!("{:?}", plan.launches());
    };
    assert_eq!(first.kernel.name(), "euhedral_q3_p2e2_decode_rows4");
    assert_eq!(second.kernel.name(), "euhedral_q3_p2e2_decode_rows2");
    assert_eq!((first.grid, first.block), ([1, 1, 1], [128, 1, 1]));
    assert_eq!(first.args.words(), [INPUT, WEIGHTS, OUTPUT, 1024, 16]);
    // The second launch starts four rows on: 4 x 1024 BF16 values of input and 4 x 16 of output.
    assert_eq!(
        second.args.words(),
        [INPUT + 4 * 1024 * 2, WEIGHTS, OUTPUT + 4 * 16 * 2, 1024, 16]
    );
}

#[test]
fn p2e2_decode_declines_what_it_does_not_run() {
    let operands = Q3Operands {
        weights_byte_size: P2E2_BYTES,
        ..operands(Bits::Q3, 6, 1024, 16)
    };
    assert_eq!(plan_p2e2_decode(&operands, true), Err(PlanError::RouteUnavailable));
    let nine = Q3Operands { rows: 9, ..operands };
    assert_eq!(plan_p2e2_decode(&nine, false), Err(PlanError::RouteUnavailable));
    let misaligned = Q3Operands {
        weights: at(WEIGHTS + 8),
        ..operands
    };
    assert_eq!(plan_p2e2_decode(&misaligned, false), Err(PlanError::RouteUnavailable));
    assert!(!p2e2_decode_runs(&nine, false));
    assert!(p2e2_decode_runs(&operands, false));
    // A size outside the bounds of a payload is a mismatch, not a decline.
    let wrong = Q3Operands {
        weights_byte_size: P2E2_BYTES - 4,
        ..operands
    };
    assert_eq!(plan_p2e2_decode(&wrong, false), Err(PlanError::FormatMismatch));
}

#[test]
fn p2e2_expansion_takes_four_rows_a_block() {
    let plan = plan_p2e2_expand(
        at(WEIGHTS),
        P2E2_BYTES,
        16,
        1024,
        4,
        10,
        at(SCRATCH),
        q3_byte_size(10, 1024),
    )
    .unwrap();
    let launch = only(&plan);
    assert_eq!(launch.kernel.name(), "euhedral_q3_p2e2_expand");
    assert_eq!((launch.grid, launch.block), ([3, 1, 1], [128, 1, 1]));
    assert_eq!(launch.args.words(), [WEIGHTS, SCRATCH, 16, 1024, 4, 10]);
    // Rows beyond the tensor, and a destination too small for them.
    assert_eq!(
        plan_p2e2_expand(at(WEIGHTS), P2E2_BYTES, 16, 1024, 10, 7, at(SCRATCH), u64::MAX),
        Err(PlanError::InvalidArgument)
    );
    assert_eq!(
        plan_p2e2_expand(at(WEIGHTS), P2E2_BYTES, 16, 1024, 0, 16, at(SCRATCH), 100),
        Err(PlanError::SizeOverflow)
    );
}
