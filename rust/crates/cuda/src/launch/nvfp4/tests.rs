//! Host tests of the NVFP4 planner. Every expected value is worked by hand from `native/src/host/nvfp4_linear.c` (the
//! line is named where it is used), not taken from the planner.

use super::*;

const INPUT: u64 = 0x1000;
const WEIGHTS: u64 = 0x2000;
const OUTPUT: u64 = 0x3000;
const SCRATCH: u64 = 0x4000;

fn at(address: u64) -> DevicePtr {
    // SAFETY: the plan only does arithmetic on the address and never dereferences it.
    unsafe { DevicePtr::from_raw(address) }
}

fn operands(rows: u32, in_features: u32, out_features: u32, layout: Nvfp4Layout) -> Nvfp4Operands {
    Nvfp4Operands {
        input: at(INPUT),
        weights: at(WEIGHTS),
        output: at(OUTPUT),
        rows,
        in_features,
        out_features,
        weights_byte_size: nvfp4_byte_size(out_features, in_features, layout),
    }
}

fn linear(rows: u32, in_features: u32, out_features: u32) -> Nvfp4Plan {
    plan_linear(&operands(rows, in_features, out_features, Nvfp4Layout::Plain), false).unwrap()
}

fn native(rows: u32, in_features: u32, out_features: u32, layout: Nvfp4Layout) -> Nvfp4Plan {
    plan_native(
        &operands(rows, in_features, out_features, layout),
        at(SCRATCH),
        u64::MAX,
        false,
    )
    .unwrap()
}

fn only(plan: &Nvfp4Plan) -> &Launch {
    assert_eq!(plan.launches().len(), 1);
    &plan.launches()[0]
}

#[test]
fn layout_from_the_byte_size() {
    // nvfp4_size, 16 rows of 128: k = 128, the code plane is 16 * 128 / 2 = 1024 bytes (already a multiple of
    // 256); plain adds 16 * 128 / 16 = 128 bytes of scales, 1152, rounded to 1280, plus the 4-byte global; SD4 adds 64,
    // 1088, rounded to 1280, plus the 20-byte table and global.
    assert_eq!(nvfp4_byte_size(16, 128, Nvfp4Layout::Plain), 1284);
    assert_eq!(nvfp4_byte_size(16, 128, Nvfp4Layout::Sd4), 1300);
    // One row of 32: k pads to 128, the code plane is 64 bytes, rounded to 256; scales 8 or 4 bytes, rounded to 512.
    assert_eq!(nvfp4_byte_size(1, 32, Nvfp4Layout::Plain), 516);
    assert_eq!(nvfp4_byte_size(1, 32, Nvfp4Layout::Sd4), 532);
    // nvfp4_layout.
    assert_eq!(nvfp4_layout(16, 128, 1284), Some(Nvfp4Layout::Plain));
    assert_eq!(nvfp4_layout(16, 128, 1300), Some(Nvfp4Layout::Sd4));
    assert_eq!(nvfp4_layout(16, 128, 1292), None);
}

#[test]
fn decode_tile_choice() {
    // decode_tile: the 4-warp tile above 5120 inputs with whole 32 outputs.
    assert!(!decode_tile(5120, 5120));
    assert!(decode_tile(5121, 32));
    assert!(decode_tile(5248, 64));
    assert!(!decode_tile(5248, 48));
    assert!(!decode_tile(4096, 64));
    // Grid out / 32 and 128 threads, or out / 16 and 256.
    let wide = linear(1, 5248, 64);
    let wide = only(&wide);
    assert_eq!(wide.kernel.name(), "euhedral_nvfp4_decode_rows1_w4");
    assert_eq!((wide.grid, wide.block, wide.shared_bytes), ([2, 1, 1], [128, 1, 1], 0));
    let narrow = linear(1, 5248, 48);
    let narrow = only(&narrow);
    assert_eq!(narrow.kernel.name(), "euhedral_nvfp4_decode_rows1_w8");
    assert_eq!((narrow.grid, narrow.block), ([3, 1, 1], [256, 1, 1]));
    let plain = linear(4, 5120, 5120);
    let plain = only(&plain);
    assert_eq!(plain.kernel.name(), "euhedral_nvfp4_decode_rows4_w8");
    assert_eq!((plain.grid, plain.block), ([320, 1, 1], [256, 1, 1]));
}

#[test]
fn decode_arguments_are_the_pointers_then_the_widths() {
    let plan = linear(3, 128, 16);
    let launch = only(&plan);
    assert_eq!(launch.kernel.name(), "euhedral_nvfp4_decode_rows3_w8");
    assert_eq!(launch.args.words(), [INPUT, WEIGHTS, OUTPUT, 128, 16]);
    assert_eq!(launch.args.sizes(), [8, 8, 8, 4, 4]);
    assert!(launch.programmatic);
}

#[test]
fn row_edges_of_the_decode_kernels() {
    // 1 to 8 rows the rows kernels, 9 to 32 the groups kernel of ceil(rows / 8), 33 and up the reference.
    for (rows, kernel) in [
        (1, "euhedral_nvfp4_decode_rows1_w8"),
        (7, "euhedral_nvfp4_decode_rows7_w8"),
        (8, "euhedral_nvfp4_decode_rows8_w8"),
        (9, "euhedral_nvfp4_decode_groups2_w8"),
        (16, "euhedral_nvfp4_decode_groups2_w8"),
        (17, "euhedral_nvfp4_decode_groups3_w8"),
        (24, "euhedral_nvfp4_decode_groups3_w8"),
        (25, "euhedral_nvfp4_decode_groups4_w8"),
        (32, "euhedral_nvfp4_decode_groups4_w8"),
        (33, "euhedral_nvfp4_reference"),
    ] {
        let plan = linear(rows, 128, 16);
        assert_eq!(only(&plan).kernel.name(), kernel, "rows {rows}");
    }
    // The grouped kernels take the rows as a last 32-bit argument.
    let plan = linear(9, 128, 16);
    let grouped = only(&plan);
    assert_eq!(grouped.args.words(), [INPUT, WEIGHTS, OUTPUT, 128, 16, 9]);
    assert_eq!(grouped.args.sizes(), [8, 8, 8, 4, 4, 4]);
    // The SD4 twins and the wide tile.
    let plan = plan_linear(&operands(20, 5248, 64, Nvfp4Layout::Sd4), false).unwrap();
    let launch = only(&plan);
    assert_eq!(launch.kernel.name(), "euhedral_nvfp4_decode_groups3_w4_sd4");
    assert_eq!((launch.grid, launch.block), ([2, 1, 1], [128, 1, 1]));
}

#[test]
fn reference_selection() {
    // Exact numerics, an input width that is not a multiple of 128, an output count that is not a multiple of
    // 16: the reference, grid rows * out, 128 threads, arguments rows, in, out (reference.c).
    let exact = plan_linear(&operands(1, 128, 16, Nvfp4Layout::Plain), true).unwrap();
    let launch = only(&exact);
    assert_eq!(launch.kernel.name(), "euhedral_nvfp4_reference");
    assert_eq!(
        (launch.grid, launch.block, launch.shared_bytes),
        ([16, 1, 1], [128, 1, 1], 0)
    );
    assert_eq!(launch.args.words(), [INPUT, WEIGHTS, OUTPUT, 1, 128, 16]);
    assert!(!launch.programmatic);
    let narrow = linear(2, 160, 16);
    assert_eq!(only(&narrow).kernel.name(), "euhedral_nvfp4_reference");
    assert_eq!(only(&narrow).grid, [32, 1, 1]);
    let odd = linear(2, 128, 24);
    assert_eq!(only(&odd).grid, [48, 1, 1]);
    let sd4 = plan_linear(&operands(40, 128, 16, Nvfp4Layout::Sd4), false).unwrap();
    assert_eq!(only(&sd4).kernel.name(), "euhedral_nvfp4_reference_sd4");
    // The grid is capped at 2^31 - 1: 200 million rows of 16 outputs is 3.2 billion.
    let big = linear(200_000_000, 128, 16);
    assert_eq!(only(&big).grid, [2_147_483_647, 1, 1]);
}

#[test]
fn refusals_before_any_launch() {
    let good = operands(1, 128, 16, Nvfp4Layout::Plain);
    assert!(plan_linear(&good, false).is_ok());
    let refused = |change: &dyn Fn(&mut Nvfp4Operands)| {
        let mut changed = good;
        change(&mut changed);
        plan_linear(&changed, false).unwrap_err()
    };
    assert_eq!(refused(&|o| o.input = DevicePtr::NULL), Nvfp4Refusal::InvalidArgument);
    assert_eq!(refused(&|o| o.rows = 0), Nvfp4Refusal::InvalidArgument);
    assert_eq!(refused(&|o| o.weights_byte_size += 4), Nvfp4Refusal::FormatMismatch);
    assert_eq!(refused(&|o| o.input = at(INPUT + 8)), Nvfp4Refusal::InvalidArgument);
    assert_eq!(refused(&|o| o.weights = at(WEIGHTS + 4)), Nvfp4Refusal::InvalidArgument);
    // 48 inputs: a valid size for the shape, but not a multiple of 32.
    let narrow = operands(1, 48, 16, Nvfp4Layout::Plain);
    assert_eq!(plan_linear(&narrow, false).unwrap_err(), Nvfp4Refusal::InvalidArgument);
}

#[test]
fn skinny_split_counts() {
    // skinny_splits: tiles = ceil(out / 64), splits = ceil(140 / tiles) capped at k_tiles / 2, at least 1.
    assert_eq!(skinny_splits(4096, 4096), 3); // 64 tiles: ceil(140 / 64) = 3, cap 16 / 2 = 8
    assert_eq!(skinny_splits(4096, 8896), 2); // 139 tiles: ceil(140 / 139) = 2
    assert_eq!(skinny_splits(4096, 8897), 1); // 140 tiles: 1
    assert_eq!(skinny_splits(4096, 64), 8); // 1 tile: 140, capped at 8
    assert_eq!(skinny_splits(1024, 64), 2); // cap 4 / 2
    assert_eq!(skinny_splits(512, 64), 1); // cap 2 / 2
    assert_eq!(skinny_splits(256, 64), 1); // cap 0, raised to 1
}

#[test]
fn skinny_shared_bytes_of_the_three_kernels() {
    // skinny_shared_bytes, two terms. 16 rows: stage (32 + 64) * 160 = 15360, 101376 / 15360 = 6, capped at
    // 4 stages: 61440. 32 rows: 128 * 160 = 20480, 4 stages: 81920. 64 rows: 192 * 160 = 30720, 3 stages: 92160.
    assert_eq!(skinny_shared_bytes(2, 16), 61440);
    assert_eq!(skinny_shared_bytes(2, 32), 81920);
    assert_eq!(skinny_shared_bytes(2, 64), 92160);
}

#[test]
fn native_skinny_plan_with_a_split() {
    // 12 rows, 5120 by 5120: 80 tiles, ceil(140 / 80) = 2 splits (cap 10); the 16-row kernel; the activations take
    // 143408 bytes (k = 5120, 24 planes: 61440 of codes, 40 * 2 * 128 * 8 = 81920 of scales, 4 * 12 of globals), so the
    // partials start at the next 256: 143616 past the scratch.
    let plan = native(12, 5120, 5120, Nvfp4Layout::Plain);
    assert_eq!(plan.tensor_maps(), None);
    let [quantize, skinny, finish] = plan.launches() else {
        panic!("three launches")
    };
    assert_eq!(quantize.kernel.name(), "euhedral_nvfp4n_quantize_rows");
    assert_eq!(
        (quantize.grid, quantize.block, quantize.shared_bytes),
        ([12, 1, 1], [128, 1, 1], 0)
    );
    assert_eq!(quantize.args.words(), [INPUT, SCRATCH, 12, 5120]);
    assert_eq!(quantize.args.sizes(), [8, 8, 4, 4]);
    assert_eq!(skinny.kernel.name(), "euhedral_nvfp4n_skinny_16");
    assert_eq!(
        (skinny.grid, skinny.block, skinny.shared_bytes),
        ([80, 2, 1], [128, 1, 1], 61440)
    );
    let partials = SCRATCH + 143_616;
    assert_eq!(
        skinny.args.words(),
        [SCRATCH, WEIGHTS, OUTPUT, partials, 12, 5120, 5120]
    );
    assert_eq!(skinny.args.sizes(), [8, 8, 8, 8, 4, 4, 4]);
    // The finish: 12 * 5120 = 61440 values, 240 blocks of 256.
    assert_eq!(finish.kernel.name(), "euhedral_nvfp4n_skinny_finish");
    assert_eq!(
        (finish.grid, finish.block, finish.shared_bytes),
        ([240, 1, 1], [256, 1, 1], 0)
    );
    assert_eq!(finish.args.words(), [partials, OUTPUT, 61440, 2]);
    assert_eq!(finish.args.sizes(), [8, 8, 4, 4]);
    assert!(plan.launches().iter().all(|launch| !launch.programmatic));
}

#[test]
fn native_skinny_without_a_split_has_no_finish_and_null_partials() {
    // 256 inputs: k_tiles / 2 = 0, one split; 4096 outputs: 64 blocks.
    let plan = native(16, 256, 4096, Nvfp4Layout::Plain);
    let [quantize, skinny] = plan.launches() else {
        panic!("two launches")
    };
    assert_eq!(quantize.grid, [16, 1, 1]);
    assert_eq!(skinny.grid, [64, 1, 1]);
    assert_eq!(skinny.args.words()[3], 0);
    // The scratch it needs is the activations alone: k = 256, 32 planes: 4096 + 4096 = 8192, plus 64 = 8256.
    assert_eq!(nvfp4_native_scratch_bytes(16, 256, 4096), 8256);
    let operands = operands(16, 256, 4096, Nvfp4Layout::Plain);
    assert!(plan_native(&operands, at(SCRATCH), 8256, false).is_ok());
    assert_eq!(
        plan_native(&operands, at(SCRATCH), 8255, false).unwrap_err(),
        Nvfp4Refusal::InvalidArgument
    );
}

#[test]
fn native_skinny_kernel_by_rows() {
    // up to 16 rows the 16-row kernel, up to 32 the 32-row, then the 64-row (shared 61440, 81920, 92160); from
    // 65 rows the tile kernel (64 rows at most).
    for (rows, kernel, shared) in [
        (9, "euhedral_nvfp4n_skinny_16", 61440),
        (16, "euhedral_nvfp4n_skinny_16", 61440),
        (17, "euhedral_nvfp4n_skinny_32", 81920),
        (32, "euhedral_nvfp4n_skinny_32", 81920),
        (33, "euhedral_nvfp4n_skinny_64", 92160),
        (64, "euhedral_nvfp4n_skinny_64", 92160),
    ] {
        let plan = native(rows, 1024, 4096, Nvfp4Layout::Plain);
        assert_eq!(plan.launches()[1].kernel.name(), kernel, "rows {rows}");
        assert_eq!(plan.launches()[1].shared_bytes, shared, "rows {rows}");
        assert_eq!(plan.tensor_maps(), None);
    }
    let plan = native(65, 1024, 4096, Nvfp4Layout::Plain);
    assert_eq!(plan.launches()[1].kernel.name(), "euhedral_nvfp4n_linear_128x128");
    let sd4 = native(17, 1024, 4096, Nvfp4Layout::Sd4);
    assert_eq!(sd4.launches()[1].kernel.name(), "euhedral_nvfp4n_skinny_32_sd4");
}

#[test]
fn native_scratch_bytes() {
    // 16 rows, 1024 by 4096: k = 1024, 32 planes: 16384 of codes, 8 * 2 * 128 * 8 = 16384 of scales, 64 of globals:
    // 32832. Two splits (64 tiles: 3, cap 4 / 2 = 2), so the partials follow at the next 256 (33024) and take
    // 4 * 2 * 16 * 4096 = 524288 bytes.
    assert_eq!(nvfp4_native_scratch_bytes(16, 1024, 4096), 33_024 + 524_288);
    // 65 rows is the tile route: k = 512, 130 planes: 33280 of codes, 4 * 2 * 128 * 8 = 8192 of scales, 260 of globals.
    assert_eq!(nvfp4_native_scratch_bytes(65, 512, 256), 41_732);
}

#[test]
fn native_tile_plan_and_its_tensor_maps() {
    // 65 rows, 512 by 256 outputs, plain. Grid ceil(65 / 128) * ceil(256 / 128) = 2, 288 threads, 85040 shared.
    let plan = native(65, 512, 256, Nvfp4Layout::Plain);
    let [quantize, tile] = plan.launches() else {
        panic!("two launches")
    };
    assert_eq!((quantize.grid, quantize.block), ([65, 1, 1], [128, 1, 1]));
    assert_eq!(tile.kernel.name(), "euhedral_nvfp4n_linear_128x128");
    assert_eq!(
        (tile.grid, tile.block, tile.shared_bytes),
        ([2, 1, 1], [288, 1, 1], 85040)
    );
    // The maps are not among the arguments: scratch, weights, output, then the three extents.
    assert_eq!(tile.args.words(), [SCRATCH, WEIGHTS, OUTPUT, 65, 512, 256]);
    assert_eq!(tile.args.sizes(), [8, 8, 8, 4, 4, 4]);
    // row_bytes = 256, row_scales = 512 / 16 = 32, scale_offset = align256(256 * 256) = 65536.
    let [a, b, scales] = plan.tensor_maps().unwrap();
    assert_eq!(a.address, SCRATCH);
    assert_eq!(
        (a.dims(), a.strides(), a.box_dims()),
        (&[256, 65, 2][..], &[256, 16640][..], &[64, 128, 1][..])
    );
    assert_eq!(a.swizzle, TensorMapSwizzle::B64);
    assert_eq!(b.address, WEIGHTS);
    assert_eq!(
        (b.dims(), b.strides(), b.box_dims()),
        (&[256, 256][..], &[256][..], &[64, 128][..])
    );
    assert_eq!(b.swizzle, TensorMapSwizzle::B64);
    assert_eq!(scales.address, WEIGHTS + 65536);
    assert_eq!(
        (scales.dims(), scales.strides(), scales.box_dims()),
        (&[32, 256][..], &[32][..], &[16, 128][..])
    );
    assert_eq!(scales.swizzle, TensorMapSwizzle::None);
    // SD4 scale rows are 512 / 32 = 16 bytes.
    let sd4 = native(129, 512, 256, Nvfp4Layout::Sd4);
    assert_eq!(sd4.launches()[1].kernel.name(), "euhedral_nvfp4n_linear_128x128_sd4");
    assert_eq!(sd4.launches()[1].grid, [4, 1, 1]);
    assert_eq!(sd4.tensor_maps().unwrap()[2].dims(), &[16, 256]);
}

#[test]
fn native_declines_and_refuses() {
    let tile = operands(65, 512, 256, Nvfp4Layout::Plain);
    let scratch = |size: u64| plan_native(&tile, at(SCRATCH), size, false);
    // The tile kernel's scratch is the activations alone: 41732 bytes.
    assert!(scratch(41_732).is_ok());
    assert_eq!(scratch(41_731).unwrap_err(), Nvfp4Refusal::InvalidArgument);
    assert_eq!(
        plan_native(&tile, DevicePtr::NULL, u64::MAX, false).unwrap_err(),
        Nvfp4Refusal::InvalidArgument
    );
    // A scratch off 16 bytes declines a tile launch but not a skinny one.
    assert_eq!(
        plan_native(&tile, at(SCRATCH + 8), u64::MAX, false).unwrap_err(),
        Nvfp4Refusal::RouteUnavailable
    );
    let skinny = operands(16, 256, 4096, Nvfp4Layout::Plain);
    assert!(plan_native(&skinny, at(SCRATCH + 8), u64::MAX, false).is_ok());
    // Exact numerics, a width that is not a multiple of 128.
    assert_eq!(
        plan_native(&tile, at(SCRATCH), u64::MAX, true).unwrap_err(),
        Nvfp4Refusal::RouteUnavailable
    );
    let odd = operands(16, 192, 4096, Nvfp4Layout::Plain);
    assert_eq!(
        plan_native(&odd, at(SCRATCH), u64::MAX, false).unwrap_err(),
        Nvfp4Refusal::RouteUnavailable
    );
    // The tile kernels need 256 inputs (plain) or 512 (SD4): 128 inputs with 65 rows, or SD4 with 256, decline.
    for (in_features, layout) in [(128, Nvfp4Layout::Plain), (256, Nvfp4Layout::Sd4)] {
        let shape = operands(65, in_features, 256, layout);
        assert_eq!(
            plan_native(&shape, at(SCRATCH), u64::MAX, false).unwrap_err(),
            Nvfp4Refusal::RouteUnavailable
        );
    }
    // 256 inputs, plain, 65 rows: accepted.
    assert!(
        plan_native(
            &operands(65, 256, 256, Nvfp4Layout::Plain),
            at(SCRATCH),
            u64::MAX,
            false
        )
        .is_ok()
    );
    // Format and null checks come from `prepare`.
    let mut wrong = tile;
    wrong.weights_byte_size += 1;
    assert_eq!(
        plan_native(&wrong, at(SCRATCH), u64::MAX, false).unwrap_err(),
        Nvfp4Refusal::FormatMismatch
    );
    // 2^31 rows over 65536 outputs: 2^24 * 2^9 tiles = 2^33 blocks.
    let huge = operands(1 << 31, 512, 1 << 16, Nvfp4Layout::Plain);
    assert_eq!(
        plan_native(&huge, at(SCRATCH), u64::MAX, false).unwrap_err(),
        Nvfp4Refusal::SizeOverflow
    );
}

#[test]
fn route_of_the_engine_policy() {
    let route = |rows, row_exact, exact| route(rows, 5120, 5120, false, row_exact, exact, true);
    assert_eq!(route(8, false, false), Nvfp4Route::DecodeRows { rows: 8 });
    assert_eq!(route(9, false, false), Nvfp4Route::Native { skinny: true });
    assert_eq!(route(64, false, false), Nvfp4Route::Native { skinny: true });
    assert_eq!(route(65, false, false), Nvfp4Route::Native { skinny: false });
    assert_eq!(route(9, true, false), Nvfp4Route::DecodeGroups { groups: 2 });
    assert_eq!(route(32, true, false), Nvfp4Route::DecodeGroups { groups: 4 });
    assert_eq!(route(33, true, false), Nvfp4Route::Reference);
    assert_eq!(route(1, false, true), Nvfp4Route::Reference);
    assert_eq!(
        super::route(9, 5120, 5120, false, false, false, false),
        Nvfp4Route::DecodeGroups { groups: 2 }
    );
}

#[test]
fn lines_of_the_recorder() {
    assert_eq!(
        Nvfp4Route::Native { skinny: true }.lines(12, 5120, 5120, false),
        [
            "euhedral_nvfp4n_quantize_rows grid=12,1,1 block=128,1,1 smem=0",
            "euhedral_nvfp4n_skinny_16 grid=80,2,1 block=128,1,1 smem=61440",
            "euhedral_nvfp4n_skinny_finish grid=240,1,1 block=256,1,1 smem=0",
        ]
    );
}

#[test]
fn a_skinny_shape_with_one_split_is_planned_whatever_its_product() {
    // 64 rows over 70 million outputs: rows * out = 4.48e9 does not fit 32 bits. With 256 inputs the skinny kernel has
    // one split (k_tiles / 2 = 0), and `launch_native` computes the finish count only past that return, so C plans
    // the quantize and the skinny launch and nothing else.
    let plan = native(64, 256, 70_000_000, Nvfp4Layout::Plain);
    let [quantize, skinny] = plan.launches() else {
        panic!("two launches")
    };
    assert_eq!(quantize.grid, [64, 1, 1]);
    assert_eq!(skinny.kernel.name(), "euhedral_nvfp4n_skinny_64");
    // ceil(70e6 / 64) = 1093750 blocks, one split.
    assert_eq!(skinny.grid, [1_093_750, 1, 1]);
    assert_eq!(skinny.args.words()[3], 0);
}

#[test]
fn plans_compare_by_the_launches_in_use() {
    let one = linear(1, 128, 16);
    assert_eq!(one, linear(1, 128, 16));
    assert_ne!(one, linear(2, 128, 16));
    // A plan of two launches is not equal to a plan of one whose filler slots repeat the first launch.
    let two = native(16, 256, 4096, Nvfp4Layout::Plain);
    let first = Nvfp4Plan::single(two.launches()[0]);
    assert_ne!(two, first);
}
