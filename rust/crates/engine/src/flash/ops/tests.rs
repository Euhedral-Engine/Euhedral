use super::*;

fn at(address: u64) -> DevicePtr {
    // SAFETY: the tests never dereference it.
    unsafe { DevicePtr::from_raw(address) }
}

const A: u64 = 0x1000;
const B: u64 = 0x2000;
const C: u64 = 0x3000;
const D: u64 = 0x4000;
const E: u64 = 0x5000;
const F: u64 = 0x6000;

fn linear(rows: u32, k: u32, n: u32, exact: bool) -> Launch {
    linear_bf16(at(A), at(B), at(C), rows, k, n, exact).unwrap()
}

#[test]
fn a_decode_linear_splits_k_by_the_shape_alone() {
    // Narrow output or long K: eight slices; mid K and a mid output: four; else one warp per column.
    assert_eq!(
        (
            slices(1024, 32),
            slices(8192, 4096),
            slices(2048, 2048),
            slices(1024, 1024)
        ),
        (8, 8, 4, 1)
    );
    let launch = linear(4, 4096, 32, false);
    assert_eq!(launch.symbol, "euhedral_q4_linear_split8_bf16");
    // ceil(n / (8 / slices)) columns, ceil(rows / 8) row blocks.
    assert_eq!((launch.grid, launch.block), ([32, 1, 1], [256, 1, 1]));
    let launch = linear(8, 2048, 2048, false);
    assert_eq!(launch.symbol, "euhedral_q4_linear_split4_bf16");
    assert_eq!(launch.grid, [1024, 1, 1]);
    let launch = linear(1, 1024, 1024, false);
    assert_eq!(launch.symbol, "euhedral_q4_linear_bf16");
    assert_eq!(launch.grid, [128, 1, 1]);
}

#[test]
fn exact_numerics_keep_one_warp_per_column_at_every_row_count() {
    for rows in [1, 8, 9, 40] {
        let launch = linear(rows, 4096, 32, true);
        assert_eq!(launch.symbol, "euhedral_q4_linear_bf16", "{rows} rows");
        assert_eq!(launch.grid, [4, rows.div_ceil(8), 1]);
    }
}

#[test]
fn a_prefill_linear_takes_the_tensor_core_kernel_the_shape_selects() {
    // (rows, n) -> kernel: split below 1536 columns or a width that is not a multiple of 32; four row tiles above 16.
    let cases = [
        (9, 4096, "euhedral_q4_linear_tc_bf16", [128, 1, 1]),
        (16, 1536, "euhedral_q4_linear_tc_split_bf16", [192, 1, 1]),
        (17, 4096, "euhedral_q4_linear_tc_rows_bf16", [128, 1, 1]),
        (65, 1000, "euhedral_q4_linear_tc_rows_split_bf16", [125, 2, 1]),
    ];
    for (rows, n, symbol, grid) in cases {
        let launch = linear(rows, 4096, n, false);
        assert_eq!(launch.symbol, symbol, "{rows} rows, {n} columns");
        let tiles = if rows > 16 { 64 } else { 16 };
        assert_eq!(launch.grid, [grid[0], rows.div_ceil(tiles), 1]);
        assert_eq!(launch.block, [128, 1, 1]);
    }
}

#[test]
fn a_linear_the_tensor_cores_cannot_take_falls_back_to_the_warp_kernel() {
    // K not a multiple of 32, N not a multiple of 8, or an output not 4-byte aligned.
    for (k, n, output) in [(40, 64, C), (64, 12, C), (64, 64, C + 2)] {
        let launch = linear_bf16(at(A), at(B), at(output), 20, k, n, false).unwrap();
        assert_eq!(
            launch.symbol, "euhedral_q4_linear_bf16",
            "k {k} n {n} output {output:#x}"
        );
    }
}

#[test]
fn a_linear_refuses_what_its_kernels_cannot_read() {
    let refused = |result: Result<Launch, FlashError>| result.unwrap_err().to_string();
    assert!(refused(linear_bf16(at(A), at(B), at(C), 0, 64, 64, false)).contains("rows must be positive"));
    assert!(refused(linear_bf16(at(A), at(B), at(C), 1, 12, 64, false)).contains("divisible by 8, got 12"));
    assert!(refused(linear_bf16(at(A + 8), at(B), at(C), 1, 64, 64, false)).contains("input 0x1008 is not 16-byte"));
    assert!(refused(linear_bf16(at(A), DevicePtr::NULL, at(C), 1, 64, 64, false)).contains("weights 0x0"));
}

#[test]
fn launches_carry_javas_arguments_in_javas_order() {
    let line = |launch: Result<Launch, FlashError>| launch.unwrap().line();
    assert_eq!(
        line(grouped_rms_norm(at(A), at(B), at(C), 3, 4, 128, 1.0)),
        "euhedral_q4_grouped_rms_norm_bf16 grid=12,1,1 block=256,1,1 smem=0 args=p1000,p2000,p3000,i3,i4,i128,i1065353216"
    );
    assert_eq!(
        line(scaled_silu(at(A), at(B), 1000, 4.0)),
        "euhedral_q4_scaled_silu_bf16 grid=4,1,1 block=256,1,1 smem=0 args=p1000,p2000,i1000,i1082130432"
    );
    assert_eq!(
        line(hc_mix(at(A), at(B), at(C), 5, 4, 600)),
        "euhedral_q4_hc_mix_bf16 grid=3,5,1 block=256,1,1 smem=0 args=p1000,p2000,p3000,i5,i4,i600"
    );
    assert_eq!(
        line(hc_inject(at(A), at(B), at(C), at(D), 5, 4, 256)),
        "euhedral_q4_hc_inject_bf16 grid=1,5,1 block=256,1,1 smem=0 args=p1000,p2000,p3000,p4000,i5,i4,i256"
    );
    assert_eq!(
        line(repeat_streams(at(A), at(B), 3, 4, 100)),
        "euhedral_q4_repeat_streams_bf16 grid=5,1,1 block=256,1,1 smem=0 args=p1000,p2000,i3,i4,i100"
    );
    assert_eq!(
        line(ngram_expand(at(A), at(B), 24, 64, 36)),
        "euhedral_q4_ngram_expand_bf16 grid=24,1,1 block=256,1,1 smem=0 args=p1000,p2000,i24,i64,i36"
    );
    assert_eq!(
        line(ple_gate(at(A), at(B), at(C), at(D), 3, 4, 128)),
        "euhedral_q4_ple_gate_bf16 grid=12,1,1 block=256,1,1 smem=0 args=p1000,p2000,p3000,p4000,i3,i4,i128"
    );
    assert_eq!(
        line(ple_conv(at(A), at(B), at(C), at(D), at(E), 3, 512, 4, 2)),
        "euhedral_q4_ple_conv_bf16 grid=2,3,1 block=256,1,1 smem=0 args=p1000,p2000,p3000,p4000,p5000,i3,i512,i4,i2"
    );
    assert_eq!(
        line(conv_history(at(A), at(B), 3, 512, 6)),
        "euhedral_q4_conv_history_bf16 grid=2,1,1 block=256,1,1 smem=0 args=p1000,p2000,i3,i512,i6"
    );
    assert_eq!(
        line(gdn_conv(at(A), at(B), at(C), at(D), 3, 512, 4)),
        "euhedral_q4_gdn_conv_bf16 grid=2,3,1 block=256,1,1 smem=0 args=p1000,p2000,p3000,p4000,i3,i512,i4"
    );
    assert_eq!(
        line(gdn_control(at(A), at(B), at(C), at(D), at(E), at(F), 3, 32)),
        "euhedral_q4_gdn_control_bf16 grid=1,1,1 block=256,1,1 smem=0 args=p1000,p2000,p3000,p4000,p5000,p6000,i3,i32"
    );
    assert_eq!(
        line(gdn_gated_norm(at(A), at(B), at(C), at(D), 3, 32, 128, 1.0, 1)),
        "euhedral_q4_gdn_gated_norm_bf16 grid=96,1,1 block=128,1,1 smem=0 args=p1000,p2000,p3000,p4000,i3,i32,i128,i1065353216,i1"
    );
    assert_eq!(
        line(embedding(at(A), at(B), at(C), 7, 64, 1000)),
        "euhedral_q4_embedding_bf16 grid=7,1,1 block=256,1,1 smem=0 args=p1000,p2000,p3000,i7,i64,i1000"
    );
}

#[test]
fn every_launch_is_one_the_catalog_plans() {
    let launches = [
        grouped_rms_norm(at(A), at(B), at(C), 3, 4, 128, 1e-6),
        scaled_silu(at(A), at(B), 1000, 4.0),
        hc_mix(at(A), at(B), at(C), 5, 4, 600),
        hc_inject(at(A), at(B), at(C), at(D), 5, 4, 256),
        repeat_streams(at(A), at(B), 3, 4, 100),
        ngram_expand(at(A), at(B), 24, 64, 36),
        ple_gate(at(A), at(B), at(C), at(D), 3, 4, 128),
        ple_conv(at(A), at(B), at(C), at(D), at(E), 3, 512, 4, 2),
        conv_history(at(A), at(B), 3, 512, 6),
        gdn_conv(at(A), at(B), at(C), at(D), 3, 512, 4),
        gdn_control(at(A), at(B), at(C), at(D), at(E), at(F), 3, 32),
        gdn_gated_norm(at(A), at(B), at(C), at(D), 3, 32, 128, 1e-6, 0),
        embedding(at(A), at(B), at(C), 7, 64, 1000),
        linear_bf16(at(A), at(B), at(C), 20, 64, 64, false),
    ];
    for launch in launches {
        let launch = launch.unwrap();
        let planned = launch
            .planned()
            .unwrap_or_else(|error| panic!("{}: {error:?}", launch.symbol));
        assert_eq!(planned.kernel.name(), launch.symbol);
    }
}

#[test]
fn the_other_ops_refuse_a_zero_extent_and_a_value_their_kernel_cannot_take() {
    let refused = |result: Result<Launch, FlashError>| result.unwrap_err().to_string();
    assert!(refused(grouped_rms_norm(at(A), at(B), at(C), 3, 0, 128, 1.0)).contains("groups must be positive"));
    assert!(refused(scaled_silu(at(A), at(B), 0, 4.0)).contains("count must be positive"));
    assert!(refused(ngram_expand(at(A), at(B), 4, 64, 34)).contains("multiples of four bytes, got 34"));
    assert!(refused(conv_history(at(A), at(B), 3, 512, 33)).contains("at most 32 rows, got 33"));
    assert!(refused(embedding(at(A), at(B), at(C), 7, 60, 1000)).contains("multiples of 8 values, got 60"));
    assert!(refused(embedding(at(A + 4), at(B), at(C), 7, 64, 1000)).contains("table 0x1004 is not 16-byte"));
    assert!(refused(repeat_streams(at(A), at(B), 0, 4, 100)).contains("rows must be positive"));
}
