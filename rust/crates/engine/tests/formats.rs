//! The linear formats' routes, decided on the host: the K split, the trellis launches over the RCO artifact's layer
//! shapes, row counts and thresholds, and the NVFP4 routes by rows. None of it needs a GPU.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

mod formats {
    use std::sync::Arc;

    use euhedral_inference_cuda::DevicePtr;
    use euhedral_inference_cuda::launch::nvfp4::skinny_target_ctas;
    use euhedral_inference_edrl::{TensorDataType, WeightFormat, WeightLayout};
    use euhedral_inference_engine::formats::{
        Launch, LinearMode, Nvfp4Linear, Nvfp4Route, ScratchRegion, TrellisLinear, TrellisOperands, TrellisShape,
        nvfp4_native_scratch_bytes,
    };
    /// The grid the portable linears assume: two blocks on each of 70 multiprocessors.
    const TARGET_CTAS: u64 = skinny_target_ctas(70);

    use euhedral_inference_engine::model::{Residence, TrellisParts, WeightRef};
    use euhedral_inference_engine::residency::scratch::splits;
    use euhedral_inference_settings::{Auto, TrellisKernelSettings};

    /// A device address that is never dereferenced.
    fn address(value: u64) -> DevicePtr {
        // SAFETY: the address is only compared and printed; no operation is queued on it.
        unsafe { DevicePtr::from_raw(value) }
    }

    /// The K split is a count of pieces of at least 256 inputs each, as many as it takes the thread blocks to reach
    /// the target (the columns already give one block per 128), never more, and one when nothing can be split.
    #[test]
    fn the_k_split_fills_the_target_with_the_fewest_pieces_of_whole_tiles() {
        let sizes = [
            128u64, 255, 256, 384, 512, 1024, 4096, 5120, 12_288, 17_408, 34_816, 248_320,
        ];
        let targets = [1u64, 2, 3, 70, 139, 140, 280, 384, 560, 1120, 100_000];
        for k in sizes {
            for n in sizes {
                let blocks = (n / 128).max(1);
                let most = (k / 256).max(1);
                let mut previous = 0;
                for target in targets {
                    let count = splits(k, n, target);
                    let at = format!("k={k} n={n} target={target}");
                    assert!((1..=most).contains(&count), "{at}: {count} pieces outside 1..={most}");
                    assert!(
                        count * blocks >= target || count == most,
                        "{at}: {count} pieces leave the target unmet below the limit"
                    );
                    assert!(
                        count == 1 || (count - 1) * blocks < target,
                        "{at}: {count} pieces are more than the target needs"
                    );
                    if count > 1 {
                        assert!(k / count >= 256, "{at}: a piece is smaller than a tile of 256 inputs");
                    }
                    assert!(count >= previous, "{at}: a larger target gives fewer pieces");
                    previous = count;
                }
                // More columns carry more blocks, so they never ask for more pieces.
                for target in targets {
                    assert!(
                        splits(k, n * 2, target) <= splits(k, n, target),
                        "k={k} n={n} target={target}"
                    );
                }
            }
        }
    }

    /// One trellis linear of the RCO artifact's layer shapes at a row count, under one set of kernel thresholds.
    struct Case {
        name: &'static str,
        n: u64,
        k: u64,
        w: u64,
        ends: Vec<u32>,
        rows: usize,
        target: usize,
        mma: usize,
        whole_k: usize,
        wide: usize,
        /// Bytes of the scratch region the plan is given.
        scratch: u64,
    }

    /// The layer shapes of the RCO artifact: name, columns, inputs, tile words and segment ends.
    const LAYERS: [(&str, u64, u64, u64, &[u32]); 9] = [
        ("attention/gate_value", 7168, 5120, 64, &[6144, 7168]),
        ("attention/output", 5120, 6144, 40, &[5120]),
        ("attention/query_key", 7168, 5120, 64, &[6144, 7168]),
        ("gdn/output", 5120, 6144, 40, &[5120]),
        ("gdn/query_key", 4096, 5120, 40, &[4096]),
        ("gdn/value_z", 12288, 5120, 40, &[6144, 12288]),
        ("mlp/down", 5120, 17408, 40, &[5120]),
        ("mlp/down-w56", 5120, 17408, 56, &[5120]),
        ("mlp/gate_up", 34816, 5120, 40, &[17408, 34816]),
    ];

    /// The kernel thresholds the engine runs with: target blocks, tensor-core row limit, whole-K rows and wide rows.
    const THRESHOLDS: [(usize, usize, usize, usize); 4] = [
        (560, 512, 64, 128),
        (384, 512, 64, 128),
        (1120, 512, 16, 32),
        (560, 16, 32, 64),
    ];

    /// Row counts either side of every threshold and of the tile and skinny limits.
    const ROWS: [usize; 23] = [
        1, 2, 3, 4, 5, 6, 8, 9, 16, 17, 20, 31, 32, 33, 63, 64, 65, 100, 127, 128, 512, 513, 1024,
    ];

    fn cases() -> Vec<Case> {
        let mut cases = Vec::new();
        for (name, n, k, w, ends) in LAYERS {
            for (target, mma, whole_k, wide) in THRESHOLDS {
                for rows in ROWS {
                    let ends = ends.to_vec();
                    let shape = TrellisShape {
                        n,
                        k,
                        w,
                        segments: ends.len() as u64,
                        ends: &ends,
                    };
                    let scratch = TrellisLinear::new(target, mma, whole_k, wide).scratch_for(&shape, rows as u64, true);
                    cases.push(Case {
                        name,
                        n,
                        k,
                        w,
                        ends,
                        rows,
                        target,
                        mma,
                        whole_k,
                        wide,
                        scratch,
                    });
                }
            }
        }
        cases
    }

    const WORDS: u64 = 0x2_0000_0000;
    const SUH: u64 = 0x3_0000_0000;
    const SVH: u64 = 0x4_0000_0000;
    const ENDS: u64 = 0x5_0000_0000;
    const INPUT: u64 = 0x1_0000_0000;
    const OUTPUT: u64 = 0x6_0000_0000;
    const SCRATCH: u64 = 0x7_0000_0000;

    fn plan_lines(case: &Case, format: &TrellisLinear, fp8: bool, row_exact: bool) -> Vec<String> {
        let shape = TrellisShape {
            n: case.n,
            k: case.k,
            w: case.w,
            segments: case.ends.len() as u64,
            ends: &case.ends,
        };
        let operands = TrellisOperands {
            input: address(INPUT),
            output: address(OUTPUT),
            words: address(WORDS),
            suh: address(SUH),
            svh: address(SVH),
            ends: address(ENDS),
            scratch: ScratchRegion {
                ptr: address(SCRATCH),
                bytes: case.scratch as usize,
            },
        };
        let mode = LinearMode {
            row_exact,
            prefill: true,
        };
        let mut lines = Vec::new();
        format
            .plan(&shape, &operands, case.rows, mode, fp8, &mut |launch: Launch| {
                lines.push(launch.line());
                Ok(())
            })
            .unwrap();
        lines
    }

    fn format_of(case: &Case) -> TrellisLinear {
        TrellisLinear::new(case.target, case.mma, case.whole_k, case.wide)
    }

    #[test]
    fn verify_rows_never_use_fp8() {
        for case in cases().iter() {
            for row_exact in [true, false] {
                let format = format_of(case);
                let lines = plan_lines(case, &format, true, row_exact);
                let fp8 = lines.iter().any(|line| {
                    line.contains("_fp8") || (line.starts_with("euhedral_trellis_prepass") && line.ends_with(",i2"))
                });
                if row_exact {
                    assert!(!fp8, "{} rows={} takes FP8 in a verification", case.name, case.rows);
                } else if case.rows >= case.whole_k && case.rows <= case.mma {
                    assert!(
                        fp8,
                        "{} rows={} should take FP8 on a device that has it",
                        case.name, case.rows
                    );
                }
                // Without the device's FP8 kernels no launch takes them whatever the mode.
                let without = plan_lines(case, &format, false, row_exact);
                assert!(without.iter().all(|line| !line.contains("_fp8")), "{}", case.name);
            }
        }
    }

    #[test]
    fn a_verification_beyond_the_tensor_core_limit_stays_on_the_tensor_core_route() {
        let format = TrellisLinear::new(560, 16, 64, 128);
        let case = cases().into_iter().find(|case| case.name == "mlp/down").unwrap();
        let big = Case {
            rows: 100,
            scratch: 1 << 30,
            ..case
        };
        let verification = plan_lines(&big, &format, true, true);
        assert!(verification.iter().all(|line| !line.contains("decode_bf16")));
        assert!(
            verification
                .iter()
                .any(|line| line.starts_with("euhedral_trellis_mma_rows "))
        );
        let prefill = plan_lines(&big, &format, true, false);
        assert!(
            prefill
                .iter()
                .any(|line| line.starts_with("euhedral_trellis_decode_bf16"))
        );
    }

    #[test]
    fn the_target_blocks_come_from_the_settings_and_auto_is_eight_per_multiprocessor() {
        let settings = TrellisKernelSettings::default();
        assert_eq!(settings.target_blocks, Auto::Auto);
        let shape_ends = [4096u32];
        let shape = TrellisShape {
            n: 4096,
            k: 5120,
            w: 40,
            segments: 1,
            ends: &shape_ends,
        };
        let operands = TrellisOperands {
            input: address(INPUT),
            output: address(OUTPUT),
            words: address(WORDS),
            suh: address(SUH),
            svh: address(SVH),
            ends: address(ENDS),
            scratch: ScratchRegion {
                ptr: address(SCRATCH),
                bytes: 1 << 24,
            },
        };
        let splits_of = |format: &TrellisLinear| {
            let mut mma = String::new();
            format
                .plan(
                    &shape,
                    &operands,
                    1,
                    LinearMode::default(),
                    false,
                    &mut |launch: Launch| {
                        if launch.symbol == "euhedral_trellis_mma" {
                            mma = launch.line();
                        }
                        Ok(())
                    },
                )
                .unwrap();
            mma
        };
        // 70 multiprocessors aim for 560 blocks, 48 for 384, and a stated value is taken as it is.
        let seventy = splits_of(&TrellisLinear::from_settings(&settings, 70));
        assert_eq!(seventy, splits_of(&TrellisLinear::new(560, 512, 64, 128)));
        assert!(seventy.contains("grid=32,18,1"), "{seventy}");
        let forty_eight = splits_of(&TrellisLinear::from_settings(&settings, 48));
        assert!(forty_eight.contains("grid=32,12,1"), "{forty_eight}");
        let stated = TrellisKernelSettings {
            target_blocks: Auto::Value(140),
            ..settings
        };
        assert!(splits_of(&TrellisLinear::from_settings(&stated, 70)).contains("grid=32,5,1"));
    }

    fn nvfp4(out: u64, input: u64, layout: WeightLayout) -> WeightRef {
        WeightRef {
            name: "w".to_string(),
            shape: vec![out, input],
            dtype: TensorDataType::Bf16,
            format: WeightFormat::Nvfp4,
            layout,
            ptr: address(0x1000),
            bytes: 1 << 20,
            residence: Residence::Device,
            trellis: None,
        }
    }

    fn route_of(format: &Nvfp4Linear, weight: &WeightRef, rows: usize, row_exact: bool) -> Nvfp4Route {
        format.route_of(weight, rows, row_exact, false).unwrap()
    }

    #[test]
    fn nvfp4_routes_by_rows() {
        let weight = nvfp4(34816, 5120, WeightLayout::RowSplitK128V1);
        let native = Nvfp4Linear::with_native_route(true, 70);
        let portable = Nvfp4Linear::with_native_route(false, 70);
        // 1 to 8 rows run the decode kernels, one kernel per row count, with or without the native route.
        for rows in 1..=8usize {
            let expected = Nvfp4Route::DecodeRows { rows: rows as u32 };
            assert_eq!(route_of(&native, &weight, rows, false), expected);
            assert_eq!(route_of(&native, &weight, rows, true), expected);
            assert_eq!(route_of(&portable, &weight, rows, false), expected);
        }
        // 9 to 32 rows of a verification, or where there is no native route, run the grouped twins.
        for rows in 9..=32usize {
            let expected = Nvfp4Route::DecodeGroups {
                groups: rows.div_ceil(8) as u32,
            };
            assert_eq!(route_of(&native, &weight, rows, true), expected, "rows {rows}");
            assert_eq!(route_of(&portable, &weight, rows, false), expected, "rows {rows}");
        }
        // From 9 rows the native route takes every other launch: skinny to 64 rows, tiles beyond.
        for rows in 9..=64usize {
            assert_eq!(
                route_of(&native, &weight, rows, false),
                Nvfp4Route::Native { skinny: true },
                "rows {rows}"
            );
        }
        for rows in [65usize, 100, 512] {
            assert_eq!(
                route_of(&native, &weight, rows, false),
                Nvfp4Route::Native { skinny: false }
            );
        }
        // Past 32 rows a verification, or a device without the native route, falls to the reference.
        for rows in [33usize, 64, 512] {
            assert_eq!(route_of(&native, &weight, rows, true), Nvfp4Route::Reference);
            assert_eq!(route_of(&portable, &weight, rows, false), Nvfp4Route::Reference);
        }
    }

    #[test]
    fn nvfp4_declines_follow_the_native_library() {
        let native = Nvfp4Linear::with_native_route(true, 70);
        // 5248 inputs are a multiple of 128 and not of 256: no skinny kernel, no tile kernel.
        let odd = nvfp4(5120, 5248, WeightLayout::RowSplitK128V1);
        assert_eq!(
            route_of(&native, &odd, 20, false),
            Nvfp4Route::DecodeGroups { groups: 3 }
        );
        assert_eq!(route_of(&native, &odd, 100, false), Nvfp4Route::Reference);
        // The SD4 tile kernel needs whole scale rows of 512 inputs: 6144 is, 5376 is not.
        let sd4_wide = nvfp4(5120, 6144, WeightLayout::RowSplitK128Sd4V1);
        let sd4_narrow = nvfp4(5120, 5376, WeightLayout::RowSplitK128Sd4V1);
        assert_eq!(
            route_of(&native, &sd4_wide, 100, false),
            Nvfp4Route::Native { skinny: false }
        );
        assert_eq!(route_of(&native, &sd4_narrow, 100, false), Nvfp4Route::Reference);
        assert_eq!(
            route_of(&native, &sd4_narrow, 40, false),
            Nvfp4Route::Native { skinny: true }
        );
        // Exact numerics run the reference whatever the rows.
        for rows in [1usize, 8, 20, 100] {
            let weight = nvfp4(5120, 5120, WeightLayout::RowSplitK128V1);
            assert_eq!(
                native.route_of(&weight, rows, false, true).unwrap(),
                Nvfp4Route::Reference
            );
        }
        // A weight that is not a two-dimensional NVFP4 object has no route.
        let mut bf16 = nvfp4(5120, 5120, WeightLayout::RowSplitK128V1);
        bf16.format = WeightFormat::Bf16;
        assert_eq!(native.route_of(&bf16, 1, false, false), None);
    }

    #[test]
    fn nvfp4_routes_list_the_launches_the_library_makes() {
        let lines = |route: Nvfp4Route, rows, k, n, sd4| route.lines(rows, k, n, sd4, TARGET_CTAS);
        // The lm head: 5120 inputs take the 16-row tile, 4 rows are the fourth decode kernel.
        assert_eq!(
            lines(Nvfp4Route::DecodeRows { rows: 4 }, 4, 5120, 248320, false),
            ["euhedral_nvfp4_decode_rows4_w8 grid=15520,1,1 block=256,1,1 smem=0"]
        );
        // Above 5120 inputs, and an output count of whole 32s, the 32-row tile with 4 warps.
        assert_eq!(
            lines(Nvfp4Route::DecodeRows { rows: 1 }, 1, 17408, 5120, true),
            ["euhedral_nvfp4_decode_rows1_w4_sd4 grid=160,1,1 block=128,1,1 smem=0"]
        );
        assert_eq!(
            lines(Nvfp4Route::DecodeGroups { groups: 3 }, 20, 5120, 5120, false),
            ["euhedral_nvfp4_decode_groups3_w8 grid=320,1,1 block=256,1,1 smem=0"]
        );
        // Skinny: 64-column blocks, split over K to reach 140 blocks, 16-row kernel to 16 rows.
        assert_eq!(
            lines(Nvfp4Route::Native { skinny: true }, 12, 5120, 5120, false),
            [
                "euhedral_nvfp4n_quantize_rows grid=12,1,1 block=128,1,1 smem=0",
                "euhedral_nvfp4n_skinny_16 grid=80,2,1 block=128,1,1 smem=61440",
                "euhedral_nvfp4n_skinny_finish grid=240,1,1 block=256,1,1 smem=0",
            ]
        );
        assert_eq!(
            lines(Nvfp4Route::Native { skinny: false }, 300, 6144, 5120, true),
            [
                "euhedral_nvfp4n_quantize_rows grid=300,1,1 block=128,1,1 smem=0",
                "euhedral_nvfp4n_linear_128x128_sd4 grid=120,1,1 block=288,1,1 smem=85040",
            ]
        );
        assert_eq!(
            lines(Nvfp4Route::Reference, 40, 5120, 248320, false),
            ["euhedral_nvfp4_reference grid=9932800,1,1 block=128,1,1 smem=0"]
        );
    }

    #[test]
    fn nvfp4_scratch_is_the_native_routes_and_only_from_nine_rows() {
        let weight = nvfp4(5120, 17408, WeightLayout::RowSplitK128V1);
        let none = LinearMode::default();
        let native = Nvfp4Linear::with_native_route(true, 70);
        let portable = Nvfp4Linear::with_native_route(false, 70);
        use euhedral_inference_engine::formats::LinearFormat;
        for rows in 1..=8usize {
            assert_eq!(native.scratch_bytes(&weight, rows, none), 0);
        }
        // The route's own scratch, or the skinny route's at 64 rows where that is larger (65 rows and more).
        for rows in [9usize, 64, 65, 512] {
            let own = nvfp4_native_scratch_bytes(rows as u64, 17408, 5120, TARGET_CTAS);
            let skinny = nvfp4_native_scratch_bytes((rows as u64).min(64), 17408, 5120, TARGET_CTAS);
            assert_eq!(native.scratch_bytes(&weight, rows, none) as u64, own.max(skinny));
            assert!(native.scratch_bytes(&weight, rows, none) > 0);
            assert_eq!(portable.scratch_bytes(&weight, rows, none), 0);
        }
    }

    #[test]
    fn nvfp4_scratch_covers_every_smaller_launch() {
        use euhedral_inference_engine::formats::LinearFormat;
        // Every NVFP4 projection of the RCO artifact: the heads and the MTP layer.
        let shapes = [
            (248_320u64, 5120u64),
            (131_072, 5120),
            (5120, 10_240),
            (7168, 5120),
            (5120, 6144),
            (34_816, 5120),
            (5120, 17_408),
        ];
        let format = Nvfp4Linear::with_native_route(true, 70);
        let none = LinearMode::default();
        for (out, input) in shapes {
            let weight = nvfp4(out, input, WeightLayout::RowSplitK128V1);
            let mut needed = 0u64;
            for rows in 1..=600usize {
                if rows >= 9 {
                    needed = needed.max(nvfp4_native_scratch_bytes(rows as u64, input, out, TARGET_CTAS));
                }
                let granted = format.scratch_bytes(&weight, rows, none) as u64;
                assert!(
                    granted >= needed,
                    "{out}x{input}: {granted} bytes for {rows} rows, a launch of fewer needs {needed}"
                );
                // And no more than the launches it covers can use.
                assert!(granted <= needed, "{out}x{input}: {granted} bytes for {rows} rows");
            }
        }
        // The skinny kernels' split-K partials make 64 rows need more than 65.
        let weight = nvfp4(5120, 17_408, WeightLayout::RowSplitK128V1);
        assert!(
            nvfp4_native_scratch_bytes(64, 17_408, 5120, TARGET_CTAS)
                > nvfp4_native_scratch_bytes(65, 17_408, 5120, TARGET_CTAS)
        );
        assert_eq!(
            format.scratch_bytes(&weight, 65, none) as u64,
            nvfp4_native_scratch_bytes(64, 17_408, 5120, TARGET_CTAS)
        );
    }

    #[test]
    fn a_trellis_weight_reports_its_scratch_from_its_shape() {
        use euhedral_inference_engine::formats::LinearFormat;
        let weight = WeightRef {
            name: "text/layers/0/mlp/down".to_string(),
            shape: vec![1088, 320, 56],
            dtype: TensorDataType::Bf16,
            format: WeightFormat::Trellis,
            layout: WeightLayout::TrellisTile16V1,
            ptr: address(WORDS),
            bytes: 1088 * 320 * 56 * 2,
            residence: Residence::Device,
            trellis: Some(TrellisParts {
                suh: address(SUH),
                svh: address(SVH),
                segments: address(ENDS),
                segment_ends: Arc::from([5120u32]),
                suh_rows: 1,
            }),
        };
        let format = TrellisLinear::new(560, 512, 64, 128);
        let decode = LinearMode::default();
        let prefill = LinearMode {
            row_exact: false,
            prefill: true,
        };
        // 17408 inputs, 5120 outputs, one segment: the activations of one row, and the split partials.
        let one = format.scratch_bytes(&weight, 1, decode);
        assert!(one > 17408 * 2);
        assert_eq!(format.scratch_bytes(&weight, 1, prefill), one);
        // Above the tensor-core limit only a prefill launch needs the decoded object and the rotated outputs.
        assert!(format.scratch_bytes(&weight, 1024, prefill) > format.scratch_bytes(&weight, 1024, decode));
        let mut nvfp4 = nvfp4(5120, 5120, WeightLayout::RowSplitK128V1);
        nvfp4.trellis = weight.trellis.clone();
        assert_eq!(format.scratch_bytes(&nvfp4, 1, decode), 0, "not a trellis object");
    }
}
