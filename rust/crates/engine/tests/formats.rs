//! The linear formats' routes, decided on the host: the K split, the trellis launches against Java's, and the NVFP4
//! routes by rows. None of it needs a GPU.
//!
//! The K-split table is produced once from Java (`TrellisFormat.splits`, Java 25, from the repository root after
//! `./gradlew :core:compileJava :core:compileTestJava`; CLASSPATH is the core module's test runtime classpath):
//!
//! ```text
//! jshell --class-path "$CLASSPATH" <<'JSH'
//! import io.euhedral_execution.inference.core.artifact.format.TrellisFormat;
//! int[][] shapes = {{5120, 4096}, {5120, 12288}, {6144, 5120}, {5120, 34816}, {17408, 5120}, {5120, 7168},
//!         {5120, 248320}, {256, 128}, {128, 128}, {255, 256}, {512, 128}, {1024, 1024}, {17408, 128}};
//! int[] targets = {1, 140, 280, 384, 560, 840, 1120, 100000};
//! for (int[] shape : shapes)
//!     for (int target : targets)
//!         System.out.println("    (" + shape[0] + ", " + shape[1] + ", " + target + ", "
//!                 + TrellisFormat.splits(shape[0], shape[1], target) + "),");
//! /exit
//! JSH
//! ```
//!
//! The launches are recorded by `fixtures/linear_launches.jsh` into `fixtures/linear_launches.txt`: the kernel, grid,
//! block, shared bytes and arguments of every launch `TrellisFormat.linear` makes for the layer shapes of the RCO
//! artifact over row counts, thresholds and routes, and the scratch it plans for each.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

mod formats {
    use std::sync::Arc;

    use euhedral_inference_cuda::DevicePtr;
    use euhedral_inference_edrl::{TensorDataType, WeightFormat, WeightLayout};
    use euhedral_inference_engine::formats::{
        Launch, LinearKernel, LinearMode, Nvfp4Linear, Nvfp4Route, ROW_EXACT_KERNELS, ScratchRegion, Table,
        TrellisLinear, TrellisOperands, TrellisShape, nvfp4_native_scratch_bytes,
    };
    use euhedral_inference_engine::model::{TrellisParts, WeightRef};
    use euhedral_inference_engine::residency::scratch::splits;
    use euhedral_inference_settings::{Auto, TrellisKernelSettings};

    /// `(k, n, target blocks, splits)`: `TrellisFormat.splits`.
    const JAVA_SPLITS: &[(u64, u64, u64, u64)] = &[
        (5120, 4096, 1, 1),
        (5120, 4096, 140, 5),
        (5120, 4096, 280, 9),
        (5120, 4096, 384, 12),
        (5120, 4096, 560, 18),
        (5120, 4096, 840, 20),
        (5120, 4096, 1120, 20),
        (5120, 4096, 100000, 20),
        (5120, 12288, 1, 1),
        (5120, 12288, 140, 2),
        (5120, 12288, 280, 3),
        (5120, 12288, 384, 4),
        (5120, 12288, 560, 6),
        (5120, 12288, 840, 9),
        (5120, 12288, 1120, 12),
        (5120, 12288, 100000, 20),
        (6144, 5120, 1, 1),
        (6144, 5120, 140, 4),
        (6144, 5120, 280, 7),
        (6144, 5120, 384, 10),
        (6144, 5120, 560, 14),
        (6144, 5120, 840, 21),
        (6144, 5120, 1120, 24),
        (6144, 5120, 100000, 24),
        (5120, 34816, 1, 1),
        (5120, 34816, 140, 1),
        (5120, 34816, 280, 2),
        (5120, 34816, 384, 2),
        (5120, 34816, 560, 3),
        (5120, 34816, 840, 4),
        (5120, 34816, 1120, 5),
        (5120, 34816, 100000, 20),
        (17408, 5120, 1, 1),
        (17408, 5120, 140, 4),
        (17408, 5120, 280, 7),
        (17408, 5120, 384, 10),
        (17408, 5120, 560, 14),
        (17408, 5120, 840, 21),
        (17408, 5120, 1120, 28),
        (17408, 5120, 100000, 68),
        (5120, 7168, 1, 1),
        (5120, 7168, 140, 3),
        (5120, 7168, 280, 5),
        (5120, 7168, 384, 7),
        (5120, 7168, 560, 10),
        (5120, 7168, 840, 15),
        (5120, 7168, 1120, 20),
        (5120, 7168, 100000, 20),
        (5120, 248320, 1, 1),
        (5120, 248320, 140, 1),
        (5120, 248320, 280, 1),
        (5120, 248320, 384, 1),
        (5120, 248320, 560, 1),
        (5120, 248320, 840, 1),
        (5120, 248320, 1120, 1),
        (5120, 248320, 100000, 20),
        (256, 128, 1, 1),
        (256, 128, 140, 1),
        (256, 128, 280, 1),
        (256, 128, 384, 1),
        (256, 128, 560, 1),
        (256, 128, 840, 1),
        (256, 128, 1120, 1),
        (256, 128, 100000, 1),
        (128, 128, 1, 1),
        (128, 128, 140, 1),
        (128, 128, 280, 1),
        (128, 128, 384, 1),
        (128, 128, 560, 1),
        (128, 128, 840, 1),
        (128, 128, 1120, 1),
        (128, 128, 100000, 1),
        (255, 256, 1, 1),
        (255, 256, 140, 1),
        (255, 256, 280, 1),
        (255, 256, 384, 1),
        (255, 256, 560, 1),
        (255, 256, 840, 1),
        (255, 256, 1120, 1),
        (255, 256, 100000, 1),
        (512, 128, 1, 1),
        (512, 128, 140, 2),
        (512, 128, 280, 2),
        (512, 128, 384, 2),
        (512, 128, 560, 2),
        (512, 128, 840, 2),
        (512, 128, 1120, 2),
        (512, 128, 100000, 2),
        (1024, 1024, 1, 1),
        (1024, 1024, 140, 4),
        (1024, 1024, 280, 4),
        (1024, 1024, 384, 4),
        (1024, 1024, 560, 4),
        (1024, 1024, 840, 4),
        (1024, 1024, 1120, 4),
        (1024, 1024, 100000, 4),
        (17408, 128, 1, 1),
        (17408, 128, 140, 68),
        (17408, 128, 280, 68),
        (17408, 128, 384, 68),
        (17408, 128, 560, 68),
        (17408, 128, 840, 68),
        (17408, 128, 1120, 68),
        (17408, 128, 100000, 68),
    ];

    /// A device address that is never dereferenced.
    fn address(value: u64) -> DevicePtr {
        // SAFETY: the address is only compared and printed; no operation is queued on it.
        unsafe { DevicePtr::from_raw(value) }
    }

    #[test]
    fn trellis_k_split_matches_java_table() {
        assert!(JAVA_SPLITS.len() > 100);
        for (k, n, target, java) in JAVA_SPLITS {
            assert_eq!(splits(*k, *n, *target), *java, "k={k} n={n} target={target}");
        }
    }

    /// One case of the fixture: a trellis linear of Java's, and the launches it made.
    struct Case {
        name: String,
        n: u64,
        k: u64,
        w: u64,
        ends: Vec<u32>,
        rows: usize,
        row_exact: bool,
        fp8: bool,
        exact: bool,
        target: usize,
        mma: usize,
        whole_k: usize,
        wide: usize,
        scratch: (u64, u64),
        lines: Vec<String>,
    }

    fn cases() -> Vec<Case> {
        let mut cases: Vec<Case> = Vec::new();
        for line in include_str!("fixtures/linear_launches.txt").lines() {
            if let Some(header) = line.strip_prefix("case ") {
                let mut parts = header.split(' ');
                let name = parts.next().unwrap().to_string();
                let field = |key: &str| -> String {
                    header
                        .split(' ')
                        .find_map(|part| part.strip_prefix(&format!("{key}=")))
                        .unwrap_or_else(|| panic!("no {key} in {header}"))
                        .to_string()
                };
                let number = |key: &str| field(key).parse::<u64>().unwrap();
                let ends = field("ends")
                    .trim_matches(|c| c == '[' || c == ']')
                    .split(',')
                    .map(|end| end.parse().unwrap())
                    .collect();
                cases.push(Case {
                    name,
                    n: number("n"),
                    k: number("k"),
                    w: number("w"),
                    ends,
                    rows: number("rows") as usize,
                    row_exact: field("rowExact") == "true",
                    fp8: field("fp8") == "true",
                    exact: field("exact") == "true",
                    target: number("target") as usize,
                    mma: number("mma") as usize,
                    whole_k: number("wholeK") as usize,
                    wide: number("wide") as usize,
                    scratch: (0, 0),
                    lines: Vec::new(),
                });
            } else if let Some(scratch) = line.strip_prefix("scratch=") {
                let (planned, decode) = scratch.split_once(',').unwrap();
                cases.last_mut().unwrap().scratch = (planned.parse().unwrap(), decode.parse().unwrap());
            } else {
                cases.last_mut().unwrap().lines.push(line.to_string());
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
                bytes: case.scratch.0 as usize,
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
        let format = TrellisLinear::new(case.target, case.mma, case.whole_k, case.wide);
        if case.exact {
            format.with_exact_numerics()
        } else {
            format
        }
    }

    #[test]
    fn trellis_launches_and_scratch_match_java() {
        let cases = cases();
        assert!(cases.len() > 500, "{} cases", cases.len());
        let mut routes = [0usize; 5];
        for case in &cases {
            let format = format_of(case);
            let label = format!(
                "{} rows={} rowExact={} fp8={} exact={} target={} mma={} wholeK={} wide={}",
                case.name,
                case.rows,
                case.row_exact,
                case.fp8,
                case.exact,
                case.target,
                case.mma,
                case.whole_k,
                case.wide
            );
            assert_eq!(
                plan_lines(case, &format, case.fp8, case.row_exact),
                case.lines,
                "{label}"
            );
            let shape = TrellisShape {
                n: case.n,
                k: case.k,
                w: case.w,
                segments: case.ends.len() as u64,
                ends: &case.ends,
            };
            assert_eq!(
                format.scratch_for(&shape, case.rows as u64, true),
                case.scratch.0,
                "{label}"
            );
            assert_eq!(
                format.scratch_for(&shape, case.rows as u64, false),
                case.scratch.1,
                "{label}"
            );
            let first = case
                .lines
                .iter()
                .find(|line| !line.starts_with("euhedral_trellis_prepass"))
                .unwrap();
            for (slot, symbol) in [
                "euhedral_trellis_mma ",
                "euhedral_trellis_mma_rows",
                "euhedral_trellis_mma_rows64",
                "euhedral_trellis_reference",
                "euhedral_trellis_decode_bf16",
            ]
            .iter()
            .enumerate()
            {
                if first.starts_with(symbol) {
                    routes[slot] += 1;
                }
            }
        }
        // The fixture reaches every route: split partials, whole-K blocks of 32 and of 64 rows, the reference and the
        // prefill route.
        assert!(routes.iter().all(|count| *count > 0), "{routes:?}");
    }

    #[test]
    fn verify_rows_never_use_fp8() {
        for case in cases().iter().filter(|case| !case.exact) {
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
        let case = cases()
            .into_iter()
            .find(|case| case.name == "mlp/down" && !case.exact)
            .unwrap();
        let big = Case {
            rows: 100,
            scratch: (1 << 30, 0),
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

    #[test]
    fn the_kernel_tables_are_in_the_native_order() {
        let names: Vec<&str> = LinearKernel::ALL.iter().map(|kernel| kernel.symbol()).collect();
        assert_eq!(names.len(), 11);
        for (index, kernel) in LinearKernel::ALL.iter().enumerate() {
            assert_eq!(kernel.index(), index);
        }
        assert_eq!(names[0], "euhedral_trellis_prepass");
        assert_eq!(names[8], "euhedral_trellis_mma_rows_fp8");
        assert_eq!(ROW_EXACT_KERNELS[0], (30, "euhedral_q4_linear_tc_bf16"));
        let launch = LinearKernel::TrellisEpilogue.launch([1, 2, 3], [4, 5, 6], 7, Default::default());
        assert_eq!((launch.table, launch.kernel), (Table::Linear, 4));
        assert_eq!(
            launch.geometry(),
            "euhedral_trellis_epilogue grid=1,2,3 block=4,5,6 smem=7"
        );
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
            trellis: None,
        }
    }

    fn route_of(format: &Nvfp4Linear, weight: &WeightRef, rows: usize, row_exact: bool) -> Nvfp4Route {
        format.route_of(weight, rows, row_exact, false).unwrap()
    }

    #[test]
    fn nvfp4_routes_by_rows() {
        let weight = nvfp4(34816, 5120, WeightLayout::RowSplitK128V1);
        let native = Nvfp4Linear::with_native_route(true);
        let portable = Nvfp4Linear::with_native_route(false);
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
        let native = Nvfp4Linear::with_native_route(true);
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
        let lines = |route: Nvfp4Route, rows, k, n, sd4| route.lines(rows, k, n, sd4);
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
        let native = Nvfp4Linear::with_native_route(true);
        let portable = Nvfp4Linear::with_native_route(false);
        use euhedral_inference_engine::formats::LinearFormat;
        for rows in 1..=8usize {
            assert_eq!(native.scratch_bytes(&weight, rows, none), 0);
        }
        // The route's own scratch, or the skinny route's at 64 rows where that is larger (65 rows and more).
        for rows in [9usize, 64, 65, 512] {
            let own = nvfp4_native_scratch_bytes(rows as u64, 17408, 5120);
            let skinny = nvfp4_native_scratch_bytes((rows as u64).min(64), 17408, 5120);
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
        let format = Nvfp4Linear::with_native_route(true);
        let none = LinearMode::default();
        for (out, input) in shapes {
            let weight = nvfp4(out, input, WeightLayout::RowSplitK128V1);
            let mut needed = 0u64;
            for rows in 1..=600usize {
                if rows >= 9 {
                    needed = needed.max(nvfp4_native_scratch_bytes(rows as u64, input, out));
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
        assert!(nvfp4_native_scratch_bytes(64, 17_408, 5120) > nvfp4_native_scratch_bytes(65, 17_408, 5120));
        assert_eq!(
            format.scratch_bytes(&weight, 65, none) as u64,
            nvfp4_native_scratch_bytes(64, 17_408, 5120)
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
