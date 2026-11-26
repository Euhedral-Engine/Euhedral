//! Host tests of the drafter's layout, its refusals and the structure of what it plans.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use super::*;
use crate::kernels::Module;
use crate::training::Step;
use crate::training::tests::{Case, Scalar, assert_arguments, declared};

const BASE: u64 = 0x10_0000;

fn config(depths: u32) -> Config {
    Config {
        shape: Shape {
            hidden: 64,
            intermediate: 96,
            heads: 4,
            kv_heads: 2,
            head_dim: 32,
            rotary_dim: 16,
            theta: 1e7,
            eps: 1e-6,
        },
        depths,
        rows: 6,
        max_keys: 16,
        max_positions: 32,
    }
}

fn window() -> Window {
    Window {
        position: 4,
        first_key: 0,
        count: 5,
    }
}

/// A drafter that has had its gradients zeroed and its scales measured, as the caller's first work does.
fn drafter(depths: u32) -> Drafter {
    let mut drafter = Drafter::new(&config(depths), BASE).unwrap();
    drafter.initialise().unwrap();
    drafter.measure().unwrap();
    drafter
}

/// The forward passes of depths `1..=depths` over `window`, each over the previous output.
fn forward_all(drafter: &mut Drafter, depths: u32, window: &Window) -> Vec<Work> {
    let mut input = BASE;
    let mut work = Vec::new();
    for depth in 1..=depths {
        let forward = drafter.forward(depth, window, input, BASE).unwrap();
        input = forward.output;
        work.extend(forward.work);
    }
    work
}

/// The kernels of `work`, in order, with a product written `"gemm"`.
fn kernels(work: &[Work]) -> Vec<&'static str> {
    work.iter()
        .flat_map(|piece| match piece {
            Work::Steps(steps) => steps.iter().map(Step::kernel).collect::<Vec<_>>(),
            Work::Product(_) | Work::Head(_) => vec!["gemm"],
        })
        .collect()
}

fn count(work: &[Work], kernel: &str) -> usize {
    kernels(work).into_iter().filter(|name| *name == kernel).count()
}

#[test]
fn parameters_come_in_the_order_and_shapes_of_the_layer() {
    let drafter = drafter(3);
    let layout: Vec<_> = drafter
        .parameters()
        .iter()
        .map(|parameter| (parameter.name, parameter.rows, parameter.columns))
        .collect();
    assert_eq!(
        layout,
        [
            ("mtp/input_projection", 64, 128),
            ("mtp/layer/attention/query_key_gate_value", 384, 64),
            ("mtp/layer/attention/output", 64, 128),
            ("mtp/layer/mlp/gate_up", 192, 64),
            ("mtp/layer/mlp/down", 64, 96),
            ("mtp/embedding_norm", 1, 64),
            ("mtp/hidden_norm", 1, 64),
            ("mtp/layer/input_norm", 1, 64),
            ("mtp/layer/post_attention_norm", 1, 64),
            ("mtp/layer/attention/query_norm", 1, 32),
            ("mtp/layer/attention/key_norm", 1, 32),
            ("mtp/final_norm", 1, 64),
        ]
    );
    assert_eq!(drafter.parameter("mtp/final_norm").unwrap().columns, 64);
    assert!(drafter.parameter("mtp/unknown").is_none());
}

#[test]
fn only_projections_carry_a_global_scale_source() {
    let drafter = drafter(1);
    for (index, parameter) in drafter.parameters().iter().enumerate() {
        assert_eq!(parameter.maximum.is_some(), index < 5, "{}", parameter.name);
    }
}

#[test]
fn every_region_lies_inside_the_arena_and_none_overlap() {
    let drafter = drafter(3);
    let end = BASE + Drafter::arena_bytes(&config(3)).unwrap();
    assert_eq!(drafter.bytes(), end - BASE);
    let mut spans: Vec<_> = drafter
        .regions()
        .iter()
        .map(|(address, bytes)| (*address, address + bytes))
        .collect();
    // Every parameter buffer is one of the regions.
    for parameter in drafter.parameters() {
        let bytes = 4 * parameter.count();
        for address in [parameter.values, parameter.gradient, parameter.first, parameter.second] {
            assert!(spans.contains(&(address, address + bytes)), "{}", parameter.name);
        }
    }
    spans.sort_unstable();
    assert!(spans.iter().all(|(start, stop)| BASE <= *start && *stop <= end));
    assert!(spans.windows(2).all(|pair| pair[0].1 <= pair[1].0), "regions overlap");
}

/// The region of `regions` that holds `address`, as `(start, stop)`.
fn region_of(regions: &[(u64, u64)], address: u64) -> (u64, u64) {
    let (start, bytes) = *regions
        .iter()
        .find(|(start, bytes)| *start <= address && address < start + bytes)
        .unwrap_or_else(|| panic!("{address:#x} lies in no region"));
    (start, start + bytes)
}

#[test]
fn column_copies_stay_inside_the_regions_they_address() {
    let mut drafter = drafter(3);
    let window = window();
    let mut work = forward_all(&mut drafter, 3, &window);
    for depth in (1..=3).rev() {
        work.extend(drafter.backward(depth, &window).unwrap());
    }
    let mut copies = 0;
    for piece in &work {
        let Work::Steps(steps) = piece else { continue };
        for step in steps.iter().filter(|step| step.kernel() == "copy_columns") {
            let args = step.args();
            let (rows, columns) = (args.read::<i32>(4).unwrap() as u64, args.read::<i32>(5).unwrap() as u64);
            for (pointer, leading) in [(0, 1), (2, 3)] {
                let address = args.read::<u64>(pointer).unwrap();
                let last = address + 4 * ((rows - 1) * args.read::<i32>(leading).unwrap() as u64 + columns);
                let (start, stop) = region_of(drafter.regions(), address);
                assert!(
                    start <= address && last <= stop,
                    "a copy of {rows}x{columns} leaves its region"
                );
            }
            copies += 1;
        }
    }
    assert!(copies > 0);
}

#[test]
fn a_deeper_drafter_takes_more_arena() {
    let bytes = |depths| Drafter::arena_bytes(&config(depths)).unwrap();
    assert!(bytes(1) < bytes(2) && bytes(2) < bytes(3));
}

#[test]
fn a_misaligned_or_null_arena_is_refused() {
    for base in [0, BASE + 4] {
        assert!(matches!(Drafter::new(&config(1), base), Err(DrafterError::Arena(refused)) if refused == base));
    }
}

#[test]
fn a_configuration_the_kernels_cannot_run_is_refused_with_the_value() {
    let refuse = |change: &dyn Fn(&mut Config)| {
        let mut bad = config(2);
        change(&mut bad);
        Drafter::arena_bytes(&bad).unwrap_err()
    };
    assert!(matches!(
        refuse(&|c| c.depths = 0),
        DrafterError::Config {
            what: "depths",
            value: 0
        }
    ));
    assert!(matches!(
        refuse(&|c| c.shape.kv_heads = 3),
        DrafterError::Config {
            what: "kv_heads",
            value: 3
        }
    ));
    assert!(matches!(
        refuse(&|c| c.shape.head_dim = 48),
        DrafterError::Config {
            what: "head_dim",
            value: 48
        }
    ));
    assert!(matches!(
        refuse(&|c| c.shape.rotary_dim = 15),
        DrafterError::Config {
            what: "rotary_dim",
            value: 15
        }
    ));
    assert!(matches!(
        refuse(&|c| c.shape.hidden = 60),
        DrafterError::Config {
            what: "hidden",
            value: 60
        }
    ));
}

#[test]
fn a_window_outside_what_the_drafter_was_built_for_is_refused() {
    let mut drafter = drafter(2);
    let mut refused = |depth: u32, change: &dyn Fn(&mut Window)| {
        let mut bad = window();
        change(&mut bad);
        drafter.forward(depth, &bad, BASE, BASE).unwrap_err()
    };
    assert!(matches!(
        refused(0, &|_| {}),
        DrafterError::Depth { depth: 0, depths: 2 }
    ));
    assert!(matches!(
        refused(3, &|_| {}),
        DrafterError::Depth { depth: 3, depths: 2 }
    ));
    assert!(matches!(
        refused(1, &|w| w.count = 0),
        DrafterError::Rows { count: 0, rows: 6 }
    ));
    assert!(matches!(
        refused(1, &|w| w.count = 7),
        DrafterError::Rows { count: 7, rows: 6 }
    ));
    assert!(matches!(
        refused(1, &|w| w.first_key = 5),
        DrafterError::FirstKey { .. }
    ));
    assert!(matches!(
        refused(1, &|w| w.position = 20),
        DrafterError::Keys { keys: 25, max: 16 }
    ));
    assert!(matches!(
        refused(1, &|w| w.position = u64::MAX),
        DrafterError::Position {
            position: u64::MAX,
            max: 32
        }
    ));
    assert!(matches!(
        refused(1, &|w| {
            w.position = 30;
            w.first_key = 30;
        }),
        DrafterError::Position { position: 35, max: 32 }
    ));
}

#[test]
fn a_backward_pass_needs_the_forward_passes_of_every_depth_up_to_it() {
    let mut drafter = drafter(3);
    assert!(matches!(
        drafter.backward(1, &window()),
        Err(DrafterError::NotForwarded(1))
    ));
    drafter.forward(1, &window(), BASE, BASE).unwrap();
    assert!(drafter.backward(1, &window()).is_ok());
    assert!(matches!(
        drafter.backward(2, &window()),
        Err(DrafterError::NotForwarded(2))
    ));
}

#[test]
fn a_depth_needs_the_previous_depth_forwarded_over_the_same_window() {
    let mut drafter = drafter(3);
    let window = window();
    assert!(matches!(
        drafter.forward(3, &window, BASE, BASE),
        Err(DrafterError::NotForwarded(2))
    ));
    drafter.forward(1, &window, BASE, BASE).unwrap();
    let other = Window { count: 4, ..window };
    assert!(matches!(
        drafter.forward(2, &other, BASE, BASE),
        Err(DrafterError::NotForwarded(1))
    ));
    drafter.forward(2, &window, BASE, BASE).unwrap();
    assert!(drafter.forward(3, &window, BASE, BASE).is_ok());
}

#[test]
fn a_new_forward_pass_drops_the_deeper_passes_that_read_what_it_wrote() {
    let mut drafter = drafter(3);
    let window = window();
    forward_all(&mut drafter, 3, &window);
    assert!(drafter.backward(3, &window).is_ok());
    let next = Window { position: 5, ..window };
    drafter.forward(1, &next, BASE, BASE).unwrap();
    assert!(matches!(drafter.backward(3, &next), Err(DrafterError::NotForwarded(2))));
    assert!(matches!(
        drafter.forward(3, &next, BASE, BASE),
        Err(DrafterError::NotForwarded(2))
    ));
    assert!(drafter.backward(1, &next).is_ok());

    // The same window again: depth 2 was planned over the first forward pass of depth 1, which this one replaces.
    drafter.forward(2, &next, BASE, BASE).unwrap();
    drafter.forward(1, &next, BASE, BASE).unwrap();
    assert!(matches!(
        drafter.forward(3, &next, BASE, BASE),
        Err(DrafterError::NotForwarded(2))
    ));
    assert!(matches!(drafter.backward(2, &next), Err(DrafterError::NotForwarded(2))));
}

#[test]
fn a_backward_pass_goes_back_over_the_window_and_rounding_it_went_forward_over() {
    let mut drafter = drafter(2);
    let window = window();
    forward_all(&mut drafter, 2, &window);
    let more_rows = Window { count: 6, ..window };
    assert!(matches!(
        drafter.backward(1, &more_rows),
        Err(DrafterError::NotForwarded(1))
    ));
    let later = Window {
        position: window.position + 1,
        ..window
    };
    assert!(matches!(
        drafter.backward(2, &later),
        Err(DrafterError::NotForwarded(1))
    ));
    drafter.set_rounded(false);
    assert!(matches!(
        drafter.backward(2, &window),
        Err(DrafterError::NotForwarded(1))
    ));
    drafter.set_rounded(true);
    assert!(drafter.backward(2, &window).is_ok());
}

#[test]
fn nothing_accumulates_or_steps_before_the_gradients_are_zeroed() {
    let mut drafter = Drafter::new(&config(1), BASE).unwrap();
    drafter.measure().unwrap();
    drafter.forward(1, &window(), BASE, BASE).unwrap();
    assert!(matches!(
        drafter.backward(1, &window()),
        Err(DrafterError::NotInitialised)
    ));
    assert!(matches!(drafter.step(1e-3, 1e-3, 1), Err(DrafterError::NotInitialised)));
    let zeroing = drafter.initialise().unwrap();
    // The gradient and both moments of each of the 12 parameters.
    assert_eq!(count(&zeroing, "fill"), 36);
    assert!(drafter.backward(1, &window()).is_ok());
    assert!(drafter.step(1e-3, 1e-3, 1).is_ok());
}

#[test]
fn a_rounded_forward_pass_needs_the_scales_measured() {
    let mut drafter = Drafter::new(&config(1), BASE).unwrap();
    drafter.initialise().unwrap();
    assert!(matches!(
        drafter.forward(1, &window(), BASE, BASE),
        Err(DrafterError::NotMeasured)
    ));
    drafter.set_rounded(false);
    assert!(drafter.forward(1, &window(), BASE, BASE).is_ok());
    drafter.set_rounded(true);
    drafter.measure().unwrap();
    assert!(drafter.forward(1, &window(), BASE, BASE).is_ok());
}

#[test]
fn a_configuration_too_large_for_32_bit_offsets_is_refused_without_overflowing() {
    for change in [
        (|c: &mut Config| c.rows = u32::MAX) as fn(&mut Config),
        |c| c.shape.hidden = u32::MAX - 15,
        |c| c.shape.intermediate = u32::MAX - 15,
        |c| c.shape.heads = u32::MAX - 1,
        |c| c.max_keys = u32::MAX,
        |c| c.max_positions = u32::MAX,
    ] {
        let mut bad = config(2);
        change(&mut bad);
        assert!(Drafter::arena_bytes(&bad).is_err());
    }
}

#[test]
fn a_refused_setting_reports_its_value() {
    let mut bad = config(1);
    bad.shape.eps = -1.5;
    assert!(matches!(Drafter::arena_bytes(&bad), Err(DrafterError::Setting { what: "eps", value }) if value == -1.5));
    bad.shape.eps = 1e-6;
    bad.shape.theta = f64::NAN;
    assert!(
        matches!(Drafter::arena_bytes(&bad), Err(DrafterError::Setting { what: "theta", value }) if value.is_nan())
    );
}

#[test]
fn output_gradients_exist_for_the_depths_only() {
    let drafter = drafter(2);
    assert!(drafter.output_gradient(0).is_none());
    let (first, second) = (drafter.output_gradient(1).unwrap(), drafter.output_gradient(2).unwrap());
    assert_ne!(first, second);
    assert!(drafter.output_gradient(3).is_none());
}

#[test]
fn a_forward_pass_returns_the_depths_own_output_buffer() {
    let mut drafter = drafter(2);
    let first = drafter.forward(1, &window(), BASE, BASE).unwrap().output;
    let second = drafter.forward(2, &window(), first, BASE).unwrap().output;
    assert_ne!(first, second);
    assert_eq!(drafter.forward(1, &window(), BASE, BASE).unwrap().output, first);
}

#[test]
fn only_depths_after_the_first_attend_the_lineage_of_earlier_depths() {
    let mut drafter = drafter(3);
    let first = drafter.forward(1, &window(), BASE, BASE).unwrap().work;
    drafter.forward(2, &window(), BASE, BASE).unwrap();
    let third = drafter.forward(3, &window(), BASE, BASE).unwrap().work;
    assert_eq!(count(&first, "lineage_dot"), 0);
    assert_eq!(count(&first, "lineage_spread"), 0);
    assert_eq!(count(&third, "lineage_dot"), 2);
    assert_eq!(count(&third, "lineage_spread"), 2);
    // Only depth 1 stores the keys and values the sequence's later windows attend.
    assert_eq!(count(&first, "copy_columns"), count(&third, "copy_columns") + 2);
}

#[test]
fn rounding_the_projections_is_a_setting_of_the_plan() {
    let mut drafter = drafter(1);
    assert!(drafter.rounded());
    let rounded = drafter.forward(1, &window(), BASE, BASE).unwrap().work;
    assert_eq!(count(&rounded, "nvfp4_round"), 5);
    drafter.set_rounded(false);
    let plain = drafter.forward(1, &window(), BASE, BASE).unwrap().work;
    assert_eq!(count(&plain, "nvfp4_round"), 0);
    assert_eq!(count(&plain, "gemm"), count(&rounded, "gemm"));
}

#[test]
fn a_step_updates_every_parameter_and_measures_every_projection() {
    let mut drafter = drafter(1);
    let work = drafter.step(1e-3, 1e-4, 1).unwrap();
    assert_eq!(count(&work, "adam"), 12);
    assert_eq!(count(&work, "absolute_maximum"), 5);
    assert_eq!(count(&drafter.measure().unwrap(), "absolute_maximum"), 5);
}

#[test]
fn arguments_match_the_parameter_lists_the_loaded_source_declares() {
    let declared = declared(Module::Drafter.source());
    let softmax = Softmax {
        scores: 1,
        ld_scores: 15,
        visible0: 9,
        extras: 2,
        extra_stride: 5,
        extra_count: 2,
        heads: 4,
        rows: 6,
        scale: 0.125,
    };
    let lineage = Lineage {
        heads: 4,
        group: 2,
        d: 64,
        rows: 6,
        stride: 7,
        slot: 3,
    };
    let softmax_scalars = vec![
        ("ldS", Scalar::Int(15)),
        ("visible0", Scalar::Int(9)),
        ("extraStride", Scalar::Int(5)),
        ("extraCount", Scalar::Int(2)),
        ("rows", Scalar::Int(6)),
        ("scale", Scalar::Float(0.125)),
    ];
    let lineage_scalars = |first: &'static str, second: &'static str| {
        vec![
            (first, Scalar::Int(0)),
            (second, Scalar::Int(0)),
            ("stride", Scalar::Int(7)),
            ("slot", Scalar::Int(3)),
            ("heads", Scalar::Int(4)),
            ("group", Scalar::Int(2)),
            ("d", Scalar::Int(64)),
            ("rows", Scalar::Int(6)),
        ]
    };
    let with_leading = |mut scalars: Vec<(&'static str, Scalar)>, x: i32, y: i32| {
        for (name, value) in &mut scalars {
            match *name {
                "lda" | "ldx" => *value = Scalar::Int(x),
                "ldb" | "ldy" => *value = Scalar::Int(y),
                _ => {}
            }
        }
        scalars
    };
    let cases: Vec<Case> = vec![
        (
            plan_absolute_maximum(1, 77, 2).unwrap(),
            vec![("count", Scalar::Long(77))],
        ),
        (
            plan_nvfp4_round(1, 2, Some(Nvfp4Codes { packed: 3, scales: 4 }), 5, 32, 5).unwrap(),
            vec![("blocks", Scalar::Long(10))],
        ),
        (plan_softmax(&softmax).unwrap(), softmax_scalars.clone()),
        (
            plan_softmax_backward(&Softmax { extras: 3, ..softmax }, 2, 4).unwrap(),
            {
                // The backward kernel takes the probabilities, then their gradients.
                softmax_scalars
            },
        ),
        (
            plan_lineage_dot((1, 256), (2, 128), 3, &lineage).unwrap(),
            with_leading(lineage_scalars("lda", "ldb"), 256, 128),
        ),
        (
            plan_lineage_spread(1, (2, 128), (3, 256), &lineage).unwrap(),
            with_leading(lineage_scalars("ldx", "ldy"), 128, 256),
        ),
        (
            plan_lineage_gather(1, (2, 256), (3, 128), &lineage).unwrap(),
            with_leading(lineage_scalars("ldx", "ldy"), 256, 128),
        ),
        (
            plan_copy_columns((1, 40), (2, 24), 3, 17, true).unwrap(),
            vec![
                ("ldx", Scalar::Int(40)),
                ("ldy", Scalar::Int(24)),
                ("rows", Scalar::Int(3)),
                ("columns", Scalar::Int(17)),
                ("add", Scalar::Int(1)),
            ],
        ),
        (
            plan_cross_entropy(1, 4, 50, 2, 0.5, 3, 4, 5).unwrap(),
            vec![("vocabulary", Scalar::Int(50)), ("scale", Scalar::Float(0.5))],
        ),
        (
            plan_nvfp4_decode_bf16(1, 2, 3, 7, 48, 40, 4).unwrap(),
            vec![
                ("rows", Scalar::Long(7)),
                ("kPad", Scalar::Int(48)),
                ("k", Scalar::Int(40)),
            ],
        ),
        (plan_float_to_bf16(1, 99, 2).unwrap(), vec![("count", Scalar::Long(99))]),
        (
            plan_q3_embedding_rows(1, 2, 5, 128, 4096, 3).unwrap(),
            vec![
                ("rows", Scalar::Int(5)),
                ("width", Scalar::Int(128)),
                ("scaleOffset", Scalar::Long(4096)),
            ],
        ),
    ];
    let mut seen = assert_arguments(&declared, &cases);
    seen.sort_unstable();
    seen.dedup();
    // Every kernel of the module has a planner under test here.
    let unplanned: Vec<_> = Module::Drafter
        .kernels()
        .iter()
        .copied()
        .filter(|name| !seen.contains(name))
        .collect();
    assert!(unplanned.is_empty(), "no case for {unplanned:?}");
}
