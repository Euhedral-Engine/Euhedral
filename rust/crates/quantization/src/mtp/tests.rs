//! Host tests of the drafter's layout, its refusals and the structure of what it plans.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use super::*;
use crate::training::Step;

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

fn drafter(depths: u32) -> Drafter {
    Drafter::new(&config(depths), BASE).unwrap()
}

/// The kernels of `work`, in order, with a product written `"gemm"`.
fn kernels(work: &[Work]) -> Vec<&'static str> {
    work.iter()
        .flat_map(|piece| match piece {
            Work::Steps(steps) => steps.iter().map(Step::kernel).collect::<Vec<_>>(),
            Work::Product(_) => vec!["gemm"],
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
fn every_buffer_lies_inside_the_arena_and_none_overlap() {
    let drafter = drafter(3);
    let end = BASE + Drafter::arena_bytes(&config(3)).unwrap();
    assert_eq!(drafter.bytes(), end - BASE);
    let mut spans = Vec::new();
    for parameter in drafter.parameters() {
        let bytes = 4 * parameter.count();
        for address in [parameter.values, parameter.gradient, parameter.first, parameter.second] {
            spans.push((address, address + bytes));
        }
        if let Some(maximum) = parameter.maximum {
            spans.push((maximum, maximum + 4));
        }
    }
    spans.sort_unstable();
    assert!(spans.iter().all(|(start, stop)| BASE <= *start && *stop <= end));
    assert!(
        spans.windows(2).all(|pair| pair[0].1 <= pair[1].0),
        "parameter buffers overlap"
    );
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
    drafter.forward(3, &window(), BASE, BASE).unwrap();
    assert!(drafter.backward(1, &window()).is_ok());
    assert!(matches!(
        drafter.backward(3, &window()),
        Err(DrafterError::NotForwarded(2))
    ));
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
    let drafter = drafter(1);
    let work = drafter.step(1e-3, 1e-4, 1).unwrap();
    assert_eq!(count(&work, "adam"), 12);
    assert_eq!(count(&work, "absolute_maximum"), 5);
    assert_eq!(count(&drafter.measure().unwrap(), "absolute_maximum"), 5);
}
