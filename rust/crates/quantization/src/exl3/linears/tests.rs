#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use std::collections::BTreeSet;

use super::*;

// A mutant that drops the MLP or lists the GDN a/b projections changes the count: 48 GDN layers of 6 linears and 16
// attention layers of 7.
#[test]
fn the_built_in_model_has_the_linears_of_each_layer_kind() {
    let all = qwen38(64).unwrap();
    assert_eq!(all.len(), 48 * 6 + 16 * 7);
    assert!(
        all.iter()
            .all(|linear| !linear.name.ends_with("in_proj_a") && !linear.name.ends_with("in_proj_b"))
    );
    let layer3: Vec<&Budgeted> = all.iter().filter(|linear| linear.layer == 3).collect();
    assert_eq!(layer3.len(), 7);
    assert!(
        layer3
            .iter()
            .filter(|l| l.name.contains("self_attn"))
            .all(|l| l.priority == 2)
    );
    let layer0: Vec<&Budgeted> = all.iter().filter(|linear| linear.layer == 0).collect();
    assert!(
        layer0
            .iter()
            .filter(|l| l.name.contains("linear_attn"))
            .all(|l| l.priority == 1)
    );
    assert!(
        layer0
            .iter()
            .filter(|l| l.name.contains("mlp"))
            .all(|l| l.priority == 0)
    );
}

// A mutant that groups q, k, v, o together, or up and down, changes these groups.
#[test]
fn promotion_groups_follow_the_module_tree() {
    let all = qwen38(4).unwrap();
    let group = |name: &str| {
        all.iter()
            .find(|linear| linear.name.ends_with(name) && linear.layer == 3)
            .unwrap()
            .group
            .clone()
    };
    assert_eq!(group("q_proj"), group("k_proj"));
    assert_eq!(group("q_proj"), group("v_proj"));
    assert_ne!(group("q_proj"), group("o_proj"));
    assert_eq!(group("up_proj"), group("gate_proj"));
    assert_ne!(group("up_proj"), group("down_proj"));
    let groups: BTreeSet<&str> = all.iter().map(|linear| linear.group.as_str()).collect();
    assert_eq!(groups.len(), 4 * 4);
}

#[test]
fn a_role_without_a_shape_is_refused() {
    let shapes = |_: &str| -> Result<[u64; 2], String> { Err("no shape".to_owned()) };
    assert_eq!(budget(2, |_| false, shapes), Err("no shape".to_owned()));
}
