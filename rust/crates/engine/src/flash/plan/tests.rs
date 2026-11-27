use euhedral_inference_edrl::tiny_flash::tiny_config;

use super::*;

fn kinds(plan: &DensePlan) -> Vec<NodeKind> {
    plan.nodes().iter().map(|node| node.kind).collect()
}

#[test]
fn the_tiny_model_has_the_embedding_its_n_gram_roots_both_layers_the_final_mix_and_the_head() {
    let plan = DensePlan::from_config(&tiny_config()).unwrap();
    let mut expected = vec![NodeKind::Embed, NodeKind::Block(0), NodeKind::End(0), NodeKind::PleIds];
    expected.extend((0..PARTS).map(NodeKind::PleGather));
    expected.extend([
        NodeKind::Ple,
        NodeKind::Block(1),
        NodeKind::End(1),
        NodeKind::FinalMix,
        NodeKind::Head,
    ]);
    assert_eq!(kinds(&plan), expected);
}

#[test]
fn the_n_gram_embedding_waits_for_every_part_and_the_layer_before_it() {
    let plan = DensePlan::from_config(&tiny_config()).unwrap();
    let at = |kind| plan.index_of(kind).unwrap();
    let ple = &plan.nodes()[at(NodeKind::Ple)];
    let mut expected: Vec<usize> = (0..PARTS).map(|part| at(NodeKind::PleGather(part))).collect();
    expected.push(at(NodeKind::End(0)));
    assert_eq!(ple.after, expected);
    for part in 0..PARTS {
        assert_eq!(
            plan.nodes()[at(NodeKind::PleGather(part))].after,
            [at(NodeKind::PleIds)]
        );
    }
    assert!(
        plan.nodes()[at(NodeKind::PleIds)].after.is_empty(),
        "the ids root has no predecessor"
    );
    assert_eq!(plan.nodes()[at(NodeKind::Block(1))].after, [at(NodeKind::Ple)]);
}

#[test]
fn each_layer_runs_its_block_then_its_end_injection_then_the_next_layer() {
    let plan = DensePlan::from_config(&tiny_config()).unwrap();
    let at = |kind| plan.index_of(kind).unwrap();
    assert_eq!(plan.nodes()[at(NodeKind::Block(0))].after, [at(NodeKind::Embed)]);
    assert_eq!(plan.nodes()[at(NodeKind::End(0))].after, [at(NodeKind::Block(0))]);
    assert_eq!(plan.nodes()[at(NodeKind::FinalMix)].after, [at(NodeKind::End(1))]);
    assert_eq!(plan.nodes()[at(NodeKind::Head)].after, [at(NodeKind::FinalMix)]);
    assert_eq!(at(NodeKind::Head), plan.nodes().len() - 1, "the head is the last node");
}

#[test]
fn a_model_without_the_n_gram_layer_has_no_host_roots() {
    let mut config = tiny_config();
    config.ple.layers.clear();
    let plan = DensePlan::from_config(&config).unwrap();
    assert_eq!(
        kinds(&plan),
        [
            NodeKind::Embed,
            NodeKind::Block(0),
            NodeKind::End(0),
            NodeKind::Block(1),
            NodeKind::End(1),
            NodeKind::FinalMix,
            NodeKind::Head
        ]
    );
}

#[test]
fn a_model_with_two_n_gram_layers_or_one_past_the_last_is_refused_with_them() {
    let mut config = tiny_config();
    config.ple.layers = vec![0, 1];
    assert!(
        DensePlan::from_config(&config)
            .unwrap_err()
            .to_string()
            .contains("[0, 1]")
    );
    config.ple.layers = vec![2];
    assert!(
        DensePlan::from_config(&config)
            .unwrap_err()
            .to_string()
            .contains("layer 2")
    );
    config.ple.layers = vec![-1];
    assert!(DensePlan::from_config(&config).is_err());
}

#[test]
fn the_graph_validates_and_holds_the_host_nodes_without_lanes() {
    let plan = DensePlan::from_config(&tiny_config()).unwrap();
    plan.spec(LaneId(0)).unwrap();
    let host = plan.nodes().iter().filter(|node| node.kind.host()).count();
    assert_eq!(host, PARTS + 1);
}

#[test]
fn a_moe_layer_has_a_route_a_shared_expert_a_plan_a_fetch_and_an_expert_per_slot_and_a_finish_between_its_block_and_its_end()
 {
    let plan = DensePlan::with_moe(&tiny_config(), 2).unwrap();
    let at = |kind| plan.index_of(kind).unwrap();
    let after = |kind| plan.nodes()[at(kind)].after.clone();
    let completed = |kind| plan.nodes()[at(kind)].completed.clone();
    assert_eq!(after(NodeKind::MoeRoute(1)), [at(NodeKind::Block(1))]);
    assert_eq!(after(NodeKind::MoeShared(1)), [at(NodeKind::Block(1))]);
    // The plan reads what the route copied to the host, so it waits for the route's device work, not its submission.
    assert!(after(NodeKind::MoePlan(1)).is_empty());
    assert_eq!(completed(NodeKind::MoePlan(1)), [at(NodeKind::MoeRoute(1))]);
    for slot in 0..2 {
        assert_eq!(after(NodeKind::MoeFetch(1, slot)), [at(NodeKind::MoePlan(1))]);
        assert_eq!(
            after(NodeKind::MoeExpert(1, slot)),
            [at(NodeKind::MoeFetch(1, slot)), at(NodeKind::MoePlan(1))]
        );
    }
    assert_eq!(
        after(NodeKind::MoeFinish(1)),
        [
            at(NodeKind::MoeShared(1)),
            at(NodeKind::MoeExpert(1, 0)),
            at(NodeKind::MoeExpert(1, 1))
        ]
    );
    assert_eq!(after(NodeKind::End(1)), [at(NodeKind::MoeFinish(1))]);
    assert_eq!(after(NodeKind::Block(1)), [at(NodeKind::Ple)]);
    assert!(plan.index_of(NodeKind::MoeFetch(1, 2)).is_none(), "the width is fixed");
}

#[test]
fn the_next_layers_block_waits_for_the_end_of_the_layer_before_it() {
    let plan = DensePlan::with_moe(&tiny_config(), 2).unwrap();
    let at = |kind| plan.index_of(kind).unwrap();
    assert_eq!(plan.nodes()[at(NodeKind::Block(0))].after, [at(NodeKind::Embed)]);
    assert_eq!(plan.nodes()[at(NodeKind::End(0))].after, [at(NodeKind::MoeFinish(0))]);
    assert_eq!(plan.nodes()[at(NodeKind::FinalMix)].after, [at(NodeKind::End(1))]);
}

#[test]
fn the_moe_graph_validates_with_experts_on_one_lane_and_on_several_and_the_fetches_have_none() {
    let plan = DensePlan::with_moe(&tiny_config(), 3).unwrap();
    plan.spec(LaneId(0)).unwrap();
    plan.spec_on(LaneId(0), &[LaneId(1), LaneId(2)]).unwrap();
    let fetches = plan
        .nodes()
        .iter()
        .filter(|node| matches!(node.kind, NodeKind::MoeFetch(..)))
        .count();
    assert_eq!(fetches, 2 * 3);
    assert!(
        plan.nodes()
            .iter()
            .filter(|node| matches!(node.kind, NodeKind::MoeFetch(..)))
            .all(|node| node.kind.host())
    );
}

#[test]
fn a_moe_plan_of_no_experts_is_refused() {
    assert!(DensePlan::with_moe(&tiny_config(), 0).is_err());
}
