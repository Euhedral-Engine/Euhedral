use std::cell::RefCell;
use std::mem::replace;

use euhedral_graph::{StageCx, StageError, Submitted};
use euhedral_inference_edrl::tiny_flash::tiny_config;

use super::*;
use crate::quantum::Predecessor;

#[test]
fn a_row_count_takes_the_bucket_javas_row_bucket_gives_it() {
    let buckets = Buckets::new(512, 256, 8).unwrap();
    let chosen: Vec<u32> = [1, 2, 9, 16, 17, 29, 512]
        .into_iter()
        .map(|rows| buckets.bucket_for(rows).unwrap())
        .collect();
    assert_eq!(chosen, [1, 16, 16, 16, 512, 512, 512]);
}

#[test]
fn a_step_starts_on_the_graph_in_the_place_of_its_bucket_among_the_sizes() {
    let buckets = Buckets::new(512, 256, 8).unwrap();
    let places: Vec<usize> = [1, 2, 16, 17, 512]
        .into_iter()
        .map(|rows| buckets.index_for(rows).unwrap())
        .collect();
    assert_eq!(places, [0, 1, 1, 2, 2]);
    // With no middle bucket the graphs are two, and the second serves every step above one row.
    let small = Buckets::new(8, 3, 2).unwrap();
    assert_eq!([1, 2, 8].map(|rows| small.index_for(rows).unwrap()), [0, 1, 1]);
    assert_eq!(Buckets::new(1, 3, 2).unwrap().index_for(1).unwrap(), 0);
}

#[test]
fn a_step_with_no_graph_to_start_on_is_refused_naming_the_rows() {
    let buckets = Buckets::new(512, 256, 8).unwrap();
    assert!(buckets.index_for(0).is_err());
    assert!(buckets.index_for(513).unwrap_err().to_string().contains("513"));
    let wide = buckets.index_for(usize::MAX).unwrap_err().to_string();
    assert!(wide.contains(&usize::MAX.to_string()), "{wide}");
}

#[test]
fn a_model_whose_largest_step_is_under_sixteen_rows_has_no_middle_bucket() {
    let buckets = Buckets::new(8, 3, 2).unwrap();
    assert_eq!(buckets.sizes().as_slice(), [1, 8]);
    assert_eq!(buckets.bucket_for(2).unwrap(), 8);
    assert_eq!(Buckets::new(1, 3, 2).unwrap().sizes().as_slice(), [1]);
    assert_eq!(Buckets::new(512, 3, 2).unwrap().sizes().as_slice(), [1, 16, 512]);
}

#[test]
fn a_step_of_no_rows_or_of_more_than_the_largest_bucket_is_refused_naming_the_count_and_the_limit() {
    let buckets = Buckets::new(512, 256, 8).unwrap();
    assert!(buckets.bucket_for(0).is_err());
    let message = buckets.bucket_for(513).unwrap_err().to_string();
    assert!(message.contains("513") && message.contains("512"), "{message}");
}

#[test]
fn buckets_over_no_rows_or_no_experts_or_more_experts_per_token_than_experts_are_refused() {
    assert!(Buckets::new(0, 3, 2).is_err());
    assert!(Buckets::new(8, 0, 0).is_err());
    assert!(Buckets::new(8, 3, 0).is_err());
    assert!(Buckets::new(8, 3, 4).is_err());
}

#[test]
fn the_moe_block_of_a_bucket_names_the_experts_its_rows_can_choose_and_no_more_than_the_model_has() {
    let buckets = Buckets::new(512, 256, 8).unwrap();
    assert_eq!(buckets.expert_width(1), 8);
    assert_eq!(buckets.expert_width(16), 128);
    assert_eq!(buckets.expert_width(512), 256);
}

#[test]
fn the_plan_of_a_bucket_has_its_expert_width_and_ends_in_the_head() {
    let buckets = Buckets::new(32, 3, 2).unwrap();
    let config = tiny_config();
    for (bucket, width) in [(1, 2), (16, 3), (32, 3)] {
        let plan = buckets.plan(&config, bucket).unwrap();
        let experts = plan
            .nodes()
            .iter()
            .filter(|node| matches!(node.kind, NodeKind::MoeExpert(1, _)))
            .count();
        assert_eq!(experts, width, "bucket {bucket}");
        assert_eq!(plan.nodes().last().unwrap().kind, NodeKind::Head);
    }
    assert!(buckets.plan(&config, 8).is_err(), "8 rows is no bucket of this model");
}

#[test]
fn every_layer_runs_its_block_then_its_experts_then_its_end_injection_before_the_next_layer_and_the_head_is_last() {
    let buckets = Buckets::new(32, 3, 2).unwrap();
    let plan = buckets.plan(&tiny_config(), 32).unwrap();
    let order: Vec<NodeKind> = plan.nodes().iter().map(|node| node.kind).collect();
    let at = |kind| order.iter().position(|&candidate| candidate == kind).unwrap();
    for layer in 0..2 {
        assert!(at(NodeKind::Block(layer)) < at(NodeKind::MoeRoute(layer)));
        assert!(at(NodeKind::MoeRoute(layer)) < at(NodeKind::MoeFinish(layer)));
        assert!(at(NodeKind::MoeFinish(layer)) < at(NodeKind::End(layer)));
    }
    assert!(at(NodeKind::End(0)) < at(NodeKind::Block(1)));
    assert!(at(NodeKind::End(1)) < at(NodeKind::FinalMix));
    assert_eq!(at(NodeKind::Head), order.len() - 1);
    assert_eq!(
        at(NodeKind::Embed),
        0,
        "the embedding is the stage a predecessor's sink is wired to"
    );
}

/// A stage that does nothing.
struct Marked;

impl Stage<Quantum> for Marked {
    fn run(&self, _quantum: &Quantum, _cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        Ok(Submitted::done())
    }
}

/// A source that records the nodes it was asked for.
#[derive(Default)]
struct Recording(RefCell<Vec<(u32, NodeKind)>>);

impl NodeStages for Recording {
    fn stage(&self, bucket: u32, kind: NodeKind) -> Result<Arc<dyn Stage<Quantum>>, FlashError> {
        self.0.borrow_mut().push((bucket, kind));
        Ok(Arc::new(Marked))
    }
}

#[test]
fn the_stages_of_a_plan_are_asked_for_in_node_order_with_the_bucket() {
    let plan = DensePlan::with_moe(&tiny_config(), 2).unwrap();
    let source = Recording::default();
    let stages = stages_of(&plan, 16, &source).unwrap();
    assert_eq!(stages.len(), plan.nodes().len());
    let asked = source.0.borrow();
    let expected: Vec<(u32, NodeKind)> = plan.nodes().iter().map(|node| (16, node.kind)).collect();
    assert_eq!(*asked, expected);
}

#[test]
fn a_source_that_refuses_a_node_refuses_the_plan() {
    struct Refusing;
    impl NodeStages for Refusing {
        fn stage(&self, _bucket: u32, kind: NodeKind) -> Result<Arc<dyn Stage<Quantum>>, FlashError> {
            Err(FlashError::refused(OP, format!("no stage for {kind:?}")))
        }
    }
    let plan = DensePlan::with_moe(&tiny_config(), 2).unwrap();
    assert!(stages_of(&plan, 16, &Refusing).is_err());
}

#[test]
fn a_new_quantum_waits_for_the_sink_of_the_one_before_it_at_its_embedding_and_nowhere_else() {
    let readers = [None];
    let before = Predecessor {
        root: StageId(0),
        sink: StageId(41),
        slot_readers: &readers,
    };
    let edges = dependencies(&[before]);
    assert_eq!(
        edges.as_slice(),
        [Dependency {
            predecessor: 0,
            from_stage: StageId(41),
            to_stage: StageId(0),
        }]
    );
    assert!(dependencies(&[]).is_empty(), "a retired predecessor orders nothing");
}

#[test]
fn each_bucket_names_the_last_stage_of_its_own_graph_as_the_sink_a_started_quantum_carries() {
    let buckets = Buckets::new(32, 3, 2).unwrap();
    let config = tiny_config();
    let sinks: Vec<StageId> = buckets
        .sizes()
        .iter()
        .map(|bucket| buckets.sink(&config, *bucket).unwrap())
        .collect();
    for (bucket, sink) in buckets.sizes().iter().zip(&sinks) {
        let nodes = buckets.plan(&config, *bucket).unwrap().nodes().len();
        assert_eq!(*sink, StageId((nodes - 1) as u16), "bucket {bucket}");
    }
    assert_ne!(
        sinks[0], sinks[1],
        "the buckets' graphs differ in size, and so in their sinks"
    );
    assert!(buckets.sink(&config, 8).is_err(), "8 rows is no bucket of this model");
}

#[test]
fn the_sink_is_the_last_node() {
    let plan = DensePlan::with_moe(&tiny_config(), 2).unwrap();
    assert_eq!(sink_of(&plan).unwrap(), StageId((plan.nodes().len() - 1) as u16));
}

/// Whether `from` reaches `to` over the submission and completion edges of `plan`.
fn reaches(plan: &DensePlan, from: usize, to: usize) -> bool {
    let mut seen = vec![false; plan.nodes().len()];
    let mut open = vec![to];
    while let Some(node) = open.pop() {
        if node == from {
            return true;
        }
        if replace(&mut seen[node], true) {
            continue;
        }
        let node = &plan.nodes()[node];
        open.extend(node.after.iter().chain(&node.completed));
    }
    false
}

#[test]
fn the_edges_of_a_bucket_order_each_layer_through_its_experts_and_every_node_ends_at_the_head() {
    let buckets = Buckets::new(32, 3, 2).unwrap();
    let plan = buckets.plan(&tiny_config(), 16).unwrap();
    let at = |kind| plan.index_of(kind).unwrap();
    for layer in 0..2 {
        for slot in 0..3 {
            let expert = at(NodeKind::MoeExpert(layer, slot));
            assert!(reaches(&plan, at(NodeKind::Block(layer)), expert));
            assert!(reaches(&plan, expert, at(NodeKind::End(layer))));
        }
        assert!(reaches(
            &plan,
            at(NodeKind::MoeRoute(layer)),
            at(NodeKind::MoePlan(layer))
        ));
        assert!(reaches(&plan, at(NodeKind::MoeShared(layer)), at(NodeKind::End(layer))));
    }
    assert!(reaches(&plan, at(NodeKind::End(0)), at(NodeKind::Block(1))));
    let head = at(NodeKind::Head);
    assert!(
        (0..plan.nodes().len()).all(|node| reaches(&plan, node, head)),
        "a node does not lead to the head, so no marker covers its work"
    );
    assert!(
        (1..plan.nodes().len())
            .filter(|&node| plan.nodes()[node].after.is_empty() && plan.nodes()[node].completed.is_empty())
            .all(|node| plan.nodes()[node].kind.host()),
        "a device node has no predecessor, so it would run beside the quantum before it"
    );
}
