use super::fixtures::{arena, tokens};
use super::{NodeId, PrefixTree, ReserveError, Speculation};

const SLOT: u64 = 4096;

fn tree(slots: u32) -> PrefixTree {
    PrefixTree::new(arena(slots, SLOT))
}

fn add(tree: &mut PrefixTree, parent: NodeId, tokens: &[u32], position: u32) -> NodeId {
    let node = tree.reserve(parent, tokens, position, None, 1).unwrap();
    tree.publish(node);
    node
}

#[test]
fn an_empty_tree_misses() {
    let mut tree = tree(16);
    assert!(tree.lookup(&tokens(0, 100), None).is_none());
}

#[test]
fn matches_the_deepest_chain() {
    let mut tree = tree(64);
    let stored = tokens(0, 6000);
    let root = tree.root();
    let a = add(&mut tree, root, &stored, 2048);
    let b = add(&mut tree, a, &stored, 4096);
    let matched = tree.lookup(&tokens(0, 5000), None).unwrap();
    assert_eq!(matched.leaf(), b);
    assert_eq!(tree.node(matched.leaf()).position(), 4096);
    assert_eq!(matched.chain(), &[a, b]);
}

#[test]
fn stops_at_the_first_span_that_differs() {
    let mut tree = tree(64);
    let stored = tokens(0, 6000);
    let root = tree.root();
    let a = add(&mut tree, root, &stored, 2048);
    add(&mut tree, a, &stored, 4096);
    let mut prompt = tokens(0, 5000);
    prompt[3000] = u32::MAX;
    assert_eq!(tree.lookup(&prompt, None).unwrap().leaf(), a);
}

#[test]
fn a_prompt_that_shares_only_the_first_token_misses() {
    let mut tree = tree(16);
    let root = tree.root();
    add(&mut tree, root, &tokens(0, 3000), 2048);
    let mut prompt = tokens(0, 3000);
    prompt[1] = u32::MAX;
    assert!(tree.lookup(&prompt, None).is_none());
}

#[test]
fn a_prompt_never_matches_at_or_beyond_its_own_length() {
    let mut tree = tree(64);
    let stored = tokens(0, 6000);
    let root = tree.root();
    let a = add(&mut tree, root, &stored, 2048);
    add(&mut tree, a, &stored, 4096);
    assert_eq!(
        tree.lookup(&tokens(0, 4096), None).unwrap().leaf(),
        a,
        "a prompt that ends on a node uses the one before"
    );
    assert!(
        tree.lookup(&tokens(0, 2048), None).is_none(),
        "and a prompt that ends on the first node has none"
    );
    assert_eq!(tree.lookup(&tokens(0, 2049), None).unwrap().leaf(), a);
}

#[test]
fn prefers_the_child_that_reaches_further() {
    let mut tree = tree(64);
    let stored = tokens(0, 6000);
    let root = tree.root();
    add(&mut tree, root, &stored, 2048);
    let far = add(&mut tree, root, &stored, 3072);
    let matched = tree.lookup(&tokens(0, 5000), None).unwrap();
    assert_eq!(matched.leaf(), far);
    assert_eq!(matched.chain().len(), 1);
}

#[test]
fn mtp_needs_an_unbroken_mtp_chain() {
    let mut tree = tree(64);
    let stored = tokens(0, 6000);
    let root = tree.root();
    let a = tree.reserve(root, &stored, 2048, Some(Speculation::Mtp), 1).unwrap();
    tree.publish(a);
    let b = tree.reserve(a, &stored, 4096, None, 1).unwrap();
    tree.publish(b);
    let c = tree.reserve(b, &stored, 5120, Some(Speculation::Mtp), 1).unwrap();
    assert_eq!(
        tree.node(c).speculation(),
        None,
        "an MTP node under a node without MTP state is unusable"
    );
    tree.abort(c);
    assert_eq!(tree.lookup(&tokens(0, 5000), Some(Speculation::Mtp)).unwrap().leaf(), a);
    assert_eq!(tree.lookup(&tokens(0, 5000), None).unwrap().leaf(), b);
}

#[test]
fn find_locates_a_published_child_by_span() {
    let mut tree = tree(16);
    let stored = tokens(0, 3000);
    let root = tree.root();
    let a = add(&mut tree, root, &stored, 2048);
    assert_eq!(tree.find(root, &stored, 2048, None), Some(a));
    assert_eq!(tree.find(root, &stored, 1024, None), None);
    let mut other = tokens(0, 3000);
    other[10] = u32::MAX;
    assert_eq!(tree.find(root, &other, 2048, None), None);
    assert_eq!(
        tree.find(root, &tokens(0, 1000), 2048, None),
        None,
        "tokens shorter than the position"
    );
}

#[test]
fn find_any_ignores_the_speculative_kind() {
    let mut tree = tree(16);
    let stored = tokens(0, 3000);
    let root = tree.root();
    let with = tree.reserve(root, &stored, 2048, Some(Speculation::Mtp), 1).unwrap();
    tree.publish(with);
    assert_eq!(tree.find(root, &stored, 2048, None), None);
    assert_eq!(tree.find_any(root, &stored, 2048), Some(with));
}

#[test]
fn a_node_with_and_without_mtp_state_can_coexist_for_the_same_span() {
    let mut tree = tree(16);
    let stored = tokens(0, 3000);
    let root = tree.root();
    let plain = add(&mut tree, root, &stored, 2048);
    let with_mtp = tree.reserve(root, &stored, 2048, Some(Speculation::Mtp), 1).unwrap();
    tree.publish(with_mtp);
    assert_eq!(tree.find(root, &stored, 2048, None), Some(plain));
    assert_eq!(tree.find(root, &stored, 2048, Some(Speculation::Mtp)), Some(with_mtp));
    assert_eq!(
        tree.lookup(&tokens(0, 3000), Some(Speculation::Mtp)).unwrap().leaf(),
        with_mtp
    );
}

#[test]
fn an_unpublished_node_is_invisible_until_published() {
    let mut tree = tree(16);
    let root = tree.root();
    let node = tree.reserve(root, &tokens(0, 3000), 2048, None, 1).unwrap();
    assert!(tree.lookup(&tokens(0, 3000), None).is_none());
    assert_eq!(tree.size(), 0);
    tree.publish(node);
    assert_eq!(tree.lookup(&tokens(0, 3000), None).unwrap().leaf(), node);
    assert_eq!(tree.size(), 1);
}

#[test]
fn evicts_the_least_recently_used_leaf_for_room() {
    let mut tree = tree(2);
    let root = tree.root();
    let first = add(&mut tree, root, &tokens(0, 3000), 2048);
    add(&mut tree, root, &tokens(10_000, 3000), 2048);
    let held = tree.lookup(&tokens(0, 3000), None).unwrap();
    tree.release(held);
    add(&mut tree, root, &tokens(20_000, 3000), 2048);
    assert!(
        tree.lookup(&tokens(10_000, 3000), None).is_none(),
        "the older entry was evicted"
    );
    assert_eq!(tree.lookup(&tokens(0, 3000), None).unwrap().leaf(), first);
    assert_eq!(tree.evictions(), 1);
}

#[test]
fn never_evicts_a_node_with_children() {
    let mut tree = tree(3);
    let stored = tokens(0, 6000);
    let root = tree.root();
    let a = add(&mut tree, root, &stored, 2048);
    let child = add(&mut tree, a, &stored, 4096);
    add(&mut tree, root, &tokens(10_000, 3000), 2048);
    let more = tree.reserve(root, &tokens(20_000, 3000), 2048, None, 1);
    assert!(more.is_ok(), "the childless node made room");
    assert_eq!(
        tree.find(a, &stored, 4096, None),
        None,
        "the child went, not the parent it hung from"
    );
    assert!(tree.get(child).is_none());
    assert_eq!(tree.find(root, &stored, 2048, None), Some(a));
}

#[test]
fn never_evicts_a_pinned_match() {
    let mut tree = tree(1);
    let stored = tokens(0, 3000);
    let root = tree.root();
    add(&mut tree, root, &stored, 2048);
    let matched = tree.lookup(&tokens(0, 3000), None).unwrap();
    assert_eq!(
        tree.reserve(root, &tokens(10_000, 3000), 2048, None, 1),
        Err(ReserveError::NoRoom),
        "nothing evictable"
    );
    assert_eq!(tree.find(root, &stored, 2048, None), Some(matched.leaf()));
    tree.release(matched);
    assert!(tree.reserve(root, &tokens(10_000, 3000), 2048, None, 1).is_ok());
}

#[test]
fn the_parent_of_a_reservation_is_not_evicted_for_it() {
    let mut tree = tree(2);
    let stored = tokens(0, 6000);
    let root = tree.root();
    let a = add(&mut tree, root, &stored, 2048);
    let b = tree.reserve(a, &stored, 4096, None, 1).unwrap();
    assert_eq!(
        tree.reserve(a, &stored, 4352, None, 1),
        Err(ReserveError::NoRoom),
        "the only evictable node is the pinned parent"
    );
    assert_eq!(tree.find(root, &stored, 2048, None), Some(a));
    tree.abort(b);
    assert!(
        tree.reserve(a, &stored, 4352, None, 1).is_ok(),
        "its pin was released with the abort"
    );
}

#[test]
fn refuses_a_reservation_larger_than_the_arena() {
    let mut tree = tree(1);
    let root = tree.root();
    assert_eq!(
        tree.reserve(root, &tokens(0, 3000), 2048, None, 2),
        Err(ReserveError::NoRoom)
    );
}

#[test]
fn rejects_spans_that_do_not_advance_overrun_the_tokens_or_leave_the_page_grid() {
    let mut tree = tree(16);
    let stored = tokens(0, 3000);
    let root = tree.root();
    for position in [0, 3072, 2000] {
        assert_eq!(
            tree.reserve(root, &stored, position, None, 1),
            Err(ReserveError::Span),
            "position {position}"
        );
    }
    assert!(!tree.node(root).pinned(), "a refusal leaves no pin");
}

// Slot accounting: what the byte-extent allocator's tests covered, over slot leases.

#[test]
fn a_reservation_leases_its_slots_and_abort_returns_them() {
    let arena = arena(4, SLOT);
    let mut tree = PrefixTree::new(arena.clone());
    let root = tree.root();
    let node = tree.reserve(root, &tokens(0, 3000), 2048, None, 3).unwrap();
    assert_eq!(arena.leased(), 3);
    assert_eq!(tree.node(node).slots().len(), 3);
    tree.abort(node);
    assert_eq!(arena.leased(), 0);
    assert_eq!(tree.size(), 0);
    assert!(tree.lookup(&tokens(0, 3000), None).is_none());
    assert!(tree.get(node).is_none(), "the id is stale after the abort");
}

#[test]
fn a_node_holds_distinct_slots() {
    let arena = arena(4, SLOT);
    let mut tree = PrefixTree::new(arena.clone());
    let root = tree.root();
    let node = tree.reserve(root, &tokens(0, 3000), 2048, None, 4).unwrap();
    let mut indices: Vec<u32> = tree.node(node).slots().iter().map(|s| s.index()).collect();
    indices.sort_unstable();
    assert_eq!(indices, [0, 1, 2, 3]);
}

#[test]
fn eviction_returns_every_slot_of_the_victim() {
    let arena = arena(4, SLOT);
    let mut tree = PrefixTree::new(arena.clone());
    let root = tree.root();
    let big = tree.reserve(root, &tokens(0, 3000), 2048, None, 3).unwrap();
    tree.publish(big);
    let small = add(&mut tree, root, &tokens(10_000, 3000), 2048);
    assert_eq!(arena.leased(), 4);
    let held = tree.lookup(&tokens(10_000, 3000), None).unwrap();
    // Two slots needed, one free: the big node is the only evictable one and gives three.
    let next = tree.reserve(root, &tokens(20_000, 3000), 2048, None, 2).unwrap();
    assert_eq!(tree.evictions(), 1);
    assert_eq!(arena.leased(), 3, "one pinned node and the new two");
    assert!(tree.get(big).is_none());
    tree.release(held);
    assert!(tree.get(small).is_some());
    tree.abort(next);
}

#[test]
fn evicts_several_nodes_for_a_large_reservation() {
    let arena = arena(4, SLOT);
    let mut tree = PrefixTree::new(arena.clone());
    let root = tree.root();
    for i in 0..4 {
        add(&mut tree, root, &tokens(i * 10_000, 3000), 2048);
    }
    let node = tree.reserve(root, &tokens(90_000, 3000), 2048, None, 3).unwrap();
    assert_eq!(tree.evictions(), 3);
    assert_eq!(tree.size(), 1);
    assert_eq!(tree.node(node).slots().len(), 3);
    assert_eq!(arena.leased(), 4);
}

#[test]
fn a_failed_reservation_returns_the_slots_it_took() {
    let arena = arena(3, SLOT);
    let mut tree = PrefixTree::new(arena.clone());
    let root = tree.root();
    add(&mut tree, root, &tokens(0, 3000), 2048);
    let pinned = tree.lookup(&tokens(0, 3000), None).unwrap();
    assert_eq!(arena.leased(), 1);
    assert_eq!(
        tree.reserve(root, &tokens(10_000, 3000), 2048, None, 3),
        Err(ReserveError::NoRoom),
        "two slots free, one pinned"
    );
    assert_eq!(arena.leased(), 1, "the two it took went back");
    tree.release(pinned);
}

#[test]
fn a_larger_than_the_arena_reservation_evicts_nothing() {
    let arena = arena(2, SLOT);
    let mut tree = PrefixTree::new(arena);
    let root = tree.root();
    add(&mut tree, root, &tokens(0, 3000), 2048);
    assert_eq!(
        tree.reserve(root, &tokens(10_000, 3000), 2048, None, 3),
        Err(ReserveError::NoRoom)
    );
    assert_eq!(tree.evictions(), 0);
    assert_eq!(tree.size(), 1);
}

#[test]
fn a_reused_table_entry_does_not_answer_to_a_stale_id() {
    let mut tree = tree(1);
    let root = tree.root();
    let first = add(&mut tree, root, &tokens(0, 3000), 2048);
    let second = add(&mut tree, root, &tokens(10_000, 3000), 2048);
    assert_eq!(tree.evictions(), 1);
    assert_ne!(first, second);
    assert!(tree.get(first).is_none());
    assert!(tree.get(second).is_some());
}

#[test]
fn dropping_the_tree_returns_every_slot() {
    let arena = arena(4, SLOT);
    let mut tree = PrefixTree::new(arena.clone());
    let root = tree.root();
    add(&mut tree, root, &tokens(0, 3000), 2048);
    add(&mut tree, root, &tokens(10_000, 3000), 2048);
    assert_eq!(arena.leased(), 2);
    drop(tree);
    assert_eq!(arena.leased(), 0);
}
