use super::*;

fn at(address: u64) -> DevicePtr {
    // SAFETY: the tests never dereference it.
    unsafe { DevicePtr::from_raw(address) }
}

#[test]
fn the_state_holds_a_key_per_block_and_two_tails_of_three_keys_each() {
    // 2048 tokens: 512 blocks of 256 bytes, two tails of 3 x 256 bytes.
    assert_eq!(QsaState::bytes(2048), 512 * 256 + 2 * 3 * 256);
    // 10 tokens complete 2 blocks; the incomplete one is the tail's.
    assert_eq!(QsaState::bytes(10), 2 * 256 + 1536);
    // Fewer tokens than a block still hold one key.
    assert_eq!(QsaState::bytes(1), 256 + 1536);
    assert_eq!(max_blocks(3), 1);
    assert_eq!(max_blocks(4), 1);
    assert_eq!(max_blocks(11), 2);
}

#[test]
fn the_plans_formula_is_the_states_bytes_with_the_indexers_shape() {
    // Blocks of 4 tokens, one key of 128 BF16 values (256 bytes).
    for tokens in [1u32, 3, 10, 2048, 262_144] {
        assert_eq!(indexer_layer_bytes(u64::from(tokens), 4, 256), Some(QsaState::bytes(tokens)));
    }
    assert_eq!(indexer_layer_bytes(u64::MAX, 1, u64::MAX), None);
}

#[test]
fn buffers_are_aligned_disjoint_and_end_inside_the_size() {
    let state = QsaState::over(at(0x10000), 1000).unwrap();
    let buffers = state.buffers();
    assert_eq!(buffers.block_keys, at(0x10000));
    let keys_end = 0x10000 + 250 * 256;
    assert!(buffers.tails.get() >= keys_end);
    assert_eq!(buffers.tails.get() % ALIGNMENT, 0);
    assert!(buffers.tails.get() + 2 * QsaBuffers::TAIL_BYTES <= 0x10000 + QsaState::bytes(1000));
    assert_eq!(state.max_tokens(), 1000);
}

#[test]
fn a_base_a_kernel_cannot_read_or_no_tokens_are_refused_with_the_values() {
    assert!(QsaState::over(at(0x10008), 100).unwrap_err().to_string().contains("0x10008"));
    assert!(QsaState::over(DevicePtr::NULL, 100).is_err());
    assert!(QsaState::over(at(0x10000), 0).unwrap_err().to_string().contains("0 tokens"));
}

#[test]
fn a_chunk_reads_the_committed_tail_and_a_commit_swaps_it() {
    let mut state = QsaState::over(at(0x10000), 64).unwrap();
    assert_eq!(state.begin_chunk(), 0);
    state.commit();
    assert_eq!(state.begin_chunk(), 1);
    state.commit();
    assert_eq!(state.begin_chunk(), 0);
}

#[test]
fn the_chunks_of_one_quantum_chain_through_the_two_tails() {
    let mut state = QsaState::over(at(0x10000), 64).unwrap();
    // Chunk k reads what chunk k - 1 wrote: the slots alternate.
    assert_eq!([state.begin_chunk(), state.begin_chunk(), state.begin_chunk()], [0, 1, 0]);
    // Three chunks wrote slot 1 last (0 -> 1 -> 0 -> 1): that is the committed tail.
    state.commit();
    assert_eq!(state.begin_chunk(), 1);
    // Two chunks leave the committed tail where it was.
    let mut state = QsaState::over(at(0x10000), 64).unwrap();
    state.begin_chunk();
    state.begin_chunk();
    state.commit();
    assert_eq!(state.begin_chunk(), 0);
}

#[test]
fn a_discarded_chunk_leaves_the_committed_tail_and_a_commit_with_nothing_pending_changes_nothing() {
    let mut state = QsaState::over(at(0x10000), 64).unwrap();
    state.begin_chunk();
    state.commit();
    state.commit();
    state.begin_chunk();
    state.discard();
    // Slot 1 is committed: the discarded chunk wrote slot 0.
    assert_eq!(state.begin_chunk(), 1);
}

#[test]
fn reset_refuses_while_a_chunk_is_in_flight_and_then_starts_over() {
    let mut state = QsaState::over(at(0x10000), 64).unwrap();
    state.begin_chunk();
    state.commit();
    state.begin_chunk();
    assert!(state.reset().unwrap_err().to_string().contains("1 chunks are in flight"));
    state.discard();
    state.reset().unwrap();
    assert_eq!(state.begin_chunk(), 0);
}
