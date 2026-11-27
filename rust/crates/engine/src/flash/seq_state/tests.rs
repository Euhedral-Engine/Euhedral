use super::*;
use crate::flash::buffers::ALIGNMENT;

fn at(address: u64) -> DevicePtr {
    // SAFETY: a test address, never dereferenced.
    unsafe { DevicePtr::from_raw(address) }
}

fn spec() -> SequenceSpec {
    SequenceSpec {
        layers: vec![
            LayerSizes::GatedDeltaNet {
                history_bytes: 1000,
                recurrent_bytes: 5000,
            },
            LayerSizes::Attention,
            LayerSizes::GatedDeltaNet {
                history_bytes: 1000,
                recurrent_bytes: 5000,
            },
            LayerSizes::Attention,
        ],
        max_tokens: 64,
        ple_history_bytes: 300,
    }
}

fn tracker(max_tokens: u32) -> Tracker {
    let layout = SequenceLayout::over(&SequenceSpec { max_tokens, ..spec() }, at(0x10_0000)).unwrap();
    Tracker::new(max_tokens, vec![7; 3].into(), layout.attention().collect()).unwrap()
}

#[test]
fn the_layout_hands_out_aligned_disjoint_ranges_inside_its_bytes() {
    let spec = spec();
    let base = at(0x10_0000);
    let layout = SequenceLayout::over(&spec, base).unwrap();
    assert_eq!(layout.bytes(), spec.bytes());
    let mut starts = Vec::new();
    for state in (0..4).filter_map(|index| layout.layer(index)) {
        match state {
            LayerState::Gdn(gdn) => {
                starts.push((gdn.conv_history.get(), 1000));
                starts.push((gdn.recurrent.get(), 5000));
            }
            LayerState::Attention(qsa) => {
                let buffers = qsa.buffers();
                starts.push((buffers.block_keys.get(), 1));
                starts.push((
                    buffers.tails.get(),
                    2 * super::super::qsa::state::QsaBuffers::TAIL_BYTES,
                ));
            }
        }
    }
    starts.push((layout.ple_history().get(), 300));
    starts.sort_unstable();
    for (address, _) in &starts {
        assert_eq!(address % ALIGNMENT, 0);
    }
    for pair in starts.windows(2) {
        assert!(pair[0].0 + pair[0].1 <= pair[1].0, "{pair:?} overlap");
    }
    let (last, size) = starts[starts.len() - 1];
    assert!(last + size <= base.get() + layout.bytes());
}

#[test]
fn a_layout_of_no_positions_or_over_an_unaligned_base_is_refused() {
    let none = SequenceSpec {
        max_tokens: 0,
        ..spec()
    };
    assert!(SequenceLayout::over(&none, at(0x10_0000)).is_err());
    assert!(SequenceLayout::over(&spec(), at(0x10_0010)).is_err());
    assert!(SequenceLayout::over(&spec(), DevicePtr::NULL).is_err());
}

#[test]
fn the_position_and_the_context_advance_only_on_a_commit() {
    let mut tracker = tracker(64);
    let first = tracker.begin(16).unwrap();
    assert_eq!((first.start, tracker.position()), (0, 0));
    assert_eq!(tracker.commit(&[1, 2, 3]).unwrap(), 16);
    assert_eq!((tracker.position(), tracker.context()), (16, &[1, 2, 3][..]));
    let second = tracker.begin(1).unwrap();
    assert_eq!(second.start, 16);
}

#[test]
fn a_discarded_chunk_leaves_the_position_context_and_tail_as_they_were() {
    let mut tracker = tracker(64);
    tracker.begin(4).unwrap();
    tracker.commit(&[1, 2, 3]).unwrap();
    let before = tracker.begin(4).unwrap();
    tracker.discard();
    assert!(!tracker.in_flight());
    assert_eq!((tracker.position(), tracker.context()), (4, &[1, 2, 3][..]));
    assert_eq!(tracker.begin(4).unwrap(), before);
}

#[test]
fn the_tail_slot_flips_with_each_commit() {
    let mut tracker = tracker(64);
    let first = tracker.begin(4).unwrap();
    tracker.commit(&[0, 0, 0]).unwrap();
    let second = tracker.begin(4).unwrap();
    assert_eq!(first.tail_slot ^ 1, second.tail_slot);
}

#[test]
fn a_second_chunk_in_flight_zero_rows_and_a_chunk_past_the_end_are_refused() {
    let mut tracker = tracker(8);
    assert!(tracker.begin(0).is_err());
    assert!(tracker.begin(9).is_err());
    tracker.begin(8).unwrap();
    assert!(tracker.begin(1).is_err());
    tracker.commit(&[0, 0, 0]).unwrap();
    assert!(tracker.begin(1).is_err(), "the sequence is full");
}

#[test]
fn a_commit_with_nothing_in_flight_or_a_context_of_another_length_is_refused() {
    let mut tracker = tracker(8);
    assert!(tracker.commit(&[0, 0, 0]).is_err());
    tracker.begin(2).unwrap();
    assert!(tracker.commit(&[0, 0]).is_err());
    assert!(tracker.in_flight(), "a refused commit leaves the chunk in flight");
}

#[test]
fn a_reset_returns_to_position_zero_and_is_refused_while_a_chunk_is_in_flight() {
    let mut tracker = tracker(8);
    tracker.begin(2).unwrap();
    assert!(tracker.reset(&[7, 7, 7]).is_err());
    tracker.commit(&[1, 1, 1]).unwrap();
    tracker.reset(&[7, 7, 7]).unwrap();
    assert_eq!((tracker.position(), tracker.context()), (0, &[7, 7, 7][..]));
    assert_eq!(tracker.begin(2).unwrap().tail_slot, 0);
    assert!(tracker.in_flight());
}

#[test]
fn a_tracker_of_no_positions_or_over_states_of_another_length_is_refused() {
    let layout = SequenceLayout::over(&spec(), at(0x10_0000)).unwrap();
    assert!(Tracker::new(0, Box::new([]), layout.attention().collect()).is_err());
    assert!(Tracker::new(32, Box::new([]), layout.attention().collect()).is_err());
}
