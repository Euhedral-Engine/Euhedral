//! The contract of the device slots' bookkeeping, through its interface.

use std::sync::Arc;

use euhedral_graph::{LaneId, Marker, QueueId};

use super::CacheError;
use super::shard::{Claimed, Reserved, Shard};
use super::stats::{CacheStats, Cause};

fn marker(queue: QueueId, lane: u16, value: u64) -> Marker {
    Marker::new(queue, LaneId(lane), value)
}

fn shard(keys: usize, slots: usize) -> (Shard, Arc<CacheStats>) {
    let stats = Arc::new(CacheStats::default());
    let shard = Shard::new(keys, slots, Arc::clone(&stats)).unwrap_or_else(|error| panic!("{error}"));
    (shard, stats)
}

fn miss(claimed: Claimed) -> Reserved {
    match claimed {
        Claimed::Miss(reserved) => reserved,
        other => panic!("expected a miss, got {other:?}"),
    }
}

/// Loads `key` to residency and returns its lease as (slot, generation), pinned once.
fn load(shard: &mut Shard, queue: QueueId, key: usize) -> (u32, u64) {
    let reserved = miss(shard.claim(key, 100, None).unwrap());
    shard
        .submitted(reserved.slot, reserved.generation, marker(queue, 0, key as u64 + 1))
        .unwrap();
    shard.retired(reserved.slot, reserved.generation).unwrap();
    (reserved.slot, reserved.generation)
}

#[test]
fn a_cache_with_no_experts_or_no_slots_is_refused() {
    let stats = Arc::new(CacheStats::default());
    assert!(matches!(
        Shard::new(0, 4, Arc::clone(&stats)),
        Err(CacheError::Keys { keys: 0, .. })
    ));
    assert!(matches!(
        Shard::new(4, 0, stats),
        Err(CacheError::Slots { slots: 0, .. })
    ));
}

#[test]
fn a_key_outside_the_cache_is_refused_naming_it() {
    let (mut shard, _) = shard(3, 2);
    assert_eq!(shard.claim(3, 10, None), Err(CacheError::Key { key: 3, keys: 3 }));
}

#[test]
fn a_miss_then_a_hit_are_counted_and_the_hit_is_a_lease_at_once() {
    let queue = QueueId::next();
    let (mut shard, stats) = shard(8, 4);
    let (slot, generation) = load(&mut shard, queue, 5);
    shard.release(slot, generation, None);
    assert_eq!(
        shard.claim(5, 100, None).unwrap(),
        Claimed::Resident { slot, generation }
    );
    let snapshot = stats.snapshot();
    assert_eq!((snapshot.misses, snapshot.hits, snapshot.transfers), (1, 1, 1));
    shard.check_invariants().unwrap();
}

#[test]
fn a_claim_that_finds_the_load_waits_for_its_copy_then_joins_with_its_marker() {
    let queue = QueueId::next();
    let (mut shard, stats) = shard(8, 4);
    let reserved = miss(shard.claim(2, 100, None).unwrap());
    assert_eq!(shard.claim(2, 100, None).unwrap(), Claimed::Wait(Cause::Copying));
    let ready = marker(queue, 1, 7);
    shard.submitted(reserved.slot, reserved.generation, ready).unwrap();
    assert_eq!(
        shard.claim(2, 100, None).unwrap(),
        Claimed::Joined {
            slot: reserved.slot,
            generation: reserved.generation,
            ready
        }
    );
    assert_eq!(stats.snapshot().coalesced, 1);
    // Two leases are out: the load's own and the joiner's.
    assert_eq!(shard.leases_open(), 2);
    shard.check_invariants().unwrap();
}

#[test]
fn a_slot_that_is_loading_or_leased_is_never_evicted() {
    let queue = QueueId::next();
    let (mut shard, _) = shard(16, 3);
    let first = load(&mut shard, queue, 0);
    let second = load(&mut shard, queue, 1);
    // The third slot is loading; the first two are leased (their loads' own leases are open).
    let third = miss(shard.claim(2, 100, None).unwrap());
    assert_eq!(shard.claim(3, 100, None).unwrap(), Claimed::Wait(Cause::Slots));
    // One lease closes: only that slot may be taken.
    shard.release(second.0, second.1, None);
    let taken = miss(shard.claim(3, 100, None).unwrap());
    assert_eq!(taken.slot, second.0);
    assert_ne!(taken.slot, first.0);
    assert_ne!(taken.slot, third.slot);
    assert!(!shard.holds(1));
    shard.check_invariants().unwrap();
}

#[test]
fn a_slot_whose_copy_has_not_retired_is_not_evictable_once_its_leases_are_let_go() {
    let queue = QueueId::next();
    let (mut shard, _) = shard(8, 1);
    let reserved = miss(shard.claim(0, 100, None).unwrap());
    shard
        .submitted(reserved.slot, reserved.generation, marker(queue, 0, 1))
        .unwrap();
    shard.release(reserved.slot, reserved.generation, None);
    assert_eq!(shard.evictable(), 0);
    assert_eq!(shard.claim(1, 100, None).unwrap(), Claimed::Wait(Cause::Slots));
    shard.retired(reserved.slot, reserved.generation).unwrap();
    assert_eq!(shard.evictable(), 1);
    assert!(matches!(shard.claim(1, 100, None).unwrap(), Claimed::Miss(_)));
}

#[test]
fn a_miss_the_caller_cannot_start_waits_for_the_cause_it_names_and_changes_nothing() {
    let (mut shard, stats) = shard(8, 2);
    assert_eq!(
        shard.claim(1, 100, Some(Cause::Staging)).unwrap(),
        Claimed::Wait(Cause::Staging)
    );
    assert_eq!(
        shard.claim(1, 100, Some(Cause::Reads)).unwrap(),
        Claimed::Wait(Cause::Reads)
    );
    assert!(!shard.holds(1));
    assert_eq!(stats.snapshot().misses, 0);
    shard.check_invariants().unwrap();
}

#[test]
fn a_resident_is_leased_even_when_loads_may_not_start() {
    let queue = QueueId::next();
    let (mut shard, _) = shard(8, 2);
    let (slot, generation) = load(&mut shard, queue, 4);
    shard.release(slot, generation, None);
    assert!(matches!(
        shard.claim(4, 100, Some(Cause::Staging)).unwrap(),
        Claimed::Resident { .. }
    ));
}

#[test]
fn what_was_asked_for_again_survives_what_was_loaded_once() {
    let queue = QueueId::next();
    let (mut shard, _) = shard(64, 10);
    // Ten residents; key 0 is asked for again.
    for key in 0..10 {
        let (slot, generation) = load(&mut shard, queue, key);
        shard.release(slot, generation, None);
    }
    let Claimed::Resident { slot, generation } = shard.claim(0, 100, None).unwrap() else {
        panic!("key 0 is resident")
    };
    shard.release(slot, generation, None);
    // Ten more one-time keys pass through: key 0 stays, the others it loaded with are replaced.
    for key in 10..19 {
        let (slot, generation) = load(&mut shard, queue, key);
        shard.release(slot, generation, None);
    }
    assert!(shard.holds(0));
    assert!(!shard.holds(1));
    shard.check_invariants().unwrap();
}

#[test]
fn a_record_evicted_unused_returns_to_the_main_queue() {
    let queue = QueueId::next();
    let (mut shard, _) = shard(64, 10);
    for key in 0..11 {
        let (slot, generation) = load(&mut shard, queue, key);
        shard.release(slot, generation, None);
    }
    // Key 0 was evicted unused for key 10; its return is remembered, so it joins the main queue and the next one-time
    // keys replace the small queue's, not it.
    assert!(!shard.holds(0));
    let (slot, generation) = load(&mut shard, queue, 0);
    shard.release(slot, generation, None);
    for key in 11..16 {
        let (slot, generation) = load(&mut shard, queue, key);
        shard.release(slot, generation, None);
    }
    assert!(shard.holds(0));
    shard.check_invariants().unwrap();
}

#[test]
fn the_device_work_that_read_a_slot_is_waited_for_by_the_next_copy_into_it() {
    let queue = QueueId::next();
    let (mut shard, _) = shard(8, 1);
    let (slot, generation) = load(&mut shard, queue, 0);
    shard.release(slot, generation, Some(marker(queue, 2, 5)));
    // A second reader on the same lane later: the later marker stands for both. One on another lane is kept beside it.
    let Claimed::Resident { slot, generation } = shard.claim(0, 100, None).unwrap() else {
        panic!("the expert is resident")
    };
    shard.release(slot, generation, Some(marker(queue, 2, 9)));
    assert_eq!(shard.fence_of(slot), [marker(queue, 2, 9)]);
    let Claimed::Resident { slot, generation } = shard.claim(0, 100, None).unwrap() else {
        panic!("the expert is resident")
    };
    shard.release(slot, generation, Some(marker(queue, 3, 1)));
    let reserved = miss(shard.claim(1, 100, None).unwrap());
    assert_eq!(reserved.fences, vec![marker(queue, 2, 9), marker(queue, 3, 1)]);
}

#[test]
fn a_load_that_never_submitted_gives_the_slot_back_with_its_fences() {
    let queue = QueueId::next();
    let (mut shard, stats) = shard(8, 1);
    let (slot, generation) = load(&mut shard, queue, 0);
    shard.release(slot, generation, Some(marker(queue, 1, 3)));
    let reserved = miss(shard.claim(1, 100, None).unwrap());
    assert_eq!(reserved.fences, vec![marker(queue, 1, 3)]);
    shard.abandon(reserved.slot, reserved.generation).unwrap();
    assert!(!shard.holds(1));
    assert!(!shard.holds(0));
    let retried = miss(shard.claim(1, 100, None).unwrap());
    assert_eq!(retried.fences, vec![marker(queue, 1, 3)]);
    assert_eq!(stats.snapshot().evictions, 1);
    shard.check_invariants().unwrap();
}

#[test]
fn a_load_is_ended_once_and_only_as_the_state_it_is_in() {
    let queue = QueueId::next();
    let (mut shard, _) = shard(8, 2);
    let reserved = miss(shard.claim(0, 100, None).unwrap());
    assert!(matches!(
        shard.retired(reserved.slot, reserved.generation),
        Err(CacheError::BadLoad { .. })
    ));
    shard
        .submitted(reserved.slot, reserved.generation, marker(queue, 0, 1))
        .unwrap();
    assert!(matches!(
        shard.submitted(reserved.slot, reserved.generation, marker(queue, 0, 2)),
        Err(CacheError::BadLoad { .. })
    ));
    assert!(matches!(
        shard.abandon(reserved.slot, reserved.generation),
        Err(CacheError::BadLoad { .. })
    ));
    shard.retired(reserved.slot, reserved.generation).unwrap();
    assert!(matches!(
        shard.retired(reserved.slot, reserved.generation),
        Err(CacheError::BadLoad { .. })
    ));
    assert!(matches!(shard.retired(9, 1), Err(CacheError::BadLoad { slot: 9, .. })));
}

#[test]
fn closing_refuses_while_a_load_is_in_flight_then_forces_the_leases_shut() {
    let queue = QueueId::next();
    let (mut shard, stats) = shard(8, 2);
    let (slot, generation) = load(&mut shard, queue, 0);
    let reserved = miss(shard.claim(1, 100, None).unwrap());
    assert_eq!(shard.close(), Err(CacheError::LoadsInFlight { loads: 1 }));
    shard.abandon(reserved.slot, reserved.generation).unwrap();
    assert_eq!(shard.close(), Ok(1));
    assert_eq!(stats.snapshot().forced_leases, 1);
    // A lease of the closed shard is let go without effect, and nothing more is asked of it.
    shard.release(slot, generation, None);
    assert_eq!(shard.claim(0, 100, None), Err(CacheError::Closed));
}

#[test]
fn the_bookkeeping_holds_through_a_long_mixed_run() {
    let queue = QueueId::next();
    let (mut shard, _) = shard(40, 7);
    let mut open: Vec<(u32, u64)> = Vec::new();
    let mut loading: Vec<Reserved> = Vec::new();
    let mut seed = 0x2545_f491_4f6c_dd1du64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        seed
    };
    for step in 0..4000u64 {
        match next() % 5 {
            0 | 1 => match shard.claim((next() % 40) as usize, 64, None).unwrap() {
                Claimed::Resident { slot, generation } | Claimed::Joined { slot, generation, .. } => {
                    open.push((slot, generation));
                }
                Claimed::Miss(reserved) => loading.push(reserved),
                Claimed::Wait(_) => {}
            },
            2 => {
                if let Some(reserved) = loading.pop() {
                    let (slot, generation) = (reserved.slot, reserved.generation);
                    if next() % 4 == 0 {
                        shard.abandon(slot, generation).unwrap();
                    } else {
                        shard.submitted(slot, generation, marker(queue, 0, step + 1)).unwrap();
                        open.push((slot, generation));
                        shard.retired(slot, generation).unwrap();
                    }
                }
            }
            _ => {
                if !open.is_empty() {
                    let (slot, generation) = open.swap_remove((next() as usize) % open.len());
                    shard.release(slot, generation, None);
                }
            }
        }
        shard
            .check_invariants()
            .unwrap_or_else(|why| panic!("step {step}: {why}"));
    }
}
