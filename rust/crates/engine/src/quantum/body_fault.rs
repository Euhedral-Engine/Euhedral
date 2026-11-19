//! A test hook: one graph body fails or panics in the middle of its launches, and the life of its hold is logged.

use std::sync::atomic::{AtomicU8, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};

/// How the armed body ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyFault {
    /// A launch refuses the quantum.
    Refuse,
    /// A launch panics.
    Panic,
}

/// Which sighting of its key the armed quantum is, as the graph cache sees it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Sighting {
    /// The first: the body runs directly, as launches on the lane.
    First,
    /// The second: the cache records the body, and the recording pass runs it.
    Recording,
    /// A later one, with a recording in the cache: the quantum is a replay, which runs no body. The fault applies after
    /// the replay queued its work.
    Replay,
}

/// What the armed fault is applied to, taken for one quantum.
#[derive(Clone, Debug)]
pub(crate) struct Planned {
    pub fault: BodyFault,
    pub sighting: Sighting,
    /// A test's value the body keeps alive in the hold of its run.
    pub token: Option<Arc<()>>,
    /// The passes of the body to leave alone first: a probe pass the cache makes before it records.
    pub skip: u8,
}

/// What a quantum met when it asked the hook: whether its key is new to the hook, and the fault, if it is for this
/// sighting.
pub(crate) struct Met {
    pub newly_seen: bool,
    pub planned: Option<Planned>,
}

/// What happened to the hold of the quantum the fault was applied to, in the order it happened.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HoldEvent {
    /// The run's hold was stored in the quantum.
    Stored,
    /// The quantum retired: its device work ended.
    Retired,
    /// The quantum dropped, and its hold with it.
    Released,
}

/// The armed fault, and the log of the quantum it was applied to.
#[derive(Debug, Default)]
pub struct BodyFaults {
    armed: AtomicU8,
    /// The events by place in the order, 0 for none yet, else the event's number.
    events: [AtomicU8; 8],
    /// The count of the test's value at each event, in the same places.
    holds: [AtomicUsize; 8],
    next: AtomicUsize,
    /// The sighting the armed fault waits for.
    sighting: AtomicU8,
    /// The test's value a faulted body keeps alive; set once, by the first arming that gives one.
    token: OnceLock<Weak<()>>,
    /// The keys the hook has seen run, by a nonzero hash of each (0 is an empty place).
    seen: [AtomicU64; 32],
}

impl BodyFaults {
    /// The next graph body that runs for a key the cache has not seen ends as `fault`.
    pub fn arm(&self, fault: BodyFault) {
        self.arm_at(fault, Sighting::First, None);
    }

    /// The next quantum whose key is at `sighting` ends as `fault`. A body that runs keeps `token` alive in its hold
    /// (the count of its strong handles at each event is logged); a replay runs no body, so it keeps none.
    pub fn arm_at(&self, fault: BodyFault, sighting: Sighting, token: Option<&Arc<()>>) {
        if let Some(token) = token {
            let _ = self.token.set(Arc::downgrade(token));
        }
        self.sighting.store(sighting as u8, Ordering::Release);
        self.armed.store(fault as u8 + 1, Ordering::Release);
    }

    /// Puts the fault back after a quantum that nothing ran for.
    pub(crate) fn rearm(&self, planned: &Planned) {
        self.sighting.store(planned.sighting as u8, Ordering::Release);
        self.armed.store(planned.fault as u8 + 1, Ordering::Release);
    }

    /// The quantum with `key` asks the hook what sighting it is (`contained`: the cache holds a recording of it) and
    /// whether the armed fault is for it.
    pub(crate) fn meet(&self, key: u128, contained: bool) -> Met {
        let mark = (key as u64 ^ (key >> 64) as u64) | 1;
        let newly_seen = !contained && !self.seen.iter().any(|place| place.load(Ordering::Acquire) == mark);
        let sighting = if contained {
            Sighting::Replay
        } else if newly_seen {
            for place in &self.seen {
                if place
                    .compare_exchange(0, mark, Ordering::AcqRel, Ordering::Acquire)
                    .is_ok()
                {
                    break;
                }
            }
            Sighting::First
        } else {
            Sighting::Recording
        };
        let planned = (self.sighting.load(Ordering::Acquire) == sighting as u8)
            .then(|| match self.armed.swap(0, Ordering::AcqRel) {
                1 => Some(BodyFault::Refuse),
                2 => Some(BodyFault::Panic),
                _ => None,
            })
            .flatten()
            .map(|fault| Planned {
                fault,
                sighting,
                token: self.token.get().and_then(Weak::upgrade),
                skip: u8::from(sighting == Sighting::Recording),
            });
        Met { newly_seen, planned }
    }

    /// The quantum with `key` ran nothing (a busy lane): the cache did not see it.
    pub(crate) fn unsee(&self, key: u128) {
        let mark = (key as u64 ^ (key >> 64) as u64) | 1;
        for place in &self.seen {
            let _ = place.compare_exchange(mark, 0, Ordering::AcqRel, Ordering::Acquire);
        }
    }

    pub(crate) fn log(&self, event: HoldEvent) {
        let at = self.next.fetch_add(1, Ordering::AcqRel);
        if let Some(place) = self.events.get(at) {
            let count = self.token.get().map_or(0, Weak::strong_count);
            self.holds[at].store(count, Ordering::Release);
            place.store(event as u8 + 1, Ordering::Release);
        }
    }

    /// The strong handles of the test's value at each event of [`events`](Self::events), in the same order.
    pub fn held_counts(&self) -> Vec<usize> {
        self.events
            .iter()
            .zip(&self.holds)
            .filter(|(place, _)| place.load(Ordering::Acquire) != 0)
            .map(|(_, count)| count.load(Ordering::Acquire))
            .collect()
    }

    /// The events of the faulted quantum so far.
    pub fn events(&self) -> Vec<HoldEvent> {
        self.events
            .iter()
            .filter_map(|place| match place.load(Ordering::Acquire) {
                1 => Some(HoldEvent::Stored),
                2 => Some(HoldEvent::Retired),
                3 => Some(HoldEvent::Released),
                _ => None,
            })
            .collect()
    }
}
