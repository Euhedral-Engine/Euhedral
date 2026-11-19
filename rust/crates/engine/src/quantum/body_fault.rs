//! A test hook: one graph body fails or panics in the middle of its launches, and the life of its hold is logged.

use std::sync::atomic::{AtomicU8, AtomicUsize, Ordering};

/// How the armed body ends.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BodyFault {
    /// A launch refuses the quantum.
    Refuse,
    /// A launch panics.
    Panic,
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
    next: AtomicUsize,
}

impl BodyFaults {
    /// The next graph body that runs for a key the cache has not seen ends as `fault`.
    pub fn arm(&self, fault: BodyFault) {
        self.armed.store(fault as u8 + 1, Ordering::Release);
    }

    pub(crate) fn take(&self) -> Option<BodyFault> {
        match self.armed.swap(0, Ordering::AcqRel) {
            1 => Some(BodyFault::Refuse),
            2 => Some(BodyFault::Panic),
            _ => None,
        }
    }

    pub(crate) fn log(&self, event: HoldEvent) {
        let at = self.next.fetch_add(1, Ordering::AcqRel);
        if let Some(place) = self.events.get(at) {
            place.store(event as u8 + 1, Ordering::Release);
        }
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
