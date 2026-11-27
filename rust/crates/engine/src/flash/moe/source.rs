//! Where a block's experts come from: the seam between the fetch and expert stages and the device cache.
//!
//! The stages ask an [`ExpertSource`] for an expert and are told, on a worker and without waiting, how it ended
//! ([`Arrived`]). The cache is the source ([`ExpertCache`]) and a lease is the held expert ([`ExpertLease`]); the seam
//! exists because a lease can be made by the cache alone, so that the stages' unit tests (an expert that arrives late,
//! out of order or never) drive them through a source of their own.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use euhedral_graph::Marker;

use crate::residency::expert_cache::{ExpertCache, ExpertLease, Outcome};
use crate::residency::experts::ExpertId;

/// An expert's record held in a device slot, which the slot's owner neither evicts nor refills while this is alive.
/// Dropping it gives the slot back; [`HeldExpert::fence_after`] first says which device work still reads it.
pub trait HeldExpert: Send {
    /// The device address of the record.
    fn device_address(&self) -> u64;

    /// The marker of the copy that is still writing the slot, when one is: work that reads the slot waits for it.
    fn ready(&self) -> Option<Marker>;

    /// Records that device work ending at `marker` reads the slot, so the next copy into it waits behind `marker`. Call
    /// it once, with the last work submitted that reads the slot, before dropping.
    fn fence_after(&mut self, marker: Marker);
}

impl HeldExpert for ExpertLease {
    fn device_address(&self) -> u64 {
        ExpertLease::device_address(self)
    }

    fn ready(&self) -> Option<Marker> {
        ExpertLease::ready(self)
    }

    fn fence_after(&mut self, marker: Marker) {
        ExpertLease::fence_after(self, marker);
    }
}

/// How a request for an expert ended.
pub enum Arrived {
    /// The expert is held.
    Held(Box<dyn HeldExpert>),
    /// The expert could not be had; the words say why.
    Failed(String),
    /// The request was cancelled before the expert arrived.
    Cancelled,
}

impl Debug for Arrived {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        match self {
            Self::Held(lease) => write!(formatter, "Held({:#x})", lease.device_address()),
            Self::Failed(why) => write!(formatter, "Failed({why})"),
            Self::Cancelled => formatter.write_str("Cancelled"),
        }
    }
}

/// What a source calls when a request ends: once, on any thread, and it must not wait.
pub type Arrival = Box<dyn FnOnce(Arrived) + Send>;

/// A source of experts.
pub trait ExpertSource: Send + Sync {
    /// Asks for `expert`; `arrival` is told how it ended. `scan` marks a prefill scan, which the source may refuse to
    /// keep; `cancel` ends the request when set. Refuses a request it cannot make, with `arrival` not called.
    fn fetch(&self, expert: ExpertId, scan: bool, cancel: Arc<AtomicBool>, arrival: Arrival) -> Result<(), String>;
}

impl ExpertSource for ExpertCache {
    fn fetch(&self, expert: ExpertId, scan: bool, cancel: Arc<AtomicBool>, arrival: Arrival) -> Result<(), String> {
        ExpertCache::fetch(
            self,
            expert,
            scan,
            cancel,
            Box::new(move |outcome| {
                arrival(match outcome {
                    Outcome::Leased(lease) => Arrived::Held(Box::new(lease)),
                    Outcome::Failed(error) => Arrived::Failed(error.to_string()),
                    Outcome::Cancelled => Arrived::Cancelled,
                })
            }),
        )
        .map_err(|error| error.to_string())
    }
}
