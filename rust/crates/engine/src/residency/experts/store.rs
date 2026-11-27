//! One load of an expert's record, as the cache owner asks it: the shard's directive says where it comes from.
//!
//! - **Hit**: the record is in a tier slot. One frame copies it out of host memory into the staging slot.
//! - **Fill**: the record is read from the artifact into the tier slot reserved for it, and each part is copied on to
//!   the staging slot as it arrives.
//! - **Bypass**: the record is read from the artifact into the staging slot alone.
//!
//! The load ends once, through `done`, with a [`LoadEnd`]: how it ended, and the [`TierSettle`] the owner applies to its
//! shard. The staging slot's leases are the load's, so the owner's free list needs no further record of it: a slot whose
//! write lease can be taken is free. Choosing the staging slot, and bounding the loads in flight, are the owner's.

use std::fmt::{self, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use super::arena::HostArena;
use super::layout::{ExpertError, ExpertId, ExpertLayout};
use super::source::{FileRecordSource, ReadLoad, ReadOutcome, Started};
use super::tier::{TierDirective, TierSettle};
use crate::frames::{Kind, step};

/// One load.
pub struct StoreLoad {
    pub id: ExpertId,
    /// What the shard planned for `id`.
    pub directive: TierDirective,
    /// The staging slot the record goes to.
    pub staging_slot: usize,
    /// Set by whoever no longer needs the record.
    pub cancel: Arc<AtomicBool>,
    /// Called once, from the frame that ended the load, with the leases already let go.
    pub done: Box<dyn FnOnce(LoadEnd) + Send>,
}

/// How a load ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadEnd {
    pub id: ExpertId,
    pub outcome: ReadOutcome,
    /// What the owner applies to its shard; `None` for a bypass.
    pub settle: Option<TierSettle>,
}

impl TierDirective {
    /// The settle of a load that planned this directive and ended as `outcome`.
    pub fn settle(&self, outcome: &ReadOutcome) -> Option<TierSettle> {
        match *self {
            TierDirective::Hit { slot } => Some(TierSettle::Used(slot)),
            TierDirective::Fill { slot } => Some(match outcome {
                ReadOutcome::Complete | ReadOutcome::Cancelled { filled: true } => TierSettle::Filled(slot),
                ReadOutcome::Lost => TierSettle::Lost(slot),
                ReadOutcome::Cancelled { filled: false } | ReadOutcome::Failed(_) => TierSettle::Abandoned(slot),
            }),
            TierDirective::Bypass => None,
        }
    }

    /// The settle of a load that planned this directive and was refused before it started.
    pub fn settle_unstarted(&self) -> Option<TierSettle> {
        self.settle(&ReadOutcome::Cancelled { filled: false })
    }
}

/// Loads records from the tier's memory or the artifact into staging. Shared by the owners; it has no state of its own
/// beyond what it was built from.
pub struct ExpertStore {
    layout: Arc<ExpertLayout>,
    source: FileRecordSource,
    /// The tier's slots; `None` when there is no tier (every load is a bypass).
    tier: Option<Arc<HostArena>>,
    staging: Arc<HostArena>,
}

impl ExpertStore {
    /// A store over `source`'s layout, with the tier's memory (if any) and the staging memory loads end in.
    pub fn new(source: FileRecordSource, tier: Option<Arc<HostArena>>, staging: Arc<HostArena>) -> ExpertStore {
        ExpertStore {
            layout: Arc::clone(source.layout()),
            source,
            tier,
            staging,
        }
    }

    pub fn layout(&self) -> &Arc<ExpertLayout> {
        &self.layout
    }

    pub fn source(&self) -> &FileRecordSource {
        &self.source
    }

    /// Starts `load`. Refuses an unknown id, a slot out of range or leased, or a directive that needs a tier the store
    /// lacks, before anything starts: `done` is then dropped uncalled, and the owner applies
    /// [`TierDirective::settle_unstarted`] to its shard.
    pub fn begin(&self, load: StoreLoad) -> Result<Started, ExpertError> {
        let StoreLoad {
            id,
            directive,
            staging_slot,
            cancel,
            done,
        } = load;
        let record = self.layout.record(id)?;
        let bytes = record.bytes as usize;
        let tier = |slot: usize| self.tier.as_ref().ok_or(ExpertError::SlotOutOfRange { slot, slots: 0 });
        let finish = move |outcome: ReadOutcome| {
            let settle = directive.settle(&outcome);
            done(LoadEnd { id, outcome, settle });
        };
        match directive {
            TierDirective::Hit { slot } => {
                let from = tier(slot)?.read(slot)?;
                let mut to = self.staging.write(staging_slot)?;
                if to.capacity() < bytes {
                    return Err(ExpertError::SlotTooSmall {
                        bank: self.layout.bank_name(id.bank).to_owned(),
                        expert: id.expert,
                        slot_bytes: to.capacity(),
                        needed: bytes,
                    });
                }
                // The copy is the one frame; `chain` orders the copies of one tier slot.
                Ok(Started::Frame(step(Kind::ExpertCopy, slot, move || {
                    let outcome = if cancel.load(Ordering::Acquire) {
                        ReadOutcome::Cancelled { filled: false }
                    } else {
                        match to.bytes_mut() {
                            Some(to_bytes) => {
                                to_bytes[..bytes].copy_from_slice(&from.bytes()[..bytes]);
                                ReadOutcome::Complete
                            }
                            // No part of a slot only this load leases can be out: refuse loudly if one is.
                            None => ReadOutcome::Failed(ExpertError::SlotBusy {
                                slot: staging_slot,
                                wanted: "copying",
                            }),
                        }
                    };
                    drop((from, to));
                    finish(outcome);
                })))
            }
            TierDirective::Fill { slot } => {
                let destination = tier(slot)?.write(slot)?;
                let copy_to = self.staging.write(staging_slot)?;
                self.source.begin(ReadLoad {
                    id,
                    destination,
                    copy_to: Some(copy_to),
                    cancel,
                    done: Box::new(finish),
                })
            }
            TierDirective::Bypass => {
                let destination = self.staging.write(staging_slot)?;
                self.source.begin(ReadLoad {
                    id,
                    destination,
                    copy_to: None,
                    cancel,
                    done: Box::new(finish),
                })
            }
        }
    }
}

impl fmt::Debug for StoreLoad {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("StoreLoad")
            .field("id", &self.id)
            .field("directive", &self.directive)
            .finish_non_exhaustive()
    }
}

impl fmt::Debug for ExpertStore {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("ExpertStore").finish_non_exhaustive()
    }
}
