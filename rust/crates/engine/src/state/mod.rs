//! What a sequence owns on the device: its attention cache, its Gated DeltaNet state and its attention scratch.

pub mod attention;
pub mod gdn;
pub mod kv;
pub mod released;

use std::fmt::{self, Display};
use std::iter;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use euhedral_memory::Claim;
use thiserror::Error;

use euhedral_inference_cuda::{AllocationId, CudaError, CudaLibrary, KernelOps, PinnedPool};

use crate::error::EngineError;
use crate::model::ModelConfig;
use crate::residency::footprint::{DRAFT_TWIN_ROWS, MINIMUM_VERIFY_ROWS};
use crate::residency::planner::ResidencyPlan;
use crate::stages::SequenceRefs;
use crate::stages::attention::KvSource;
use crate::stages::sampling::{RowReadbackPool, SelectionPool};
use attention::AttentionScratch;
use gdn::GdnStates;
use kv::KvCache;
pub use kv::KvView;
use released::ReleasedBuffers;

/// A sequence of a runtime, the unit a step belongs to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct SequenceId(pub usize);

/// What the stages of a quantum read of a sequence: its GDN state and its attention scratch. Allocated once, at
/// full size, and shared by every quantum of the sequence; nothing about it changes while quanta run, except the
/// contents of the buffers the stages write.
#[derive(Debug)]
pub struct SequenceDevice {
    pub gdn: GdnStates,
    pub scratch: AttentionScratch,
}

impl SequenceDevice {
    /// The allocation serials of every buffer the stages of a quantum read or write here, in a fixed order: what a
    /// captured graph of the sequence is keyed on.
    pub fn generations(&self, out: &mut Vec<AllocationId>) {
        self.gdn.generations(out);
        self.scratch.generations(out);
    }
}

/// A quantum's cache was not reserved exactly once since the quantum before it.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ReserveEpochError {
    #[error("the sequence has no {0:?} cache")]
    NoCache(KvSource),
    #[error(
        "the {cache:?} cache was reserved {calls} times for this quantum, not once: a view says whether its reservation grew only for the latest call"
    )]
    Calls { cache: KvSource, calls: u64 },
}

/// The quanta of one sequence that are in flight: admitted, and not yet dropped by the owner after their retirement.
#[derive(Default)]
pub struct Flight {
    count: AtomicUsize,
    /// Set by the owner that waits for the count to reach zero ([`Flight::settled`]): the guard that brings it there
    /// clears it and calls `wake`.
    awaited: AtomicBool,
    /// What tells the owner that the last guard was dropped; set once, by the owner.
    wake: OnceLock<Box<dyn Fn() + Send + Sync>>,
}

impl fmt::Debug for Flight {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Flight")
            .field("count", &self.count())
            .finish_non_exhaustive()
    }
}

impl Flight {
    /// Counts one more quantum in flight until the guard is dropped.
    pub fn begin(self: &Arc<Flight>) -> FlightGuard {
        self.count.fetch_add(1, Ordering::AcqRel);
        FlightGuard {
            flight: Arc::clone(self),
        }
    }

    /// Quanta in flight now.
    pub fn count(&self) -> usize {
        self.count.load(Ordering::Acquire)
    }

    /// Sets what is called when the last guard is dropped after [`Flight::settled`] answered no. Once.
    pub fn on_settled(&self, wake: impl Fn() + Send + Sync + 'static) {
        let first = self.wake.set(Box::new(wake)).is_ok();
        debug_assert!(first, "a flight's wake is set once, by its owner");
    }

    /// Whether nothing is in flight. When something is, the owner is woken once, by the drop of the last guard, and asks
    /// again. The flag is set before the count is read and the guard decrements before it reads the flag (all
    /// sequentially consistent), so one of the two sees the other: the wake is never lost.
    pub fn settled(&self) -> bool {
        self.awaited.store(true, Ordering::SeqCst);
        if self.count.load(Ordering::SeqCst) == 0 {
            self.awaited.store(false, Ordering::SeqCst);
            return true;
        }
        false
    }
}

/// One quantum in flight on a sequence ([`Flight::begin`]); the quantum holds it until it is dropped.
#[derive(Debug)]
pub struct FlightGuard {
    flight: Arc<Flight>,
}

impl Clone for FlightGuard {
    fn clone(&self) -> Self {
        self.flight.begin()
    }
}

impl Drop for FlightGuard {
    fn drop(&mut self) {
        let flight = &self.flight;
        if flight.count.fetch_sub(1, Ordering::SeqCst) == 1
            && flight.awaited.swap(false, Ordering::SeqCst)
            && let Some(wake) = flight.wake.get()
        {
            wake();
        }
    }
}

/// The device state of one sequence, as its owner holds it.
///
/// The owner holds what changes between quanta: the attention caches, whose frontiers it moves and whose pages and
/// page tables it grows (`reserve`, `appended`, `commit`), and the rows the next quantum replays. A quantum gets
/// the shared [`SequenceDevice`] and a [`KvView`] of each cache, copied at its admission (`KvCache::view`).
///
/// Drop it only after its last quantum has retired and been dropped ([`SequenceState::in_flight`] is 0): a view keeps
/// the pages and the host arena alive, and the last release of the arena and the host region is a blocking free,
/// which must happen in the owner's teardown and not on a worker that drops the last quantum.
#[derive(Debug)]
pub struct SequenceState {
    /// The caches of the full-attention layers (the first `attention layers` of the plan's cache).
    pub kv: KvCache,
    /// The MTP layer's own cache, when the sequence drafts: it follows the base layers' at layer index 64.
    pub mtp_kv: Option<KvCache>,
    pub device: Arc<SequenceDevice>,
    /// The selection words the sequence's quanta take in turn.
    pub selections: SelectionPool,
    /// The pinned readbacks a sampled quantum's logits row is copied to.
    pub row_readbacks: RowReadbackPool,
    flight: Arc<Flight>,
    /// The reserve epoch of each cache at the last [`SequenceState::refs`] that appended to it.
    claimed: [AtomicU64; 2],
    /// The sequence's bytes in the engine's memory account, released when the sequence is dropped (declared last).
    claims: Vec<Claim>,
}

impl SequenceState {
    /// Allocates the state `plan` sized: the cache with its host pages, the GDN state (zeroed through `kernels`, on the owner's lane
    /// before any quantum), the decode scratch and, for a drafting sequence, the draft buffers and the MTP layer's
    /// cache. Pages and page tables are freed through `released` when their last holder lets go. Waits on the allocators:
    /// startup only. The caller waits for the zeroing's marker.
    pub fn allocate(
        library: &Arc<CudaLibrary>,
        released: &Arc<ReleasedBuffers>,
        config: &ModelConfig,
        plan: &ResidencyPlan,
        kernels: &mut dyn KernelOps,
    ) -> Result<SequenceState, EngineError> {
        let drafts = plan.draft_depth > 0;
        let verified_rows = plan.verified_rows();
        let kv = KvCache::new(
            library,
            released,
            config.attention_layers(),
            config.kv_width() as u64,
            plan.kv_format,
            plan.kv_device_tokens as u64,
            plan.kv_host_tokens as u64,
        )
        .map_err(|error| EngineError::Runtime(error.to_string()))?;
        let mtp_kv = if drafts {
            Some(
                KvCache::new(
                    library,
                    released,
                    1,
                    config.kv_width() as u64,
                    plan.kv_format,
                    plan.max_context_tokens as u64,
                    0,
                )
                .map_err(|error| EngineError::Runtime(error.to_string()))?,
            )
        } else {
            None
        };
        let gdn = GdnStates::allocate(library, config, verified_rows, kernels)?;
        // A drafting sequence's catch-up runs up to the decode row twins' rows on the same scratch.
        let decode_rows = if drafts {
            verified_rows.max(DRAFT_TWIN_ROWS)
        } else {
            verified_rows.max(1)
        };
        let scratch = AttentionScratch::allocate(
            library,
            config,
            decode_rows,
            drafts,
            plan.rows.rows.max(MINIMUM_VERIFY_ROWS),
        )?;
        // A step's quanta that select (a catch-up, the recursive drafts and the verification) are all admitted before
        // the first retires, and a decode chain keeps one more behind the one it follows.
        let selections = SelectionPool::new(library, (plan.draft_depth + 2).max(2), (verified_rows as usize).max(1))
            .map_err(|error| EngineError::Runtime(error.to_string()))?;
        // A sampled request keeps one quantum in flight: the readback is the owner's from its admission until the host
        // has selected from the row, and the next quantum waits for that. The rows are those of a verification.
        let row_readbacks = RowReadbackPool::new(library, 1, (verified_rows as usize).max(1), config.vocab_size)
            .map_err(|error| EngineError::Runtime(error.to_string()))?;
        Ok(
            SequenceState::from_parts(kv, mtp_kv, Arc::new(SequenceDevice { gdn, scratch }), selections)
                .with_row_readbacks(row_readbacks),
        )
    }

    /// The state of the given parts, with no quantum in flight.
    pub fn from_parts(
        kv: KvCache,
        mtp_kv: Option<KvCache>,
        device: Arc<SequenceDevice>,
        selections: SelectionPool,
    ) -> SequenceState {
        SequenceState {
            kv,
            mtp_kv,
            device,
            selections,
            row_readbacks: RowReadbackPool::none(),
            flight: Arc::new(Flight::default()),
            claimed: [AtomicU64::new(0), AtomicU64::new(0)],
            claims: Vec::new(),
        }
    }

    /// The sequence, holding `claim` for some of its bytes until it is dropped.
    pub fn with_claim(mut self, claim: Claim) -> SequenceState {
        self.claims.push(claim);
        self
    }

    /// The sequence with `row_readbacks` for the quanta of sampling requests.
    pub fn with_row_readbacks(mut self, row_readbacks: RowReadbackPool) -> SequenceState {
        self.row_readbacks = row_readbacks;
        self
    }

    /// The references a quantum of this sequence reads: the shared device part and a view of each cache as it is now,
    /// for a quantum that appends to the cache `appends_to`. The quantum is counted in flight until its references
    /// are dropped.
    ///
    /// The owner reserves the rows of every quantum with one [`KvCache::reserve`] call (it allocates nothing when
    /// the rows are already backed) and takes the references after it: a view says whether its reservation had to
    /// grow ([`KvView::grew_for`]), and so whether a graph may repeat the quantum, only for the latest call. This
    /// refuses a quantum whose cache was reserved twice, or not at all, since the references of the quantum before.
    pub fn refs(&self, appends_to: KvSource) -> Result<SequenceRefs, ReserveEpochError> {
        let (cache, claimed) = match appends_to {
            KvSource::Base => (Some(&self.kv), &self.claimed[0]),
            KvSource::Mtp => (self.mtp_kv.as_ref(), &self.claimed[1]),
        };
        let cache = cache.ok_or(ReserveEpochError::NoCache(appends_to))?;
        let epoch = cache.epoch();
        let before = claimed.swap(epoch, Ordering::AcqRel);
        if epoch != before + 1 {
            return Err(ReserveEpochError::Calls {
                cache: appends_to,
                calls: epoch.wrapping_sub(before),
            });
        }
        let mut refs = SequenceRefs::new(Arc::clone(&self.device))
            .with_kv(self.kv.view())
            .in_flight(self.flight.begin());
        if let Some(mtp) = &self.mtp_kv {
            refs = refs.with_mtp_kv(mtp.view());
        }
        Ok(refs)
    }

    /// Makes the sequence one that has seen no token: the caches are empty (their pages stay reserved) and the GDN
    /// state is zeroed through `kernels`, on the owner's lane. The owner calls it between requests, with no quantum in flight, and
    /// orders the sequence's next quantum behind the marker of the work that queued it.
    pub fn reset(&mut self, kernels: &mut dyn KernelOps) -> Result<(), EngineError> {
        if self.in_flight() > 0 {
            return Err(EngineError::Runtime(format!(
                "a sequence with {} quanta in flight cannot be reset",
                self.in_flight()
            )));
        }
        let fail = |error: &dyn Display| EngineError::Runtime(format!("resetting the sequence: {error}"));
        for cache in iter::once(&mut self.kv).chain(self.mtp_kv.as_mut()) {
            cache.discard();
            cache.truncate(0).map_err(|error| fail(&error))?;
        }
        self.device.gdn.zero(kernels).map_err(|error| fail(&error))
    }

    /// Ends the sequence: frees its caches, the one place their blocking frees may happen. Refused, giving the state
    /// back, while quanta are in flight: the owner closes it after the last quantum has retired and been dropped.
    pub fn close(self) -> Result<(), SequenceBusy> {
        let in_flight = self.in_flight();
        if in_flight > 0 {
            return Err(SequenceBusy {
                state: Box::new(self),
                in_flight,
            });
        }
        drop(self);
        Ok(())
    }

    /// Quanta admitted on this sequence and not yet dropped.
    pub fn in_flight(&self) -> usize {
        self.flight.count()
    }

    /// Whether no quantum is in flight. If one is, `wake` (set by [`SequenceState::on_settled`]) is called once when
    /// the last of them is dropped, whatever thread drops it: the owner that ends a request asks again then.
    pub fn settled(&self) -> bool {
        self.flight.settled()
    }

    /// The sequence's count of quanta in flight.
    pub fn flight(&self) -> &Flight {
        &self.flight
    }

    /// Sets what is called when the last quantum in flight is dropped after [`SequenceState::settled`] answered no.
    /// Once; a second call changes nothing.
    pub fn on_settled(&self, wake: impl Fn() + Send + Sync + 'static) {
        self.flight.on_settled(wake);
    }

    /// Allocates the pinned staging the caches' reservations take (two sets per cache and per sequence of
    /// `sequences` of this shape: one in flight, one being reserved), so that no reservation on the request path
    /// allocates. The sets are held together, so the pool keeps that many. Startup only.
    pub fn prime_staging(&self, pool: &Arc<PinnedPool>, sequences: usize) -> Result<(), CudaError> {
        let copies = 2 * sequences.max(1);
        self.kv.prime_staging(pool, copies)?;
        if let Some(mtp) = &self.mtp_kv {
            mtp.prime_staging(pool, copies)?;
        }
        Ok(())
    }

    /// The device bytes the state holds now (the cache grows as it is reserved).
    pub fn device_bytes(&self) -> u64 {
        self.kv.device_bytes()
            + self.mtp_kv.as_ref().map_or(0, KvCache::device_bytes)
            + self.device.gdn.device_bytes()
            + self.device.scratch.device_bytes()
    }
}

/// A sequence cannot be closed while quanta are in flight.
#[derive(Debug)]
pub struct SequenceBusy {
    /// The sequence, untouched.
    pub state: Box<SequenceState>,
    /// Quanta in flight.
    pub in_flight: usize,
}

impl Drop for SequenceState {
    fn drop(&mut self) {
        let flying = self.flight.count();
        if flying > 0 {
            log::error!(
                "a sequence was dropped with {flying} quanta in flight: its host arena is freed by whichever worker \
                 drops the last quantum. Drop the state after the last quantum retired."
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_quantum_is_in_flight_until_its_guard_is_dropped() {
        let flight = Arc::new(Flight::default());
        assert_eq!(flight.count(), 0);
        let first = flight.begin();
        let second = first.clone();
        assert_eq!(
            flight.count(),
            2,
            "a clone of the references is the same quantum, counted again"
        );
        drop(first);
        assert_eq!(flight.count(), 1);
        drop(second);
        assert_eq!(flight.count(), 0, "the owner may free the caches");
    }

    #[test]
    fn an_owner_that_asks_while_a_quantum_flies_is_woken_once_by_the_last_drop() {
        let flight = Arc::new(Flight::default());
        let woken = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&woken);
        assert!(
            flight
                .wake
                .set(Box::new(move || {
                    counted.fetch_add(1, Ordering::SeqCst);
                }))
                .is_ok()
        );
        let first = flight.begin();
        let second = flight.begin();
        assert!(!flight.settled(), "two quanta fly");
        drop(first);
        assert_eq!(woken.load(Ordering::SeqCst), 0, "one still flies");
        drop(second);
        assert_eq!(woken.load(Ordering::SeqCst), 1, "the last drop woke the owner");
        assert!(flight.settled());
        let third = flight.begin();
        drop(third);
        assert_eq!(woken.load(Ordering::SeqCst), 1, "nobody asked: nobody is woken");
    }

    #[test]
    fn an_owner_that_asks_with_nothing_in_flight_is_not_woken_later() {
        let flight = Arc::new(Flight::default());
        let woken = Arc::new(AtomicUsize::new(0));
        let counted = Arc::clone(&woken);
        assert!(
            flight
                .wake
                .set(Box::new(move || {
                    counted.fetch_add(1, Ordering::SeqCst);
                }))
                .is_ok()
        );
        assert!(flight.settled());
        drop(flight.begin());
        assert_eq!(woken.load(Ordering::SeqCst), 0);
    }
}
