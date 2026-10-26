//! What a sequence owns on the device: its attention cache, its Gated DeltaNet state and its attention scratch.

pub mod attention;
pub mod gdn;
pub mod kv;

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_inference_cuda::{AllocationId, CudaError, CudaLibrary, PinnedPool, RetireList, Submission};

use crate::error::EngineError;
use crate::model::ModelConfig;
use crate::residency::footprint::{DRAFT_TWIN_ROWS, MINIMUM_VERIFY_ROWS};
use crate::residency::planner::ResidencyPlan;
use crate::stages::SequenceRefs;
use crate::stages::sampling::SelectionPool;
use attention::AttentionScratch;
use gdn::GdnStates;
use kv::KvCache;
pub use kv::{KvView, StaleReservation};

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

/// The quanta of one sequence that are in flight: admitted, and not yet dropped by the owner after their retirement.
#[derive(Debug, Default)]
pub struct Flight {
    count: AtomicUsize,
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
        self.flight.count.fetch_sub(1, Ordering::AcqRel);
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
    flight: Arc<Flight>,
}

impl SequenceState {
    /// Allocates the state `plan` sized: the cache with its host pages, the GDN state (zeroed on `submission`'s lane
    /// before any quantum), the decode scratch and, for a drafting sequence, the draft buffers and the MTP layer's
    /// cache. Pages and page tables are freed through `retire` when their last holder lets go. Waits on the allocators:
    /// startup only. The caller waits for the zeroing's marker.
    pub fn allocate(
        library: &Arc<CudaLibrary>,
        retire: &Arc<RetireList>,
        config: &ModelConfig,
        plan: &ResidencyPlan,
        submission: &Submission<'_>,
    ) -> Result<SequenceState, EngineError> {
        let drafts = plan.draft_depth > 0;
        let verified_rows = if drafts { 1 + plan.draft_depth as u64 } else { 0 };
        let kv = KvCache::new(
            library,
            retire,
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
                    retire,
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
        let gdn = GdnStates::allocate(library, config, verified_rows, submission)?;
        // A drafting sequence's catch-up runs up to the decode row twins' rows on the same scratch.
        let decode_rows = if drafts { verified_rows.max(DRAFT_TWIN_ROWS) } else { 1 };
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
        Ok(SequenceState::from_parts(
            kv,
            mtp_kv,
            Arc::new(SequenceDevice { gdn, scratch }),
            selections,
        ))
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
            flight: Arc::new(Flight::default()),
        }
    }

    /// The references a quantum of this sequence reads: the shared device part and a view of each cache as it is now
    /// (take it after the reservation). The quantum is counted in flight until its references are dropped.
    pub fn refs(&self) -> SequenceRefs {
        let mut refs = SequenceRefs::new(Arc::clone(&self.device))
            .with_kv(self.kv.view())
            .in_flight(self.flight.begin());
        if let Some(mtp) = &self.mtp_kv {
            refs = refs.with_mtp_kv(mtp.view());
        }
        refs
    }

    /// Quanta admitted on this sequence and not yet dropped.
    pub fn in_flight(&self) -> usize {
        self.flight.count()
    }

    /// Allocates the pinned staging the caches' reservations take (two sets per cache: one in flight, one being
    /// reserved), so that no reservation on the request path allocates. Startup only.
    pub fn prime_staging(&self, pool: &Arc<PinnedPool>) -> Result<(), CudaError> {
        self.kv.prime_staging(pool, 2)?;
        if let Some(mtp) = &self.mtp_kv {
            mtp.prime_staging(pool, 2)?;
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
