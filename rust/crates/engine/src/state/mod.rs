//! What a sequence owns on the device: its attention cache, its Gated DeltaNet state and its attention scratch.

pub mod attention;
pub mod gdn;
pub mod kv;

use std::sync::Arc;

use euhedral_inference_cuda::{CudaError, CudaLibrary, PinnedPool, RetireList, Submission};

use crate::error::EngineError;
use crate::model::ModelConfig;
use crate::residency::footprint::{DRAFT_TWIN_ROWS, MINIMUM_VERIFY_ROWS};
use crate::residency::planner::ResidencyPlan;
use attention::AttentionScratch;
use gdn::GdnStates;
use kv::KvCache;
pub use kv::KvView;

/// What the stages of a quantum read of a sequence: its GDN state and its attention scratch. Allocated once, at
/// full size, and shared by every quantum of the sequence; nothing about it changes while quanta run, except the
/// contents of the buffers the stages write.
#[derive(Debug)]
pub struct SequenceDevice {
    pub gdn: GdnStates,
    pub scratch: AttentionScratch,
}

/// The device state of one sequence, as its owner holds it.
///
/// The owner holds what changes between quanta: the attention caches, whose frontiers it moves and whose pages and
/// page tables it grows (`reserve`, `appended`, `commit`), and the rows the next quantum replays. A quantum gets
/// the shared [`SequenceDevice`] and a [`KvView`] of each cache, copied at its admission (`KvCache::view`).
#[derive(Debug)]
pub struct SequenceState {
    /// The caches of the full-attention layers (the first `attention layers` of the plan's cache).
    pub kv: KvCache,
    /// The MTP layer's own cache, when the sequence drafts: it follows the base layers' at layer index 64.
    pub mtp_kv: Option<KvCache>,
    pub device: Arc<SequenceDevice>,
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
        Ok(SequenceState {
            kv,
            mtp_kv,
            device: Arc::new(SequenceDevice { gdn, scratch }),
        })
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
