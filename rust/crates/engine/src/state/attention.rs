//! The attention scratch of a sequence: the split-KV decode area every attention layer shares, and the buffers
//! speculative drafting passes between quanta (`AttentionStates`).

use std::sync::Arc;

use euhedral_inference_cuda::{AllocationId, CudaError, CudaLibrary, DeviceBuffer};

use crate::model::ModelConfig;
use crate::residency::footprint::decode_scratch_bytes;

/// Slots of the draft token buffer: the step's current token and up to 16 drafts
/// (`AttentionStates.DRAFT_TOKEN_SLOTS`).
pub const DRAFT_TOKEN_SLOTS: u64 = 17;

/// What drafting adds to a sequence's scratch.
#[derive(Debug)]
pub struct DraftBuffers {
    /// Base post-final-norm hidden rows (BF16) that seed the next MTP catch-up.
    pub seed_rows: DeviceBuffer,
    /// Rows `seed_rows` holds.
    pub seed_capacity: u64,
    /// The latest MTP row's post-`mtp.norm` hidden (BF16), seeding the next recursive draft row.
    pub recursion_hidden: DeviceBuffer,
    /// The tokens of the step being drafted, int32: the current token, then its drafts in order.
    pub tokens: DeviceBuffer,
}

/// Device scratch owned by one sequence, allocated once at load.
#[derive(Debug)]
pub struct AttentionScratch {
    decode: DeviceBuffer,
    decode_rows: u64,
    query_heads: u64,
    draft: Option<DraftBuffers>,
}

impl AttentionScratch {
    /// Allocates the decode scratch for `decode_rows` rows, and, when the sequence drafts, `seed_rows` draft seed
    /// rows and the draft buffers. The waiting allocator: startup only.
    pub fn allocate(
        library: &Arc<CudaLibrary>,
        config: &ModelConfig,
        decode_rows: u64,
        drafts: bool,
        seed_rows: u64,
    ) -> Result<AttentionScratch, CudaError> {
        let query_heads = config.num_attention_heads as u64;
        let decode = DeviceBuffer::alloc(library, decode_scratch_bytes(query_heads, decode_rows) as usize)?;
        let draft = if drafts {
            let hidden = config.hidden_size as u64;
            Some(DraftBuffers {
                seed_rows: DeviceBuffer::alloc(library, (seed_rows * hidden * 2) as usize)?,
                seed_capacity: seed_rows,
                recursion_hidden: DeviceBuffer::alloc(library, (hidden * 2) as usize)?,
                tokens: DeviceBuffer::alloc(library, (4 * DRAFT_TOKEN_SLOTS) as usize)?,
            })
        } else {
            None
        };
        Ok(AttentionScratch {
            decode,
            decode_rows,
            query_heads,
            draft,
        })
    }

    /// The split-KV decode scratch: one area of `query heads x 64 splits x 258 floats` per row. One serves every
    /// attention layer of a quantum, which run one after another.
    pub fn decode(&self) -> &DeviceBuffer {
        &self.decode
    }

    /// The rows the decode scratch holds.
    pub fn decode_rows(&self) -> u64 {
        self.decode_rows
    }

    /// The query heads the scratch was sized for.
    pub fn query_heads(&self) -> u64 {
        self.query_heads
    }

    /// The drafting buffers, when the sequence drafts.
    pub fn draft(&self) -> Option<&DraftBuffers> {
        self.draft.as_ref()
    }

    /// The allocation serial of every buffer of the scratch, in a fixed order.
    pub fn generations(&self, out: &mut Vec<AllocationId>) {
        out.push(self.decode.generation());
        if let Some(draft) = &self.draft {
            out.extend([
                draft.seed_rows.generation(),
                draft.recursion_hidden.generation(),
                draft.tokens.generation(),
            ]);
        }
    }

    /// The device bytes the scratch holds.
    pub fn device_bytes(&self) -> u64 {
        self.decode.bytes() as u64
            + self.draft.as_ref().map_or(0, |draft| {
                (draft.seed_rows.bytes() + draft.recursion_hidden.bytes() + draft.tokens.bytes()) as u64
            })
    }
}
