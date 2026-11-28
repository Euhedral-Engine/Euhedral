//! The attention scratch of a sequence: the split-KV decode area every attention layer shares, and the buffers
//! speculative drafting passes between quanta (`AttentionStates`).

use std::sync::Arc;

use euhedral_inference_cuda::{AllocationId, CudaError, CudaLibrary, DeviceBuffer, DevicePtr};

use crate::model::ModelConfig;
use crate::model::dflash2::Dflash2Config;
use crate::residency::footprint::decode_scratch_bytes;

/// Slots of the draft token buffer: the step's current token and up to 16 drafts.
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

/// What the DFlash2 drafter keeps for one sequence: the tap rows of the base model's committed rows, which the
/// drafter's context update reads, and the drafter's key/value rings (`DFlash2State`).
#[derive(Debug)]
pub struct DrafterBuffers {
    taps: DeviceBuffer,
    tap_capacity: u64,
    ring_keys: DeviceBuffer,
    ring_values: DeviceBuffer,
    ring_bytes: u64,
    layers: usize,
}

impl DrafterBuffers {
    /// Bytes of one layer's ring of keys (or of values): the window's rows of the key/value width, BF16.
    pub fn ring_bytes_for(config: &Dflash2Config) -> u64 {
        (config.sliding_window * config.kv_width() * 2) as u64
    }

    /// Bytes the buffers hold for `tap_rows` tap rows.
    pub fn bytes(config: &Dflash2Config, tap_rows: u64) -> u64 {
        tap_rows * config.tap_width() as u64 * 2 + 2 * config.layers as u64 * Self::ring_bytes_for(config)
    }

    /// Buffers for `tap_rows` tap rows (the most rows one base quantum has). The waiting allocator: startup only.
    pub fn allocate(library: &Arc<CudaLibrary>, config: &Dflash2Config, tap_rows: u64) -> Result<Self, CudaError> {
        let ring_bytes = Self::ring_bytes_for(config);
        let rings = ring_bytes as usize * config.layers;
        Ok(Self {
            taps: DeviceBuffer::alloc(library, tap_rows as usize * config.tap_width() * 2)?,
            tap_capacity: tap_rows,
            ring_keys: DeviceBuffer::alloc(library, rings)?,
            ring_values: DeviceBuffer::alloc(library, rings)?,
            ring_bytes,
            layers: config.layers,
        })
    }

    /// The tap rows: one row of `tap width` BF16 values per row of the base quantum that wrote them.
    pub fn taps(&self) -> DevicePtr {
        self.taps.ptr()
    }

    /// The rows the taps hold.
    pub fn tap_capacity(&self) -> u64 {
        self.tap_capacity
    }

    /// The drafter's layers: each has a ring of keys and one of values.
    pub fn layers(&self) -> usize {
        self.layers
    }

    /// Bytes of one ring.
    pub fn ring_bytes(&self) -> u64 {
        self.ring_bytes
    }

    /// The ring of keys of the drafter's layer `layer`.
    pub fn ring_keys(&self, layer: usize) -> DevicePtr {
        self.ring_keys.ptr().offset(layer as u64 * self.ring_bytes)
    }

    /// The ring of values of the drafter's layer `layer`.
    pub fn ring_values(&self, layer: usize) -> DevicePtr {
        self.ring_values.ptr().offset(layer as u64 * self.ring_bytes)
    }

    fn generations(&self) -> [AllocationId; 3] {
        [
            self.taps.generation(),
            self.ring_keys.generation(),
            self.ring_values.generation(),
        ]
    }

    fn device_bytes(&self) -> u64 {
        (self.taps.bytes() + self.ring_keys.bytes() + self.ring_values.bytes()) as u64
    }
}

/// Device scratch owned by one sequence, allocated once at load.
#[derive(Debug)]
pub struct AttentionScratch {
    decode: DeviceBuffer,
    decode_rows: u64,
    query_heads: u64,
    draft: Option<DraftBuffers>,
    drafter: Option<DrafterBuffers>,
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
            drafter: None,
        })
    }

    /// The same scratch with the DFlash2 drafter's buffers.
    pub fn with_drafter(mut self, drafter: DrafterBuffers) -> AttentionScratch {
        self.drafter = Some(drafter);
        self
    }

    /// The DFlash2 drafter's buffers, when the sequence drafts with it.
    pub fn drafter(&self) -> Option<&DrafterBuffers> {
        self.drafter.as_ref()
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
        if let Some(drafter) = &self.drafter {
            out.extend(drafter.generations());
        }
    }

    /// The device bytes the scratch holds.
    pub fn device_bytes(&self) -> u64 {
        self.decode.bytes() as u64
            + self.draft.as_ref().map_or(0, |draft| {
                (draft.seed_rows.bytes() + draft.recursion_hidden.bytes() + draft.tokens.bytes()) as u64
            })
            + self.drafter.as_ref().map_or(0, DrafterBuffers::device_bytes)
    }
}
