//! The workspace: one device buffer per slot of the execution plan, sized once at load for a prefill chunk's rows.
//!
//! Nothing grows while quanta run: a quantum that needs more rows than the workspace holds is refused when it is
//! admitted. Two views of one slot may alias only where the plan's edges order their users; that ordering belongs to
//! the stage graph, not to this crate. The slots are `ExecutionPlan.Buffer`'s, the sizes those of
//! `SharedWorkspace` and `Workspace.slotBytes`: every slot holds the widest use any view of the plan makes of it, at
//! the workspace's rows.

use std::sync::Arc;

use euhedral_inference_cuda::{AllocationId, CudaError, CudaLibrary, DeviceBuffer, DevicePtr};
use thiserror::Error;

use crate::model::ModelConfig;

/// The element a slot holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Element {
    Bf16,
    Fp32,
}

impl Element {
    /// Bytes of one element.
    pub fn bytes(self) -> u64 {
        match self {
            Element::Bf16 => 2,
            Element::Fp32 => 4,
        }
    }
}

/// A named buffer of the execution plan (`ExecutionPlan.Buffer`) the pilot's views use, and the input record.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum BufferSlot {
    HiddenState,
    MixerHidden,
    FinalHiddenState,
    InputNormalized,
    QkProjected,
    ValueZProjected,
    AProjected,
    BProjected,
    GdnAlpha,
    GdnBeta,
    GdnConvolved,
    GdnRecurrent,
    GdnNormalized,
    AttentionQkNormalized,
    AttentionContext,
    MixerDelta,
    PostMixerNormalized,
    GateUp,
    Swiglu,
    FfnDelta,
    FinalNormalized,
    Logits,
    /// MTP stem scratch: one normalized half.
    MtpNormed,
    /// MTP stem scratch: the packed `[embedding; hidden]` rows.
    MtpPacked,
    /// The quantum's input record: its token ids (32-bit), then its start positions (64-bit, 8-byte aligned).
    TokenIds,
}

impl BufferSlot {
    /// Every slot, in allocation order.
    pub const ALL: [BufferSlot; 25] = [
        BufferSlot::HiddenState,
        BufferSlot::MixerHidden,
        BufferSlot::FinalHiddenState,
        BufferSlot::InputNormalized,
        BufferSlot::QkProjected,
        BufferSlot::ValueZProjected,
        BufferSlot::AProjected,
        BufferSlot::BProjected,
        BufferSlot::GdnAlpha,
        BufferSlot::GdnBeta,
        BufferSlot::GdnConvolved,
        BufferSlot::GdnRecurrent,
        BufferSlot::GdnNormalized,
        BufferSlot::AttentionQkNormalized,
        BufferSlot::AttentionContext,
        BufferSlot::MixerDelta,
        BufferSlot::PostMixerNormalized,
        BufferSlot::GateUp,
        BufferSlot::Swiglu,
        BufferSlot::FfnDelta,
        BufferSlot::FinalNormalized,
        BufferSlot::Logits,
        BufferSlot::MtpNormed,
        BufferSlot::MtpPacked,
        BufferSlot::TokenIds,
    ];

    /// The index of the slot in [`BufferSlot::ALL`].
    pub fn index(self) -> usize {
        self as usize
    }

    /// The width in elements of one row of the slot, and the element, for `config`: the widest use any view makes
    /// of it (`ExecutionPlan.fullBufferSpecs`). `None` for the input record, which is not rows of one width.
    pub fn row(self, config: &ModelConfig) -> Option<(u64, Element)> {
        let hidden = config.hidden_size as u64;
        let intermediate = config.intermediate_size as u64;
        let heads = config.linear_num_value_heads as u64;
        let attention_query = config.attention_query_width() as u64;
        let attention_key = (config.num_key_value_heads * config.attention_head_dim) as u64;
        let attention_qk = attention_query + attention_key;
        let gdn_value = config.gdn_value_width() as u64;
        Some(match self {
            BufferSlot::HiddenState
            | BufferSlot::MixerHidden
            | BufferSlot::FinalHiddenState
            | BufferSlot::InputNormalized
            | BufferSlot::MixerDelta
            | BufferSlot::PostMixerNormalized
            | BufferSlot::FfnDelta
            | BufferSlot::FinalNormalized
            | BufferSlot::MtpNormed => (hidden, Element::Bf16),
            // The GDN query and key rows, or the attention's: the wider.
            BufferSlot::QkProjected => ((config.gdn_query_key_width() as u64).max(attention_qk), Element::Bf16),
            // The GDN value and z rows, or the attention's gate and value rows: the wider.
            BufferSlot::ValueZProjected => ((2 * gdn_value).max(attention_query + attention_key), Element::Bf16),
            BufferSlot::AProjected | BufferSlot::BProjected | BufferSlot::GdnAlpha | BufferSlot::GdnBeta => {
                (heads, Element::Fp32)
            }
            BufferSlot::GdnConvolved => (config.gdn_convolution_channels() as u64, Element::Bf16),
            BufferSlot::GdnRecurrent | BufferSlot::GdnNormalized => (gdn_value, Element::Bf16),
            BufferSlot::AttentionQkNormalized => (attention_qk, Element::Bf16),
            BufferSlot::AttentionContext => (attention_query, Element::Bf16),
            BufferSlot::GateUp => (2 * intermediate, Element::Bf16),
            BufferSlot::Swiglu => (intermediate, Element::Bf16),
            BufferSlot::Logits => (config.vocab_size as u64, Element::Bf16),
            BufferSlot::MtpPacked => (2 * hidden, Element::Bf16),
            BufferSlot::TokenIds => return None,
        })
    }
}

/// The rows a workspace holds.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkspaceRows {
    /// Rows of every per-row slot: a prefill chunk's.
    pub rows: u64,
    /// Rows of the logits: the most any quantum scores (a verification's, never a whole chunk's).
    pub logits_rows: u64,
}

impl WorkspaceRows {
    /// Refuses a quantum of more rows than the workspace was sized for.
    pub fn check(&self, rows: u64) -> Result<(), WorkspaceError> {
        if rows > self.rows {
            return Err(WorkspaceError::TooManyRows {
                rows,
                capacity: self.rows,
            });
        }
        Ok(())
    }

    /// Refuses a quantum that scores more rows than the logits hold.
    pub fn check_logits(&self, rows: u64) -> Result<(), WorkspaceError> {
        if rows > self.logits_rows {
            return Err(WorkspaceError::TooManyLogitsRows {
                rows,
                capacity: self.logits_rows,
            });
        }
        Ok(())
    }
}

/// Bytes the input record of up to `rows` token ids and one start position holds: the ids, padded to 8 bytes, then
/// the position (`Workspace.inputByteSize`).
pub fn input_record_bytes(rows: u64) -> u64 {
    (rows * 4).div_ceil(8) * 8 + 8
}

/// Bytes of `slot` in a workspace of `rows`.
pub fn slot_bytes(config: &ModelConfig, slot: BufferSlot, rows: WorkspaceRows) -> u64 {
    match slot.row(config) {
        None => input_record_bytes(rows.rows),
        Some((width, element)) => {
            let slot_rows = if slot == BufferSlot::Logits {
                rows.logits_rows
            } else {
                rows.rows
            };
            slot_rows * width * element.bytes()
        }
    }
}

/// Bytes of the whole workspace for `rows` (without the expansion scratch).
pub fn workspace_bytes(config: &ModelConfig, rows: WorkspaceRows) -> u64 {
    BufferSlot::ALL.iter().map(|slot| slot_bytes(config, *slot, rows)).sum()
}

/// A quantum asked for more than the workspace was sized for.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum WorkspaceError {
    #[error("the quantum needs {rows} rows but the workspace was sized at load for {capacity}")]
    TooManyRows { rows: u64, capacity: u64 },
    #[error("the quantum needs {rows} logits rows but the workspace holds {capacity}")]
    TooManyLogitsRows { rows: u64, capacity: u64 },
}

/// The device buffers of the workspace.
#[derive(Debug)]
pub struct Workspace {
    rows: WorkspaceRows,
    slots: Vec<DeviceBuffer>,
    /// The expansion scratch regions of the linear routes.
    scratch: Vec<DeviceBuffer>,
}

impl Workspace {
    /// Allocates one buffer per slot for `rows`, and one per scratch region of `scratch_bytes`. A region of 0 bytes
    /// is not allocated. This allocates with the waiting `cudaMalloc`: startup only.
    pub fn allocate(
        library: &Arc<CudaLibrary>,
        config: &ModelConfig,
        rows: WorkspaceRows,
        scratch_bytes: &[u64],
    ) -> Result<Workspace, CudaError> {
        let mut slots = Vec::with_capacity(BufferSlot::ALL.len());
        for slot in BufferSlot::ALL {
            slots.push(DeviceBuffer::alloc(library, slot_bytes(config, slot, rows) as usize)?);
        }
        let mut scratch = Vec::with_capacity(scratch_bytes.len());
        for bytes in scratch_bytes.iter().copied().filter(|bytes| *bytes > 0) {
            scratch.push(DeviceBuffer::alloc(library, bytes as usize)?);
        }
        Ok(Workspace { rows, slots, scratch })
    }

    /// The rows the workspace was sized for.
    pub fn rows(&self) -> WorkspaceRows {
        self.rows
    }

    /// The view of a quantum of `rows` rows, or the refusal when the workspace is too small for it.
    pub fn view(&self, rows: u64) -> Result<WorkspaceView<'_>, WorkspaceError> {
        self.rows.check(rows)?;
        Ok(WorkspaceView { workspace: self, rows })
    }

    /// The expansion scratch region `region`, if the plan has one.
    pub fn scratch(&self, region: usize) -> Option<&DeviceBuffer> {
        self.scratch.get(region)
    }

    /// The allocation serial of every slot and scratch region, in slot order: what a captured graph that reads the
    /// workspace is keyed on.
    pub fn generations(&self) -> impl Iterator<Item = AllocationId> + '_ {
        self.slots.iter().chain(&self.scratch).map(DeviceBuffer::generation)
    }

    /// The device bytes the workspace holds.
    pub fn retained_bytes(&self) -> u64 {
        self.slots
            .iter()
            .chain(&self.scratch)
            .map(|buffer| buffer.bytes() as u64)
            .sum()
    }

    /// The buffer of `slot`.
    pub fn buffer(&self, slot: BufferSlot) -> &DeviceBuffer {
        &self.slots[slot.index()]
    }
}

/// A quantum's view of the workspace: its slots at its own row count.
#[derive(Clone, Copy, Debug)]
pub struct WorkspaceView<'w> {
    workspace: &'w Workspace,
    rows: u64,
}

impl WorkspaceView<'_> {
    /// The rows of the quantum.
    pub fn rows(&self) -> u64 {
        self.rows
    }

    /// The device address of `slot`.
    pub fn ptr(&self, slot: BufferSlot) -> DevicePtr {
        self.workspace.buffer(slot).ptr()
    }

    /// The bytes of `slot` this view uses: its rows of the slot's width.
    pub fn bytes(&self, config: &ModelConfig, slot: BufferSlot) -> u64 {
        match slot.row(config) {
            None => input_record_bytes(self.rows),
            Some((width, element)) => {
                let rows = if slot == BufferSlot::Logits {
                    self.workspace.rows.logits_rows
                } else {
                    self.rows
                };
                rows * width * element.bytes()
            }
        }
    }

    /// The logits rows the workspace holds; asking for more is refused.
    pub fn logits(&self, rows: u64) -> Result<DevicePtr, WorkspaceError> {
        self.workspace.rows.check_logits(rows)?;
        Ok(self.ptr(BufferSlot::Logits))
    }
}
