//! Greedy selection on the device (`generation/HostLogits`, `native/src/sampling/argmax.cuh`): an argmax key per
//! logits row, the copy of the keys into pinned memory, and the pass of a drafted token to the next quantum of its
//! step. No logits row is copied to the host.
//!
//! A key is 64 bits whose low word is `0xFFFFFFFF` minus the selected token id, so that among equal maxima the
//! lowest id has the highest key; it is 0 when no logit is selectable (all NaN or negative infinity).

use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use euhedral_graph::Marker;
use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{
    AllocationId, CudaError, CudaLibrary, DeviceBuffer, DevicePtr, PinnedBuffer, Submission, ops,
};

use super::{Launch, Quantum, eager, fatal, launch_u32};
use crate::state::attention::DRAFT_TOKEN_SLOTS;
use crate::workspace::BufferSlot;

/// Bytes of one key.
const KEY_BYTES: usize = 8;

/// The words a quantum's argmax writes on the device and the pinned memory their copy lands in.
///
/// The host reads a key only after the marker of the [`Argmax`] stage that wrote it has been reached (a completion
/// edge, or the lane's completion word): the stage finishes with a signalled marker, so the lane's completion word
/// reaches the marker once the copy has retired.
#[derive(Debug)]
pub struct Selection {
    keys: DeviceBuffer,
    // Declared after the device words, which are freed first; freeing pinned memory waits for the device.
    readback: PinnedBuffer,
    capacity: usize,
    /// Whether a quantum in flight fills the words ([`SelectionPool::acquire`]): until it is released, no other
    /// quantum is given them.
    held: AtomicBool,
}

impl Selection {
    /// Words for up to `rows` rows (a verification's rows, at least 1). The allocations wait: startup only, and so
    /// is the drop.
    pub fn new(library: &Arc<CudaLibrary>, rows: usize) -> Result<Selection, CudaError> {
        let rows = rows.max(1);
        Ok(Selection {
            keys: DeviceBuffer::alloc(library, rows * KEY_BYTES)?,
            readback: PinnedBuffer::alloc(library, rows * KEY_BYTES)?,
            capacity: rows,
            held: AtomicBool::new(false),
        })
    }

    /// The rows the words hold.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The key of `row`, once its argmax has retired. `None` for a row beyond the capacity.
    pub fn key(&self, row: usize) -> Option<u64> {
        if row >= self.capacity {
            return None;
        }
        let low = self.readback.word(2 * row).load(Ordering::Acquire);
        let high = self.readback.word(2 * row + 1).load(Ordering::Acquire);
        let mut bytes = [0u8; KEY_BYTES];
        bytes[..4].copy_from_slice(&low.to_ne_bytes());
        bytes[4..].copy_from_slice(&high.to_ne_bytes());
        Some(u64::from_ne_bytes(bytes))
    }

    /// The token `row` selected, once its argmax has retired; `None` for a row with no selectable logit, or beyond
    /// the capacity.
    pub fn token(&self, row: usize) -> Option<u32> {
        token_of(self.key(row)?)
    }

    /// The device address of the keys.
    pub(crate) fn device_keys(&self) -> DevicePtr {
        self.keys.ptr()
    }

    /// The allocation serial of the device words, which a graph that writes them is keyed on.
    pub fn generation(&self) -> AllocationId {
        self.keys.generation()
    }

    pub(crate) fn readback(&self) -> &PinnedBuffer {
        &self.readback
    }
}

/// The selection words of one sequence: a few [`Selection`]s that quanta take in turn.
///
/// A quantum's argmax writes the device keys and copies them to the selection's pinned words; the host reads them once
/// the quantum retired. A quantum admitted behind it on the device (the zero-gap chain) must not reach the same words
/// before they were read, so a selection is leased to one quantum from its admission until the quantum is dropped,
/// which the owner does after it read the keys (`HostLogits.admitted`). With every selection leased the owner holds
/// the admission back.
#[derive(Debug)]
pub struct SelectionPool {
    selections: Vec<Arc<Selection>>,
}

/// Every selection of the pool is leased to a quantum in flight.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SelectionsBusy;

impl fmt::Display for SelectionsBusy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("every selection of the sequence is held by a quantum in flight")
    }
}

impl Error for SelectionsBusy {}

impl SelectionPool {
    /// `count` selections of up to `rows` rows each. The allocations wait: startup only.
    pub fn new(library: &Arc<CudaLibrary>, count: usize, rows: usize) -> Result<SelectionPool, CudaError> {
        let selections = (0..count.max(1))
            .map(|_| Selection::new(library, rows).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(SelectionPool { selections })
    }

    /// Selections in the pool.
    pub fn len(&self) -> usize {
        self.selections.len()
    }

    pub fn is_empty(&self) -> bool {
        self.selections.is_empty()
    }

    /// Selections leased to quanta in flight.
    pub fn leased(&self) -> usize {
        self.selections
            .iter()
            .filter(|selection| selection.held.load(Ordering::Acquire))
            .count()
    }

    /// Leases a selection that no quantum holds, or refuses when every one is held. Called by the sequence's owner,
    /// one admission at a time.
    pub fn acquire(&self) -> Result<SelectionLease, SelectionsBusy> {
        for selection in &self.selections {
            if selection
                .held
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(SelectionLease {
                    selection: Arc::clone(selection),
                });
            }
        }
        Err(SelectionsBusy)
    }
}

/// A selection leased to one quantum; dropping it (with the quantum, once its keys were read) returns the selection to
/// its pool.
#[derive(Debug)]
pub struct SelectionLease {
    selection: Arc<Selection>,
}

impl SelectionLease {
    /// The leased selection.
    pub fn selection(&self) -> &Arc<Selection> {
        &self.selection
    }
}

impl Drop for SelectionLease {
    fn drop(&mut self) {
        self.selection.held.store(false, Ordering::Release);
    }
}

/// The token a key chose, or `None` for the key of a row with no selectable logit.
pub fn token_of(key: u64) -> Option<u32> {
    if key == 0 {
        None
    } else {
        Some(0xFFFF_FFFF - (key & 0xFFFF_FFFF) as u32)
    }
}

/// Which logits rows an argmax selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SelectedRows {
    /// The last logits row: decode, and the last row of a prompt.
    Last,
    /// Every logits row: a verification reads back only token ids, never vocabulary rows.
    Each,
}

/// The greedy selection over the quantum's logits, and the copy of its keys to pinned memory.
///
/// The stage's marker is signalled, so that a completion edge to the stage that follows it (the sequence's
/// continuation) is found by the lane's completion word.
#[derive(Clone, Debug)]
pub struct Argmax {
    logits: BufferSlot,
    vocabulary: usize,
    rows: SelectedRows,
}

impl Argmax {
    /// Selects over the logits rows of `logits`, each `vocabulary` entries wide.
    pub fn new(logits: BufferSlot, vocabulary: usize, rows: SelectedRows) -> Self {
        Self {
            logits,
            vocabulary,
            rows,
        }
    }
}

impl Argmax {
    /// The signal step of a selection that was launched in a graph: finishes `submission`, on which the graph was
    /// launched, with a marker whose completion word is written. Eager submission does the same after
    /// [`Launch::launch`]; a graph capture cannot hold the marker or the signal, so the launch of the graph is
    /// followed by this.
    pub fn signal(submission: Submission<'_>) -> Result<Marker, CudaError> {
        submission.finish_signalled()
    }
}

impl Launch for Argmax {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.logits_rows == 0
    }

    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError> {
        if self.idle(quantum) {
            return Ok(());
        }
        let logits_rows = quantum.logits_rows;
        let selection = quantum
            .selection
            .as_ref()
            .ok_or_else(|| fatal("a quantum that selects has no selection words"))?;
        if self.vocabulary == 0 || self.vocabulary > quantum.config.vocab_size {
            return Err(fatal(format!(
                "a vocabulary of {} entries does not fit the logits rows of {}",
                self.vocabulary, quantum.config.vocab_size
            )));
        }
        let (first, count) = match self.rows {
            SelectedRows::Last => (logits_rows - 1, 1),
            SelectedRows::Each => (0, logits_rows),
        };
        if count > selection.capacity() {
            return Err(fatal(format!(
                "{count} selected rows exceed the {} the selection holds",
                selection.capacity()
            )));
        }
        let entries = launch_u32(self.vocabulary, "the vocabulary")?;
        // The slot's rows are the model's vocabulary wide, whatever head wrote fewer entries into them.
        let row_bytes = quantum.config.vocab_size as u64 * 2;
        let logits = quantum.slot(self.logits);
        for row in 0..count {
            ops::argmax_bf16(
                submission,
                logits.offset((first + row) as u64 * row_bytes),
                entries,
                selection.device_keys().offset((row * KEY_BYTES) as u64),
            )
            .map_err(fatal)?;
        }
        // SAFETY: the selection is held by the quantum (an `Arc`), which retires only after the marker that follows
        // this launch is reached; the host reads the keys through atomic words, and only once that marker has been
        // reached.
        unsafe {
            ops::copy_device_to_readback_shared(
                submission,
                selection.readback(),
                0,
                selection.device_keys(),
                count * KEY_BYTES,
            )
        }
        .map_err(fatal)
    }
}

/// Passes the drafted token of the quantum's selection to the sequence's step tokens on the device, where the next
/// quantum of the step reads it (`selected_token`): `ids[row]` of the selected row of the draft head, or the row
/// itself when the head has no ids. Does nothing for a quantum that drafts no token.
#[derive(Clone, Copy, Debug, Default)]
pub struct SelectedToken;

impl Launch for SelectedToken {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.draft_token_write.is_none()
    }

    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError> {
        let Some(write) = quantum.draft_token_write else {
            return Ok(());
        };
        let selection = quantum
            .selection
            .as_ref()
            .ok_or_else(|| fatal("a quantum that passes a drafted token has no selection words"))?;
        let draft = quantum
            .seq
            .device()
            .scratch
            .draft()
            .ok_or_else(|| fatal("the sequence has no draft buffers"))?;
        if write.slot as u64 >= DRAFT_TOKEN_SLOTS {
            return Err(fatal(format!(
                "draft token slot {} is out of {DRAFT_TOKEN_SLOTS}",
                write.slot
            )));
        }
        ops::selected_token(
            submission,
            selection.device_keys(),
            write.ids,
            draft.tokens.ptr().offset(4 * write.slot as u64),
        )
        .map_err(fatal)
    }
}

eager!(Argmax, SelectedToken);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_key_holds_the_complement_of_the_lowest_selected_id() {
        assert_eq!(token_of(0xFFFF_FFFF - 5), Some(5));
        assert_eq!(token_of((0x3f80 << 32) | (0xFFFF_FFFF - 248_319)), Some(248_319));
        assert_eq!(token_of(0xFFFF_FFFF), Some(0));
    }

    #[test]
    fn a_zero_key_selects_no_token() {
        assert_eq!(token_of(0), None);
    }
}
