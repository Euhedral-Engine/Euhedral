//! Selection. A greedy quantum selects on the device: an
//! argmax key per logits row, the copy of the keys into pinned memory, and the pass of a drafted token to the next
//! quantum of its step; no logits row is copied to the host. A quantum of a sampling or teacher-forced request ([`CopyRows`]) copies its
//! logits row to pinned memory instead, inside its graph, and the host selects from the row.
//!
//! A key is 64 bits whose low word is `0xFFFFFFFF` minus the selected token id, so that among equal maxima the
//! lowest id has the highest key; it is 0 when no logit is selectable (all NaN or negative infinity).

use std::error::Error;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{AllocationId, CudaError, CudaLibrary, DeviceBuffer, DevicePtr, KernelOps, PinnedBuffer};

use super::{Launch, Quantum, eager, fatal, launch_u32};

use crate::state::attention::DRAFT_TOKEN_SLOTS;
use crate::workspace::BufferSlot;

/// Bytes of one key.
pub(crate) const KEY_BYTES: usize = 8;

/// The words a quantum's argmax writes on the device and the pinned memory their copy lands in.
///
/// The host reads a key only after the quantum that wrote it has retired, which is after the marker of the [`Argmax`]
/// stage's work has been reached.
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

    /// The 32-bit word `index` of the readback as the host sees it, once the quantum that wrote it has retired: for a
    /// stage that writes plain words in place of keys. `None` beyond the readback.
    pub fn word(&self, index: usize) -> Option<u32> {
        (index < 2 * self.capacity).then(|| self.readback.word(index).load(Ordering::Acquire))
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
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum SelectedRows {
    /// The last logits row: decode, and the last row of a prompt.
    Last,
    /// Every logits row: a verification reads back only token ids, never vocabulary rows.
    Each,
}

/// The greedy selection over the quantum's logits, and the copy of its keys to pinned memory.
///
/// The quantum's retirement is behind the stage's marker, so the keys are readable once the quantum retired.
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

impl Launch for Argmax {
    fn idle(&self, quantum: &Quantum) -> bool {
        // A quantum whose rows go to the host selects there.
        quantum.logits_rows == 0 || quantum.segments().iter().all(|segment| segment.row_readback.is_some())
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        if self.idle(quantum) {
            return Ok(());
        }
        if self.vocabulary == 0 || self.vocabulary > quantum.config.vocab_size {
            return Err(fatal(format!(
                "a vocabulary of {} entries does not fit the logits rows of {}",
                self.vocabulary, quantum.config.vocab_size
            )));
        }
        let entries = launch_u32(self.vocabulary, "the vocabulary")?;
        // The slot's rows are the model's vocabulary wide, whatever head wrote fewer entries into them.
        let row_bytes = quantum.config.vocab_size as u64 * 2;
        let logits = quantum.slot(self.logits);
        let single = quantum.segments().len() == 1;
        for segment in quantum.segments() {
            let selection = segment
                .selection
                .as_ref()
                .ok_or_else(|| fatal("a quantum that selects has no selection words"))?;
            // A quantum of one sequence scores the rows it was given; a segment of a batch scores all its rows, at its
            // offset among the logits rows.
            let (scored, base) = if single {
                (quantum.logits_rows, 0)
            } else {
                (segment.rows, segment.row_offset)
            };
            let (first, count) = match self.rows {
                SelectedRows::Last => (scored - 1, 1),
                SelectedRows::Each => (0, scored),
            };
            if count > selection.capacity() {
                return Err(fatal(format!(
                    "{count} selected rows exceed the {} the selection holds",
                    selection.capacity()
                )));
            }
            for row in 0..count {
                kernels
                    .argmax_bf16(
                        logits.offset((base + first + row) as u64 * row_bytes),
                        entries,
                        selection.device_keys().offset((row * KEY_BYTES) as u64),
                    )
                    .map_err(fatal)?;
            }
            // SAFETY: the selection is held by the quantum (an `Arc`), which retires only after the marker that follows
            // this launch is reached; the host reads the keys through atomic words, and only once that marker has been
            // reached.
            unsafe {
                kernels.copy_device_to_readback_shared(
                    selection.readback(),
                    0,
                    selection.device_keys(),
                    count * KEY_BYTES,
                )
            }
            .map_err(fatal)?;
        }
        Ok(())
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

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let Some(write) = quantum.draft_token_write else {
            return Ok(());
        };
        let selection = quantum
            .segment()
            .selection
            .as_ref()
            .ok_or_else(|| fatal("a quantum that passes a drafted token has no selection words"))?;
        let draft = quantum
            .segment()
            .refs
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
        kernels
            .selected_token(
                selection.device_keys(),
                write.ids,
                draft.tokens.ptr().offset(4 * write.slot as u64),
            )
            .map_err(fatal)
    }
}

/// Pinned memory the logits rows of a quantum that the host reads (sampled or teacher-forced) are copied to, and the allocation identity its graph is keyed
/// on. The host reads the rows only after the quantum that wrote them has retired.
#[derive(Debug)]
pub struct RowReadback {
    readback: PinnedBuffer,
    /// Bytes of one row: the model's vocabulary at two bytes an entry.
    row_bytes: usize,
    capacity: usize,
    /// Bytes after the rows that hold the last row's final hidden state: 0 for a plain readback, the hidden
    /// row's size for one a teacher-forced run that takes the hidden state leases.
    hidden_bytes: usize,
    /// Whether a quantum in flight, or the frame that selects from its rows, holds the readback
    /// ([`RowReadbackPool::acquire`]).
    held: AtomicBool,
}

impl RowReadback {
    /// Pinned memory for up to `rows` rows (at least 1) of `vocabulary` BF16 entries. The allocation waits: startup
    /// only, and so is the drop.
    pub fn new(library: &Arc<CudaLibrary>, rows: usize, vocabulary: usize) -> Result<RowReadback, CudaError> {
        RowReadback::with_hidden(library, rows, vocabulary, 0)
    }

    /// [`RowReadback::new`] that also holds `hidden_size` BF16 entries after the rows: the final hidden state of the
    /// quantum's last row, copied by the quantum beside its logits ([`CopyRows`]).
    pub fn with_hidden(
        library: &Arc<CudaLibrary>,
        rows: usize,
        vocabulary: usize,
        hidden_size: usize,
    ) -> Result<RowReadback, CudaError> {
        let rows = rows.max(1);
        let row_bytes = vocabulary * 2;
        let hidden_bytes = hidden_size * 2;
        Ok(RowReadback {
            readback: PinnedBuffer::alloc(library, rows * row_bytes + hidden_bytes)?,
            row_bytes,
            capacity: rows,
            hidden_bytes,
            held: AtomicBool::new(false),
        })
    }

    /// The rows the readback holds.
    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// The BF16 bytes of `row`, once the quantum that copied it has retired; `None` beyond the capacity.
    pub fn row(&self, row: usize) -> Option<&[u8]> {
        if row >= self.capacity {
            return None;
        }
        self.readback
            .as_slice()
            .get(row * self.row_bytes..(row + 1) * self.row_bytes)
    }

    /// The BF16 bytes of the last row's final hidden state, once the quantum that copied it has retired; `None` for a
    /// readback that holds none.
    pub fn hidden(&self) -> Option<&[u8]> {
        if self.hidden_bytes == 0 {
            return None;
        }
        let start = self.capacity * self.row_bytes;
        self.readback.as_slice().get(start..start + self.hidden_bytes)
    }

    /// The identity a graph that copies into the readback is keyed on: its address, which the launch bakes in, and
    /// which no other live readback shares.
    pub fn generation(&self) -> AllocationId {
        AllocationId(self.readback.as_ptr() as u64)
    }

    pub(crate) fn pinned(&self) -> &PinnedBuffer {
        &self.readback
    }
}

/// The row readbacks of one sequence: a quantum whose rows the host reads leases one from its admission until the host has selected from
/// its row, which is later than the quantum's retirement, so the lease belongs to the owner and then to the frame that
/// selects, not to the quantum.
#[derive(Debug, Default)]
pub struct RowReadbackPool {
    readbacks: Vec<Arc<RowReadback>>,
    /// Readbacks of one row that also hold the last row's final hidden state, for a teacher-forced run that takes it.
    /// Their own allocations: a quantum that copies the hidden row is a different graph from one that does not.
    hidden: Vec<Arc<RowReadback>>,
}

/// Every row readback of the pool is leased.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadbacksBusy;

impl fmt::Display for ReadbacksBusy {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the sequence has no free row readback")
    }
}

impl Error for ReadbacksBusy {}

impl RowReadbackPool {
    /// `count` readbacks of up to `rows` rows of `vocabulary` entries. The allocations wait: startup only.
    pub fn new(
        library: &Arc<CudaLibrary>,
        count: usize,
        rows: usize,
        vocabulary: usize,
    ) -> Result<RowReadbackPool, CudaError> {
        let readbacks = (0..count)
            .map(|_| RowReadback::new(library, rows, vocabulary).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(RowReadbackPool {
            readbacks,
            hidden: Vec::new(),
        })
    }

    /// The pool with `count` more readbacks of one row of `vocabulary` entries and the final hidden state of
    /// `hidden_size` entries ([`RowReadbackPool::acquire_hidden`]). The allocations wait: startup only.
    pub fn with_hidden(
        mut self,
        library: &Arc<CudaLibrary>,
        count: usize,
        vocabulary: usize,
        hidden_size: usize,
    ) -> Result<RowReadbackPool, CudaError> {
        if vocabulary == 0 || hidden_size == 0 {
            return Err(CudaError::Argument(format!(
                "a readback that holds the hidden state needs a vocabulary and a hidden size, not {vocabulary} and {hidden_size}"
            )));
        }
        self.hidden = (0..count)
            .map(|_| RowReadback::with_hidden(library, 1, vocabulary, hidden_size).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(self)
    }

    /// A pool with no readback: a sequence that cannot serve requests whose rows the host reads.
    pub fn none() -> RowReadbackPool {
        RowReadbackPool::default()
    }

    /// The readbacks of the pool, those that hold the hidden state too.
    pub fn len(&self) -> usize {
        self.readbacks.len() + self.hidden.len()
    }

    /// The rows each readback of the pool holds, the hidden ones aside (they hold one); 0 for a pool with none.
    pub fn rows(&self) -> usize {
        self.readbacks.first().map_or(0, |readback| readback.capacity())
    }

    pub fn is_empty(&self) -> bool {
        self.readbacks.is_empty() && self.hidden.is_empty()
    }

    /// Whether no readback without a hidden tail is free to lease: the pool has none, or every one is held.
    pub fn is_exhausted(&self) -> bool {
        RowReadbackPool::held(&self.readbacks) >= self.readbacks.len()
    }

    /// Readbacks leased now, the hidden ones too.
    pub fn leased(&self) -> usize {
        RowReadbackPool::held(&self.readbacks) + RowReadbackPool::held(&self.hidden)
    }

    fn held(readbacks: &[Arc<RowReadback>]) -> usize {
        readbacks
            .iter()
            .filter(|readback| readback.held.load(Ordering::Acquire))
            .count()
    }

    /// Whether no readback that holds the hidden state is free to lease.
    pub fn hidden_is_exhausted(&self) -> bool {
        RowReadbackPool::held(&self.hidden) >= self.hidden.len()
    }

    /// Leases a readback no one holds, or refuses. Called by the sequence's owner, one admission at a time.
    pub fn acquire(&self) -> Result<RowReadbackLease, ReadbacksBusy> {
        RowReadbackPool::lease(&self.readbacks)
    }

    /// [`RowReadbackPool::acquire`] for a readback that also holds the last row's final hidden state.
    pub fn acquire_hidden(&self) -> Result<RowReadbackLease, ReadbacksBusy> {
        RowReadbackPool::lease(&self.hidden)
    }

    fn lease(readbacks: &[Arc<RowReadback>]) -> Result<RowReadbackLease, ReadbacksBusy> {
        for readback in readbacks {
            if readback
                .held
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Ok(RowReadbackLease {
                    readback: Arc::clone(readback),
                });
            }
        }
        Err(ReadbacksBusy)
    }
}

/// A row readback leased by the owner; dropping it returns the readback to its pool.
#[derive(Debug)]
pub struct RowReadbackLease {
    readback: Arc<RowReadback>,
}

impl RowReadbackLease {
    pub fn readback(&self) -> &Arc<RowReadback> {
        &self.readback
    }
}

impl Drop for RowReadbackLease {
    fn drop(&mut self) {
        self.readback.held.store(false, Ordering::Release);
    }
}

/// Copies the quantum's logits rows to the pinned [`RowReadback`] the quantum was given, for a host that reads them
/// (sampling, teacher forcing). Idle for a quantum with no readback, which is every greedy quantum: the stage is in the graph's list
/// and queues nothing.
#[derive(Clone, Debug)]
pub struct CopyRows {
    logits: BufferSlot,
    /// The slot of the final hidden state, whose last row goes beside the logits to a readback that holds it.
    hidden: BufferSlot,
    rows: SelectedRows,
}

impl CopyRows {
    pub fn new(logits: BufferSlot, hidden: BufferSlot, rows: SelectedRows) -> Self {
        Self { logits, hidden, rows }
    }
}

impl Launch for CopyRows {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.logits_rows == 0 || quantum.segments().iter().all(|segment| segment.row_readback.is_none())
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        if self.idle(quantum) {
            return Ok(());
        }
        // The slot's rows are the model's vocabulary wide, whatever head wrote fewer entries into them.
        let row_bytes = quantum.config.vocab_size * 2;
        let logits = quantum.slot(self.logits);
        let single = quantum.segments().len() == 1;
        for segment in quantum.segments() {
            let readback = segment
                .row_readback
                .as_ref()
                .ok_or_else(|| fatal("a quantum that copies rows has no readback"))?;
            // As the selection: a segment of a batch copies all its rows, from its offset among the logits rows.
            let (scored, base) = if single {
                (quantum.logits_rows, 0)
            } else {
                (segment.rows, segment.row_offset)
            };
            let (first, count) = match self.rows {
                SelectedRows::Last => (scored - 1, 1),
                SelectedRows::Each => (0, scored),
            };
            if count > readback.capacity() {
                return Err(fatal(format!(
                    "{count} copied rows exceed the {} the readback holds",
                    readback.capacity()
                )));
            }
            if row_bytes != readback.row_bytes {
                return Err(fatal(format!(
                    "a readback of {} bytes a row does not fit the logits rows of {row_bytes}",
                    readback.row_bytes
                )));
            }
            for row in 0..count {
                // SAFETY: the readback is held by the quantum (an `Arc`), which retires only after the marker that
                // follows this launch is reached, and by the lease of the owner after it; the host reads the row only
                // once that marker has been reached, and no other copy writes the range.
                unsafe {
                    kernels.copy_device_to_readback_shared(
                        readback.pinned(),
                        row * row_bytes,
                        logits.offset(((base + first + row) * row_bytes) as u64),
                        row_bytes,
                    )
                }
                .map_err(fatal)?;
            }
            let copy = hidden_copy(
                readback.hidden_bytes,
                readback.capacity * row_bytes,
                quantum.config.hidden_size * 2,
                quantum.rows,
                single,
                self.rows,
            )
            .map_err(fatal)?;
            if let Some(copy) = copy {
                // SAFETY: as above; the hidden row follows the logits rows in the readback, and the quantum's own
                // stages wrote the slot before this launch, so it is this quantum's row and not another's.
                unsafe {
                    kernels.copy_device_to_readback_shared(
                        readback.pinned(),
                        copy.destination,
                        quantum.slot(self.hidden).offset(copy.source as u64),
                        copy.bytes,
                    )
                }
                .map_err(fatal)?;
            }
        }
        Ok(())
    }
}

/// The copy of a quantum's last final-hidden row into the tail of a readback.
#[derive(Debug, PartialEq, Eq)]
struct HiddenCopy {
    /// Byte offset in the readback: after its logits rows.
    destination: usize,
    /// Byte offset in the hidden slot: the start of the quantum's last row.
    source: usize,
    bytes: usize,
}

/// The hidden-row copy of a quantum of `quantum_rows` rows, or `None` for a readback with no hidden tail
/// (`tail_bytes` 0). `logits_bytes` is the readback's logits rows, `hidden_bytes` one hidden row as the model has it,
/// `single` whether the quantum is one sequence's own and `selected` which logits rows it copies. Refuses what cannot
/// be honoured: a tail that does not fit the model's hidden row, a batch, or a quantum that copies every row (the
/// hidden state is the last row's only).
fn hidden_copy(
    tail_bytes: usize,
    logits_bytes: usize,
    hidden_bytes: usize,
    quantum_rows: usize,
    single: bool,
    selected: SelectedRows,
) -> Result<Option<HiddenCopy>, String> {
    if tail_bytes == 0 {
        return Ok(None);
    }
    if !single {
        return Err("a readback that holds the hidden state is a quantum of one sequence's own".to_string());
    }
    if selected == SelectedRows::Each {
        return Err("the hidden state is copied for a quantum's last row only".to_string());
    }
    if tail_bytes != hidden_bytes {
        return Err(format!(
            "a readback of {tail_bytes} hidden bytes does not fit the hidden rows of {hidden_bytes}"
        ));
    }
    Ok(Some(HiddenCopy {
        destination: logits_bytes,
        source: quantum_rows.max(1).saturating_sub(1) * hidden_bytes,
        bytes: hidden_bytes,
    }))
}

eager!(Argmax, SelectedToken, CopyRows);

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
    fn a_pool_with_no_readback_has_none_free() {
        let pool = RowReadbackPool::none();
        assert!(pool.is_empty());
        assert!(
            pool.is_exhausted(),
            "a request that samples is refused before it prefills"
        );
    }

    #[test]
    fn a_plain_readback_copies_no_hidden_row() {
        assert_eq!(hidden_copy(0, 800, 16, 3, true, SelectedRows::Last), Ok(None));
        assert_eq!(hidden_copy(0, 800, 16, 3, false, SelectedRows::Each), Ok(None));
    }

    #[test]
    fn a_hidden_readback_takes_the_last_row_after_the_logits() {
        let copy = |rows| hidden_copy(16, 800, 16, rows, true, SelectedRows::Last);
        let at = |source| {
            Ok(Some(HiddenCopy {
                destination: 800,
                source,
                bytes: 16,
            }))
        };
        assert_eq!(copy(1), at(0));
        assert_eq!(copy(5), at(64));
        assert_eq!(copy(0), at(0));
    }

    #[test]
    fn a_hidden_readback_refuses_a_batch_every_row_and_a_wrong_size() {
        assert!(hidden_copy(16, 800, 16, 4, false, SelectedRows::Last).is_err());
        assert!(hidden_copy(16, 800, 16, 4, true, SelectedRows::Each).is_err());
        let error = hidden_copy(16, 800, 32, 4, true, SelectedRows::Last).unwrap_err();
        assert!(error.contains("16") && error.contains("32"), "{error}");
    }

    #[test]
    fn a_zero_key_selects_no_token() {
        assert_eq!(token_of(0), None);
    }
}
