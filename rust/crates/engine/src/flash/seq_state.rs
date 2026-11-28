//! What one sequence carries between quanta on a Flash-Next engine: the GDN convolution and recurrent state of each
//! GDN layer, the indexer buffers of each sparse-attention layer ([`QsaState`]), the n-gram embedding's convolution
//! history, the KV pages ([`KvCache`]) and, on the host, the position and the n-gram context.
//!
//! The device state is one allocation carved by [`SequenceLayout`]: the stages of a step hold its addresses from the
//! day they are built, so the sequence's state lives as long as the engine and a request resets it. [`Tracker`] is the host side of the lifecycle and has no device:
//! a chunk is begun, then committed once its quantum has retired or discarded when it failed, and the position and
//! the context advance only on a commit, so a failed chunk leaves them as they were. One chunk is in flight at a
//! time. [`SequenceState`] adds the device: it reserves the KV pages of a chunk and hands the chunk's placement (the
//! `ChunkAt` of each cache layer) to the quantum.

use std::fmt::Display;
use std::sync::Arc;

use euhedral_inference_cuda::{CudaError, DeviceBuffer, DevicePtr, KernelOps, LaneOps, PinnedPool};

use super::FlashError;
use super::block::ChunkPlacement;
use super::buffers::{ALIGNMENT, Carver};
use super::gdn::GdnState;
use super::qsa::state::QsaState;
use crate::state::kv::{KvCache, KvError, KvRetained};

const OP: &str = "sequence state";

/// What one layer keeps of a sequence.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerSizes {
    /// A GDN layer: bytes of the convolution history and of the recurrent state.
    GatedDeltaNet { history_bytes: u64, recurrent_bytes: u64 },
    /// A sparse-attention layer: its indexer state; its KV pages are the cache's.
    Attention,
}

/// The sizes of the state of one sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceSpec {
    pub layers: Vec<LayerSizes>,
    /// Positions the sequence may reach.
    pub max_tokens: u32,
    /// Bytes of the n-gram embedding's convolution history.
    pub ple_history_bytes: u64,
}

impl SequenceSpec {
    /// The bytes of the device state: the carve of the layout, each range padded to its alignment.
    pub fn bytes(&self) -> u64 {
        let mut carver = Carver::new(DevicePtr::NULL);
        for sizes in &self.layers {
            match *sizes {
                LayerSizes::GatedDeltaNet {
                    history_bytes,
                    recurrent_bytes,
                } => {
                    carver.take(history_bytes);
                    carver.take(recurrent_bytes);
                }
                LayerSizes::Attention => {
                    carver.take(QsaState::bytes(self.max_tokens));
                }
            }
        }
        carver.take(self.ple_history_bytes);
        carver.bytes()
    }
}

/// What one layer keeps, at its addresses.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum LayerState {
    Gdn(GdnState),
    Attention(QsaState),
}

/// The addresses of a sequence's device state.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SequenceLayout {
    base: DevicePtr,
    bytes: u64,
    layers: Vec<LayerState>,
    ple_history: DevicePtr,
}

impl SequenceLayout {
    /// The state of `spec` over `base`, which holds [`SequenceSpec::bytes`] bytes. Refuses a sequence of no positions
    /// and a base that is null or not aligned.
    pub fn over(spec: &SequenceSpec, base: DevicePtr) -> Result<Self, FlashError> {
        if spec.max_tokens == 0 || base.is_null() || !base.get().is_multiple_of(ALIGNMENT) {
            return Err(FlashError::refused(
                OP,
                format!(
                    "{} positions over {:#x}: positions are positive and the base is a multiple of {ALIGNMENT}",
                    spec.max_tokens,
                    base.get()
                ),
            ));
        }
        let mut carver = Carver::new(base);
        let mut layers = Vec::with_capacity(spec.layers.len());
        for sizes in &spec.layers {
            layers.push(match *sizes {
                LayerSizes::GatedDeltaNet {
                    history_bytes,
                    recurrent_bytes,
                } => LayerState::Gdn(GdnState {
                    conv_history: carver.take(history_bytes),
                    recurrent: carver.take(recurrent_bytes),
                }),
                LayerSizes::Attention => {
                    let at = carver.take(QsaState::bytes(spec.max_tokens));
                    LayerState::Attention(QsaState::over(at, spec.max_tokens)?)
                }
            });
        }
        let ple_history = carver.take(spec.ple_history_bytes);
        Ok(Self {
            base,
            bytes: carver.bytes(),
            layers,
            ple_history,
        })
    }

    /// The first byte of the state.
    pub fn base(&self) -> DevicePtr {
        self.base
    }

    /// The bytes the state spans.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The state of layer `index`.
    pub fn layer(&self, index: usize) -> Option<&LayerState> {
        self.layers.get(index)
    }

    /// The n-gram embedding's convolution history.
    pub fn ple_history(&self) -> DevicePtr {
        self.ple_history
    }

    /// The indexer state of each sparse-attention layer, in layer order.
    pub fn attention(&self) -> impl Iterator<Item = QsaState> + '_ {
        self.layers.iter().filter_map(|layer| match layer {
            LayerState::Attention(state) => Some(state.clone()),
            LayerState::Gdn(_) => None,
        })
    }
}

/// Where a chunk starts, and the raw tail slot its sparse-attention layers read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Begun {
    pub start: u32,
    pub tail_slot: u32,
}

/// The host side of one sequence: its position, the n-gram context, and the chunk in flight.
#[derive(Debug)]
pub struct Tracker {
    max_tokens: u32,
    position: u32,
    in_flight: u32,
    context: Box<[i32]>,
    qsa: Vec<QsaState>,
}

impl Tracker {
    /// A sequence at position 0 with `context` (the n-gram context of a fresh sequence), whose sparse-attention layers
    /// keep `qsa`. Refuses no positions and states that disagree on them.
    pub fn new(max_tokens: u32, context: Box<[i32]>, qsa: Vec<QsaState>) -> Result<Self, FlashError> {
        if max_tokens == 0 || qsa.iter().any(|state| state.max_tokens() != max_tokens) {
            return Err(FlashError::refused(
                OP,
                format!("a sequence of {max_tokens} positions (at least one) over indexer states of the same length"),
            ));
        }
        Ok(Self {
            max_tokens,
            position: 0,
            in_flight: 0,
            context,
            qsa,
        })
    }

    /// The positions the sequence has committed.
    pub fn position(&self) -> u32 {
        self.position
    }

    /// The n-gram context after the committed positions.
    pub fn context(&self) -> &[i32] {
        &self.context
    }

    /// Whether a chunk is begun and neither committed nor discarded.
    pub fn in_flight(&self) -> bool {
        self.in_flight != 0
    }

    /// Begins a chunk of `rows` rows at the position. Refuses no rows, a chunk already in flight and a chunk that
    /// ends past the sequence's positions.
    pub fn begin(&mut self, rows: u32) -> Result<Begun, FlashError> {
        let end = u64::from(self.position) + u64::from(rows);
        if rows == 0 || self.in_flight() || end > u64::from(self.max_tokens) {
            return Err(FlashError::refused(
                OP,
                format!(
                    "a chunk of {rows} rows at position {} of {} ({} in flight)",
                    self.position, self.max_tokens, self.in_flight
                ),
            ));
        }
        self.in_flight = rows;
        let mut tail_slot = 0;
        for state in &mut self.qsa {
            // The layers move in lockstep, so they all read the same slot.
            tail_slot = state.begin_chunk();
        }
        Ok(Begun {
            start: self.position,
            tail_slot,
        })
    }

    /// Publishes the chunk in flight, whose quantum retired: the position advances by its rows and the context is
    /// `next_context`. Returns the rows. Refuses with no chunk in flight and a context of another length.
    pub fn commit(&mut self, next_context: &[i32]) -> Result<u32, FlashError> {
        if !self.in_flight() || next_context.len() != self.context.len() {
            return Err(FlashError::refused(
                OP,
                format!(
                    "a commit with {} rows in flight and a context of {} entries, not {}",
                    self.in_flight,
                    next_context.len(),
                    self.context.len()
                ),
            ));
        }
        let rows = self.in_flight;
        self.position += rows;
        self.in_flight = 0;
        self.context.copy_from_slice(next_context);
        for state in &mut self.qsa {
            state.commit();
        }
        Ok(rows)
    }

    /// Drops the chunk in flight, which failed: the position and the context stay. With none in flight it does nothing.
    pub fn discard(&mut self) {
        self.in_flight = 0;
        for state in &mut self.qsa {
            state.discard();
        }
    }

    /// Returns the sequence to position 0 with `context`. Refuses a chunk in flight and a context of another length.
    pub fn reset(&mut self, context: &[i32]) -> Result<(), FlashError> {
        if self.in_flight() || context.len() != self.context.len() {
            return Err(FlashError::refused(
                OP,
                format!(
                    "a reset with {} rows in flight and a context of {} entries, not {}",
                    self.in_flight,
                    context.len(),
                    self.context.len()
                ),
            ));
        }
        for state in &mut self.qsa {
            state.reset()?;
        }
        self.position = 0;
        self.context.copy_from_slice(context);
        Ok(())
    }
}

/// One sequence: its device state, its cache and its [`Tracker`].
#[derive(Debug)]
pub struct SequenceState {
    // Held for the layout's addresses: freed with the state.
    buffer: DeviceBuffer,
    layout: SequenceLayout,
    kv: KvCache,
    tracker: Tracker,
}

impl SequenceState {
    /// The state over `buffer`, which the layout was carved from, and `kv`, whose cache layers are the layout's
    /// sparse-attention layers in order. The state holds whatever the buffer held: reset it before the first chunk.
    pub fn new(
        buffer: DeviceBuffer,
        layout: SequenceLayout,
        kv: KvCache,
        context: Box<[i32]>,
    ) -> Result<Self, FlashError> {
        let attention: Vec<QsaState> = layout.attention().collect();
        let max_tokens = attention.first().map_or(0, QsaState::max_tokens);
        if buffer.ptr() != layout.base() || (buffer.bytes() as u64) < layout.bytes() || kv.layers() != attention.len() {
            return Err(FlashError::refused(
                OP,
                format!(
                    "a buffer of {} bytes at {:#x} for a layout of {} bytes at {:#x}, and a cache of {} layers for {} attention layers",
                    buffer.bytes(),
                    buffer.ptr().get(),
                    layout.bytes(),
                    layout.base().get(),
                    kv.layers(),
                    attention.len()
                ),
            ));
        }
        let tracker = Tracker::new(max_tokens, context, attention)?;
        Ok(Self {
            buffer,
            layout,
            kv,
            tracker,
        })
    }

    pub fn layout(&self) -> &SequenceLayout {
        &self.layout
    }

    pub fn tracker(&self) -> &Tracker {
        &self.tracker
    }

    /// Returns the sequence to position 0: zeroes the device state on `kernels` (the caller waits for the lane) and
    /// forgets the cache's rows. Refuses a chunk in flight.
    pub fn reset(&mut self, kernels: &mut dyn KernelOps, context: &[i32]) -> Result<(), FlashError> {
        let rejected = |error: &dyn Display| FlashError::refused(OP, format!("reset: {error}"));
        self.tracker.reset(context)?;
        self.kv.truncate(0).map_err(|error| rejected(&error))?;
        kernels
            .zero_device_memory(self.buffer.ptr(), self.layout.bytes())
            .map_err(|error: CudaError| rejected(&error))
    }

    /// Begins a chunk of `rows` rows: reserves its cache rows on `lane` (the lane that runs the quantum's input
    /// preparation, so the reservation and the table upload are ordered before the stages) and returns its placement.
    /// A failure leaves the sequence as it was.
    pub fn begin_chunk(
        &mut self,
        rows: u32,
        lane: &mut dyn LaneOps,
        pool: &Arc<PinnedPool>,
        retained: &mut KvRetained,
    ) -> Result<ChunkPlacement, FlashError> {
        let begun = self.tracker.begin(rows)?;
        let reserved = self
            .kv
            .reserve(lane, pool, u64::from(begun.start), u64::from(rows), retained)
            .and_then(|()| self.kv.appended(u64::from(rows)));
        if let Err(error) = reserved {
            self.discard();
            return Err(FlashError::refused(OP, format!("the cache refused the chunk: {error}")));
        }
        let tables = (0..self.kv.layers())
            .map(|layer| self.kv.key_table(layer).zip(self.kv.value_table(layer)))
            .collect::<Option<Vec<_>>>();
        match tables {
            Some(tables) => ChunkPlacement::new(begun.start, begun.tail_slot, tables),
            None => {
                self.discard();
                Err(FlashError::refused(
                    OP,
                    "the cache has no page table for a layer after reserving",
                ))
            }
        }
    }

    /// Publishes the chunk, whose quantum retired, with the context after it. Returns its rows.
    pub fn commit(&mut self, next_context: &[i32]) -> Result<u32, FlashError> {
        let rows = self.tracker.commit(next_context)?;
        self.kv.commit_submitted();
        Ok(rows)
    }

    /// Drops the chunk in flight (a failed quantum's), cache rows included.
    pub fn discard(&mut self) {
        self.tracker.discard();
        self.kv.discard();
    }
}

impl From<KvError> for FlashError {
    fn from(error: KvError) -> Self {
        FlashError::refused("kv cache", error.to_string())
    }
}

#[cfg(test)]
mod tests;
