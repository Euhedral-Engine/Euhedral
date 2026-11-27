//! What one sparse-attention layer keeps of one sequence beside the KV pages (`QsaState`): the pooled indexer key of
//! every complete block of four tokens and the raw indexer keys of the up to three tokens of the incomplete trailing
//! block.
//!
//! The KV pages are the sequence's [`KvCache`](crate::state::kv::KvCache), shared with the other layers, which has
//! the lengths and the frontiers; this state has the two buffers and the tail's double buffering. A chunk reads the
//! committed tail and writes the other one, and a commit makes that one the committed tail, so a discarded chunk
//! leaves the committed state intact. Block keys at or past the committed length are overwritten by the next chunk, so
//! they need no undoing.
//!
//! Java mutates this state inside `QsaLayer.run`; here the state belongs to the sequence's owner, which calls
//! [`QsaState::begin_chunk`] when it builds a chunk's operations and [`QsaState::commit`] or [`QsaState::discard`]
//! once the quantum retired, and the body is a function of the [`ChunkAt`](crate::flash::block::ChunkAt) it is
//! handed.
//!
//! Bytes per token: 64 of block keys (256 per block of four tokens).

use euhedral_inference_cuda::DevicePtr;

use super::ops::{BLOCK_TOKENS, INDEX_WIDTH};
use crate::flash::FlashError;
use crate::flash::buffers::{ALIGNMENT, Carver};

/// Raw keys kept of the incomplete block.
pub const TAIL_SLOTS: u32 = BLOCK_TOKENS - 1;
/// Bytes of one indexer key.
const KEY_BYTES: u64 = INDEX_WIDTH as u64 * 2;

/// Blocks a sequence of `tokens` positions completes, at least one.
pub fn max_blocks(tokens: u32) -> u32 {
    (tokens / BLOCK_TOKENS).max(1)
}

/// The bytes a layer's indexer keeps of a sequence of `tokens` positions with blocks of `ratio` tokens and keys of
/// `key_bytes` bytes: one key per complete block, and the two raw tails of up to `ratio - 1` keys. The one formula
/// the residency plan and [`QsaState`] share.
pub fn indexer_layer_bytes(tokens: u64, ratio: u64, key_bytes: u64) -> Option<u64> {
    let blocks = (tokens / ratio).max(1);
    let tails = 2 * (ratio - 1);
    blocks.checked_add(tails)?.checked_mul(key_bytes)
}

/// The addresses of a layer's indexer buffers, which its body reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QsaBuffers {
    /// The pooled block keys: `[block][128]` BF16.
    pub block_keys: DevicePtr,
    /// The two raw tails: `[2][3][128]` BF16, token `4 (start / 4) + i` at slot `i`.
    pub tails: DevicePtr,
}

impl QsaBuffers {
    /// The bytes of one raw tail.
    pub const TAIL_BYTES: u64 = TAIL_SLOTS as u64 * KEY_BYTES;
}

/// The indexer state of one layer of one sequence.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QsaState {
    buffers: QsaBuffers,
    max_tokens: u32,
    /// The tail the committed state reads (0 or 1).
    live_tail: u32,
    /// Chunks begun since the last commit or discard: chunk `k` reads the tail chunk `k - 1` wrote.
    pending_chunks: u32,
}

impl QsaState {
    /// The bytes of the state of a sequence of up to `max_tokens` positions.
    pub fn bytes(max_tokens: u32) -> u64 {
        Self::carve(DevicePtr::NULL, max_tokens).1
    }

    /// The state laid out over `base`, which holds [`QsaState::bytes`] bytes.
    pub fn over(base: DevicePtr, max_tokens: u32) -> Result<Self, FlashError> {
        if max_tokens == 0 || base.is_null() || !base.get().is_multiple_of(ALIGNMENT) {
            return Err(FlashError::refused(
                "qsa state",
                format!(
                    "{max_tokens} tokens over {:#x}: tokens are positive and the base is a multiple of {ALIGNMENT}",
                    base.get()
                ),
            ));
        }
        Ok(Self::carve(base, max_tokens).0)
    }

    fn carve(base: DevicePtr, max_tokens: u32) -> (Self, u64) {
        let mut carver = Carver::new(base);
        let buffers = QsaBuffers {
            block_keys: carver.take(u64::from(max_blocks(max_tokens)) * KEY_BYTES),
            tails: carver.take(2 * QsaBuffers::TAIL_BYTES),
        };
        let state = Self {
            buffers,
            max_tokens,
            live_tail: 0,
            pending_chunks: 0,
        };
        (state, carver.bytes())
    }

    /// The addresses of the buffers.
    pub fn buffers(&self) -> QsaBuffers {
        self.buffers
    }

    /// The positions the state holds.
    pub fn max_tokens(&self) -> u32 {
        self.max_tokens
    }

    /// Begins a chunk and returns the slot of the tail it reads (the other is the one it writes): the committed tail
    /// for a quantum's first chunk, the previous chunk's for a prompt's later ones.
    pub fn begin_chunk(&mut self) -> u32 {
        let slot = self.live_tail ^ (self.pending_chunks & 1);
        self.pending_chunks += 1;
        slot
    }

    /// Publishes every pending chunk: the quantum's device work has retired. With none pending it does nothing.
    pub fn commit(&mut self) {
        self.live_tail ^= self.pending_chunks & 1;
        self.pending_chunks = 0;
    }

    /// Drops every pending chunk. A prompt's chunks after the first rewrite the tail the committed state reads, so
    /// only a quantum of one chunk leaves the committed state intact; a failed prompt leaves its sequence terminal.
    pub fn discard(&mut self) {
        self.pending_chunks = 0;
    }

    /// Forgets the sequence: the next chunk starts at position 0. Refused while a chunk is in flight.
    pub fn reset(&mut self) -> Result<(), FlashError> {
        if self.pending_chunks != 0 {
            return Err(FlashError::refused(
                "qsa state",
                format!("{} chunks are in flight", self.pending_chunks),
            ));
        }
        self.live_tail = 0;
        Ok(())
    }
}

#[cfg(test)]
mod tests;
