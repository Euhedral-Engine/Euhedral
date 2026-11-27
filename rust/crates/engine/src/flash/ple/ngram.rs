//! The host side of the per-layer n-gram embedding: the hashed row ids of the tokens ([`NgramIds`], Java's `NgramIds`),
//! the table of records the ids address ([`NgramRows`], Java's `NgramStore.gatherRecordsRange`), and the split of one
//! quantum's gather into parts that are filled apart and joined once all have arrived ([`PartSlots`]).

use std::fmt;
use std::ops::Range;
use std::sync::{Arc, OnceLock};

use euhedral_inference_edrl::NgramLayout;

use crate::flash::FlashError;

/// The most parts one gather splits into (`Shape.PLE_PARTS`).
pub const PARTS: usize = 8;
/// Rows below which a gather is not worth splitting: a part is a frame, and its hop costs more than a few dozen rows
/// of copying.
const MIN_PART_ROWS: usize = 64;

/// The parts a gather of `count` rows splits into: one per 64 rows, from one to [`PARTS`].
pub fn part_count(count: usize) -> usize {
    (count / MIN_PART_ROWS).clamp(1, PARTS)
}

/// The rows of part `part` of a gather of `count` rows: empty for a part past [`part_count`] and for a part of no
/// rows.
pub fn part_range(count: usize, part: usize) -> Range<usize> {
    let parts = part_count(count);
    if part >= parts {
        return 0..0;
    }
    // 128-bit so that `count * part` cannot overflow.
    let edge = |index: usize| (count as u128 * index as u128 / parts as u128) as usize;
    edge(part)..edge(part + 1)
}

/// The hashed row ids of the n-gram heads (`NgramIds`), over a context of the tokens before the first one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct NgramIds {
    size: u32,
    heads_per_ngram: u32,
    multipliers: Box<[i64]>,
    vocabularies: Box<[i64]>,
    offsets: Box<[i64]>,
    end_of_sequence: i32,
}

impl NgramIds {
    /// The ids of an n-gram of `size` tokens with `heads_per_ngram` heads for each of the sizes 2 to `size`: one
    /// multiplier per position, and a vocabulary size and a row offset per head. Refuses a size below 2, and a
    /// vocabulary that is not positive or an offset that is negative.
    pub fn new(
        size: u32,
        heads_per_ngram: u32,
        multipliers: &[i64],
        vocabularies: &[i64],
        offsets: &[i64],
        end_of_sequence: i32,
    ) -> Result<Self, FlashError> {
        const OP: &str = "ngram ids";
        let heads = u64::from(size.saturating_sub(1)) * u64::from(heads_per_ngram);
        if size < 2 || heads_per_ngram == 0 {
            return Err(FlashError::refused(
                OP,
                format!("{heads_per_ngram} heads of {size}-grams: the size is at least 2 and the heads positive"),
            ));
        }
        if multipliers.len() != size as usize || vocabularies.len() as u64 != heads || offsets.len() as u64 != heads {
            return Err(FlashError::refused(
                OP,
                format!(
                    "{} multipliers, {} vocabularies and {} offsets for {size} positions and {heads} heads",
                    multipliers.len(),
                    vocabularies.len(),
                    offsets.len()
                ),
            ));
        }
        if let Some(bad) = vocabularies.iter().find(|vocabulary| **vocabulary <= 0) {
            return Err(FlashError::refused(
                OP,
                format!("a head's vocabulary is {bad}, which is not positive"),
            ));
        }
        if let Some(bad) = offsets.iter().find(|offset| **offset < 0) {
            return Err(FlashError::refused(
                OP,
                format!("a head's offset is {bad}, which is negative"),
            ));
        }
        Ok(Self {
            size,
            heads_per_ngram,
            multipliers: multipliers.into(),
            vocabularies: vocabularies.into(),
            offsets: offsets.into(),
            end_of_sequence,
        })
    }

    /// The hash heads: one set of `heads_per_ngram` for each n-gram size from 2.
    pub fn heads(&self) -> usize {
        (self.size as usize - 1) * self.heads_per_ngram as usize
    }

    /// The tokens a sequence carries between calls.
    pub fn context_length(&self) -> usize {
        self.size as usize - 1
    }

    /// The context of a new sequence: all end-of-sequence tokens, as upstream's cache starts.
    pub fn fresh_context(&self) -> Box<[i32]> {
        vec![self.end_of_sequence; self.context_length()].into()
    }

    /// The `heads()` ids of each of `tokens`, row by row, and the context after them: the last `context_length`
    /// entries of `context` followed by `tokens`. Nothing is changed in place, so a quantum that is abandoned leaves
    /// the sequence's context as it was. Refuses a context of the wrong length.
    pub fn compute(&self, context: &[i32], tokens: &[u32]) -> Result<(Vec<u64>, Box<[i32]>), FlashError> {
        let history = self.context_length();
        if context.len() != history {
            return Err(FlashError::refused(
                "ngram ids",
                format!("a context of {} tokens, expected {history}", context.len()),
            ));
        }
        let heads = self.heads();
        let mut ids = Vec::with_capacity(tokens.len() * heads);
        for position in 0..tokens.len() {
            let mut mixed = self
                .token_at(context, tokens, position, 0)
                .wrapping_mul(self.multipliers[0]);
            let mut column = 0;
            for ngram in 2..=self.size as usize {
                // The mix of an n-gram extends the previous one by the token n - 1 positions back.
                mixed ^= self
                    .token_at(context, tokens, position, ngram - 1)
                    .wrapping_mul(self.multipliers[ngram - 1]);
                for _ in 0..self.heads_per_ngram {
                    let id = mixed.rem_euclid(self.vocabularies[column]) + self.offsets[column];
                    ids.push(id as u64);
                    column += 1;
                }
            }
        }
        let next = (0..history)
            .map(|index| {
                let combined = tokens.len() + index;
                if combined >= history {
                    tokens[combined - history] as i32
                } else {
                    context[combined]
                }
            })
            .collect();
        Ok((ids, next))
    }

    /// The token `shift` positions before `position` when no end-of-sequence token lies between them, else the
    /// end-of-sequence token. Shift 0 is the token itself.
    fn token_at(&self, context: &[i32], tokens: &[u32], position: usize, shift: usize) -> i64 {
        if shift == 0 {
            return i64::from(tokens[position] as i32);
        }
        let end = i64::from(self.end_of_sequence);
        let before = |back: usize| -> i64 {
            match position.checked_sub(back) {
                Some(index) => i64::from(tokens[index] as i32),
                None => i64::from(context[context.len() + position - back]),
            }
        };
        if (1..=shift).any(|back| before(back) == end) {
            return end;
        }
        before(shift)
    }
}

/// The bytes of one shard of the table: a mapped file, or memory.
pub trait ShardBytes: AsRef<[u8]> + Send + Sync + fmt::Debug {}

impl<T: AsRef<[u8]> + Send + Sync + fmt::Debug> ShardBytes for T {}

/// The n-gram table: shards of `shard_rows` rows each (the last may hold fewer), every shard laid out as
/// [`NgramLayout`] describes and carrying its own global scale. Reads are shared and never block.
#[derive(Debug)]
pub struct NgramRows {
    shards: Box<[Arc<dyn ShardBytes>]>,
    layouts: Box<[NgramLayout]>,
    scales: Box<[f32]>,
    shard_rows: u64,
    total_rows: u64,
    row_width: u32,
    row_bytes: u32,
    record_bytes: u32,
}

impl NgramRows {
    /// The table of `total_rows` rows of `row_width` values over `shards`. Refuses a shard too short for its rows and a
    /// shard count that does not cover the rows.
    pub fn new(
        shards: Vec<Arc<dyn ShardBytes>>,
        shard_rows: u64,
        total_rows: u64,
        row_width: u32,
    ) -> Result<Self, FlashError> {
        const OP: &str = "ngram table";
        if shard_rows == 0 || total_rows == 0 || shards.len() as u64 != total_rows.div_ceil(shard_rows) {
            return Err(FlashError::refused(
                OP,
                format!(
                    "{} shards of {shard_rows} rows do not hold {total_rows} rows",
                    shards.len()
                ),
            ));
        }
        let mut layouts = Vec::with_capacity(shards.len());
        let mut scales = Vec::with_capacity(shards.len());
        for (index, shard) in shards.iter().enumerate() {
            let rows = shard_rows.min(total_rows - index as u64 * shard_rows);
            let layout = NgramLayout::new(rows, u64::from(row_width))
                .map_err(|why| FlashError::refused(OP, format!("shard {index}: {why}")))?;
            let bytes = shard.as_ref().as_ref();
            if (bytes.len() as u64) < layout.byte_size() {
                return Err(FlashError::refused(
                    OP,
                    format!(
                        "shard {index} holds {} bytes, {rows} rows of {row_width} values take {}",
                        bytes.len(),
                        layout.byte_size()
                    ),
                ));
            }
            let at = layout.trailer_offset() as usize;
            let mut scale = [0u8; 4];
            scale.copy_from_slice(&bytes[at..at + 4]);
            scales.push(f32::from_le_bytes(scale));
            layouts.push(layout);
        }
        let row_bytes = layouts[0].row_bytes() as u32;
        Ok(Self {
            shards: shards.into(),
            layouts: layouts.into(),
            scales: scales.into(),
            shard_rows,
            total_rows,
            row_width,
            row_bytes,
            record_bytes: (row_bytes + 4).next_multiple_of(4),
        })
    }

    /// The values of one row.
    pub fn row_width(&self) -> u32 {
        self.row_width
    }

    /// The bytes of one record: the packed row, zeros, then the shard's scale as a little-endian `f32`, in a multiple
    /// of four bytes.
    pub fn record_bytes(&self) -> u32 {
        self.record_bytes
    }

    /// The rows of the table.
    pub fn total_rows(&self) -> u64 {
        self.total_rows
    }

    /// Writes the records of `ids[range]` to `out`, one [`NgramRows::record_bytes`] each in order
    /// (`NgramStore.gatherRecordsRange`). Refuses a row outside the table, a range outside `ids` and an `out` of the
    /// wrong length.
    pub fn gather(&self, ids: &[u64], range: Range<usize>, out: &mut [u8]) -> Result<(), FlashError> {
        const OP: &str = "ngram gather";
        let record = self.record_bytes as usize;
        if range.end > ids.len() || range.start > range.end || out.len() != (range.end - range.start) * record {
            return Err(FlashError::refused(
                OP,
                format!(
                    "rows {}..{} of {} ids into {} bytes of {record}-byte records",
                    range.start,
                    range.end,
                    ids.len(),
                    out.len()
                ),
            ));
        }
        let row_bytes = self.row_bytes as usize;
        for (slot, &row) in out.chunks_exact_mut(record).zip(&ids[range]) {
            if row >= self.total_rows {
                return Err(FlashError::refused(OP, format!("row {row} of {}", self.total_rows)));
            }
            let shard = (row / self.shard_rows) as usize;
            let local = row % self.shard_rows;
            // The layout of the shard knows every row of it: the shard of this row was checked at construction.
            let at = self.layouts[shard].row_offset(local).unwrap_or(0) as usize;
            let bytes: &[u8] = self.shards[shard].as_ref().as_ref();
            slot[..row_bytes].copy_from_slice(&bytes[at..at + row_bytes]);
            slot[row_bytes..record - 4].fill(0);
            slot[record - 4..].copy_from_slice(&self.scales[shard].to_le_bytes());
        }
        Ok(())
    }
}

/// What a part of a gather's publication refuses.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PublishError {
    /// The part is not one of the slots.
    OutOfRange { part: usize, parts: usize },
    /// The part was published already.
    Twice { part: usize },
}

/// One slot per part of a gather, each filled once by the part's own stage and read by the stage that joins them.
/// A slot is a one-shot cell: the publication releases what the part wrote and the read acquires it.
#[derive(Debug)]
pub struct PartSlots<T> {
    slots: Box<[OnceLock<T>]>,
}

impl<T> PartSlots<T> {
    /// `parts` empty slots.
    pub fn new(parts: usize) -> Self {
        Self {
            slots: (0..parts).map(|_| OnceLock::new()).collect(),
        }
    }

    /// Publishes `value` as part `part`. Hands `value` back with the refusal, so the caller keeps what it could not
    /// publish.
    pub fn publish(&self, part: usize, value: T) -> Result<(), (PublishError, T)> {
        let Some(slot) = self.slots.get(part) else {
            let parts = self.slots.len();
            return Err((PublishError::OutOfRange { part, parts }, value));
        };
        slot.set(value).map_err(|value| (PublishError::Twice { part }, value))
    }

    /// The value of part `part`, once published.
    pub fn get(&self, part: usize) -> Option<&T> {
        self.slots.get(part).and_then(OnceLock::get)
    }

    /// The published parts of a gather of `count` rows, in order, skipping the parts of no rows. `Err` names the first
    /// part with rows that has not been published.
    pub fn join(&self, count: usize) -> Result<Vec<(Range<usize>, &T)>, usize> {
        let mut joined = Vec::new();
        for part in 0..part_count(count) {
            let range = part_range(count, part);
            if range.is_empty() {
                continue;
            }
            match self.get(part) {
                Some(value) => joined.push((range, value)),
                None => return Err(part),
            }
        }
        Ok(joined)
    }
}

#[cfg(test)]
mod tests;
