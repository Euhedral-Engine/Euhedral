//! The host side of the per-layer n-gram embedding: the hashed row ids of the tokens ([`NgramIds`]),
//! the table of records the ids address ([`NgramRows`]), and the split of one
//! quantum's gather into parts that are filled apart and joined once all have arrived ([`PartSlots`]).

use std::fmt;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use euhedral_data_structures::sync::ClaimCell;
use euhedral_inference_edrl::NgramLayout;
use euhedral_io::Continuation;

use crate::flash::FlashError;
use crate::frames::{Kind, step};

/// The most parts one gather splits into.
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

/// What a started read tells: the bytes read, or why they were not. It returns the frame that runs the rest of the
/// work, which whoever ended the read runs.
pub type RowRead = Continuation<Result<Vec<u8>, String>>;

/// The bytes of one shard of the table, read asynchronously: from the artifact file through RustEE's I/O
/// ([`FileShard`]), or a fake in a test.
pub trait ShardBytes: Send + Sync + fmt::Debug {
    /// The shard's length in bytes.
    fn size(&self) -> u64;

    /// Starts reading `len` bytes at `offset` of the shard and returns at once; `then` runs when the read has ended
    /// (it may be before this call returns), with the bytes or the failure, naming the cause. Refuses a range outside
    /// the shard, or an empty one, starting nothing: `then` is dropped unrun.
    fn start(&self, offset: u64, len: usize, then: RowRead) -> Result<(), String>;
}

/// A shard held in memory, whose reads end inside the call that starts them (the frame that continues the work runs
/// there, as a test or a tool runs it): for tables built from bytes, not for the artifact file.
#[derive(Debug)]
pub struct MemoryShard(pub Vec<u8>);

impl ShardBytes for MemoryShard {
    fn size(&self) -> u64 {
        self.0.len() as u64
    }

    fn start(&self, offset: u64, len: usize, then: RowRead) -> Result<(), String> {
        let end = offset.checked_add(len as u64).filter(|end| *end <= self.size());
        let (Some(end), true) = (end, len > 0) else {
            return Err(format!("{len} bytes at {offset} of a shard of {}", self.size()));
        };
        // `end` fits the shard, whose length fits `usize`.
        let bytes = self.0[offset as usize..end as usize].to_vec();
        then(Ok(bytes))
            .execute()
            .map_err(|error| format!("the frame that continued the read failed: {error:?}"))
    }
}

/// The n-gram table: shards of `shard_rows` rows each (the last may hold fewer), every shard laid out as
/// [`NgramLayout`] describes and carrying its own global scale. Reads are shared; a read of a file shard waits for the
/// page cache as a read of a mapped page does.
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
    /// The bytes shard `shard` of a table of `total_rows` rows in shards of `shard_rows` takes, and where its scale (a
    /// little-endian `f32`) is: the figures to size a shard and to read its scale by. Refuses a shard past the table
    /// and a table the layout cannot hold.
    pub fn shard_extent(
        shard: u64,
        shard_rows: u64,
        total_rows: u64,
        row_width: u32,
    ) -> Result<(u64, u64), FlashError> {
        let layout = Self::shard_layout(shard, shard_rows, total_rows, row_width)?;
        Ok((layout.byte_size(), layout.trailer_offset()))
    }

    fn shard_layout(shard: u64, shard_rows: u64, total_rows: u64, row_width: u32) -> Result<NgramLayout, FlashError> {
        const OP: &str = "ngram table";
        let first = shard.checked_mul(shard_rows).filter(|first| *first < total_rows);
        let Some(first) = first else {
            return Err(FlashError::refused(
                OP,
                format!("shard {shard} of {shard_rows} rows is past the table's {total_rows} rows"),
            ));
        };
        NgramLayout::new(shard_rows.min(total_rows - first), u64::from(row_width))
            .map_err(|why| FlashError::refused(OP, format!("shard {shard}: {why}")))
    }

    /// The table of `total_rows` rows of `row_width` values over `shards`, `scales` holding each shard's scale (read
    /// from the shard's [`shard_extent`](Self::shard_extent) trailer by the caller: building a table reads nothing).
    /// Refuses a shard too short for its rows, a shard count that does not cover the rows and a scale count that is
    /// not the shards'.
    pub fn new(
        shards: Vec<Arc<dyn ShardBytes>>,
        shard_rows: u64,
        total_rows: u64,
        row_width: u32,
        scales: Vec<f32>,
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
        if scales.len() != shards.len() {
            return Err(FlashError::refused(
                OP,
                format!("{} scales for {} shards", scales.len(), shards.len()),
            ));
        }
        let mut layouts = Vec::with_capacity(shards.len());
        for (index, shard) in shards.iter().enumerate() {
            let layout = Self::shard_layout(index as u64, shard_rows, total_rows, row_width)?;
            if shard.size() < layout.byte_size() {
                return Err(FlashError::refused(
                    OP,
                    format!(
                        "shard {index} holds {} bytes, its rows of {row_width} values take {}",
                        shard.size(),
                        layout.byte_size()
                    ),
                ));
            }
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

    /// The shard and the offset in it of each row of `ids[range]`. Refuses a row outside the table and a range outside
    /// `ids`.
    fn places(&self, ids: &[u64], range: Range<usize>) -> Result<Vec<(usize, u64)>, FlashError> {
        const OP: &str = "ngram gather";
        let Some(rows) = ids.get(range.clone()) else {
            return Err(FlashError::refused(
                OP,
                format!("rows {}..{} of {} ids", range.start, range.end, ids.len()),
            ));
        };
        rows.iter()
            .map(|&row| {
                if row >= self.total_rows {
                    return Err(FlashError::refused(OP, format!("row {row} of {}", self.total_rows)));
                }
                let shard = (row / self.shard_rows) as usize;
                // The layout of the shard knows every row of it: the shard of this row was checked at construction.
                let at = self.layouts[shard]
                    .row_offset(row % self.shard_rows)
                    .ok_or_else(|| FlashError::refused(OP, format!("row {row} is not in shard {shard}")))?;
                Ok((shard, at))
            })
            .collect()
    }

    /// Starts reading the records of `ids[range]` and returns at once; `done` runs, as the work of a frame, when the last
    /// row has been read, with the records, one [`NgramRows::record_bytes`] each in order,
    /// or the first failure. Every row is its own read, started here: the I/O engine queues what its ring cannot take.
    /// Refuses a row outside the table and a range outside `ids`, starting nothing (`done` is dropped unrun); a range of
    /// no rows ends at once, in this call.
    pub fn gather(self: &Arc<Self>, ids: &[u64], range: Range<usize>, done: GatherDone) -> Result<(), FlashError> {
        let places = self.places(ids, range)?;
        let count = places.len();
        let gather = Arc::new(Gather {
            table: Arc::clone(self),
            rows: (0..count).map(|_| OnceLock::new()).collect(),
            shards: places.iter().map(|place| place.0).collect(),
            // The call's own count: no row ends the gather inside it unless the call has started every row.
            pending: AtomicUsize::new(count + 1),
            failure: OnceLock::new(),
            done: ClaimCell::new(Some(done)),
        });
        for (index, (shard, at)) in places.into_iter().enumerate() {
            let ended = Arc::clone(&gather);
            let then: RowRead = Box::new(move |result| {
                step(Kind::NgramRead, index, move || {
                    match result {
                        Ok(bytes) => {
                            let _ = ended.rows[index].set(bytes);
                        }
                        Err(why) => ended.fail(index, why),
                    }
                    ended.tick();
                })
            });
            if let Err(why) = self.shards[shard].start(at, self.row_bytes as usize, then) {
                // Nothing started: the continuation is gone, unrun.
                gather.fail(index, why);
                gather.tick();
            }
        }
        gather.tick();
        Ok(())
    }
}

impl NgramRows {
    /// The records of `ids[range]`, for a table whose shards end their reads inside the call ([`MemoryShard`]). Refuses
    /// what [`gather`](Self::gather) refuses, and a table with a read still out.
    pub fn gather_in_memory(self: &Arc<Self>, ids: &[u64], range: Range<usize>) -> Result<Vec<u8>, FlashError> {
        let slot: Arc<OnceLock<Result<Vec<u8>, FlashError>>> = Arc::new(OnceLock::new());
        let told = Arc::clone(&slot);
        self.gather(
            ids,
            range,
            Box::new(move |records| {
                let _ = told.set(records);
            }),
        )?;
        slot.get()
            .cloned()
            .unwrap_or_else(|| Err(FlashError::refused("ngram gather", "a read was still out".to_owned())))
    }
}

/// What a gather tells when its last row has been read.
pub type GatherDone = Box<dyn FnOnce(Result<Vec<u8>, FlashError>) + Send>;

/// The rows of one gather in flight, and what joins them.
struct Gather {
    table: Arc<NgramRows>,
    rows: Box<[OnceLock<Vec<u8>>]>,
    shards: Vec<usize>,
    /// Rows not ended, and the starting call's own count.
    pending: AtomicUsize,
    /// The first row's failure.
    failure: OnceLock<FlashError>,
    /// Taken once, by the one that brings `pending` to zero.
    done: ClaimCell<Option<GatherDone>>,
}

impl Gather {
    fn fail(&self, row: usize, why: String) {
        // The first failure is the one told; a later one is a consequence or a duplicate.
        let _ = self.failure.set(FlashError::refused(
            "ngram gather",
            format!("row {row} of the gather: {why}"),
        ));
    }

    /// One row ended (or was never started): the last one ends the gather.
    fn tick(&self) {
        if self.pending.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        let Some(mut claim) = self.done.try_claim() else {
            return;
        };
        let Some(done) = claim.take() else {
            return;
        };
        drop(claim);
        done(self.join());
    }

    /// The records of the rows, each the row's bytes, zeros, and its shard's scale.
    fn join(&self) -> Result<Vec<u8>, FlashError> {
        if let Some(error) = self.failure.get() {
            return Err(error.clone());
        }
        let table = &self.table;
        let (row_bytes, record) = (table.row_bytes as usize, table.record_bytes as usize);
        let mut out = vec![0u8; self.rows.len() * record];
        for ((slot, row), shard) in out.chunks_exact_mut(record).zip(&self.rows).zip(&self.shards) {
            let bytes = row
                .get()
                .filter(|bytes| bytes.len() == row_bytes)
                .ok_or_else(|| FlashError::refused("ngram gather", "a row ended without its bytes".to_owned()))?;
            slot[..row_bytes].copy_from_slice(bytes);
            slot[record - 4..].copy_from_slice(&table.scales[*shard].to_le_bytes());
        }
        Ok(out)
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
