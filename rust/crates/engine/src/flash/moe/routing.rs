//! The host-side grouping of a block's routed (row, expert) pairs by expert, and the description the expert kernels and
//! the combine read (`ExpertRouting`).
//!
//! The router's output is `ids` and `weights`, `[rows][top_k]` row-major, ids in `[0, experts]` (the id `experts` is
//! padding and is skipped) and BF16 weight bits. [`Routing::plan`] groups the pairs by expert: the experts that
//! received pairs are the block's active experts, in ascending id order, and the `i`-th of them is computed on its own,
//! over work items of at most [`ITEM_PAIRS`] pairs that name slot `i` of the description's slot table, which holds the
//! expert's record address. A pair's place inside an expert is ascending row, then ascending position in its row. The
//! combine adds each row's pairs in ascending expert order, so the sum is the same whichever expert finishes first.
//!
//! The description is one block of host memory (little-endian, 16-byte aligned sections) that the device reads in place
//! after one copy ([`Routing::fill`] writes it as 32-bit words through a sink, so that the caller owns where they go):
//!
//! | section | content | bytes |
//! |---|---|---|
//! | slots | device address of each active expert's record | 8 per expert |
//! | items | per work item: slot index, first pair, pair count (1..16), 0 | 16 per item |
//! | pairs | per pair: row, BF16 routing weight (low 16 bits of a word) | 8 per pair |
//! | row offsets | `rows + 1` offsets into the row lists | 4 per row, plus one |
//! | row pairs | each row's pair indices, ascending expert order | 4 per pair |

use crate::flash::FlashError;

/// Pairs per work item: the token columns of two MMA tiles.
pub const ITEM_PAIRS: u32 = 16;

const OP: &str = "moe routing";

/// The sections of a description, by byte offset, for blocks of up to `max_rows` rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DescriptorLayout {
    experts: u32,
    top_k: u32,
    max_rows: u32,
    max_pairs: u32,
    max_active: u32,
    items: usize,
    pairs: usize,
    row_offsets: usize,
    row_pairs: usize,
    bytes: usize,
}

fn align(bytes: usize) -> usize {
    bytes.next_multiple_of(16)
}

impl DescriptorLayout {
    /// The layout of `experts` routed experts, `top_k` per row, up to `max_rows` rows. Refuses a zero extent and sizes
    /// that overflow 32 bits.
    pub fn new(experts: u32, top_k: u32, max_rows: u32) -> Result<Self, FlashError> {
        if experts == 0 || top_k == 0 || max_rows == 0 {
            return Err(FlashError::refused(
                OP,
                format!("{experts} experts, {top_k} per row, {max_rows} rows: every extent is positive"),
            ));
        }
        let max_pairs = max_rows
            .checked_mul(top_k)
            .filter(|pairs| *pairs <= u32::MAX / 16)
            .ok_or_else(|| FlashError::refused(OP, format!("{max_rows} rows of {top_k} experts overflow the pairs")))?;
        let max_active = experts.min(max_pairs);
        let item_capacity = (max_pairs / ITEM_PAIRS + max_active) as usize;
        let items = align(8 * max_active as usize);
        let pairs = align(items + 16 * item_capacity);
        let row_offsets = align(pairs + 8 * max_pairs as usize);
        let row_pairs = align(row_offsets + 4 * (max_rows as usize + 1));
        let bytes = align(row_pairs + 4 * max_pairs as usize);
        Ok(Self {
            experts,
            top_k,
            max_rows,
            max_pairs,
            max_active,
            items,
            pairs,
            row_offsets,
            row_pairs,
            bytes,
        })
    }

    /// The bytes of one description.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    pub fn experts(&self) -> u32 {
        self.experts
    }

    pub fn top_k(&self) -> u32 {
        self.top_k
    }

    pub fn max_rows(&self) -> u32 {
        self.max_rows
    }

    /// The most pairs a block can have.
    pub fn max_pairs(&self) -> u32 {
        self.max_pairs
    }

    /// The most experts a block can name: one per pair, at most every expert. This is the width of the layer's fetch
    /// and expert stages.
    pub fn max_active(&self) -> u32 {
        self.max_active
    }

    /// The byte offsets of the sections. The slot table is first.
    pub fn slots_offset(&self) -> usize {
        0
    }

    pub fn items_offset(&self) -> usize {
        self.items
    }

    pub fn pairs_offset(&self) -> usize {
        self.pairs
    }

    pub fn row_offsets_offset(&self) -> usize {
        self.row_offsets
    }

    pub fn row_pairs_offset(&self) -> usize {
        self.row_pairs
    }

    /// The byte offset of the slot of the `index`-th active expert.
    pub fn slot_offset(&self, index: u32) -> usize {
        8 * index as usize
    }

    /// The two words (index, value) that set the slot of the `index`-th active expert to `address`.
    pub fn slot_words(&self, index: u32, address: u64) -> [(usize, u32); 2] {
        let word = self.slot_offset(index) / 4;
        [(word, address as u32), (word + 1, (address >> 32) as u32)]
    }
}

/// The grouping of one block, with the scratch it takes preallocated.
#[derive(Debug)]
pub struct Routing {
    layout: DescriptorLayout,
    expert_start: Vec<u32>,
    expert_fill: Vec<u32>,
    sorted_row: Vec<u32>,
    sorted_weight: Vec<u32>,
    active: Vec<u32>,
    item_start: Vec<u32>,
    row_cursor: Vec<u32>,
    rows: u32,
    active_count: u32,
}

impl Routing {
    pub fn new(layout: DescriptorLayout) -> Self {
        let (experts, pairs, active) = (
            layout.experts as usize,
            layout.max_pairs as usize,
            layout.max_active as usize,
        );
        Self {
            layout,
            expert_start: vec![0; experts + 2],
            expert_fill: vec![0; experts + 1],
            sorted_row: vec![0; pairs],
            sorted_weight: vec![0; pairs],
            active: vec![0; active],
            item_start: vec![0; active + 1],
            row_cursor: vec![0; layout.max_rows as usize],
            rows: 0,
            active_count: 0,
        }
    }

    pub fn layout(&self) -> &DescriptorLayout {
        &self.layout
    }

    /// Groups the pairs of `rows` rows by expert: `id(i)` and `weight(i)` are the `i`-th entry of the router's output,
    /// `i < rows * top_k`. Refuses more rows than the layout holds and an id outside `[0, experts]`, naming it, and
    /// then holds no plan.
    pub fn plan(
        &mut self,
        rows: u32,
        id: impl Fn(usize) -> i32,
        weight: impl Fn(usize) -> u16,
    ) -> Result<(), FlashError> {
        self.rows = 0;
        self.active_count = 0;
        if rows > self.layout.max_rows {
            return Err(FlashError::refused(
                OP,
                format!("{rows} rows exceed the {} the layout holds", self.layout.max_rows),
            ));
        }
        let experts = self.layout.experts;
        let top_k = self.layout.top_k as usize;
        let pairs = rows as usize * top_k;
        self.expert_start.fill(0);
        for i in 0..pairs {
            let found = id(i);
            match u32::try_from(found) {
                Ok(found) if found < experts => self.expert_start[found as usize + 2] += 1,
                Ok(found) if found == experts => {}
                _ => {
                    self.expert_start.fill(0);
                    return Err(FlashError::refused(
                        OP,
                        format!("expert id {found} at pair {i} is outside [0, {experts}]"),
                    ));
                }
            }
        }
        // `expert_start[e + 1]` is the first pair of expert `e` after the prefix sum; `expert_fill` walks it.
        for e in 0..experts as usize {
            self.expert_start[e + 2] += self.expert_start[e + 1];
        }
        self.expert_fill[..experts as usize].copy_from_slice(&self.expert_start[1..=experts as usize]);
        for i in 0..pairs {
            let found = id(i) as u32;
            if found == experts {
                continue;
            }
            let at = self.expert_fill[found as usize] as usize;
            self.expert_fill[found as usize] += 1;
            self.sorted_row[at] = (i / top_k) as u32;
            self.sorted_weight[at] = u32::from(weight(i));
        }
        let mut items = 0;
        for e in 0..experts as usize {
            let count = self.expert_start[e + 2] - self.expert_start[e + 1];
            if count == 0 {
                continue;
            }
            self.item_start[self.active_count as usize] = items;
            self.active[self.active_count as usize] = e as u32;
            self.active_count += 1;
            items += count.div_ceil(ITEM_PAIRS);
        }
        self.item_start[self.active_count as usize] = items;
        self.rows = rows;
        Ok(())
    }

    pub fn rows(&self) -> u32 {
        self.rows
    }

    /// The experts that received at least one pair.
    pub fn active_experts(&self) -> u32 {
        self.active_count
    }

    /// The id of the `index`-th active expert (ascending), or `None` past the active ones.
    pub fn active_expert(&self, index: u32) -> Option<u32> {
        (index < self.active_count).then(|| self.active[index as usize])
    }

    /// The first work item of the `index`-th active expert.
    pub fn item_start(&self, index: u32) -> Option<u32> {
        (index < self.active_count).then(|| self.item_start[index as usize])
    }

    /// How many work items the `index`-th active expert has.
    pub fn item_count(&self, index: u32) -> Option<u32> {
        (index < self.active_count).then(|| self.item_start[index as usize + 1] - self.item_start[index as usize])
    }

    /// The pairs of the block that were not padding.
    pub fn pair_count(&self) -> u32 {
        self.expert_start[self.layout.experts as usize + 1]
    }

    /// Writes the description as `(word index, value)` into `put`: everything but the slot addresses. Every word of the
    /// sections the block uses is written, and the row offsets of the rows past the block's hold its pair count.
    pub fn fill(&mut self, put: &mut impl FnMut(usize, u32)) {
        let layout = self.layout;
        let word = |offset: usize, index: usize, field: usize| (offset + index) / 4 + field;
        let mut item = 0;
        for index in 0..self.active_count as usize {
            let e = self.active[index] as usize;
            let begin = self.expert_start[e + 1];
            let count = self.expert_start[e + 2] - begin;
            for group in (0..count).step_by(ITEM_PAIRS as usize) {
                let at = layout.items + 16 * item;
                put(at / 4, index as u32);
                put(at / 4 + 1, begin + group);
                put(at / 4 + 2, ITEM_PAIRS.min(count - group));
                put(at / 4 + 3, 0);
                item += 1;
            }
        }
        let pairs = self.pair_count() as usize;
        for p in 0..pairs {
            put(word(layout.pairs, 8 * p, 0), self.sorted_row[p]);
            put(word(layout.pairs, 8 * p, 1), self.sorted_weight[p]);
        }
        // The row lists: a counting sort of the pairs by row, ascending pair index (ascending expert) inside a row.
        let rows = self.rows as usize;
        self.row_cursor[..rows].fill(0);
        for p in 0..pairs {
            self.row_cursor[self.sorted_row[p] as usize] += 1;
        }
        let mut running = 0;
        for r in 0..rows {
            put(word(layout.row_offsets, 4 * r, 0), running);
            let count = self.row_cursor[r];
            self.row_cursor[r] = running;
            running += count;
        }
        for r in rows..=layout.max_rows as usize {
            put(word(layout.row_offsets, 4 * r, 0), running);
        }
        for p in 0..pairs {
            let row = self.sorted_row[p] as usize;
            put(word(layout.row_pairs, 4 * self.row_cursor[row] as usize, 0), p as u32);
            self.row_cursor[row] += 1;
        }
    }
}

#[cfg(test)]
mod tests;
