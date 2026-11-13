//! The key of a captured graph: everything its submissions bake in.
//!
//! A graph repeats the launches of one quantum, so it can serve another only when every launch parameter that is not
//! read from device memory is the same. Java's key (`Quantum.captureKey` with `CaptureFingerprint`) holds the kind,
//! rows, logits, selection, draft seed and token slots, the split counts of decode attention, and a fingerprint of the
//! workspace and the sequence's device allocations, each with its allocation's identity. This key holds the same, and
//! what the Rust engine bakes in beyond it: the allocation serials of the page tables (a table the cache outgrew is
//! another allocation), the host pages the staging copies span, and the rows each GDN layer replays first.

use euhedral_hashing::hasher_api::hash_bytes;
use euhedral_inference_cuda::AllocationId;

use crate::plan::View;
use crate::stages::QuantumKind;

/// Keys with the same value record and replay the same graph.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CaptureKey {
    pub view: View,
    pub kind: QuantumKind,
    pub rows: usize,
    /// The logits rows the quantum scores.
    pub logits: usize,
    /// Whether the quantum selects: its argmax and the copy of its keys are in the graph.
    pub selects: bool,
    pub seeds_draft: bool,
    /// The address of the rows that seed a draft quantum (a recursive draft's hidden).
    pub draft_seed: Option<u64>,
    /// The slot of the step's tokens the rows read their ids from.
    pub draft_read: Option<u8>,
    /// The slot a drafted token is passed on to, and the address of the head's row ids that map it.
    pub draft_write: Option<(u8, u64)>,
    pub writes_current_token: bool,
    /// The input record the quantum's ids and positions are read from: its launches bake the address.
    pub input_record: u8,
    /// The selection whose keys the quantum's token is read from, on the device: its launch bakes the address.
    pub token_from: Option<AllocationId>,
    /// Whether the quantum runs row-exact (a verification): the Linear stages take other routes.
    pub row_exact: bool,
    /// The split counts of decode attention and whether the fused verifier serves the rows ([`split_geometry`]).
    pub split_geometry: u64,
    /// Host pages the reserved rows of the base cache and the MTP cache reach: the sizes of the staging copies.
    pub reserved_host_pages: [u64; 2],
    /// The rows each GDN layer replays first, by layer, without the trailing layers that replay none.
    pub replay_rows: Vec<u32>,
    /// The allocation serial of every buffer the graph reads or writes: the workspace, the sequence's GDN state and
    /// attention scratch, the selection words, and the page tables.
    pub buffer_generations: Vec<AllocationId>,
}

impl CaptureKey {
    /// Whether `other` is the key of the same quantum at another point of the sequence: equal in everything but the
    /// attention geometry.
    pub fn same_but_geometry(&self, other: &CaptureKey) -> bool {
        self.view == other.view
            && self.kind == other.kind
            && self.rows == other.rows
            && self.logits == other.logits
            && self.selects == other.selects
            && self.seeds_draft == other.seeds_draft
            && self.draft_seed == other.draft_seed
            && self.draft_read == other.draft_read
            && self.draft_write == other.draft_write
            && self.writes_current_token == other.writes_current_token
            && self.input_record == other.input_record
            && self.token_from == other.token_from
            && self.row_exact == other.row_exact
            && self.reserved_host_pages == other.reserved_host_pages
            && self.replay_rows == other.replay_rows
            && self.buffer_generations == other.buffer_generations
    }
    /// A 64-bit digest of every field of the key, for a cache keyed by a fixed-width integer.
    ///
    /// The key is destructured without `..`, so a field added to it does not compile until it is hashed here. Each
    /// field is written at a fixed width, enums with a tag and options with a presence byte, and each variable-length
    /// part with its length first, so no two keys differ in their bytes without differing in the digest's input. The
    /// digest is xxHash64 of those bytes (`euhedral-hashing`), which is fixed by value and does not depend on the
    /// process, the platform's endianness or the hasher's random state: it is stable across processes and runs. Equal
    /// keys give equal digests; distinct keys collide only by chance of a 64-bit hash.
    pub fn digest(&self) -> u64 {
        let CaptureKey {
            view,
            kind,
            rows,
            logits,
            selects,
            seeds_draft,
            draft_seed,
            draft_read,
            draft_write,
            writes_current_token,
            input_record,
            token_from,
            row_exact,
            split_geometry,
            reserved_host_pages,
            replay_rows,
            buffer_generations,
        } = self;
        let mut bytes = Vec::with_capacity(160 + 4 * replay_rows.len() + 8 * buffer_generations.len());
        let (view_tag, view_rows) = match *view {
            View::PrefillChunk => (0u8, 0),
            View::Decode => (1, 0),
            View::Verify { rows } => (2, rows),
            View::Draft { rows } => (3, rows),
            View::CatchUp { rows } => (4, rows),
        };
        bytes.push(view_tag);
        bytes.extend_from_slice(&(view_rows as u64).to_le_bytes());
        bytes.push(match kind {
            QuantumKind::Prefill => 0,
            QuantumKind::Decode => 1,
            QuantumKind::Verify => 2,
            QuantumKind::Draft => 3,
        });
        bytes.extend_from_slice(&(*rows as u64).to_le_bytes());
        bytes.extend_from_slice(&(*logits as u64).to_le_bytes());
        bytes.push(u8::from(*selects));
        bytes.push(u8::from(*seeds_draft));
        push_option(&mut bytes, draft_seed.map(u64::to_le_bytes));
        push_option(&mut bytes, draft_read.map(|slot| [slot]));
        match draft_write {
            Some((slot, address)) => {
                bytes.push(1);
                bytes.push(*slot);
                bytes.extend_from_slice(&address.to_le_bytes());
            }
            None => bytes.push(0),
        }
        bytes.push(u8::from(*writes_current_token));
        bytes.push(*input_record);
        push_option(&mut bytes, token_from.map(|id| id.0.to_le_bytes()));
        bytes.push(u8::from(*row_exact));
        bytes.extend_from_slice(&split_geometry.to_le_bytes());
        for pages in reserved_host_pages {
            bytes.extend_from_slice(&pages.to_le_bytes());
        }
        bytes.extend_from_slice(&(replay_rows.len() as u64).to_le_bytes());
        for rows in replay_rows {
            bytes.extend_from_slice(&rows.to_le_bytes());
        }
        bytes.extend_from_slice(&(buffer_generations.len() as u64).to_le_bytes());
        for id in buffer_generations {
            bytes.extend_from_slice(&id.0.to_le_bytes());
        }
        hash_bytes(&bytes)
    }
}

/// Writes an option as a presence byte and, when present, its bytes.
fn push_option<const N: usize>(bytes: &mut Vec<u8>, value: Option<[u8; N]>) {
    match value {
        Some(value) => {
            bytes.push(1);
            bytes.extend_from_slice(&value);
        }
        None => bytes.push(0),
    }
}

/// The fewest keys at which the fused verifier kernel serves rows of a verification or draft, by rows
/// (`fused_from` in `qwen_layer_ops.c`: rows that fill their 8-column mma tiles win from 2048 keys, the others on
/// longer caches).
fn fused_from(rows: u64) -> u64 {
    const FROM: [u64; 9] = [u64::MAX, u64::MAX, 49152, 16384, 4096, 2048, 16384, 16384, 12288];
    usize::try_from(rows)
        .ok()
        .and_then(|rows| FROM.get(rows).copied())
        .unwrap_or(u64::MAX)
}

/// What the host dispatch of decode attention (`attention_causal` in `qwen_layer_ops.c`) bakes into its launches for a
/// quantum of `rows` rows at `start`, which attends over `start + 1` to `start + rows` keys, packed as
/// `fused << 16 | below << 8 | above`:
/// - the split counts: rows below 2048 keys split into 48-key spans, those from 2048 keys (query-head groups up to
///   8) into 32-key spans, at most 64 each;
/// - whether the fused verifier kernel runs for the rows from 2048 keys (two to eight rows whose mma tiles fit, and
///   whose last row reaches the keys from which the fused kernel is faster than the row twins), a different launch
///   from the row twin's.
///
/// Every other position-dependent value is read from the input record.
pub fn split_geometry(start: u64, rows: u64, group: u64) -> u64 {
    let first = start + 1;
    let last = start + rows;
    let from: u64 = if group <= 8 { 2048 } else { u64::MAX };
    let below = if first < from {
        64.min(last.min(from - 1).div_ceil(48))
    } else {
        0
    };
    let above = if last >= from { 64.min(last.div_ceil(32)) } else { 0 };
    let tiles = (rows * group).div_ceil(8).max(2);
    let fused = rows > 1 && last >= from && tiles <= 6 && last >= fused_from(rows);
    u64::from(fused) << 16 | below << 8 | above
}

/// How far a sequence has come through the geometries of one view: a geometry of a lower rank is one the sequence has
/// passed. The split counts below 2048 keys rise with the position, then the quanta that cross 2048 keys carry both,
/// and from there the counts are constant (the fused verifier comes in at its own thresholds).
pub fn geometry_rank(geometry: u64) -> u64 {
    let (fused, below, above) = (geometry >> 16, geometry >> 8 & 0xff, geometry & 0xff);
    if below == 0 {
        1 << 24 | fused << 16 | above
    } else {
        below << 8 | above
    }
}

/// The rows each layer replays, without the trailing layers that replay none.
pub(crate) fn trimmed(replay: &[u32]) -> Vec<u32> {
    let keep = replay.iter().rposition(|rows| *rows > 0).map_or(0, |last| last + 1);
    replay[..keep].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    type Change = (&'static str, fn(&mut CaptureKey));

    const GROUP: u64 = 6;

    fn key() -> CaptureKey {
        CaptureKey {
            view: View::Verify { rows: 4 },
            kind: QuantumKind::Verify,
            rows: 4,
            logits: 4,
            selects: true,
            seeds_draft: false,
            draft_seed: Some(0x10),
            draft_read: Some(1),
            draft_write: Some((2, 0x20)),
            writes_current_token: false,
            input_record: 1,
            token_from: Some(AllocationId(7)),
            row_exact: true,
            split_geometry: 64,
            reserved_host_pages: [3, 5],
            replay_rows: vec![0, 2],
            buffer_generations: vec![AllocationId(1), AllocationId(2)],
        }
    }

    #[test]
    fn equal_keys_have_equal_digests() {
        assert_eq!(key().digest(), key().clone().digest());
    }

    #[test]
    fn changing_any_one_field_changes_the_digest() {
        let base = key().digest();
        let changes: Vec<Change> = vec![
            ("view", |k| k.view = View::Draft { rows: 4 }),
            ("view rows", |k| k.view = View::Verify { rows: 5 }),
            ("kind", |k| k.kind = QuantumKind::Draft),
            ("rows", |k| k.rows = 5),
            ("logits", |k| k.logits = 5),
            ("selects", |k| k.selects = false),
            ("seeds_draft", |k| k.seeds_draft = true),
            ("draft_seed", |k| k.draft_seed = Some(0x11)),
            ("draft_seed absent", |k| k.draft_seed = None),
            ("draft_read", |k| k.draft_read = Some(2)),
            ("draft_write slot", |k| k.draft_write = Some((3, 0x20))),
            ("draft_write address", |k| k.draft_write = Some((2, 0x21))),
            ("draft_write absent", |k| k.draft_write = None),
            ("writes_current_token", |k| k.writes_current_token = true),
            ("input_record", |k| k.input_record = 0),
            ("token_from", |k| k.token_from = Some(AllocationId(8))),
            ("token_from absent", |k| k.token_from = None),
            ("row_exact", |k| k.row_exact = false),
            ("split_geometry", |k| k.split_geometry = 65),
            ("reserved_host_pages first", |k| k.reserved_host_pages = [4, 5]),
            ("reserved_host_pages second", |k| k.reserved_host_pages = [3, 6]),
            ("replay_rows", |k| k.replay_rows = vec![0, 3]),
            ("buffer_generations", |k| {
                k.buffer_generations = vec![AllocationId(1), AllocationId(3)]
            }),
        ];
        for (name, change) in changes {
            let mut other = key();
            change(&mut other);
            assert_ne!(other.digest(), base, "{name}");
        }
    }

    #[test]
    fn variable_length_parts_do_not_run_into_each_other() {
        let mut a = key();
        a.replay_rows = vec![1, 2];
        a.buffer_generations = vec![AllocationId(3)];
        let mut b = key();
        b.replay_rows = vec![1];
        b.buffer_generations = vec![AllocationId(2), AllocationId(3)];
        assert_ne!(a.digest(), b.digest());
    }

    #[test]
    fn the_split_counts_change_every_48_positions_below_2048_keys() {
        let mut changes = Vec::new();
        let mut previous = split_geometry(0, 1, GROUP);
        assert_eq!(previous, 1 << 8, "one key, one split of 48");
        for start in 1..2047u64 {
            let geometry = split_geometry(start, 1, GROUP);
            if geometry != previous {
                changes.push(start);
                previous = geometry;
            }
        }
        assert_eq!(changes.len(), 42, "43 values in all");
        assert!(changes.iter().all(|start| start % 48 == 0), "{changes:?}");
        assert_eq!(changes[0], 48);
        assert_eq!(previous, 43 << 8, "2047 keys take 43 splits of 48");
    }

    #[test]
    fn the_split_counts_are_constant_from_2048_keys() {
        let wanted = split_geometry(2047, 1, GROUP);
        assert_eq!(
            wanted, 64,
            "no span below, 64 of 32 keys, and one row never takes the fused verifier"
        );
        for start in (2047..140_000u64).step_by(7) {
            assert_eq!(split_geometry(start, 1, GROUP), wanted, "{start}");
        }
        for rows in 2..=8u64 {
            // Past every fused threshold the key is constant too.
            let beyond = split_geometry(131_000, rows, GROUP);
            assert_eq!(beyond, split_geometry(100_000, rows, GROUP), "{rows} rows");
            assert_eq!(beyond & 0xffff, 64, "{rows} rows");
        }
    }

    /// The thresholds of `fused_from` in `qwen_layer_ops.c` and the tiles of rows x 6 query heads per KV head.
    #[test]
    fn the_fused_verifier_choice_flips_at_its_thresholds_for_two_to_eight_rows() {
        let thresholds = [
            (2u64, 49152u64),
            (3, 16384),
            (4, 4096),
            (5, 2048),
            (6, 16384),
            (7, 16384),
            (8, 12288),
        ];
        for (rows, from) in thresholds {
            let mut flips = Vec::new();
            let mut previous = split_geometry(2047, rows, GROUP) >> 16;
            assert_eq!(previous, u64::from(from <= 2047 + rows), "{rows} rows");
            for start in 2048..70_000u64 {
                let fused = split_geometry(start, rows, GROUP) >> 16;
                if fused != previous {
                    flips.push(start);
                    previous = fused;
                }
            }
            // The last row reaches `from` keys at start + rows = from.
            let expected = if from > 2047 + rows {
                vec![from - rows]
            } else {
                Vec::new()
            };
            assert_eq!(
                flips, expected,
                "{rows} rows flip where the last row reaches {from} keys"
            );
            assert_eq!(previous, 1, "{rows} rows end on the fused verifier");
        }
        // One row has no fused verifier, and a group of more than 8 has no GQA kernel at all.
        assert_eq!(split_geometry(60_000, 1, GROUP) >> 16, 0);
        assert_eq!(split_geometry(60_000, 4, 9) >> 16, 0);
    }

    #[test]
    fn the_rank_orders_a_sequence_through_its_geometries() {
        let mut previous = 0;
        for start in 0..70_000u64 {
            let rank = geometry_rank(split_geometry(start, 5, GROUP));
            assert!(rank >= previous, "{start}");
            previous = rank;
        }
        assert!(geometry_rank(split_geometry(0, 1, GROUP)) < geometry_rank(split_geometry(2047, 1, GROUP)));
    }

    #[test]
    fn rows_that_cross_2048_keys_have_both_counts() {
        // 2044 + 5 reaches 2049 keys: 5 rows serve the fused verifier from 2048 keys.
        assert_eq!(split_geometry(2044, 5, GROUP), 1 << 16 | 43 << 8 | 64);
        // A group wider than 8 never takes the 32-key spans.
        assert_eq!(split_geometry(5000, 1, 9), 64 << 8);
    }

    #[test]
    fn replay_rows_compare_without_their_trailing_zeros() {
        assert_eq!(trimmed(&[0, 3, 0, 0]), vec![0, 3]);
        assert_eq!(trimmed(&[0, 0]), Vec::<u32>::new());
        assert_eq!(trimmed(&[]), Vec::<u32>::new());
    }
}
