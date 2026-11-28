//! The key of a captured graph: everything its submissions bake in.
//!
//! A graph repeats the launches of one quantum, so it can serve another only when every launch parameter that is not
//! read from device memory is the same. Java's key (`Quantum.captureKey` with `CaptureFingerprint`) holds the kind,
//! rows, logits, selection, draft seed and token slots, the split counts of decode attention, and a fingerprint of the
//! workspace and the sequence's device allocations, each with its allocation's identity. This key holds the same, and
//! what the Rust engine bakes in beyond it: the allocation serials of the page tables (a table the cache outgrew is
//! another allocation), the host pages the staging copies span, and the rows each GDN layer replays first.

use euhedral_hashing::hasher_api::hash_bytes_with_seed;
use euhedral_inference_cuda::AllocationId;
use euhedral_inference_cuda::launch::attention::FUSED_ROWS;

use crate::plan::View;
use crate::stages::QuantumKind;
use crate::stages::sampling::SelectedRows;

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
    /// Which logits rows the quantum copies to the host's readback (a sampling request's quantum, whose graph holds
    /// the copy and no argmax): the readback's allocation is among the `buffer_generations`.
    pub copies_rows: Option<SelectedRows>,
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
    /// Whether the launches the catalog marks overlap their predecessors (programmatic dependent launch): a recording
    /// bakes the launch attribute.
    pub programmatic: bool,
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
    /// A 128-bit digest of every field of the key, for a cache that replays a hit without checking the body.
    ///
    /// The key is destructured without `..`, so a field added to it does not compile until it is hashed here. Each
    /// field is written at a fixed width, enums with a tag and options with a presence byte, and each variable-length
    /// part with its length first, so no two keys differ in their bytes without differing in the digest's input. The
    /// digest is built from two xxHash64 passes (`euhedral-hashing` has no 128-bit hash) over those bytes with two different
    /// fixed seeds, the first in the high half and the second in the low half. xxHash64 is fixed by value and does not depend on the
    /// process, the platform's endianness or the hasher's random state: it is stable across processes and runs. Equal
    /// keys give equal digests; distinct keys collide only by chance of a 128-bit hash.
    pub fn digest(&self) -> u128 {
        let CaptureKey {
            view,
            kind,
            rows,
            logits,
            selects,
            copies_rows,
            seeds_draft,
            draft_seed,
            draft_read,
            draft_write,
            writes_current_token,
            input_record,
            token_from,
            row_exact,
            programmatic,
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
            View::Context { rows } => (5, rows),
            View::Block => (6, 0),
        };
        bytes.push(view_tag);
        bytes.extend_from_slice(&(view_rows as u64).to_le_bytes());
        bytes.push(match kind {
            QuantumKind::Prefill => 0,
            QuantumKind::Decode => 1,
            QuantumKind::Verify => 2,
            QuantumKind::Draft => 3,
            QuantumKind::Context => 4,
            QuantumKind::Block => 5,
        });
        bytes.extend_from_slice(&(*rows as u64).to_le_bytes());
        bytes.extend_from_slice(&(*logits as u64).to_le_bytes());
        bytes.push(u8::from(*selects));
        bytes.push(match copies_rows {
            None => 0,
            Some(SelectedRows::Last) => 1,
            Some(SelectedRows::Each) => 2,
        });
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
        bytes.push(u8::from(*programmatic));
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
        let high = hash_bytes_with_seed(&bytes, HIGH_SEED);
        let low = hash_bytes_with_seed(&bytes, LOW_SEED);
        u128::from(high) << 64 | u128::from(low)
    }
}

/// The seeds of the two passes of [`CaptureKey::digest`].
const HIGH_SEED: u64 = 0x9e37_79b9_7f4a_7c15;
const LOW_SEED: u64 = 0xc2b2_ae3d_27d4_eb4f;

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

/// The fewest keys at which the fused verifier kernel serves `rows` rows under `table` (`kernels.fused-verify-from`);
/// `u32::MAX` is never.
fn fused_from(table: &[u32; FUSED_ROWS], rows: u64) -> u64 {
    match usize::try_from(rows).ok().and_then(|rows| table.get(rows)) {
        Some(&keys) if keys != u32::MAX => u64::from(keys),
        _ => u64::MAX,
    }
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
/// `fused_from_keys` is the kernels' `fused-verify-from` table. Every other position-dependent value is read from the
/// input record.
pub fn split_geometry(start: u64, rows: u64, group: u64, fused_from_keys: &[u32; FUSED_ROWS]) -> u64 {
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
    let fused = rows > 1 && last >= from && tiles <= 6 && last >= fused_from(fused_from_keys, rows);
    u64::from(fused) << 16 | below << 8 | above
}

/// The split geometry a key names: [`split_geometry`] for a quantum whose attention reads keys of a cache, and 0 for one
/// with no such attention (the drafter's context and block, a model with no attention layers), whose launches bake no key
/// count, so its position must not re-key it.
pub fn keyed_split_geometry(
    attends_cache: bool,
    start: u64,
    rows: u64,
    group: u64,
    fused_from_keys: &[u32; FUSED_ROWS],
) -> u64 {
    if attends_cache {
        split_geometry(start, rows, group, fused_from_keys)
    } else {
        0
    }
}

/// The rows each layer replays, without the trailing layers that replay none.
pub(crate) fn trimmed(replay: &[u32]) -> Vec<u32> {
    let keep = replay.iter().rposition(|rows| *rows > 0).map_or(0, |last| last + 1);
    replay[..keep].to_vec()
}

#[cfg(test)]
mod tests {
    use euhedral_inference_cuda::launch::attention::DEFAULT_FUSED_VERIFY_FROM;

    use super::*;

    fn default_geometry(start: u64, rows: u64, group: u64) -> u64 {
        split_geometry(start, rows, group, &DEFAULT_FUSED_VERIFY_FROM)
    }

    #[test]
    fn a_configured_table_moves_where_the_key_says_the_fused_verifier_runs() {
        // Rows 4, group 6: fused from 4096 keys by default, from 2048 under an earlier table, never under a late one.
        let mut early = DEFAULT_FUSED_VERIFY_FROM;
        early[4] = 2048;
        let mut never = DEFAULT_FUSED_VERIFY_FROM;
        never[4] = u32::MAX;
        let fused = |table: &[u32; FUSED_ROWS], last: u64| split_geometry(last - 4, 4, GROUP, table) >> 16 == 1;
        assert!(!fused(&DEFAULT_FUSED_VERIFY_FROM, 3000));
        assert!(fused(&early, 3000));
        assert!(!fused(&early, 2047));
        assert!(fused(&DEFAULT_FUSED_VERIFY_FROM, 5000));
        assert!(!fused(&never, 5000));
    }

    type Change = (&'static str, fn(&mut CaptureKey));

    const GROUP: u64 = 6;

    fn key() -> CaptureKey {
        CaptureKey {
            view: View::Verify { rows: 4 },
            kind: QuantumKind::Verify,
            rows: 4,
            logits: 4,
            selects: true,
            copies_rows: None,
            seeds_draft: false,
            draft_seed: Some(0x10),
            draft_read: Some(1),
            draft_write: Some((2, 0x20)),
            writes_current_token: false,
            input_record: 1,
            token_from: Some(AllocationId(7)),
            row_exact: true,
            programmatic: true,
            split_geometry: 64,
            reserved_host_pages: [3, 5],
            replay_rows: vec![0, 2],
            buffer_generations: vec![AllocationId(1), AllocationId(2)],
        }
    }

    #[test]
    fn the_two_halves_of_the_digest_differ() {
        let digest = key().digest();
        assert_ne!(digest >> 64, digest & u128::from(u64::MAX));
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
            ("copies_rows last", |k| k.copies_rows = Some(SelectedRows::Last)),
            ("copies_rows each", |k| k.copies_rows = Some(SelectedRows::Each)),
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
            ("programmatic", |k| k.programmatic = false),
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
    fn a_sampled_verification_is_keyed_apart_from_a_greedy_one_and_the_greedy_key_names_no_copy() {
        let greedy = key();
        assert!(
            greedy.selects && greedy.copies_rows.is_none(),
            "a greedy quantum selects on the device"
        );
        // A sampled verification copies every row to its readback and runs no argmax; its graph holds another readback.
        let mut sampled = key();
        sampled.selects = false;
        sampled.copies_rows = Some(SelectedRows::Each);
        sampled.buffer_generations.push(AllocationId(9));
        assert_ne!(sampled.digest(), greedy.digest());
        // Two readbacks are two graphs: the launch bakes the address in.
        let mut other = sampled.clone();
        other.buffer_generations.pop();
        other.buffer_generations.push(AllocationId(10));
        assert_ne!(sampled.digest(), other.digest());
        // A decode that copies its last row is not a verification that copies each.
        let mut decode = sampled.clone();
        decode.copies_rows = Some(SelectedRows::Last);
        assert_ne!(sampled.digest(), decode.digest());
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
        let mut previous = default_geometry(0, 1, GROUP);
        assert_eq!(previous, 1 << 8, "one key, one split of 48");
        for start in 1..2047u64 {
            let geometry = default_geometry(start, 1, GROUP);
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
        let wanted = default_geometry(2047, 1, GROUP);
        assert_eq!(
            wanted, 64,
            "no span below, 64 of 32 keys, and one row never takes the fused verifier"
        );
        for start in (2047..140_000u64).step_by(7) {
            assert_eq!(default_geometry(start, 1, GROUP), wanted, "{start}");
        }
        for rows in 2..=8u64 {
            // Past every fused threshold the key is constant too.
            let beyond = default_geometry(131_000, rows, GROUP);
            assert_eq!(beyond, default_geometry(100_000, rows, GROUP), "{rows} rows");
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
            let mut previous = default_geometry(2047, rows, GROUP) >> 16;
            assert_eq!(previous, u64::from(from <= 2047 + rows), "{rows} rows");
            for start in 2048..70_000u64 {
                let fused = default_geometry(start, rows, GROUP) >> 16;
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
        assert_eq!(default_geometry(60_000, 1, GROUP) >> 16, 0);
        assert_eq!(default_geometry(60_000, 4, 9) >> 16, 0);
    }

    #[test]
    fn rows_that_cross_2048_keys_have_both_counts() {
        // 2044 + 5 reaches 2049 keys: 5 rows serve the fused verifier from 2048 keys.
        assert_eq!(default_geometry(2044, 5, GROUP), 1 << 16 | 43 << 8 | 64);
        // A group wider than 8 never takes the 32-key spans.
        assert_eq!(default_geometry(5000, 1, 9), 64 << 8);
    }

    #[test]
    fn a_quantum_with_no_cache_attention_is_keyed_the_same_at_every_position() {
        for rows in [1, 8] {
            let wanted = keyed_split_geometry(false, 0, rows, GROUP, &DEFAULT_FUSED_VERIFY_FROM);
            for start in [1, 47, 48, 2047, 9000, 131_000] {
                assert_eq!(
                    keyed_split_geometry(false, start, rows, GROUP, &DEFAULT_FUSED_VERIFY_FROM),
                    wanted,
                    "{rows} rows at {start}"
                );
            }
        }
        assert_eq!(
            keyed_split_geometry(true, 100, 1, GROUP, &DEFAULT_FUSED_VERIFY_FROM),
            default_geometry(100, 1, GROUP)
        );
    }

    #[test]
    fn replay_rows_compare_without_their_trailing_zeros() {
        assert_eq!(trimmed(&[0, 3, 0, 0]), vec![0, 3]);
        assert_eq!(trimmed(&[0, 0]), Vec::<u32>::new());
        assert_eq!(trimmed(&[]), Vec::<u32>::new());
    }
}
