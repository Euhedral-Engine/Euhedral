//! The key of a captured graph: everything its submissions bake in.
//!
//! A graph repeats the launches of one quantum, so it can serve another only when every launch parameter that is not
//! read from device memory is the same. Java's key (`Quantum.captureKey` with `CaptureFingerprint`) holds the kind,
//! rows, logits, selection, draft seed and token slots, the split counts of decode attention, and a fingerprint of the
//! workspace and the sequence's device allocations, each with its allocation's identity. This key holds the same, and
//! what the Rust engine bakes in beyond it: the allocation serials of the page tables (a table the cache outgrew is
//! another allocation), the host pages the staging copies span, and the rows each GDN layer replays first.

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
    /// The split counts of decode attention ([`split_geometry`]).
    pub split_geometry: u64,
    /// Host pages the reserved rows of the base cache and the MTP cache reach: the sizes of the staging copies.
    pub reserved_host_pages: [u64; 2],
    /// The rows each GDN layer replays first, by layer, without the trailing layers that replay none.
    pub replay_rows: Vec<u32>,
    /// The allocation serial of every buffer the graph reads or writes: the workspace, the sequence's GDN state and
    /// attention scratch, the selection words, and the page tables.
    pub buffer_generations: Vec<AllocationId>,
}

/// The split counts that size decode attention's launches (host dispatch in `qwen_layer_ops.c`): the rows below 2048
/// keys split into 48-key spans, those from 2048 keys (query-head groups up to 8) into 32-key spans, at most 64 each,
/// packed as `below << 8 | above`. Every other position-dependent value is read from the input record. A quantum of
/// `rows` rows at `start` attends over `start + 1` to `start + rows` keys.
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
    below << 8 | above
}

/// The rows each layer replays, without the trailing layers that replay none.
pub(crate) fn trimmed(replay: &[u32]) -> Vec<u32> {
    let keep = replay.iter().rposition(|rows| *rows > 0).map_or(0, |last| last + 1);
    replay[..keep].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    const GROUP: u64 = 6;

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
        assert_eq!(wanted, 64, "no span below, 64 of 32 keys");
        for start in (2047..140_000u64).step_by(7) {
            assert_eq!(split_geometry(start, 1, GROUP), wanted, "{start}");
        }
        for rows in 1..=8 {
            for start in [2047u64, 4096, 131_000] {
                assert_eq!(split_geometry(start, rows, GROUP), wanted, "{rows} rows at {start}");
            }
        }
    }

    #[test]
    fn rows_that_cross_2048_keys_have_both_counts() {
        assert_eq!(split_geometry(2044, 5, GROUP), 43 << 8 | 64);
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
