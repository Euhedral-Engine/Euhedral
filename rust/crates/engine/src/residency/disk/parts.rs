//! A direct read of a window slot, cut into parts that the ring reads side by side.
//!
//! The ring reads a region in pieces of a quarter of a MiB, each piece's completion submitting the next, so one large read
//! keeps a single small request on the device at a time. A direct read has no page cache to batch it, so it is cut here
//! into parts (each its own read, each landing in its own range of the slot) and the slot's read ends when its last part
//! has. [`parts_of`] decides the cut, [`Join`] collects the parts' ends into the one result the reader is told.

use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};

use super::ReadDone;
use crate::handoff::Handoff;
use crate::residency::offload::HOST_ALIGNMENT;

/// The most bytes one part of a direct read holds (a multiple of [`HOST_ALIGNMENT`]): a read of up to this many bytes is one
/// part, a larger one is cut into as many parts as it takes, as even as the blocks allow.
pub(super) const PART_BYTES: usize = 4 << 20;

/// The parts a direct read of `len` bytes (whole blocks) is cut into: `(start, len)` within the read, in order, adjacent,
/// each a whole number of blocks and none longer than [`PART_BYTES`]; none for an empty read.
pub(super) fn parts_of(len: usize) -> Vec<(usize, usize)> {
    debug_assert!(
        len.is_multiple_of(HOST_ALIGNMENT as usize),
        "a direct read is whole blocks"
    );
    let count = len.div_ceil(PART_BYTES);
    if count == 0 {
        return Vec::new();
    }
    let block = HOST_ALIGNMENT as usize;
    let each = (len / block).div_ceil(count) * block;
    (0..count)
        .map(|part| {
            let start = part * each;
            (start, each.min(len - start))
        })
        .filter(|&(_, bytes)| bytes > 0)
        .collect()
}

/// What the last part to end hands on: the reader's completion and the read's result.
pub(super) type Ended = (ReadDone, io::Result<usize>);

/// The parts of one read and how they ended. Shared by the parts' completions; no call waits.
pub(super) struct Join {
    /// Each part's length.
    lens: Vec<usize>,
    /// Each part's end, until the join is complete.
    results: Vec<Handoff<io::Result<usize>>>,
    /// The parts that have not ended.
    left: AtomicUsize,
    done: Handoff<ReadDone>,
}

impl Join {
    /// A join of parts of `lens` bytes that tells `done` when they all ended.
    pub(super) fn new(lens: Vec<usize>, done: ReadDone) -> Join {
        debug_assert!(!lens.is_empty(), "a read has a part");
        let handoff = Handoff::empty();
        handoff.send(done);
        Join {
            results: lens.iter().map(|_| Handoff::empty()).collect(),
            left: AtomicUsize::new(lens.len()),
            lens,
            done: handoff,
        }
    }

    /// Part `part` ended with `result`. When it was the last, the completion to run and the read's result: the bytes
    /// moved from the start of the read up to the first part that fell short, or the error of the first part that failed
    /// before it.
    pub(super) fn ended(&self, part: usize, result: io::Result<usize>) -> Option<Ended> {
        self.results[part].send(result);
        // AcqRel: the last part to end sees every other part's result.
        if self.left.fetch_sub(1, Ordering::AcqRel) > 1 {
            return None;
        }
        let mut moved = 0;
        for (part, result) in self.results.iter().enumerate() {
            // Every part ended before the last one counted down.
            match result.take().unwrap_or(Ok(0)) {
                Ok(bytes) => {
                    moved += bytes;
                    if bytes < self.lens[part] {
                        break;
                    }
                }
                Err(failure) => return self.done.take().map(|done| (done, Err(failure))),
            }
        }
        self.done.take().map(|done| (done, Ok(moved)))
    }

    /// The ring refused part `from` and none of the parts after it started: they all end with `refusal` (the first with the
    /// error itself).
    pub(super) fn refused(&self, from: usize, refusal: io::Error) -> Option<Ended> {
        let (kind, why) = (refusal.kind(), refusal.to_string());
        let mut ended = self.ended(from, Err(refusal));
        for part in from + 1..self.lens.len() {
            ended = self.ended(part, Err(io::Error::new(kind, why.clone()))).or(ended);
        }
        ended
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    const BLOCK: usize = HOST_ALIGNMENT as usize;

    fn counted() -> (ReadDone, Arc<AtomicUsize>) {
        let calls = Arc::new(AtomicUsize::new(0));
        let told = Arc::clone(&calls);
        (
            Box::new(move |_| {
                told.fetch_add(1, Ordering::SeqCst);
            }),
            calls,
        )
    }

    fn result_of(ended: Option<Ended>) -> io::Result<usize> {
        ended.expect("the last part ended the read").1
    }

    #[test]
    fn a_read_within_one_part_is_not_cut() {
        assert_eq!(parts_of(BLOCK), vec![(0, BLOCK)]);
        assert_eq!(parts_of(PART_BYTES), vec![(0, PART_BYTES)]);
        assert!(parts_of(0).is_empty());
    }

    #[test]
    fn a_larger_read_is_cut_into_adjacent_whole_blocks_none_over_the_part_size() {
        for len in [
            PART_BYTES + BLOCK,
            3 * PART_BYTES,
            14 * (1 << 20) + 5 * BLOCK,
            100 * PART_BYTES + BLOCK,
        ] {
            let parts = parts_of(len);
            assert_eq!(parts.len(), len.div_ceil(PART_BYTES), "{len} bytes");
            let mut next = 0;
            for &(start, bytes) in &parts {
                assert_eq!(start, next, "the parts are adjacent");
                assert!(
                    bytes > 0 && bytes <= PART_BYTES,
                    "a part holds up to {PART_BYTES} bytes: {bytes}"
                );
                assert_eq!(bytes % BLOCK, 0, "a part is whole blocks");
                next += bytes;
            }
            assert_eq!(next, len, "the parts cover the read");
        }
    }

    #[test]
    fn the_read_ends_when_its_last_part_does_whatever_the_order() {
        let (done, calls) = counted();
        let join = Join::new(vec![BLOCK, BLOCK, BLOCK], done);
        assert!(join.ended(2, Ok(BLOCK)).is_none());
        assert!(join.ended(0, Ok(BLOCK)).is_none());
        let (done, result) = join.ended(1, Ok(BLOCK)).expect("the last part ended the read");
        assert_eq!(result.expect("whole"), 3 * BLOCK);
        assert_eq!(
            calls.load(Ordering::SeqCst),
            0,
            "the join hands the completion on, it does not run it"
        );
        done(Ok(0));
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_short_part_ends_the_bytes_moved_at_it() {
        let (done, _) = counted();
        let join = Join::new(vec![BLOCK, BLOCK, BLOCK], done);
        assert!(join.ended(0, Ok(BLOCK)).is_none());
        assert!(join.ended(2, Ok(BLOCK)).is_none());
        // The file ended inside part 1: what lies past it is not the read's.
        assert_eq!(
            result_of(join.ended(1, Ok(BLOCK / 2))).expect("short"),
            BLOCK + BLOCK / 2
        );
    }

    #[test]
    fn the_first_failed_part_is_the_reads_failure_with_its_error_intact() {
        let (done, _) = counted();
        let join = Join::new(vec![BLOCK, BLOCK, BLOCK], done);
        assert!(join.ended(2, Err(io::Error::from_raw_os_error(5))).is_none());
        assert!(join.ended(1, Err(io::Error::from_raw_os_error(22))).is_none());
        let failure = result_of(join.ended(0, Ok(BLOCK))).expect_err("failed");
        assert_eq!(
            failure.raw_os_error(),
            Some(22),
            "the alignment refusal the reader falls back on survives"
        );
    }

    #[test]
    fn a_failure_past_a_short_part_is_the_end_of_the_file_not_a_failure() {
        let (done, _) = counted();
        let join = Join::new(vec![BLOCK, BLOCK], done);
        assert!(
            join.ended(1, Err(io::Error::from(io::ErrorKind::UnexpectedEof)))
                .is_none()
        );
        assert_eq!(result_of(join.ended(0, Ok(100))).expect("short"), 100);
    }

    #[test]
    fn parts_the_ring_refused_end_with_its_refusal_and_end_the_read_when_none_is_in_flight() {
        let (done, _) = counted();
        let join = Join::new(vec![BLOCK; 4], done);
        let failure = result_of(join.refused(0, io::Error::from_raw_os_error(11))).expect_err("refused");
        assert_eq!(failure.raw_os_error(), Some(11));

        let (done, _) = counted();
        let join = Join::new(vec![BLOCK; 4], done);
        assert!(join.ended(0, Ok(BLOCK)).is_none());
        let failure = result_of(join.refused(1, io::Error::from_raw_os_error(11))).expect_err("refused");
        assert_eq!(
            failure.raw_os_error(),
            Some(11),
            "the parts that ran do not hide the refusal"
        );
    }

    #[test]
    fn a_part_in_flight_when_the_rest_were_refused_ends_the_read_with_the_refusal() {
        let (done, _) = counted();
        let join = Join::new(vec![BLOCK; 3], done);
        assert!(
            join.refused(1, io::Error::from_raw_os_error(11)).is_none(),
            "part 0 is still in flight"
        );
        let failure = result_of(join.ended(0, Ok(BLOCK))).expect_err("refused");
        assert_eq!(failure.raw_os_error(), Some(11));
    }
}
