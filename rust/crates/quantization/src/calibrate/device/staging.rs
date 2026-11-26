//! The pinned memory the embedding read, the MTP block and the heads read their tensors into.
//!
//! These tensors are as large as a layer's, or larger (the embedding table and the output head are 2.5 GB each), and
//! they are read a few times in a run. The layers' pool rounds a loan up to a power of two and allocates it in the
//! frame that asks, and then keeps it until a trim that no calibration calls. So the heads have a buffer of their own:
//! [`HeadStaging`] holds one buffer of the size the largest group of head tensors needs, allocated at open, where a
//! blocking allocation is allowed, and freed with the calibration. A group is the tensors one call reads together.
//! [`HeadStaging::lease`] lends the whole buffer to one group at a time, cut into disjoint parts that start on a page,
//! and the [`Lease`] returns it when it is dropped. No frame allocates or frees pinned memory for these reads, and
//! nothing is rounded up.
//!
//! Call sequence: `open` once; per group `lease`, read each part, use the parts, drop the lease.

use std::slice;
use std::sync::Arc;

use euhedral_inference_cuda::{CudaError, CudaLibrary, PinnedBuffer, PinnedRange};
use euhedral_inference_engine::Handoff;

use crate::calibrate::CalibrateError;

/// Each part starts on a page, as the layers' loans do.
const PART_ALIGN: usize = 4096;

/// Where each of `sizes` starts, and how many bytes the buffer needs for all of them; `None` when that overflows.
fn layout(sizes: &[usize]) -> Option<(Vec<(usize, usize)>, usize)> {
    let mut parts = Vec::with_capacity(sizes.len());
    let mut end = 0usize;
    for &bytes in sizes {
        let start = end.checked_next_multiple_of(PART_ALIGN)?;
        end = start.checked_add(bytes)?;
        parts.push((start, bytes));
    }
    Some((parts, end))
}

/// The bytes of the buffer the largest of `groups` needs; 0 for no group.
pub(super) fn capacity_of(groups: &[Vec<usize>]) -> Result<usize, CalibrateError> {
    groups
        .iter()
        .map(|sizes| layout(sizes).map(|(_, total)| total))
        .try_fold(0, |most: usize, total| total.map(|total| most.max(total)))
        .ok_or_else(|| CalibrateError::Invalid("the head tensors are too large for one buffer".to_owned()))
}

/// The one buffer of the heads' reads, lent to one group at a time. `B` is the buffer: a [`PinnedBuffer`] in a run.
pub(super) struct HeadStaging<B> {
    home: Arc<Handoff<B>>,
    capacity: usize,
}

impl<B> HeadStaging<B> {
    /// A staging that owns `buffer`, which holds `capacity` bytes.
    fn holding(buffer: Option<B>, capacity: usize) -> Self {
        let home = Arc::new(Handoff::empty());
        if let Some(buffer) = buffer {
            home.send(buffer);
        }
        HeadStaging { home, capacity }
    }

    /// Lends the buffer, cut into one part for each of `sizes`. Refused, with nothing taken, when the parts do not fit
    /// in the buffer, and when it is already lent.
    pub(super) fn lease(&self, sizes: &[usize]) -> Result<Lease<B>, CalibrateError> {
        let fits = layout(sizes).filter(|(_, total)| *total <= self.capacity);
        let Some((parts, _)) = fits else {
            return Err(CalibrateError::Failed(format!(
                "the head tensors {sizes:?} do not fit the staging buffer of {} bytes",
                self.capacity
            )));
        };
        let buffer = self
            .home
            .take()
            .ok_or_else(|| CalibrateError::Failed("the heads' staging buffer is in use".to_owned()))?;
        Ok(Lease {
            buffer: Some(Arc::new(buffer)),
            home: Arc::clone(&self.home),
            claimed: vec![false; parts.len()],
            parts,
        })
    }
}

#[cfg(test)]
impl<B> HeadStaging<B> {
    /// A staging that owns `buffer`, for tests with a stand-in for the pinned buffer.
    pub(super) fn open_with(buffer: B, capacity: usize) -> Self {
        Self::holding(Some(buffer), capacity)
    }
}

impl HeadStaging<PinnedBuffer> {
    /// Allocates the buffer for the largest of `groups` of tensor sizes. Blocks: call it from setup code.
    pub(super) fn open(library: &Arc<CudaLibrary>, groups: &[Vec<usize>]) -> Result<Self, CalibrateError> {
        let capacity = capacity_of(groups)?;
        let buffer = (capacity > 0)
            .then(|| PinnedBuffer::alloc(library, capacity))
            .transpose()
            .map_err(|error: CudaError| CalibrateError::failed("the heads' staging buffer", error))?;
        Ok(Self::holding(buffer, capacity))
    }
}

/// The staging buffer on loan, cut into parts. Dropping it gives the buffer back, unless a clone of it is still out
/// (a range a read never returned): the buffer then stays with that holder, and later leases are refused.
pub(super) struct Lease<B> {
    buffer: Option<Arc<B>>,
    home: Arc<Handoff<B>>,
    parts: Vec<(usize, usize)>,
    /// Which parts a read has been given: a part is given once.
    claimed: Vec<bool>,
}

impl<B> Lease<B> {
    /// Where part `index` starts in the buffer, and its bytes.
    pub(super) fn part(&self, index: usize) -> Option<(usize, usize)> {
        self.parts.get(index).copied()
    }

    /// Hands part `index` to a read: where it starts, its bytes, and a clone of the buffer that the read holds until it
    /// ends. Refused (`None`) for a part that does not exist and for one already given, so two reads never share bytes.
    pub(super) fn claim(&mut self, index: usize) -> Option<(usize, usize, Arc<B>)> {
        let (start, bytes) = self.part(index)?;
        let taken = self.claimed.get_mut(index).filter(|taken| !**taken)?;
        *taken = true;
        Some((start, bytes, Arc::clone(self.buffer())))
    }

    /// Where part `index` starts and its bytes, for the host to read them: refused (`None`) while a clone of the buffer
    /// is out, that is while a read may still be writing.
    pub(super) fn settled(&self, index: usize) -> Option<(usize, usize)> {
        (Arc::strong_count(self.buffer()) == 1)
            .then(|| self.part(index))
            .flatten()
    }

    pub(super) fn buffer(&self) -> &Arc<B> {
        #[allow(
            clippy::expect_used,
            reason = "invariant: a lease holds its buffer until it is dropped"
        )]
        self.buffer.as_ref().expect("a lease holds its buffer until dropped")
    }
}

impl<B> Drop for Lease<B> {
    fn drop(&mut self) {
        if let Some(Ok(buffer)) = self.buffer.take().map(Arc::try_unwrap) {
            self.home.send(buffer);
        }
    }
}

/// What the tensors are read from: it reads the tensor `name` into part `index` of `lease` and gives the lease back to
/// `then`, once, after the read has ended and let go of its part (or at once, with the error, when it cannot start).
pub(super) trait PartReader<B>: Send + Sync + 'static {
    fn read(
        &self,
        name: &str,
        index: usize,
        lease: Lease<B>,
        then: Box<dyn FnOnce(Result<Lease<B>, CalibrateError>) + Send>,
    );
}

/// Reads `names` one after the other, name `i` into part `i`, starting at `index`; `then` gets the lease with every
/// part read, or the first failure.
pub(super) fn read_parts<B: Send + Sync + 'static>(
    reader: Arc<dyn PartReader<B>>,
    names: Vec<String>,
    index: usize,
    lease: Lease<B>,
    then: Box<dyn FnOnce(Result<Lease<B>, CalibrateError>) + Send>,
) {
    let Some(name) = names.get(index).cloned() else {
        return then(Ok(lease));
    };
    let next = Arc::clone(&reader);
    reader.read(
        &name,
        index,
        lease,
        Box::new(move |read| match read {
            Ok(lease) => read_parts(next, names, index + 1, lease, then),
            Err(error) => then(Err(error)),
        }),
    );
}

/// The heads' tensors in pinned memory.
pub(super) type HeadMemory = Lease<PinnedBuffer>;

impl HeadMemory {
    /// The range of part `index`, for a read to fill; `None` when the part was already cut.
    pub(super) fn range(&mut self, index: usize) -> Option<PinnedRange> {
        let (start, bytes, buffer) = self.claim(index)?;
        // SAFETY: the lease is the buffer's only user, its parts do not overlap, and `claim` gave this part once.
        unsafe { PinnedRange::new(&buffer, start, bytes) }
    }

    /// The bytes of part `index`; `None` while any range is out, which is while a read may be writing.
    pub(super) fn bytes(&self, index: usize) -> Option<&[u8]> {
        let (start, bytes) = self.settled(index)?;
        // SAFETY: the part lies inside the buffer (the lease was cut to fit), and no range is out, so nothing writes.
        Some(unsafe { slice::from_raw_parts(self.buffer().as_ptr().add(start), bytes) })
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn staging(capacity: usize) -> HeadStaging<Vec<u8>> {
        HeadStaging::holding(Some(vec![0; capacity]), capacity)
    }

    #[test]
    fn the_buffer_is_the_size_of_the_largest_group_with_each_part_on_a_page() {
        let groups = vec![vec![10_000], vec![5000, 4096, 10], vec![]];
        // 5000 -> page 8192; 4096 starts at 8192, ends 12288; 10 starts at 12288.
        assert_eq!(capacity_of(&groups).unwrap(), 12_298);
        assert_eq!(capacity_of(&[]).unwrap(), 0);
        assert!(capacity_of(&[vec![usize::MAX, 1]]).is_err());
    }

    #[test]
    fn a_lease_cuts_exact_disjoint_parts_in_order() {
        let staging = staging(12_298);
        let lease = staging.lease(&[5000, 4096, 10]).unwrap();
        assert_eq!(lease.part(0), Some((0, 5000)));
        assert_eq!(lease.part(1), Some((8192, 4096)));
        assert_eq!(lease.part(2), Some((12_288, 10)));
        assert_eq!(lease.part(3), None);
    }

    #[test]
    fn a_group_larger_than_the_buffer_is_refused_and_takes_nothing() {
        let staging = staging(8192);
        assert!(staging.lease(&[8193]).is_err());
        assert!(staging.lease(&[4097, 4097]).is_err());
        assert!(staging.lease(&[8192]).is_ok(), "a refusal takes nothing");
    }

    #[test]
    fn the_buffer_is_lent_to_one_group_at_a_time_and_comes_back_on_drop() {
        let staging = staging(4096);
        let lease = staging.lease(&[100]).unwrap();
        assert!(staging.lease(&[100]).is_err(), "a lent buffer is not lent again");
        drop(lease);
        assert!(staging.lease(&[100]).is_ok());
    }

    #[test]
    fn a_buffer_a_read_still_holds_is_not_given_back() {
        let staging = staging(4096);
        let mut lease = staging.lease(&[100]).unwrap();
        let (.., held) = lease.claim(0).unwrap();
        drop(lease);
        assert!(
            staging.lease(&[100]).is_err(),
            "memory a read may still write is not lent"
        );
        drop(held);
    }

    #[test]
    fn a_part_is_given_to_one_read_only() {
        let staging = staging(8192);
        let mut lease = staging.lease(&[10, 10]).unwrap();
        assert!(lease.claim(0).is_some());
        assert!(lease.claim(0).is_none(), "a second read of the same bytes is refused");
        assert!(lease.claim(1).is_some(), "another part is still free");
        assert!(lease.claim(2).is_none(), "there is no third part");
    }

    #[test]
    fn the_host_reads_a_part_only_when_no_read_is_out() {
        let staging = staging(8192);
        let mut lease = staging.lease(&[10, 10]).unwrap();
        assert_eq!(lease.settled(1), Some((4096, 10)));
        let (.., read) = lease.claim(0).unwrap();
        assert_eq!(lease.settled(1), None, "a read of part 0 may still be writing");
        drop(read);
        assert_eq!(lease.settled(1), Some((4096, 10)));
        assert_eq!(lease.settled(2), None);
    }

    /// Records, in order, the first byte of each tensor name and the part it was given to, and ends each read at once
    /// or fails the one named `fail`.
    struct Fake {
        asked: [AtomicUsize; 4],
        count: AtomicUsize,
        fail: Option<&'static str>,
    }

    impl PartReader<Vec<u8>> for Fake {
        fn read(
            &self,
            name: &str,
            index: usize,
            mut lease: Lease<Vec<u8>>,
            then: Box<dyn FnOnce(Result<Lease<Vec<u8>>, CalibrateError>) + Send>,
        ) {
            let at = self.count.fetch_add(1, Ordering::Relaxed);
            self.asked[at].store(usize::from(name.as_bytes()[0]) * 10 + index, Ordering::Relaxed);
            assert!(lease.claim(index).is_some(), "each part is claimed once");
            then(if self.fail == Some(name) {
                Err(CalibrateError::Failed(name.to_owned()))
            } else {
                Ok(lease)
            });
        }
    }

    /// Reads `names` through a fake that fails `fail`; what it was asked to read as name byte times ten plus part, in
    /// order, and whether the read ended well.
    fn read_with(names: &[&str], fail: Option<&'static str>) -> (Vec<usize>, bool) {
        let staging = staging(3 * 4096);
        let lease = staging.lease(&[1, 1, 1][..names.len()]).unwrap();
        let fake = Arc::new(Fake {
            asked: Default::default(),
            count: AtomicUsize::new(0),
            fail,
        });
        let reader: Arc<dyn PartReader<Vec<u8>>> = Arc::clone(&fake) as _;
        let done = Arc::new(AtomicUsize::new(0));
        let ended = Arc::clone(&done);
        read_parts(
            reader,
            names.iter().map(|name| (*name).to_owned()).collect(),
            0,
            lease,
            Box::new(move |read| ended.store(if read.is_ok() { 1 } else { 2 }, Ordering::Relaxed)),
        );
        let count = fake.count.load(Ordering::Relaxed);
        let asked = fake.asked[..count]
            .iter()
            .map(|at| at.load(Ordering::Relaxed))
            .collect();
        (asked, done.load(Ordering::Relaxed) == 1)
    }

    #[test]
    fn tensors_are_read_one_part_each_in_the_order_named() {
        let (asked, done) = read_with(&["a", "b", "c"], None);
        assert!(done);
        assert_eq!(asked, [970, 981, 992]);
    }

    #[test]
    fn a_failed_read_stops_the_rest() {
        let (asked, done) = read_with(&["a", "b", "c"], Some("b"));
        assert!(!done);
        assert_eq!(asked, [970, 981], "c was never read");
    }
}
