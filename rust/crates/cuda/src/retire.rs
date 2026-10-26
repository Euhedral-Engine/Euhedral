//! Freeing memory after the device is done with it, without waiting.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_graph::Marker;

use crate::error::CudaError;
use crate::stack::Stack;
use crate::submission::Submission;

/// Something the retire list can release without blocking.
///
/// A plain `Drop` of a [`DeviceBuffer`](crate::DeviceBuffer), [`PinnedBuffer`](crate::PinnedBuffer) or
/// [`HostWeightsArena`](crate::HostWeightsArena) calls `cudaFree`, `cudaFreeHost` or `cudaHostUnregister`, which wait
/// for the whole device under the driver's lock. That is for startup and shutdown only; on a running engine it can
/// deadlock against a lane parked on a host flag whose raising thread is stuck in CUDA. So only types with a
/// non-blocking release implement this trait: a [`PooledPinned`](crate::PooledPinned) (returns to its pool) and a
/// stream-ordered [`DeviceBuffer`](crate::DeviceBuffer) (`cudaFreeAsync`). A raw `PinnedBuffer` is not retirable.
pub trait Retire: Send + 'static {
    /// Whether this item can be released without blocking; checked when it is retired.
    fn retirable(&self) -> Result<(), CudaError> {
        Ok(())
    }

    /// Releases the item. `submission` is the lane a stream-ordered free is queued on.
    fn release(self, submission: &Submission<'_>) -> Result<(), CudaError>;
}

/// The object-safe form of [`Retire`].
trait Boxed: Send {
    fn release_boxed(self: Box<Self>, submission: &Submission<'_>) -> Result<(), CudaError>;
}

impl<T: Retire> Boxed for T {
    fn release_boxed(self: Box<Self>, submission: &Submission<'_>) -> Result<(), CudaError> {
        (*self).release(submission)
    }
}

/// Buffers waiting for the lane work that last used them.
///
/// [`RetireList::retire`] takes an item with the marker of its last user, and [`RetireList::reap`] releases the ones
/// whose markers have been reached, from a poll, through the item's non-blocking release ([`Retire`]). Nothing here
/// calls a synchronizing free.
///
/// The list is its own object (the runtime owns one) and not a field of the library, because the items it holds keep
/// the library alive: a list inside the library would keep it alive forever. Dropping a list that still holds items
/// drops them, which for a device buffer is the synchronizing free: shutdown only.
pub struct RetireList {
    waiting: Stack<(Marker, Box<dyn Boxed>)>,
    pending: AtomicUsize,
}

impl fmt::Debug for RetireList {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RetireList")
            .field("pending", &self.pending())
            .finish()
    }
}

impl Default for RetireList {
    fn default() -> Self {
        Self::new()
    }
}

impl RetireList {
    pub fn new() -> RetireList {
        RetireList {
            waiting: Stack::new(),
            pending: AtomicUsize::new(0),
        }
    }

    /// Holds `item` until `marker` is reached. Any thread may call it. An item with no non-blocking release is handed
    /// back with the reason (dropping it would be the synchronizing free).
    pub fn retire<T: Retire>(&self, item: T, marker: Marker) -> Result<(), (CudaError, T)> {
        if let Err(error) = item.retirable() {
            return Err((error, item));
        }
        self.pending.fetch_add(1, Ordering::Relaxed);
        self.waiting.push((marker, Box::new(item)));
        Ok(())
    }

    /// Releases every item whose marker `reached` accepts (`|marker| pool.reached(marker)`) on `submission`'s lane,
    /// and returns how many; the rest stay. Nothing blocks. Every ready item is released even when one fails; the
    /// first failure is returned.
    pub fn reap(&self, reached: impl Fn(Marker) -> bool, submission: &Submission<'_>) -> Result<usize, CudaError> {
        self.reap_with(reached, |item| item.release_boxed(submission))
    }

    fn reap_with(
        &self,
        reached: impl Fn(Marker) -> bool,
        mut release: impl FnMut(Box<dyn Boxed>) -> Result<(), CudaError>,
    ) -> Result<usize, CudaError> {
        let (mut freed, mut first_error) = (0, None);
        for (marker, item) in self.waiting.take_all() {
            if !reached(marker) {
                self.waiting.push((marker, item));
                continue;
            }
            freed += 1;
            if let Err(error) = release(item) {
                first_error.get_or_insert(error);
            }
        }
        self.pending.fetch_sub(freed, Ordering::Relaxed);
        first_error.map_or(Ok(freed), Err)
    }

    /// Items still waiting for their markers.
    pub fn pending(&self) -> usize {
        self.pending.load(Ordering::Relaxed)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;

    use euhedral_graph::LaneId;

    use super::*;

    struct Flag(Arc<AtomicBool>, bool);

    impl Drop for Flag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    impl Retire for Flag {
        fn retirable(&self) -> Result<(), CudaError> {
            if self.1 {
                Ok(())
            } else {
                Err(CudaError::Argument("blocking".to_string()))
            }
        }

        fn release(self, _submission: &Submission<'_>) -> Result<(), CudaError> {
            Ok(())
        }
    }

    fn marker(value: u64) -> Marker {
        Marker { lane: LaneId(0), value }
    }

    fn reap(list: &RetireList, limit: u64) -> usize {
        list.reap_with(|m| m.value <= limit, drop_boxed).unwrap()
    }

    /// Stands in for `release_boxed`, which needs a submission: dropping runs the flag's `Drop`.
    fn drop_boxed(item: Box<dyn Boxed>) -> Result<(), CudaError> {
        drop(item);
        Ok(())
    }

    #[test]
    fn an_item_is_released_once_its_marker_is_reached_and_not_before() {
        let list = RetireList::new();
        let (early, late) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        list.retire(Flag(Arc::clone(&early), true), marker(1)).ok().unwrap();
        list.retire(Flag(Arc::clone(&late), true), marker(5)).ok().unwrap();
        assert_eq!(list.pending(), 2);

        assert_eq!(reap(&list, 3), 1);
        assert!(early.load(Ordering::SeqCst));
        assert!(!late.load(Ordering::SeqCst));
        assert_eq!(list.pending(), 1);

        assert_eq!(reap(&list, 3), 0);
        assert_eq!(reap(&list, 9), 1);
        assert!(late.load(Ordering::SeqCst));
        assert_eq!(list.pending(), 0);
    }

    #[test]
    fn an_item_without_a_non_blocking_release_is_handed_back() {
        let list = RetireList::new();
        let dropped = Arc::new(AtomicBool::new(false));
        let (error, item) = list.retire(Flag(Arc::clone(&dropped), false), marker(1)).unwrap_err();
        assert!(error.to_string().contains("blocking"));
        assert!(!dropped.load(Ordering::SeqCst), "handed back, not dropped");
        assert_eq!(list.pending(), 0);
        drop(item);
    }

    #[test]
    fn dropping_the_list_drops_what_it_holds() {
        let dropped = Arc::new(AtomicBool::new(false));
        let list = RetireList::new();
        list.retire(Flag(Arc::clone(&dropped), true), marker(9)).ok().unwrap();
        drop(list);
        assert!(dropped.load(Ordering::SeqCst));
    }
}
