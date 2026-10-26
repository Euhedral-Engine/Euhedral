//! Freeing memory after the device is done with it, without waiting.

use std::fmt;
use std::sync::atomic::{AtomicUsize, Ordering};

use euhedral_graph::Marker;

use crate::stack::Stack;

/// Buffers waiting for the lane work that last used them.
///
/// Dropping a [`DeviceBuffer`](crate::DeviceBuffer) or [`PinnedBuffer`](crate::PinnedBuffer) frees it at once, and
/// `cudaFree` and `cudaFreeHost` wait for the device. A buffer whose last user is still queued goes here instead:
/// [`RetireList::retire`] takes it with that user's marker, and [`RetireList::reap`] drops the ones whose markers have
/// been reached, from a poll, without blocking.
///
/// The list is its own object (the runtime owns one) and not a field of the library, because the buffers it holds keep
/// the library alive: a list inside the library would keep it alive forever.
pub struct RetireList {
    waiting: Stack<(Marker, Box<dyn Send>)>,
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

    /// Holds `item` until `marker` is reached. Any thread may call it.
    pub fn retire(&self, item: impl Send + 'static, marker: Marker) {
        self.pending.fetch_add(1, Ordering::Relaxed);
        self.waiting.push((marker, Box::new(item)));
    }

    /// Drops every item whose marker `reached` accepts (`|marker| pool.reached(marker)`), and returns how many; the
    /// rest stay. The dropped items' last users have finished, so freeing them waits for no queued work (a pinned
    /// buffer or a device buffer still frees through `cudaFreeHost` or `cudaFree`, which also synchronize the device
    /// with other lanes' work: recycle through a [`PinnedPool`](crate::PinnedPool) to avoid that).
    pub fn reap(&self, reached: impl Fn(Marker) -> bool) -> usize {
        let mut freed = 0;
        for (marker, item) in self.waiting.take_all() {
            if reached(marker) {
                drop(item);
                freed += 1;
            } else {
                self.waiting.push((marker, item));
            }
        }
        self.pending.fetch_sub(freed, Ordering::Relaxed);
        freed
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

    struct Flag(Arc<AtomicBool>);

    impl Drop for Flag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    fn marker(value: u64) -> Marker {
        Marker { lane: LaneId(0), value }
    }

    #[test]
    fn an_item_is_dropped_once_its_marker_is_reached_and_not_before() {
        let list = RetireList::new();
        let (early, late) = (Arc::new(AtomicBool::new(false)), Arc::new(AtomicBool::new(false)));
        list.retire(Flag(Arc::clone(&early)), marker(1));
        list.retire(Flag(Arc::clone(&late)), marker(5));
        assert_eq!(list.pending(), 2);

        assert_eq!(list.reap(|m| m.value <= 3), 1);
        assert!(early.load(Ordering::SeqCst));
        assert!(!late.load(Ordering::SeqCst));
        assert_eq!(list.pending(), 1);

        assert_eq!(list.reap(|m| m.value <= 3), 0);
        assert_eq!(list.reap(|_| true), 1);
        assert!(late.load(Ordering::SeqCst));
        assert_eq!(list.pending(), 0);
    }

    #[test]
    fn dropping_the_list_drops_what_it_holds() {
        let dropped = Arc::new(AtomicBool::new(false));
        let list = RetireList::new();
        list.retire(Flag(Arc::clone(&dropped)), marker(9));
        drop(list);
        assert!(dropped.load(Ordering::SeqCst));
    }
}
