//! A value handed from a frame (or any holder of the sending end) to a thread outside the lattice.
//!
//! RustEE removed its latch: the user holds what they hand over. This is the engine's hold: an `AtomicPtr` to a boxed
//! value, set once and taken once. The thread that waits is a caller's own (the binary's `main`, a test), never a
//! worker; every holder of a sending end must send in its `Drop` when it has not sent, or the waiter never ends.

use std::error::Error;
use std::fmt;
use std::marker::PhantomData;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};
use std::thread;
use std::time::{Duration, Instant};

use euhedral_data_structures::sync::HotScope;

/// The first pause between two checks of a waiting call; it doubles up to [`LONGEST_PAUSE`].
const SHORTEST_PAUSE: Duration = Duration::from_micros(50);
const LONGEST_PAUSE: Duration = Duration::from_millis(1);

/// The deadline of a [`Handoff::wait`] passed with nothing sent.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TimedOut;

impl fmt::Display for TimedOut {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the hand-off was not sent before its deadline")
    }
}

impl Error for TimedOut {}

/// The mark of an abandoned hand-off: the address of this static is stored where a value's box would be. It is never
/// read through.
static ABANDONED: u8 = 0;

/// A cell that takes one value from any thread and gives it to one. The taker may give up first
/// ([`abandon`](Handoff::abandon)): a value that arrives after that goes to the sender's own release, never into the
/// cell.
pub struct Handoff<T> {
    value: AtomicPtr<T>,
    /// The cell owns a `T` that crosses threads: `Send` only when `T` is.
    owns: PhantomData<T>,
}

// SAFETY: a value goes in by a release store of a pointer to a box nothing else holds and out by an acquire swap that
// gives the box to one thread, so the cell only moves `T`s between threads, which `T: Send` allows.
unsafe impl<T: Send> Sync for Handoff<T> {}

impl<T> fmt::Debug for Handoff<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Handoff").field("set", &self.is_set()).finish()
    }
}

impl<T> Handoff<T> {
    fn abandoned_mark() -> *mut T {
        ptr::from_ref(&ABANDONED).cast_mut().cast::<T>()
    }

    /// An empty hand-off, to share between the sending end and the waiting end.
    pub fn new() -> Arc<Handoff<T>> {
        Arc::new(Handoff::empty())
    }

    /// An empty hand-off held by value, as one field of a larger shared object.
    pub fn empty() -> Handoff<T> {
        Handoff {
            value: AtomicPtr::new(ptr::null_mut()),
            owns: PhantomData,
        }
    }

    /// Hands `value` over. Only the first send counts: a later value is dropped (and a debug build panics, since two
    /// senders mean a holder sent without checking that it had not).
    pub fn send(&self, value: T) {
        self.deliver(value, drop);
    }

    /// Hands `value` over like [`send`](Handoff::send), unless the taker has abandoned the hand-off: then `release`
    /// gets the value, on this thread and at once, because nobody will take it. Either the taker or `release` ends up
    /// with a value, never both and never neither.
    pub fn deliver(&self, value: T, release: impl FnOnce(T)) {
        let new = Box::into_raw(Box::new(value));
        match self
            .value
            .compare_exchange(ptr::null_mut(), new, Ordering::AcqRel, Ordering::Acquire)
        {
            Ok(_) => {}
            Err(current) => {
                // SAFETY: `new` came from `Box::into_raw` above and was not published.
                let value = *unsafe { Box::from_raw(new) };
                if current == Self::abandoned_mark() {
                    release(value);
                } else {
                    drop(value);
                    debug_assert!(false, "a hand-off was sent twice");
                }
            }
        }
    }

    /// Gives up the hand-off: returns the value if one was sent and not yet taken, and otherwise marks the hand-off
    /// so that the value still to come goes to its sender's release ([`deliver`](Handoff::deliver)). Nothing waits.
    pub fn abandon(&self) -> Option<T> {
        let mark = Self::abandoned_mark();
        let mut current = self.value.load(Ordering::Acquire);
        loop {
            if current == mark {
                return None;
            }
            match self
                .value
                .compare_exchange(current, mark, Ordering::AcqRel, Ordering::Acquire)
            {
                // SAFETY: a non-null pointer other than the mark was published by `deliver` from `Box::into_raw`,
                // and the exchange gave it to this call alone.
                Ok(taken) => return (!taken.is_null()).then(|| *unsafe { Box::from_raw(taken) }),
                Err(seen) => current = seen,
            }
        }
    }

    /// Takes the value if it was sent and not yet taken.
    pub fn take(&self) -> Option<T> {
        let mark = Self::abandoned_mark();
        let mut current = self.value.load(Ordering::Acquire);
        loop {
            if current.is_null() || current == mark {
                return None;
            }
            match self
                .value
                .compare_exchange(current, ptr::null_mut(), Ordering::AcqRel, Ordering::Acquire)
            {
                // SAFETY: a non-null pointer other than the mark was published by `deliver` from `Box::into_raw`,
                // and the exchange gave it to this call alone.
                Ok(taken) => return Some(*unsafe { Box::from_raw(taken) }),
                Err(seen) => current = seen,
            }
        }
    }

    /// Whether a value waits to be taken.
    pub fn is_set(&self) -> bool {
        let current = self.value.load(Ordering::Acquire);
        !current.is_null() && current != Self::abandoned_mark()
    }

    /// Waits for the value for at most `deadline`, checking at growing pauses (50 microseconds up to 1 ms).
    ///
    /// For threads outside the lattice only: a debug build panics when a hot scope is open on the calling thread.
    #[allow(
        clippy::disallowed_methods,
        reason = "outside the lattice: the caller's own thread waits, never a worker"
    )]
    pub fn wait(&self, deadline: Duration) -> Result<T, TimedOut> {
        assert!(
            !HotScope::active(),
            "Handoff::wait inside the lattice: only a thread outside it may wait"
        );
        let start = Instant::now();
        let mut pause = SHORTEST_PAUSE;
        loop {
            if let Some(value) = self.take() {
                return Ok(value);
            }
            let left = deadline.saturating_sub(start.elapsed());
            if left.is_zero() {
                return Err(TimedOut);
            }
            thread::sleep(pause.min(left));
            pause = (pause * 2).min(LONGEST_PAUSE);
        }
    }
}

impl<T> Drop for Handoff<T> {
    fn drop(&mut self) {
        drop(self.take());
    }
}

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    reason = "the tests start threads to race senders, and join them"
)]
mod tests {
    #[allow(clippy::disallowed_types, reason = "the test lines two threads up so that they race")]
    use std::sync::Barrier;
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    use super::*;

    const BOUND: Duration = Duration::from_secs(5);

    #[test]
    fn a_value_sent_is_taken_once() {
        let handoff = Handoff::new();
        assert!(!handoff.is_set());
        handoff.send(7);
        assert!(handoff.is_set());
        assert_eq!(handoff.take(), Some(7));
        assert_eq!(handoff.take(), None);
        assert!(!handoff.is_set());
    }

    #[test]
    fn a_take_before_the_send_finds_nothing() {
        let handoff = Handoff::<u32>::new();
        assert_eq!(handoff.take(), None);
        handoff.send(1);
        assert_eq!(handoff.take(), Some(1));
    }

    #[test]
    fn a_wait_ends_when_the_value_comes_from_another_thread() {
        let handoff = Handoff::new();
        let sender = Arc::clone(&handoff);
        let sending = thread::spawn(move || sender.send(42));
        assert_eq!(handoff.wait(BOUND), Ok(42));
        sending.join().unwrap();
    }

    #[test]
    fn a_wait_with_no_sender_fails_at_its_deadline() {
        let handoff = Handoff::<u32>::new();
        let start = Instant::now();
        assert_eq!(handoff.wait(Duration::from_millis(50)), Err(TimedOut));
        assert!(start.elapsed() < Duration::from_secs(1), "the deadline is not honoured");
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "sent twice")]
    fn a_second_send_panics_in_a_debug_build() {
        let handoff = Handoff::new();
        handoff.send(1);
        handoff.send(2);
    }

    #[test]
    fn a_value_left_in_the_hand_off_is_dropped_once() {
        struct Counted(Arc<AtomicUsize>);
        impl Drop for Counted {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let dropped = Arc::new(AtomicUsize::new(0));
        let handoff = Handoff::new();
        handoff.send(Counted(Arc::clone(&dropped)));
        drop(handoff);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }

    struct Counted(Arc<AtomicUsize>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn a_value_that_arrives_after_the_abandon_is_released_once_by_its_sender() {
        let freed = Arc::new(AtomicUsize::new(0));
        let handoff = Handoff::new();
        assert!(handoff.abandon().is_none());
        assert!(handoff.abandon().is_none(), "abandoning twice changes nothing");
        let counted = Arc::clone(&freed);
        handoff.deliver(Counted(Arc::clone(&freed)), move |value| {
            // The sender frees it: a count of frees by the release, apart from the drop.
            counted.fetch_add(100, Ordering::SeqCst);
            drop(value);
        });
        assert_eq!(freed.load(Ordering::SeqCst), 101);
        assert!(!handoff.is_set());
        assert!(handoff.take().is_none());
        drop(handoff);
        assert_eq!(freed.load(Ordering::SeqCst), 101, "nothing is freed again");
    }

    #[test]
    fn a_value_that_arrived_before_the_abandon_comes_back_from_it() {
        let freed = Arc::new(AtomicUsize::new(0));
        let handoff = Handoff::new();
        handoff.send(Counted(Arc::clone(&freed)));
        let value = handoff.abandon();
        assert!(value.is_some());
        drop(value);
        drop(handoff);
        assert_eq!(freed.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[allow(clippy::disallowed_types, reason = "the test lines two threads up so that they race")]
    fn a_race_between_the_abandon_and_the_arrival_frees_exactly_once() {
        for _ in 0..500 {
            let freed = Arc::new(AtomicUsize::new(0));
            let handoff = Handoff::new();
            let sender = Arc::clone(&handoff);
            let released = Arc::clone(&freed);
            let gate = Arc::new(Barrier::new(2));
            let arrival_gate = Arc::clone(&gate);
            let arriving = thread::spawn(move || {
                arrival_gate.wait();
                sender.deliver(Counted(Arc::clone(&released)), |value| {
                    released.fetch_add(100, Ordering::SeqCst);
                    drop(value);
                });
            });
            // The abandoning side frees what it gets back.
            gate.wait();
            let got = handoff.abandon();
            let gave = got.is_some();
            drop(got);
            arriving.join().unwrap();
            drop(handoff);
            let count = freed.load(Ordering::SeqCst);
            assert_eq!(count, if gave { 1 } else { 101 }, "freed {count} times");
        }
    }

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "only a thread outside it may wait")]
    fn a_wait_inside_a_hot_scope_panics_in_a_debug_build() {
        let handoff = Handoff::<u32>::new();
        let _hot = HotScope::enter();
        let _ = handoff.wait(Duration::from_millis(1));
    }
}
