//! A value handed from a frame (or any holder of the sending end) to a thread outside the lattice.
//!
//! RustEE removed its latch: the user holds what they hand over. This is the engine's hold: an `AtomicPtr` to a boxed
//! value, set once and taken once. The thread that waits is a caller's own (the binary's `main`, a test), never a
//! worker; every holder of a sending end must send in its `Drop` when it has not sent, or the waiter never ends.

use std::fmt;
use std::ptr;
use std::sync::Arc;
use std::sync::atomic::{AtomicPtr, Ordering};
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

impl std::error::Error for TimedOut {}

/// A cell that takes one value from any thread and gives it to one.
pub struct Handoff<T> {
    value: AtomicPtr<T>,
}

impl<T> fmt::Debug for Handoff<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Handoff").field("set", &self.is_set()).finish()
    }
}

impl<T> Handoff<T> {
    /// An empty hand-off, to share between the sending end and the waiting end.
    pub fn new() -> Arc<Handoff<T>> {
        Arc::new(Handoff::empty())
    }

    /// An empty hand-off held by value, as one field of a larger shared object.
    pub fn empty() -> Handoff<T> {
        Handoff {
            value: AtomicPtr::new(ptr::null_mut()),
        }
    }

    /// Hands `value` over. Only the first send counts: a later value is dropped (and a debug build panics, since two
    /// senders mean a holder sent without checking that it had not).
    pub fn send(&self, value: T) {
        let new = Box::into_raw(Box::new(value));
        if self
            .value
            .compare_exchange(ptr::null_mut(), new, Ordering::AcqRel, Ordering::Acquire)
            .is_err()
        {
            // SAFETY: `new` came from `Box::into_raw` above and was not published.
            drop(unsafe { Box::from_raw(new) });
            debug_assert!(false, "a hand-off was sent twice");
        }
    }

    /// Takes the value if it was sent and not yet taken.
    pub fn take(&self) -> Option<T> {
        let taken = self.value.swap(ptr::null_mut(), Ordering::Acquire);
        // SAFETY: a non-null pointer was published by `send` from `Box::into_raw`, and the swap gave it to this call
        // alone.
        (!taken.is_null()).then(|| *unsafe { Box::from_raw(taken) })
    }

    /// Whether a value waits to be taken.
    pub fn is_set(&self) -> bool {
        !self.value.load(Ordering::Acquire).is_null()
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
            std::thread::sleep(pause.min(left));
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
mod tests {
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

    /// The holder of a sending end the way the engine writes one: it sends the failure in `Drop` when it has not
    /// sent a result.
    struct Holder {
        ready: Option<Arc<Handoff<Result<u32, &'static str>>>>,
    }

    impl Drop for Holder {
        fn drop(&mut self) {
            if let Some(ready) = self.ready.take() {
                ready.send(Err("abandoned"));
            }
        }
    }

    #[test]
    fn a_holder_dropped_without_a_result_ends_the_waiter() {
        let handoff = Handoff::new();
        let holder = Holder {
            ready: Some(Arc::clone(&handoff)),
        };
        let dropper = thread::spawn(move || drop(holder));
        // Without the send in `Drop` this returns `TimedOut` after the bound.
        assert_eq!(handoff.wait(BOUND), Ok(Err("abandoned")));
        dropper.join().unwrap();
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

    #[cfg(debug_assertions)]
    #[test]
    #[should_panic(expected = "only a thread outside it may wait")]
    fn a_wait_inside_a_hot_scope_panics_in_a_debug_build() {
        let handoff = Handoff::<u32>::new();
        let _hot = HotScope::enter();
        let _ = handoff.wait(Duration::from_millis(1));
    }
}
