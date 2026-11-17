//! A maximum over a sliding window: the largest value of the last few windows of time, so a spike is reported for a while
//! and then forgotten.
//!
//! The window is the one the metrics of the server this one replaces use: three windows of one minute each (the
//! reporting step), so a value is reported for between two and three minutes after it was recorded. Time is the wall
//! clock in milliseconds, passed in by the caller; a clock that goes backwards rotates nothing until it passes the last
//! rotation again.

use std::array::from_fn;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, AtomicUsize, Ordering};

/// The length of one window.
pub const WINDOW_MS: i64 = 60_000;

/// How many windows the ring holds.
pub const WINDOWS: usize = 3;

/// The ring of maxima. Every window holds the largest value recorded since it was last cleared, and the oldest one is
/// the one reported; one thread at a time rotates, and a thread that finds a rotation under way goes on without it.
#[derive(Debug)]
pub struct WindowMax {
    ring: [AtomicU64; WINDOWS],
    current: AtomicUsize,
    last_rotate_ms: AtomicI64,
    rotating: AtomicBool,
}

impl WindowMax {
    /// An empty ring whose first window began at `now_ms`.
    pub fn new(now_ms: i64) -> WindowMax {
        WindowMax {
            ring: from_fn(|_| AtomicU64::new(0)),
            current: AtomicUsize::new(0),
            last_rotate_ms: AtomicI64::new(now_ms),
            rotating: AtomicBool::new(false),
        }
    }

    /// Records `value` at `now_ms`.
    pub fn record(&self, value: u64, now_ms: i64) {
        self.rotate(now_ms);
        for window in &self.ring {
            window.fetch_max(value, Ordering::Relaxed);
        }
    }

    /// The largest value of the windows still held at `now_ms`; 0 when there is none.
    pub fn poll(&self, now_ms: i64) -> u64 {
        self.rotate(now_ms);
        self.ring[self.current.load(Ordering::Relaxed)].load(Ordering::Relaxed)
    }

    fn rotate(&self, now_ms: i64) {
        let mut elapsed = now_ms.saturating_sub(self.last_rotate_ms.load(Ordering::Relaxed));
        if elapsed < WINDOW_MS {
            return;
        }
        if self
            .rotating
            .compare_exchange(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            return;
        }
        // Another rotation may have finished since the first look.
        let last = self.last_rotate_ms.load(Ordering::Relaxed);
        elapsed = now_ms.saturating_sub(last);
        if elapsed >= WINDOW_MS * WINDOWS as i64 {
            for window in &self.ring {
                window.store(0, Ordering::Relaxed);
            }
            self.current.store(0, Ordering::Relaxed);
            self.last_rotate_ms
                .store(now_ms - elapsed % WINDOW_MS, Ordering::Relaxed);
        } else if elapsed >= WINDOW_MS {
            let mut at = self.current.load(Ordering::Relaxed);
            let mut began = last;
            while elapsed >= WINDOW_MS {
                self.ring[at].store(0, Ordering::Relaxed);
                at = (at + 1) % WINDOWS;
                elapsed -= WINDOW_MS;
                began += WINDOW_MS;
            }
            self.current.store(at, Ordering::Relaxed);
            self.last_rotate_ms.store(began, Ordering::Relaxed);
        }
        self.rotating.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MINUTE: i64 = WINDOW_MS;

    #[test]
    fn a_maximum_is_reported_until_its_windows_pass() {
        let max = WindowMax::new(0);
        max.record(5, 0);
        assert_eq!(max.poll(0), 5);
        assert_eq!(max.poll(3 * MINUTE - 1), 5);
        assert_eq!(max.poll(3 * MINUTE), 0);
    }

    #[test]
    fn a_smaller_value_takes_over_when_the_larger_one_expires() {
        let max = WindowMax::new(0);
        max.record(5, 0);
        max.record(2, MINUTE + MINUTE / 2);
        assert_eq!(max.poll(MINUTE + MINUTE / 2), 5);
        assert_eq!(max.poll(3 * MINUTE - 1), 5);
        assert_eq!(max.poll(3 * MINUTE), 2);
        assert_eq!(max.poll(4 * MINUTE - 1), 2);
        assert_eq!(max.poll(4 * MINUTE), 0);
    }

    #[test]
    fn windows_rotate_with_nothing_recorded() {
        let max = WindowMax::new(1_000);
        max.record(9, 1_000);
        // Polling every minute rotates one window at a time; the value is gone after three.
        assert_eq!(max.poll(1_000 + MINUTE), 9);
        assert_eq!(max.poll(1_000 + 2 * MINUTE), 9);
        assert_eq!(max.poll(1_000 + 3 * MINUTE), 0);
        // A long silence clears everything at once and keeps the windows aligned to the start.
        max.record(4, 1_000 + 3 * MINUTE);
        assert_eq!(max.poll(1_000 + 100 * MINUTE + 5), 0);
        max.record(6, 1_000 + 100 * MINUTE + 5);
        assert_eq!(max.poll(1_000 + 103 * MINUTE - 1), 6);
        assert_eq!(max.poll(1_000 + 103 * MINUTE), 0);
    }

    #[test]
    fn a_clock_that_goes_backwards_rotates_nothing() {
        let max = WindowMax::new(10 * MINUTE);
        max.record(7, 10 * MINUTE);
        assert_eq!(max.poll(0), 7);
        max.record(3, 5 * MINUTE);
        assert_eq!(max.poll(5 * MINUTE), 7);
        // The windows still end three minutes after they began, by the clock that began them.
        assert_eq!(max.poll(13 * MINUTE - 1), 7);
        assert_eq!(max.poll(13 * MINUTE), 0);
    }
}
