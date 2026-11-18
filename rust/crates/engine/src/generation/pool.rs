//! The pool of idle sequences: which of the engine's sequences can take a request now.
//!
//! A request takes a sequence from the pool when it is admitted and holds a [`Lease`] on it until its run is dropped, so
//! that the sequence goes back on every end: finished, cancelled, failed, refused or withdrawn. A request whose owner
//! panicked twice leaves quanta in flight that can no longer be accounted for: its lease is forgotten, and the sequence
//! stays out of the pool for good, so that no next request races them. The state is one flag per sequence, taken and given back with atomic operations: no lock and no wait. The
//! lowest idle sequence is taken first, so requests that follow one another stay on the same sequence and its graphs.
//!
//! Whether a request is admitted depends on the pool alone, never on whether a turn of an owner holds its claim at the
//! moment.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::state::SequenceId;

/// The idle sequences of an engine.
#[derive(Debug)]
pub(crate) struct SequencePool {
    running: Vec<Arc<AtomicBool>>,
}

impl SequencePool {
    /// A pool of `sequences` sequences, all idle.
    pub(crate) fn new(sequences: usize) -> SequencePool {
        SequencePool {
            running: (0..sequences).map(|_| Arc::new(AtomicBool::new(false))).collect(),
        }
    }

    /// The number of sequences, idle or not.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.running.len()
    }

    /// Takes the lowest idle sequence, or `None` when every sequence is running a request.
    pub(crate) fn take(&self) -> Option<(SequenceId, Lease)> {
        self.running.iter().enumerate().find_map(|(index, flag)| {
            flag.compare_exchange(false, true, Ordering::AcqRel, Ordering::Relaxed)
                .ok()
                .map(|_| (SequenceId(index), Lease { flag: Arc::clone(flag) }))
        })
    }

    /// How many sequences are running a request.
    #[cfg(test)]
    pub(crate) fn running(&self) -> usize {
        self.running.iter().filter(|flag| flag.load(Ordering::Acquire)).count()
    }
}

/// A sequence taken from the pool: dropping it gives the sequence back.
#[derive(Debug)]
pub(crate) struct Lease {
    flag: Arc<AtomicBool>,
}

impl Drop for Lease {
    fn drop(&mut self) {
        self.flag.store(false, Ordering::Release);
    }
}

#[cfg(test)]
mod tests {
    use std::mem;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    use super::*;

    #[test]
    fn the_lowest_idle_sequence_is_taken_and_comes_back_when_the_lease_drops() {
        let pool = SequencePool::new(3);
        let (first, lease_first) = pool.take().unwrap();
        let (second, lease_second) = pool.take().unwrap();
        assert_eq!((first, second), (SequenceId(0), SequenceId(1)));
        drop(lease_first);
        let (again, _lease) = pool.take().unwrap();
        assert_eq!(again, SequenceId(0), "the lowest idle sequence goes first");
        drop(lease_second);
        assert_eq!(pool.running(), 1);
    }

    #[test]
    fn one_more_request_than_sequences_finds_the_pool_empty_exactly_once() {
        let pool = SequencePool::new(4);
        let leases: Vec<_> = (0..4).map(|_| pool.take().unwrap()).collect();
        assert_eq!(pool.running(), pool.len());
        assert!(pool.take().is_none(), "the fifth request is refused");
        assert!(pool.take().is_none(), "and a sixth: nothing was given back");
        drop(leases);
        assert_eq!(pool.running(), 0, "every lease gave its sequence back");
        assert!(pool.take().is_some());
    }

    #[test]
    fn a_lease_dropped_on_any_path_gives_the_sequence_back() {
        let pool = SequencePool::new(1);
        // Refused before it ran: the lease goes out of scope with the request.
        let refused: Result<(), Lease> = Err(pool.take().unwrap().1);
        assert!(refused.is_err() && pool.take().is_none());
        drop(refused);
        // A panic that unwinds through the holder.
        let (_, lease) = pool.take().unwrap();
        let unwound = catch_unwind(AssertUnwindSafe(move || {
            let _held = lease;
            panic!("the holder panics");
        }));
        assert!(unwound.is_err());
        assert_eq!(pool.running(), 0);
        // Forgotten (an owner that panicked twice leaks its run): the sequence stays out, never handed to a second request.
        mem::forget(pool.take().unwrap().1);
        assert!(pool.take().is_none());
    }

    #[test]
    #[allow(
        clippy::disallowed_methods,
        reason = "peer threads race for the pool: a test fixture"
    )]
    fn threads_taking_and_dropping_never_hold_one_sequence_twice() {
        const SEQUENCES: usize = 3;
        const THREADS: usize = SEQUENCES + 1;
        const ROUNDS: usize = 2000;
        let pool = Arc::new(SequencePool::new(SEQUENCES));
        let holders: Arc<Vec<AtomicUsize>> = Arc::new((0..SEQUENCES).map(|_| AtomicUsize::new(0)).collect());
        let refused = Arc::new(AtomicUsize::new(0));
        let taken = Arc::new(AtomicUsize::new(0));
        let workers: Vec<_> = (0..THREADS)
            .map(|_| {
                let (pool, holders, refused, taken) = (
                    Arc::clone(&pool),
                    Arc::clone(&holders),
                    Arc::clone(&refused),
                    Arc::clone(&taken),
                );
                thread::spawn(move || {
                    for _ in 0..ROUNDS {
                        match pool.take() {
                            Some((id, lease)) => {
                                assert_eq!(holders[id.0].fetch_add(1, Ordering::SeqCst), 0, "held twice");
                                taken.fetch_add(1, Ordering::SeqCst);
                                thread::yield_now();
                                holders[id.0].fetch_sub(1, Ordering::SeqCst);
                                drop(lease);
                            }
                            None => {
                                refused.fetch_add(1, Ordering::SeqCst);
                            }
                        }
                    }
                })
            })
            .collect();
        for worker in workers {
            worker.join().unwrap();
        }
        assert_eq!(
            taken.load(Ordering::SeqCst) + refused.load(Ordering::SeqCst),
            THREADS * ROUNDS
        );
        assert!(taken.load(Ordering::SeqCst) >= ROUNDS, "the pool served requests");
        assert_eq!(pool.running(), 0, "every lease came back");
        // With every sequence held, exactly the extra callers are refused.
        let held: Vec<_> = (0..SEQUENCES).map(|_| pool.take().unwrap()).collect();
        let refusals = (0..THREADS - SEQUENCES).filter(|_| pool.take().is_none()).count();
        assert_eq!(refusals, THREADS - SEQUENCES);
        drop(held);
    }
}
