//! The continuation of a call, held until one step ends the call.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use euhedral_core::frames::Frame;
use euhedral_data_structures::sync::ClaimCell;

use super::{ResumeError, Then};
use crate::step::idle;

/// No step is being submitted.
const QUIET: u8 = 0;
/// A step is being submitted: a watch dropped now was dropped by the submission itself, which tells its caller.
const SUBMITTING: u8 = 1;
/// A watch was dropped while a step was being submitted.
const DROPPED: u8 = 2;

struct Shared<T> {
    then: ClaimCell<Option<Then<T>>>,
    state: AtomicU8,
}

/// A continuation that waits to be taken by whichever of the steps of a call ends the call, so that it runs once. A
/// step that is dropped without running (the lake or the I/O engine is closing) ends the call with
/// [`ResumeError::LakeClosed`]: see [`Slot::submit`] and [`Watch`].
pub(super) struct Slot<T>(Arc<Shared<T>>);

impl<T> Clone for Slot<T> {
    fn clone(&self) -> Slot<T> {
        Slot(Arc::clone(&self.0))
    }
}

impl<T> Slot<T> {
    pub(super) fn new(then: Then<T>) -> Slot<T> {
        Slot(Arc::new(Shared {
            then: ClaimCell::new(Some(then)),
            state: AtomicU8::new(QUIET),
        }))
    }

    /// Ends the call with `result`: runs the continuation if no step has yet, and returns its frame.
    pub(super) fn finish(&self, result: Result<T, ResumeError>) -> Box<dyn Frame> {
        let taken = self.0.then.try_claim().and_then(|mut claim| claim.take());
        match taken {
            Some(then) => then(result),
            None => idle(),
        }
    }

    /// Ends the call with the error of a step that could not start; a step that started is nothing to do.
    pub(super) fn fail<U>(&self, started: Result<U, ResumeError>) -> Box<dyn Frame> {
        match started {
            Ok(_) => idle(),
            Err(error) => self.finish(Err(error)),
        }
    }

    /// Starts a step. `held` is what the step keeps until it runs; `start` gets the watch the step's closure must own
    /// and submits the closure.
    ///
    /// A submission that returns an error has dropped the closure, and tells its caller by the error: the watch says
    /// nothing then. A closure that is dropped after the submission returned (a closing lake discards the frame) ends
    /// the call with `LakeClosed`, whichever thread drops it, even if that happens before `start` returns.
    pub(super) fn submit<H, U>(
        &self,
        held: H,
        start: impl FnOnce(Watch<T, H>) -> Result<U, ResumeError>,
    ) -> Result<U, ResumeError> {
        self.0.state.store(SUBMITTING, Ordering::SeqCst);
        let started = start(Watch {
            held,
            ending: Ending {
                slot: self.clone(),
                armed: true,
            },
        });
        let before = self.0.state.swap(QUIET, Ordering::SeqCst);
        if before == DROPPED && started.is_ok() {
            drop(self.finish(Err(ResumeError::LakeClosed)));
        }
        started
    }
}

/// What a step's closure owns: what the step keeps (`held`) and the duty to end the call if the closure is dropped
/// without running. `held` is dropped first, so a continuation that ends the call finds no handle of the step.
pub(super) struct Watch<T, H> {
    held: H,
    ending: Ending<T>,
}

impl<T, H> Watch<T, H> {
    /// The closure runs: the watch no longer ends the call. Returns the call's slot and what the step kept.
    pub(super) fn take(self) -> (Slot<T>, H) {
        let Watch { held, ending } = self;
        (ending.disarm(), held)
    }
}

struct Ending<T> {
    slot: Slot<T>,
    armed: bool,
}

impl<T> Ending<T> {
    fn disarm(mut self) -> Slot<T> {
        self.armed = false;
        self.slot.clone()
    }
}

impl<T> Drop for Ending<T> {
    // This runs the caller's continuation inside a drop, on the closing lake's thread: a panic there aborts.
    fn drop(&mut self) {
        if !self.armed {
            return;
        }
        let swapped = self
            .slot
            .0
            .state
            .compare_exchange(SUBMITTING, DROPPED, Ordering::SeqCst, Ordering::SeqCst);
        if swapped.is_err() {
            drop(self.slot.finish(Err(ResumeError::LakeClosed)));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;

    use super::*;

    /// Counts what the continuation was told: calls, and how many of them were `LakeClosed`.
    #[derive(Default)]
    struct Log {
        told: AtomicUsize,
        closed: AtomicUsize,
        value: AtomicUsize,
    }

    fn recorder() -> (Arc<Log>, Slot<usize>) {
        let log = Arc::new(Log::default());
        let seen = Arc::clone(&log);
        let slot = Slot::new(Box::new(move |result: Result<usize, ResumeError>| {
            seen.told.fetch_add(1, Ordering::SeqCst);
            match result {
                Ok(value) => seen.value.store(value, Ordering::SeqCst),
                Err(ResumeError::LakeClosed) => {
                    seen.closed.fetch_add(1, Ordering::SeqCst);
                }
                Err(_) => {}
            }
            idle()
        }));
        (log, slot)
    }

    /// A step accepted and then discarded ends the call with `LakeClosed`, once.
    #[test]
    fn a_step_dropped_after_submission_ends_the_call() {
        let (log, slot) = recorder();
        let mut kept = None;
        slot.submit((), |watch| {
            kept = Some(watch);
            Ok(())
        })
        .unwrap();
        assert_eq!(log.told.load(Ordering::SeqCst), 0);
        drop(kept);
        assert_eq!(log.told.load(Ordering::SeqCst), 1);
        assert_eq!(log.closed.load(Ordering::SeqCst), 1);
    }

    /// A step dropped by the submission that failed says nothing: the error is the caller's.
    #[test]
    fn a_step_dropped_by_a_failed_submission_is_silent() {
        let (log, slot) = recorder();
        let started = slot.submit((), |watch| {
            drop(watch);
            Err::<(), _>(ResumeError::Busy)
        });
        assert!(matches!(started, Err(ResumeError::Busy)));
        assert_eq!(log.told.load(Ordering::SeqCst), 0);
    }

    /// A step dropped before its submission returns, which then succeeds, still ends the call.
    #[test]
    fn a_step_dropped_inside_a_submission_that_succeeds_ends_the_call() {
        let (log, slot) = recorder();
        slot.submit((), |watch| {
            drop(watch);
            Ok(())
        })
        .unwrap();
        assert_eq!(log.told.load(Ordering::SeqCst), 1);
        assert_eq!(log.closed.load(Ordering::SeqCst), 1);
    }

    /// A step that runs takes the call over: dropping what is left says nothing, and the call ends once.
    #[test]
    fn a_step_that_ran_ends_the_call_once() {
        let (log, slot) = recorder();
        slot.submit((), |watch| {
            let (slot, ()) = watch.take();
            drop(slot.finish(Ok(7)));
            drop(slot.finish(Ok(8)));
            Ok(())
        })
        .unwrap();
        assert_eq!(log.told.load(Ordering::SeqCst), 1);
        assert_eq!(log.value.load(Ordering::SeqCst), 7);
    }
}
