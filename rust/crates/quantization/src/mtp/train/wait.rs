//! The one place the run's startup and shutdown wait: the calling thread starts an asynchronous call and blocks until
//! the frame that ends it answers. A frame never uses this.
//!
//! A single call is bounded by [`LONGEST_CALL`]. The run, which may last days, is not bounded by a time: its frames
//! [`beat`](Reply::beat) at every window, update, read and write they finish, and [`follow`](Wait::follow) fails it
//! only when a whole [`STALL`] passes with no beat.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Duration;

use euhedral_inference_engine::Handoff;

use super::error::RunError;
use crate::resume::Then;
use crate::step::idle;

/// The longest the startup or the shutdown waits for one call: a bound on the wait, not a rate.
const LONGEST_CALL: Duration = Duration::from_secs(24 * 60 * 60);

/// How long a run may go without finishing any call before it counts as hung: a window, an update, a read or a write
/// takes seconds, and the largest piece of a save a few more.
const STALL: Duration = Duration::from_secs(30 * 60);

/// The waiting end.
pub struct Wait<T> {
    cell: Arc<Handoff<Result<T, RunError>>>,
    beats: Arc<AtomicU64>,
}

/// The answering end: the first answer counts, and a call dropped without one answers that it ended.
pub struct Reply<T> {
    cell: Arc<Handoff<Result<T, RunError>>>,
    sent: AtomicBool,
    beats: Arc<AtomicU64>,
}

impl<T> Debug for Wait<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Wait").finish_non_exhaustive()
    }
}

impl<T> Debug for Reply<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Reply").finish_non_exhaustive()
    }
}

/// A waiting end and the answering end that reaches it.
pub fn pair<T>() -> (Wait<T>, Arc<Reply<T>>) {
    let (cell, beats) = (Handoff::new(), Arc::new(AtomicU64::new(0)));
    let reply = Arc::new(Reply {
        cell: Arc::clone(&cell),
        sent: AtomicBool::new(false),
        beats: Arc::clone(&beats),
    });
    (Wait { cell, beats }, reply)
}

/// Tells whether a run has stopped finishing calls: it has when the beats are the same at two looks.
#[derive(Debug)]
struct Watch {
    last: u64,
}

impl Watch {
    /// Looks at the beats; true when they have not moved since the look before.
    fn stalled(&mut self, beats: u64) -> bool {
        let same = beats == self.last;
        self.last = beats;
        same
    }
}

impl<T> Reply<T> {
    /// Tells the waiting end that the run finished a call, so that it is not hung.
    pub fn beat(&self) {
        self.beats.fetch_add(1, Ordering::Relaxed);
    }

    /// Answers, unless an answer has been given.
    pub fn send(&self, answer: Result<T, RunError>) {
        if !self.sent.swap(true, Ordering::AcqRel) {
            self.cell.send(answer);
        }
    }
}

impl<T> Drop for Reply<T> {
    fn drop(&mut self) {
        self.send(Err(RunError::Broken("a call ended without answering".to_owned())));
    }
}

impl<T: Send + 'static> Wait<T> {
    /// Blocks until the call answers.
    #[allow(
        clippy::disallowed_methods,
        reason = "startup and shutdown: the calling thread is the binary's own, never a worker"
    )]
    pub fn block(self) -> Result<T, RunError> {
        self.cell
            .wait(LONGEST_CALL)
            .map_err(|_| RunError::Broken("a call did not answer in time".to_owned()))?
    }
}

impl<T: Send + 'static> Wait<T> {
    /// Blocks until the run answers, however long it takes, and fails only when it stops finishing calls: the error
    /// names what never happened.
    pub fn follow(self) -> Result<T, RunError> {
        self.follow_within(STALL)
    }

    /// [`follow`](Wait::follow) with `stall` as the look interval: fails when the beats did not move over one.
    #[allow(
        clippy::disallowed_methods,
        reason = "shutdown: the calling thread is the binary's own, never a worker"
    )]
    pub(crate) fn follow_within(self, stall: Duration) -> Result<T, RunError> {
        let mut watch = Watch {
            last: self.beats.load(Ordering::Relaxed),
        };
        loop {
            if let Ok(answer) = self.cell.wait(stall) {
                return answer;
            }
            if watch.stalled(self.beats.load(Ordering::Relaxed)) {
                return Err(RunError::Broken(format!(
                    "the run finished no window, update, read or write in {} seconds and did not end",
                    stall.as_secs()
                )));
            }
        }
    }
}

/// The `Done` of a call that reports to `reply`.
pub fn done<T, E>(reply: &Arc<Reply<T>>) -> Arc<dyn Fn(Result<T, E>) + Send + Sync>
where
    T: Send + 'static,
    E: Into<RunError> + 'static,
{
    let reply = Arc::clone(reply);
    Arc::new(move |result| reply.send(result.map_err(Into::into)))
}

/// The continuation of a resume file's call that reports to `reply`.
pub fn then<T: Send + 'static>(reply: &Arc<Reply<T>>) -> Then<T> {
    let reply = Arc::clone(reply);
    Box::new(move |result| {
        reply.send(result.map_err(RunError::from));
        idle()
    })
}

/// Runs a call that starts with `start(reply)` and waits for its answer. A call that cannot start returns its error.
pub fn call<T, E>(start: impl FnOnce(Arc<Reply<T>>) -> Result<(), E>) -> Result<T, RunError>
where
    T: Send + 'static,
    E: Into<RunError>,
{
    let (wait, reply) = pair();
    start(reply).map_err(Into::into)?;
    wait.block()
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]

    use super::*;

    #[test]
    fn a_run_is_stalled_only_when_no_call_finished_since_the_last_look() {
        let mut watch = Watch { last: 0 };
        assert!(watch.stalled(0), "nothing finished at all");
        assert!(!watch.stalled(3));
        assert!(!watch.stalled(4));
        assert!(watch.stalled(4));
        assert!(!watch.stalled(5), "a beat after a stall counts again");
    }

    #[test]
    fn an_answer_already_given_ends_the_following_at_once() {
        let (wait, reply) = pair();
        reply.beat();
        reply.send(Ok(3));
        assert_eq!(wait.follow().unwrap(), 3);
    }

    #[test]
    fn a_run_dropped_without_an_answer_ends_the_following_with_the_reason() {
        let (wait, reply) = pair::<u32>();
        drop(reply);
        assert!(matches!(wait.follow(), Err(RunError::Broken(message)) if message.contains("without answering")));
    }

    #[test]
    fn a_run_that_finishes_no_call_and_gives_no_answer_fails_naming_what_never_happened() {
        let (wait, reply) = pair::<u32>();
        let error = wait.follow_within(Duration::from_millis(20)).unwrap_err();
        assert!(
            matches!(&error, RunError::Broken(message) if message.contains("finished no window, update, read or write")),
            "{error}"
        );
        drop(reply);
    }

    #[test]
    fn the_first_answer_counts() {
        let (wait, reply) = pair();
        reply.send(Ok(1));
        reply.send(Ok(2));
        reply.send(Err(RunError::Broken("late".to_owned())));
        assert_eq!(wait.block().unwrap(), 1);
    }

    #[test]
    fn a_reply_dropped_unanswered_answers_that_it_ended() {
        let (wait, reply) = pair::<u32>();
        drop(reply);
        let error = wait.block().unwrap_err();
        assert!(matches!(error, RunError::Broken(message) if message.contains("without answering")));
    }

    #[test]
    fn an_answer_survives_the_drop_of_its_reply() {
        let (wait, reply) = pair();
        reply.send(Ok(7));
        drop(reply);
        assert_eq!(wait.block().unwrap(), 7);
    }

    #[test]
    fn a_call_that_cannot_start_returns_its_own_error() {
        let result = call::<u32, RunError>(|_| Err(RunError::Broken("no start".to_owned())));
        assert!(matches!(result, Err(RunError::Broken(message)) if message == "no start"));
    }

    #[test]
    fn a_call_returns_what_its_continuation_answers() {
        let result = call::<u32, RunError>(|reply| {
            reply.send(Ok(5));
            Ok(())
        });
        assert_eq!(result.unwrap(), 5);
    }
}
