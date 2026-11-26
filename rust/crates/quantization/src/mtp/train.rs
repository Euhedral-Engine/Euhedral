//! Training the drafter from a corpus of the base model's hidden states and tokens, as `quantize train-drafter` does.
//!
//! - [`load`]: the startup, which waits: the device, the drafter's parameters, the artifact, the corpus, the state.
//! - [`run`]: the run as frames, from the first window to the last save.
//! - [`lane`]: the lane's stream with the drafter, and what it runs for a window, an update and a read.
//! - [`data`]: the corpus in memory, checked, and its split into training and held-out sequences.
//! - [`state`]: where each part of the parameters lies in the saved files.
//! - [`progress`]: where a run is, what each outcome of the device does to it, and the header of the saved state.
//! - [`stage`]: the device inputs of a window and the rows its drafts are checked against.
//! - [`settings`]: the run's settings and their refusals, and the host arithmetic of windows, shuffle and schedule.

pub mod data;
pub mod error;
pub mod lane;
pub mod load;
pub mod options;
pub mod progress;
pub mod run;
pub mod settings;
pub mod stage;
pub mod state;
pub mod wait;
pub mod window;

use std::sync::Arc;
use std::time::Duration;

pub use error::RunError;
pub use lane::Lane;
pub use options::{Host, Log, Options};
pub use run::{Ended, Summary};
use wait::{Reply, pair};

/// Trains the drafter as `options` say: loads, runs and closes the lane. Waits for the run, so the caller is the
/// binary's own thread, never a worker.
pub fn train(options: &Options, host: &Host<'_>, log: &Log) -> Result<Summary, RunError> {
    let ready = load::load(options, host, log)?;
    drive(options.stall, |reply| {
        run::start(ready, options, host, Arc::clone(log), reply)
    })
}

/// Starts a run with `start`, follows it until it ends and closes the lane it gives back, here on the calling
/// thread. The run's error comes before the close's. A run that finishes no call over `stall` is told to end at its
/// next one, which returns its lane to be closed; one that never finishes a call fails, naming that no window, update,
/// read or write finished.
pub fn drive<L: Lane>(stall: Duration, start: impl FnOnce(Arc<Reply<Ended<L>>>)) -> Result<Summary, RunError> {
    let (wait, reply) = pair();
    start(reply);
    let ended = wait.follow(stall)?;
    let closed = ended.lane.close();
    let summary = ended.outcome?;
    closed?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    #![allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]

    use std::sync::atomic::{AtomicU32, Ordering};

    use super::lane::{Done, Job, Outcome};
    use super::progress::Update;
    use super::wait::STALL;
    use super::*;

    /// A lane that only counts its closes.
    struct Counting {
        closes: Arc<AtomicU32>,
        close_error: bool,
    }

    impl Lane for Counting {
        fn window(self, _: Job, _: Done<Self, Outcome>) {}
        fn update(self, _: Update, _: Done<Self, ()>) {}
        fn read(self, _: u64, _: usize, _: Done<Self, Vec<u8>>) {}
        fn close(self) -> Result<(), RunError> {
            self.closes.fetch_add(1, Ordering::SeqCst);
            if self.close_error {
                Err(RunError::Broken("the close failed".to_owned()))
            } else {
                Ok(())
            }
        }
    }

    fn summary() -> Summary {
        Summary {
            saves: 2,
            stopped: false,
            best: 0.5,
        }
    }

    fn drive_with(outcome: Result<Summary, RunError>, close_error: bool) -> (Result<Summary, RunError>, u32) {
        let closes = Arc::new(AtomicU32::new(0));
        let lane = Counting {
            closes: Arc::clone(&closes),
            close_error,
        };
        let result = drive(STALL, |reply| reply.send(Ok(Ended { lane, outcome })));
        (result, closes.load(Ordering::SeqCst))
    }

    #[test]
    fn a_run_that_ends_gives_its_summary_and_its_lane_is_closed_once() {
        let (result, closes) = drive_with(Ok(summary()), false);
        assert_eq!(result.unwrap(), summary());
        assert_eq!(closes, 1);
    }

    #[test]
    fn a_run_that_fails_closes_its_lane_and_reports_the_run_error_before_the_close_error() {
        let (result, closes) = drive_with(Err(RunError::Broken("the run failed".to_owned())), true);
        assert!(matches!(result, Err(RunError::Broken(message)) if message == "the run failed"));
        assert_eq!(closes, 1);
    }

    #[test]
    fn a_close_that_fails_after_a_good_run_is_reported() {
        let (result, closes) = drive_with(Ok(summary()), true);
        assert!(matches!(result, Err(RunError::Broken(message)) if message == "the close failed"));
        assert_eq!(closes, 1);
    }

    #[test]
    fn a_run_dropped_without_an_answer_fails_without_a_lane_to_close() {
        let result = drive::<Counting>(STALL, drop);
        assert!(matches!(result, Err(RunError::Broken(message)) if message.contains("without answering")));
    }
}
