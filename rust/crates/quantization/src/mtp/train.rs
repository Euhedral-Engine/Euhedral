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

pub use error::RunError;
pub use options::{Host, Log, Options};
pub use run::Summary;
use wait::pair;

/// Trains the drafter as `options` say: loads, runs and closes the lane. Waits for the run, so the caller is the
/// binary's own thread, never a worker.
pub fn train(options: &Options, host: &Host<'_>, log: &Log) -> Result<Summary, RunError> {
    let loaded = load::load(options, host, log)?;
    let (wait, reply) = pair();
    run::start(loaded, options, host, Arc::clone(log), reply);
    let ended = wait.block()?;
    let closed = ended.lane.close();
    drop(ended.keep);
    let summary = ended.outcome?;
    closed?;
    Ok(summary)
}
