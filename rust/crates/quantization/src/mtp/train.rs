//! Training the drafter from a corpus of the base model's hidden states and tokens, as `quantize train-drafter` does.
//!
//! - [`progress`]: where a run is, what each outcome of the device does to it, and the header of the saved state.
//! - [`settings`]: the run's settings and their refusals, and the host arithmetic of windows, shuffle and schedule.

pub mod progress;
pub mod settings;
