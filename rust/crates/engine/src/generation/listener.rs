//! Where a generation's events go.

use std::time::Instant;

use super::request::{GenerationOutcome, QuantumEvent};

/// What a generation tells its caller. Every method runs on a worker of the engine's lattice, one call at a time and in
/// the order of the events, and must not wait: hand the event on (set a latch, offer a frame) and return.
pub trait GenerationListener: Send {
    /// The last prompt chunk has retired, at `at`; the first token is read next.
    fn prefill_done(&mut self, at: Instant);

    /// The generation selected token `id`, found at `at`.
    fn token(&mut self, id: u32, at: Instant);

    /// A quantum of the generation retired.
    fn quantum(&mut self, event: QuantumEvent);

    /// The generation ended. This is the last call, made after the engine's sequence can take another generation.
    fn finished(&mut self, outcome: GenerationOutcome);
}
