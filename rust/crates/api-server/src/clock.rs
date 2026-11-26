//! The time admission reads and the timers it arms, behind one trait so that a test can own both.

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;
use std::time::Instant;

use euhedral_core::frames::Frame;
use euhedral_graph::Completions;

/// What admission knows of time: the moment now, and a frame to emit at a later moment. A clock whose `now` and `at`
/// disagree with each other would arm timers that never fire, so an implementation owns both.
pub trait Clock: Send + Sync {
    /// The moment now.
    fn now(&self) -> Instant;

    /// Emits `frame` into the lattice at or after `deadline`. A frame whose deadline has passed is emitted at once.
    fn at(&self, deadline: Instant, frame: Box<dyn Frame>);
}

/// The wall clock, with the lattice's timers.
pub struct Timers {
    completions: Arc<Completions>,
}

impl Debug for Timers {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("Timers").finish_non_exhaustive()
    }
}

impl Timers {
    /// The wall clock, whose timers are `completions` (attached to the lattice).
    pub fn new(completions: Arc<Completions>) -> Timers {
        Timers { completions }
    }
}

impl Clock for Timers {
    fn now(&self) -> Instant {
        Instant::now()
    }

    fn at(&self, deadline: Instant, frame: Box<dyn Frame>) {
        self.completions.at(deadline, frame);
    }
}
