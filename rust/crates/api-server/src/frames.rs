//! One small frame: a closure run once on a worker.
//!
//! Each frame carries a routing hash of its own, so no two frames of the server are tied to one worker's order: they
//! run wherever a worker is free. Work that must stay in order (the pieces of one reply) is chained by the closure
//! that offers the next piece, not by a shared hash.

use std::sync::atomic::{AtomicU64, Ordering};

use euhedral_core::frames::{Frame, FrameError, FrameHeader};

/// The id the frames of this crate carry.
const ID: u64 = 0x6170_6973;

/// The seed of the next frame's hash.
static SEEDS: AtomicU64 = AtomicU64::new(0);

struct Once {
    header: FrameHeader,
    work: Option<Box<dyn FnOnce(bool) + Send>>,
}

impl Frame for Once {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(work) = self.work.take() {
            work(true);
        }
        Ok(())
    }
}

impl Drop for Once {
    /// A frame that never ran (the lattice refused it or let it go) tells its work so.
    fn drop(&mut self) {
        if let Some(work) = self.work.take() {
            work(false);
        }
    }
}

/// A frame that runs `work` once, on whichever worker takes it. Nothing runs if the frame is refused or discarded.
pub fn once(work: impl FnOnce() + Send + 'static) -> Box<dyn Frame> {
    once_or_refused(move |ran| {
        if ran {
            work();
        }
    })
}

/// A frame that calls `work(true)` once on a worker, or `work(false)` once, on whichever thread drops it, if it
/// never ran: so a continuation inside it is called exactly once whatever the lattice does with the frame.
pub fn once_or_refused(work: impl FnOnce(bool) + Send + 'static) -> Box<dyn Frame> {
    let mut header = FrameHeader::new(ID);
    header.randomize_hash(SEEDS.fetch_add(1, Ordering::Relaxed));
    Box::new(Once {
        header,
        work: Some(Box::new(work)),
    })
}
