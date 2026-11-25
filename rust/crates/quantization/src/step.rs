//! A frame that runs a closure once: the way the library's own steps are offered to a lake.

use std::fmt::{Debug, Formatter, Result as FmtResult};

use euhedral_core::frames::{DummyFrame, Frame, FrameError, FrameHeader};
use euhedral_hashing::hasher_api;

/// What a frame of this library is, for its id hash.
#[derive(Clone, Copy, Debug)]
pub(crate) enum Kind {
    /// A durability, truncation or removal call, run on the host lake.
    Durable = 1,
    /// A step that lets go of what a finished call held, after the call's completion was offered.
    Release = 2,
    /// The continuation of a finished call, offered to the host lake.
    Resolve = 3,
}

struct Step {
    header: FrameHeader,
    work: Option<Box<dyn FnOnce() + Send>>,
}

/// A frame of `kind` that runs `work` once.
pub(crate) fn step(kind: Kind, work: impl FnOnce() + Send + 'static) -> Box<dyn Frame> {
    Box::new(Step {
        header: FrameHeader::new(hasher_api::mix((kind as u64) << 32 | 1 << 63)),
        work: Some(Box::new(work)),
    })
}

/// A frame that does nothing: what a completion returns when it has handed the chain to another call.
pub(crate) fn idle() -> Box<dyn Frame> {
    Box::new(DummyFrame::INSTANCE)
}

impl Frame for Step {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(work) = self.work.take() {
            work();
        }
        Ok(())
    }
}

impl Debug for Step {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Step").finish_non_exhaustive()
    }
}
