//! One small frame: a closure run once on a worker.

use euhedral_core::frames::{Frame, FrameError, FrameHeader};

/// The id the frames of this crate carry; they need no order among themselves.
const ID: u64 = 0x6170_6973;

struct Once {
    header: FrameHeader,
    work: Option<Box<dyn FnOnce() + Send>>,
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
            work();
        }
        Ok(())
    }
}

/// A frame that runs `work` once, on whichever worker takes it.
pub fn once(work: impl FnOnce() + Send + 'static) -> Box<dyn Frame> {
    Box::new(Once {
        header: FrameHeader::new(ID),
        work: Some(Box::new(work)),
    })
}
