//! Where the cache's loads run.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_graph::{Lake, UnitGuard};
use euhedral_hashing::hasher_api;

/// A place to run a task as a frame of its own.
///
/// A task that is refused is dropped without running; the owner of what it holds reports that (the cache's loader
/// does, in its `Drop`).
pub trait Frames: Send + Sync {
    fn submit(&self, task: Box<dyn FnOnce() + Send>);
}

/// Tasks as frames of the `host` lake, each with an id hash of its own so that they spread over the workers.
#[derive(Debug)]
pub struct HostFrames {
    host: Arc<Lake>,
    next: AtomicU64,
}

impl HostFrames {
    pub fn new(host: Arc<Lake>) -> HostFrames {
        HostFrames {
            host,
            next: AtomicU64::new(0),
        }
    }
}

impl Frames for HostFrames {
    fn submit(&self, task: Box<dyn FnOnce() + Send>) {
        // A lake that finished admits nothing: the task is dropped unrun.
        let Ok(unit) = self.host.admit() else { return };
        let id = self.next.fetch_add(1, Ordering::Relaxed);
        let frame = Task {
            header: FrameHeader::new(hasher_api::mix(id | 1 << 62)),
            task: Some(task),
            _unit: unit,
        };
        // A lake that finished hands the frame back; dropping it drops the task.
        drop(self.host.offer(Box::new(frame)));
    }
}

struct Task {
    header: FrameHeader,
    task: Option<Box<dyn FnOnce() + Send>>,
    /// The lake counts the task until it is dropped.
    _unit: UnitGuard,
}

impl Frame for Task {
    fn header(&self) -> &FrameHeader {
        &self.header
    }

    fn header_mut(&mut self) -> &mut FrameHeader {
        &mut self.header
    }

    fn execute(&mut self) -> Result<(), FrameError> {
        if let Some(task) = self.task.take() {
            task();
        }
        Ok(())
    }
}

impl fmt::Debug for Task {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Task").finish_non_exhaustive()
    }
}
