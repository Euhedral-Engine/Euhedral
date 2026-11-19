//! The file operations `AsyncIo` has no call for: cutting a file to a length, making it durable and removing it.
//!
//! This module is the one place the quantization library blocks on a file. Each operation runs as a frame of the host
//! lake, so no lattice worker that serves other work waits on it, and its completion continues the chain that asked.
//! It is replaced by the asynchronous calls when `AsyncIo` has them (truncate, fsync and unlink).

use std::fmt;
use std::fs::{self, OpenOptions};
use std::io;
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_graph::Lake;
use euhedral_hashing::hasher_api;

/// What a blocking operation does to the file at `path`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Operation {
    /// Cut the file to `len` bytes (or extend it with zeros).
    SetLen(u64),
    /// Write everything the file holds, and its size, to the device.
    Sync,
    /// Remove the file.
    Remove,
}

#[allow(
    clippy::disallowed_methods,
    reason = "AsyncIo has no truncate, fsync or unlink: scoped blocking calls, each in a host-lake frame, until it does"
)]
fn perform(operation: Operation, path: &PathBuf) -> io::Result<()> {
    match operation {
        Operation::SetLen(len) => OpenOptions::new().write(true).open(path)?.set_len(len),
        Operation::Sync => OpenOptions::new().write(true).open(path)?.sync_all(),
        Operation::Remove => fs::remove_file(path),
    }
}

struct Blocking {
    header: FrameHeader,
    work: Option<Box<dyn FnOnce() + Send>>,
}

impl Frame for Blocking {
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

impl fmt::Debug for Blocking {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Blocking").finish_non_exhaustive()
    }
}

/// A frame that runs `work` once on a host-lake worker. `kind` tells the frames of different steps apart.
pub fn frame(kind: u64, work: impl FnOnce() + Send + 'static) -> Box<dyn Frame> {
    Box::new(Blocking {
        header: FrameHeader::new(hasher_api::mix(kind | 1 << 62)),
        work: Some(Box::new(work)),
    })
}

/// Offers the frame `work` to `host`. A lake that is closed refuses it and the work is dropped unrun, which the run
/// that owns the work reports as lost.
pub fn offer(host: &Arc<Lake>, kind: u64, work: impl FnOnce() + Send + 'static) {
    drop(host.offer(frame(kind, work)));
}

/// Runs `operation` on `path` as a host-lake frame and continues with its result.
pub fn run(host: &Arc<Lake>, operation: Operation, path: PathBuf, then: impl FnOnce(io::Result<()>) + Send + 'static) {
    offer(host, 1, move || then(perform(operation, &path)));
}
