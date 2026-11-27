//! The frames the engine's host work is made of: one closure each, run once, with the id hash of its kind.
//!
//! A frame owns what it needs by value and hands it to the next frame it publishes; nothing is shared that a frame
//! mutates. Frames with one id hash are ordered: they run in the order they were offered, on one CPU. Frames of
//! different chains differ in their hash, so the chains spread over the CPUs.

use std::fmt;

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_hashing::hasher_api;

/// What a frame of this engine does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Kind {
    /// The steps of the startup that follow one another: open, index, plan, allocate, assemble, ready.
    Startup,
    /// The reads of the trellis segment companions.
    Segments,
    /// One chain of the weight upload: its read completions.
    UploadRead,
    /// One chain of the weight upload: its piece checks on the host.
    UploadCheck,
    /// One chain of the weight upload: its copies on the device.
    UploadCopy,
    /// One chain of the weight upload: its copy completions.
    UploadDone,
    /// The owner of the engine's sequence: one turn of its work.
    Owner,
    /// The arrival of a teacher-forced step's logits row.
    Logits,
    /// The selection of a sampled request's token from its logits row, on the host.
    Sample,
    /// The running of the work a parked quantum waited for on the owner's lane.
    Reaped,
    /// The preparation of a constrained request's next mask, on the host, while the device computes the row it applies to.
    Prepare,
    /// The prefix cache's copy jobs of one sequence: their stages and their completions, in the order they were admitted.
    Copy,
    /// The completions of the reads of the disk window, one chain per slot.
    DiskRead,
    /// The completions of the copies out of the disk window, one chain per slot.
    DiskCopy,
    /// The completions of the parts of one routed-expert record's read, one chain per part.
    ExpertRead,
    /// The copy of a routed-expert record out of the host tier into a staging buffer.
    ExpertCopy,
    /// The delivery of a hidden-rows request's rows to its sink.
    Hidden,
    /// The submission of a routed expert's copy to the device on the copy lane.
    ExpertSubmit,
    /// The retirement of a routed expert's copy: its staging buffer is free and its load ends.
    ExpertRetire,
    /// The turn of the routed-expert cache's owner: it applies the records posted to it.
    ExpertOwner,
}

/// The id hash of a frame of `kind` on chain `chain`: distinct per kind and chain, mixed so that the chains spread
/// over the lattice's partitions.
pub(crate) fn id_hash(kind: Kind, chain: usize) -> u64 {
    hasher_api::mix(((kind as u64) << 32) | chain as u64 | 1 << 63)
}

/// A frame that runs a closure once.
pub(crate) struct Step {
    header: FrameHeader,
    work: Option<Box<dyn FnOnce() + Send>>,
}

/// A frame of `kind` on `chain` that runs `work` once.
pub(crate) fn step(kind: Kind, chain: usize, work: impl FnOnce() + Send + 'static) -> Box<dyn Frame> {
    Box::new(Step {
        header: FrameHeader::new(id_hash(kind, chain)),
        work: Some(Box::new(work)),
    })
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

impl fmt::Debug for Step {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Step").finish_non_exhaustive()
    }
}
