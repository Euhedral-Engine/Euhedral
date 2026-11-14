//! A whole quantum as one body of RustEE's graph cache.
//!
//! The launches of every stage that has work, in the order of the view, are one [`GraphBody`]: the cache runs it
//! directly the first time a key is seen, records it the second time, and replays the recording after, as recorded
//! (the key covers every argument, see [`CaptureKey`](super::CaptureKey)). The run is one marked submission on the
//! quantum's home lane; its marker is recorded in the quantum's [`LaneEnds`] like any stage's, so a quantum that is
//! stopped or fails settles and retires as it does stage by stage.

use std::fmt;
use std::hash::Hash;

use euhedral_gpu::{Enqueue, Gpu, GpuError, GraphBody, GraphHold, Lane, Ran, SharedGraphCache, Stream};
use euhedral_graph::{Marker, StageError};

use crate::stages::{LaneEnds, fatal};

/// The launches of a quantum, which the cache runs as a graph body.
pub trait Launches<G: Gpu> {
    /// Enqueues every launch on `enq`, in order, with the quantum's arguments.
    fn launch<E: Enqueue<Kernel = G::Kernel>>(&mut self, enq: &mut E) -> Result<(), StageError>;
}

/// The body the cache runs: the quantum's launches, and the error a launch refused with.
struct Body<'a, T> {
    launches: &'a mut T,
    refused: Option<StageError>,
}

impl<G: Gpu, T: Launches<G>> GraphBody<G> for Body<'_, T> {
    fn enqueue<E: Enqueue<Kernel = G::Kernel>>(&mut self, enq: &mut E) -> Result<(), GpuError> {
        self.launches.launch(enq).map_err(|error| {
            self.refused = Some(error);
            GpuError::Invalid("a stage refused its quantum".to_string())
        })
    }
}

/// Where a quantum's body runs: the lane, behind what, and where its marker is recorded.
pub struct Site<'a, G: Gpu> {
    /// The device the cache records on.
    pub gpu: &'a G,
    /// The lane the body is submitted on.
    pub lane: &'a <G::Stream as Stream>::Lane,
    /// The markers the body is ordered behind.
    pub waits: &'a [Marker],
    /// The quantum's record of its submissions.
    pub ends: &'a LaneEnds,
}

impl<G: Gpu> fmt::Debug for Site<'_, G> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Site")
            .field("waits", &self.waits)
            .finish_non_exhaustive()
    }
}

/// Runs `launches` at `site` through `cache` under `key` as one marked submission, and records the marker in the
/// site's `ends` as signalled: it is the quantum's only submission on the lane, so nothing follows it to settle.
///
/// What the run keeps alive for the device goes to `hold`, which the caller holds until the quantum retired. Nothing
/// waits: a lane with no free marker answers [`StageError::Retry`] before anything was queued. A launch that fails
/// gives its error back as the stage's and flags the lane, whose queued work has no marker, to be settled.
pub fn run_cached<G, K, T>(
    site: &Site<'_, G>,
    cache: &SharedGraphCache<G, K>,
    key: K,
    launches: &mut T,
    hold: &mut GraphHold,
) -> Result<(Ran, Marker), StageError>
where
    G: Gpu,
    K: Copy + Eq + Hash + Send,
    T: Launches<G>,
{
    let Site { gpu, lane, waits, ends } = site;
    let mut body = Body {
        launches,
        refused: None,
    };
    let made = lane.submit_marked(|_| cache.run_after(gpu, lane, waits, key, &mut body, hold));
    let made = made.map_err(|error| match error {
        GpuError::Busy => StageError::Retry,
        other => body.refused.take().unwrap_or_else(|| fatal(other)),
    });
    ends.ended(lane.lane(), true, made)
}
