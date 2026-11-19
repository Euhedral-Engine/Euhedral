//! A whole quantum as one body of RustEE's graph cache.
//!
//! The launches of every stage that has work, in the order of the view, are one [`GraphBody`]: the cache runs it
//! directly the first time a key is seen, records it the second time, and replays the recording after, as recorded
//! (the key covers every argument, see [`CaptureKey`](super::CaptureKey)). The run is one marked submission on the
//! quantum's home lane, and RustEE's retirement of the quantum is armed on that lane, behind whatever the run queued
//! even when it failed, so a quantum that is stopped or fails retires as it does stage by stage.

use std::fmt;
use std::hash::Hash;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::OnceLock;

use euhedral_gpu::{Enqueue, Gpu, GpuError, GraphBody, GraphHold, Lane, Ran, SharedGraphCache, Stream};
use euhedral_graph::{Marker, StageError};

use crate::stages::fatal;

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

/// Where a quantum's body runs: the lane, and behind what.
pub struct Site<'a, G: Gpu> {
    /// The device the cache records on.
    pub gpu: &'a G,
    /// The lane the body is submitted on.
    pub lane: &'a <G::Stream as Stream>::Lane,
    /// The markers the body is ordered behind.
    pub waits: &'a [Marker],
}

impl<G: Gpu> fmt::Debug for Site<'_, G> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Site")
            .field("waits", &self.waits)
            .finish_non_exhaustive()
    }
}

/// Runs `launches` at `site` through `cache` under `key` as one marked submission, the quantum's only one on its home
/// lane.
///
/// What the run keeps alive for the device goes to `hold`, which the caller holds until the quantum retired. Nothing
/// waits: a lane with no free marker answers [`StageError::Retry`] before anything was queued. A launch that fails
/// gives its error back as the stage's; the work it queued is on the home lane, which RustEE's retirement covers.
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
    let Site { gpu, lane, waits } = site;
    let mut body = Body {
        launches,
        refused: None,
    };
    let made = lane.submit_marked(|_| cache.run_after(gpu, lane, waits, key, &mut body, hold));
    made.map_err(|error| match error {
        GpuError::Busy => StageError::Retry,
        other => body.refused.take().unwrap_or_else(|| fatal(other)),
    })
}

/// As [`run_cached`], with the hold made here and put in `store` before the result (or a panic of the body) goes on:
/// the hold is the quantum's, dropped after it retired, never at an unwind. A run that did not start (a retry) keeps
/// nothing and stores nothing; one that failed or panicked may have queued work that reads what it kept, so its hold
/// is stored.
pub fn run_held<G, K, T>(
    site: &Site<'_, G>,
    cache: &SharedGraphCache<G, K>,
    key: K,
    launches: &mut T,
    store: &OnceLock<GraphHold>,
) -> Result<(Ran, Marker), StageError>
where
    G: Gpu,
    K: Copy + Eq + Hash + Send,
    T: Launches<G>,
{
    let mut hold = GraphHold::new();
    let made = catch_unwind(AssertUnwindSafe(|| run_cached(site, cache, key, launches, &mut hold)));
    if !matches!(made, Ok(Err(StageError::Retry))) {
        let _ = store.set(hold);
    }
    match made {
        Ok(made) => made,
        Err(panic) => resume_unwind(panic),
    }
}
