//! CUDA graphs: capture a lane's submissions once, replay them.

use std::sync::Arc;

use crate::error::{CudaError, check};
use crate::lane::CudaLane;
use crate::library::CudaLibrary;
use crate::submission::{Submission, SubmitMode};

/// A capture in progress on a lane.
///
/// Between [`GraphRecorder::begin`] and [`GraphRecorder::end`] the lane's stream records instead of running: every
/// submission made through [`GraphRecorder::submission`] queues its launches, copies and memsets into the graph, and
/// nothing executes. The recorder holds the lane for the whole capture, so [`Submission::begin`] on it (or another
/// capture) fails with `LaneBusy`. The capture is relaxed, so other threads' CUDA calls are unaffected. Capture on a
/// lane dedicated to it; work is not submitted twice and no shadow stream is involved.
///
/// Warm every kernel (the first launch of a kernel loads its module) before capturing. A capture cannot take
/// stream-ordered allocations or frees, synchronous copies, markers, waits on other lanes' markers, completion
/// signals or lane synchronization; those return `Capturing`. Dropping a recorder without ending it ends the capture
/// and discards the graph.
#[derive(Debug)]
pub struct GraphRecorder<'l> {
    lane: &'l CudaLane,
    ended: bool,
}

impl<'l> GraphRecorder<'l> {
    /// Takes `lane` and starts capturing its stream. Fails with `LaneBusy` when a submission or capture holds it.
    pub fn begin(lane: &'l CudaLane) -> Result<GraphRecorder<'l>, CudaError> {
        lane.try_own()?;
        lane.set_capturing(true);
        // SAFETY: the stream is the lane's live stream.
        let status = unsafe { lane.library().api().euhedral_cuda_graph_capture_begin(lane.stream()) };
        if let Err(error) = check("euhedral_cuda_graph_capture_begin", status) {
            lane.set_capturing(false);
            lane.disown();
            return Err(error);
        }
        Ok(GraphRecorder { lane, ended: false })
    }

    /// A submission on the captured lane; what it queues is recorded. Drop it before ending the capture.
    pub fn submission(&self, mode: SubmitMode) -> Result<Submission<'_>, CudaError> {
        Submission::borrowing(self.lane, mode)
    }

    /// Ends the capture and instantiates it. Any [`Submission`] on the lane has been dropped by then.
    pub fn end(mut self) -> Result<GraphExec, CudaError> {
        let exec = self.finish_capture()?;
        Ok(GraphExec {
            library: Arc::clone(self.lane.library()),
            exec,
        })
    }

    /// Ends the capture; the lane is usable afterwards even when the capture failed.
    fn finish_capture(&mut self) -> Result<u64, CudaError> {
        self.ended = true;
        let mut exec = 0u64;
        // SAFETY: the stream is the lane's, in capture; `exec` is a live local.
        let status = unsafe {
            self.lane
                .library()
                .api()
                .euhedral_cuda_graph_capture_end(self.lane.stream(), &mut exec)
        };
        self.lane.set_capturing(false);
        self.lane.disown();
        check("euhedral_cuda_graph_capture_end", status)?;
        if exec == 0 {
            return Err(CudaError::Argument("the capture instantiated no graph".to_string()));
        }
        Ok(exec)
    }
}

impl Drop for GraphRecorder<'_> {
    fn drop(&mut self) {
        if self.ended {
            return;
        }
        if let Ok(exec) = self.finish_capture() {
            // SAFETY: the graph was instantiated just now and nothing else holds it.
            let _ = unsafe { self.lane.library().api().euhedral_cuda_graph_destroy(exec) };
        }
    }
}

/// An instantiated graph, destroyed on drop.
///
/// A graph replays the device addresses it captured: every buffer it read or wrote must stay alive, and be the same
/// allocation, for as long as the graph can be launched. A freed buffer whose address a later allocation reuses would
/// be read or written by the replay. The engine's graph cache keys a graph on the [`AllocationId`](crate::AllocationId)
/// of each buffer it captured and drops the graph when one changes; nothing in this type checks it.
#[derive(Debug)]
pub struct GraphExec {
    library: Arc<CudaLibrary>,
    exec: u64,
}

impl GraphExec {
    /// Queues one replay on the submission's lane. The addresses the capture saw are the addresses it replays, so
    /// the buffers it read and wrote must be the same allocations.
    pub fn launch(&self, submission: &Submission<'_>) -> Result<(), CudaError> {
        // SAFETY: the graph is live and the stream is the submission's lane's live stream.
        let status = unsafe {
            self.library
                .api()
                .euhedral_cuda_graph_launch(self.exec, submission.stream())
        };
        check("euhedral_cuda_graph_launch", status)
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        // SAFETY: the graph is live and destroyed once. Destroying a graph with replays queued is allowed: CUDA
        // releases it when they finish.
        let _ = unsafe { self.library.api().euhedral_cuda_graph_destroy(self.exec) };
    }
}
