//! The engine's implementations of the cache's two interfaces: the frames go to the `stages` lake and the copies run on
//! one transfer lane of the runtime, retiring through its completions.

use std::cell::Cell;
use std::fmt::{self, Formatter};
use std::sync::Arc;

use euhedral_core::frames::Frame;
use euhedral_gpu::Lane;
use euhedral_gpu::cuda::CudaQueue;
use euhedral_graph::{LaneControl, Marker};
use euhedral_inference_cuda::DevicePtr;

use super::host::{CopyJob, CopyLane, Frames, SubmitFailure};
use crate::residency::loader::retry_or_offer;
use crate::runtime::Core;
use crate::stages::SubmitError;

/// The runtime's lake, completions and one transfer lane, as the cache's frames and copy lane.
#[expect(
    dead_code,
    reason = "the MoE layer that builds the cache over the runtime is a later change"
)]
pub(crate) struct RuntimeLanes {
    core: Arc<Core>,
    chain: usize,
}

impl fmt::Debug for RuntimeLanes {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimeLanes")
            .field("chain", &self.chain)
            .finish_non_exhaustive()
    }
}

#[expect(
    dead_code,
    reason = "the MoE layer that builds the cache over the runtime is a later change"
)]
impl RuntimeLanes {
    /// The lanes of `core`, copying on upload chain `chain`'s transfer lane. Refuses a chain the runtime has no lane for.
    pub(crate) fn new(core: Arc<Core>, chain: usize) -> Result<RuntimeLanes, String> {
        if chain >= core.transfer_lanes() {
            return Err(format!(
                "transfer chain {chain} was asked for: the runtime has {} transfer lanes",
                core.transfer_lanes()
            ));
        }
        Ok(RuntimeLanes { core, chain })
    }

    fn lane(&self) -> &Arc<CudaQueue> {
        self.core.transfer_lane(self.chain)
    }
}

impl Frames for RuntimeLanes {
    fn offer(&self, frame: Box<dyn Frame>) -> Result<(), Box<dyn Frame>> {
        self.core.stages.offer(frame)
    }

    fn after_marker(&self, marker: Marker, frame: Box<dyn Frame>) {
        self.core.completions().after_marker(marker, frame);
    }

    fn admit(&self) -> Result<Box<dyn Send>, String> {
        self.core
            .stages
            .admit()
            .map(|guard| Box::new(guard) as Box<dyn Send>)
            .map_err(|error| error.to_string())
    }
}

impl CopyLane for RuntimeLanes {
    fn submit(&self, job: &CopyJob) -> Result<Marker, SubmitFailure> {
        let Some((buffer, offset)) = job.staging.pinned() else {
            return Err(SubmitFailure::Refused(
                "the staging memory is not page-locked: a device copy cannot read it".to_owned(),
            ));
        };
        let lane = self.lane();
        for wait in &job.waits {
            if wait.lane != Lane::lane(&**lane) {
                lane.await_marker(*wait)
                    .map_err(|error| SubmitFailure::Refused(error.to_string()))?;
            }
        }
        // SAFETY: the device address is the slot of a pool the cache leased; the cache never leases a slot out for
        // anything else while this load is out.
        let destination = unsafe { DevicePtr::from_raw(job.device) };
        let copied = Cell::new(false);
        let queued = self.core.lane_table().submit_ops(lane, false, |ops| {
            // SAFETY: the job, which holds the staging slot's read lease, is kept by the caller until the marker this
            // submission returns has completed, so the bytes stay alive and unmodified until the copy has run.
            let result = unsafe { ops.copy_upload_to_device(destination, buffer, offset, job.bytes) };
            copied.set(result.is_ok());
            result
        });
        match queued {
            Ok(((), marker)) => Ok(marker),
            Err(SubmitError::Busy) => Err(SubmitFailure::Busy),
            // The work's own refusal (validation, or the copy call failing): the lane queued nothing of it.
            Err(SubmitError::Work(why)) => Err(SubmitFailure::Refused(why)),
            // The copy was queued and the lane failed after it (the marker, the claim): it may still run, with no
            // marker to say when. A failure before the copy was made queued nothing.
            Err(SubmitError::Lane(error)) if copied.get() => Err(SubmitFailure::Uncertain(error.to_string())),
            Err(SubmitError::Lane(error)) => Err(SubmitFailure::Refused(error.to_string())),
        }
    }

    fn mark(&self) -> Result<Marker, SubmitFailure> {
        match self
            .core
            .lane_table()
            .submit_ops(self.lane(), false, |_| Ok::<(), String>(()))
        {
            Ok(((), marker)) => Ok(marker),
            Err(SubmitError::Busy) => Err(SubmitFailure::Busy),
            Err(error) => Err(SubmitFailure::Refused(error.to_string())),
        }
    }

    fn when_room(&self, frame: Box<dyn Frame>) {
        let stages = Arc::clone(&self.core.stages);
        retry_or_offer(&**self.lane(), self.core.completions(), frame, |frame| {
            // The lake took none (it is closed): the frame runs here, so the load it carries ends rather than hangs.
            if let Err(mut refused) = stages.offer(frame)
                && let Err(error) = refused.execute()
            {
                log::error!("a retried expert copy failed on the calling thread: {error:?}");
            }
        });
    }
}
