//! Retiring a quantum that ended before its stages did.
//!
//! A quantum retires when every lane it used has reached the quantum's latest marker on it, and a lane reaches only
//! the markers that were signalled (`Submission::finish_signalled`): the plan signals the ends of the lanes' chains and
//! no other stage, so a completed quantum's last marker on each lane is signalled. A quantum that stops early (its kill
//! switch was set, or a stage failed) can have submitted a stage that is not a chain's end and then never submitted
//! the stage that would have signalled behind it; its last marker on that lane is unsignalled, the lane's completion
//! word does not reach it, and the quantum never retires.
//!
//! Each quantum therefore records, per lane, the latest marker it submitted and the latest it signalled
//! ([`LaneEnds`]). When its stages are abandoned, or a stage fails or panics, the lanes whose last marker is
//! unsignalled are *settled*: an empty submission on the lane ends with a signalled marker, which the lane reaches
//! only after everything queued before it, the quantum's unsignalled marker included. The settling is a frame published
//! through the completions ([`Settler`]); a lane held by another submission answers `LaneBusy`, and the frame is
//! published again behind the progress of any lane, so nothing waits. The graph's retirement boundary needs no change:
//! it names the quantum's own last marker, and the settling marker is a later value on the same lane.
//!
//! Quanta that run to their end signal as before and settle nothing.

use std::fmt;
use std::panic::{AssertUnwindSafe, catch_unwind, resume_unwind};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

use euhedral_graph::{GraphRuntime, LaneId, LanePool, Marker, Ready, Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{CudaError, CudaLane, Submission, SubmitMode};

use super::{LaneTable, Quantum};
use crate::frames::{self, Kind};

/// What settling a lane found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Settled {
    /// The lane's last marker of the quantum is signalled already.
    Clean,
    /// The lane ended with a signalled marker.
    Signalled,
    /// Another submission holds the lane: try again.
    Busy,
}

/// The latest marker a quantum submitted on each lane, and the latest it signalled.
#[derive(Debug)]
pub struct LaneEnds {
    lanes: Box<[LaneEnd]>,
    /// Set by the first abandoned stage, which settles for all of them.
    abandoned: AtomicBool,
    /// Set by the first stage that failed.
    failed: AtomicBool,
}

#[derive(Debug, Default)]
struct LaneEnd {
    submitted: AtomicU64,
    signalled: AtomicU64,
}

impl LaneEnds {
    /// The record of a quantum over `lanes` lanes.
    pub fn new(lanes: usize) -> Self {
        Self {
            lanes: (0..lanes).map(|_| LaneEnd::default()).collect(),
            abandoned: AtomicBool::new(false),
            failed: AtomicBool::new(false),
        }
    }

    /// The quantum submitted `marker`, signalled or not.
    pub fn record(&self, marker: Marker, signalled: bool) {
        let Some(lane) = self.lanes.get(usize::from(marker.lane.0)) else {
            return;
        };
        lane.submitted.fetch_max(marker.value, Ordering::AcqRel);
        if signalled {
            lane.signalled.fetch_max(marker.value, Ordering::AcqRel);
        }
    }

    /// Whether the quantum's latest marker on `lane` is unsignalled: the lane's completion word will not reach it until
    /// a signalled marker follows.
    pub fn unsignalled(&self, lane: LaneId) -> bool {
        self.lanes
            .get(usize::from(lane.0))
            .is_some_and(|lane| lane.submitted.load(Ordering::Acquire) > lane.signalled.load(Ordering::Acquire))
    }

    /// The lanes whose latest marker of the quantum is unsignalled.
    pub fn unsignalled_lanes(&self) -> Vec<LaneId> {
        (0..self.lanes.len())
            .filter_map(|lane| u16::try_from(lane).ok().map(LaneId))
            .filter(|&lane| self.unsignalled(lane))
            .collect()
    }

    /// Whether this is the first abandoned stage of the quantum (the one that settles).
    fn first_abandoned(&self) -> bool {
        !self.abandoned.swap(true, Ordering::AcqRel)
    }

    /// Whether this is the first failed stage of the quantum.
    fn first_failed(&self) -> bool {
        !self.failed.swap(true, Ordering::AcqRel)
    }

    /// Ends `lane` with a signalled marker if the quantum's latest marker on it is not signalled: an empty submission,
    /// queued behind the lane's work, that writes its marker to the completion word. Never waits: a held lane answers
    /// [`Settled::Busy`].
    pub fn settle(&self, lane: &CudaLane) -> Result<Settled, CudaError> {
        let id = lane.id();
        if !self.unsignalled(id) {
            return Ok(Settled::Clean);
        }
        let submission = match Submission::begin(lane, SubmitMode::default()) {
            Ok(submission) => submission,
            Err(CudaError::LaneBusy(_)) => return Ok(Settled::Busy),
            Err(error) => return Err(error),
        };
        // The stage that held the lane before may have signalled in the meantime.
        if !self.unsignalled(id) {
            return Ok(Settled::Clean);
        }
        let marker = submission.finish_signalled()?;
        self.record(marker, true);
        Ok(Settled::Signalled)
    }
}

/// Settles the lanes of quanta that ended early, on the runtime their graphs run on.
pub struct Settler {
    runtime: GraphRuntime,
}

impl fmt::Debug for Settler {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Settler").finish_non_exhaustive()
    }
}

impl Settler {
    /// A settler publishing through `runtime`'s completions.
    pub fn new(runtime: &GraphRuntime) -> Arc<Self> {
        Arc::new(Self {
            runtime: runtime.clone(),
        })
    }

    /// Settles every lane of `quantum` whose latest marker is unsignalled, each in a frame of its own.
    fn settle(self: &Arc<Self>, quantum: &Quantum) {
        for lane in quantum.ends.unsignalled_lanes() {
            self.publish(lane, Arc::clone(&quantum.ends), quantum.lanes.clone(), Box::new(Now));
        }
    }

    fn publish(self: &Arc<Self>, lane: LaneId, ends: Arc<LaneEnds>, table: LaneTable, ready: Box<dyn Ready>) {
        let settler = Arc::clone(self);
        let frame = frames::step(Kind::Settle, usize::from(lane.0), move || {
            settler.attempt(lane, ends, table);
        });
        self.runtime.completions.after(ready, frame);
    }

    /// One try at settling `lane`; a held lane is tried again once any lane has made progress.
    fn attempt(self: &Arc<Self>, lane: LaneId, ends: Arc<LaneEnds>, table: LaneTable) {
        let Some(device) = table.get(lane) else {
            log::error!(
                "lane {} of a quantum that ended early is not in its lane table; the quantum cannot retire",
                lane.0
            );
            return;
        };
        match ends.settle(device) {
            Ok(Settled::Busy) => {
                let progress = Progress::new(&self.runtime);
                self.publish(lane, ends, table, Box::new(progress));
            }
            Ok(Settled::Clean | Settled::Signalled) => {}
            Err(error) => log::error!(
                "lane {} of a quantum that ended early could not be settled, so the quantum cannot retire: {error}",
                lane.0
            ),
        }
    }
}

/// Ready at once: the frame goes out at the next poll.
struct Now;

impl Ready for Now {
    fn is_ready(&mut self) -> bool {
        true
    }
}

/// Ready once any lane of the pool completed more than it had, or the runtime's retry backstop passed.
struct Progress {
    lanes: Arc<LanePool>,
    seen: Vec<u64>,
    deadline: Instant,
}

impl Progress {
    fn new(runtime: &GraphRuntime) -> Self {
        let seen = (0..runtime.lanes.len())
            .filter_map(|lane| u16::try_from(lane).ok())
            .map(|lane| runtime.lanes.lane(LaneId(lane)).completed())
            .collect();
        Self {
            lanes: Arc::clone(&runtime.lanes),
            seen,
            deadline: Instant::now() + runtime.retry_backstop,
        }
    }
}

impl Ready for Progress {
    fn is_ready(&mut self) -> bool {
        let progressed = self.seen.iter().enumerate().any(|(lane, &seen)| {
            u16::try_from(lane).is_ok_and(|lane| self.lanes.lane(LaneId(lane)).completed() > seen)
        });
        progressed || Instant::now() >= self.deadline
    }
}

/// A stage of a plan's graph that settles its quantum's lanes when the quantum ends early: when the stage is abandoned
/// (the quantum was killed or a stage failed before it ran), and when it fails or panics itself. Otherwise it is the
/// stage it wraps.
pub struct Settling {
    stage: Arc<dyn Stage<Quantum>>,
    settler: Arc<Settler>,
}

impl Settling {
    /// `stage`, settling through `settler`.
    pub fn new(stage: Arc<dyn Stage<Quantum>>, settler: Arc<Settler>) -> Self {
        Self { stage, settler }
    }
}

impl fmt::Debug for Settling {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Settling").finish_non_exhaustive()
    }
}

impl Stage<Quantum> for Settling {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        match catch_unwind(AssertUnwindSafe(|| self.stage.run(quantum, cx))) {
            // A retry runs again, or is abandoned.
            Ok(Err(StageError::Retry)) => Err(StageError::Retry),
            Ok(Err(error)) => {
                // Stages behind this one are abandoned and settle; this covers a failure with none behind it.
                if quantum.ends.first_failed() {
                    self.settler.settle(quantum);
                }
                Err(error)
            }
            Ok(result) => result,
            Err(panic) => {
                if quantum.ends.first_failed() {
                    self.settler.settle(quantum);
                }
                resume_unwind(panic)
            }
        }
    }

    fn abandoned(&self, quantum: &Quantum) {
        self.stage.abandoned(quantum);
        if quantum.ends.first_abandoned() {
            self.settler.settle(quantum);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn marker(lane: u16, value: u64) -> Marker {
        Marker {
            lane: LaneId(lane),
            value,
        }
    }

    #[test]
    fn a_lane_with_no_marker_of_the_quantum_is_not_unsignalled() {
        let ends = LaneEnds::new(3);
        assert!(ends.unsignalled_lanes().is_empty());
    }

    #[test]
    fn a_lane_whose_latest_marker_is_signalled_is_settled_and_one_whose_latest_is_not_is_not() {
        let ends = LaneEnds::new(3);
        ends.record(marker(0, 4), false);
        ends.record(marker(0, 5), true);
        ends.record(marker(1, 7), true);
        ends.record(marker(1, 9), false);
        assert!(!ends.unsignalled(LaneId(0)));
        assert!(ends.unsignalled(LaneId(1)));
        assert!(!ends.unsignalled(LaneId(2)));
        assert_eq!(ends.unsignalled_lanes(), vec![LaneId(1)]);
    }

    #[test]
    fn a_signalled_marker_after_the_unsignalled_one_covers_it() {
        let ends = LaneEnds::new(1);
        ends.record(marker(0, 3), false);
        assert!(ends.unsignalled(LaneId(0)));
        // The settling marker, or the end of a chain: a later value, signalled.
        ends.record(marker(0, 4), true);
        assert!(!ends.unsignalled(LaneId(0)));
    }

    #[test]
    fn markers_recorded_out_of_order_keep_the_latest_of_each_kind() {
        let ends = LaneEnds::new(1);
        // Two stages of one lane finish their records in either order.
        ends.record(marker(0, 6), true);
        ends.record(marker(0, 5), false);
        assert!(!ends.unsignalled(LaneId(0)), "the signalled marker is the later");
        ends.record(marker(0, 8), false);
        ends.record(marker(0, 7), true);
        assert!(ends.unsignalled(LaneId(0)), "the unsignalled marker is the later");
    }

    #[test]
    fn a_marker_of_a_lane_the_quantum_has_no_record_for_is_ignored() {
        let ends = LaneEnds::new(1);
        ends.record(marker(5, 2), false);
        assert!(ends.unsignalled_lanes().is_empty());
    }

    #[test]
    fn the_first_abandoned_stage_settles_and_the_rest_do_not() {
        let ends = LaneEnds::new(1);
        assert!(ends.first_abandoned());
        assert!(!ends.first_abandoned());
        assert!(ends.first_failed(), "a failure is counted apart from the abandonments");
        assert!(!ends.first_failed());
    }
}
