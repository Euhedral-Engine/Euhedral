//! Submissions: the guard that selects a lane's stream for the calling thread.

use std::cell::Cell;
use std::marker::PhantomData;

use euhedral_graph::Marker;

use crate::error::{CudaError, check};
use crate::lane::CudaLane;

thread_local! {
    /// Whether the thread's live submission is on a lane being captured.
    static CAPTURE_SELECTED: Cell<bool> = const { Cell::new(false) };
}

/// Whether the calling thread holds a submission on a lane that is being captured into a graph.
pub(crate) fn capture_selected() -> bool {
    CAPTURE_SELECTED.with(Cell::get)
}

/// How a submission launches.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SubmitMode {
    /// Programmatic dependent launch: a registered kernel starts while the previous kernel on the lane drains.
    pub pdl: bool,
    /// Row-exact execution (speculative verification): every operator whose kernel changes with the row count runs
    /// each row through its single-row path.
    pub row_exact: bool,
}

/// The launch state of the calling thread for one lane: its stream, and the PDL and row-exact selections.
///
/// The C library keeps that state per thread. A `Submission` selects it when created and clears it when dropped, and
/// every operation in [`ops`](crate::ops) takes one, so a launch cannot happen without a selected stream and a
/// selection cannot leak into the next frame that runs on the same worker. It is created and dropped inside one
/// frame's `execute`, never held across a poll, and is `!Send`: it names the thread whose state it selected.
///
/// ```compile_fail
/// fn assert_send<T: Send>() {}
/// assert_send::<euhedral_inference_cuda::Submission<'static>>();
/// ```
///
/// A lane has one submission at a time, and a thread has one at a time: [`Submission::begin`] fails with
/// `LaneBusy` when the lane is held, and with a status error when the thread already selected a stream. Nothing
/// waits for the lane to come free. That exclusivity is what keeps the order work lands in the stream the same as the
/// order the lane's markers were claimed in, which [`CudaLane`]'s completion word relies on.
#[derive(Debug)]
pub struct Submission<'l> {
    lane: &'l CudaLane,
    mode: SubmitMode,
    /// Whether dropping this submission gives the lane back (a capture's submissions borrow the capture's hold).
    owns_lane: bool,
    /// Raw pointers are neither `Send` nor `Sync`.
    thread_bound: PhantomData<*const ()>,
}

impl<'l> Submission<'l> {
    /// Takes `lane`, and selects its stream and `mode`, for the calling thread.
    pub fn begin(lane: &'l CudaLane, mode: SubmitMode) -> Result<Submission<'l>, CudaError> {
        lane.try_own()?;
        Self::select(lane, mode, true)
    }

    /// A submission on a lane the caller already holds (a graph capture's).
    pub(crate) fn borrowing(lane: &'l CudaLane, mode: SubmitMode) -> Result<Submission<'l>, CudaError> {
        Self::select(lane, mode, false)
    }

    fn select(lane: &'l CudaLane, mode: SubmitMode, owns_lane: bool) -> Result<Submission<'l>, CudaError> {
        let api = lane.library().api();
        // SAFETY: the stream is the lane's live stream.
        let status = unsafe { api.euhedral_cuda_stream_select(lane.stream()) };
        if let Err(error) = check("euhedral_cuda_stream_select", status) {
            if owns_lane {
                lane.disown();
            }
            return Err(error);
        }
        // From here the guard exists, so a failure below still clears the stream on drop.
        CAPTURE_SELECTED.with(|selected| selected.set(lane.is_capturing()));
        let submission = Submission {
            lane,
            mode,
            owns_lane,
            thread_bound: PhantomData,
        };
        // SAFETY: the calls take integers and set this thread's state.
        unsafe {
            if mode.pdl {
                api.euhedral_cuda_pdl_select(1);
            }
            if mode.row_exact {
                api.euhedral_cuda_row_exact_select(1);
            }
        }
        Ok(submission)
    }

    /// The lane this submission queues on.
    pub fn lane(&self) -> &'l CudaLane {
        self.lane
    }

    pub fn mode(&self) -> SubmitMode {
        self.mode
    }

    /// Orders the work queued after this call behind `marker` on the device. The host does not wait. A marker of this
    /// lane, or value 0, needs no wait: the lane runs in order. A marker the lane has not issued, or one older than its
    /// event ring (its event now belongs to a later marker), is an error. Not allowed while the lane is captured into a graph
    /// (a capture is one lane's work; the graph's launch is ordered by the submission that launches it).
    pub fn wait(&self, marker: Marker) -> Result<(), CudaError> {
        if self.lane.is_capturing() {
            return Err(CudaError::Capturing("a wait for another lane's marker"));
        }
        if marker.lane == self.lane.id() {
            return Ok(());
        }
        let ring = self.lane.library().ring(marker.lane)?;
        // `waitable` reads the lane's issued count before the wait is queued. A concurrent record on that lane can
        // reuse the event in between, which only widens the wait to the later work: conservative, never wrong.
        let Some(event) = ring.waitable(marker)? else {
            return Ok(());
        };
        // SAFETY: both handles are live: events are destroyed only when the library goes.
        let status = unsafe {
            self.lane
                .library()
                .api()
                .euhedral_cuda_stream_wait_event(self.lane.stream(), event)
        };
        check("euhedral_cuda_stream_wait_event", status)
    }

    /// Records a marker on the lane after the work queued so far, and ends the submission. Not allowed while the lane
    /// is captured into a graph; record the marker after the graph's launch.
    pub fn finish(self) -> Result<Marker, CudaError> {
        self.record()
    }

    /// Like [`Submission::finish`], and also queues a write of the marker's value to the lane's completion word, so
    /// [`DeviceQueue::completed`](euhedral_graph::DeviceQueue::completed) reaches the marker once the work retires.
    /// Do this on the markers the host polls, not on every marker.
    pub fn finish_signalled(self) -> Result<Marker, CudaError> {
        let marker = self.record()?;
        self.signal(marker.value)?;
        Ok(marker)
    }

    fn record(&self) -> Result<Marker, CudaError> {
        if self.lane.is_capturing() {
            return Err(CudaError::Capturing("a marker"));
        }
        let library = self.lane.library();
        let value = self.lane.next_value()?;
        let ring = library.ring(self.lane.id())?;
        let event = ring
            .event(value)
            .ok_or_else(|| CudaError::Argument("marker value 0".to_string()))?;
        // SAFETY: the event is one of this lane's live events and the stream is the lane's.
        let status = unsafe {
            library
                .api()
                .euhedral_cuda_completion_event_record(event, self.lane.stream())
        };
        check("euhedral_cuda_completion_event_record", status)?;
        Ok(Marker {
            lane: self.lane.id(),
            value,
        })
    }

    /// Queues a write of `value` (its low 32 bits) to the lane's host-mapped completion word, after the work queued
    /// so far. `DeviceQueue::completed` widens the word, so values must not move backwards or jump by 2^31 or more.
    ///
    /// Refused while the lane is captured into a graph: the graph would write the same value on every replay. Signal
    /// after the graph's launch.
    pub fn signal(&self, value: u64) -> Result<(), CudaError> {
        if self.lane.is_capturing() {
            return Err(CudaError::Capturing("a completion signal"));
        }
        // SAFETY: the address is the lane's mapped word; the submission has selected the lane's stream.
        let status = unsafe {
            self.lane
                .library()
                .api()
                .euhedral_cuda_write_host_flag(self.lane.word_device().get(), value as u32)
        };
        check("euhedral_cuda_write_host_flag", status)
    }

    pub(crate) fn stream(&self) -> u64 {
        self.lane.stream()
    }

    pub(crate) fn api(&self) -> &crate::sys::EuhedralCuda {
        self.lane.library().api()
    }
}

impl Drop for Submission<'_> {
    fn drop(&mut self) {
        let api = self.lane.library().api();
        // SAFETY: the calls clear this thread's state, which `begin` set.
        unsafe {
            if self.mode.pdl {
                api.euhedral_cuda_pdl_select(0);
            }
            if self.mode.row_exact {
                api.euhedral_cuda_row_exact_select(0);
            }
            api.euhedral_cuda_stream_clear();
        }
        CAPTURE_SELECTED.with(|selected| selected.set(false));
        if self.owns_lane {
            self.lane.disown();
        }
    }
}
