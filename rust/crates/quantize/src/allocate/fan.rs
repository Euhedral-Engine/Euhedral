//! The search's host tasks (the members' fits and the RCO cells) as frames on the host lake.

use std::fmt::{self, Formatter};
use std::sync::Arc;
use std::time::Duration;

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_graph::Lake;
use euhedral_hashing::hasher_api;
use euhedral_inference_engine::Handoff;
use euhedral_inference_quantization::mixed::search::{Fan, SearchFailure};

/// The longest a wave of host tasks takes: a bound on the wait, not a rate.
const LONGEST_WAVE: Duration = Duration::from_secs(24 * 60 * 60);

/// A frame that runs one task and sends its answer.
struct Task {
    header: FrameHeader,
    work: Option<Box<dyn FnOnce() + Send>>,
}

impl Frame for Task {
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

impl fmt::Debug for Task {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Task").finish_non_exhaustive()
    }
}

/// Runs the tasks of the search on the host lake, `sinks` at a time, and joins them in the order they were given.
/// The calling thread, outside the lattice, waits for each wave.
pub(super) struct LakeFan {
    host: Arc<Lake>,
    sinks: usize,
}

impl LakeFan {
    pub(super) fn new(host: Arc<Lake>, sinks: usize) -> LakeFan {
        LakeFan {
            host,
            sinks: sinks.max(1),
        }
    }
}

impl Fan for LakeFan {
    fn map<T: Send + 'static>(&self, tasks: Vec<Box<dyn FnOnce() -> T + Send>>) -> Result<Vec<T>, SearchFailure> {
        let lost = |why: &str| SearchFailure::Host(why.to_owned());
        let _unit = self
            .host
            .admit()
            .map_err(|error| SearchFailure::Host(format!("the host lake would not take work: {error}")))?;
        let mut answers = Vec::with_capacity(tasks.len());
        let mut waiting = tasks.into_iter().enumerate();
        loop {
            let mut wave = Vec::with_capacity(self.sinks);
            for (index, task) in waiting.by_ref().take(self.sinks) {
                let done = Handoff::new();
                let sent = Arc::clone(&done);
                let frame = Task {
                    header: FrameHeader::new(hasher_api::mix((index as u64) | 1 << 63)),
                    work: Some(Box::new(move || sent.send(task()))),
                };
                if self.host.offer(Box::new(frame)).is_err() {
                    return Err(lost("the host lake refused a task"));
                }
                wave.push(done);
            }
            if wave.is_empty() {
                return Ok(answers);
            }
            for done in wave {
                answers.push(
                    done.wait(LONGEST_WAVE)
                        .map_err(|_| lost("a host task did not come back"))?,
                );
            }
        }
    }
}
