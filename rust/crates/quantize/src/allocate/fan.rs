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

/// Why a task has no answer: its frame was dropped before it ran, or its closure ended without sending (it panicked).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Lost;

/// The sending end of one task's answer. Whoever holds it sends in `Drop` when nothing was sent, so the waiter hears
/// of a task that panicked or was dropped unrun at once, never after its bound.
struct Slot<T> {
    done: Arc<Handoff<Result<T, Lost>>>,
    sent: bool,
}

impl<T> Slot<T> {
    fn new(done: &Arc<Handoff<Result<T, Lost>>>) -> Slot<T> {
        Slot {
            done: Arc::clone(done),
            sent: false,
        }
    }

    fn send(mut self, answer: T) {
        self.sent = true;
        self.done.send(Ok(answer));
    }
}

impl<T> Drop for Slot<T> {
    fn drop(&mut self) {
        if !self.sent {
            self.done.send(Err(Lost));
        }
    }
}

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
                let slot = Slot::new(&done);
                let frame = Task {
                    header: FrameHeader::new(hasher_api::mix((index as u64) | 1 << 63)),
                    work: Some(Box::new(move || slot.send(task()))),
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
                let answer = done
                    .wait(LONGEST_WAVE)
                    .map_err(|_| lost("a host task did not come back in time"))?
                    .map_err(|Lost| lost("a host task panicked or was dropped before it ran"))?;
                answers.push(answer);
            }
        }
    }
}

#[cfg(test)]
#[allow(clippy::unwrap_used, reason = "a test: a failed unwrap is the failure")]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use euhedral_inference_settings::load_for_tool;

    use super::*;
    use crate::rig::Rig;

    fn rig() -> Rig {
        Rig::start(&load_for_tool(&[], &|_| None).unwrap()).unwrap()
    }

    type Job<T> = Box<dyn FnOnce() -> T + Send>;

    #[test]
    fn answers_come_back_in_the_order_the_tasks_were_given() {
        let rig = rig();
        let fan = LakeFan::new(Arc::clone(&rig.host), 3);
        let tasks: Vec<Job<usize>> = (0..10usize).map(|i| Box::new(move || i * i) as Job<usize>).collect();
        assert_eq!(fan.map(tasks).unwrap(), (0..10usize).map(|i| i * i).collect::<Vec<_>>());
        assert_eq!(fan.map(Vec::<Job<usize>>::new()).unwrap(), Vec::<usize>::new());
        rig.stop();
    }

    #[test]
    fn no_more_than_the_sinks_are_in_flight_at_once() {
        let rig = rig();
        let (sinks, count) = (2, 9);
        let fan = LakeFan::new(Arc::clone(&rig.host), sinks);
        let finished = Arc::new(AtomicUsize::new(0));
        let tasks: Vec<Job<usize>> = (0..count)
            .map(|index| {
                let finished = Arc::clone(&finished);
                Box::new(move || {
                    // Tasks of one wave start only after the whole wave before it was joined.
                    let before = finished.load(Ordering::Acquire);
                    finished.fetch_add(1, Ordering::AcqRel);
                    assert!(
                        before >= index / sinks * sinks,
                        "task {index} started after only {before} finished"
                    );
                    before
                }) as Job<usize>
            })
            .collect();
        assert_eq!(fan.map(tasks).unwrap().len(), count);
        rig.stop();
    }

    #[test]
    fn a_panicking_task_fails_the_wave_at_once_naming_the_cause() {
        let rig = rig();
        let fan = LakeFan::new(Arc::clone(&rig.host), 2);
        let tasks: Vec<Job<usize>> = vec![
            Box::new(|| 1usize),
            Box::new(|| panic!("a task's own panic")),
            Box::new(|| 3),
        ];
        let refusal = fan.map(tasks).unwrap_err();
        assert!(
            matches!(&refusal, SearchFailure::Host(why) if why.contains("panicked")),
            "{refusal}"
        );
        // The lake still takes work after a task was lost.
        assert_eq!(fan.map(vec![Box::new(|| 7usize) as Job<usize>]).unwrap(), [7]);
        rig.stop();
    }

    #[test]
    fn a_task_dropped_before_it_ran_sends_its_loss() {
        let done = Handoff::<Result<usize, Lost>>::new();
        drop(Slot::new(&done));
        assert_eq!(
            done.wait(Duration::from_secs(10)).unwrap(),
            Err(Lost),
            "no loss was sent"
        );
        let done = Handoff::<Result<usize, Lost>>::new();
        Slot::new(&done).send(5);
        assert_eq!(
            done.wait(Duration::from_secs(10)).unwrap(),
            Ok(5),
            "the answer was replaced by a loss"
        );
    }

    #[test]
    fn a_lake_that_has_stopped_refuses_the_wave() {
        let rig = rig();
        let host = Arc::clone(&rig.host);
        rig.stop();
        let refusal = LakeFan::new(host, 2)
            .map(vec![Box::new(|| 1usize) as Job<usize>])
            .unwrap_err();
        assert!(matches!(refusal, SearchFailure::Host(_)), "{refusal}");
    }
}
