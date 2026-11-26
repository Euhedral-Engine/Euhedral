//! The search's host tasks (the members' fits and the RCO cells) as frames on the host lake.

use std::fmt::{self, Formatter};
use std::sync::Arc;
use std::time::Duration;

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_graph::Lake;
use euhedral_hashing::hasher_api;
use euhedral_inference_engine::Handoff;
use euhedral_inference_quantization::mixed::search::{Fan, SearchFailure};

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
/// The calling thread, outside the lattice, waits for each wave, at most `longest_wave` for a task.
pub(super) struct LakeFan {
    host: Arc<Lake>,
    sinks: usize,
    longest_wave: Duration,
}

impl LakeFan {
    /// Refuses no sinks and no time to wait: either would fail every wave.
    pub(super) fn new(host: Arc<Lake>, sinks: usize, longest_wave: Duration) -> Result<LakeFan, SearchFailure> {
        if sinks == 0 {
            return Err(SearchFailure::Host(
                "a host fan needs at least one sink, not 0".to_owned(),
            ));
        }
        if longest_wave.is_zero() {
            return Err(SearchFailure::Host(
                "a host fan needs a wave bound above 0 seconds, not 0".to_owned(),
            ));
        }
        Ok(LakeFan {
            host,
            sinks,
            longest_wave,
        })
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
                    .wait(self.longest_wave)
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
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::{Condvar, Mutex};
    use std::thread;

    use euhedral_inference_settings::load_for_tool;

    use super::*;
    use crate::rig::Rig;

    fn rig() -> Rig {
        Rig::start(&load_for_tool(&[], &|_| None).unwrap()).unwrap()
    }

    /// The deadline for a step that must happen; a run that reaches it is hung, and its failure says which step.
    const HANG: Duration = Duration::from_secs(30);
    /// The wave bound of the test that must hear of a lost task at once: far under `HANG`.
    const SHORT_WAVE: Duration = Duration::from_secs(5);

    type Job<T> = Box<dyn FnOnce() -> T + Send>;

    #[test]
    fn answers_come_back_in_the_order_the_tasks_were_given() {
        let rig = rig();
        let fan = LakeFan::new(Arc::clone(&rig.host), 3, HANG).unwrap();
        let tasks: Vec<Job<usize>> = (0..10usize).map(|i| Box::new(move || i * i) as Job<usize>).collect();
        assert_eq!(fan.map(tasks).unwrap(), (0..10usize).map(|i| i * i).collect::<Vec<_>>());
        assert_eq!(fan.map(Vec::<Job<usize>>::new()).unwrap(), Vec::<usize>::new());
        rig.stop();
    }

    /// A gate the tasks wait at until the test opens it; the deadline is only for a hang, with a message that says
    /// what never happened.
    struct Gate {
        state: Mutex<(usize, bool)>,
        changed: Condvar,
    }

    impl Gate {
        fn new() -> Arc<Gate> {
            Arc::new(Gate {
                state: Mutex::new((0, false)),
                changed: Condvar::new(),
            })
        }

        /// A task arrives and waits for the gate to open.
        fn arrive_and_wait(&self) {
            let mut state = self.state.lock().unwrap();
            state.0 += 1;
            self.changed.notify_all();
            let (state, timeout) = self.changed.wait_timeout_while(state, HANG, |state| !state.1).unwrap();
            assert!(!timeout.timed_out() && state.1, "the gate was never opened");
        }

        /// Waits until `count` tasks have arrived.
        fn await_arrivals(&self, count: usize) {
            let state = self.state.lock().unwrap();
            let (state, timeout) = self
                .changed
                .wait_timeout_while(state, HANG, |state| state.0 < count)
                .unwrap();
            assert!(
                !timeout.timed_out() && state.0 >= count,
                "only {} of {count} tasks of the first wave started",
                state.0
            );
        }

        fn open(&self) {
            self.state.lock().unwrap().1 = true;
            self.changed.notify_all();
        }
    }

    #[test]
    fn a_wave_is_not_offered_before_the_one_before_it_is_joined() {
        let mut settings = load_for_tool(&[], &|_| None).unwrap();
        // Workers enough that a task offered too early would start while the first wave still holds the gate.
        settings.host.task_parallelism = 3;
        let rig = Rig::start(&settings).unwrap();
        let (sinks, count) = (1, 4);
        let fan = Arc::new(LakeFan::new(Arc::clone(&rig.host), sinks, HANG).unwrap());
        let gate = Gate::new();
        let opened = Arc::new(AtomicBool::new(false));
        let tasks: Vec<Job<bool>> = (0..count)
            .map(|index| {
                let (gate, opened) = (Arc::clone(&gate), Arc::clone(&opened));
                Box::new(move || {
                    if index < sinks {
                        gate.arrive_and_wait();
                    }
                    // Only the first wave runs before the gate opens.
                    index < sinks || opened.load(Ordering::Acquire)
                }) as Job<bool>
            })
            .collect();
        let running = {
            let fan = Arc::clone(&fan);
            thread::spawn(move || fan.map(tasks))
        };
        gate.await_arrivals(sinks);
        opened.store(true, Ordering::Release);
        gate.open();
        let answers = running.join().unwrap().unwrap();
        assert_eq!(
            answers,
            vec![true; count],
            "a task outside the first wave started before the gate opened"
        );
        rig.stop();
    }

    #[test]
    fn a_fan_without_sinks_or_without_time_is_refused() {
        let rig = rig();
        for (sinks, wave, named) in [(0, HANG, "sink"), (1, Duration::ZERO, "wave bound")] {
            let refusal = LakeFan::new(Arc::clone(&rig.host), sinks, wave).err().unwrap();
            assert!(
                matches!(&refusal, SearchFailure::Host(why) if why.contains(named)),
                "{refusal}"
            );
        }
        rig.stop();
    }

    #[test]
    fn a_panicking_task_fails_the_wave_at_once_naming_the_cause() {
        let rig = rig();
        let fan = LakeFan::new(Arc::clone(&rig.host), 2, SHORT_WAVE).unwrap();
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
        assert_eq!(done.wait(HANG).unwrap(), Err(Lost), "no loss was sent");
        let done = Handoff::<Result<usize, Lost>>::new();
        Slot::new(&done).send(5);
        assert_eq!(done.wait(HANG).unwrap(), Ok(5), "the answer was replaced by a loss");
    }

    #[test]
    fn a_lake_that_has_stopped_refuses_the_wave() {
        let rig = rig();
        let host = Arc::clone(&rig.host);
        rig.stop();
        let refusal = LakeFan::new(host, 2, HANG)
            .unwrap()
            .map(vec![Box::new(|| 1usize) as Job<usize>])
            .unwrap_err();
        assert!(matches!(refusal, SearchFailure::Host(_)), "{refusal}");
    }
}
