//! A layer's pass over every batch of sequences as lattice frames, each frame's finalization publishing the next:
//!
//! - `Start` runs the batch's head on the main lane and publishes one `Sequence` frame per sequence;
//! - `Sequence` runs one sequence on its slot's own lane, behind the head: the recurrences or attention of different
//!   sequences overlap on the device; the last to arrive publishes `Finish`;
//! - `Finish` runs the batch's tail on the main lane, behind every slot, and publishes the next batch's `Start`, or
//!   hands the end of the pass to the lanes' completions behind the pass's last marker.
//!
//! No frame waits: a frame only submits device work, ordered across lanes by markers, and the completion source emits
//! the last frame when the device is through. What a frame owns travels with it: the main lane's state is taken from
//! its cell by `Start` and `Finish` (never both at once), a slot by the one `Sequence` that runs on it. Nothing is
//! locked.
//!
//! [`Scheduler`] is that machine over [`Stages`], the device-facing calls; [`device::ForwardPipeline`] is it over the
//! layer forward and the lanes.
//!
//! A failure ends the pass behind a fence: the error is delivered only after the device has run everything the pass
//! enqueued, on the main lane and every slot lane, because that work still reads the caller's buffers.

pub mod device;

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use euhedral_core::frames::{Frame, FrameError, FrameHeader};
use euhedral_graph::{Lake, UnitGuard};
use euhedral_hashing::hasher_api;
use euhedral_inference_engine::Handoff;

use super::ForwardError;

/// What a stage call came to: its work was queued, or the lane's marker ring was full and nothing was queued, so the
/// call is made again.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Attempt<T> {
    Ready(T),
    Busy,
}

/// What a pass ends with, called once from a frame of the lattice; it must not wait.
pub type Done = Box<dyn FnOnce(Result<(), ForwardError>) + Send>;

/// The device-facing calls of a pass.
pub trait Stages: Send + Sync + 'static {
    /// What `Start` and `Finish` own: the main lane's state.
    type Main: Send + 'static;
    /// What one sequence owns while it runs.
    type Slot: Send + 'static;
    /// A point on a lane that later work on another lane waits for.
    type Mark: Copy + Send + Sync + 'static;
    /// What every frame of one pass reads.
    type Pass: Send + Sync + 'static;

    /// The head of the batch `first..first + count`, queued on the main lane. Returns the mark after it.
    fn head(
        &self,
        pass: &Self::Pass,
        main: &mut Self::Main,
        first: u32,
        count: u32,
    ) -> Result<Attempt<Self::Mark>, ForwardError>;

    /// Sequence `index` of the batch on its slot's lane, behind `after`. Returns the mark after it.
    fn sequence(
        &self,
        pass: &Self::Pass,
        slot: &mut Self::Slot,
        index: u32,
        after: Self::Mark,
    ) -> Result<Attempt<Self::Mark>, ForwardError>;

    /// The tail of the batch on the main lane, behind every sequence's mark. With `last`, the mark after the pass's
    /// final work.
    fn tail(
        &self,
        pass: &Self::Pass,
        main: &mut Self::Main,
        first: u32,
        count: u32,
        last: bool,
        marks: &[Self::Mark],
    ) -> Result<Attempt<Option<Self::Mark>>, ForwardError>;

    /// A mark after everything queued so far on the main lane and on every slot lane.
    fn fence(&self, main: &mut Self::Main) -> Result<Attempt<Self::Mark>, ForwardError>;

    /// Emits `then` once the lanes have reached `mark`.
    fn after(&self, mark: Self::Mark, then: Box<dyn Frame>) -> Result<(), ForwardError>;

    /// The device is through with the pass: publish what it appended.
    fn retire(&self, slots: &mut [Self::Slot]);
}

/// What a scheduler gives back: the stages, the main lane's state and the slots.
pub type Parts<S> = (S, <S as Stages>::Main, Vec<<S as Stages>::Slot>);

type Resident<S> = (<S as Stages>::Main, Vec<<S as Stages>::Slot>);

/// Runs passes over a [`Stages`]: the lanes' state rests in the scheduler between passes.
pub struct Scheduler<S: Stages> {
    stages: Arc<S>,
    lake: Arc<Lake>,
    home: Arc<Handoff<Resident<S>>>,
    seeds: Arc<AtomicU64>,
}

impl<S: Stages> Debug for Scheduler<S> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Scheduler").finish_non_exhaustive()
    }
}

impl<S: Stages> Scheduler<S> {
    /// A scheduler of `stages` whose frames go to `lake`, with the main lane's state and one slot per sequence of a
    /// batch.
    pub fn new(stages: S, lake: Arc<Lake>, main: S::Main, slots: Vec<S::Slot>) -> Self {
        let home = Handoff::new();
        home.send((main, slots));
        Scheduler {
            stages: Arc::new(stages),
            lake,
            home,
            seeds: Arc::new(AtomicU64::new(1)),
        }
    }

    /// The stages.
    pub fn stages(&self) -> &S {
        &self.stages
    }

    /// Gives back the stages and the lanes' state once no pass is running, or the scheduler itself while one is (or
    /// after one lost the state). A pass is over for this call when its `done` was called.
    pub fn into_parts(self) -> Result<Parts<S>, Self> {
        match Arc::try_unwrap(self.stages) {
            Ok(stages) => match self.home.take() {
                Some((main, slots)) => Ok((stages, main, slots)),
                None => Err(Scheduler {
                    stages: Arc::new(stages),
                    lake: self.lake,
                    home: self.home,
                    seeds: self.seeds,
                }),
            },
            Err(stages) => Err(Scheduler {
                stages,
                lake: self.lake,
                home: self.home,
                seeds: self.seeds,
            }),
        }
    }

    /// Starts a pass of `sequences` sequences, `batch` at a time. `done` is called once, from a frame, when the pass
    /// has ended and the device is through with it; `Err` here started nothing and `done` is dropped. One pass runs
    /// at a time.
    pub fn run(&self, shared: S::Pass, sequences: u32, batch: u32, done: Done) -> Result<(), ForwardError> {
        if sequences == 0 || batch == 0 {
            return Err(ForwardError::Invalid("a pass of no sequences".to_owned()));
        }
        let Some((main, slots)) = self.home.take() else {
            return Err(ForwardError::Invalid("a pass is already running".to_owned()));
        };
        if batch as usize > slots.len() {
            self.home.send((main, slots));
            return Err(ForwardError::Invalid("more sequences per batch than slots".to_owned()));
        }
        let unit = match self.lake.admit() {
            Ok(unit) => unit,
            Err(error) => {
                self.home.send((main, slots));
                return Err(ForwardError::Invalid(format!(
                    "the lake would not take the pass: {error}"
                )));
            }
        };
        let pass = Arc::new(Pass {
            stages: Arc::clone(&self.stages),
            lake: Arc::clone(&self.lake),
            home: Arc::clone(&self.home),
            seeds: Arc::clone(&self.seeds),
            shared,
            sequences,
            batch,
            main: Handoff::empty(),
            slots: slots
                .into_iter()
                .map(|slot| {
                    let cell = Handoff::empty();
                    cell.send(slot);
                    cell
                })
                .collect(),
            failed: AtomicBool::new(false),
            failure: Handoff::empty(),
            unit: Handoff::empty(),
            ending: Ending {
                result: Handoff::empty(),
                done: Handoff::empty(),
            },
        });
        pass.main.send(main);
        pass.unit.send(unit);
        pass.ending.done.send(done);
        let first = Pass::start(&pass, 0);
        if let Err(frame) = pass.lake.offer(first) {
            drop(frame);
            // Nothing ran: the caller is told by the return, not by `done`.
            drop(pass.ending.done.take());
            pass.give_back();
            return Err(ForwardError::Invalid(
                "the lake is closed: the pass was not started".to_owned(),
            ));
        }
        Ok(())
    }
}

/// What one batch shares: the mark the sequences wait for, and what they leave for the finish.
struct Join<S: Stages> {
    first: u32,
    count: u32,
    start: S::Mark,
    remaining: AtomicU32,
    marks: Vec<OnceLock<S::Mark>>,
}

struct Pass<S: Stages> {
    stages: Arc<S>,
    lake: Arc<Lake>,
    home: Arc<Handoff<Resident<S>>>,
    seeds: Arc<AtomicU64>,
    shared: S::Pass,
    sequences: u32,
    batch: u32,
    main: Handoff<S::Main>,
    slots: Vec<Handoff<S::Slot>>,
    failed: AtomicBool,
    failure: Handoff<ForwardError>,
    unit: Handoff<UnitGuard>,
    /// Last: it tells the caller when the pass's last frame is gone, after the stages and the lake were let go.
    ending: Ending,
}

/// Tells the caller how the pass ended, when nothing of the pass is left. A pass whose frames were dropped before it
/// ended (the lake shut down) still tells its caller.
struct Ending {
    result: Handoff<Result<(), ForwardError>>,
    done: Handoff<Done>,
}

impl Drop for Ending {
    fn drop(&mut self) {
        if let Some(done) = self.done.take() {
            let result = self.result.take().unwrap_or_else(|| {
                Err(ForwardError::Invalid(
                    "the lake closed before the pass finished".to_owned(),
                ))
            });
            done(result);
        }
    }
}

/// A frame of a pass: `work` runs in `execute` and returns what the finalization does.
struct Step {
    header: FrameHeader,
    work: Option<Box<dyn FnOnce() -> Then + Send>>,
    then: Option<Then>,
}

type Then = Box<dyn FnOnce() + Send>;

impl Debug for Step {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Step").finish_non_exhaustive()
    }
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
            self.then = Some(work());
        }
        Ok(())
    }

    fn do_finally(mut self: Box<Self>) {
        if let Some(then) = self.then.take() {
            then();
        }
    }
}

impl<S: Stages> Pass<S> {
    fn frame(&self, work: impl FnOnce() -> Then + Send + 'static) -> Box<dyn Frame> {
        let mut header = FrameHeader::new(hasher_api::mix(0x0f0a_5e11 | 1 << 63));
        header.randomize_hash(self.seeds.fetch_add(1, Ordering::Relaxed));
        Box::new(Step {
            header,
            work: Some(Box::new(work)),
            then: None,
        })
    }

    fn count(&self, first: u32) -> u32 {
        self.batch.min(self.sequences - first)
    }

    /// The first failure wins.
    fn fail(&self, error: ForwardError) {
        if !self.failed.swap(true, Ordering::AcqRel) {
            self.failure.send(error);
        }
    }

    fn has_failed(&self) -> bool {
        self.failed.load(Ordering::Acquire)
    }

    /// Offers `frame`; a refusal (the lake closed) ends the pass with `refused`.
    fn publish(pass: &Arc<Self>, frame: Box<dyn Frame>, refused: impl FnOnce(&Arc<Self>)) {
        if pass.lake.offer(frame).is_err() {
            pass.fail(ForwardError::Invalid(
                "the lake is closed: a frame of the pass was refused".to_owned(),
            ));
            refused(pass);
        }
    }

    fn start(pass: &Arc<Self>, first: u32) -> Box<dyn Frame> {
        let held = Arc::clone(pass);
        pass.frame(move || {
            let pass = held;
            let Some(mut main) = pass.main.take() else {
                return Box::new(|| {});
            };
            if pass.has_failed() {
                return Box::new(move || {
                    pass.main.send(main);
                    Self::end_failed(&pass);
                });
            }
            let count = pass.count(first);
            match pass.stages.head(&pass.shared, &mut main, first, count) {
                Ok(Attempt::Ready(mark)) => Box::new(move || {
                    pass.main.send(main);
                    Self::fan_out(&pass, first, count, mark);
                }),
                Ok(Attempt::Busy) => Box::new(move || {
                    pass.main.send(main);
                    let frame = Self::start(&pass, first);
                    Self::publish(&pass, frame, Self::end_failed);
                }),
                Err(error) => Box::new(move || {
                    pass.fail(error);
                    pass.main.send(main);
                    Self::end_failed(&pass);
                }),
            }
        })
    }

    /// Publishes the batch's sequences. A refused offer counts as that sequence's arrival, failed.
    fn fan_out(pass: &Arc<Self>, first: u32, count: u32, start: S::Mark) {
        let join = Arc::new(Join::<S> {
            first,
            count,
            start,
            remaining: AtomicU32::new(count),
            marks: (0..count).map(|_| OnceLock::new()).collect(),
        });
        for index in 0..count {
            let frame = Self::sequence(pass, &join, index);
            if pass.lake.offer(frame).is_err() {
                pass.fail(ForwardError::Invalid(
                    "the lake is closed: a frame of the pass was refused".to_owned(),
                ));
                Self::arrive(pass, &join);
            }
        }
    }

    fn sequence(pass: &Arc<Self>, join: &Arc<Join<S>>, index: u32) -> Box<dyn Frame> {
        let (pass, join) = (Arc::clone(pass), Arc::clone(join));
        let held = Arc::clone(&pass);
        held.frame(move || {
            if pass.has_failed() {
                return Box::new(move || Self::arrive(&pass, &join));
            }
            let Some(mut slot) = pass.slots[index as usize].take() else {
                pass.fail(ForwardError::Invalid(format!("slot {index} is missing")));
                return Box::new(move || Self::arrive(&pass, &join));
            };
            let outcome = pass.stages.sequence(&pass.shared, &mut slot, index, join.start);
            pass.slots[index as usize].send(slot);
            match outcome {
                Ok(Attempt::Ready(mark)) => {
                    let _ = join.marks[index as usize].set(mark);
                    Box::new(move || Self::arrive(&pass, &join))
                }
                Ok(Attempt::Busy) => Box::new(move || {
                    let frame = Self::sequence(&pass, &join, index);
                    if pass.lake.offer(frame).is_err() {
                        pass.fail(ForwardError::Invalid(
                            "the lake is closed: a frame of the pass was refused".to_owned(),
                        ));
                        Self::arrive(&pass, &join);
                    }
                }),
                Err(error) => Box::new(move || {
                    pass.fail(error);
                    Self::arrive(&pass, &join);
                }),
            }
        })
    }

    /// A sequence is through; the last publishes the finish, or ends a failed pass.
    fn arrive(pass: &Arc<Self>, join: &Arc<Join<S>>) {
        if join.remaining.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        if pass.has_failed() {
            return Self::end_failed(pass);
        }
        let frame = Self::finish(pass, join);
        Self::publish(pass, frame, Self::end_failed);
    }

    fn finish(pass: &Arc<Self>, join: &Arc<Join<S>>) -> Box<dyn Frame> {
        let (pass, join) = (Arc::clone(pass), Arc::clone(join));
        let held = Arc::clone(&pass);
        held.frame(move || {
            let Some(mut main) = pass.main.take() else {
                return Box::new(|| {});
            };
            let marks: Vec<S::Mark> = join.marks.iter().filter_map(|mark| mark.get().copied()).collect();
            let following = join.first + join.count;
            let last = following >= pass.sequences;
            if marks.len() != join.count as usize {
                pass.fail(ForwardError::Invalid("a sequence ended without its mark".to_owned()));
                return Box::new(move || {
                    pass.main.send(main);
                    Self::end_failed(&pass);
                });
            }
            match pass
                .stages
                .tail(&pass.shared, &mut main, join.first, join.count, last, &marks)
            {
                Ok(Attempt::Ready(mark)) => Box::new(move || {
                    pass.main.send(main);
                    if !last {
                        let frame = Self::start(&pass, following);
                        return Self::publish(&pass, frame, Self::end_failed);
                    }
                    match mark {
                        Some(mark) => Self::end_ok(&pass, mark),
                        None => {
                            pass.fail(ForwardError::Invalid("the last tail gave no mark".to_owned()));
                            Self::end_failed(&pass);
                        }
                    }
                }),
                Ok(Attempt::Busy) => Box::new(move || {
                    pass.main.send(main);
                    let frame = Self::finish(&pass, &join);
                    Self::publish(&pass, frame, Self::end_failed);
                }),
                Err(error) => Box::new(move || {
                    pass.fail(error);
                    pass.main.send(main);
                    Self::end_failed(&pass);
                }),
            }
        })
    }

    /// The pass queued everything: its end is the frame the lanes' completions emit behind `mark`.
    fn end_ok(pass: &Arc<Self>, mark: S::Mark) {
        let frame = Self::last(pass);
        if let Err(error) = pass.stages.after(mark, frame) {
            // The frame went with the refusal; the device may still be running the pass: fence, then end.
            pass.fail(error);
            Self::end_failed(pass);
        }
    }

    /// Ends a failed pass behind a fence over every lane it used: the error reaches the caller only once the device
    /// no longer reads the caller's buffers.
    fn end_failed(pass: &Arc<Self>) {
        let Some(mut main) = pass.main.take() else {
            return Self::finalize(pass);
        };
        let outcome = pass.stages.fence(&mut main);
        pass.main.send(main);
        match outcome {
            Ok(Attempt::Ready(mark)) => {
                let frame = Self::last(pass);
                if pass.stages.after(mark, frame).is_err() {
                    Self::finalize(pass);
                }
            }
            Ok(Attempt::Busy) => {
                let held = Arc::clone(pass);
                let frame = pass.frame(move || Box::new(move || Self::end_failed(&held)));
                if pass.lake.offer(frame).is_err() {
                    Self::finalize(pass);
                }
            }
            Err(_) => Self::finalize(pass),
        }
    }

    /// The frame that ends the pass, emitted when the device is through.
    fn last(pass: &Arc<Self>) -> Box<dyn Frame> {
        let held = Arc::clone(pass);
        pass.frame(move || Box::new(move || Self::finalize(&held)))
    }

    /// Returns the lanes' state, releases the lake and tells the caller.
    fn finalize(pass: &Arc<Self>) {
        let result = if pass.has_failed() {
            Err(pass
                .failure
                .take()
                .unwrap_or_else(|| ForwardError::Invalid("the pass failed".to_owned())))
        } else {
            Ok(())
        };
        let retired = result.is_ok();
        let lost = pass.give_back_with(retired);
        drop(pass.unit.take());
        let result = match (result, lost) {
            (Ok(()), true) => Err(ForwardError::Invalid("the pass lost a lane's state".to_owned())),
            (result, _) => result,
        };
        pass.ending.result.send(result);
    }

    fn give_back(&self) {
        self.give_back_with(false);
    }

    /// Puts the main state and the slots back at the scheduler; true when one was missing (then none are put back).
    fn give_back_with(&self, retire: bool) -> bool {
        let main = self.main.take();
        let mut slots = Vec::with_capacity(self.slots.len());
        for cell in &self.slots {
            if let Some(slot) = cell.take() {
                slots.push(slot);
            }
        }
        match main {
            Some(main) if slots.len() == self.slots.len() => {
                if retire {
                    self.stages.retire(&mut slots);
                }
                self.home.send((main, slots));
                false
            }
            _ => true,
        }
    }
}
