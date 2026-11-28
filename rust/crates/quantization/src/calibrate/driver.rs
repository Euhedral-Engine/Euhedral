//! The layer loop: the calls of a calibration in their order, each ending in the continuation of the one before.
//!
//! Layer `i` runs while layer `i + 1` is read from the checkpoint: the two calls end in either
//! order and the later one goes on, so no thread waits for the other. The loop checkpoints after every
//! [`Plan::checkpoint_every`] layers and, at the end of a run of the whole model, checkpoints the last layer's states
//! and finishes.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

use euhedral_inference_engine::Handoff;

use super::CalibrateError;
use super::plan::Plan;

/// What a call tells its continuation.
pub type Then<T> = Box<dyn FnOnce(Result<T, CalibrateError>) + Send>;

/// A continuation taken once, from whichever call ends the chain.
pub type Once<T> = Arc<Handoff<Then<T>>>;

/// `then` as a continuation any of several calls may take.
pub fn single<T>(then: Then<T>) -> Once<T> {
    let cell = Arc::new(Handoff::empty());
    cell.send(then);
    cell
}

/// Ends the chain with `result`, unless a call already did.
pub fn end<T>(cell: &Once<T>, result: Result<T, CalibrateError>) {
    if let Some(then) = cell.take() {
        then(result);
    }
}

/// The calls of a calibration. Each takes the continuation to run once its work, which may run on other threads,
/// is over; none waits.
pub trait Stages: Send + Sync + 'static {
    /// A layer's weights read into host memory.
    type Staged: Send + 'static;

    /// Resumes from the checkpoint when it is the run's, else embeds the tokens as the first layer's input; gives
    /// the first layer to run.
    fn begin(&self, plan: &Plan, then: Then<usize>);

    /// Reads layer `layer` from the checkpoint.
    fn stage(&self, layer: usize, then: Then<Self::Staged>);

    /// Layer `layer`: its Hessians observed, its linears rounded, written and swapped in, then its pass over the
    /// states. The states are the next layer's input when it is over.
    fn layer(&self, layer: usize, staged: Self::Staged, then: Then<()>);

    /// Saves the states after `done` layers as the point to resume from.
    fn checkpoint(&self, done: usize, then: Then<()>);

    /// After the last layer of a run of the whole model, whose states are checkpointed (or were resumed): the heads
    /// and the MTP block. A run that stops early does not call it.
    fn finish(&self, then: Then<()>);
}

/// Runs the calibration `plan` describes; `done` gets its result from the frame that ended it.
pub fn drive<S: Stages>(stages: Arc<S>, plan: Plan, done: Then<()>) {
    let run = Arc::new(Run {
        stages,
        plan,
        done: Handoff::empty(),
    });
    run.done.send(done);
    let started = Arc::clone(&run);
    run.stages.begin(
        &run.plan,
        Box::new(move |first| match first {
            Err(error) => started.fail(error),
            Ok(first) if first >= started.plan.stop() => started.end(),
            Ok(first) => {
                let reading = Arc::clone(&started);
                started.stages.stage(
                    first,
                    Box::new(move |staged| match staged {
                        Ok(staged) => reading.step(first, staged),
                        Err(error) => reading.fail(error),
                    }),
                );
            }
        }),
    );
}

struct Run<S: Stages> {
    stages: Arc<S>,
    plan: Plan,
    done: Handoff<Then<()>>,
}

/// The two calls of a step, layer `i` and the read of layer `i + 1`: the second to arrive goes on.
struct Join<T> {
    arrivals: AtomicU8,
    expected: u8,
    ran: Handoff<Result<(), CalibrateError>>,
    staged: Handoff<Result<T, CalibrateError>>,
}

impl<S: Stages> Debug for Run<S> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("Run")
            .field("plan", &self.plan)
            .finish_non_exhaustive()
    }
}

impl<T> Debug for Join<T> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Join").finish_non_exhaustive()
    }
}

impl<S: Stages> Run<S> {
    /// Tells the caller the run ended with `result`; the first call counts.
    fn complete(&self, result: Result<(), CalibrateError>) {
        if let Some(done) = self.done.take() {
            done(result);
        }
    }

    fn fail(&self, error: CalibrateError) {
        self.complete(Err(error));
    }

    /// The last layer of the run ran. A run of the whole model checkpoints the states first (unless it never
    /// checkpoints), so that a failure of the heads resumes at the heads, not at an earlier layer.
    fn last_layer_done(self: &Arc<Self>) {
        if self.plan.stops_early() || self.plan.checkpoint_every == 0 {
            return self.end();
        }
        let run = Arc::clone(self);
        self.stages.checkpoint(
            self.plan.layers,
            Box::new(move |saved| match saved {
                Ok(()) => run.end(),
                Err(error) => run.fail(error),
            }),
        );
    }

    /// Every layer ran: a full run finishes, one that stopped early ends with its checkpoint left.
    fn end(self: &Arc<Self>) {
        if self.plan.stops_early() {
            self.complete(Ok(()));
            return;
        }
        let run = Arc::clone(self);
        self.stages.finish(Box::new(move |result| run.complete(result)));
    }

    /// Layer `layer` runs over `staged` while the next layer is read.
    fn step(self: &Arc<Self>, layer: usize, staged: S::Staged) {
        let reads = layer + 1 < self.plan.stop();
        let join = Arc::new(Join {
            arrivals: AtomicU8::new(0),
            expected: if reads { 2 } else { 1 },
            ran: Handoff::empty(),
            staged: Handoff::empty(),
        });
        if reads {
            let (run, join) = (Arc::clone(self), Arc::clone(&join));
            self.stages.stage(
                layer + 1,
                Box::new(move |staged| {
                    join.staged.send(staged);
                    run.arrive(&join, layer);
                }),
            );
        }
        let (run, join) = (Arc::clone(self), join);
        self.stages.layer(
            layer,
            staged,
            Box::new(move |result| {
                join.ran.send(result);
                run.arrive(&join, layer);
            }),
        );
    }

    /// One of a step's calls ended; the second to end goes on to the checkpoint, then the next layer.
    fn arrive(self: &Arc<Self>, join: &Join<S::Staged>, layer: usize) {
        if join.arrivals.fetch_add(1, Ordering::AcqRel) + 1 < join.expected {
            return;
        }
        // Both hand-offs were sent before the second arrival, so both hold a value.
        let (Some(ran), staged) = (join.ran.take(), join.staged.take()) else {
            self.fail(CalibrateError::Failed("a layer ended without its result".to_owned()));
            return;
        };
        if let Err(error) = ran {
            self.fail(error);
            return;
        }
        let next = layer + 1;
        let run = Arc::clone(self);
        let onward = move || match staged {
            None => run.last_layer_done(),
            Some(Ok(staged)) => run.step(next, staged),
            Some(Err(error)) => run.fail(error),
        };
        if !self.plan.checkpoint_due(next) {
            onward();
            return;
        }
        let run = Arc::clone(self);
        self.stages.checkpoint(
            next,
            Box::new(move |saved| match saved {
                Ok(()) => onward(),
                Err(error) => run.fail(error),
            }),
        );
    }
}
