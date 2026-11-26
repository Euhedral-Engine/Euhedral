//! Which linear each lane takes next, and when the round ends: the part of a layer's rounding that does not touch the
//! card, generic over the lane and over the buffer a linear's decoded weight lives in.
//!
//! Every lane pulls the next linear of the order, works on it, and pulls again until none is left; a failure stops
//! every lane from pulling, lets the ones in flight end, and ends the round with the first error. The last lane to give
//! itself back ends the round.

use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use euhedral_inference_engine::Handoff;

use super::CalibrateError;
use super::driver::Then;

/// What a round gives back once every lane has ended without a failure.
#[derive(Debug)]
pub struct Settled<L, B> {
    /// The lanes, in their own order.
    pub lanes: Vec<L>,
    /// Each linear's decoded buffer, if it was given back, in the linears' own order.
    pub decoded: Vec<Option<B>>,
}

/// A round in progress, shared by its lanes.
pub struct Schedule<L, B> {
    names: Vec<String>,
    /// The linears' indices, in the order they are taken.
    order: Vec<usize>,
    next: AtomicUsize,
    running: AtomicUsize,
    failed: AtomicBool,
    error: Handoff<CalibrateError>,
    lanes: Vec<Handoff<L>>,
    decoded: Vec<Handoff<B>>,
    ended: Handoff<Then<Settled<L, B>>>,
}

impl<L, B> Debug for Schedule<L, B> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Schedule").finish_non_exhaustive()
    }
}

impl<L: Send + 'static, B: Send + 'static> Schedule<L, B> {
    /// A round of the linears `names` (each with its `decoded` buffer, if it has one) in `order`, over `lanes`;
    /// `then` gets the outcome. At most as many lanes as linears are started: the others are given back at once.
    /// The count is how many lanes start.
    pub fn new(
        names: Vec<String>,
        order: Vec<usize>,
        decoded: Vec<Handoff<B>>,
        lanes: usize,
        then: Then<Settled<L, B>>,
    ) -> (Schedule<L, B>, usize) {
        let started = lanes.min(names.len()).max(1);
        let schedule = Schedule {
            names,
            order,
            next: AtomicUsize::new(0),
            running: AtomicUsize::new(started),
            failed: AtomicBool::new(false),
            error: Handoff::empty(),
            lanes: (0..lanes).map(|_| Handoff::empty()).collect(),
            decoded,
            ended: Handoff::empty(),
        };
        schedule.ended.send(then);
        (schedule, started)
    }

    /// Parks a lane that does not work this round.
    pub fn park(&self, number: usize, lane: L) {
        self.lanes[number].send(lane);
    }

    /// Takes the next linear for lane `number` and hands it to `work` with its decoded buffer, or ends the lane when
    /// none is left or the round has failed.
    pub fn pull(&self, number: usize, lane: L, work: impl FnOnce(usize, B, L)) {
        if self.failed.load(Ordering::Acquire) {
            self.give_back(number, lane);
            return;
        }
        let at = self.next.fetch_add(1, Ordering::AcqRel);
        let Some(&index) = self.order.get(at) else {
            self.give_back(number, lane);
            return;
        };
        match self.decoded[index].take() {
            Some(buffer) => work(index, buffer, lane),
            None => self.abort(
                number,
                lane,
                CalibrateError::failed(&self.names[index], "no buffer was given for the decoded weight"),
            ),
        }
    }

    /// Gives back `buffer`, the decoded weight of linear `index`, once its linear is rounded.
    pub fn keep(&self, index: usize, buffer: B) {
        self.decoded[index].send(buffer);
    }

    /// Lane `number` failed: the first error is the round's, and the lane ends.
    pub fn abort(&self, number: usize, lane: L, error: CalibrateError) {
        if !self.failed.swap(true, Ordering::AcqRel) {
            self.error.send(error);
        }
        self.give_back(number, lane);
    }

    /// The lane is done: it goes back to the round, and the last lane to end ends the round.
    pub fn give_back(&self, number: usize, lane: L) {
        self.lanes[number].send(lane);
        if self.running.fetch_sub(1, Ordering::AcqRel) != 1 {
            return;
        }
        let Some(then) = self.ended.take() else { return };
        if self.failed.load(Ordering::Acquire) {
            let error = self
                .error
                .take()
                .unwrap_or_else(|| CalibrateError::Failed("rounding failed".to_owned()));
            then(Err(error));
            return;
        }
        // Every lane has sent its end before the count reached 0, so no work touches the round any more.
        let lanes = self.lanes.iter().filter_map(Handoff::take).collect();
        let decoded = self.decoded.iter().map(Handoff::take).collect();
        then(Ok(Settled { lanes, decoded }));
    }
}
