//! The host's part of a sampled request: the frames that select tokens from logits rows, and the bookkeeping that
//! tells the owner which selections it is waiting for.
//!
//! A sampled quantum copies its logits rows to pinned memory inside its graph. When the quantum has retired the owner
//! publishes one [`Kind::Sample`] frame for each row, which owns a share of the rows' lease, that row's sampler and the
//! step and position it selects for. The frames of a quantum are independent and run side by side: each runs
//! [`TokenSampler::select_bf16`], lets go of its share of the lease, and posts its answer to the owner. The owner joins
//! the answers ([`SampledRows`]) and admits nothing until all have arrived: the tokens decide what comes next. A frame
//! that is dropped without running, or whose selection panics, still posts, with an error, so the request ends and never
//! waits for a selection that will not come.

use std::mem;
use std::ops::{Deref, DerefMut};
use std::panic::{self, AssertUnwindSafe};
use std::sync::Arc;

use euhedral_core::frames::Frame;

use super::constraint::{ConstraintError, TokenConstraint};
use crate::frames::{Kind, step};
use crate::sampling::{TokenFilter, TokenSampler};
use crate::stages::sampling::RowReadbackLease;

/// The logits row a [`Sample`] frame selects from.
pub(crate) trait Row: Send {
    /// The row's BF16 bytes, exactly the vocabulary wide; `None` when there is no such row.
    fn bytes(&self) -> Option<&[u8]>;
}

/// One row of a readback that the frames of a quantum share: the readback goes back to its pool when the last frame
/// has let go of its row.
pub(crate) struct LeasedRow {
    lease: Arc<RowReadbackLease>,
    index: usize,
}

impl LeasedRow {
    pub(crate) fn new(lease: Arc<RowReadbackLease>, index: usize) -> LeasedRow {
        LeasedRow { lease, index }
    }
}

impl Row for LeasedRow {
    fn bytes(&self) -> Option<&[u8]> {
        self.lease.readback().row(self.index)
    }
}

/// What the frame tells the owner.
pub(crate) struct Sampled {
    /// The selection the answer is for.
    pub(crate) step: u64,
    /// The token and the sampler, which the request keeps for its next selection, or why there is no token.
    pub(crate) result: Result<(u32, Box<TokenSampler>), String>,
    /// The request's constraint, which has accepted the token, when the request has one and the selection succeeded.
    pub(crate) constraint: Option<Guarded>,
}

/// What the preparation of a constraint's next mask tells the owner.
pub(crate) struct Prepared {
    /// The preparation the answer is for.
    pub(crate) step: u64,
    /// The constraint, ready to select from, or why it is not.
    pub(crate) result: Result<Guarded, String>,
}

/// A constraint on its way through the request's frames and the owner. Its `Drop` may panic (it is a caller's code), and
/// a drop on the owner's turn would unwind past the request's end and lose the outcome: it is let go of inside
/// `catch_unwind`, wherever the constraint ends up.
pub(crate) struct Guarded(Box<dyn TokenConstraint>);

/// What takes the place of a constraint that was let go of.
struct Gone;

impl TokenFilter for Gone {
    fn allows(&self, _id: u32) -> bool {
        false
    }
}

impl TokenConstraint for Gone {
    fn accept(&mut self, _id: u32) -> Result<(), ConstraintError> {
        Ok(())
    }
}

impl Guarded {
    pub(crate) fn new(constraint: Box<dyn TokenConstraint>) -> Guarded {
        Guarded(constraint)
    }
}

impl Deref for Guarded {
    type Target = dyn TokenConstraint;

    fn deref(&self) -> &(dyn TokenConstraint + 'static) {
        &*self.0
    }
}

impl DerefMut for Guarded {
    fn deref_mut(&mut self) -> &mut (dyn TokenConstraint + 'static) {
        &mut *self.0
    }
}

impl Drop for Guarded {
    fn drop(&mut self) {
        let inner = mem::replace(&mut self.0, Box::new(Gone));
        let _ = panic::catch_unwind(AssertUnwindSafe(move || drop(inner)));
    }
}

/// The selections of one quantum that are waiting for their frames: `count` of them, numbered from `first`.
#[derive(Debug, Default)]
pub(crate) struct Pending {
    first: u64,
    answered: Vec<bool>,
    outstanding: usize,
}

impl Pending {
    /// Starts the selections numbered `first` to `first + count`, which the owner draws from a counter that never
    /// restarts. One quantum's at a time: a request keeps one quantum in flight.
    pub(crate) fn begin(&mut self, first: u64, count: usize) {
        self.first = first;
        self.answered.clear();
        self.answered.resize(count, false);
        self.outstanding = count;
    }

    /// The row of selection `step` if it is one awaited, which it then no longer is. Any other step, and an answer that
    /// comes twice, is stale.
    pub(crate) fn take(&mut self, step: u64) -> Option<usize> {
        let row = usize::try_from(step.checked_sub(self.first)?).ok()?;
        let answered = self.answered.get_mut(row)?;
        if *answered {
            return None;
        }
        *answered = true;
        self.outstanding -= 1;
        Some(row)
    }

    /// Whether a selection is awaited.
    pub(crate) fn waiting(&self) -> bool {
        self.outstanding > 0
    }

    /// Gives up the selections from `row` on, which were never published, so only the published ones are awaited.
    pub(crate) fn abandon_from(&mut self, row: usize) {
        for answered in self.answered.iter_mut().skip(row).filter(|answered| !**answered) {
            *answered = true;
            self.outstanding -= 1;
        }
    }
}

/// What an answer did to the join.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Answered {
    /// The answer is not for a selection awaited.
    Stale,
    /// Others are awaited.
    Waiting,
    /// This was the last: the rows' tokens are known, or [`SampledRows::failure`] says why not.
    Joined,
}

/// The samplers of a sampled request, one for each row its quanta select from at once, and the join of a quantum's
/// answers. Made once with the request: a selection takes its row's sampler and the answer gives it back.
#[derive(Debug)]
pub(crate) struct SampledRows {
    samplers: Vec<Option<Box<TokenSampler>>>,
    pending: Pending,
    tokens: Vec<u32>,
    failure: Option<String>,
}

impl SampledRows {
    pub(crate) fn new(samplers: Vec<TokenSampler>) -> SampledRows {
        let rows = samplers.len();
        SampledRows {
            samplers: samplers.into_iter().map(|sampler| Some(Box::new(sampler))).collect(),
            pending: Pending::default(),
            tokens: vec![0; rows],
            failure: None,
        }
    }

    /// Starts the selections of `count` rows numbered from `first`; [`SampledRows::sampler`] then gives each row's
    /// sampler. Nothing starts when the request has not got that many samplers at hand.
    pub(crate) fn begin(&mut self, first: u64, count: usize) -> Result<(), String> {
        if count == 0 || count > self.samplers.len() || self.samplers[..count].iter().any(Option::is_none) {
            return Err(format!(
                "the request has no sampler for each of {count} rows ({} in all)",
                self.samplers.len()
            ));
        }
        self.pending.begin(first, count);
        self.failure = None;
        Ok(())
    }

    /// The sampler of `row`, for the frame that selects from it.
    pub(crate) fn sampler(&mut self, row: usize) -> Option<Box<TokenSampler>> {
        self.samplers.get_mut(row)?.take()
    }

    /// Takes in an answer: the token or why there is none, and the row's sampler.
    pub(crate) fn answer(&mut self, sampled: Sampled) -> Answered {
        let Some(row) = self.pending.take(sampled.step) else {
            return Answered::Stale;
        };
        match sampled.result {
            Ok((token, sampler)) => {
                self.tokens[row] = token;
                self.samplers[row] = Some(sampler);
            }
            Err(why) => {
                self.failure.get_or_insert(why);
            }
        }
        if self.pending.waiting() {
            Answered::Waiting
        } else {
            Answered::Joined
        }
    }

    /// Whether answers are awaited.
    pub(crate) fn waiting(&self) -> bool {
        self.pending.waiting()
    }

    /// Ends a quantum whose frames from `row` on could not be published, saying `why`: the frames published answer, and
    /// the join then reports the failure. Whether any answer is still awaited.
    pub(crate) fn abandon_from(&mut self, row: usize, why: String) -> bool {
        self.pending.abandon_from(row);
        self.failure.get_or_insert(why);
        self.pending.waiting()
    }

    /// Why a selection of the quantum failed, when one did.
    pub(crate) fn failure(&self) -> Option<&str> {
        self.failure.as_deref()
    }

    /// The tokens of the rows, in order, once joined; the first of them for a quantum that selects one.
    pub(crate) fn tokens(&self) -> &[u32] {
        &self.tokens
    }

    /// The tokens, to read while the owner changes the request; [`SampledRows::restore`] gives them back.
    pub(crate) fn take_tokens(&mut self) -> Vec<u32> {
        mem::take(&mut self.tokens)
    }

    pub(crate) fn restore(&mut self, tokens: Vec<u32>) {
        self.tokens = tokens;
    }
}

/// What the frame owns, and posts the answer however it ends.
struct Sample<R: Row> {
    step: u64,
    position: u64,
    row: Option<R>,
    sampler: Option<Box<TokenSampler>>,
    constraint: Option<Guarded>,
    post: Option<Box<dyn FnOnce(Sampled) + Send>>,
}

impl<R: Row> Sample<R> {
    fn run(mut self) {
        let (Some(row), Some(mut sampler)) = (self.row.take(), self.sampler.take()) else {
            return;
        };
        let position = self.position;
        let mut constraint = self.constraint.take();
        let selected = panic::catch_unwind(AssertUnwindSafe(|| {
            let bytes = row
                .bytes()
                .ok_or_else(|| "the logits row is not in the readback".to_string())?;
            let filter = constraint.as_deref().map(|constraint| constraint as &dyn TokenFilter);
            let token = sampler
                .select_bf16(bytes, position, filter)
                .map_err(|error| error.to_string())?;
            // The token is committed as soon as it is chosen: the next mask depends on it.
            if let Some(constraint) = constraint.as_deref_mut() {
                constraint
                    .accept(token)
                    .map_err(|error| format!("the constraint refused token {token}: {error}"))?;
            }
            Ok(token)
        }))
        .unwrap_or_else(|_| {
            Err(if constraint.is_some() {
                "the constraint or the sampler panicked"
            } else {
                "the sampler panicked"
            }
            .to_string())
        });
        // The row goes back to its pool before the owner hears: its next quantum may take it at once.
        drop(row);
        match selected {
            Ok(token) => self.answer(Ok((token, sampler)), constraint),
            Err(why) => {
                drop(constraint);
                self.answer(Err(why), None);
            }
        }
    }

    fn answer(&mut self, result: Result<(u32, Box<TokenSampler>), String>, constraint: Option<Guarded>) {
        if let Some(post) = self.post.take() {
            post(Sampled {
                step: self.step,
                result,
                constraint,
            });
        }
    }
}

impl<R: Row> Drop for Sample<R> {
    fn drop(&mut self) {
        if self.post.is_some() {
            // Dropped before it ran (the lattice ended, or the frame was discarded): the request must still end.
            self.row = None;
            self.sampler = None;
            self.constraint = None;
            self.answer(Err("the selection was dropped before it ran".to_string()), None);
        }
    }
}

/// The frame that selects the token at `position` from `row` with `sampler` and calls `post` with the answer.
pub(crate) fn sample_frame<R: Row + 'static>(
    step_number: u64,
    position: u64,
    sampler: Box<TokenSampler>,
    constraint: Option<Guarded>,
    row: R,
    post: impl FnOnce(Sampled) + Send + 'static,
) -> Box<dyn Frame> {
    let sample = Sample {
        step: step_number,
        position,
        row: Some(row),
        sampler: Some(sampler),
        constraint,
        post: Some(Box::new(post)),
    };
    step(Kind::Sample, 0, move || sample.run())
}

/// What the frame that prepares a constraint owns, and posts the answer however it ends.
struct Prepare {
    step: u64,
    constraint: Option<Guarded>,
    post: Option<Box<dyn FnOnce(Prepared) + Send>>,
}

impl Prepare {
    fn run(mut self) {
        let Some(mut constraint) = self.constraint.take() else {
            return;
        };
        let prepared = panic::catch_unwind(AssertUnwindSafe(|| constraint.prepare()));
        match prepared {
            Ok(Ok(())) => self.answer(Ok(constraint)),
            Ok(Err(error)) => {
                drop(constraint);
                self.answer(Err(format!("preparing the constraint: {error}")));
            }
            Err(_) => {
                drop(constraint);
                self.answer(Err("the constraint panicked while preparing".to_string()));
            }
        }
    }

    fn answer(&mut self, result: Result<Guarded, String>) {
        if let Some(post) = self.post.take() {
            post(Prepared {
                step: self.step,
                result,
            });
        }
    }
}

impl Drop for Prepare {
    fn drop(&mut self) {
        if self.post.is_some() {
            // Dropped before it ran: the request must still end.
            self.constraint = None;
            self.answer(Err("the preparation was dropped before it ran".to_string()));
        }
    }
}

/// The frame that prepares `constraint` for the next selection and calls `post` with it, or with why it is not ready.
pub(crate) fn prepare_frame(
    step_number: u64,
    constraint: Guarded,
    post: impl FnOnce(Prepared) + Send + 'static,
) -> Box<dyn Frame> {
    let prepare = Prepare {
        step: step_number,
        constraint: Some(constraint),
        post: Some(Box::new(post)),
    };
    step(Kind::Prepare, 0, move || prepare.run())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};

    use super::*;
    use crate::generation::ConstraintError;
    use crate::handoff::Handoff;
    use crate::sampling::SamplingConfig;

    /// What the frames under test posted: the first answer and how many came.
    #[derive(Clone)]
    struct Answers {
        first: Arc<Handoff<Sampled>>,
        count: Arc<AtomicUsize>,
    }

    impl Default for Answers {
        fn default() -> Answers {
            Answers {
                first: Handoff::new(),
                count: Arc::default(),
            }
        }
    }

    impl Answers {
        fn poster(&self) -> impl FnOnce(Sampled) + Send + 'static {
            let (first, count) = (Arc::clone(&self.first), Arc::clone(&self.count));
            move |sampled| {
                if count.fetch_add(1, Ordering::SeqCst) == 0 {
                    first.send(sampled);
                }
            }
        }

        /// The answer posted, which must be the only one.
        fn only(&self) -> Sampled {
            assert_eq!(self.count.load(Ordering::SeqCst), 1, "one answer");
            self.first.take().unwrap()
        }
    }

    struct Bytes(Vec<u8>);

    impl Row for Bytes {
        fn bytes(&self) -> Option<&[u8]> {
            Some(&self.0)
        }
    }

    struct Panics;

    impl Row for Panics {
        fn bytes(&self) -> Option<&[u8]> {
            panic!("a row that cannot be read")
        }
    }

    fn bf16(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn sampler(vocabulary: usize, top_k: u32) -> Box<TokenSampler> {
        let config = SamplingConfig::new(1.0, top_k, 1.0, 7, false).unwrap();
        Box::new(TokenSampler::new(config, vocabulary).unwrap())
    }

    fn run(frame: &mut Box<dyn Frame>) {
        frame.execute().unwrap();
    }

    #[test]
    fn a_frame_selects_the_row_and_hands_the_sampler_back() {
        let answers = Answers::default();
        let row = Bytes(bf16(&[0.5, 3.0, -1.0, 2.0]));
        let mut frame = sample_frame(4, 9, sampler(4, 1), None, row, answers.poster());
        run(&mut frame);
        let sampled = answers.only();
        assert_eq!(sampled.step, 4);
        let (token, sampler) = sampled.result.as_ref().unwrap();
        assert_eq!(*token, 1, "the highest logit under top-k 1");
        assert_eq!(sampler.vocabulary(), 4);
    }

    #[test]
    fn a_row_of_the_wrong_width_is_an_error() {
        let answers = Answers::default();
        let mut frame = sample_frame(0, 0, sampler(4, 1), None, Bytes(bf16(&[1.0, 2.0])), answers.poster());
        run(&mut frame);
        let sampled = answers.only();
        assert!(sampled.result.is_err());
    }

    #[test]
    fn a_frame_dropped_before_it_ran_still_answers_once_with_an_error() {
        let answers = Answers::default();
        let frame = sample_frame(2, 0, sampler(4, 1), None, Bytes(bf16(&[1.0; 4])), answers.poster());
        drop(frame);
        let sampled = answers.only();
        assert_eq!(sampled.step, 2);
        assert!(sampled.result.is_err());
    }

    #[test]
    fn a_selection_that_panics_answers_with_an_error() {
        let answers = Answers::default();
        let mut frame = sample_frame(1, 0, sampler(4, 1), None, Panics, answers.poster());
        run(&mut frame);
        let sampled = answers.only();
        assert_eq!(
            sampled.result.as_ref().err().map(String::as_str),
            Some("the sampler panicked")
        );
    }

    #[test]
    fn only_the_awaited_steps_are_taken_and_a_stale_one_is_ignored() {
        let mut pending = Pending::default();
        assert!(!pending.waiting());
        pending.begin(10, 3);
        assert!(pending.waiting());
        assert_eq!(pending.take(9), None, "a step before the quantum's");
        assert_eq!(pending.take(13), None, "a step after the quantum's");
        assert_eq!(pending.take(u64::MAX), None);
        assert!(pending.waiting(), "a stale answer leaves the wait");
        assert_eq!(pending.take(12), Some(2), "answers come in any order");
        assert_eq!(pending.take(12), None, "an answer comes once");
        assert_eq!(pending.take(10), Some(0));
        assert!(pending.waiting());
        assert_eq!(pending.take(11), Some(1));
        assert!(!pending.waiting());
        pending.begin(13, 1);
        assert_eq!(pending.take(12), None, "the answer of an earlier quantum is stale");
        assert_eq!(pending.take(13), Some(0));
    }

    #[test]
    fn an_answer_of_an_earlier_request_does_not_match_the_next_requests_selection() {
        // Two requests of one owner number their selections from one counter: the second's first selection is not
        // the first's.
        let mut earlier = Pending::default();
        earlier.begin(0, 1);
        let mut next = Pending::default();
        next.begin(1, 4);
        assert_eq!(next.take(0), None, "the late answer is stale");
        assert_eq!(next.take(1), Some(0));
    }

    /// A sampled request's rows: `count` samplers of top-k 1, which pick the largest logit of a row.
    fn rows(count: usize) -> SampledRows {
        SampledRows::new((0..count).map(|_| *sampler(4, 1)).collect())
    }

    /// The answer of the frame of `row`, which selected `token`.
    fn answered(step: u64, token: u32) -> Sampled {
        Sampled {
            step,
            result: Ok((token, sampler(4, 1))),
            constraint: None,
        }
    }

    #[test]
    fn the_join_gives_the_tokens_of_the_rows_whatever_the_order_of_the_answers() {
        let mut join = rows(3);
        join.begin(20, 3).unwrap();
        assert_eq!(join.answer(answered(22, 7)), Answered::Waiting);
        assert!(join.waiting());
        assert_eq!(join.answer(answered(20, 5)), Answered::Waiting);
        assert_eq!(join.answer(answered(21, 6)), Answered::Joined);
        assert!(!join.waiting());
        assert_eq!(join.failure(), None);
        assert_eq!(join.tokens(), [5, 6, 7]);
        // Every sampler came back: the next quantum selects again.
        join.begin(23, 3).unwrap();
        assert!((0..3).all(|row| join.sampler(row).is_some()));
    }

    #[test]
    fn a_missing_answer_keeps_the_join_waiting() {
        let mut join = rows(2);
        join.begin(0, 2).unwrap();
        assert_eq!(join.answer(answered(1, 3)), Answered::Waiting);
        assert!(join.waiting(), "the row that never answered is awaited");
    }

    #[test]
    fn an_error_in_one_row_is_kept_until_the_join_and_names_the_failure() {
        let mut join = rows(3);
        join.begin(5, 3).unwrap();
        assert!(
            (0..3).all(|row| join.sampler(row).is_some()),
            "the frames take the samplers"
        );
        let failed = Sampled {
            step: 6,
            result: Err("no token".to_string()),
            constraint: None,
        };
        assert_eq!(join.answer(failed), Answered::Waiting);
        assert_eq!(join.failure(), Some("no token"), "known as soon as it arrives");
        assert_eq!(join.answer(answered(5, 1)), Answered::Waiting);
        assert_eq!(join.answer(answered(7, 2)), Answered::Joined);
        assert_eq!(join.failure(), Some("no token"), "and still known at the join");
        // The next quantum starts clean; the lost sampler leaves it short.
        assert!(join.begin(8, 3).is_err(), "the failed row's sampler did not come back");
        assert!(join.begin(8, 1).is_ok());
        assert_eq!(join.failure(), None);
    }

    #[test]
    fn rows_never_published_are_no_longer_awaited() {
        let mut join = rows(3);
        join.begin(5, 3).unwrap();
        assert!(
            join.abandon_from(2, "no sampler".to_string()),
            "two published rows answer"
        );
        assert_eq!(join.answer(answered(5, 1)), Answered::Waiting);
        assert_eq!(join.answer(answered(6, 2)), Answered::Joined);
        assert_eq!(join.failure(), Some("no sampler"));
        join.begin(8, 3).unwrap();
        assert!(
            !join.abandon_from(0, "none".to_string()),
            "nothing was published: nothing answers"
        );
    }

    #[test]
    fn a_stale_step_changes_nothing_in_the_join() {
        let mut join = rows(2);
        join.begin(10, 2).unwrap();
        assert_eq!(join.answer(answered(9, 3)), Answered::Stale);
        assert_eq!(join.answer(answered(12, 3)), Answered::Stale);
        assert_eq!(join.answer(answered(10, 1)), Answered::Waiting);
        assert_eq!(join.answer(answered(10, 2)), Answered::Stale, "a row answers once");
        assert!(join.waiting());
        assert_eq!(join.answer(answered(11, 2)), Answered::Joined);
        assert_eq!(join.tokens(), [1, 2]);
    }

    #[test]
    fn the_join_refuses_more_rows_than_it_has_samplers() {
        let mut join = rows(2);
        assert!(join.begin(0, 3).is_err());
        assert!(join.begin(0, 0).is_err());
        assert!(!join.waiting(), "a refused quantum waits for nothing");
    }

    #[test]
    fn frames_of_one_quantum_select_each_its_row_at_its_position_and_the_last_to_finish_frees_the_lease() {
        // Two rows of one readback; the lease is held by the frames only.
        let shared = Arc::new(Bytes2(vec![bf16(&[0.0, 9.0, 1.0, 2.0]), bf16(&[9.0, 0.0, 1.0, 2.0])]));
        let answers = Answers::default();
        let mut first = sample_frame(
            0,
            5,
            sampler(4, 1),
            None,
            Indexed(Arc::clone(&shared), 0),
            answers.poster(),
        );
        let mut second = sample_frame(
            1,
            6,
            sampler(4, 1),
            None,
            Indexed(Arc::clone(&shared), 1),
            answers.poster(),
        );
        assert_eq!(Arc::strong_count(&shared), 3);
        // The second frame runs first.
        run(&mut second);
        assert_eq!(
            Arc::strong_count(&shared),
            2,
            "a frame lets go of its share before it answers"
        );
        run(&mut first);
        assert_eq!(Arc::strong_count(&shared), 1);
        assert_eq!(answers.count.load(Ordering::SeqCst), 2);
    }

    /// What a fake constraint does.
    #[derive(Clone, Copy, PartialEq)]
    enum Fault {
        None,
        RefusesTokens,
        PanicsOnAllows,
        PanicsOnAccept,
        ErrorsOnPrepare,
        PanicsOnPrepare,
    }

    /// A constraint that allows the ids of `allowed`, and counts its calls.
    struct Fake {
        allowed: Vec<u32>,
        fault: Fault,
        accepted: Arc<AtomicUsize>,
        last: Arc<AtomicU32>,
        prepared: Arc<AtomicUsize>,
    }

    /// What a fake constraint did, which the test reads once the frames ran.
    #[derive(Default)]
    struct Watch {
        accepted: Arc<AtomicUsize>,
        last: Arc<AtomicU32>,
        prepared: Arc<AtomicUsize>,
    }

    impl Fake {
        fn new(allowed: &[u32], fault: Fault) -> (Fake, Watch) {
            let watch = Watch::default();
            let fake = Fake {
                allowed: allowed.to_vec(),
                fault,
                accepted: Arc::clone(&watch.accepted),
                last: Arc::clone(&watch.last),
                prepared: Arc::clone(&watch.prepared),
            };
            (fake, watch)
        }
    }

    impl TokenFilter for Fake {
        fn allows(&self, id: u32) -> bool {
            if self.fault == Fault::PanicsOnAllows {
                panic!("a constraint that cannot answer");
            }
            self.allowed.contains(&id)
        }
    }

    impl TokenConstraint for Fake {
        fn prepare(&mut self) -> Result<(), ConstraintError> {
            self.prepared.fetch_add(1, Ordering::SeqCst);
            match self.fault {
                Fault::ErrorsOnPrepare => Err(ConstraintError::new("no mask")),
                Fault::PanicsOnPrepare => panic!("a constraint that cannot prepare"),
                _ => Ok(()),
            }
        }

        fn accept(&mut self, id: u32) -> Result<(), ConstraintError> {
            match self.fault {
                Fault::RefusesTokens => return Err(ConstraintError::new("refused")),
                Fault::PanicsOnAccept => panic!("a constraint that cannot accept"),
                _ => {}
            }
            self.accepted.fetch_add(1, Ordering::SeqCst);
            self.last.store(id, Ordering::SeqCst);
            Ok(())
        }
    }

    fn greedy_sampler(vocabulary: usize) -> Box<TokenSampler> {
        Box::new(TokenSampler::new(SamplingConfig::greedy(0), vocabulary).unwrap())
    }

    #[test]
    fn a_greedy_constrained_frame_takes_the_best_allowed_token_and_hands_the_constraint_back_having_accepted_it() {
        // The row's best token (1) is not allowed: the frame takes the best of the one the constraint allows.
        let (fake, watch) = Fake::new(&[3], Fault::None);
        let answers = Answers::default();
        let row = Bytes(bf16(&[0.5, 3.0, -1.0, 2.0]));
        let mut frame = sample_frame(
            0,
            4,
            greedy_sampler(4),
            Some(Guarded::new(Box::new(fake))),
            row,
            answers.poster(),
        );
        run(&mut frame);
        let sampled = answers.only();
        assert_eq!(sampled.result.as_ref().unwrap().0, 3);
        assert!(sampled.constraint.is_some(), "the constraint comes back");
        assert_eq!(watch.accepted.load(Ordering::SeqCst), 1, "accepted once");
        assert_eq!(watch.last.load(Ordering::SeqCst), 3, "the token that was selected");
    }

    #[test]
    fn a_constraint_that_allows_one_token_a_step_is_followed_step_by_step() {
        let mut constraint: Option<Guarded> = Some(Guarded::new(Box::new(Fake::new(&[2], Fault::None).0)));
        let mut sampler = greedy_sampler(4);
        for step in 0..3u64 {
            let answers = Answers::default();
            let mut frame = sample_frame(
                step,
                step,
                sampler,
                constraint.take(),
                Bytes(bf16(&[9.0, 8.0, -5.0, 7.0])),
                answers.poster(),
            );
            run(&mut frame);
            let sampled = answers.only();
            let (token, back) = sampled.result.unwrap();
            assert_eq!(token, 2, "the only allowed token, however low its logit");
            sampler = back;
            constraint = sampled.constraint;
            assert!(constraint.is_some());
        }
    }

    #[test]
    fn a_constraint_with_nothing_allowed_ends_the_selection_with_an_error() {
        let answers = Answers::default();
        let (fake, watch) = Fake::new(&[], Fault::None);
        let mut frame = sample_frame(
            0,
            0,
            greedy_sampler(4),
            Some(Guarded::new(Box::new(fake))),
            Bytes(bf16(&[1.0, 2.0, 3.0, 4.0])),
            answers.poster(),
        );
        run(&mut frame);
        let sampled = answers.only();
        assert!(sampled.result.is_err());
        assert!(sampled.constraint.is_none());
        assert_eq!(watch.accepted.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_constraint_that_refuses_the_token_ends_the_selection_once_with_an_error() {
        let answers = Answers::default();
        let (fake, _) = Fake::new(&[1], Fault::RefusesTokens);
        let mut frame = sample_frame(
            0,
            0,
            greedy_sampler(4),
            Some(Guarded::new(Box::new(fake))),
            Bytes(bf16(&[1.0, 2.0, 3.0, 4.0])),
            answers.poster(),
        );
        run(&mut frame);
        let sampled = answers.only();
        let why = sampled.result.err().unwrap();
        assert!(why.contains("refused"), "{why}");
        assert!(sampled.constraint.is_none());
    }

    #[test]
    fn a_constraint_that_panics_answers_once_with_an_error_and_the_frame_does_not_hang() {
        for fault in [Fault::PanicsOnAllows, Fault::PanicsOnAccept] {
            let answers = Answers::default();
            let (fake, _) = Fake::new(&[1], fault);
            let mut frame = sample_frame(
                0,
                0,
                greedy_sampler(4),
                Some(Guarded::new(Box::new(fake))),
                Bytes(bf16(&[1.0, 2.0, 3.0, 4.0])),
                answers.poster(),
            );
            // The frame runs to its end on this thread: a panic that escaped would fail the test here, one that
            // swallowed the answer would fail the count.
            run(&mut frame);
            let sampled = answers.only();
            assert!(sampled.result.is_err());
            assert!(sampled.constraint.is_none());
        }
    }

    #[test]
    fn a_selection_frame_with_a_constraint_dropped_before_it_ran_answers_once() {
        let answers = Answers::default();
        let (fake, _) = Fake::new(&[1], Fault::None);
        let frame = sample_frame(
            3,
            0,
            greedy_sampler(4),
            Some(Guarded::new(Box::new(fake))),
            Bytes(bf16(&[1.0; 4])),
            answers.poster(),
        );
        drop(frame);
        let sampled = answers.only();
        assert_eq!(sampled.step, 3);
        assert!(sampled.result.is_err() && sampled.constraint.is_none());
    }

    /// What a preparation frame posted.
    #[derive(Clone)]
    struct Prepares {
        first: Arc<Handoff<Prepared>>,
        count: Arc<AtomicUsize>,
    }

    impl Default for Prepares {
        fn default() -> Prepares {
            Prepares {
                first: Handoff::new(),
                count: Arc::default(),
            }
        }
    }

    impl Prepares {
        fn poster(&self) -> impl FnOnce(Prepared) + Send + 'static {
            let (first, count) = (Arc::clone(&self.first), Arc::clone(&self.count));
            move |prepared| {
                if count.fetch_add(1, Ordering::SeqCst) == 0 {
                    first.send(prepared);
                }
            }
        }

        fn only(&self) -> Prepared {
            assert_eq!(self.count.load(Ordering::SeqCst), 1, "one answer");
            self.first.take().unwrap()
        }
    }

    #[test]
    fn a_preparation_frame_prepares_once_and_hands_the_constraint_back() {
        let (fake, watch) = Fake::new(&[1], Fault::None);
        let posts = Prepares::default();
        let mut frame = prepare_frame(7, Guarded::new(Box::new(fake)), posts.poster());
        run(&mut frame);
        let answer = posts.only();
        assert_eq!(answer.step, 7);
        assert!(answer.result.is_ok());
        assert_eq!(watch.prepared.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn a_preparation_that_fails_panics_or_is_dropped_answers_once_with_an_error() {
        for fault in [Fault::ErrorsOnPrepare, Fault::PanicsOnPrepare] {
            let (fake, _) = Fake::new(&[1], fault);
            let posts = Prepares::default();
            let mut frame = prepare_frame(1, Guarded::new(Box::new(fake)), posts.poster());
            run(&mut frame);
            assert!(posts.only().result.is_err());
        }
        let (fake, watch) = Fake::new(&[1], Fault::None);
        let posts = Prepares::default();
        drop(prepare_frame(2, Guarded::new(Box::new(fake)), posts.poster()));
        let answer = posts.only();
        assert_eq!(answer.step, 2);
        assert!(answer.result.is_err());
        assert_eq!(watch.prepared.load(Ordering::SeqCst), 0, "it never ran");
    }

    /// A constraint whose `Drop` panics, as caller code may.
    struct Bomb;

    impl TokenFilter for Bomb {
        fn allows(&self, _id: u32) -> bool {
            true
        }
    }

    impl TokenConstraint for Bomb {
        fn accept(&mut self, _id: u32) -> Result<(), ConstraintError> {
            Ok(())
        }
    }

    impl Drop for Bomb {
        fn drop(&mut self) {
            panic!("a constraint that cannot be dropped");
        }
    }

    #[test]
    fn a_constraint_whose_drop_panics_is_let_go_of_without_unwinding() {
        drop(Guarded::new(Box::new(Bomb)));
        let mut constraint = Some(Guarded::new(Box::new(Bomb)));
        assert!(constraint.take().is_some());
        drop(constraint);
        let posts = Prepares::default();
        drop(prepare_frame(3, Guarded::new(Box::new(Bomb)), posts.poster()));
        assert!(posts.only().result.is_err());
    }

    struct Bytes2(Vec<Vec<u8>>);

    struct Indexed(Arc<Bytes2>, usize);

    impl Row for Indexed {
        fn bytes(&self) -> Option<&[u8]> {
            self.0.0.get(self.1).map(Vec::as_slice)
        }
    }
}
