//! Forms passes from the ready steps of concurrent sequences.
//!
//! The assembler is plain state confined to the frame that owns the workspace: every event is applied by that frame
//! and nothing waits. Rule: no pass forms while one is in flight, nor while a member of the last retired pass has not
//! reported (offered its next step or left). A pass then takes the pending offers in arrival order, up to `max_rows`
//! rows and `max_parts` offers; a step over `max_rows` forms a pass of its own.
//!
//! The owner drives it: after every event, including a refused [`offer`](PassAssembler::offer) (a step of no rows
//! releases its sequence's wait), and after [`leave`](PassAssembler::leave) and [`retired`](PassAssembler::retired),
//! it calls [`next_pass`](PassAssembler::next_pass), runs what it returns, and tells the assembler how many
//! retirements that pass will report with [`admitted`](PassAssembler::admitted). [`withdraw`](PassAssembler::withdraw)
//! only shrinks the pending offers and never releases a pass.
//!
//! Obligations of the owner:
//! - Between `next_pass` and `admitted` it must always call `admitted`; a run that failed or was abandoned is
//!   `admitted(0)`, after which it asks for the next pass at once. Otherwise `next_pass` returns `None` forever.
//! - The count must be the number of retirements the pass really owes. A run that fails after `admitted` must still
//!   call `retired` for every retirement it owes.
//! - The caps given to `new` must be non-zero: validate the settings first, since `new` panics otherwise.

use std::collections::{HashSet, VecDeque};
use std::error::Error;
use std::fmt;

pub use crate::state::SequenceId;

/// What the passes did, as numbers: each pass counts once, however many sequences it holds. A pass is one quantum, so
/// `passes` is the count of quanta, `segments` the sequences' parts they held and `batched` the quanta that held more
/// than one part, as the same fields do in the Java engine's per-kind statistics. Only the steps offered to the
/// assembler (the verifications) are counted here, as one total: the other kinds are admitted directly and are not
/// counted, and the disk and staging byte counts are not kept.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PassStatistics {
    /// Passes started (quanta).
    pub passes: u64,
    /// Rows the passes held.
    pub rows: u64,
    /// Sequences' parts the passes held (a pass of one has one).
    pub segments: u64,
    /// Passes that held more than one part.
    pub batched: u64,
}

/// A ready step offered for a pass.
pub trait PassOffer {
    /// The sequence the step belongs to.
    fn sequence(&self) -> SequenceId;
    /// The rows the step occupies in a pass.
    fn rows(&self) -> usize;
    /// The class of passes the step can join: only steps of one class share a pass.
    fn class(&self) -> u64 {
        0
    }
}

/// An offer refused before it was queued: nothing of it remains in the assembler.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Refusal {
    /// The step has no rows. It still counts as its sequence's report.
    NoRows,
    /// The sequence already has a pending step.
    Duplicate,
}

impl fmt::Display for Refusal {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NoRows => "a step of no rows",
            Self::Duplicate => "a sequence already has a pending step",
        })
    }
}

impl Error for Refusal {}

/// A call made in a state where it has no meaning.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Misuse {
    /// [`PassAssembler::retired`] with no retirement owed.
    NothingInFlight,
    /// [`PassAssembler::admitted`] with no pass handed out and unreported.
    NoPassForming,
}

impl fmt::Display for Misuse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NothingInFlight => "no pass is in flight",
            Self::NoPassForming => "no pass was handed out awaiting its retirement count",
        })
    }
}

impl Error for Misuse {}

/// Forms passes from offers.
#[derive(Debug)]
pub struct PassAssembler<O> {
    max_rows: usize,
    max_parts: usize,
    pending: VecDeque<O>,
    pending_sequences: HashSet<SequenceId>,
    /// Members of the last retired pass that have not offered or left.
    awaiting: HashSet<SequenceId>,
    /// Members of the pass in flight.
    flight: HashSet<SequenceId>,
    /// Members of the pass in flight that left before it retired.
    departed: HashSet<SequenceId>,
    /// The members of a pass handed out by `next_pass` whose retirement count is not yet known.
    forming: Option<Vec<SequenceId>>,
    in_flight: usize,
}

impl<O: PassOffer> PassAssembler<O> {
    /// An assembler whose passes hold at most `max_rows` rows and `max_parts` offers.
    ///
    /// # Panics
    /// When either cap is zero: a pass needs at least one row and one part.
    pub fn new(max_rows: usize, max_parts: usize) -> Self {
        assert!(
            max_rows >= 1 && max_parts >= 1,
            "a pass needs at least one row and one part"
        );
        Self {
            max_rows,
            max_parts,
            pending: VecDeque::new(),
            pending_sequences: HashSet::new(),
            awaiting: HashSet::new(),
            flight: HashSet::new(),
            departed: HashSet::new(),
            forming: None,
            in_flight: 0,
        }
    }

    /// Queues a ready step, which also counts as its sequence's report. A refused step with no rows is still the
    /// sequence's report, so it never holds the others back.
    ///
    /// # Errors
    /// [`Refusal::NoRows`] for a step of no rows and [`Refusal::Duplicate`] for a sequence that already has a pending
    /// step.
    pub fn offer(&mut self, offer: O) -> Result<(), Refusal> {
        let sequence = offer.sequence();
        if offer.rows() < 1 {
            self.awaiting.remove(&sequence);
            return Err(Refusal::NoRows);
        }
        if !self.pending_sequences.insert(sequence) {
            return Err(Refusal::Duplicate);
        }
        self.awaiting.remove(&sequence);
        self.pending.push_back(offer);
        Ok(())
    }

    /// Releases the wait for a sequence that will not offer again, and returns its pending step, if it has one, so the
    /// owner gets the step back. A member of the pass in flight is recorded as departed so that its retirement does
    /// not await it.
    pub fn leave(&mut self, sequence: SequenceId) -> Option<O> {
        self.awaiting.remove(&sequence);
        let in_pass = self.flight.contains(&sequence) || self.forming.as_ref().is_some_and(|f| f.contains(&sequence));
        if in_pass {
            self.departed.insert(sequence);
        }
        if !self.pending_sequences.remove(&sequence) {
            return None;
        }
        let at = self.pending.iter().position(|offer| offer.sequence() == sequence)?;
        self.pending.remove(at)
    }

    /// Retires one quantum of the pass in flight. Its members must report before the next pass forms, unless they
    /// already offered or left.
    ///
    /// # Errors
    /// [`Misuse::NothingInFlight`] when no retirement is owed.
    pub fn retired(&mut self, members: &[SequenceId]) -> Result<(), Misuse> {
        if self.in_flight == 0 {
            return Err(Misuse::NothingInFlight);
        }
        self.in_flight -= 1;
        for member in members {
            self.flight.remove(member);
            if !self.departed.remove(member) && !self.pending_sequences.contains(member) {
                self.awaiting.insert(*member);
            }
        }
        if self.in_flight == 0 {
            self.flight.clear();
            self.departed.clear();
        }
        Ok(())
    }

    /// Removes the pending offers `selected` accepts, in arrival order. Each is removed as it is chosen, so the state
    /// stays consistent if `selected` panics.
    pub fn withdraw(&mut self, mut selected: impl FnMut(&O) -> bool) -> Vec<O> {
        let mut out = Vec::new();
        let mut at = 0;
        while at < self.pending.len() {
            if selected(&self.pending[at]) {
                if let Some(offer) = self.pending.remove(at) {
                    self.pending_sequences.remove(&offer.sequence());
                    out.push(offer);
                }
            } else {
                at += 1;
            }
        }
        out
    }

    /// Forms the next pass, if the rule allows one: none while a pass is in flight or its count is outstanding, none
    /// while a member of the last retired pass has yet to report. The caller runs the pass and then calls
    /// [`admitted`](Self::admitted).
    pub fn next_pass(&mut self) -> Option<Vec<O>> {
        if self.forming.is_some() || self.in_flight > 0 || self.pending.is_empty() || !self.awaiting.is_empty() {
            return None;
        }
        let class = self.pending.front().map(PassOffer::class)?;
        let mut pass = Vec::new();
        let mut rows = 0usize;
        let mut at = 0;
        while at < self.pending.len() {
            let next = &self.pending[at];
            if next.class() != class {
                at += 1;
                continue;
            }
            if pass.len() >= self.max_parts || (!pass.is_empty() && rows + next.rows() > self.max_rows) {
                break;
            }
            let Some(next) = self.pending.remove(at) else { break };
            rows += next.rows();
            self.pending_sequences.remove(&next.sequence());
            pass.push(next);
        }
        self.forming = Some(pass.iter().map(PassOffer::sequence).collect());
        Some(pass)
    }

    /// Records that the pass last handed out admitted work that will report `retirements` retirements (one per
    /// quantum). With none, nothing retires to wake the offers behind it: ask for the next pass at once.
    ///
    /// # Errors
    /// [`Misuse::NoPassForming`] when no pass awaits its count.
    pub fn admitted(&mut self, retirements: usize) -> Result<(), Misuse> {
        let members = self.forming.take().ok_or(Misuse::NoPassForming)?;
        if retirements > 0 {
            self.in_flight += retirements;
            self.flight.extend(members);
        } else {
            for member in members {
                self.departed.remove(&member);
            }
        }
        Ok(())
    }

    /// Whether `sequence` has an offer waiting for a pass.
    pub fn has_pending(&self, sequence: SequenceId) -> bool {
        self.pending_sequences.contains(&sequence)
    }

    /// The number of offers waiting for a pass.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// The number of members of the last retired pass yet to report.
    pub fn awaiting(&self) -> usize {
        self.awaiting.len()
    }

    /// The number of retirements the passes in flight still owe.
    pub fn in_flight(&self) -> usize {
        self.in_flight
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{AssertUnwindSafe, catch_unwind};

    use super::*;

    const A: SequenceId = SequenceId(0);
    const B: SequenceId = SequenceId(1);
    const C: SequenceId = SequenceId(2);
    const D: SequenceId = SequenceId(3);
    const X: SequenceId = SequenceId(4);
    const Z: SequenceId = SequenceId(25);

    #[derive(Clone, Debug, PartialEq, Eq)]
    struct Step {
        sequence: SequenceId,
        rows: usize,
        serial: usize,
        class: u64,
    }

    impl PassOffer for Step {
        fn sequence(&self) -> SequenceId {
            self.sequence
        }
        fn rows(&self) -> usize {
            self.rows
        }
        fn class(&self) -> u64 {
            self.class
        }
    }

    /// An assembler with the passes it formed, each admitted as one retirement.
    struct Rig {
        assembler: PassAssembler<Step>,
        passes: Vec<Vec<Step>>,
        serial: usize,
    }

    impl Rig {
        fn new(max_rows: usize, max_parts: usize) -> Self {
            Self {
                assembler: PassAssembler::new(max_rows, max_parts),
                passes: Vec::new(),
                serial: 0,
            }
        }

        fn step(&mut self, sequence: SequenceId, rows: usize) -> Step {
            self.serial += 1;
            Step {
                sequence,
                rows,
                serial: self.serial,
                class: 0,
            }
        }

        /// Forms every pass the rule allows now, each admitted as one retirement.
        fn settle(&mut self) {
            while let Some(pass) = self.assembler.next_pass() {
                self.assembler.admitted(1).unwrap();
                self.passes.push(pass);
            }
        }

        fn offer(&mut self, sequence: SequenceId, rows: usize) -> Result<(), Refusal> {
            let step = self.step(sequence, rows);
            let result = self.assembler.offer(step);
            self.settle();
            result
        }

        fn leave(&mut self, sequence: SequenceId) -> Option<Step> {
            let returned = self.assembler.leave(sequence);
            self.settle();
            returned
        }

        fn retired(&mut self, members: &[SequenceId]) -> Result<(), Misuse> {
            let result = self.assembler.retired(members);
            self.settle();
            result
        }

        fn names(&self, pass: usize) -> Vec<SequenceId> {
            self.passes[pass].iter().map(PassOffer::sequence).collect()
        }
    }

    #[test]
    fn a_first_offer_runs_at_once_alone() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        assert_eq!(rig.passes.len(), 1);
        assert_eq!(rig.names(0), [A]);
        assert_eq!(rig.assembler.in_flight(), 1);
        assert_eq!(rig.assembler.pending(), 0);
    }

    #[test]
    fn offers_during_a_pass_wait_for_every_member_to_report() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        assert_eq!(rig.passes.len(), 1);
        rig.retired(&[A]).unwrap();
        assert_eq!(rig.passes.len(), 1);
        assert_eq!(rig.assembler.awaiting(), 1);
        rig.offer(A, 1).unwrap();
        assert_eq!(rig.passes.len(), 2);
        assert_eq!(rig.names(1), [B, A]);
    }

    #[test]
    fn a_member_that_leaves_releases_the_wait() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        rig.retired(&[A]).unwrap();
        assert_eq!(rig.passes.len(), 1);
        rig.leave(A);
        assert_eq!(rig.passes.len(), 2);
        assert_eq!(rig.names(1), [B]);
        assert_eq!(rig.assembler.awaiting(), 0);
    }

    #[test]
    fn leaving_without_being_awaited_changes_nothing() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        assert!(rig.leave(Z).is_none());
        assert_eq!(rig.passes.len(), 1);
        assert_eq!(rig.assembler.pending(), 1);
        assert_eq!(rig.assembler.awaiting(), 0);
    }

    #[test]
    fn the_row_cap_splits_in_arrival_order_and_stops_at_the_first_misfit() {
        let mut rig = Rig::new(4, 8);
        rig.offer(X, 1).unwrap();
        rig.offer(A, 3).unwrap();
        rig.offer(B, 3).unwrap();
        rig.offer(C, 1).unwrap();
        rig.retired(&[]).unwrap();
        assert_eq!(rig.passes.len(), 2);
        assert_eq!(rig.names(1), [A]);
        rig.retired(&[A]).unwrap();
        assert_eq!(rig.passes.len(), 2);
        rig.offer(A, 3).unwrap();
        assert_eq!(rig.passes.len(), 3);
        assert_eq!(rig.names(2), [B, C]);
        assert_eq!(rig.assembler.pending(), 1);
    }

    #[test]
    fn max_parts_one_runs_singles() {
        let mut rig = Rig::new(8, 1);
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        rig.offer(C, 1).unwrap();
        rig.retired(&[]).unwrap();
        assert_eq!(rig.names(1), [B]);
        rig.retired(&[]).unwrap();
        assert_eq!(rig.names(2), [C]);
    }

    #[test]
    fn an_oversized_step_runs_alone_and_an_empty_one_is_refused() {
        let mut rig = Rig::new(4, 8);
        rig.offer(A, 5).unwrap();
        assert_eq!(rig.passes.len(), 1);
        assert_eq!(rig.passes[0][0].rows, 5);
        assert_eq!(rig.offer(B, 0), Err(Refusal::NoRows));
        assert_eq!(rig.assembler.pending(), 0);
        assert_eq!(rig.passes.len(), 1);
    }

    #[test]
    fn an_empty_step_still_reports_its_sequence() {
        let mut rig = Rig::new(4, 8);
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        rig.retired(&[A]).unwrap();
        assert_eq!(rig.passes.len(), 1);
        assert_eq!(rig.offer(A, 0), Err(Refusal::NoRows));
        assert_eq!(rig.passes.len(), 2);
    }

    #[test]
    fn a_pass_takes_only_the_offers_of_the_class_of_the_first() {
        let mut assembler = PassAssembler::<Step>::new(8, 8);
        for (serial, (sequence, class)) in [(A, 1), (B, 2), (X, 1)].into_iter().enumerate() {
            assembler
                .offer(Step {
                    sequence,
                    rows: 1,
                    serial,
                    class,
                })
                .unwrap();
        }
        let first = assembler.next_pass().unwrap();
        assert_eq!(first.iter().map(|step| step.sequence).collect::<Vec<_>>(), vec![A, X]);
        assert!(assembler.has_pending(B), "the other class keeps its place");
        assembler.admitted(1).unwrap();
        assembler.retired(&[A, X]).unwrap();
        assert!(assembler.next_pass().is_none(), "A and X have not reported");
        assembler.leave(A);
        assembler.leave(X);
        let second = assembler.next_pass().unwrap();
        assert_eq!(second.iter().map(|step| step.sequence).collect::<Vec<_>>(), vec![B]);
    }

    #[test]
    fn a_pass_that_admits_nothing_does_not_strand_the_offers_behind_it() {
        let mut assembler = PassAssembler::<Step>::new(8, 1);
        let mut seen = Vec::new();
        for (serial, sequence) in [X, SequenceId(9), B].into_iter().enumerate() {
            assembler
                .offer(Step {
                    sequence,
                    rows: 1,
                    serial,
                    class: 0,
                })
                .unwrap();
        }
        // The first pass admits one quantum; the second admits nothing.
        let first = assembler.next_pass().unwrap();
        assembler.admitted(1).unwrap();
        seen.push(first[0].sequence);
        assert!(assembler.next_pass().is_none());
        assembler.retired(&[]).unwrap();
        while let Some(pass) = assembler.next_pass() {
            seen.push(pass[0].sequence);
            let bad = pass[0].sequence == SequenceId(9);
            assembler.admitted(usize::from(!bad)).unwrap();
        }
        assert_eq!(seen, [X, SequenceId(9), B]);
        assert_eq!(assembler.pending(), 0);
        assert_eq!(assembler.in_flight(), 1);
    }

    #[test]
    fn a_step_exactly_the_row_cap_runs_alone() {
        let mut rig = Rig::new(4, 8);
        rig.offer(X, 1).unwrap();
        rig.offer(A, 1).unwrap();
        rig.offer(B, 4).unwrap();
        rig.retired(&[]).unwrap();
        assert_eq!(rig.names(1), [A]);
        rig.retired(&[]).unwrap();
        assert_eq!(rig.names(2), [B]);
    }

    #[test]
    fn withdraw_removes_only_matching_pending_offers() {
        let mut rig = Rig::new(8, 8);
        rig.offer(X, 1).unwrap();
        let (b, c, d) = (rig.step(B, 1), rig.step(C, 1), rig.step(D, 1));
        for step in [&b, &c, &d] {
            rig.assembler.offer(step.clone()).unwrap();
        }
        assert_eq!(rig.assembler.withdraw(|s| s.sequence == B || s.sequence == D), [b, d]);
        assert_eq!(rig.assembler.pending(), 1);
        rig.retired(&[]).unwrap();
        assert_eq!(rig.names(1), [C]);
        assert!(rig.assembler.withdraw(|_| true).is_empty());
    }

    #[test]
    fn a_withdrawn_sequence_can_offer_again() {
        let mut rig = Rig::new(8, 8);
        rig.offer(X, 1).unwrap();
        rig.offer(B, 1).unwrap();
        assert_eq!(rig.offer(B, 1), Err(Refusal::Duplicate));
        assert_eq!(rig.assembler.withdraw(|s| s.sequence == B).len(), 1);
        rig.offer(B, 1).unwrap();
        assert_eq!(rig.assembler.pending(), 1);
    }

    #[test]
    fn withdrawing_everything_pending_forms_no_empty_pass() {
        let mut rig = Rig::new(8, 8);
        rig.offer(X, 1).unwrap();
        rig.offer(B, 1).unwrap();
        rig.assembler.withdraw(|_| true);
        rig.retired(&[]).unwrap();
        assert_eq!(rig.passes.len(), 1);
        assert_eq!(rig.assembler.in_flight(), 0);
    }

    #[test]
    fn an_empty_retirement_forms_the_next_pass_at_once() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        rig.retired(&[]).unwrap();
        assert_eq!(rig.passes.len(), 2);
        assert_eq!(rig.names(1), [B]);
    }

    #[test]
    fn retiring_with_nothing_in_flight_is_refused() {
        let mut rig = Rig::new(8, 8);
        assert_eq!(rig.retired(&[]), Err(Misuse::NothingInFlight));
        rig.offer(A, 1).unwrap();
        rig.retired(&[]).unwrap();
        assert_eq!(rig.retired(&[]), Err(Misuse::NothingInFlight));
    }

    #[test]
    fn a_last_member_leaving_with_nothing_pending_forms_nothing() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        rig.retired(&[A]).unwrap();
        rig.leave(A);
        assert_eq!(rig.passes.len(), 1);
        rig.offer(B, 1).unwrap();
        assert_eq!(rig.passes.len(), 2);
    }

    #[test]
    fn offers_are_held_while_a_pass_is_in_flight_even_when_nobody_is_awaited() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        rig.offer(C, 1).unwrap();
        assert_eq!(rig.passes.len(), 1);
        assert_eq!(rig.assembler.pending(), 2);
        rig.retired(&[A]).unwrap();
        rig.offer(A, 1).unwrap();
        assert_eq!(rig.names(1), [B, C, A]);
    }

    #[test]
    fn an_offer_made_before_the_count_is_known_waits_for_the_retirement() {
        let mut assembler = PassAssembler::<Step>::new(4, 8);
        assembler
            .offer(Step {
                sequence: A,
                rows: 1,
                serial: 0,
                class: 0,
            })
            .unwrap();
        let first = assembler.next_pass().unwrap();
        assert_eq!(first.len(), 1);
        assembler
            .offer(Step {
                sequence: B,
                rows: 1,
                serial: 1,
                class: 0,
            })
            .unwrap();
        assert!(assembler.next_pass().is_none());
        assert_eq!(assembler.retired(&[]), Err(Misuse::NothingInFlight));
        assembler.admitted(1).unwrap();
        assert!(assembler.next_pass().is_none());
        assert_eq!(assembler.pending(), 1);
        assembler.retired(&[]).unwrap();
        let second = assembler.next_pass().unwrap();
        assert_eq!(second[0].sequence, B);
    }

    #[test]
    fn a_count_without_a_pass_is_refused() {
        let mut assembler = PassAssembler::<Step>::new(4, 8);
        assert_eq!(assembler.admitted(1), Err(Misuse::NoPassForming));
        assembler
            .offer(Step {
                sequence: A,
                rows: 1,
                serial: 0,
                class: 0,
            })
            .unwrap();
        assembler.next_pass().unwrap();
        assembler.admitted(1).unwrap();
        assert_eq!(assembler.admitted(1), Err(Misuse::NoPassForming));
        assert_eq!(assembler.in_flight(), 1);
    }

    #[test]
    fn a_pass_that_admits_nothing_leaves_nothing_in_flight() {
        let mut assembler = PassAssembler::<Step>::new(4, 8);
        let mut seen = 0;
        for (serial, sequence) in [A, B].into_iter().enumerate() {
            assembler
                .offer(Step {
                    sequence,
                    rows: 1,
                    serial,
                    class: 0,
                })
                .unwrap();
        }
        // Both offers are queued before any pass forms, so they form one pass.
        while let Some(pass) = assembler.next_pass() {
            seen += pass.len();
            assembler.admitted(0).unwrap();
        }
        assert_eq!(seen, 2);
        assert_eq!(assembler.in_flight(), 0);
        assert_eq!(assembler.pending(), 0);
        assembler
            .offer(Step {
                sequence: A,
                rows: 1,
                serial: 2,
                class: 0,
            })
            .unwrap();
        assert!(assembler.next_pass().is_some());
    }

    #[test]
    fn a_member_that_leaves_before_its_pass_retires_is_not_awaited() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        rig.leave(A);
        rig.retired(&[A]).unwrap();
        assert_eq!(rig.assembler.awaiting(), 0);
        assert_eq!(rig.passes.len(), 2);
        assert_eq!(rig.names(1), [B]);
    }

    #[test]
    fn a_member_that_left_while_its_pass_was_forming_is_not_awaited() {
        let mut assembler = PassAssembler::<Step>::new(8, 8);
        assembler
            .offer(Step {
                sequence: A,
                rows: 1,
                serial: 0,
                class: 0,
            })
            .unwrap();
        assembler
            .offer(Step {
                sequence: B,
                rows: 1,
                serial: 1,
                class: 0,
            })
            .unwrap();
        assert_eq!(assembler.next_pass().unwrap().len(), 2);
        assembler.leave(A);
        assembler.admitted(1).unwrap();
        assembler.retired(&[A, B]).unwrap();
        assert_eq!(assembler.awaiting(), 1);
    }

    #[test]
    fn a_member_that_offered_before_its_pass_retires_is_not_awaited() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        rig.offer(A, 1).unwrap();
        rig.retired(&[A]).unwrap();
        assert_eq!(rig.assembler.awaiting(), 0);
        assert_eq!(rig.passes.len(), 2);
    }

    #[test]
    fn a_member_that_never_reports_and_then_leaves_does_not_block_the_others() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        rig.offer(C, 1).unwrap();
        // The pass was [A]; B and C are pending. A reports nothing and then leaves.
        rig.retired(&[A]).unwrap();
        assert_eq!(rig.passes.len(), 1);
        assert_eq!(rig.assembler.awaiting(), 1);
        rig.leave(A);
        assert_eq!(rig.passes.len(), 2);
        assert_eq!(rig.names(1), [B, C]);
        // Both members then retire and report; the one that never does is released by leaving.
        rig.retired(&[B, C]).unwrap();
        rig.offer(B, 1).unwrap();
        assert_eq!(rig.passes.len(), 2);
        assert_eq!(rig.assembler.awaiting(), 1);
        rig.leave(C);
        assert_eq!(rig.passes.len(), 3);
        assert_eq!(rig.names(2), [B]);
    }

    #[test]
    fn two_retirements_owed_hold_the_next_pass_and_keep_departures() {
        let mut assembler = PassAssembler::<Step>::new(8, 8);
        for (serial, sequence) in [A, B].into_iter().enumerate() {
            assembler
                .offer(Step {
                    sequence,
                    rows: 1,
                    serial,
                    class: 0,
                })
                .unwrap();
        }
        let pass = assembler.next_pass().unwrap();
        assert_eq!(pass.len(), 2);
        assembler.admitted(2).unwrap();
        assembler
            .offer(Step {
                sequence: C,
                rows: 1,
                serial: 3,
                class: 0,
            })
            .unwrap();
        assert!(assembler.leave(A).is_none());
        assembler.retired(&[B]).unwrap();
        assert!(assembler.next_pass().is_none());
        assert_eq!(assembler.in_flight(), 1);
        assembler.retired(&[A]).unwrap();
        assert_eq!(assembler.awaiting(), 1);
        assert_eq!(assembler.in_flight(), 0);
        assert!(assembler.next_pass().is_none());
        assembler
            .offer(Step {
                sequence: B,
                rows: 1,
                serial: 4,
                class: 0,
            })
            .unwrap();
        assert_eq!(assembler.next_pass().unwrap().len(), 2);
    }

    #[test]
    fn a_member_that_offered_and_then_leaves_gets_its_step_back() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        rig.offer(A, 3).unwrap();
        assert_eq!(rig.assembler.pending(), 2);
        let back = rig.leave(A).unwrap();
        assert_eq!((back.sequence, back.rows), (A, 3));
        assert_eq!(rig.assembler.pending(), 1);
        // A is released at its pass's retirement, so B's pass forms and A may offer again later.
        rig.retired(&[A]).unwrap();
        assert_eq!(rig.names(1), [B]);
        assert_eq!(rig.assembler.awaiting(), 0);
        rig.offer(A, 1).unwrap();
        assert_eq!(rig.assembler.pending(), 1);
    }

    #[test]
    fn leaving_with_a_pending_step_returns_it_and_frees_the_slot() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        let back = rig.leave(B).unwrap();
        assert_eq!(back.sequence, B);
        assert_eq!(rig.assembler.pending(), 0);
        rig.retired(&[A]).unwrap();
        assert_eq!(rig.assembler.awaiting(), 1);
        rig.leave(A);
        assert_eq!(rig.assembler.awaiting(), 0);
        rig.offer(B, 1).unwrap();
        assert_eq!(rig.passes.len(), 2);
    }

    #[test]
    fn a_panicking_predicate_leaves_the_pending_state_consistent() {
        let mut rig = Rig::new(8, 8);
        rig.offer(X, 1).unwrap();
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        let mut seen = 0;
        let result = catch_unwind(AssertUnwindSafe(|| {
            rig.assembler.withdraw(|_| {
                seen += 1;
                assert!(seen < 2, "boom");
                true
            })
        }));
        assert!(result.is_err());
        // A was withdrawn and may offer again; B is still pending and still refused as a duplicate.
        assert_eq!(rig.assembler.pending(), 1);
        rig.offer(A, 1).unwrap();
        assert_eq!(rig.offer(B, 1), Err(Refusal::Duplicate));
    }

    #[test]
    fn a_failed_run_leaves_the_gate_consistent() {
        let mut assembler = PassAssembler::<Step>::new(4, 8);
        assembler
            .offer(Step {
                sequence: A,
                rows: 1,
                serial: 0,
                class: 0,
            })
            .unwrap();
        assert_eq!(assembler.next_pass().unwrap().len(), 1);
        assembler.admitted(0).unwrap();
        assert_eq!(assembler.in_flight(), 0);
        assert_eq!(assembler.pending(), 0);
        assembler
            .offer(Step {
                sequence: A,
                rows: 1,
                serial: 1,
                class: 0,
            })
            .unwrap();
        let pass = assembler.next_pass().unwrap();
        assert_eq!(pass[0].sequence, A);
        assembler.admitted(1).unwrap();
        assert_eq!(assembler.in_flight(), 1);
    }

    #[test]
    fn a_parts_cap_above_one_splits_after_that_many_offers() {
        let mut rig = Rig::new(8, 2);
        rig.offer(X, 1).unwrap();
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        rig.offer(C, 1).unwrap();
        rig.retired(&[]).unwrap();
        assert_eq!(rig.names(1), [A, B]);
        rig.retired(&[]).unwrap();
        assert_eq!(rig.names(2), [C]);
    }

    #[test]
    fn a_second_pending_offer_from_one_sequence_is_refused() {
        let mut rig = Rig::new(8, 8);
        rig.offer(A, 1).unwrap();
        rig.offer(B, 1).unwrap();
        assert_eq!(rig.offer(B, 1), Err(Refusal::Duplicate));
        assert_eq!(rig.assembler.pending(), 1);
    }

    #[test]
    #[should_panic(expected = "at least one row")]
    fn a_pass_with_no_row_cap_is_refused() {
        let _ = PassAssembler::<Step>::new(0, 1);
    }

    #[test]
    #[should_panic(expected = "at least one row")]
    fn a_pass_with_no_part_cap_is_refused() {
        let _ = PassAssembler::<Step>::new(1, 0);
    }
}
