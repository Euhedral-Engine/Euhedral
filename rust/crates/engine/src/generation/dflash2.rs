//! Speculative decoding with the DFlash2 drafter (`ExecutionPlan.dflash2Context` and `dflash2Block`): the owner's side
//! of a speculative request on an artifact whose drafter is DFlash2, beside [`mtp`](super::mtp)'s.
//!
//! The drafter reads the base model's hidden rows at its tapped layers. Every base quantum writes the taps of its rows
//! to the sequence's tap buffer, from row 0, and the drafter's key/value rings follow the committed rows only. A step
//! is three quanta, admitted together as one chain on the device:
//!
//! 1. the **context** ([`Role::Context`]): the rows the last verification committed, whose taps are still in the tap
//!    buffer, are projected into the rings. It is left out when no row is pending;
//! 2. the **block** ([`Role::Block`]): the drafter over the anchor (the current token) and the mask rows, scored with the
//!    base model's head and selected on the device. It writes the anchor and the drafts to the step tokens, and the
//!    drafts to its selection, which the host reads when it retires;
//! 3. the **verification** ([`Role::Verify`]): the base model over `[current, d1 .. dn]`, read from the step tokens, as
//!    MTP's.
//!
//! The rings hold the committed rows of the sequence, so the drafter's context length is the anchor's position when
//! the block runs ([`Drafter`]): a verification's rows past the accepted ones are never read, and the rings are not
//! truncated, as the positions they would hold are written again, at the same place in the window, by the rows that
//! take them. The prompt's chunks are projected as they are prefilled, as the next chunk overwrites the taps; the last
//! chunk's are the first step's context. A request that samples or is constrained does not speculate.

use std::mem;
use std::sync::Arc;

use super::prefix_plan::{Carrier, carrier};
use super::source::{Admission, Flying, Mode, OVERLAP_MAX_START_POSITION, Owner, Role, Run};
use crate::prefix::SequenceAddresses;
use crate::stages::attention::KvSource;
use crate::stages::{QuantumKind, SequenceRefs};

/// How far the drafter's context is of the sequence's committed rows.
#[derive(Debug, Default)]
pub(super) struct Drafter {
    /// Rows in the rings: the positions `0 .. context_len` are projected.
    context_len: u64,
    /// The committed rows after them, whose taps are the first rows of the tap buffer.
    pending: usize,
    /// Drafts of the step in flight.
    drafts: usize,
}

impl Drafter {
    /// A base quantum committed `rows` rows after the context, whose taps it left in the tap buffer: a prompt chunk.
    pub(super) fn tapped(&mut self, rows: usize) -> Result<(), String> {
        if self.pending != 0 {
            return Err(format!(
                "{rows} rows were tapped over {} pending ones, which the context has not taken",
                self.pending
            ));
        }
        self.pending = rows;
        Ok(())
    }

    /// The prompt resumes from a checkpoint at `position`, whose rings hold the rows before it as a cold run's do.
    pub(super) fn resumed_at(&mut self, position: u64) {
        *self = Drafter {
            context_len: position,
            ..Drafter::default()
        };
    }

    /// The rows the context update takes, as their first position and their count: all that are pending, which it
    /// leaves in the rings. `None` when there are none.
    pub(super) fn take_context(&mut self) -> Option<(u64, usize)> {
        let rows = mem::take(&mut self.pending);
        let start = self.context_len;
        self.context_len += rows as u64;
        (rows > 0).then_some((start, rows))
    }

    /// Starts the step of `drafts` drafts after the committed rows up to `base`: the context update it needs, and the
    /// refusal when the rings would not hold `base` rows once it ran.
    pub(super) fn begin_step(&mut self, base: u64, drafts: usize) -> Result<Option<(u64, usize)>, String> {
        let context = self.take_context();
        if self.context_len != base {
            return Err(format!(
                "the drafter's context of {} rows does not end at the anchor's position {base}",
                self.context_len
            ));
        }
        self.drafts = drafts;
        Ok(context)
    }

    /// The verification of the step started at `start` committed `count` rows, whose taps are the first rows of the
    /// tap buffer.
    pub(super) fn committed(&mut self, start: u64, count: usize) -> Result<(), String> {
        if self.context_len != start || self.pending != 0 {
            return Err(format!(
                "a verification at {start} committed rows over a drafter context of {} rows with {} pending",
                self.context_len, self.pending
            ));
        }
        self.pending = count;
        Ok(())
    }

    /// Drafts the step in flight makes.
    pub(super) fn drafts(&self) -> usize {
        self.drafts
    }

    /// The rows after the context, which the next step's context update takes.
    #[cfg(test)]
    fn position(&self) -> u64 {
        self.context_len + self.pending as u64
    }
}

/// What a step of the drafter takes of the request's state.
pub(super) struct DrafterStep {
    /// The context update: the position of its first row and its rows.
    pub(super) context: Option<(u64, usize)>,
    /// The anchor: the token the block and the verification feed first.
    pub(super) current: u32,
    /// The rows each GDN layer replays first.
    pub(super) replay: Vec<u32>,
}

impl Owner {
    /// Whether the live request drafts with the DFlash2 drafter.
    pub(super) fn dflash2(&self) -> bool {
        matches!(
            self.run.as_ref().map(|run| &run.mode),
            Some(Mode::Speculative { state, .. }) if state.drafter().is_some()
        )
    }

    /// The context update of the `rows` prompt rows just prefilled: their taps go to the rings before the next chunk
    /// overwrites them. Nothing is scored.
    pub(super) fn admit_prompt_context(&mut self, rows: usize) -> Result<(), String> {
        let Some(run) = self.run.as_mut() else { return Ok(()) };
        let Mode::Speculative { state, .. } = &mut run.mode else {
            return Err("a prompt context in a request that does not speculate".to_string());
        };
        let drafter = state
            .drafter_mut()
            .ok_or("a prompt context in a request without the drafter")?;
        let (start, taken) = drafter.take_context().ok_or("a prompt context with no rows tapped")?;
        if taken != rows {
            return Err(format!("a prompt context of {rows} rows with {taken} tapped"));
        }
        state.prompt_chunk_taken();
        // The chunk that ends at a checkpoint is captured once its rows are in the rings: nothing is chained behind
        // the update, and the next chunk waits for the capture.
        let end = start as usize + rows;
        let checkpoint = carrier(state.draft_kind(), false) == Carrier::DraftUpdate
            && run
                .checkpoints
                .as_ref()
                .is_some_and(|checkpoints| checkpoints.wants(end));
        let refs = self.admit_context(start, rows, checkpoint)?;
        if let Some(refs) = refs
            && let Some(run) = self.run.as_mut()
            && let Some(addresses) = SequenceAddresses::new(refs, &self.shared.config)
            && let Some(checkpoints) = run.checkpoints.as_mut()
        {
            checkpoints.expect(end, Arc::new(addresses));
            if let Some(flying) = run.in_flight.back_mut() {
                flying.checkpoint = true;
            }
        }
        Ok(())
    }

    /// Admits the context update of `rows` rows from the position `start`, chained to the quantum behind it unless
    /// `capture` is set, which returns the references the update was admitted with.
    fn admit_context(&mut self, start: u64, rows: usize, capture: bool) -> Result<Option<SequenceRefs>, String> {
        let admission = Admission {
            role: Role::Context,
            kind: QuantumKind::Context,
            rows,
            start,
            source: None,
            pdl: false,
            logits: 0,
            replay: Vec::new(),
            last_chunk: false,
            more: !capture,
        };
        self.admit_quantum_taking(admission, move |quantum| quantum.with_tokens(vec![0; rows]), capture)
    }

    /// Admits a step of `drafts` drafts: the context update, the block and the verification, one chain.
    pub(super) fn admit_dflash2_speculation(&mut self, drafts: usize) -> Result<(), String> {
        let config = Arc::clone(&self.shared.config);
        let drafter = config.dflash2.as_ref().ok_or("the model has no DFlash2 drafter")?;
        let Some(run) = self.run.as_mut() else { return Ok(()) };
        let base = run.position;
        let Mode::Speculative { state, .. } = &mut run.mode else {
            return Err("a speculative step in a request that does not speculate".to_string());
        };
        let step = state.begin_drafter_step(base, drafts, &config)?;
        if let Some((start, rows)) = step.context {
            self.admit_context(start, rows, false)?;
        }
        let mut tokens = vec![drafter.mask_token; drafter.block_size];
        tokens[0] = step.current;
        // The block drafts a token for each mask row; the verification takes the first `drafts` of them.
        let admission = Admission {
            role: Role::Block,
            kind: QuantumKind::Block,
            rows: drafter.block_size,
            start: base,
            source: None,
            pdl: false,
            logits: drafter.drafts_per_block(),
            replay: Vec::new(),
            last_chunk: false,
            more: true,
        };
        self.admit_quantum(admission, move |quantum| quantum.with_tokens(tokens))?;
        // The verification reads the anchor and the drafts from the step tokens, which the block wrote.
        let admission = Admission {
            role: Role::Verify,
            kind: QuantumKind::Verify,
            rows: drafts + 1,
            start: base,
            source: Some(KvSource::Base),
            pdl: base < OVERLAP_MAX_START_POSITION,
            logits: drafts + 1,
            replay: step.replay,
            last_chunk: false,
            more: false,
        };
        self.admit_quantum(admission, |quantum| quantum.reading_draft_tokens(0))
    }

    /// The block of the live step retired: its selection holds the drafts the drafter selected, as tokens of the model.
    pub(super) fn blocked(&mut self, flying: &Flying) {
        let vocabulary = self.shared.config.vocab_size;
        let Some(Run {
            mode: Mode::Speculative { state, .. },
            ..
        }) = self.run.as_mut()
        else {
            return;
        };
        let Some(drafts) = state.drafter().map(Drafter::drafts) else {
            return self.fail_step("a block retired in a request without the drafter".to_string());
        };
        let tokens: Option<Vec<u32>> = flying
            .selection
            .as_ref()
            .and_then(|selection| (0..drafts).map(|index| selection.word(index)).collect());
        match tokens {
            Some(tokens) if tokens.iter().all(|token| (*token as usize) < vocabulary) => {
                state.proposed(tokens, flying.retired_at);
            }
            Some(tokens) => {
                self.fail_step(format!(
                    "the drafter selected {tokens:?}, outside the {vocabulary} tokens of the model"
                ));
            }
            None => self.fail_step("a block's selection holds no drafts".to_string()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A fake of the owner's bookkeeping across a request: `prompt` rows in chunks, then steps that each commit `commits`.
    fn run(prompt: &[usize], commits: &[usize]) -> (Drafter, u64) {
        let mut drafter = Drafter::default();
        let mut position = 0u64;
        for (index, rows) in prompt.iter().enumerate() {
            drafter.tapped(*rows).unwrap();
            position += *rows as u64;
            if index + 1 < prompt.len() {
                let (start, taken) = drafter.take_context().unwrap();
                assert_eq!((start + taken as u64), position);
            }
        }
        for count in commits {
            let context = drafter.begin_step(position, 6).unwrap();
            if let Some((start, rows)) = context {
                assert_eq!(start + rows as u64, position);
            }
            drafter.committed(position, *count).unwrap();
            position += *count as u64;
            assert_eq!(drafter.position(), position);
        }
        (drafter, position)
    }

    #[test]
    fn the_context_ends_at_the_anchor_through_every_step() {
        let (drafter, position) = run(&[512, 512, 37], &[1, 7, 3, 2, 7]);
        assert_eq!(drafter.position(), position);
        assert_eq!(position, 1061 + 20);
    }

    #[test]
    fn the_first_step_takes_the_last_prompt_chunk_and_later_steps_the_committed_rows() {
        let mut drafter = Drafter::default();
        drafter.tapped(40).unwrap();
        assert_eq!(drafter.begin_step(40, 6).unwrap(), Some((0, 40)));
        drafter.committed(40, 4).unwrap();
        assert_eq!(drafter.begin_step(44, 6).unwrap(), Some((40, 4)));
    }

    #[test]
    fn a_step_at_another_position_than_the_context_ends_at_is_refused() {
        let mut drafter = Drafter::default();
        drafter.tapped(40).unwrap();
        let why = drafter.begin_step(41, 6).unwrap_err();
        assert!(why.contains("40") && why.contains("41"), "{why}");
    }

    #[test]
    fn a_verification_that_commits_over_pending_rows_is_refused() {
        let mut drafter = Drafter::default();
        drafter.tapped(40).unwrap();
        assert!(drafter.committed(0, 3).is_err());
        assert!(drafter.tapped(5).is_err());
    }

    #[test]
    fn a_prompt_resumed_from_a_checkpoint_has_its_context_end_there() {
        let mut drafter = Drafter::default();
        drafter.resumed_at(1024);
        // The prompt's last chunk after the checkpoint is the first step's context, from the checkpoint's position.
        drafter.tapped(40).unwrap();
        assert_eq!(drafter.begin_step(1064, 6).unwrap(), Some((1024, 40)));
        assert!(
            Drafter::default().begin_step(1064, 6).is_err(),
            "a cold context does not end there"
        );
    }

    #[test]
    fn a_step_with_nothing_pending_has_no_context_update() {
        let mut drafter = Drafter::default();
        assert_eq!(drafter.begin_step(0, 6).unwrap(), None);
    }
}
