//! Speculative decoding with the artifact's MTP layer (`MtpDecoder`, `docs/speculation/MTP_CONTRACT.md` section 4): the
//! owner's side of a speculative request.
//!
//! A step is `drafts` drafts and one verification. Its quanta are admitted together, one chain on the device:
//!
//! 1. the **catch-up** (`Role::CatchUp`): the MTP layer over the rows the last verification committed. Its rows pair the
//!    base model's final hidden of each position (the seed rows, written by the verification) with the token that
//!    followed it; its last row makes the first draft. It writes the step's current token (its last input) to slot 0
//!    of the sequence's step tokens and the first draft to slot 1;
//! 2. the **recursive drafts** (`Role::Draft`): one row each, seeded by the MTP hidden of the row before, reading the
//!    draft before it from the step tokens and writing the next;
//! 3. the **verification** (`Role::Verify`): the base model over `[current, d1 .. dn]`, row-exact, read from the step
//!    tokens, with the argmax of every row. It seeds the next step's catch-up.
//!
//! The host reads nothing until the verification retires. It then reads the drafts (the draft head's rows, mapped to
//! tokens) and the verifier's keys, accepts the drafts that equal what the model selects ([`accept`]), commits that
//! many rows of the base cache, drops the MTP cache's speculative rows, and goes on from the verifier's token. A GDN
//! layer's state has run past the rows that were not accepted: the next base quantum replays the accepted rows from
//! the checkpoint the verification took (`SequenceRefs::with_replay_rows`).
//!
//! A sampled request (a [`SampledRows`] in the state) speculates the same way. Its drafts stay greedy, on the device;
//! its prompt's last row and its verification's rows are copied to the host, one [`Kind::Sample`](crate::frames::Kind)
//! frame selects each row at the position of the token it chooses, and the owner joins the answers and resolves them
//! with [`accept_tokens`], the function the greedy path's keys reach too. The first token is the host's selection of the
//! prompt's last row, and the catch-up pairs the rows with it.
//!
//! The prompt is prefilled in chunks, each followed by its catch-up (the next chunk's seed rows overwrite its own); the
//! last chunk's catch-up is the first step's, which waits for the first token. The output is greedy decode's.

use std::mem;
use std::sync::Arc;
use std::time::{Duration, Instant};

use euhedral_inference_cuda::DevicePtr;

use super::acceptance::{Accepted, accept, accept_tokens};
use super::request::{FinishReason, Verification};
use super::sampled::SampledRows;
use super::source::{Admission, Flying, Mode, OVERLAP_MAX_START_POSITION, Owner, Role, Run, Step};
use crate::model::{LayerKind, ModelConfig};
use crate::prefix::SequenceAddresses;
use crate::stages::attention::KvSource;
use crate::stages::{DraftSeed, DraftTokenWrite, QuantumKind};

/// The state of a speculative request beyond a greedy one's.
#[derive(Debug)]
pub(super) struct Speculating {
    /// Drafts a step makes and its verification checks, at most.
    depth: usize,
    /// Whether the request drafts at all: it wants more than the first token, and the prompt leaves room for a row.
    drafting: bool,
    /// The last committed token, not yet fed to the model. `None` until the prompt has selected the first.
    current: Option<u32>,
    /// The position of the next catch-up's first row in the MTP cache.
    mtp_position: u64,
    /// The rows of the prompt chunk just prefilled, whose catch-up is next.
    prompt_catch_up: Option<usize>,
    /// The prompt resumes from a restored checkpoint, and the catch-up that is next re-pairs the checkpoint's last row.
    resuming: bool,
    /// The tokens the next step's catch-up pairs with the base hiddens of the rows from `mtp_position`: the tokens
    /// that followed them.
    catch_up: Vec<u32>,
    /// The rows each GDN layer replays first: the rows the last verification committed, when its state ran past them.
    replay: u32,
    /// The drafts of the step in flight, as tokens of the model, in order.
    drafts: Vec<u32>,
    /// When the step in flight was admitted, and when its last draft quantum retired.
    admitted: Option<Instant>,
    drafted_at: Option<Instant>,
    /// Whether the decode that feeds the last token has been admitted (a test hook).
    settled: bool,
    /// The samplers and the join of the host's selections, for a request that samples.
    rows: Option<SampledRows>,
    /// What the selections awaited are for.
    joining: Option<Joining>,
}

/// What the host's selections of a quantum's rows decide.
#[derive(Debug)]
pub(super) enum Joining {
    /// The token after the prompt (`first`), or after the last row the cache holds.
    Token { first: bool },
    /// A verification, which the selections resolve.
    Verification(Verifying),
}

/// The verification of a step, as far as its resolution needs it: the quantum that ran it has been concluded.
#[derive(Debug)]
pub(super) struct Verifying {
    start: u64,
    rows: usize,
    admitted: Instant,
    retired_at: Option<Instant>,
}

impl Verifying {
    pub(super) fn of(flying: &Flying) -> Verifying {
        Verifying {
            start: flying.start,
            rows: flying.rows,
            admitted: flying.admitted,
            retired_at: flying.retired_at,
        }
    }
}

impl Speculating {
    /// The state of a request for `max_tokens` tokens after a prompt of `prompt_len` in a context of `capacity` rows,
    /// that drafts `depth` tokens a step.
    pub(super) fn new(depth: usize, capacity: u64, prompt_len: usize, max_tokens: usize) -> Speculating {
        Speculating {
            depth,
            drafting: max_tokens > 1 && (prompt_len as u64) < capacity,
            current: None,
            mtp_position: 0,
            prompt_catch_up: None,
            resuming: false,
            catch_up: Vec::new(),
            replay: 0,
            drafts: Vec::new(),
            admitted: None,
            drafted_at: None,
            settled: false,
            rows: None,
            joining: None,
        }
    }

    /// The state of a request whose tokens the host selects with `rows`.
    pub(super) fn sampled(mut self, rows: SampledRows) -> Speculating {
        self.rows = Some(rows);
        self
    }

    /// Whether the host selects the request's tokens.
    pub(super) fn samples(&self) -> bool {
        self.rows.is_some()
    }

    pub(super) fn rows_mut(&mut self) -> Option<&mut SampledRows> {
        self.rows.as_mut()
    }

    /// Whether the host's selections of a quantum's rows are awaited.
    pub(super) fn selecting(&self) -> bool {
        self.rows.as_ref().is_some_and(SampledRows::waiting)
    }

    /// The selections about to be awaited are for `joining`.
    pub(super) fn joining(&mut self, joining: Joining) {
        self.joining = Some(joining);
    }

    /// Whether the request drafts: the prompt's chunks seed it.
    pub(super) fn drafting(&self) -> bool {
        self.drafting
    }

    /// The token the next step feeds first.
    pub(super) fn current(&self) -> Option<u32> {
        self.current
    }

    /// The prompt chunk of `rows` rows was admitted: its catch-up is next.
    pub(super) fn prefilled(&mut self, rows: usize) {
        self.prompt_catch_up = Some(rows);
    }

    /// The prompt resumes from a checkpoint at `position`, whose draft cache holds the rows below `position - 1` and
    /// whose seed row holds the base hidden row of `position - 1`. The row `position - 1` pairs that hidden row with the
    /// token at `position`. When the checkpoint paired it with the prompt's token (`whole`) the cache holds that row
    /// too and the chunks go on from `position`; otherwise it is re-paired first, in a catch-up of one row.
    pub(super) fn resumed_at(&mut self, position: u32, whole: bool) {
        if whole {
            self.mtp_position = u64::from(position);
            self.prompt_catch_up = None;
            self.resuming = false;
            return;
        }
        self.mtp_position = u64::from(position) - 1;
        self.prompt_catch_up = Some(1);
        self.resuming = true;
    }

    /// The restore that was asked for was refused: the prompt starts from its first row.
    pub(super) fn start_over(&mut self) {
        self.mtp_position = 0;
        self.prompt_catch_up = None;
        self.resuming = false;
    }

    /// The prompt selected its first token: the steps go on from it. The last chunk's rows pair with the prompt that
    /// follows them, and with the first token for the last.
    pub(super) fn first_token(&mut self, token: u32, prompt: &[u32]) {
        self.current = Some(token);
        if self.drafting {
            let from = self.mtp_position as usize + 1;
            self.catch_up = prompt[from..].to_vec();
            self.catch_up.push(token);
        }
    }

    /// The rows each GDN layer replays first, by layer, leaving none pending.
    pub(super) fn take_replay(&mut self, config: &ModelConfig) -> Vec<u32> {
        let rows = mem::take(&mut self.replay);
        if rows == 0 {
            return Vec::new();
        }
        config
            .layer_kinds
            .iter()
            .map(|kind| if *kind == LayerKind::GatedDeltaNet { rows } else { 0 })
            .collect()
    }

    /// The next step of the request.
    pub(super) fn next_step(&self, run: &Run, chunk: usize) -> Option<Step> {
        if let Some(rows) = self.prompt_catch_up {
            return Some(Step::PromptCatchUp { rows });
        }
        if run.offset < run.prompt.len() {
            let rows = chunk.min(run.prompt.len() - run.offset);
            return Some(Step::Prefill {
                rows,
                last: run.offset + rows == run.prompt.len(),
            });
        }
        // Until the prompt has selected the first token there is nothing to feed.
        self.current?;
        if run.generated >= run.wanted || run.position >= run.capacity {
            return None;
        }
        // The verification feeds the current token and the drafts, and the cache holds `room` more rows.
        let room = (run.capacity - run.position) as usize;
        let drafts = self.depth.min(room - 1);
        Some(if drafts == 0 {
            Step::Decode
        } else {
            Step::Speculate { drafts }
        })
    }

    /// Whether the request still has to feed its last token (the test hook).
    #[cfg_attr(
        not(feature = "test-oracles"),
        allow(dead_code, reason = "only the test hook asks for it")
    )]
    fn needs_settling(&self, run: &Run) -> bool {
        !self.settled && self.current.is_some() && run.position < run.capacity
    }
}

/// Where the sequence keeps what the draft quanta read and write on the device.
#[derive(Clone, Copy)]
struct DraftPointers {
    seed_rows: DevicePtr,
    recursion_hidden: DevicePtr,
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

impl Owner {
    /// The step that feeds the last token of a speculative request that ended at its budget, when the test hook asks
    /// for it.
    #[cfg(feature = "test-oracles")]
    pub(super) fn settle_step(&self) -> Option<Step> {
        use std::sync::atomic::Ordering;

        let run = self.run.as_ref()?;
        if !self.shared.settle_after_speculation.load(Ordering::Acquire) {
            return None;
        }
        match (&run.mode, &run.stopping) {
            (Mode::Speculative { state, .. }, Some(FinishReason::Length)) if state.needs_settling(run) => {
                Some(Step::Settle)
            }
            _ => None,
        }
    }

    /// There is no such step outside the tests.
    #[cfg(not(feature = "test-oracles"))]
    pub(super) fn settle_step(&self) -> Option<Step> {
        None
    }

    /// Where the sequence keeps the rows that seed the MTP layer.
    fn draft_pointers(&self) -> Result<DraftPointers, String> {
        let draft = self
            .sequence
            .device
            .scratch
            .draft()
            .ok_or_else(|| "the sequence has no draft buffers".to_string())?;
        Ok(DraftPointers {
            seed_rows: draft.seed_rows.ptr(),
            recursion_hidden: draft.recursion_hidden.ptr(),
        })
    }

    /// The MTP catch-up of the prompt chunk just prefilled: its rows pair the chunk's base hiddens (the seed rows) with
    /// the tokens that follow them. Nothing is scored.
    pub(super) fn admit_prompt_catch_up(&mut self, rows: usize) -> Result<(), String> {
        let seed = self.draft_pointers()?.seed_rows;
        let Some(run) = self.run.as_mut() else { return Ok(()) };
        let Mode::Speculative { state, .. } = &mut run.mode else {
            return Err("a prompt catch-up in a request that does not speculate".to_string());
        };
        let start = state.mtp_position;
        let first = start as usize + 1;
        let tokens = run.prompt[first..first + rows].to_vec();
        // The chunk this catch-up completes ends at a checkpoint when it is wanted: the state is whole once the draft's
        // rows are, and the next chunk (which overwrites the seed rows) waits for the capture. The re-pairing of a
        // restored checkpoint's last row completes no chunk.
        let end = first + rows - 1;
        let resuming = mem::take(&mut state.resuming);
        let checkpoint = !resuming
            && run
                .checkpoints
                .as_ref()
                .is_some_and(|checkpoints| checkpoints.wants(end));
        let admission = Admission {
            role: Role::PromptCatchUp,
            kind: QuantumKind::Draft,
            rows,
            start,
            source: KvSource::Mtp,
            pdl: false,
            logits: 0,
            replay: Vec::new(),
            last_chunk: false,
            more: !checkpoint,
        };
        let refs = self.admit_quantum_taking(
            admission,
            move |quantum| quantum.with_tokens(tokens)?.with_draft_seed(DraftSeed::Address(seed)),
            checkpoint,
        )?;
        if let Some(refs) = refs
            && let Some(run) = self.run.as_mut()
            // The seed row of the checkpoint is the last row of the chunk's seed rows, which is the last row of this
            // catch-up.
            && let Some(addresses) = SequenceAddresses::new(refs, &self.shared.config).map(|a| a.with_seed_row(rows as u32 - 1))
            && let Some(checkpoints) = run.checkpoints.as_mut()
        {
            checkpoints.expect(end, Arc::new(addresses));
            if let Some(flying) = run.in_flight.back_mut() {
                flying.checkpoint = true;
            }
        }
        if let Some(Run {
            mode: Mode::Speculative { state, .. },
            ..
        }) = self.run.as_mut()
        {
            state.mtp_position += rows as u64;
            state.prompt_catch_up = None;
        }
        Ok(())
    }

    /// The decode that feeds the last token of a speculative request (a test hook): it replays what the last
    /// verification left, and scores nothing.
    pub(super) fn admit_settle(&mut self) -> Result<(), String> {
        let config = Arc::clone(&self.shared.config);
        let Some(run) = self.run.as_mut() else { return Ok(()) };
        let Mode::Speculative { state, .. } = &mut run.mode else {
            return Err("a settling decode in a request that does not speculate".to_string());
        };
        let token = state.current.ok_or("the request has no token to feed")?;
        state.settled = true;
        let start = run.position;
        let admission = Admission {
            role: Role::Settle,
            kind: QuantumKind::Decode,
            rows: 1,
            start,
            source: KvSource::Base,
            pdl: start < OVERLAP_MAX_START_POSITION,
            logits: 0,
            replay: state.take_replay(&config),
            last_chunk: false,
            more: false,
        };
        self.admit_quantum(admission, move |quantum| quantum.with_tokens(vec![token]))?;
        if let Some(run) = self.run.as_mut() {
            run.position += 1;
        }
        Ok(())
    }

    /// Admits a step of `drafts` drafts: the catch-up, the recursive drafts and the verification, one chain.
    pub(super) fn admit_speculation(&mut self, drafts: usize) -> Result<(), String> {
        let pointers = self.draft_pointers()?;
        let ids = self
            .shared
            .drafts
            .as_ref()
            .ok_or_else(|| "the artifact has no draft head ids".to_string())?
            .device_token_ids;
        let config = Arc::clone(&self.shared.config);
        let Some(run) = self.run.as_mut() else { return Ok(()) };
        let base = run.position;
        let Mode::Speculative { state, .. } = &mut run.mode else {
            return Err("a speculative step in a request that does not speculate".to_string());
        };
        let catch_up = mem::take(&mut state.catch_up);
        if catch_up.is_empty() || state.mtp_position + catch_up.len() as u64 != base {
            return Err(format!(
                "the catch-up of {} rows from MTP position {} does not end at the verification's position {base}",
                catch_up.len(),
                state.mtp_position
            ));
        }
        let replay = state.take_replay(&config);
        state.drafts.clear();
        state.admitted = Some(Instant::now());
        state.drafted_at = None;
        let mtp_start = state.mtp_position;
        let catch_up_rows = catch_up.len();

        // The catch-up writes the step's current token and the first draft to the step tokens.
        let admission = Admission {
            role: Role::CatchUp,
            kind: QuantumKind::Draft,
            rows: catch_up_rows,
            start: mtp_start,
            source: KvSource::Mtp,
            pdl: false,
            logits: 1,
            replay: Vec::new(),
            last_chunk: false,
            more: true,
        };
        self.admit_quantum(admission, move |quantum| {
            quantum
                .with_tokens(catch_up)?
                .with_draft_seed(DraftSeed::Address(pointers.seed_rows))?
                .writing_current_token()?
                .with_draft_token_write(DraftTokenWrite { slot: 1, ids })
        })?;
        // Each recursive draft reads the draft before it and writes the next, seeded by the MTP hidden of the row before.
        for index in 1..drafts {
            let admission = Admission {
                role: Role::Draft,
                kind: QuantumKind::Draft,
                rows: 1,
                start: mtp_start + catch_up_rows as u64 + index as u64 - 1,
                source: KvSource::Mtp,
                pdl: false,
                logits: 1,
                replay: Vec::new(),
                last_chunk: false,
                more: true,
            };
            self.admit_quantum(admission, move |quantum| {
                quantum
                    .reading_draft_tokens(index)?
                    .with_draft_seed(DraftSeed::Address(pointers.recursion_hidden))?
                    .with_draft_token_write(DraftTokenWrite { slot: index + 1, ids })
            })?;
        }
        // The verification reads the current token and the drafts from the step tokens and seeds the next catch-up.
        let admission = Admission {
            role: Role::Verify,
            kind: QuantumKind::Verify,
            rows: drafts + 1,
            start: base,
            source: KvSource::Base,
            pdl: base < OVERLAP_MAX_START_POSITION,
            logits: drafts + 1,
            replay,
            last_chunk: false,
            more: false,
        };
        self.admit_quantum(admission, |quantum| quantum.reading_draft_tokens(0)?.seeding_draft())
    }

    /// A catch-up or a recursive draft of the live step retired: its selected row of the draft head is a token.
    pub(super) fn drafted(&mut self, flying: &Flying) {
        let row = flying.selection.as_ref().and_then(|selection| selection.token(0));
        let token = row.and_then(|row| {
            self.shared
                .drafts
                .as_ref()
                .and_then(|drafts| drafts.token_ids.get(row as usize).copied())
        });
        let Some(token) = token else {
            return self.fail_step("the draft head selected no token of its shortlist".to_string());
        };
        if let Some(Run {
            mode: Mode::Speculative { state, .. },
            ..
        }) = self.run.as_mut()
        {
            state.drafts.push(token);
            state.drafted_at = flying.retired_at;
        }
    }

    fn fail_step(&mut self, why: String) {
        if let Some(run) = self.run.as_mut() {
            run.stop(FinishReason::Failed(why));
        }
    }

    /// The verification of the live step retired: accepts the drafts the model's own tokens confirm, commits that many
    /// rows, reports the tokens and goes on from the verifier's.
    pub(super) fn verified(&mut self, flying: &Flying) {
        let Some(selection) = flying.selection.as_ref() else {
            return self.fail_step("a verification retired with no selection".to_string());
        };
        let keys: Option<Vec<u64>> = (0..flying.rows).map(|row| selection.key(row)).collect();
        let Some(keys) = keys else {
            return self.fail_step("a verification's keys are out of its selection".to_string());
        };
        self.resolve(Verifying::of(flying), |drafts, eos, budget| {
            accept(drafts, &keys, eos, budget)
        });
    }

    /// The host selected the rows of a quantum of a sampled speculative request: the token after the prompt, which the
    /// steps go on from, or the tokens of a verification's rows, which are resolved against its drafts.
    pub(super) fn joined_rows(&mut self) {
        let Some(run) = self.run.as_mut() else { return };
        let Mode::Speculative { state, .. } = &mut run.mode else {
            return;
        };
        let Some(rows) = state.rows.as_mut() else { return };
        let tokens = rows.take_tokens();
        let joining = state.joining.take();
        match joining {
            Some(Joining::Token { first }) => {
                if let Some(token) = tokens.first().copied() {
                    self.selected_token(token, first);
                }
            }
            Some(Joining::Verification(verifying)) => {
                let count = verifying.rows.min(tokens.len());
                self.resolve(verifying, |drafts, eos, budget| {
                    accept_tokens(drafts, &tokens[..count], eos, budget)
                });
            }
            None => self.fail_step("the host selected rows nothing waited for".to_string()),
        }
        if let Some(Run {
            mode: Mode::Speculative { state, .. },
            ..
        }) = self.run.as_mut()
            && let Some(rows) = state.rows.as_mut()
        {
            rows.restore(tokens);
        }
    }

    /// The host selected the token after the prompt (`first`) or after the last row: reports it, and goes on from it.
    fn selected_token(&mut self, token: u32, first: bool) {
        let Some(run) = self.run.as_mut() else { return };
        run.generated += 1;
        if let Some(listener) = run.listener() {
            listener.token(token, Instant::now());
        }
        if run.stop_tokens.contains(&token) {
            run.stop(FinishReason::Stop);
        } else if run.generated >= run.wanted {
            run.stop(FinishReason::Length);
        } else if let Mode::Speculative { state, .. } = &mut run.mode {
            if first {
                state.first_token(token, &run.prompt);
            } else {
                state.current = Some(token);
            }
        }
    }

    /// Resolves the verification of the live step with `accept`, which is given the drafts, the end tokens and the
    /// budget: commits the rows of the accepted outputs, reports them and goes on from the last.
    fn resolve(&mut self, verifying: Verifying, accept: impl FnOnce(&[u32], &[u32], u32) -> Accepted) {
        let now = Instant::now();
        let Some(run) = self.run.as_mut() else { return };
        let Mode::Speculative { listener, state } = &mut run.mode else {
            return;
        };
        let drafts = mem::take(&mut state.drafts);
        if drafts.len() + 1 != verifying.rows {
            return self.fail_step(format!(
                "a verification of {} rows retired with {} drafts",
                verifying.rows,
                drafts.len()
            ));
        }
        let budget = u32::try_from(run.wanted - run.generated).unwrap_or(u32::MAX);
        let accepted = accept(&drafts, &run.stop_tokens, budget);
        if accepted.unselected {
            return self.fail_step("the head selected no token: every logit was NaN or negative infinity".to_string());
        }
        // The rows of the accepted outputs become part of the base cache; the rest of the verification's rows, and the
        // MTP cache's speculative rows, are dropped.
        if let Err(error) = self.sequence.kv.commit_prefix(accepted.count as u64) {
            return self.fail_step(format!("committing the verified rows: {error}"));
        }
        match self.sequence.mtp_kv.as_mut() {
            Some(cache) => {
                if let Err(error) = cache.truncate(verifying.start) {
                    return self.fail_step(format!("dropping the speculative MTP rows: {error}"));
                }
            }
            None => return self.fail_step("a verification retired on a sequence with no MTP cache".to_string()),
        }
        for token in &accepted.outputs {
            run.generated += 1;
            listener.token(*token, now);
        }
        let clock = state.admitted.unwrap_or(verifying.admitted);
        let drafted_at = state.drafted_at.unwrap_or(clock);
        let retired = verifying.retired_at.unwrap_or(now);
        listener.verification(Verification {
            accepted: accepted.accepted_drafts,
            drafted: drafts.len(),
            committed: accepted.count,
            verify_nanos: nanos(retired.saturating_duration_since(drafted_at)),
            draft_nanos: nanos(drafted_at.saturating_duration_since(clock)),
        });
        state.current = accepted.outputs.last().copied();
        state.mtp_position = verifying.start;
        state.replay = if accepted.count < verifying.rows {
            accepted.count as u32
        } else {
            0
        };
        run.position = verifying.start + accepted.count as u64;
        let ended = accepted.eos_at.is_some();
        state.catch_up = accepted.outputs;
        if ended {
            run.stop(FinishReason::Stop);
        } else if run.generated >= run.wanted {
            run.stop(FinishReason::Length);
        }
    }
}
