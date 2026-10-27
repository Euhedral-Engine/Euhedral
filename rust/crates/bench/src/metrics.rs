//! Converts the boundaries of one generation into a row's work, timings and throughput. Every duration comes from its own
//! pair of boundaries, as in the Java benchmark; the boundaries are the engine's listener events, taken on a worker the
//! moment the owner finds the event.
//!
//! - `tokenization`: generate entry to prompt encoded.
//! - `prefill`: admission of the first prefill quantum to retirement of the last; excludes tokenization and the first
//!   token. Rate: prompt tokens prefilled. The engine selects the first token with a device argmax inside the last
//!   prefill quantum, so this window includes that selection (and `firstTokenSample` is about 0); the Java session selects
//!   on the host after its last prefill quantum, outside this window.
//! - `firstTokenSample`: retirement of the last prefill quantum to the first token found.
//! - `timeToFirstToken`: entry to the first token found (tokenization, prefill, selection).
//! - `decode`: **the first token found to the last token found.** Excludes the first token (it comes from prefill) and
//!   everything after the last token (the engine ends a request when its last token is found; it runs no commit-only
//!   quantum). Rate: tokens 2..N over this window, so a window of `n - 1` intervals carries `n - 1` tokens. With
//!   speculative steps the window runs from the first token to the last step's retirement and the numerator is the
//!   tokens the steps committed.
//! - `decodeQuantaSum`: the sampling decode quanta's admission-to-retirement times summed. This engine keeps two quanta
//!   in flight, so the sum can exceed `decode`; it is the latency of the quanta, not the window. With steps: the
//!   verification quanta alone.
//! - `finalCommit`: always null; `finalCommitQuanta` is 0.
//! - `timeToLastToken`: entry to the last token found.
//! - `endToEnd`: entry to the end of the request, as the listener saw it. Rate: returned tokens over this window.

use std::collections::BTreeMap;
use std::time::{Duration, Instant};

use crate::config::{Scenario, ScenarioKind};
use crate::result::{DraftPhase, FAILED, INELIGIBLE, SUCCESS, Speculative, Throughput, Timings, Work, output_sha256};

/// What a quantum was, for the benchmark.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Phase {
    Prefill,
    /// A quantum that selected one token.
    Decode,
    /// A quantum of another kind (drafting, verification); counted nowhere unless it is a [`Step`].
    Other,
}

/// A quantum that retired for the request.
#[derive(Clone, Copy, Debug)]
pub struct QuantumRecord {
    pub phase: Phase,
    pub rows: usize,
    pub admitted: Instant,
    pub retired: Instant,
}

/// One speculative verification step.
#[derive(Clone, Copy, Debug)]
pub struct Step {
    pub start: Instant,
    pub executed: Instant,
    /// Tokens the step committed.
    pub outputs: usize,
    /// Drafts it accepted.
    pub accepted: usize,
    /// Why it rejected a draft: 0 the base token was in the draft head's shortlist, 1 it was not, -1 no rejection.
    pub rejection: i32,
}

/// How the request ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Ending {
    /// It produced what it could: its tokens, a terminator or the end of the context included.
    Completed,
    /// A quantum failed or the engine refused: the text says why.
    Failed(String),
}

/// The boundaries of one generation, filled by the listener's events.
#[derive(Clone, Debug)]
pub struct Recorder {
    /// Immediately before the prompt is tokenized and the request made.
    pub entry: Instant,
    /// When the prompt was encoded, and its token count.
    pub encoded: Option<(Instant, usize)>,
    pub quanta: Vec<QuantumRecord>,
    /// Every token found, in order, with the moment it was found.
    pub tokens: Vec<(u32, Instant)>,
    pub steps: Vec<Step>,
    /// Drafting quanta by phase: count and summed admission-to-retirement time.
    pub draft_phases: BTreeMap<String, (u64, Duration)>,
    pub finished: Option<(Instant, Ending)>,
}

impl Recorder {
    pub fn new(entry: Instant) -> Recorder {
        Recorder {
            entry,
            encoded: None,
            quanta: Vec::new(),
            tokens: Vec::new(),
            steps: Vec::new(),
            draft_phases: BTreeMap::new(),
            finished: None,
        }
    }
}

/// The row's measured part.
#[derive(Clone, Debug, PartialEq)]
pub struct Outcome {
    pub status: &'static str,
    pub reason: Option<String>,
    pub work: Work,
    pub timings: Option<Timings>,
    pub throughput: Option<Throughput>,
}

/// Nanoseconds from `from` to `to`, zero when `to` is earlier.
fn nanos(from: Instant, to: Instant) -> i64 {
    to.saturating_duration_since(from).as_nanos() as i64
}

/// Tokens per second over `nanos`, or `None` for no tokens or no time.
pub fn per_second(count: usize, nanos: Option<i64>) -> Option<f64> {
    match nanos {
        Some(nanos) if count > 0 && nanos > 0 => Some(count as f64 * 1e9 / nanos as f64),
        _ => None,
    }
}

/// Evaluates one generation against its scenario. `is_eos` says whether a token id ends generation.
pub fn evaluate(scenario: &Scenario, recorder: &Recorder, is_eos: impl Fn(u32) -> bool) -> Outcome {
    let prompt_tokens = recorder.encoded.map(|(_, count)| count);
    let prefill: Vec<&QuantumRecord> = recorder.quanta.iter().filter(|q| q.phase == Phase::Prefill).collect();
    let prefill_tokens: usize = prefill.iter().map(|q| q.rows).sum();
    let tokens: Vec<u32> = recorder.tokens.iter().map(|(id, _)| *id).collect();

    let failure = match &recorder.finished {
        None => Some("generation_did_not_return".to_string()),
        Some((_, Ending::Failed(reason))) => Some(reason.clone()),
        Some((_, Ending::Completed)) => None,
    };
    if let Some(reason) = failure {
        return Outcome {
            status: FAILED,
            reason: Some(reason),
            work: Work {
                prompt_tokens,
                prefill_quanta: Some(prefill.len()),
                prefill_tokens: Some(prefill_tokens),
                generated_tokens: None,
                decode_sampled_tokens: None,
                final_commit_quanta: None,
                eos_observed: None,
                speculative: None,
                output_sha256: None,
                prefix_restored_tokens: None,
            },
            timings: None,
            throughput: None,
        };
    }
    let finished_at = recorder.finished.as_ref().map_or(recorder.entry, |(at, _)| *at);

    let eos = tokens.last().is_some_and(|last| is_eos(*last));
    let first = recorder.tokens.first().map(|(_, at)| *at);
    let last = recorder.tokens.last().map(|(_, at)| *at);
    let decode_quanta: Vec<&QuantumRecord> = recorder.quanta.iter().filter(|q| q.phase == Phase::Decode).collect();

    // Tokens 2..N come from decode; with speculative steps, the tokens the steps committed.
    let mut sampled = tokens.len().saturating_sub(1);
    let mut decode_end = last;
    let mut decode_quanta_sum = (sampled > 0 && !decode_quanta.is_empty()).then(|| {
        decode_quanta
            .iter()
            .take(sampled)
            .map(|q| nanos(q.admitted, q.retired))
            .sum::<i64>()
    });
    let mut speculative = None;
    if !recorder.steps.is_empty() {
        let steps = &recorder.steps;
        let mut histogram = [0u64; 8];
        let (mut committed, mut verify, mut draft) = (0usize, 0i64, 0i64);
        let (mut in_shortlist, mut outside_shortlist) = (0, 0);
        for (index, step) in steps.iter().enumerate() {
            histogram[step.accepted.min(7)] += 1;
            committed += step.outputs;
            match step.rejection {
                0 => in_shortlist += 1,
                1 => outside_shortlist += 1,
                _ => {}
            }
            verify += nanos(step.start, step.executed);
            let before = index.checked_sub(1).map(|previous| steps[previous].executed).or(first);
            draft += before.map_or(0, |before| nanos(before, step.start));
        }
        let depth = histogram.iter().rposition(|count| *count > 0).unwrap_or(0);
        speculative = Some(Speculative {
            verifications: steps.len(),
            accepted_drafts: histogram[..(depth + 1).max(4)].to_vec(),
            committed_tokens: committed,
            verify_nanos: verify,
            draft_nanos: draft,
            rejections_in_shortlist: in_shortlist,
            rejections_outside_shortlist: outside_shortlist,
            draft_phases: recorder
                .draft_phases
                .iter()
                .map(|(phase, (quanta, time))| {
                    (
                        phase.clone(),
                        DraftPhase {
                            quanta: *quanta,
                            nanos: time.as_nanos() as i64,
                        },
                    )
                })
                .collect(),
        });
        sampled = committed;
        decode_end = steps.last().map(|step| step.executed);
        decode_quanta_sum = Some(verify);
    }

    let work = Work {
        prompt_tokens,
        prefill_quanta: Some(prefill.len()),
        prefill_tokens: Some(prefill_tokens),
        generated_tokens: Some(tokens.len()),
        decode_sampled_tokens: Some(sampled),
        final_commit_quanta: Some(0),
        eos_observed: Some(eos),
        speculative,
        output_sha256: Some(output_sha256(&tokens)),
        prefix_restored_tokens: None,
    };

    let prefill_time = match (prefill.first(), prefill.last()) {
        (Some(first_quantum), Some(last_quantum)) => Some(nanos(first_quantum.admitted, last_quantum.retired)),
        _ => None,
    };
    let first_sample = match (first, prefill.last()) {
        (Some(first), Some(last_quantum)) => Some(nanos(last_quantum.retired, first)),
        _ => None,
    };
    let decode_window = match (first, decode_end) {
        (Some(first), Some(end)) if sampled > 0 => Some(nanos(first, end)),
        _ => None,
    };
    let end_to_end = nanos(recorder.entry, finished_at);
    let timings = Timings {
        tokenization: recorder.encoded.map(|(at, _)| nanos(recorder.entry, at)),
        prefill: prefill_time,
        first_token_sample: first_sample,
        time_to_first_token: first.map(|first| nanos(recorder.entry, first)),
        decode: decode_window,
        decode_quanta_sum: decode_window.and(decode_quanta_sum),
        final_commit: None,
        time_to_last_token: if decode_window.is_some() {
            decode_end.map(|end| nanos(recorder.entry, end))
        } else {
            first.map(|first| nanos(recorder.entry, first))
        },
        end_to_end: Some(end_to_end),
        prefix_restore: None,
    };
    let throughput = Throughput {
        prefill_tokens_per_second: per_second(prefill_tokens, prefill_time),
        decode_tokens_per_second: per_second(sampled, decode_window),
        end_to_end_output_tokens_per_second: per_second(tokens.len(), Some(end_to_end)),
    };

    let requested = scenario.requested_new_tokens;
    let mut status = SUCCESS;
    let mut reason = None;
    if prompt_tokens.is_none() || prefill_tokens != prompt_tokens.unwrap_or(0) || prefill.is_empty() {
        status = FAILED;
        reason = Some("incomplete_prefill".to_string());
    } else {
        match scenario.kind {
            ScenarioKind::Prefill => {
                if !tokens.is_empty() {
                    status = FAILED;
                    reason = Some("inconsistent_generation_boundaries".to_string());
                }
            }
            ScenarioKind::FirstToken => {
                if tokens.len() != 1 {
                    status = FAILED;
                    reason = Some("no_first_token".to_string());
                }
            }
            ScenarioKind::PromptToN | ScenarioKind::SustainedDecode => {
                if tokens.is_empty() {
                    status = FAILED;
                    reason = Some("inconsistent_generation_boundaries".to_string());
                } else if eos {
                    status = INELIGIBLE;
                    reason = Some(format!("eos_after_{}_of_{requested}_tokens", tokens.len()));
                } else if tokens.len() < requested {
                    status = INELIGIBLE;
                    reason = Some(format!("stopped_after_{}_of_{requested}_tokens", tokens.len()));
                }
            }
        }
    }
    if status == FAILED {
        return Outcome {
            status,
            reason,
            work,
            timings: None,
            throughput: None,
        };
    }
    Outcome {
        status,
        reason,
        work,
        timings: Some(timings),
        throughput: Some(throughput),
    }
}
