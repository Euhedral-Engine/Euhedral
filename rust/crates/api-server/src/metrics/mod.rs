//! The server's own metrics: the series of a request and of the engine, recorded with atomics and rendered as
//! Prometheus text (format 0.0.4) at `/metrics`.
//!
//! Recording takes no lock, allocates nothing and never waits: every series is a fixed set of atomic words, laid out when
//! the registry is made. Rendering reads the words one by one, so a series that changes during a scrape is read at one
//! moment or the next. A series is exported with its value from the start (zero before anything is recorded).
//!
//! The series, their names, labels, types and help are those of the server this one replaces; the process's own series
//! (virtual machine, process, HTTP server, logging) are not part of it.

mod buckets;
mod http;
pub mod process;
mod render;
mod window;

use std::array::from_fn;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use buckets::{DURATION_BOUNDS_NS, RATE_BOUNDS};

pub use http::{RequestTimer, Uri, outcome};
pub use render::CONTENT_TYPE;
pub use window::{WINDOW_MS, WINDOWS, WindowMax};

/// The value of the `application` label every series carries.
pub const APPLICATION: &str = "euhedral-inference";

/// The API surface a request came in on, the `api` label. The order is the order of the label values as text.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Api {
    ChatCompletions,
    Messages,
    Responses,
}

impl Api {
    pub const ALL: [Api; 3] = [Api::ChatCompletions, Api::Messages, Api::Responses];

    /// The label value.
    pub fn label(self) -> &'static str {
        match self {
            Api::ChatCompletions => "chat_completions",
            Api::Messages => "messages",
            Api::Responses => "responses",
        }
    }

    fn index(self) -> usize {
        self as usize
    }
}

/// How a request ended, the `outcome` label.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Outcome {
    /// Answered (any finish reason).
    Success,
    /// Refused while planning: invalid, unsupported, or over the context.
    Invalid,
    /// Refused for capacity or shutdown.
    Rejected,
    /// The client left or the request timed out.
    Cancelled,
    /// The server or the model's output failed.
    Failed,
}

impl Outcome {
    /// In the order of the label values as text.
    const BY_LABEL: [Outcome; 5] = [
        Outcome::Cancelled,
        Outcome::Failed,
        Outcome::Invalid,
        Outcome::Rejected,
        Outcome::Success,
    ];

    /// The label value.
    pub fn label(self) -> &'static str {
        match self {
            Outcome::Success => "success",
            Outcome::Invalid => "invalid",
            Outcome::Rejected => "rejected",
            Outcome::Cancelled => "cancelled",
            Outcome::Failed => "failed",
        }
    }

    fn index(self) -> usize {
        Outcome::BY_LABEL.iter().position(|each| *each == self).unwrap_or(0)
    }
}

/// Why a generation finished, the `reason` label of the finishes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FinishReason {
    /// The model ended its turn.
    End,
    /// A stop string matched.
    StopSequence,
    /// The completion budget ran out.
    Length,
    /// The answer is tool calls.
    ToolCalls,
}

impl FinishReason {
    /// In the order of the label values as text.
    const BY_LABEL: [FinishReason; 4] = [
        FinishReason::End,
        FinishReason::Length,
        FinishReason::StopSequence,
        FinishReason::ToolCalls,
    ];

    /// The label value.
    pub fn label(self) -> &'static str {
        match self {
            FinishReason::End => "end",
            FinishReason::StopSequence => "stop_sequence",
            FinishReason::Length => "length",
            FinishReason::ToolCalls => "tool_calls",
        }
    }

    fn index(self) -> usize {
        FinishReason::BY_LABEL
            .iter()
            .position(|each| *each == self)
            .unwrap_or(0)
    }
}

/// The check a finished output failed, the `reason` label of the constrained failures.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ConstraintFailure {
    InvalidToolCall,
    InvalidStructuredOutput,
}

impl ConstraintFailure {
    /// In the order of the label values as text.
    const BY_LABEL: [ConstraintFailure; 2] = [
        ConstraintFailure::InvalidStructuredOutput,
        ConstraintFailure::InvalidToolCall,
    ];

    /// The label value (the error code of the refusal).
    pub fn label(self) -> &'static str {
        match self {
            ConstraintFailure::InvalidToolCall => "invalid_tool_call",
            ConstraintFailure::InvalidStructuredOutput => "invalid_structured_output",
        }
    }

    fn index(self) -> usize {
        match self {
            ConstraintFailure::InvalidStructuredOutput => 0,
            ConstraintFailure::InvalidToolCall => 1,
        }
    }
}

/// One answered generation: its finish, token counts and the instants of its stages, in nanoseconds on one monotonic
/// clock (only their differences are used). `first_token_ns` is 0 when it produced no token.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Generation {
    pub api: Api,
    pub reason: FinishReason,
    pub prompt_tokens: u64,
    pub cached_prompt_tokens: u64,
    pub completion_tokens: u64,
    /// The generated tokens that were reasoning; `None` without a think block.
    pub reasoning_tokens: Option<u64>,
    /// When the request arrived.
    pub queued_ns: u64,
    /// When generation started.
    pub started_ns: u64,
    /// When the first token came; 0 for none.
    pub first_token_ns: u64,
    /// When the last token came.
    pub ended_ns: u64,
}

/// A duration histogram with the sum and the largest value, in nanoseconds. Each bucket counts the values between its
/// bound and the one before; the last counts those beyond the largest bound.
#[derive(Debug)]
struct Durations {
    buckets: [AtomicU64; DURATION_BOUNDS_NS.len() + 1],
    sum_ns: AtomicU64,
    max_ns: WindowMax,
}

impl Durations {
    fn new(now_ms: i64) -> Durations {
        Durations {
            buckets: from_fn(|_| AtomicU64::new(0)),
            sum_ns: AtomicU64::new(0),
            max_ns: WindowMax::new(now_ms),
        }
    }

    fn record(&self, nanos: u64, now_ms: i64) {
        let at = DURATION_BOUNDS_NS.partition_point(|bound| *bound < nanos);
        self.buckets[at].fetch_add(1, Ordering::Relaxed);
        self.sum_ns.fetch_add(nanos, Ordering::Relaxed);
        self.max_ns.record(nanos, now_ms);
    }
}

/// A histogram of rates (tokens per second) with the sum and the largest value, both kept as the bits of an `f64`.
#[derive(Debug)]
struct Rates {
    buckets: [AtomicU64; RATE_BOUNDS.len() + 1],
    sum: AtomicU64,
    max: WindowMax,
}

impl Rates {
    fn new(now_ms: i64) -> Rates {
        Rates {
            buckets: from_fn(|_| AtomicU64::new(0)),
            sum: AtomicU64::new(0f64.to_bits()),
            max: WindowMax::new(now_ms),
        }
    }

    fn record(&self, value: f64, now_ms: i64) {
        if value.is_nan() || value < 0.0 {
            return;
        }
        let at = RATE_BOUNDS.partition_point(|bound| *bound < value);
        self.buckets[at].fetch_add(1, Ordering::Relaxed);
        let added = |bits: u64| Some((f64::from_bits(bits) + value).to_bits());
        let _ = self.sum.try_update(Ordering::Relaxed, Ordering::Relaxed, added);
        // The bits of a non-negative `f64` order as the values do.
        self.max.record(value.to_bits(), now_ms);
    }
}

/// The registry of the series of a request and of the engine.
#[derive(Debug)]
pub struct Metrics {
    active: AtomicUsize,
    queued: AtomicUsize,
    requests: [[AtomicU64; Outcome::BY_LABEL.len()]; Api::ALL.len()],
    schema_rejections: [AtomicU64; Api::ALL.len()],
    constrained_failures: [AtomicU64; ConstraintFailure::BY_LABEL.len()],
    tool_calls: [AtomicU64; Api::ALL.len()],
    finishes: [[AtomicU64; FinishReason::BY_LABEL.len()]; Api::ALL.len()],
    prompt_tokens: AtomicU64,
    prompt_cached_tokens: AtomicU64,
    prompt_prefilled_tokens: AtomicU64,
    completion_tokens: AtomicU64,
    reasoning_tokens: AtomicU64,
    request_duration: [Durations; Api::ALL.len()],
    queue_wait: [Durations; Api::ALL.len()],
    time_to_first_token: [Durations; Api::ALL.len()],
    prefill_rate: Rates,
    decode_rate: Rates,
    speculative_verifications: AtomicU64,
    /// Verification steps by the number of drafted tokens accepted, `0` up to the speculative depth.
    speculative_accepted: Box<[AtomicU64]>,
    /// The indices of `speculative_accepted` in the order of their labels as text.
    drafts_by_label: Box<[usize]>,
    speculative_accepted_tokens: AtomicU64,
    http: http::Table,
    /// The process's CPU time and uptime, in nanoseconds, at the last scrape.
    last_cpu_ns: AtomicU64,
    last_wall_ns: AtomicU64,
}

impl Metrics {
    /// A registry with every series at zero, made at `now_ms` (the wall clock in milliseconds, which the largest values
    /// of the histograms are windowed by). `speculative_depth` is the tokens drafted per verification (0 without
    /// speculation): the accepted counts are exported for 0 up to that many drafts.
    pub fn new(speculative_depth: usize, now_ms: i64) -> Metrics {
        let counters = |count: usize| -> Box<[AtomicU64]> { (0..count).map(|_| AtomicU64::new(0)).collect() };
        let mut drafts_by_label: Vec<usize> = (0..=speculative_depth).collect();
        drafts_by_label.sort_by_key(ToString::to_string);
        Metrics {
            active: AtomicUsize::new(0),
            queued: AtomicUsize::new(0),
            requests: from_fn(|_| from_fn(|_| AtomicU64::new(0))),
            schema_rejections: from_fn(|_| AtomicU64::new(0)),
            constrained_failures: from_fn(|_| AtomicU64::new(0)),
            tool_calls: from_fn(|_| AtomicU64::new(0)),
            finishes: from_fn(|_| from_fn(|_| AtomicU64::new(0))),
            prompt_tokens: AtomicU64::new(0),
            prompt_cached_tokens: AtomicU64::new(0),
            prompt_prefilled_tokens: AtomicU64::new(0),
            completion_tokens: AtomicU64::new(0),
            reasoning_tokens: AtomicU64::new(0),
            request_duration: from_fn(|_| Durations::new(now_ms)),
            queue_wait: from_fn(|_| Durations::new(now_ms)),
            time_to_first_token: from_fn(|_| Durations::new(now_ms)),
            prefill_rate: Rates::new(now_ms),
            decode_rate: Rates::new(now_ms),
            speculative_verifications: AtomicU64::new(0),
            speculative_accepted: counters(speculative_depth + 1),
            drafts_by_label: drafts_by_label.into_boxed_slice(),
            speculative_accepted_tokens: AtomicU64::new(0),
            http: http::Table::new(now_ms),
            last_cpu_ns: AtomicU64::new(0),
            last_wall_ns: AtomicU64::new(0),
        }
    }

    /// The generations running and the requests waiting for a slot, as they are now.
    pub fn set_queue(&self, active: usize, queued: usize) {
        self.active.store(active, Ordering::Relaxed);
        self.queued.store(queued, Ordering::Relaxed);
    }

    /// The wall clock now in milliseconds since the epoch, for the windows of the largest values.
    pub fn wall_clock_ms() -> i64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| i64::try_from(since.as_millis()).unwrap_or(i64::MAX))
    }

    /// A request that ended, or was refused, with `outcome`.
    pub fn request(&self, api: Api, outcome: Outcome) {
        self.requests[api.index()][outcome.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// A request whose JSON schema or grammar could not be enforced.
    pub fn schema_rejected(&self, api: Api) {
        self.schema_rejections[api.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// A finished output that failed its constraint's check.
    pub fn constrained_failure(&self, failure: ConstraintFailure) {
        self.constrained_failures[failure.index()].fetch_add(1, Ordering::Relaxed);
    }

    /// Tool calls returned in an answer; none is nothing.
    pub fn tool_calls(&self, api: Api, calls: u64) {
        if calls > 0 {
            self.tool_calls[api.index()].fetch_add(calls, Ordering::Relaxed);
        }
    }

    /// One answered generation: its finish, token counts and timings. A stage that ran backwards on the clock records no
    /// duration.
    pub fn generation(&self, generation: &Generation, now_ms: i64) {
        let api = generation.api.index();
        self.finishes[api][generation.reason.index()].fetch_add(1, Ordering::Relaxed);
        let prefilled = generation.prompt_tokens.saturating_sub(generation.cached_prompt_tokens);
        self.prompt_tokens
            .fetch_add(generation.prompt_tokens, Ordering::Relaxed);
        self.prompt_cached_tokens
            .fetch_add(generation.cached_prompt_tokens, Ordering::Relaxed);
        self.prompt_prefilled_tokens.fetch_add(prefilled, Ordering::Relaxed);
        self.completion_tokens
            .fetch_add(generation.completion_tokens, Ordering::Relaxed);
        if let Some(reasoning) = generation.reasoning_tokens {
            self.reasoning_tokens.fetch_add(reasoning, Ordering::Relaxed);
        }
        if let Some(total) = generation.ended_ns.checked_sub(generation.queued_ns) {
            self.request_duration[api].record(total, now_ms);
        }
        if let Some(waited) = generation.started_ns.checked_sub(generation.queued_ns) {
            self.queue_wait[api].record(waited, now_ms);
        }
        if generation.first_token_ns == 0 {
            return;
        }
        let to_first = generation.first_token_ns.checked_sub(generation.started_ns);
        if let Some(to_first) = to_first {
            self.time_to_first_token[api].record(to_first, now_ms);
            if prefilled > 0 && to_first > 0 {
                self.prefill_rate
                    .record(prefilled as f64 * 1e9 / to_first as f64, now_ms);
            }
        }
        let decoding = generation.ended_ns.saturating_sub(generation.first_token_ns);
        if generation.completion_tokens > 1 && decoding > 0 {
            self.decode_rate.record(
                (generation.completion_tokens - 1) as f64 * 1e9 / decoding as f64,
                now_ms,
            );
        }
    }

    /// One speculative verification that accepted `accepted` of its drafted tokens. More than the depth counts as the
    /// depth.
    pub fn speculative_step(&self, accepted: usize) {
        self.speculative_verifications.fetch_add(1, Ordering::Relaxed);
        let at = accepted.min(self.speculative_accepted.len() - 1);
        self.speculative_accepted[at].fetch_add(1, Ordering::Relaxed);
        self.speculative_accepted_tokens.fetch_add(at as u64, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests;
