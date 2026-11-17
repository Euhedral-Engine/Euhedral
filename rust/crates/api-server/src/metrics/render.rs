//! The Prometheus text format (0.0.4) of the registry, written by hand in a fixed order: the families sorted by name, each
//! with its `# HELP` and `# TYPE` lines, the samples of a family sorted by their labels, a histogram's buckets then its
//! count and sum, and its largest value as a family of its own, a gauge. Numbers are written as a Java `double` prints
//! (`1.0`, `1.5E-4`, `1.23456789012E11`); a histogram's bucket and count samples are whole numbers.

use std::fmt::Write;
use std::iter::repeat_n;
use std::sync::atomic::{AtomicU64, Ordering};

use super::buckets::{DURATION_BOUNDS_NS, RATE_BOUNDS};
use super::{APPLICATION, Api, ConstraintFailure, Durations, FinishReason, Metrics, Outcome, Rates};
use crate::backend::{EngineReport, PrefixCacheReport};

/// The content type of the text, as the route answers it.
pub const CONTENT_TYPE: &str = "text/plain;version=0.0.4;charset=utf-8";

const SECONDS: f64 = 1e9;

/// What a family is, as its `# TYPE` line says.
#[derive(Clone, Copy)]
enum Kind {
    Counter,
    Gauge,
    Histogram,
}

/// The text being written.
struct Page<'a> {
    out: &'a mut String,
}

/// Writes `value` as `\\`, `\n` (and with `quoted` also `\"`) escapes.
fn escaped(out: &mut String, text: &str, quoted: bool) {
    for ch in text.chars() {
        match ch {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '"' if quoted => out.push_str("\\\""),
            other => out.push(other),
        }
    }
}

/// Writes `value` the way a Java `double` prints; infinities and NaN as the text format spells them.
pub(super) fn push_double(out: &mut String, value: f64) {
    if value.is_nan() {
        out.push_str("NaN");
        return;
    }
    if value.is_infinite() {
        out.push_str(if value > 0.0 { "+Inf" } else { "-Inf" });
        return;
    }
    if value == 0.0 {
        out.push_str(if value.is_sign_negative() { "-0.0" } else { "0.0" });
        return;
    }
    if value < 0.0 {
        out.push('-');
    }
    let magnitude = value.abs();
    // The shortest digits that read back as the value, and the power of ten of the first.
    let mut scientific = String::new();
    let _ = write!(scientific, "{magnitude:e}");
    let (mantissa, exponent) = scientific.split_once('e').unwrap_or((&scientific, "0"));
    let exponent: i32 = exponent.parse().unwrap_or(0);
    let digits: String = mantissa.chars().filter(char::is_ascii_digit).collect();
    if (1e-3..1e7).contains(&magnitude) {
        if exponent >= 0 {
            let whole = exponent as usize + 1;
            if digits.len() <= whole {
                out.push_str(&digits);
                out.extend(repeat_n('0', whole - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..whole]);
                out.push('.');
                out.push_str(&digits[whole..]);
            }
        } else {
            out.push_str("0.");
            out.extend(repeat_n('0', (-exponent - 1) as usize));
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        out.push('.');
        out.push_str(if digits.len() > 1 { &digits[1..] } else { "0" });
        out.push('E');
        let _ = write!(out, "{exponent}");
    }
}

impl Page<'_> {
    /// The `# HELP` and `# TYPE` lines of a family.
    fn family(&mut self, name: &str, kind: Kind, help: &str) {
        self.out.push_str("# HELP ");
        self.out.push_str(name);
        self.out.push(' ');
        escaped(self.out, help, false);
        self.out.push_str("\n# TYPE ");
        self.out.push_str(name);
        self.out.push_str(match kind {
            Kind::Counter => " counter\n",
            Kind::Gauge => " gauge\n",
            Kind::Histogram => " histogram\n",
        });
    }

    /// The start of a sample: the name and its labels. `first` are the labels that sort before `application`, `last`
    /// those that sort after it, and `bound` the bucket's `le` (which is always the last).
    fn head(&mut self, name: &str, first: &[(&str, &str)], last: &[(&str, &str)], bound: Option<&str>) {
        self.out.push_str(name);
        self.out.push('{');
        for (key, value) in first {
            self.pair(key, value);
        }
        self.pair("application", APPLICATION);
        for (key, value) in last {
            self.pair(key, value);
        }
        if let Some(bound) = bound {
            self.out.push_str("le=\"");
            self.out.push_str(bound);
            self.out.push_str("\",");
        }
        self.out.pop();
        self.out.push_str("} ");
    }

    fn pair(&mut self, key: &str, value: &str) {
        self.out.push_str(key);
        self.out.push_str("=\"");
        escaped(self.out, value, true);
        self.out.push_str("\",");
    }

    /// A sample whose value is a `double`.
    fn sample(&mut self, name: &str, first: &[(&str, &str)], last: &[(&str, &str)], value: f64) {
        self.head(name, first, last, None);
        push_double(self.out, value);
        self.out.push('\n');
    }

    /// A sample whose value is a whole number.
    fn whole(&mut self, name: &str, first: &[(&str, &str)], last: &[(&str, &str)], bound: Option<&str>, value: u64) {
        self.head(name, first, last, bound);
        let _ = writeln!(self.out, "{value}");
    }

    /// A counter family of one sample without labels.
    fn counter(&mut self, name: &str, help: &str, value: f64) {
        self.family(name, Kind::Counter, help);
        self.sample(name, &[], &[], value);
    }

    /// A gauge family of one sample without labels.
    fn gauge(&mut self, name: &str, help: &str, value: f64) {
        self.family(name, Kind::Gauge, help);
        self.sample(name, &[], &[], value);
    }

    /// A counter family with one sample per `api`.
    fn per_api(&mut self, name: &str, help: &str, values: &[AtomicU64]) {
        self.family(name, Kind::Counter, help);
        for api in Api::ALL {
            self.sample(name, &[("api", api.label())], &[], load(&values[api.index()]));
        }
    }

    /// The duration histograms of every `api` (`name` is the family's, in seconds) and the family of their largest.
    fn durations(&mut self, name: &str, help: &str, series: &[Durations], now_ms: i64) {
        self.family(name, Kind::Histogram, help);
        let bucket = format!("{name}_bucket");
        let count = format!("{name}_count");
        let sum = format!("{name}_sum");
        let mut bound = String::new();
        for api in Api::ALL {
            let labels = [("api", api.label())];
            let each = &series[api.index()];
            let mut seen = 0;
            for (at, upper) in DURATION_BOUNDS_NS.iter().enumerate() {
                seen += each.buckets[at].load(Ordering::Relaxed);
                bound.clear();
                push_double(&mut bound, *upper as f64 / SECONDS);
                self.whole(&bucket, &labels, &[], Some(&bound), seen);
            }
            seen += each.buckets[DURATION_BOUNDS_NS.len()].load(Ordering::Relaxed);
            self.whole(&bucket, &labels, &[], Some("+Inf"), seen);
            self.whole(&count, &labels, &[], None, seen);
            self.sample(&sum, &labels, &[], each.sum_ns.load(Ordering::Relaxed) as f64 / SECONDS);
        }
        let max = format!("{name}_max");
        self.family(&max, Kind::Gauge, help);
        for api in Api::ALL {
            let nanos = series[api.index()].max_ns.poll(now_ms);
            self.sample(&max, &[("api", api.label())], &[], nanos as f64 / SECONDS);
        }
    }

    /// The rate histogram and the family of its largest value.
    fn rates(&mut self, name: &str, help: &str, series: &Rates, now_ms: i64) {
        self.family(name, Kind::Histogram, help);
        let bucket = format!("{name}_bucket");
        let mut bound = String::new();
        let mut seen = 0;
        for (at, upper) in RATE_BOUNDS.iter().enumerate() {
            seen += series.buckets[at].load(Ordering::Relaxed);
            bound.clear();
            push_double(&mut bound, *upper);
            self.whole(&bucket, &[], &[], Some(&bound), seen);
        }
        seen += series.buckets[RATE_BOUNDS.len()].load(Ordering::Relaxed);
        self.whole(&bucket, &[], &[], Some("+Inf"), seen);
        self.whole(&format!("{name}_count"), &[], &[], None, seen);
        self.sample(
            &format!("{name}_sum"),
            &[],
            &[],
            f64::from_bits(series.sum.load(Ordering::Relaxed)),
        );
        let max = format!("{name}_max");
        self.family(&max, Kind::Gauge, help);
        self.sample(&max, &[], &[], f64::from_bits(series.max.poll(now_ms)));
    }
}

fn load(counter: &AtomicU64) -> f64 {
    counter.load(Ordering::Relaxed) as f64
}

impl Metrics {
    /// The text of every series now. `model` and `context_tokens` are the served model's; `engine` is what the engine
    /// reports at this scrape.
    pub fn render(&self, model: &str, context_tokens: usize, engine: &EngineReport, now_ms: i64) -> String {
        let mut text = String::with_capacity(96 * 1024);
        let mut page = Page { out: &mut text };
        // The families run in the order of their names, a counter's `_total` aside.
        self.before_engine(&mut page, now_ms);
        engine_state(&mut page, model, context_tokens, engine);
        self.finishes_and_queue(&mut page, now_ms);
        if let Some(cache) = &engine.prefix_cache {
            prefix_cache(&mut page, cache);
        }
        self.tokens_and_timings(&mut page, now_ms);
        self.requests_and_speculation(&mut page, now_ms);
        text
    }

    fn before_engine(&self, page: &mut Page<'_>, now_ms: i64) {
        page.counter(
            "euhedral_completion_tokens_total",
            "Generated tokens, reasoning included",
            load(&self.completion_tokens),
        );
        let name = "euhedral_constrained_failures_total";
        page.family(name, Kind::Counter, "Constrained outputs that failed their final check");
        for failure in ConstraintFailure::BY_LABEL {
            let value = load(&self.constrained_failures[failure.index()]);
            page.sample(name, &[], &[("reason", failure.label())], value);
        }
        page.rates(
            "euhedral_decode_tokens_per_second",
            "Generated tokens per second after the first",
            &self.decode_rate,
            now_ms,
        );
    }

    fn finishes_and_queue(&self, page: &mut Page<'_>, now_ms: i64) {
        let name = "euhedral_finishes_total";
        page.family(name, Kind::Counter, "Generations by finish reason");
        for api in Api::ALL {
            for reason in FinishReason::BY_LABEL {
                let value = load(&self.finishes[api.index()][reason.index()]);
                page.sample(name, &[("api", api.label())], &[("reason", reason.label())], value);
            }
        }
        page.gauge(
            "euhedral_generations_active",
            "Generations running",
            self.active.load(Ordering::Relaxed) as f64,
        );
        page.gauge(
            "euhedral_generations_queued",
            "Requests waiting for the generation slot",
            self.queued.load(Ordering::Relaxed) as f64,
        );
        page.rates(
            "euhedral_prefill_tokens_per_second",
            "Prefilled tokens per second up to the first token",
            &self.prefill_rate,
            now_ms,
        );
    }

    fn tokens_and_timings(&self, page: &mut Page<'_>, now_ms: i64) {
        page.counter(
            "euhedral_prompt_cached_tokens_total",
            "Prompt tokens restored from the prefix cache",
            load(&self.prompt_cached_tokens),
        );
        page.counter(
            "euhedral_prompt_prefilled_tokens_total",
            "Prompt tokens prefilled",
            load(&self.prompt_prefilled_tokens),
        );
        page.counter(
            "euhedral_prompt_tokens_total",
            "Prompt tokens",
            load(&self.prompt_tokens),
        );
        page.durations(
            "euhedral_queue_wait_seconds",
            "From arrival to the start of generation",
            &self.queue_wait,
            now_ms,
        );
        page.counter(
            "euhedral_reasoning_tokens_total",
            "Generated reasoning tokens",
            load(&self.reasoning_tokens),
        );
        page.durations(
            "euhedral_request_duration_seconds",
            "From arrival to the last token",
            &self.request_duration,
            now_ms,
        );
    }

    fn requests_and_speculation(&self, page: &mut Page<'_>, now_ms: i64) {
        let name = "euhedral_requests_total";
        page.family(name, Kind::Counter, "Requests by API surface and outcome");
        for api in Api::ALL {
            for outcome in Outcome::BY_LABEL {
                let value = load(&self.requests[api.index()][outcome.index()]);
                page.sample(name, &[("api", api.label())], &[("outcome", outcome.label())], value);
            }
        }
        page.per_api(
            "euhedral_schema_rejections_total",
            "Requests refused because a schema cannot be enforced",
            &self.schema_rejections,
        );
        let name = "euhedral_speculative_accepted_total";
        page.family(
            name,
            Kind::Counter,
            "Verification steps by the number of drafted tokens accepted",
        );
        let mut drafts = String::new();
        for at in &self.drafts_by_label {
            drafts.clear();
            let _ = write!(drafts, "{at}");
            page.sample(name, &[], &[("drafts", &drafts)], load(&self.speculative_accepted[*at]));
        }
        page.counter(
            "euhedral_speculative_accepted_tokens_total",
            "Drafted tokens accepted",
            load(&self.speculative_accepted_tokens),
        );
        page.counter(
            "euhedral_speculative_verifications_total",
            "Speculative verification steps",
            load(&self.speculative_verifications),
        );
        page.durations(
            "euhedral_time_to_first_token_seconds",
            "From the start of generation to the first token",
            &self.time_to_first_token,
            now_ms,
        );
        page.per_api("euhedral_tool_calls_total", "Tool calls returned", &self.tool_calls);
    }
}

/// The engine's own gauges, from the engine's report.
fn engine_state(page: &mut Page<'_>, model: &str, context_tokens: usize, engine: &EngineReport) {
    let name = "euhedral_engine_info";
    page.family(name, Kind::Gauge, "The served model and artifact");
    let depth = engine.speculative_depth.to_string();
    let labels = [
        ("artifact", engine.artifact.as_deref().unwrap_or("unknown")),
        ("model", model),
        ("speculation", engine.speculation.as_str()),
        ("speculative_depth", depth.as_str()),
    ];
    page.head(name, &[], &labels, None);
    page.out.push_str("1\n");
    page.gauge(
        "euhedral_engine_context_tokens",
        "Longest prompt plus completion a request may use",
        context_tokens as f64,
    );
    page.gauge(
        "euhedral_engine_device_allocated_bytes",
        "Device memory the engine allocated: weights, KV cache, sequence state, workspaces",
        engine.allocated_device_bytes as f64,
    );
    page.gauge(
        "euhedral_engine_device_free_bytes",
        "Free device memory reported by the driver",
        engine.free_device_bytes as f64,
    );
    page.gauge(
        "euhedral_engine_host_backed_weight_bytes",
        "Weights held in pinned host memory and streamed to the device",
        engine.host_backed_weight_bytes as f64,
    );
    page.gauge(
        "euhedral_engine_pinned_pool_bytes",
        "Released pinned staging buffers kept for reuse",
        engine.pinned_pool_bytes as f64,
    );
    page.counter(
        "euhedral_engine_pinned_pool_overflows_total",
        "Released pinned staging buffers the pool had no room for",
        engine.pinned_pool_overflows as f64,
    );
    page.gauge(
        "euhedral_engine_workers",
        "Worker CPUs of the engine's lattice",
        engine.workers as f64,
    );
}

/// The prefix cache's families, which exist only for an engine that has the cache.
fn prefix_cache(page: &mut Page<'_>, cache: &PrefixCacheReport) {
    page.gauge(
        "euhedral_prefix_capacity_bytes",
        "Pinned arena bytes",
        cache.capacity_bytes as f64,
    );
    page.counter(
        "euhedral_prefix_capture_seconds_total",
        "Time spent storing checkpoints",
        cache.capture_nanos as f64 / SECONDS,
    );
    page.counter(
        "euhedral_prefix_captures_total",
        "Checkpoints stored",
        cache.captures as f64,
    );
    page.counter(
        "euhedral_prefix_evictions_total",
        "Checkpoints evicted",
        cache.evictions as f64,
    );
    page.counter(
        "euhedral_prefix_failed_total",
        "Checkpoints whose copies failed",
        cache.failed as f64,
    );
    page.counter(
        "euhedral_prefix_hits_total",
        "Prefix cache lookups that restored a prefix",
        cache.hits as f64,
    );
    page.counter(
        "euhedral_prefix_lookups_total",
        "Prefix cache lookups",
        cache.lookups as f64,
    );
    page.gauge("euhedral_prefix_nodes", "Checkpoints held", cache.nodes as f64);
    page.counter(
        "euhedral_prefix_restore_seconds_total",
        "Time spent restoring prefixes",
        cache.restore_nanos as f64 / SECONDS,
    );
    page.counter(
        "euhedral_prefix_restored_tokens_total",
        "Prompt tokens restored instead of prefilled",
        cache.restored_tokens as f64,
    );
    page.counter(
        "euhedral_prefix_restores_total",
        "Restores completed",
        cache.restores as f64,
    );
    page.counter(
        "euhedral_prefix_skipped_total",
        "Checkpoints not stored for lack of room",
        cache.skipped as f64,
    );
    page.gauge(
        "euhedral_prefix_used_bytes",
        "Pinned arena bytes in use",
        cache.used_bytes as f64,
    );
}
