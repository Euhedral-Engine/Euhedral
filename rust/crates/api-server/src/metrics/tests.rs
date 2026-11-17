use super::render::push_double;
use super::*;
use crate::backend::{EngineReport, PrefixCacheReport};

const RECORDED: &str = include_str!("../../tests/data/metrics_recorded.txt");
const IDLE: &str = include_str!("../../tests/data/metrics_idle.txt");

fn double(value: f64) -> String {
    let mut out = String::new();
    push_double(&mut out, value);
    out
}

/// The engine of the fixtures: speculating three tokens deep, with a prefix cache.
fn report(full: bool) -> EngineReport {
    let mut report = EngineReport {
        artifact: Some("art.edrl".to_owned()),
        speculation: "mtp".to_owned(),
        speculative_depth: 3,
        ..EngineReport::default()
    };
    if full {
        report.workers = 7;
        report.allocated_device_bytes = 1_234_567_890_123;
        report.free_device_bytes = 98_765_432_100;
        report.host_backed_weight_bytes = 555;
        report.pinned_pool_bytes = 777;
        report.pinned_pool_overflows = 9;
        report.prefix_cache = Some(PrefixCacheReport {
            lookups: 100,
            hits: 60,
            restored_tokens: 4096,
            captures: 30,
            skipped: 3,
            failed: 1,
            evictions: 12,
            restores: 59,
            capture_nanos: 1_500_000_000,
            restore_nanos: 250_000_000,
            nodes: 17,
            used_bytes: 1 << 30,
            capacity_bytes: 1 << 33,
        });
    } else {
        report.prefix_cache = Some(PrefixCacheReport::default());
    }
    report
}

/// The same recordings the reference text was made from.
fn record_everything(metrics: &Metrics) {
    for (api_at, api) in Api::ALL.into_iter().enumerate() {
        for (outcome_at, outcome) in [
            Outcome::Success,
            Outcome::Invalid,
            Outcome::Rejected,
            Outcome::Cancelled,
            Outcome::Failed,
        ]
        .into_iter()
        .enumerate()
        {
            for _ in 0..(api_at + 1) * (outcome_at + 1) {
                metrics.request(api, outcome);
            }
        }
        for _ in 0..=api_at {
            metrics.schema_rejected(api);
        }
        metrics.tool_calls(api, api_at as u64 + 2);
        for (at, reason) in [
            FinishReason::End,
            FinishReason::StopSequence,
            FinishReason::Length,
            FinishReason::ToolCalls,
        ]
        .into_iter()
        .enumerate()
        {
            let (api_at, at) = (api_at as u64, at as u64);
            let queued = 1000;
            let started = queued + 12_345_678 * (api_at + 1);
            let first = started + 87_654_321 * (at + 1);
            metrics.generation(
                &Generation {
                    api,
                    reason,
                    prompt_tokens: 50 + 7 * api_at + at,
                    cached_prompt_tokens: 10 + at,
                    completion_tokens: 20 + 3 * api_at + at,
                    reasoning_tokens: (at % 2 == 0).then_some(5 + api_at),
                    queued_ns: queued,
                    started_ns: started,
                    first_token_ns: first,
                    ended_ns: first + 1_234_567_890 * (at + 1),
                },
                0,
            );
        }
    }
    metrics.constrained_failure(ConstraintFailure::InvalidToolCall);
    metrics.constrained_failure(ConstraintFailure::InvalidToolCall);
    metrics.constrained_failure(ConstraintFailure::InvalidStructuredOutput);
    for accepted in 0..=3 {
        for _ in 0..=accepted {
            metrics.speculative_step(accepted);
        }
    }
    metrics.set_queue(2, 5);
}

#[test]
fn a_registry_that_recorded_nothing_exports_every_series_at_zero() {
    let metrics = Metrics::new(3, 0);
    assert_eq!(metrics.render("m1", 4096, &report(false), 0), IDLE);
}

#[test]
fn recordings_are_exported_as_the_reference_text_has_them() {
    let metrics = Metrics::new(3, 0);
    record_everything(&metrics);
    assert_eq!(metrics.render("m1", 4096, &report(true), 0), RECORDED);
}

#[test]
fn an_engine_without_a_prefix_cache_or_artifact_exports_neither() {
    let text = Metrics::new(0, 0).render("m", 8, &EngineReport::default(), 0);
    assert!(text.contains(
        "euhedral_engine_info{application=\"euhedral-inference\",artifact=\"unknown\",model=\"m\",speculation=\"none\",speculative_depth=\"0\"} 1\n"
    ));
    assert!(!text.contains("euhedral_prefix_"));
    assert!(
        text.contains("euhedral_speculative_accepted_total{application=\"euhedral-inference\",drafts=\"0\"} 0.0\n")
    );
    assert!(!text.contains("drafts=\"1\""));
}

#[test]
fn labels_and_help_are_escaped() {
    let text = Metrics::new(0, 0).render("a\"b\\c\nd", 1, &EngineReport::default(), 0);
    assert!(text.contains("model=\"a\\\"b\\\\c\\nd\""), "{text}");
}

#[test]
fn numbers_print_as_a_java_double_does() {
    let table = [
        (0.0, "0.0"),
        (1.0, "1.0"),
        (-2.5, "-2.5"),
        (0.001, "0.001"),
        (0.0001, "1.0E-4"),
        (0.011184809, "0.011184809"),
        (9_999_999.0, "9999999.0"),
        (10_000_000.0, "1.0E7"),
        (123_456_789.0, "1.23456789E8"),
        (1.234567890123e12, "1.234567890123E12"),
        (98_765_432_100.0, "9.87654321E10"),
        (116.03250105589575, "116.03250105589575"),
        (600.0, "600.0"),
        (1e-7, "1.0E-7"),
        (f64::INFINITY, "+Inf"),
        (f64::NAN, "NaN"),
    ];
    for (value, text) in table {
        assert_eq!(double(value), text, "{value:e}");
    }
}

/// The value of the sample of `name` whose labels contain every fragment.
fn value(text: &str, name: &str, labels: &[&str]) -> f64 {
    text.lines()
        .filter(|line| !line.starts_with('#'))
        .filter_map(|line| line.rsplit_once(' '))
        .filter(|(head, _)| {
            let (found, with) = head.split_once('{').unwrap_or((head, ""));
            found == name && labels.iter().all(|label| with.contains(label))
        })
        .map(|(_, value)| value.parse::<f64>().unwrap_or(f64::NAN))
        .next()
        .unwrap_or_else(|| panic!("no sample of {name} {labels:?}"))
}

#[test]
fn refusals_and_failures_are_counted_by_api_and_outcome() {
    let metrics = Metrics::new(0, 0);
    metrics.request(Api::ChatCompletions, Outcome::Success);
    metrics.request(Api::Messages, Outcome::Success);
    metrics.request(Api::ChatCompletions, Outcome::Invalid);
    metrics.request(Api::ChatCompletions, Outcome::Invalid);
    metrics.request(Api::ChatCompletions, Outcome::Failed);
    metrics.schema_rejected(Api::ChatCompletions);
    metrics.constrained_failure(ConstraintFailure::InvalidToolCall);
    let text = metrics.render("m", 1, &EngineReport::default(), 0);
    let requests = |api: &str, outcome: &str| {
        value(
            &text,
            "euhedral_requests_total",
            &[&format!("api=\"{api}\""), &format!("outcome=\"{outcome}\"")],
        )
    };
    assert_eq!(requests("chat_completions", "success"), 1.0);
    assert_eq!(requests("messages", "success"), 1.0);
    assert_eq!(requests("chat_completions", "invalid"), 2.0);
    assert_eq!(requests("chat_completions", "failed"), 1.0);
    assert_eq!(requests("responses", "rejected"), 0.0);
    assert_eq!(
        value(&text, "euhedral_schema_rejections_total", &["api=\"chat_completions\""]),
        1.0
    );
    assert_eq!(
        value(
            &text,
            "euhedral_constrained_failures_total",
            &["reason=\"invalid_tool_call\""]
        ),
        1.0
    );
    assert_eq!(value(&text, "euhedral_generations_active", &[]), 0.0);
    assert_eq!(value(&text, "euhedral_generations_queued", &[]), 0.0);
    metrics.set_queue(1, 4);
    let text = metrics.render("m", 1, &EngineReport::default(), 0);
    assert_eq!(value(&text, "euhedral_generations_active", &[]), 1.0);
    assert_eq!(value(&text, "euhedral_generations_queued", &[]), 4.0);
}

fn generation(api: Api, first_token_ns: u64, completion_tokens: u64) -> Generation {
    Generation {
        api,
        reason: FinishReason::End,
        prompt_tokens: 100,
        cached_prompt_tokens: 40,
        completion_tokens,
        reasoning_tokens: None,
        queued_ns: 1_000,
        started_ns: 2_001_000,
        first_token_ns,
        ended_ns: 1_003_001_000,
    }
}

#[test]
fn a_generation_without_a_token_records_no_first_token_and_no_rates() {
    let metrics = Metrics::new(0, 0);
    metrics.generation(&generation(Api::Messages, 0, 0), 0);
    let text = metrics.render("m", 1, &EngineReport::default(), 0);
    let api = ["api=\"messages\""];
    assert_eq!(value(&text, "euhedral_request_duration_seconds_count", &api), 1.0);
    assert_eq!(value(&text, "euhedral_queue_wait_seconds_count", &api), 1.0);
    assert_eq!(value(&text, "euhedral_time_to_first_token_seconds_count", &api), 0.0);
    assert_eq!(value(&text, "euhedral_prefill_tokens_per_second_count", &[]), 0.0);
    assert_eq!(value(&text, "euhedral_decode_tokens_per_second_count", &[]), 0.0);
    assert_eq!(value(&text, "euhedral_prompt_prefilled_tokens_total", &[]), 60.0);
    assert_eq!(value(&text, "euhedral_reasoning_tokens_total", &[]), 0.0);
}

#[test]
fn rates_are_tokens_per_second_of_the_stage() {
    let metrics = Metrics::new(0, 0);
    // 60 tokens prefilled in 1 ms; 9 more tokens in 1 s after the first of 10.
    metrics.generation(&generation(Api::Messages, 3_001_000, 10), 0);
    let text = metrics.render("m", 1, &EngineReport::default(), 0);
    assert_eq!(value(&text, "euhedral_prefill_tokens_per_second_sum", &[]), 60_000.0);
    assert_eq!(value(&text, "euhedral_decode_tokens_per_second_sum", &[]), 9.0);
    assert_eq!(value(&text, "euhedral_decode_tokens_per_second_max", &[]), 9.0);
    // One token is no decode rate.
    metrics.generation(&generation(Api::Messages, 3_001_000, 1), 0);
    let text = metrics.render("m", 1, &EngineReport::default(), 0);
    assert_eq!(value(&text, "euhedral_decode_tokens_per_second_count", &[]), 1.0);
}

#[test]
fn a_duration_counts_in_the_first_bucket_whose_bound_it_does_not_pass() {
    let durations = Durations::new(0);
    durations.record(10_000_000, 0);
    durations.record(10_000_001, 0);
    durations.record(600_000_000_000, 0);
    durations.record(600_000_000_001, 0);
    assert_eq!(durations.buckets[0].load(Ordering::Relaxed), 1);
    assert_eq!(durations.buckets[1].load(Ordering::Relaxed), 1);
    let last = DURATION_BOUNDS_NS.len();
    assert_eq!(durations.buckets[last - 1].load(Ordering::Relaxed), 1);
    assert_eq!(durations.buckets[last].load(Ordering::Relaxed), 1);
    assert_eq!(durations.max_ns.poll(0), 600_000_000_001);
    assert_eq!(DURATION_BOUNDS_NS[last - 1], 600_000_000_000);
    assert!(DURATION_BOUNDS_NS.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(RATE_BOUNDS.windows(2).all(|pair| pair[0] < pair[1]));
}

#[test]
fn a_clock_that_ran_backwards_and_a_zero_tool_call_record_nothing() {
    let metrics = Metrics::new(0, 0);
    let mut backwards = generation(Api::Responses, 0, 5);
    backwards.ended_ns = 0;
    backwards.started_ns = 0;
    metrics.generation(&backwards, 0);
    metrics.tool_calls(Api::Responses, 0);
    let text = metrics.render("m", 1, &EngineReport::default(), 0);
    let api = ["api=\"responses\""];
    assert_eq!(value(&text, "euhedral_request_duration_seconds_count", &api), 0.0);
    assert_eq!(value(&text, "euhedral_queue_wait_seconds_count", &api), 0.0);
    assert_eq!(value(&text, "euhedral_tool_calls_total", &api), 0.0);
    assert_eq!(value(&text, "euhedral_completion_tokens_total", &[]), 5.0);
}

#[test]
fn speculative_steps_are_counted_by_drafts_accepted_in_label_order() {
    let metrics = Metrics::new(11, 0);
    metrics.speculative_step(2);
    metrics.speculative_step(11);
    metrics.speculative_step(40);
    let text = metrics.render("m", 1, &EngineReport::default(), 0);
    let drafts: Vec<&str> = text
        .lines()
        .filter(|line| line.starts_with("euhedral_speculative_accepted_total{"))
        .filter_map(|line| line.split("drafts=\"").nth(1)?.split('"').next())
        .collect();
    assert_eq!(drafts, ["0", "1", "10", "11", "2", "3", "4", "5", "6", "7", "8", "9"]);
    assert_eq!(
        value(&text, "euhedral_speculative_accepted_total", &["drafts=\"11\""]),
        2.0
    );
    assert_eq!(value(&text, "euhedral_speculative_verifications_total", &[]), 3.0);
    assert_eq!(value(&text, "euhedral_speculative_accepted_tokens_total", &[]), 24.0);
}

#[test]
fn the_largest_values_expire_by_windows_of_a_minute() {
    const MINUTE: i64 = 60_000;
    let max_of = |text: &str| value(text, "euhedral_request_duration_seconds_max", &["api=\"messages\""]);
    let metrics = Metrics::new(0, 0);
    metrics.generation(&generation(Api::Messages, 3_001_000, 10), 0);
    let report = EngineReport::default();
    let first = max_of(&metrics.render("m", 1, &report, 0));
    assert!(first > 0.0);
    // Still reported just before the third window ends, gone when it has.
    assert_eq!(max_of(&metrics.render("m", 1, &report, 3 * MINUTE - 1)), first);
    assert_eq!(max_of(&metrics.render("m", 1, &report, 3 * MINUTE)), 0.0);
    // The rates have their own windows.
    assert!(
        value(
            &metrics.render("m", 1, &report, 0),
            "euhedral_decode_tokens_per_second_max",
            &[]
        ) == 0.0
    );
}
