//! `GET /metrics` over loopback: status, content type and body, answered in the route frame.
#![allow(
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixture: the test thread is not a lattice worker; it waits on a channel with a bound"
)]

mod common;

use std::thread::sleep;
use std::time::{Duration, Instant};

use common::{Fixture, PATIENCE, Raw, assert_json, headers, small_limits, text};
use euhedral_inference_api_server::metrics::{Api, ConstraintFailure, Outcome};
use euhedral_inference_api_server::scripted::Script;
use euhedral_inference_api_server::{Counts, EngineReport, PrefixCacheReport};
use euhedral_net::Method;

const TYPE: &str = "text/plain;version=0.0.4;charset=utf-8";

#[test]
fn the_metrics_are_prometheus_text_with_the_series_of_the_server_and_the_engine() {
    let fixture = Fixture::start();
    let answer = fixture.get("/metrics");
    assert_eq!(answer.status, 200, "{}", text(&answer));
    assert_eq!(
        headers(&answer),
        [
            ("Content-Type".to_owned(), TYPE.to_owned()),
            ("Content-Length".to_owned(), answer.body.len().to_string())
        ]
    );
    let body = text(&answer);
    let app = "application=\"euhedral-inference\"";
    assert!(body.starts_with("# HELP euhedral_completion_tokens_total Generated tokens, reasoning included\n# TYPE euhedral_completion_tokens_total counter\n"));
    let info = format!(
        "euhedral_engine_info{{{app},artifact=\"unknown\",model=\"euhedral-test-model\",speculation=\"none\",speculative_depth=\"0\"}} 1\n"
    );
    assert!(body.contains(&info), "{body}");
    assert!(body.contains(&format!("euhedral_engine_context_tokens{{{app}}} 4096.0\n")));
    assert!(body.contains(&format!(
        "euhedral_requests_total{{api=\"messages\",{app},outcome=\"success\"}} 0.0\n"
    )));
    assert!(!body.contains("euhedral_prefix_"));
    assert!(body.ends_with('\n'));
    fixture.finish();
}

#[test]
fn recordings_show_in_the_next_scrape() {
    let fixture = Fixture::start();
    let metrics = fixture.server.metrics();
    metrics.request(Api::ChatCompletions, Outcome::Invalid);
    metrics.request(Api::ChatCompletions, Outcome::Invalid);
    metrics.constrained_failure(ConstraintFailure::InvalidToolCall);
    let answer = fixture.get("/metrics");
    let body = text(&answer);
    let app = "application=\"euhedral-inference\"";
    assert!(body.contains(&format!(
        "euhedral_requests_total{{api=\"chat_completions\",{app},outcome=\"invalid\"}} 2.0\n"
    )));
    assert!(body.contains(&format!(
        "euhedral_constrained_failures_total{{{app},reason=\"invalid_tool_call\"}} 1.0\n"
    )));
    fixture.finish();
}

#[test]
fn the_queue_gauges_show_the_slots_in_use_and_the_requests_waiting() {
    let mut limits = small_limits();
    limits.max_body_bytes = 1 << 20;
    let fixture = Fixture::with(limits, Some(2));
    fixture.backend.set_max_active(1);
    fixture.backend.set_context_length(1 << 30);
    fixture.backend.set_script(Script::Endless);
    let body = r#"{"model":"euhedral-test-model","reasoning_effort":"none","max_tokens":100000000,"messages":[{"role":"user","content":"Hi"}]}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let mut first = Raw::connect(fixture.addr);
    first.send(&request);
    let mut second = Raw::connect(fixture.addr);
    second.send(&request);
    let end = Instant::now() + PATIENCE;
    while fixture.server.admission().counts() != (Counts { active: 1, waiting: 1 }) {
        assert!(Instant::now() < end, "{:?}", fixture.server.admission().counts());
        sleep(Duration::from_millis(5));
    }
    let answer = fixture.get("/metrics");
    let text = text(&answer);
    let app = "application=\"euhedral-inference\"";
    assert!(
        text.contains(&format!("euhedral_generations_active{{{app}}} 1.0\n")),
        "{text}"
    );
    assert!(
        text.contains(&format!("euhedral_generations_queued{{{app}}} 1.0\n")),
        "{text}"
    );
    drop(first);
    drop(second);
    let end = Instant::now() + PATIENCE;
    while fixture.server.admission().counts() != Counts::default() {
        assert!(Instant::now() < end, "{:?}", fixture.server.admission().counts());
        sleep(Duration::from_millis(5));
    }
    fixture.finish();
}

#[test]
fn the_engine_reports_at_every_scrape() {
    let report = EngineReport {
        artifact: Some("a.edrl".to_owned()),
        speculation: "mtp".to_owned(),
        speculative_depth: 2,
        workers: 3,
        prefix_cache: Some(PrefixCacheReport {
            lookups: 5,
            ..PrefixCacheReport::default()
        }),
        ..EngineReport::default()
    };
    let fixture = Fixture::reporting(small_limits(), Some(2), report);
    let answer = fixture.get("/metrics");
    let body = text(&answer);
    let app = "application=\"euhedral-inference\"";
    assert!(body.contains(&format!("euhedral_engine_workers{{{app}}} 3.0\n")));
    assert!(body.contains(&format!("euhedral_prefix_lookups_total{{{app}}} 5.0\n")));
    assert!(body.contains(&format!(
        "euhedral_speculative_accepted_total{{{app},drafts=\"2\"}} 0.0\n"
    )));
    assert!(body.contains("speculation=\"mtp\",speculative_depth=\"2\"} 1\n"));
    fixture.backend.set_engine_report(EngineReport {
        workers: 9,
        ..EngineReport::default()
    });
    let again = fixture.get("/metrics");
    assert!(text(&again).contains(&format!("euhedral_engine_workers{{{app}}} 9.0\n")));
    assert!(!text(&again).contains("euhedral_prefix_"));
    fixture.finish();
}

#[test]
fn head_options_and_other_methods_answer_as_for_any_get_route() {
    let fixture = Fixture::start();
    let head = fixture.call(&Fixture::req(Method::Head, "/metrics"));
    assert_eq!(head.status, 200);
    assert!(head.body.is_empty());
    assert_eq!(head.headers.get("content-type"), Some(TYPE));
    let get = fixture.get("/metrics");
    assert_eq!(
        head.headers.get("content-length"),
        Some(get.body.len().to_string().as_str())
    );
    let options = fixture.call(&Fixture::req(Method::Options, "/metrics"));
    assert_eq!(options.status, 200);
    assert_eq!(options.headers.get("allow"), Some("GET,HEAD,OPTIONS"));
    let post = fixture.post("/metrics", Some("application/json"), "{}");
    assert_eq!(post.status, 405);
    fixture.finish();
}

#[test]
fn an_accept_that_excludes_the_text_is_refused_and_a_scraper_s_is_not() {
    let fixture = Fixture::start();
    let scraper = "application/openmetrics-text;version=1.0.0,text/plain;version=0.0.4;q=0.5,*/*;q=0.1";
    let accepted = Fixture::req(Method::Get, "/metrics").header("Accept", scraper);
    assert_eq!(fixture.call(&accepted).status, 200);
    let plain = Fixture::req(Method::Get, "/metrics").header("Accept", "text/plain");
    assert_eq!(fixture.call(&plain).status, 200);
    let json = Fixture::req(Method::Get, "/metrics").header("Accept", "application/json");
    let refused = r#"{"error":{"message":"Acceptable representations: [text/plain;version=0.0.4;charset=utf-8].","type":"invalid_request_error","param":null,"code":null}}"#;
    assert_json(&fixture.call(&json), 406, refused);
    fixture.finish();
}
