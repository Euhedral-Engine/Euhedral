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

use common::{Fixture, HANG_GUARD, Raw, assert_json, headers, small_limits, text};
use euhedral_inference_api_server::Generation;
use euhedral_inference_api_server::Limits;
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
        "euhedral_engine_info{{{app},artifact=\"unknown\",family=\"unknown\",model=\"euhedral-test-model\",speculation=\"none\",speculative_depth=\"0\"}} 1\n"
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
    let end = Instant::now() + HANG_GUARD;
    while fixture.server.admission().counts() != (Counts { active: 1, waiting: 1 }) {
        assert!(
            Instant::now() < end,
            "never reached the expected slots: {:?}",
            fixture.server.admission().counts()
        );
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
    let end = Instant::now() + HANG_GUARD;
    while fixture.server.admission().counts() != Counts::default() {
        assert!(
            Instant::now() < end,
            "never reached the expected slots: {:?}",
            fixture.server.admission().counts()
        );
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
    let before = fixture.get("/metrics");
    let head = fixture.call(&Fixture::req(Method::Head, "/metrics"));
    assert_eq!(head.status, 200);
    assert!(head.body.is_empty());
    assert_eq!(head.headers.get("content-type"), Some(TYPE));
    // The scrape that follows differs (the uptime and the timing of the HEAD itself), so the length is bounded rather than equal:
    // the body of the HEAD is the one before it plus the series of the request before it.
    let length: usize = head
        .headers
        .get("content-length")
        .and_then(|length| length.parse().ok())
        .expect("a length");
    let before = before.body.len();
    assert!(length > before && length < before + 4096, "{length} after {before}");
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

fn roomy_limits() -> Limits {
    let mut limits = small_limits();
    limits.max_body_bytes = 1 << 20;
    limits
}

const APP: &str = "application=\"euhedral-inference\"";

/// The value of the sample that starts with `prefix`, which ends at the space before it.
fn sample_value(body: &str, prefix: &str) -> Option<f64> {
    body.lines()
        .find_map(|line| line.strip_prefix(prefix))
        .and_then(|rest| rest.trim().parse().ok())
}

/// The scrape's lines of a family, `# HELP` and `# TYPE` included.
fn family<'a>(body: &'a str, name: &str) -> Vec<&'a str> {
    body.lines()
        .filter(|line| {
            line.starts_with(&format!("{name}{{"))
                || line.starts_with(&format!("{name}_count{{"))
                || line.starts_with(&format!("{name}_sum{{"))
                || line.starts_with(&format!("# HELP {name} "))
                || line.starts_with(&format!("# TYPE {name} "))
        })
        .collect()
}

#[test]
fn the_process_series_are_well_formed_and_the_uptime_does_not_go_back() {
    let fixture = Fixture::start();
    let first = text(&fixture.get("/metrics")).to_owned();
    let second = text(&fixture.get("/metrics")).to_owned();
    let mut names = vec![
        "process_cpu_time_ns_total",
        "process_cpu_usage",
        "process_files_open_files",
        "process_start_time_seconds",
        "process_uptime_seconds",
    ];
    // The limit of open files is read where the system has one to read.
    if cfg!(target_os = "linux") {
        names.push("process_files_max_files");
    }
    for name in names {
        assert!(second.contains(&format!("# TYPE {name} ")), "{name}\n{second}");
    }
    assert!(second.contains("# TYPE process_cpu_time_ns_total counter\n"));
    let usage = sample_value(&second, &format!("process_cpu_usage{{{APP}}} ")).expect("a cpu usage");
    assert!((0.0..=1.0).contains(&usage), "{usage}");
    for line in second.lines().filter(|line| line.starts_with("process_")) {
        let (series, value) = line.rsplit_once(' ').expect("a sample has a value");
        assert!(series.contains(&format!("{{{APP}}}")), "{line}");
        assert!(value.parse::<f64>().is_ok(), "{line}");
    }
    let uptime = format!("process_uptime_seconds{{{APP}}} ");
    let before = sample_value(&first, &uptime).expect("an uptime");
    let after = sample_value(&second, &uptime).expect("an uptime");
    assert!(before >= 0.0 && after >= before, "{before} then {after}");
    // A restart is a change of the start time: it is the same at every scrape.
    let start = format!("process_start_time_seconds{{{APP}}} ");
    let started = sample_value(&second, &start).expect("a start time");
    assert!(started > 1.0e9, "{started}");
    assert_eq!(sample_value(&first, &start), Some(started));
    fixture.finish();
}

#[test]
fn every_route_is_timed_under_its_template_and_an_unknown_path_has_one_label() {
    let fixture = Fixture::with(roomy_limits(), Some(2));
    assert_eq!(fixture.get("/health").status, 200);
    assert_eq!(fixture.get("/v1/models").status, 200);
    assert_eq!(fixture.get("/v1/models/euhedral-test-model").status, 200);
    assert_eq!(fixture.get("/v1/models/another").status, 404);
    assert_eq!(fixture.get("/nowhere/1").status, 404);
    assert_eq!(fixture.get("/nowhere/2?x=1").status, 404);
    assert_eq!(fixture.post("/v1/models", None, "").status, 405);
    let body =
        r#"{"model":"euhedral-test-model","reasoning_effort":"none","messages":[{"role":"user","content":"Hi"}]}"#;
    assert_eq!(
        fixture
            .post("/v1/chat/completions", Some("application/json"), body)
            .status,
        200
    );
    let scrape = fixture.get("/metrics");
    let body = text(&scrape);
    let line = |method: &str, outcome: &str, status: u16, uri: &str, count: u64| {
        format!(
            "http_server_requests_seconds_count{{{APP},error=\"none\",exception=\"none\",method=\"{method}\",outcome=\"{outcome}\",status=\"{status}\",uri=\"{uri}\"}} {count}"
        )
    };
    for expected in [
        line("GET", "SUCCESS", 200, "/health", 1),
        line("GET", "SUCCESS", 200, "/v1/models", 1),
        line("GET", "SUCCESS", 200, "/v1/models/{model}", 1),
        line("GET", "CLIENT_ERROR", 404, "/v1/models/{model}", 1),
        line("GET", "CLIENT_ERROR", 404, "/**", 2),
        line("POST", "CLIENT_ERROR", 405, "/v1/models", 1),
        line("POST", "SUCCESS", 200, "/v1/chat/completions", 1),
    ] {
        assert!(body.lines().any(|each| each == expected), "{expected}\n{body}");
    }
    assert!(body.contains("# HELP http_server_requests_seconds  \n# TYPE http_server_requests_seconds summary\n"));
    assert!(body.contains("# TYPE http_server_requests_seconds_max gauge\n"));
    // The samples run by method, then outcome, status and uri, as the labels sort.
    let at = |text: String| body.find(&text).expect("a sample");
    assert!(at(line("GET", "CLIENT_ERROR", 404, "/**", 2)) < at(line("GET", "SUCCESS", 200, "/health", 1)));
    assert!(
        at(line("GET", "SUCCESS", 200, "/v1/models", 1)) < at(line("POST", "SUCCESS", 200, "/v1/chat/completions", 1))
    );
    assert!(!body.contains("nowhere") && !body.contains("another"), "{body}");
    // The scrape that is being answered is timed after it, so the next one shows it.
    let again = fixture.get("/metrics");
    assert!(
        text(&again)
            .lines()
            .any(|each| each == line("GET", "SUCCESS", 200, "/metrics", 1)),
        "{}",
        text(&again)
    );
    fixture.finish();
}

#[test]
fn a_streamed_reply_is_timed_when_its_last_byte_is_written() {
    let fixture = Fixture::with(roomy_limits(), Some(2));
    fixture.backend.set_script(Script::Tokens {
        chunks: vec!["Hello".to_owned(), " there".to_owned()],
        stop_token: true,
    });
    let body = r#"{"model":"euhedral-test-model","stream":true,"reasoning_effort":"none","messages":[{"role":"user","content":"Hi"}]}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let mut raw = Raw::connect(fixture.addr);
    raw.send(&request);
    raw.read_until("[DONE]");
    let timed = "http_server_requests_seconds_count{";
    let end = Instant::now() + HANG_GUARD;
    let scrape = loop {
        let scrape = text(&fixture.get("/metrics")).to_owned();
        if scrape.contains("uri=\"/v1/chat/completions\"") {
            break scrape;
        }
        assert!(Instant::now() < end, "the request was never counted: {scrape}");
        sleep(Duration::from_millis(5));
    };
    assert!(scrape.contains(timed), "{scrape}");
    let chat = family(&scrape, "http_server_requests_seconds");
    let counted = chat
        .iter()
        .find(|line| line.contains("_count{") && line.contains("uri=\"/v1/chat/completions\""))
        .expect("the stream is counted");
    assert!(
        counted.contains("status=\"200\"") && counted.ends_with(" 1"),
        "{counted}"
    );
    fixture.finish();
}

/// A streaming chat request whose generation the test drives: the backend holds it after the launch, and the test
/// gives it its one piece, `tok0`, once the launch has happened. An unthrottled generator would instead outrun a
/// loaded host's write of the head and the client's reads, and the server would rightly cut the stream of a
/// reader that falls behind (`Limits::max_held_bytes`), which is not what these tests look at.
fn open_stream(fixture: &Fixture) -> Raw {
    fixture.backend.set_max_active(1);
    fixture.backend.set_context_length(1 << 30);
    fixture.backend.set_script(Script::Hung);
    let body = r#"{"model":"euhedral-test-model","stream":true,"reasoning_effort":"none","max_tokens":100000000,"messages":[{"role":"user","content":"Hi"}]}"#;
    let request = format!(
        "POST /v1/chat/completions HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    );
    let mut raw = Raw::connect(fixture.addr);
    raw.send(&request);
    let end = Instant::now() + HANG_GUARD;
    while !fixture
        .backend
        .generations()
        .first()
        .is_some_and(|g| g.started().is_some())
    {
        assert!(Instant::now() < end, "the generation was never launched");
        sleep(Duration::from_millis(5));
    }
    assert!(
        fixture.backend.generations()[0].emit_held("tok0 "),
        "the launched generation holds its listener"
    );
    raw.read_until("tok0");
    raw
}

#[test]
fn a_stream_still_open_is_not_yet_in_the_timings() {
    let fixture = Fixture::with(roomy_limits(), Some(2));
    let raw = open_stream(&fixture);
    let scrape = text(&fixture.get("/metrics")).to_owned();
    assert!(!scrape.contains("uri=\"/v1/chat/completions\""), "{scrape}");
    drop(raw);
    assert!(fixture.backend.generations()[0].finish_held_ok(1, false));
    fixture.finish();
}

#[test]
fn a_stream_cut_after_its_head_is_timed_once() {
    let fixture = Fixture::with(roomy_limits(), Some(2));
    let raw = open_stream(&fixture);
    drop(raw);
    // The server learns that the client is gone when a write fails, so the generation keeps saying a piece until the
    // cut cancels it; the test then ends it as a cancelled one ends, and the request ends with it.
    let end = Instant::now() + HANG_GUARD;
    while !fixture.backend.generations()[0].is_cancelled() {
        assert!(Instant::now() < end, "the cut never cancelled the generation");
        assert!(fixture.backend.generations()[0].emit_held("tok "));
        sleep(Duration::from_millis(5));
    }
    assert!(fixture.backend.generations()[0].finish_held_ok(1, false));
    let end = Instant::now() + HANG_GUARD;
    let counted = format!(
        "http_server_requests_seconds_count{{{APP},error=\"none\",exception=\"none\",method=\"POST\",outcome=\"SUCCESS\",status=\"200\",uri=\"/v1/chat/completions\"}} 1\n"
    );
    loop {
        let scrape = text(&fixture.get("/metrics")).to_owned();
        if scrape.contains(&counted) {
            break;
        }
        assert!(Instant::now() < end, "the request was never counted: {scrape}");
        sleep(Duration::from_millis(5));
    }
    // The stop returns once every connection has ended: nothing is left that could count the request again.
    fixture.stop();
    let mut last = fixture.server.metrics().render("m", 1, &EngineReport::default(), 0);
    last.push_str(&fixture.server.metrics().render_platform(0));
    assert!(last.contains(&counted), "{last}");
    fixture.finish_after_stop();
}

#[test]
fn each_method_has_its_own_label() {
    let fixture = Fixture::start();
    assert_eq!(fixture.call(&Fixture::req(Method::Put, "/v1/models")).status, 405);
    assert_eq!(fixture.call(&Fixture::req(Method::Delete, "/v1/models")).status, 405);
    let body = text(&fixture.get("/metrics")).to_owned();
    for method in ["PUT", "DELETE"] {
        assert!(
            body.contains(&format!(
                "method=\"{method}\",outcome=\"CLIENT_ERROR\",status=\"405\",uri=\"/v1/models\""
            )),
            "{body}"
        );
    }
    fixture.finish();
}
