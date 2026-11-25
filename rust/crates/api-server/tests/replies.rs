//! The JSON replies of the generating routes, over loopback, against a scripted backend.
#![allow(
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixture: the test thread is not a lattice worker; it waits with a bound"
)]

mod common;

use std::thread::sleep;
use std::time::{Duration, Instant};

use common::{Fixture, PATIENCE, Raw, headers, json_headers, text};
use euhedral_inference_api_server::answer::{owner_fault_is_held, release_owner_fault};
use euhedral_inference_api_server::scripted::{LaunchFault, MODEL_ID, Script};
use euhedral_inference_api_server::{Backend, BackendError, Counts, Generation, Limits};
use euhedral_net::Response;

const CHAT: &str = "/v1/chat/completions";
const MESSAGES: &str = "/v1/messages";
const RESPONSES: &str = "/v1/responses";
/// A budget no test reaches.
const BIG: &str = r#","max_tokens":100000000"#;
const JSON: Option<&str> = Some("application/json");

/// The limit a test makes pass by moving its clock.
const LIMIT: Duration = Duration::from_millis(500);

fn limits() -> Limits {
    let mut limits = common::small_limits();
    limits.max_body_bytes = 1 << 20;
    limits
}

fn tokens(chunks: &[&str], stop_token: bool) -> Script {
    Script::Tokens {
        chunks: chunks.iter().map(|chunk| (*chunk).to_owned()).collect(),
        stop_token,
    }
}

/// The ids and times of a body replaced by constants: the only thing the comparison ignores.
fn norm(body: &str) -> String {
    let mut out = String::new();
    let mut rest = body;
    while !rest.is_empty() {
        let mut best: Option<(usize, &str, bool)> = None;
        for (key, quoted) in [
            ("\"id\":\"", true),
            ("\"call_id\":\"", true),
            ("\"created\":", false),
            ("\"created_at\":", false),
        ] {
            if let Some(at) = rest.find(key)
                && best.is_none_or(|(first, _, _)| at < first)
            {
                best = Some((at, key, quoted));
            }
        }
        let Some((at, key, quoted)) = best else {
            out.push_str(rest);
            break;
        };
        let value = at + key.len();
        out.push_str(&rest[..value]);
        let tail = &rest[value..];
        let end = if quoted {
            tail.find('"').unwrap_or(tail.len())
        } else {
            tail.find(|c: char| !c.is_ascii_digit()).unwrap_or(tail.len())
        };
        out.push_str(if quoted { "ID" } else { "0" });
        rest = &tail[end..];
    }
    out
}

/// A chat request without reasoning; `extra` starts with a comma.
fn chat_body(extra: &str) -> String {
    format!(
        r#"{{"model":"{MODEL_ID}","reasoning_effort":"none"{extra},"messages":[{{"role":"user","content":"Hi"}}]}}"#
    )
}

fn chat(fx: &Fixture, extra: &str) -> Response {
    fx.post(CHAT, JSON, &chat_body(extra))
}

fn tools(extra: &str) -> String {
    let tool = r#"{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}"#;
    let ping = r#"{"type":"function","function":{"name":"ping","parameters":{"type":"object","properties":{}}}}"#;
    format!(r#","tools":[{tool},{ping}]{extra}"#)
}

fn call(name: &str, parameters: &[(&str, &str)]) -> String {
    let mut out = format!("<tool_call>\n<function={name}>\n");
    for (key, value) in parameters {
        out.push_str(&format!("<parameter={key}>\n{value}\n</parameter>\n"));
    }
    out.push_str("</function>\n</tool_call>");
    out
}

/// Asserts a JSON answer's status and headers and returns its body with ids and times normalised.
fn json(response: &Response, status: u16) -> String {
    assert_eq!(response.status, status, "{}", text(response));
    assert_eq!(headers(response), json_headers(response.body.len()));
    norm(text(response))
}

/// Waits, within the patience of a test, until no request holds or awaits a slot.
fn settle(fx: &Fixture) {
    let end = Instant::now() + PATIENCE;
    while fx.server.admission().counts() != Counts::default() {
        assert!(Instant::now() < end, "slots held: {:?}", fx.server.admission().counts());
        sleep(Duration::from_millis(5));
    }
}

/// Waits until `check` holds.
fn until(what: &str, check: impl Fn() -> bool) {
    let end = Instant::now() + PATIENCE;
    while !check() {
        assert!(Instant::now() < end, "never: {what}");
        sleep(Duration::from_millis(5));
    }
}

/// The scripted prompt has one token per character.
fn prompt_tokens(fx: &Fixture, index: usize) -> usize {
    until("a generation started", || {
        fx.backend
            .generations()
            .get(index)
            .is_some_and(|g| g.started().is_some())
    });
    fx.backend.generations()[index]
        .started()
        .expect("started")
        .0
        .chars()
        .count()
}

fn usage(prompt: usize, completion: usize, cached: usize, reasoning: usize) -> String {
    let details = format!(r#","prompt_tokens_details":{{"cached_tokens":{cached}}}"#);
    let more = if reasoning > 0 {
        format!(r#","completion_tokens_details":{{"reasoning_tokens":{reasoning}}}"#)
    } else {
        String::new()
    };
    format!(
        r#""usage":{{"prompt_tokens":{prompt},"completion_tokens":{completion},"total_tokens":{}{details}{more}}}"#,
        prompt + completion
    )
}

fn assert_closed_once(fx: &Fixture) {
    until("every generation closed", || {
        fx.backend.generations().iter().all(|g| g.close_count() == 1)
    });
    settle(fx);
}

/// A generation that runs until it is cancelled: the budget is far beyond what a test lets it use.
fn endless(fx: &Fixture) {
    fx.backend.set_context_length(1 << 30);
    fx.backend.set_script(Script::Endless);
}

fn generating(fx: &Fixture) -> bool {
    fx.backend.generations().first().is_some_and(|g| g.emitted() > 0)
}

// Plain text, on the three surfaces.

#[test]
fn chat_plain_text() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello", ", ", "world"], true));
    let response = chat(&fx, "");
    let prompt = prompt_tokens(&fx, 0);
    let body = json(&response, 200);
    let expected = format!(
        r#"{{"id":"ID","object":"chat.completion","created":0,"model":"{MODEL_ID}","choices":[{{"index":0,"message":{{"role":"assistant","content":"Hello, world","refusal":null}},"logprobs":null,"finish_reason":"stop"}}],{}}}"#,
        usage(prompt, 4, 0, 0)
    );
    assert_eq!(body, expected);
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn chat_length() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello", ", ", "world"], true));
    let body = json(&chat(&fx, r#","max_tokens":2"#), 200);
    assert!(body.contains(r#""content":"Hello, ""#), "{body}");
    assert!(body.contains(r#""finish_reason":"length""#), "{body}");
    assert!(body.contains(r#""completion_tokens":2"#), "{body}");
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn chat_stop_sequence_cancels_the_generation() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend
        .set_script(tokens(&["one ", "tw", "o STO", "P three", " four"], true));
    let body = json(&chat(&fx, r#","stop":["STOP"]"#), 200);
    assert!(body.contains(r#""content":"one two ""#), "{body}");
    assert!(body.contains(r#""finish_reason":"stop""#), "{body}");
    assert_closed_once(&fx);
    assert!(fx.backend.generations()[0].is_cancelled());
    fx.finish();
}

#[test]
fn chat_reasoning_on_and_off() {
    let fx = Fixture::with(limits(), Some(2));
    let chunks = ["\nLet me", " think.\n", "</think>", "\n\n", "Hello."];
    fx.backend.set_script(tokens(&chunks, true));
    fx.backend.set_counts(3, 0);
    let on = format!(r#"{{"model":"{MODEL_ID}","messages":[{{"role":"user","content":"Hi"}}]}}"#);
    let body = json(&fx.post(CHAT, JSON, &on), 200);
    assert!(body.contains(r#""content":"Hello.""#), "{body}");
    assert!(body.contains(r#""reasoning_content":"Let me think.""#), "{body}");
    assert!(body.contains(r#""reasoning_tokens":3"#), "{body}");
    assert!(body.contains(r#""finish_reason":"stop""#), "{body}");

    // With reasoning off, the model's text is the answer and no reasoning is reported.
    fx.backend.set_script(tokens(&["Hello."], true));
    fx.backend.set_counts(0, 0);
    let body = json(&chat(&fx, ""), 200);
    assert!(body.contains(r#""content":"Hello.""#), "{body}");
    assert!(!body.contains("reasoning"), "{body}");
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn chat_reasoning_cut_by_the_budget_is_length_with_no_content() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend
        .set_script(tokens(&["\nLet me", " think.\n", "</think>", "Hello."], true));
    let on = format!(r#"{{"model":"{MODEL_ID}","max_tokens":2,"messages":[{{"role":"user","content":"Hi"}}]}}"#);
    let body = json(&fx.post(CHAT, JSON, &on), 200);
    assert!(body.contains(r#""content":"""#), "{body}");
    assert!(body.contains(r#""finish_reason":"length""#), "{body}");
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn chat_usage_reports_cached_prompt_tokens() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hi"], true));
    fx.backend.set_counts(0, 5);
    let response = chat(&fx, "");
    let prompt = prompt_tokens(&fx, 0);
    let body = json(&response, 200);
    assert!(body.contains(&usage(prompt, 2, 5, 0)), "{body}");
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn chat_token_ids_only_when_asked() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello", "!"], true));
    let body = json(&chat(&fx, r#","return_token_ids":true"#), 200);
    assert!(body.contains(r#""prompt_token_ids":[100,"#), "{body}");
    assert!(body.contains(r#""token_ids":[5000,5001,5002]"#), "{body}");
    let body = json(&chat(&fx, ""), 200);
    assert!(!body.contains("token_ids"), "{body}");
    assert_closed_once(&fx);
    fx.finish();
}

// Tool calls.

#[test]
fn chat_one_tool_call() {
    let fx = Fixture::with(limits(), Some(2));
    let text_call = call("get_weather", &[("city", "Paris"), ("days", "3")]);
    fx.backend.set_script(tokens(&[&text_call], true));
    let body = json(&chat(&fx, &tools("")), 200);
    assert!(body.contains(r#""content":null"#), "{body}");
    assert!(
        body.contains(r#""tool_calls":[{"id":"ID","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\",\"days\":3}"}}]"#),
        "{body}"
    );
    assert!(body.contains(r#""finish_reason":"tool_calls""#), "{body}");
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn chat_several_tool_calls_with_content() {
    let fx = Fixture::with(limits(), Some(2));
    let generated = format!(
        "Let me check.\n\n{}\n{}",
        call("get_weather", &[("city", "Line one\nline two"), ("days", "2")]),
        call("ping", &[])
    );
    let (a, b) = generated.split_at(12);
    let (b, c) = b.split_at(60);
    fx.backend.set_script(tokens(&[a, b, c], true));
    let body = json(&chat(&fx, &tools("")), 200);
    assert!(body.contains(r#""content":"Let me check.""#), "{body}");
    assert!(
        body.contains(r#""arguments":"{\"city\":\"Line one\\nline two\",\"days\":2}""#),
        "{body}"
    );
    assert!(body.contains(r#""name":"ping","arguments":"{}""#), "{body}");
    assert!(body.contains(r#""finish_reason":"tool_calls""#), "{body}");
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn chat_invalid_tool_calls_fail_and_stop_decoding() {
    let fx = Fixture::with(limits(), Some(2));
    let valid = call("get_weather", &[("city", "x")]);
    let cases = [
        call("nothing", &[]),
        call("get_weather", &[("days", "3")]),
        call("get_weather", &[("city", "x"), ("days", "three")]),
        "<tool_call>\nnot a function\n</tool_call>".to_owned(),
        format!("{valid}\nAnd more text."),
    ];
    for (index, case) in cases.iter().enumerate() {
        fx.backend.set_script(tokens(&[case, "tail1", "tail2"], true));
        let response = chat(&fx, &tools(""));
        assert_eq!(response.status, 500, "{case}: {}", text(&response));
        let body = norm(text(&response));
        assert!(body.contains(r#""type":"server_error""#), "{body}");
        assert!(body.contains(r#""code":"invalid_tool_call""#), "{body}");
        until("closed", || fx.backend.generations()[index].close_count() == 1);
        let generation = &fx.backend.generations()[index];
        assert!(generation.is_cancelled(), "{case}");
        settle(&fx);
    }
    // An unfinished call at the end of the output is just as invalid.
    fx.backend.set_script(tokens(
        &["<tool_call>\n<function=get_weather>\n<parameter=city>\n"],
        true,
    ));
    let response = chat(&fx, &tools(""));
    assert_eq!(response.status, 500, "{}", text(&response));
    assert!(text(&response).contains("invalid_tool_call"));
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn chat_parallel_tool_calls_false_ends_after_the_first_call() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&[&call("ping", &[]), "more"], false));
    let body = json(&chat(&fx, &tools(r#","parallel_tool_calls":false"#)), 200);
    assert_eq!(body.matches(r#""type":"function""#).count(), 1, "{body}");
    assert!(body.contains(r#""finish_reason":"tool_calls""#), "{body}");
    assert_closed_once(&fx);
    assert!(fx.backend.generations()[0].is_cancelled());

    // A second call in the same text is never delivered.
    let two = format!("{}\n{}", call("ping", &[]), call("ping", &[]));
    fx.backend.set_script(tokens(&[&two], true));
    let body = json(&chat(&fx, &tools(r#","parallel_tool_calls":false"#)), 200);
    assert_eq!(body.matches(r#""type":"function""#).count(), 1, "{body}");
    assert!(body.contains(r#""finish_reason":"tool_calls""#), "{body}");
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn chat_budget_keeps_a_complete_call_and_drops_a_partial_one() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&[&call("ping", &[])], false));
    let body = json(&chat(&fx, &tools(r#","max_tokens":1"#)), 200);
    assert!(body.contains(r#""content":null"#), "{body}");
    assert!(body.contains(r#""name":"ping""#), "{body}");
    assert!(body.contains(r#""finish_reason":"length""#), "{body}");

    fx.backend.set_script(tokens(
        &["<tool_call>\n<function=get_weather>\n<parameter=city>\nPa"],
        false,
    ));
    let body = json(&chat(&fx, &tools(r#","max_tokens":1,"tool_choice":"required""#)), 200);
    assert!(body.contains(r#""content":"""#), "{body}");
    assert!(!body.contains("tool_calls"), "{body}");
    assert!(body.contains(r#""finish_reason":"length""#), "{body}");
    assert_closed_once(&fx);
    fx.finish();
}

// Structured output.

#[test]
fn chat_structured_output_valid_invalid_and_truncated() {
    let fx = Fixture::with(limits(), Some(2));
    let format = r#","response_format":{"type":"json_schema","json_schema":{"name":"p","schema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}"#;
    fx.backend.set_script(tokens(&[r#"{"name":"#, r#""Ada"}"#], true));
    let body = json(&chat(&fx, format), 200);
    assert!(body.contains(r#""content":"{\"name\":\"Ada\"}""#), "{body}");
    assert!(body.contains(r#""finish_reason":"stop""#), "{body}");

    fx.backend.set_script(tokens(&[r#"{"name":"#], true));
    let response = chat(&fx, format);
    assert_eq!(response.status, 500, "{}", text(&response));
    let body = norm(text(&response));
    assert!(body.contains(r#""code":"invalid_structured_output""#), "{body}");
    assert!(
        body.contains("not valid JSON for the requested response format"),
        "{body}"
    );

    // Cut by the budget, the document is simply unfinished: length, not an error.
    fx.backend.set_script(tokens(&[r#"{"name":"#, r#""Ada"}"#], false));
    let body = json(&chat(&fx, &format!(r#"{format},"max_tokens":1"#)), 200);
    assert!(body.contains(r#""finish_reason":"length""#), "{body}");
    assert_closed_once(&fx);
    fx.finish();
}

// The other surfaces.

#[test]
fn messages_surface() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello", ", ", "world"], true));
    let ask = |extra: &str| {
        let body = format!(
            r#"{{"model":"{MODEL_ID}","max_tokens":16,"thinking":{{"type":"disabled"}}{extra},"messages":[{{"role":"user","content":"Hi"}}]}}"#
        );
        fx.post(MESSAGES, JSON, &body)
    };
    let response = ask("");
    let prompt = prompt_tokens(&fx, 0);
    let body = json(&response, 200);
    let expected = format!(
        r#"{{"id":"ID","type":"message","role":"assistant","model":"{MODEL_ID}","content":[{{"type":"text","text":"Hello, world"}}],"stop_reason":"end_turn","stop_sequence":null,"usage":{{"input_tokens":{prompt},"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":4}}}}"#
    );
    assert_eq!(body, expected);

    let body = json(&ask(r#","stop_sequences":["wor"]"#), 200);
    assert!(body.contains(r#""stop_reason":"stop_sequence""#), "{body}");
    assert!(body.contains(r#""stop_sequence":"wor""#), "{body}");
    assert!(body.contains(r#""text":"Hello, ""#), "{body}");

    let generated = call("get_weather", &[("city", "Paris")]);
    fx.backend.set_script(tokens(&[&generated], true));
    let tool = r#","tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;
    let body = json(&ask(tool), 200);
    assert!(
        body.contains(r#""content":[{"type":"tool_use","id":"ID","name":"get_weather","input":{"city":"Paris"}}]"#),
        "{body}"
    );
    assert!(body.contains(r#""stop_reason":"tool_use""#), "{body}");

    fx.backend.set_script(tokens(&["Hello", ", ", "world"], true));
    let short = format!(
        r#"{{"model":"{MODEL_ID}","max_tokens":2,"thinking":{{"type":"disabled"}},"messages":[{{"role":"user","content":"Hi"}}]}}"#
    );
    let body = json(&fx.post(MESSAGES, JSON, &short), 200);
    assert!(body.contains(r#""stop_reason":"max_tokens""#), "{body}");

    fx.backend
        .set_script(tokens(&["\nLet me", " think.\n", "</think>", "\n\n", "Hello."], true));
    let thinking = format!(
        r#"{{"model":"{MODEL_ID}","max_tokens":64,"thinking":{{"type":"enabled","budget_tokens":32}},"messages":[{{"role":"user","content":"Hi"}}]}}"#
    );
    let body = json(&fx.post(MESSAGES, JSON, &thinking), 200);
    assert!(
        body.contains(r#""content":[{"type":"thinking","thinking":"Let me think.","signature":"euhedral-"#),
        "{body}"
    );
    assert!(body.contains(r#"{"type":"text","text":"Hello."}"#), "{body}");
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn responses_surface() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello", ", ", "world"], true));
    let ask = |extra: &str| {
        let body = format!(r#"{{"model":"{MODEL_ID}","reasoning":{{"effort":"none"}},"input":"Hi"{extra}}}"#);
        fx.post(RESPONSES, JSON, &body)
    };
    let body = json(&ask(""), 200);
    assert!(body.contains(r#""object":"response""#), "{body}");
    assert!(body.contains(r#""status":"completed""#), "{body}");
    assert!(body.contains(r#""type":"output_text","text":"Hello, world""#), "{body}");
    assert!(body.contains(r#""output_tokens":4"#), "{body}");

    let body = json(&ask(r#","max_output_tokens":2"#), 200);
    assert!(body.contains(r#""status":"incomplete""#), "{body}");
    assert!(body.contains(r#""reason":"max_output_tokens""#), "{body}");

    let generated = call("get_weather", &[("city", "Paris")]);
    fx.backend.set_script(tokens(&[&generated], true));
    let tool = r#","tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;
    let body = json(&ask(tool), 200);
    assert!(body.contains(r#""type":"function_call""#), "{body}");
    assert!(
        body.contains(r#""name":"get_weather","arguments":"{\"city\":\"Paris\"}""#),
        "{body}"
    );

    fx.backend
        .set_script(tokens(&["\nLet me", " think.\n", "</think>", "\n\n", "Hello."], true));
    let on = format!(r#"{{"model":"{MODEL_ID}","input":"Hi"}}"#);
    let body = json(&fx.post(RESPONSES, JSON, &on), 200);
    assert!(body.contains(r#""type":"reasoning""#), "{body}");
    assert!(
        body.contains(r#""type":"reasoning""#) && body.contains("Let me think."),
        "{body}"
    );
    assert!(body.contains(r#""text":"Hello.""#), "{body}");
    assert_closed_once(&fx);
    fx.finish();
}

// Failures of the backend.

#[test]
fn backend_failures_answer_and_free_the_slot() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hi"], true));

    fx.backend
        .set_open_fault(Some(BackendError::Unavailable("gone".to_owned())));
    let response = chat(&fx, "");
    assert_eq!(response.status, 503, "{}", text(&response));
    assert!(text(&response).contains("engine_unavailable"), "{}", text(&response));
    settle(&fx);

    fx.backend.set_open_fault(Some(BackendError::Failed("boom".to_owned())));
    let response = chat(&fx, "");
    assert_eq!(response.status, 500, "{}", text(&response));
    assert!(!text(&response).contains("boom"));
    settle(&fx);
    fx.backend.set_open_fault(None);

    fx.backend.set_launch_fault(Some(LaunchFault::Fail("late".to_owned())));
    let response = chat(&fx, "");
    assert_eq!(response.status, 500, "{}", text(&response));
    assert!(!text(&response).contains("late"));
    assert_closed_once(&fx);

    fx.backend.set_launch_fault(Some(LaunchFault::Panic));
    let response = chat(&fx, "");
    assert_eq!(response.status, 500, "{}", text(&response));
    settle(&fx);
    fx.backend.set_launch_fault(None);

    // Still serving.
    assert_eq!(chat(&fx, "").status, 200);
    settle(&fx);
    fx.finish();
}

#[test]
fn unavailable_engine_is_a_503() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_available(false);
    let response = chat(&fx, "");
    assert_eq!(response.status, 503, "{}", text(&response));
    settle(&fx);
    fx.backend.set_available(true);
    fx.finish();
}

// A client that goes away, the time limits, refusal and shutdown.

fn raw_post(target: &str, body: &str) -> String {
    format!(
        "POST {target} HTTP/1.1\r\nHost: t\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

#[test]
fn disconnect_while_generating_cancels_and_frees_the_slot() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_max_active(1);
    endless(&fx);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&raw_post(CHAT, &chat_body(BIG)));
    until("generating", || generating(&fx));
    drop(raw);
    until("cancelled", || fx.backend.generations()[0].is_cancelled());
    assert_closed_once(&fx);
    fx.backend.set_script(tokens(&["Hi"], true));
    assert_eq!(chat(&fx, "").status, 200);
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn disconnect_while_queued_leaves_the_queue_and_the_next_request_runs() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_max_active(1);
    endless(&fx);
    let mut first = Raw::connect(fx.addr);
    first.send(&raw_post(CHAT, &chat_body(BIG)));
    until("generating", || generating(&fx));
    let mut second = Raw::connect(fx.addr);
    second.send(&raw_post(CHAT, &chat_body(BIG)));
    until("queued", || fx.server.admission().counts().waiting == 1);
    drop(second);
    until("left the queue", || fx.server.admission().counts().waiting == 0);
    assert_eq!(fx.backend.generations().len(), 1, "the queued request must never start");
    fx.backend.set_script(tokens(&["Hi"], true));
    drop(first);
    assert_eq!(chat(&fx, "").status, 200);
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn time_limit_while_generating_is_a_503() {
    let mut limits = limits();
    limits.request_limit = LIMIT;
    let (fx, clock) = Fixture::timed(limits);
    endless(&fx);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&raw_post(CHAT, &chat_body(BIG)));
    until("generating", || generating(&fx));
    // The limit passes while the generation runs, however long the machine took to start it.
    clock.advance(LIMIT);
    let seen = raw.read_until(r#""code":"timeout""#);
    assert!(seen.starts_with("HTTP/1.1 503"), "{seen}");
    until("cancelled", || {
        fx.backend.generations().first().is_some_and(|g| g.is_cancelled())
    });
    drop(raw);
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn time_limit_while_queued_is_a_503() {
    let mut limits = limits();
    // Only the queue limit is short: the request limit is far off.
    limits.request_limit = Duration::from_secs(3600);
    limits.queue_limit = Some(LIMIT);
    let (fx, clock) = Fixture::timed(limits);
    fx.backend.set_max_active(1);
    endless(&fx);
    let mut first = Raw::connect(fx.addr);
    first.send(&raw_post(CHAT, &chat_body(BIG)));
    until("generating", || generating(&fx));
    let mut second = Raw::connect(fx.addr);
    second.send(&raw_post(CHAT, &chat_body(BIG)));
    until("queued", || fx.server.admission().counts().waiting == 1);
    clock.advance(LIMIT);
    let seen = second.read_until(r#""code":"timeout""#);
    assert!(seen.starts_with("HTTP/1.1 503"), "{seen}");
    assert_eq!(fx.backend.generations().len(), 1);
    drop(first);
    drop(second);
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn full_queue_is_refused() {
    let mut limits = limits();
    limits.max_queued = 1;
    let fx = Fixture::with(limits, Some(2));
    fx.backend.set_max_active(1);
    endless(&fx);
    let mut first = Raw::connect(fx.addr);
    first.send(&raw_post(CHAT, &chat_body(BIG)));
    until("generating", || generating(&fx));
    let mut second = Raw::connect(fx.addr);
    second.send(&raw_post(CHAT, &chat_body(BIG)));
    until("queued", || fx.server.admission().counts().waiting == 1);
    let response = chat(&fx, BIG);
    assert_eq!(response.status, 503, "{}", text(&response));
    assert!(
        text(&response).contains("at generation capacity"),
        "{}",
        text(&response)
    );
    drop(first);
    drop(second);
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn shutdown_interrupts_a_running_generation() {
    let mut limits = limits();
    limits.drain_limit = Duration::from_millis(100);
    let fx = Fixture::with(limits, Some(2));
    endless(&fx);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&raw_post(CHAT, &chat_body(BIG)));
    until("generating", || generating(&fx));
    fx.stop();
    let seen = raw.read_until("interrupted by server shutdown");
    assert!(seen.starts_with("HTTP/1.1 503"), "{seen}");
    assert_closed_once(&fx);
    fx.finish_after_stop();
}

#[test]
fn a_backend_that_never_finishes_is_abandoned_at_shutdown() {
    let mut limits = limits();
    limits.drain_limit = Duration::from_millis(100);
    limits.shutdown_limit = Duration::from_millis(300);
    let fx = Fixture::with(limits, Some(2));
    fx.backend.set_context_length(1 << 30);
    fx.backend.set_script(Script::Hung);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&raw_post(CHAT, &chat_body(BIG)));
    until("opened", || {
        fx.backend.generations().first().is_some_and(|g| g.started().is_some())
    });
    fx.stop();
    settle(&fx);
    drop(raw);
    fx.finish_after_stop();
}

// What the listener settles, and the lifecycle around it.

#[test]
fn a_client_leaving_as_its_request_starts_still_cancels_the_generation() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_max_active(1);
    endless(&fx);
    fx.backend.hold_open();
    let mut raw = Raw::connect(fx.addr);
    raw.send(&raw_post(CHAT, &chat_body(BIG)));
    assert!(fx.backend.open_is_held(PATIENCE), "the open never started");
    drop(raw);
    // The owner has heard the client leave (the request is recorded as cancelled) before the open returns.
    until("the owner heard the client leave", || {
        fx.server
            .metrics()
            .render(MODEL_ID, 4096, &fx.backend.engine_report(), 0)
            .contains(r#"outcome="cancelled"}"#)
    });
    fx.backend.release_open();
    until("the generation was cancelled", || {
        fx.backend.generations().first().is_some_and(|g| g.is_cancelled())
    });
    assert_closed_once(&fx);
    fx.backend.set_script(tokens(&["Hi"], true));
    assert_eq!(chat(&fx, "").status, 200);
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn a_stop_sequence_cancels_inside_the_listener_so_the_counts_are_those_of_the_stop() {
    let fx = Fixture::with(limits(), Some(2));
    // Eight chunks go in one step: a cancel that came a turn later would let all of them and the stop token through.
    fx.backend.set_quantum(8);
    fx.backend
        .set_script(tokens(&["one ", "tw", "o STO", "P three", " four", " five"], true));
    let body = json(&chat(&fx, r#","stop":["STOP"],"return_token_ids":true"#), 200);
    assert!(body.contains(r#""content":"one two ""#), "{body}");
    assert!(body.contains(r#""finish_reason":"stop""#), "{body}");
    assert!(body.contains(r#""token_ids":[5000,5001,5002,5003]"#), "{body}");
    assert!(body.contains(r#""completion_tokens":4"#), "{body}");
    assert_closed_once(&fx);
    assert_eq!(fx.backend.generations()[0].emitted(), 4);
    fx.finish();
}

#[test]
fn invalid_tool_calls_cancel_inside_the_listener_too() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_quantum(8);
    fx.backend.set_script(tokens(
        &[
            "<tool_call>\n<function=nope>\n",
            "</function>\n</tool_call>",
            "more",
            "more",
        ],
        true,
    ));
    let response = chat(&fx, &tools(r#","return_token_ids":true"#));
    assert_eq!(response.status, 500, "{}", text(&response));
    assert!(text(&response).contains("invalid_tool_call"), "{}", text(&response));
    assert_closed_once(&fx);
    assert!(fx.backend.generations()[0].emitted() <= 2);
    fx.finish();
}

#[test]
fn a_panic_in_an_owner_turn_answers_a_server_error_and_frees_the_slot() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_max_active(1);
    fx.backend.set_script(tokens(&["Hi"], true));
    let faulty = format!(
        r#"{{"model":"{MODEL_ID}","reasoning_effort":"none","messages":[{{"role":"user","content":"owner-turn-fault"}}]}}"#
    );
    let response = fx.post(CHAT, JSON, &faulty);
    assert_eq!(response.status, 500, "{}", text(&response));
    assert_closed_once(&fx);
    assert_eq!(chat(&fx, "").status, 200);
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn a_panic_in_a_turn_with_the_end_already_queued_still_closes_and_frees() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_max_active(1);
    fx.backend.set_script(Script::Hung);
    let faulty = format!(
        r#"{{"model":"{MODEL_ID}","reasoning_effort":"none","messages":[{{"role":"user","content":"owner-turn-fault-late"}}]}}"#
    );
    let mut raw = Raw::connect(fx.addr);
    raw.send(&raw_post(CHAT, &faulty));
    // The turn that handles the launch is held; the generation ends meanwhile, so its end is queued behind the
    // command the turn then panics on.
    until("the launch was handled", owner_fault_is_held);
    assert!(fx.backend.generations()[0].finish_held());
    release_owner_fault();
    let seen = raw.read_until("server_error");
    assert!(seen.starts_with("HTTP/1.1 500"), "{seen}");
    assert_closed_once(&fx);
    fx.backend.set_script(tokens(&["Hi"], true));
    assert_eq!(chat(&fx, "").status, 200);
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn a_generation_that_finishes_after_it_was_abandoned_is_still_closed() {
    let mut limits = limits();
    limits.drain_limit = Duration::from_millis(100);
    limits.shutdown_limit = Duration::from_millis(300);
    let fx = Fixture::with(limits, Some(2));
    fx.backend.set_context_length(1 << 30);
    fx.backend.set_script(Script::Hung);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&raw_post(CHAT, &chat_body(BIG)));
    until("opened", || {
        fx.backend.generations().first().is_some_and(|g| g.started().is_some())
    });
    fx.stop();
    settle(&fx);
    assert_eq!(fx.backend.generations()[0].close_count(), 0);
    assert!(fx.backend.generations()[0].finish_held());
    until("closed after the late end", || {
        fx.backend.generations()[0].close_count() == 1
    });
    drop(raw);
    fx.finish_after_stop();
}

#[test]
fn anthropic_input_tokens_leave_out_the_cached_ones() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello"], true));
    fx.backend.set_counts(0, 1);
    let body = format!(
        r#"{{"model":"{MODEL_ID}","max_tokens":16,"thinking":{{"type":"disabled"}},"messages":[{{"role":"user","content":"Hi"}}]}}"#
    );
    let response = fx.post(MESSAGES, JSON, &body);
    let prompt = prompt_tokens(&fx, 0);
    let body = json(&response, 200);
    assert!(
        body.contains(&format!(
            r#""usage":{{"input_tokens":{},"cache_creation_input_tokens":0,"cache_read_input_tokens":1,"output_tokens":2}}"#,
            prompt - 1
        )),
        "{body}"
    );
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn responses_reply_body_is_exact() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello", ", ", "world"], true));
    let request = format!(r#"{{"model":"{MODEL_ID}","reasoning":{{"effort":"none"}},"input":"Hi"}}"#);
    let response = fx.post(RESPONSES, JSON, &request);
    let prompt = prompt_tokens(&fx, 0);
    let body = json(&response, 200);
    let expected = format!(
        r#"{{"id":"ID","object":"response","created_at":0,"status":"completed","error":null,"incomplete_details":null,"model":"{MODEL_ID}","output":[{{"id":"ID","type":"message","status":"completed","role":"assistant","content":[{{"type":"output_text","text":"Hello, world","annotations":[]}}]}}],"instructions":null,"max_output_tokens":null,"parallel_tool_calls":true,"temperature":null,"top_p":null,"tool_choice":"auto","tools":[],"text":{{"format":{{"type":"text"}}}},"reasoning":{{"effort":"none","summary":null}},"store":false,"metadata":{{}},"usage":{{"input_tokens":{prompt},"input_tokens_details":{{"cached_tokens":0}},"output_tokens":4,"output_tokens_details":{{"reasoning_tokens":0}},"total_tokens":{}}}}}"#,
        prompt + 4
    );
    assert_eq!(body, expected);
    fx.finish();
}

#[test]
fn responses_input_tokens_details_report_the_cached_ones() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello"], true));
    fx.backend.set_counts(0, 3);
    let request = format!(r#"{{"model":"{MODEL_ID}","reasoning":{{"effort":"none"}},"input":"Hi"}}"#);
    let response = fx.post(RESPONSES, JSON, &request);
    let prompt = prompt_tokens(&fx, 0);
    let body = json(&response, 200);
    assert!(
        body.contains(&format!(
            r#""usage":{{"input_tokens":{prompt},"input_tokens_details":{{"cached_tokens":3}},"output_tokens":2,"#
        )),
        "{body}"
    );
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn chat_tool_call_reply_body_is_exact() {
    let fx = Fixture::with(limits(), Some(2));
    let generated = call("get_weather", &[("city", "Paris"), ("days", "3")]);
    fx.backend.set_script(tokens(&[&generated], true));
    let response = chat(&fx, &tools(""));
    let prompt = prompt_tokens(&fx, 0);
    let body = json(&response, 200);
    let expected = format!(
        r#"{{"id":"ID","object":"chat.completion","created":0,"model":"{MODEL_ID}","choices":[{{"index":0,"message":{{"role":"assistant","content":null,"refusal":null,"tool_calls":[{{"id":"ID","type":"function","function":{{"name":"get_weather","arguments":"{{\"city\":\"Paris\",\"days\":3}}"}}}}]}},"logprobs":null,"finish_reason":"tool_calls"}}],{}}}"#,
        usage(prompt, 2, 0, 0)
    );
    assert_eq!(body, expected);
    assert_closed_once(&fx);
    fx.finish();
}
