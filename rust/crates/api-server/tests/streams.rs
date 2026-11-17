//! The server-sent event streams of the generating routes, over loopback, against a scripted backend.
#![allow(
    clippy::expect_used,
    clippy::disallowed_types,
    clippy::disallowed_methods,
    reason = "test fixture: the test thread is not a lattice worker; it waits with a bound"
)]

mod common;

use std::thread::sleep;
use std::time::{Duration, Instant};

use common::{Fixture, PATIENCE, Raw};
use euhedral_inference_api_server::answer::{owner_fault_is_held, release_owner_fault};
use euhedral_inference_api_server::scripted::{LaunchFault, MODEL_ID, Script};
use euhedral_inference_api_server::{Counts, Generation, Limits};

const CHAT: &str = "/v1/chat/completions";
const MESSAGES: &str = "/v1/messages";
const RESPONSES: &str = "/v1/responses";
const BIG_OUTPUT: &str = r#","max_output_tokens":100000000"#;
const BIG: &str = r#","max_tokens":100000000"#;

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

const END: &str = "\r\n0\r\n\r\n";
const HEAD: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream;charset=UTF-8\r\nCache-Control: no-cache\r\nX-Accel-Buffering: no\r\nTransfer-Encoding: chunked\r\n\r\n";

fn norm(body: &str) -> String {
    let mut out = String::new();
    let mut rest = body;
    while !rest.is_empty() {
        let mut best: Option<(usize, &str, bool)> = None;
        for (key, quoted) in [
            ("\"id\":\"", true),
            ("\"item_id\":\"", true),
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

/// The body of a chunked reply: the chunks joined.
fn dechunk(wire: &str) -> String {
    let mut out = String::new();
    let mut rest = wire;
    while let Some((size, tail)) = rest.split_once("\r\n") {
        let size = usize::from_str_radix(size.trim(), 16).expect("a chunk size");
        if size == 0 {
            break;
        }
        out.push_str(&tail[..size]);
        rest = &tail[size + 2..];
    }
    out
}

/// What a stream reply carried: the events, normalised, one string per event.
fn events(wire: &str) -> Vec<String> {
    let body = wire
        .strip_prefix(HEAD)
        .unwrap_or_else(|| panic!("a stream head in {wire:?}"));
    dechunk(body)
        .split("\n\n")
        .filter(|event| !event.is_empty())
        .map(norm)
        .collect()
}

fn post_raw(target: &str, body: &str) -> String {
    format!(
        "POST {target} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{body}",
        body.len()
    )
}

/// Sends a request and reads the whole stream.
fn stream(fx: &Fixture, target: &str, body: &str) -> Vec<String> {
    let mut raw = Raw::connect(fx.addr);
    raw.send(&post_raw(target, body));
    let wire = raw.read_until(END);
    let events = events(&wire);
    assert_closed_once(fx);
    events
}

fn chat_body(extra: &str) -> String {
    format!(
        r#"{{"model":"{MODEL_ID}","stream":true,"reasoning_effort":"none"{extra},"messages":[{{"role":"user","content":"Hi"}}]}}"#
    )
}

fn message_body(extra: &str) -> String {
    format!(
        r#"{{"model":"{MODEL_ID}","stream":true,"max_tokens":64,"thinking":{{"type":"disabled"}}{extra},"messages":[{{"role":"user","content":"Hi"}}]}}"#
    )
}

fn response_body(extra: &str) -> String {
    format!(r#"{{"model":"{MODEL_ID}","stream":true,"reasoning":{{"effort":"none"}}{extra},"input":"Hi"}}"#)
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

fn settle(fx: &Fixture) {
    let end = Instant::now() + PATIENCE;
    while fx.server.admission().counts() != Counts::default() {
        assert!(Instant::now() < end, "slots held: {:?}", fx.server.admission().counts());
        sleep(Duration::from_millis(5));
    }
}

fn until(what: &str, check: impl Fn() -> bool) {
    let end = Instant::now() + PATIENCE;
    while !check() {
        assert!(Instant::now() < end, "never: {what}");
        sleep(Duration::from_millis(5));
    }
}

/// Lets a hung generation end (a generation that is not held is left alone), then waits for every close and slot.
fn assert_closed_once(fx: &Fixture) {
    until("every generation closed", || {
        let generations = fx.backend.generations();
        for generation in &generations {
            generation.finish_held_ok(0, false);
        }
        // A queued request whose client left may start after the first ones ended (its generation is then hung with
        // a stream nobody reads): the loop goes on until no slot is held, so that one is ended too.
        generations.iter().all(|g| g.close_count() == 1) && fx.server.admission().counts() == Counts::default()
    });
    settle(fx);
}

fn joined(events: &[String]) -> String {
    events.join("\n\n")
}

fn named(events: &[String]) -> Vec<&str> {
    events
        .iter()
        .filter_map(|event| event.strip_prefix("event:").and_then(|rest| rest.lines().next()))
        .collect()
}

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

/// A generation that is launched and never ends by itself: the queue behind it stays where it is.
fn hung(fx: &Fixture) {
    fx.backend.set_context_length(1 << 30);
    fx.backend.set_script(Script::Hung);
}

fn launched(fx: &Fixture) -> bool {
    fx.backend.generations().first().is_some_and(|g| g.started().is_some())
}

fn chunk(delta: &str, finish: &str) -> String {
    format!(
        r#"data:{{"id":"ID","object":"chat.completion.chunk","created":0,"model":"{MODEL_ID}","choices":[{{"index":0,"delta":{delta},"logprobs":null,"finish_reason":{finish}}}]}}"#
    )
}

fn content(text: &str) -> String {
    chunk(&format!(r#"{{"content":"{text}"}}"#), "null")
}

fn role() -> String {
    chunk(r#"{"role":"assistant","content":""}"#, "null")
}

fn done() -> String {
    "data:[DONE]".to_owned()
}

// Chat.

#[test]
fn chat_plain_text_with_and_without_usage() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello", ", ", "world"], true));
    let plain = stream(&fx, CHAT, &chat_body(""));
    let prompt = prompt_tokens(&fx, 0);
    let expected = vec![
        role(),
        content("Hello"),
        content(", "),
        content("world"),
        chunk("{}", r#""stop""#),
        done(),
    ];
    assert_eq!(plain, expected);

    let counted = stream(&fx, CHAT, &chat_body(r#","stream_options":{"include_usage":true}"#));
    let usage = format!(
        r#"data:{{"id":"ID","object":"chat.completion.chunk","created":0,"model":"{MODEL_ID}","choices":[],"usage":{{"prompt_tokens":{prompt},"completion_tokens":4,"total_tokens":{},"prompt_tokens_details":{{"cached_tokens":0}}}}}}"#,
        prompt + 4
    );
    let mut with = expected;
    with.insert(5, usage);
    assert_eq!(counted, with);
    fx.finish();
}

#[test]
fn chat_reasoning_on_and_off() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend
        .set_script(tokens(&["\nLet me", " think.\n", "</think>", "\n\n", "Hello."], true));
    let on = format!(r#"{{"model":"{MODEL_ID}","stream":true,"messages":[{{"role":"user","content":"Hi"}}]}}"#);
    let reasoning = |text: &str| chunk(&format!(r#"{{"reasoning_content":"{text}"}}"#), "null");
    assert_eq!(
        stream(&fx, CHAT, &on),
        vec![
            role(),
            reasoning("Let me"),
            reasoning(" think."),
            content("Hello."),
            chunk("{}", r#""stop""#),
            done()
        ]
    );
    fx.backend.set_script(tokens(&["Hello.", " Bye."], true));
    let joined = joined(&stream(&fx, CHAT, &chat_body("")));
    assert!(!joined.contains("reasoning_content"), "{joined}");
    assert!(joined.contains(r#""content":"Hello.""#), "{joined}");
    fx.finish();
}

#[test]
fn chat_stop_sequence_and_a_held_back_partial_match() {
    let fx = Fixture::with(limits(), Some(2));
    let stop = r#","stop":["STOP"]"#;
    fx.backend.set_script(tokens(&["Hello ", "ST", "OP never"], true));
    let events = stream(&fx, CHAT, &chat_body(stop));
    assert_eq!(
        events,
        vec![role(), content("Hello "), chunk("{}", r#""stop""#), done()]
    );
    // A partial match that does not complete is delivered once it is known to be text.
    fx.backend.set_script(tokens(&["Hello ", "ST", "Xmore"], true));
    assert_eq!(
        stream(&fx, CHAT, &chat_body(stop)),
        vec![
            role(),
            content("Hello "),
            content("STXmore"),
            chunk("{}", r#""stop""#),
            done()
        ]
    );
    // A partial match at the very end of the text is text too.
    fx.backend.set_script(tokens(&["Hello ", "ST"], true));
    assert_eq!(
        stream(&fx, CHAT, &chat_body(stop)),
        vec![
            role(),
            content("Hello "),
            content("ST"),
            chunk("{}", r#""stop""#),
            done()
        ]
    );
    fx.finish();
}

#[test]
fn chat_length() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["a", "b", "c", "d"], false));
    let events = stream(&fx, CHAT, &chat_body(r#","max_tokens":2"#));
    assert_eq!(
        events,
        vec![role(), content("a"), content("b"), chunk("{}", r#""length""#), done()]
    );
    fx.finish();
}

#[test]
fn chat_tool_calls() {
    let fx = Fixture::with(limits(), Some(2));
    let one = call("get_weather", &[("city", "Paris"), ("days", "3")]);
    fx.backend.set_script(tokens(&[&one], true));
    let events = stream(&fx, CHAT, &chat_body(&tools("")));
    let calls = chunk(
        r#"{"tool_calls":[{"index":0,"id":"ID","type":"function","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\",\"days\":3}"}}]}"#,
        "null",
    );
    assert_eq!(events, vec![role(), calls, chunk("{}", r#""tool_calls""#), done()]);

    let generated = format!("Let me check.\n\n{}\n{}", one, call("ping", &[]));
    let (a, b) = generated.split_at(12);
    fx.backend.set_script(tokens(&[a, b], true));
    let joined = joined(&stream(&fx, CHAT, &chat_body(&tools(""))));
    assert!(joined.contains(r#""content":"Let me check""#), "{joined}");
    assert!(joined.contains(r#""index":0,"id":"ID""#), "{joined}");
    assert!(
        joined.contains(r#""index":1,"id":"ID","type":"function","function":{"name":"ping","arguments":"{}"}"#),
        "{joined}"
    );
    assert!(joined.ends_with("data:[DONE]"), "{joined}");

    // One call only.
    let two = format!("{}\n{}", call("ping", &[]), call("ping", &[]));
    fx.backend.set_script(tokens(&[&two], true));
    let events = stream(&fx, CHAT, &chat_body(&tools(r#","parallel_tool_calls":false"#)));
    let joined = self::joined(&events);
    assert_eq!(joined.matches(r#""type":"function""#).count(), 1, "{joined}");
    assert!(joined.contains(r#""finish_reason":"tool_calls""#), "{joined}");
    assert!(joined.ends_with("data:[DONE]"), "{joined}");
    fx.finish();
}

#[test]
fn an_invalid_tool_call_is_an_error_event_without_a_terminator() {
    let fx = Fixture::with(limits(), Some(2));
    let valid = call("get_weather", &[("city", "x")]);
    let cases = [
        call("nothing", &[]),
        call("get_weather", &[("days", "3")]),
        "<tool_call>\nnot a function\n</tool_call>".to_owned(),
        format!("{valid}\nAnd more text."),
    ];
    for case in &cases {
        fx.backend.set_script(tokens(&[case, "tail1", "tail2"], true));
        let events = stream(&fx, CHAT, &chat_body(&tools("")));
        let last = events.last().expect("an event");
        assert!(last.starts_with(r#"data:{"error":{"#), "{case}: {events:?}");
        assert!(last.contains(r#""code":"invalid_tool_call""#), "{last}");
        assert!(!joined(&events).contains("[DONE]"), "{events:?}");
    }
    for generation in fx.backend.generations() {
        assert!(generation.is_cancelled());
    }
    fx.finish();
}

#[test]
fn chat_structured_output() {
    let fx = Fixture::with(limits(), Some(2));
    let format = r#","response_format":{"type":"json_schema","json_schema":{"name":"p","schema":{"type":"object","properties":{"name":{"type":"string"}},"required":["name"]}}}"#;
    fx.backend.set_script(tokens(&[r#"{"name":"#, r#""Ada"}"#], true));
    let joined_ok = joined(&stream(&fx, CHAT, &chat_body(format)));
    assert!(joined_ok.contains(r#"{\"name\":"#), "{joined_ok}");
    assert!(joined_ok.contains(r#""finish_reason":"stop""#), "{joined_ok}");
    assert!(joined_ok.ends_with("data:[DONE]"), "{joined_ok}");

    fx.backend.set_script(tokens(&[r#"{"name":"#], true));
    let events = stream(&fx, CHAT, &chat_body(format));
    let last = events.last().expect("an event");
    assert!(last.contains(r#""code":"invalid_structured_output""#), "{last}");
    assert!(!joined(&events).contains("[DONE]"));

    fx.backend.set_script(tokens(&[r#"{"name":"#, r#""Ada"}"#], false));
    let joined_cut = joined(&stream(&fx, CHAT, &chat_body(&format!(r#"{format},"max_tokens":1"#))));
    assert!(joined_cut.contains(r#""finish_reason":"length""#), "{joined_cut}");
    fx.finish();
}

// Messages.

fn message_tools() -> &'static str {
    r#","tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#
}

fn usage(input: usize, output: usize) -> String {
    format!(
        r#""usage":{{"input_tokens":{input},"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"output_tokens":{output}}}"#
    )
}

fn event(name: &str, data: &str) -> String {
    format!("event:{name}\ndata:{data}")
}

fn message_start(input: usize) -> String {
    event(
        "message_start",
        &format!(
            r#"{{"type":"message_start","message":{{"id":"ID","type":"message","role":"assistant","model":"{MODEL_ID}","content":[],"stop_reason":null,"stop_sequence":null,{}}}}}"#,
            usage(input, 0)
        ),
    )
}

fn block_start(index: usize, block: &str) -> String {
    event(
        "content_block_start",
        &format!(r#"{{"type":"content_block_start","index":{index},"content_block":{block}}}"#),
    )
}

fn block_delta(index: usize, delta: &str) -> String {
    event(
        "content_block_delta",
        &format!(r#"{{"type":"content_block_delta","index":{index},"delta":{delta}}}"#),
    )
}

fn text_delta(index: usize, text: &str) -> String {
    block_delta(index, &format!(r#"{{"type":"text_delta","text":"{text}"}}"#))
}

fn block_stop(index: usize) -> String {
    event(
        "content_block_stop",
        &format!(r#"{{"type":"content_block_stop","index":{index}}}"#),
    )
}

fn message_end(reason: &str, sequence: &str, input: usize, output: usize) -> Vec<String> {
    vec![
        event(
            "message_delta",
            &format!(
                r#"{{"type":"message_delta","delta":{{"stop_reason":"{reason}","stop_sequence":{sequence}}},{}}}"#,
                usage(input, output)
            ),
        ),
        event("message_stop", r#"{"type":"message_stop"}"#),
    ]
}

fn text_block() -> String {
    block_start(0, r#"{"type":"text","text":""}"#)
}

#[test]
fn messages_plain_thinking_and_usage() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello", ", ", "world"], true));
    let events = stream(&fx, MESSAGES, &message_body(""));
    let prompt = prompt_tokens(&fx, 0);
    let mut expected = vec![
        message_start(prompt),
        text_block(),
        text_delta(0, "Hello"),
        text_delta(0, ", "),
        text_delta(0, "world"),
        block_stop(0),
    ];
    expected.extend(message_end("end_turn", "null", prompt, 4));
    assert_eq!(events, expected);

    fx.backend
        .set_script(tokens(&["\nLet me", " think.\n", "</think>", "\n\n", "Hello."], true));
    let thinking = format!(
        r#"{{"model":"{MODEL_ID}","stream":true,"max_tokens":64,"thinking":{{"type":"enabled","budget_tokens":32}},"messages":[{{"role":"user","content":"Hi"}}]}}"#
    );
    let events = stream(&fx, MESSAGES, &thinking);
    let prompt = prompt_tokens(&fx, 1);
    let signature = events[4]
        .split(r#""signature":""#)
        .nth(1)
        .and_then(|rest| rest.strip_suffix(r#""}}"#))
        .expect("a signature");
    assert!(signature.starts_with("euhedral-"), "{signature}");
    let mut expected = vec![
        message_start(prompt),
        block_start(0, r#"{"type":"thinking","thinking":"","signature":""}"#),
        block_delta(0, r#"{"type":"thinking_delta","thinking":"Let me"}"#),
        block_delta(0, r#"{"type":"thinking_delta","thinking":" think."}"#),
        block_delta(0, &format!(r#"{{"type":"signature_delta","signature":"{signature}"}}"#)),
        block_stop(0),
        block_start(1, r#"{"type":"text","text":""}"#),
        text_delta(1, "Hello."),
        block_stop(1),
    ];
    expected.extend(message_end("end_turn", "null", prompt, 6));
    assert_eq!(events, expected);
    fx.finish();
}

#[test]
fn messages_stop_sequence_length_tools_and_errors() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello ", "ST", "OP never"], true));
    let events = stream(&fx, MESSAGES, &message_body(r#","stop_sequences":["STOP"]"#));
    let prompt = prompt_tokens(&fx, 0);
    let mut expected = vec![
        message_start(prompt),
        text_block(),
        text_delta(0, "Hello "),
        block_stop(0),
    ];
    expected.extend(message_end("stop_sequence", r#""STOP""#, prompt, 3));
    assert_eq!(events, expected);

    fx.backend.set_script(tokens(&["a", "b", "c"], false));
    let body = message_body("").replace(r#""max_tokens":64"#, r#""max_tokens":2"#);
    let events = stream(&fx, MESSAGES, &body);
    let mut expected = vec![
        message_start(prompt),
        text_block(),
        text_delta(0, "a"),
        text_delta(0, "b"),
        block_stop(0),
    ];
    expected.extend(message_end("max_tokens", "null", prompt, 2));
    assert_eq!(events, expected);

    fx.backend
        .set_script(tokens(&[&call("get_weather", &[("city", "Paris")])], true));
    let events = stream(&fx, MESSAGES, &message_body(message_tools()));
    let prompt = prompt_tokens(&fx, 2);
    let mut expected = vec![
        message_start(prompt),
        block_start(0, r#"{"type":"tool_use","id":"ID","name":"get_weather","input":{}}"#),
        block_delta(
            0,
            r#"{"type":"input_json_delta","partial_json":"{\"city\":\"Paris\"}"}"#,
        ),
        block_stop(0),
    ];
    expected.extend(message_end("tool_use", "null", prompt, 2));
    assert_eq!(events, expected);

    fx.backend.set_script(tokens(&[&call("nothing", &[]), "tail"], true));
    let events = stream(&fx, MESSAGES, &message_body(message_tools()));
    assert_eq!(
        events,
        vec![
            message_start(prompt),
            event(
                "error",
                r#"{"type":"error","error":{"type":"api_error","message":"The model produced an invalid tool call: the model called a function that was not offered: nothing."}}"#
            )
        ]
    );
    fx.finish();
}

// Responses.

fn response_tools() -> &'static str {
    r#","tools":[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#
}

const WEATHER: &str = r#"[{"type":"function","name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]"#;

/// What differs between the responses of the tests: the parts of the response object an event carries.
struct Shape {
    effort: &'static str,
    tools: &'static str,
    max_output: &'static str,
}

impl Shape {
    fn response(&self, status: &str, incomplete: &str, output: &str, usage: &str) -> String {
        format!(
            r#"{{"id":"ID","object":"response","created_at":0,"status":"{status}","error":null,"incomplete_details":{incomplete},"model":"{MODEL_ID}","output":{output},"instructions":null,"max_output_tokens":{max},"parallel_tool_calls":true,"temperature":null,"top_p":null,"tool_choice":"auto","tools":{tools},"text":{{"format":{{"type":"text"}}}},"reasoning":{{"effort":"{effort}","summary":null}},"store":false,"metadata":{{}},"usage":{usage}}}"#,
            max = self.max_output,
            tools = self.tools,
            effort = self.effort
        )
    }

    /// `response.created` and `response.in_progress`.
    fn opening(&self) -> Vec<String> {
        let open = self.response("in_progress", "null", "[]", "null");
        vec![
            numbered("response.created", 0, &format!(r#""response":{open}"#)),
            numbered("response.in_progress", 1, &format!(r#""response":{open}"#)),
        ]
    }

    fn closing(&self, name: &str, number: usize, reason: &str, output: &str, input: usize, out: usize) -> String {
        let (status, incomplete) = if name == "response.completed" {
            ("completed", "null".to_owned())
        } else {
            ("incomplete", format!(r#"{{"reason":"{reason}"}}"#))
        };
        let usage = format!(
            r#"{{"input_tokens":{input},"input_tokens_details":{{"cached_tokens":0}},"output_tokens":{out},"output_tokens_details":{{"reasoning_tokens":0}},"total_tokens":{}}}"#,
            input + out
        );
        let response = self.response(status, &incomplete, output, &usage);
        numbered(name, number, &format!(r#""response":{response}"#))
    }
}

/// An event of the Responses stream: its name, its sequence number and the rest of its fields.
fn numbered(name: &str, number: usize, fields: &str) -> String {
    event(
        name,
        &format!(r#"{{"type":"{name}","sequence_number":{number},{fields}}}"#),
    )
}

fn message_item(text: &str, status: &str) -> String {
    let content = if status == "completed" {
        format!(r#"[{{"type":"output_text","text":"{text}","annotations":[]}}]"#)
    } else {
        "[]".to_owned()
    };
    format!(r#"{{"id":"ID","type":"message","status":"{status}","role":"assistant","content":{content}}}"#)
}

/// The events of a message item that carries `deltas` as output `index`, starting at sequence number `first`.
fn message_events(index: usize, first: usize, deltas: &[&str]) -> Vec<String> {
    let at = format!(r#""item_id":"ID","output_index":{index},"content_index":0"#);
    let text = deltas.concat();
    let mut number = first;
    let mut next = || {
        number += 1;
        number - 1
    };
    let mut out = vec![
        numbered(
            "response.output_item.added",
            next(),
            &format!(r#""output_index":{index},"item":{}"#, message_item("", "in_progress")),
        ),
        numbered(
            "response.content_part.added",
            next(),
            &format!(r#"{at},"part":{{"type":"output_text","text":"","annotations":[]}}"#),
        ),
    ];
    for delta in deltas {
        out.push(numbered(
            "response.output_text.delta",
            next(),
            &format!(r#"{at},"delta":"{delta}""#),
        ));
    }
    out.push(numbered(
        "response.output_text.done",
        next(),
        &format!(r#"{at},"text":"{text}""#),
    ));
    out.push(numbered(
        "response.content_part.done",
        next(),
        &format!(r#"{at},"part":{{"type":"output_text","text":"{text}","annotations":[]}}"#),
    ));
    out.push(numbered(
        "response.output_item.done",
        next(),
        &format!(r#""output_index":{index},"item":{}"#, message_item(&text, "completed")),
    ));
    out
}

#[test]
fn responses_plain_reasoning_and_sequence_numbers() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["Hello", ", ", "world"], true));
    let events = stream(&fx, RESPONSES, &response_body(""));
    let prompt = prompt_tokens(&fx, 0);
    let off = Shape {
        effort: "none",
        tools: "[]",
        max_output: "null",
    };
    let mut expected = off.opening();
    expected.extend(message_events(0, 2, &["Hello", ", ", "world"]));
    expected.push(off.closing(
        "response.completed",
        10,
        "",
        &format!("[{}]", message_item("Hello, world", "completed")),
        prompt,
        4,
    ));
    assert_eq!(events, expected);

    fx.backend
        .set_script(tokens(&["\nLet me", " think.\n", "</think>", "\n\n", "Hello."], true));
    let on = format!(r#"{{"model":"{MODEL_ID}","stream":true,"input":"Hi"}}"#);
    let events = stream(&fx, RESPONSES, &on);
    let prompt = prompt_tokens(&fx, 1);
    let high = Shape {
        effort: "xhigh",
        tools: "[]",
        max_output: "null",
    };
    let at = r#""item_id":"ID","output_index":0,"content_index":0"#;
    let reasoning =
        r#"{"id":"ID","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"Let me think."}]}"#;
    let mut expected = high.opening();
    expected.extend([
        numbered(
            "response.output_item.added",
            2,
            r#""output_index":0,"item":{"id":"ID","type":"reasoning","summary":[],"content":[]}"#,
        ),
        numbered(
            "response.content_part.added",
            3,
            &format!(r#"{at},"part":{{"type":"reasoning_text","text":""}}"#),
        ),
        numbered("response.reasoning_text.delta", 4, &format!(r#"{at},"delta":"Let me""#)),
        numbered(
            "response.reasoning_text.delta",
            5,
            &format!(r#"{at},"delta":" think.""#),
        ),
        numbered(
            "response.reasoning_text.done",
            6,
            &format!(r#"{at},"text":"Let me think.""#),
        ),
        numbered(
            "response.content_part.done",
            7,
            &format!(r#"{at},"part":{{"type":"reasoning_text","text":"Let me think."}}"#),
        ),
        numbered(
            "response.output_item.done",
            8,
            &format!(r#""output_index":0,"item":{reasoning}"#),
        ),
    ]);
    expected.extend(message_events(1, 9, &["Hello."]));
    expected.push(high.closing(
        "response.completed",
        15,
        "",
        &format!("[{reasoning},{}]", message_item("Hello.", "completed")),
        prompt,
        6,
    ));
    assert_eq!(events, expected);
    fx.finish();
}

#[test]
fn responses_length_tools_and_errors() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(tokens(&["a", "b", "c"], false));
    let events = stream(&fx, RESPONSES, &response_body(r#","max_output_tokens":2"#));
    let prompt = prompt_tokens(&fx, 0);
    let capped = Shape {
        effort: "none",
        tools: "[]",
        max_output: "2",
    };
    let mut expected = capped.opening();
    expected.extend(message_events(0, 2, &["a", "b"]));
    expected.push(capped.closing(
        "response.incomplete",
        9,
        "max_output_tokens",
        &format!("[{}]", message_item("ab", "completed")),
        prompt,
        2,
    ));
    assert_eq!(events, expected);

    fx.backend
        .set_script(tokens(&[&call("get_weather", &[("city", "Paris")])], true));
    let events = stream(&fx, RESPONSES, &response_body(response_tools()));
    let prompt = prompt_tokens(&fx, 1);
    let offered = Shape {
        effort: "none",
        tools: WEATHER,
        max_output: "null",
    };
    let arguments = r#"{\"city\":\"Paris\"}"#;
    let item = |status: &str, arguments: &str| {
        format!(
            r#"{{"id":"ID","type":"function_call","status":"{status}","call_id":"ID","name":"get_weather","arguments":"{arguments}"}}"#
        )
    };
    let mut expected = offered.opening();
    expected.extend([
        numbered(
            "response.output_item.added",
            2,
            &format!(r#""output_index":0,"item":{}"#, item("in_progress", "")),
        ),
        numbered(
            "response.function_call_arguments.delta",
            3,
            &format!(r#""item_id":"ID","output_index":0,"delta":"{arguments}""#),
        ),
        numbered(
            "response.function_call_arguments.done",
            4,
            &format!(r#""item_id":"ID","output_index":0,"arguments":"{arguments}""#),
        ),
        numbered(
            "response.output_item.done",
            5,
            &format!(r#""output_index":0,"item":{}"#, item("completed", arguments)),
        ),
        offered.closing(
            "response.completed",
            6,
            "",
            &format!("[{}]", item("completed", arguments)),
            prompt,
            2,
        ),
    ]);
    assert_eq!(events, expected);

    fx.backend.set_script(tokens(&[&call("nothing", &[]), "tail"], true));
    let events = stream(&fx, RESPONSES, &response_body(response_tools()));
    let mut expected = offered.opening();
    expected.push(numbered(
        "error",
        2,
        r#""code":"invalid_tool_call","message":"The model produced an invalid tool call: the model called a function that was not offered: nothing.","param":null"#,
    ));
    assert_eq!(events, expected);
    fx.finish();
}

// Chunking.

/// An event without the numbers that follow how the text was cut: the sequence number and the output tokens.
fn unnumbered(event: &str) -> String {
    let mut out = String::new();
    let mut rest = event;
    while let Some((before, after)) = rest
        .split_once(r#""sequence_number":"#)
        .or_else(|| rest.split_once(r#""output_tokens":"#))
    {
        let key = if rest[before.len()..].starts_with(r#""sequence_number":"#) {
            r#""sequence_number":"#
        } else {
            r#""output_tokens":"#
        };
        out.push_str(before);
        out.push_str(key);
        out.push('0');
        rest = after.trim_start_matches(|c: char| c.is_ascii_digit());
    }
    out.push_str(rest);
    out
}

#[test]
fn the_chunking_of_the_generated_text_does_not_change_the_stream() {
    let fx = Fixture::with(limits(), Some(2));
    let text = "\nLet me think.\n</think>\n\nHello, wor\u{e9}ld! STOP and more";
    let whole = [text];
    let chars: Vec<String> = text.chars().map(String::from).collect();
    let pieces: Vec<&str> = chars.iter().map(String::as_str).collect();
    let mut seen = Vec::new();
    for chunks in [&whole[..], &pieces[..]] {
        fx.backend.set_script(tokens(chunks, true));
        let reasoning = format!(
            r#"{{"model":"{MODEL_ID}","stream":true,"stop":["STOP"],"messages":[{{"role":"user","content":"Hi"}}]}}"#
        );
        let resp = format!(r#"{{"model":"{MODEL_ID}","stream":true,"input":"Hi"}}"#);
        let thinking = format!(
            r#"{{"model":"{MODEL_ID}","stream":true,"max_tokens":64,"stop_sequences":["STOP"],"thinking":{{"type":"enabled","budget_tokens":32}},"messages":[{{"role":"user","content":"Hi"}}]}}"#
        );
        let chat = stream(&fx, CHAT, &reasoning);
        let message = stream(&fx, MESSAGES, &thinking);
        let response = stream(&fx, RESPONSES, &resp);
        // What the deltas add up to, and the closing events, do not depend on where the text was cut.
        let deltas = |events: &[String], key: &str| -> String {
            events
                .iter()
                .filter_map(|event| event.split_once(key).and_then(|(_, tail)| tail.split_once('"')))
                .map(|(text, _)| text.to_owned())
                .collect()
        };
        seen.push((
            deltas(&chat, r#""reasoning_content":""#),
            deltas(&chat, r#""delta":{"content":""#),
            chat[chat.len() - 2..].to_vec(),
            deltas(&message, r#""thinking_delta","thinking":""#),
            deltas(&message, r#""text_delta","text":""#),
            message[message.len() - 2..]
                .iter()
                .map(|e| unnumbered(e))
                .collect::<Vec<_>>(),
            response
                .iter()
                .filter(|e| e.starts_with("event:response.output_item.done"))
                .map(|e| unnumbered(e))
                .collect::<Vec<_>>(),
        ));
    }
    assert_eq!(seen[0], seen[1]);
    assert_eq!(seen[0].0, "Let me think.");
    assert!(seen[0].1.starts_with("Hello"), "{:?}", seen[0].1);
    assert!(!seen[0].1.contains("STOP"));
    fx.finish();
}

// Keep-alive.

#[test]
fn keep_alive_comments_and_pings_come_with_prefill_quanta() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_prefill_quanta(3);
    fx.backend.set_script(tokens(&["Hi"], true));
    let chat = joined(&stream(&fx, CHAT, &chat_body("")));
    assert!(chat.matches(":keep-alive").count() >= 1, "{chat}");
    assert!(chat.find(":keep-alive") < chat.find("chat.completion.chunk"), "{chat}");
    let responses = joined(&stream(&fx, RESPONSES, &response_body("")));
    assert!(responses.contains(":keep-alive"), "{responses}");
    let events = stream(&fx, MESSAGES, &message_body(""));
    assert!(named(&events).contains(&"ping"), "{events:?}");
    assert_eq!(named(&events)[0], "ping");
    assert!(events[0].ends_with(r#"data:{"type":"ping"}"#), "{}", events[0]);
    assert!(!joined(&events).contains("keep-alive"));
    fx.finish();
}

// Ends before the stream.

#[test]
fn errors_before_the_head_are_json_replies() {
    let mut limits = limits();
    limits.max_queued = 1;
    let fx = Fixture::with(limits, Some(2));
    fx.backend.set_max_active(1);
    hung(&fx);
    let mut first = Raw::connect(fx.addr);
    first.send(&post_raw(CHAT, &chat_body(BIG)));
    until("launched", || launched(&fx));
    let mut second = Raw::connect(fx.addr);
    second.send(&post_raw(CHAT, &chat_body(BIG)));
    until("queued", || fx.server.admission().counts().waiting == 1);
    let mut third = Raw::connect(fx.addr);
    third.send(&post_raw(CHAT, &chat_body(BIG)));
    let seen = third.read_until("at generation capacity");
    assert!(seen.starts_with("HTTP/1.1 503"), "{seen}");
    assert!(seen.contains("Content-Type: application/json"), "{seen}");
    drop((first, second, third));
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn time_limit_while_queued_is_a_json_503() {
    let mut limits = limits();
    limits.request_limit = Duration::from_secs(15);
    limits.queue_limit = Some(Duration::from_millis(300));
    let fx = Fixture::with(limits, Some(2));
    fx.backend.set_max_active(1);
    hung(&fx);
    let mut first = Raw::connect(fx.addr);
    first.send(&post_raw(CHAT, &chat_body(BIG)));
    until("launched", || launched(&fx));
    let mut second = Raw::connect(fx.addr);
    second.send(&post_raw(CHAT, &chat_body(BIG)));
    let seen = second.read_until(r#""code":"timeout""#);
    assert!(seen.starts_with("HTTP/1.1 503"), "{seen}");
    assert!(!seen.contains("text/event-stream"), "{seen}");
    assert_eq!(fx.backend.generations().len(), 1);
    drop(first);
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn disconnect_while_queued_leaves_the_queue() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_max_active(1);
    hung(&fx);
    let mut first = Raw::connect(fx.addr);
    first.send(&post_raw(CHAT, &chat_body(BIG)));
    until("launched", || launched(&fx));
    let mut second = Raw::connect(fx.addr);
    second.send(&post_raw(CHAT, &chat_body(BIG)));
    until("queued", || fx.server.admission().counts().waiting == 1);
    drop(second);
    until("left the queue", || fx.server.admission().counts().waiting == 0);
    assert_eq!(fx.backend.generations().len(), 1);
    drop(first);
    assert_closed_once(&fx);
    fx.finish();
}

// Ends inside the stream.

#[test]
fn disconnect_while_generating_cancels() {
    // The generation says a word every few milliseconds, as a model does; a generator that never waits can say
    // more than the owner takes in on a loaded machine, and then it is the bound that ends the request.
    let fx = Fixture::with(limits(), Some(2));
    hung(&fx);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&post_raw(CHAT, &chat_body(BIG)));
    raw.read_until(HEAD);
    until("launched", || launched(&fx));
    assert!(fx.backend.generations()[0].emit_held("tok0 "));
    let seen = raw.read_until("tok0");
    assert!(seen.contains("tok0"), "{seen}");
    drop(raw);
    until("cancelled", || {
        fx.backend.generations()[0].emit_held("tok ");
        fx.backend.generations()[0].is_cancelled()
    });
    assert!(fx.backend.generations()[0].finish_held_ok(1, true));
    assert_closed_once(&fx);
    fx.backend.set_script(tokens(&["Hi"], true));
    assert!(!stream(&fx, CHAT, &chat_body("")).is_empty());
    fx.finish();
}

/// Feeds a hung generation 64 KiB pieces until `stop` holds, within the patience of a test.
fn feed(fx: &Fixture, stop: impl Fn() -> bool) {
    let piece = "a".repeat(64 << 10);
    let end = Instant::now() + PATIENCE;
    while !stop() {
        assert!(
            Instant::now() < end,
            "never reached: {} emitted, {:?}",
            fx.backend.generations()[0].emitted(),
            fx.server.admission().counts()
        );
        assert!(fx.backend.generations()[0].emit_held(&piece));
        sleep(Duration::from_millis(1));
    }
}

#[test]
fn disconnect_during_a_write_cancels() {
    let mut limits = limits();
    limits.max_held_bytes = 1 << 30;
    let fx = Fixture::with(limits, Some(2));
    fx.backend.set_script(Script::Hung);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&post_raw(CHAT, &chat_body("")));
    until("launched", || {
        fx.backend.generations().first().is_some_and(|g| g.started().is_some())
    });
    // The client reads nothing: the first writes fill the socket and one stays in flight.
    for _ in 0..200 {
        assert!(fx.backend.generations()[0].emit_held(&"a".repeat(64 << 10)));
    }
    sleep(Duration::from_millis(200));
    drop(raw);
    until("cancelled", || fx.backend.generations()[0].is_cancelled());
    assert!(fx.backend.generations()[0].finish_held_ok(1, true));
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn a_reader_too_slow_for_the_held_bound_ends_the_request() {
    let mut limits = limits();
    limits.max_held_bytes = 256 << 10;
    let fx = Fixture::with(limits, Some(2));
    fx.backend.set_script(Script::Hung);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&post_raw(CHAT, &chat_body("")));
    until("launched", || {
        fx.backend.generations().first().is_some_and(|g| g.started().is_some())
    });
    feed(&fx, || fx.backend.generations()[0].is_cancelled());
    assert!(fx.backend.generations()[0].finish_held_ok(1, true));
    // The stream is cut: whatever the client reads, it never gets the closing chunk.
    let mut seen = String::new();
    let end = Instant::now() + PATIENCE;
    while Instant::now() < end {
        match raw.read_for(Duration::from_millis(200)) {
            Some(more) => seen.push_str(&more),
            None => break,
        }
        assert!(!seen.ends_with(END), "the stream ended cleanly");
    }
    assert!(!seen.ends_with(END));
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn time_limit_while_generating_ends_the_stream_with_an_error() {
    // The limit is long next to what a loaded machine takes to launch a generation, and both requests wait for it
    // together; nothing below depends on how soon the limit comes, only on its coming after the launch.
    let mut limits = limits();
    limits.request_limit = Duration::from_secs(4);
    let fx = Fixture::with(limits, Some(2));
    fx.backend.set_max_active(2);
    fx.backend.set_script(Script::Hung);
    let mut raws = Vec::new();
    for (index, (target, body)) in [(CHAT, chat_body("")), (MESSAGES, message_body(""))]
        .into_iter()
        .enumerate()
    {
        let mut raw = Raw::connect(fx.addr);
        raw.send(&post_raw(target, &body));
        let head = raw.read_until(HEAD);
        until("launched", || {
            fx.backend
                .generations()
                .get(index)
                .is_some_and(|g| g.started().is_some())
        });
        raws.push((target, raw, head));
    }
    for (index, (target, mut raw, head)) in raws.into_iter().enumerate() {
        let seen = head + &raw.read_until(END);
        assert!(seen.starts_with(HEAD), "{seen}");
        if target == CHAT {
            assert!(seen.contains(r#""code":"timeout""#), "{seen}");
            assert!(!seen.contains("[DONE]"));
        } else {
            assert!(
                seen.contains("event:error\ndata:{\"type\":\"error\",\"error\":{\"type\":\"overloaded_error\""),
                "{seen}"
            );
        }
        until("cancelled", || fx.backend.generations()[index].is_cancelled());
    }
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn shutdown_ends_the_stream_with_an_error() {
    let mut limits = limits();
    limits.drain_limit = Duration::from_millis(100);
    limits.shutdown_limit = Duration::from_millis(300);
    let fx = Fixture::with(limits, Some(2));
    hung(&fx);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&post_raw(RESPONSES, &response_body(BIG_OUTPUT)));
    until("launched", || launched(&fx));
    fx.stop();
    let seen = raw.read_until(END);
    assert!(seen.contains("interrupted by server shutdown"), "{seen}");
    assert!(
        seen.starts_with("HTTP/1.1 200 OK\r\nContent-Type: text/event-stream"),
        "{seen}"
    );
    assert!(fx.backend.generations()[0].finish_held_ok(1, false));
    assert_closed_once(&fx);
    fx.finish_after_stop();
}

#[test]
fn a_backend_failure_after_the_head_is_an_error_event() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(Script::Hung);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&post_raw(CHAT, &chat_body("")));
    let seen = raw.read_until(HEAD);
    assert!(seen.starts_with(HEAD), "{seen}");
    until("launched", || {
        fx.backend.generations().first().is_some_and(|g| g.started().is_some())
    });
    assert!(fx.backend.generations()[0].emit_held("Hi"));
    assert!(fx.backend.generations()[0].finish_held());
    let seen = raw.read_until(END);
    assert!(seen.contains(r#""type":"server_error""#), "{seen}");
    assert!(!seen.contains("[DONE]"), "{seen}");
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn a_panic_in_the_owner_after_the_head_ends_the_stream_and_frees_the_slot() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_max_active(1);
    fx.backend.set_script(Script::Hung);
    let faulty = format!(
        r#"{{"model":"{MODEL_ID}","stream":true,"reasoning_effort":"none","messages":[{{"role":"user","content":"owner-turn-fault-late"}}]}}"#
    );
    let mut raw = Raw::connect(fx.addr);
    raw.send(&post_raw(CHAT, &faulty));
    until("the launch was handled", owner_fault_is_held);
    assert!(fx.backend.generations()[0].finish_held());
    release_owner_fault();
    let seen = raw.read_until(END);
    assert!(seen.contains("server_error"), "{seen}");
    assert_closed_once(&fx);
    fx.backend.set_script(tokens(&["Hi"], true));
    assert!(!stream(&fx, CHAT, &chat_body("")).is_empty());
    fx.finish();
}

// What the review asked for.

/// Reads the stream of a raw connection to its end.
fn rest(raw: &mut Raw, head: String) -> Vec<String> {
    let wire = head + &raw.read_until(END);
    events(&wire)
}

#[test]
fn a_large_reasoning_block_on_a_slow_reader_still_completes() {
    let mut limits = limits();
    limits.max_held_bytes = 512 << 10;
    let fx = Fixture::with(limits, Some(2));
    fx.backend.set_script(Script::Hung);
    let on = format!(r#"{{"model":"{MODEL_ID}","stream":true,"input":"Hi"}}"#);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&post_raw(RESPONSES, &on));
    let head = raw.read_until(HEAD);
    until("launched", || launched(&fx));
    let generation = fx.backend.generations()[0].clone();
    assert!(generation.emit_held("\nthinking "));
    // The reader reads nothing while a quarter of a megabyte of reasoning is said; the events repeat it several times.
    for _ in 0..16 {
        assert!(generation.emit_held(&"a".repeat(16 << 10)));
    }
    assert!(generation.emit_held("\n</think>\n\nDone."));
    assert!(generation.finish_held_ok(3, true));
    let events = rest(&mut raw, head);
    assert!(!generation.is_cancelled());
    let names = named(&events);
    assert_eq!(names.last(), Some(&"response.completed"), "{names:?}");
    assert!(names.contains(&"response.reasoning_text.done"), "{names:?}");
    assert!(events.last().expect("an event").contains(r#""status":"completed""#));
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn text_before_a_malformed_call_is_delivered_before_the_error() {
    let fx = Fixture::with(limits(), Some(2));
    // The call is complete and wrong: the error is found as the text arrives.
    let bad = call("nothing", &[]);
    fx.backend.set_script(tokens(&["Let me check. ", &bad], true));
    let events = stream(&fx, CHAT, &chat_body(&tools("")));
    assert_eq!(events.len(), 3, "{events:?}");
    assert_eq!(events[0], role());
    assert_eq!(events[1], content("Let me check."));
    assert!(events[2].contains(r#""code":"invalid_tool_call""#), "{events:?}");
    // The call is cut off at the end: the error is found when the pipeline finishes.
    fx.backend.set_script(tokens(
        &["Let me check.", "<tool_call>\n<function=get_weather>\n"],
        true,
    ));
    let events = stream(&fx, CHAT, &chat_body(&tools("")));
    assert_eq!(events[0], role(), "{events:?}");
    assert_eq!(events[1], content("Let me check."), "{events:?}");
    let last = events.last().expect("an event");
    assert!(last.contains(r#""code":"invalid_tool_call""#), "{events:?}");
    assert!(!joined(&events).contains("[DONE]"), "{events:?}");
    fx.finish();
}

#[test]
fn a_half_closed_client_gets_the_whole_stream() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(Script::Hung);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&post_raw(CHAT, &chat_body("")));
    let head = raw.read_until(HEAD);
    until("launched", || launched(&fx));
    raw.half_close();
    let generation = fx.backend.generations()[0].clone();
    assert!(generation.emit_held("one "));
    sleep(Duration::from_millis(200));
    assert!(generation.emit_held("two"));
    sleep(Duration::from_millis(200));
    assert!(!generation.is_cancelled());
    assert!(generation.finish_held_ok(2, true));
    let events = rest(&mut raw, head);
    assert_eq!(
        events,
        vec![
            role(),
            content("one "),
            content("two"),
            chunk("{}", r#""stop""#),
            done()
        ]
    );
    assert!(!generation.is_cancelled());
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn a_client_that_leaves_during_prefill_is_found_by_a_keep_alive() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(Script::Hung);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&post_raw(CHAT, &chat_body("")));
    raw.read_until(HEAD);
    until("launched", || launched(&fx));
    drop(raw);
    // Each quantum writes a keep-alive; the write that fails ends the request.
    until("cancelled", || {
        fx.backend.generations()[0].prefill_held();
        fx.backend.generations()[0].is_cancelled()
    });
    assert!(fx.backend.generations()[0].finish_held_ok(0, false));
    assert_closed_once(&fx);
    assert_eq!(fx.backend.generations().len(), 1);
    fx.finish();
}

#[test]
fn a_launch_failure_after_the_head_is_an_error_event() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_launch_fault(Some(LaunchFault::Fail("late".to_owned())));
    let events = stream(&fx, CHAT, &chat_body(""));
    assert_eq!(events.len(), 1, "{events:?}");
    assert!(events[0].starts_with(r#"data:{"error":{"#), "{events:?}");
    assert!(events[0].contains(r#""type":"server_error""#), "{events:?}");
    assert!(!events[0].contains("late"), "{events:?}");
    fx.backend.set_launch_fault(None);
    fx.backend.set_script(tokens(&["Hi"], true));
    assert!(!stream(&fx, CHAT, &chat_body("")).is_empty());
    fx.finish();
}

#[test]
fn text_arrives_as_it_is_generated() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_script(Script::Hung);
    let mut raw = Raw::connect(fx.addr);
    raw.send(&post_raw(CHAT, &chat_body("")));
    let mut seen = raw.read_until(HEAD);
    until("launched", || launched(&fx));
    let generation = fx.backend.generations()[0].clone();
    for word in ["alpha", "beta", "gamma"] {
        assert!(generation.emit_held(word));
        seen += &raw.read_until(&format!(r#""content":"{word}""#));
        assert!(!seen.ends_with(END));
    }
    assert!(generation.finish_held_ok(3, true));
    let events = rest(&mut raw, seen);
    assert_eq!(events.last(), Some(&done()));
    assert_closed_once(&fx);
    fx.finish();
}

#[test]
fn a_queued_request_that_starts_after_its_client_left_is_released() {
    let fx = Fixture::with(limits(), Some(2));
    fx.backend.set_max_active(1);
    hung(&fx);
    let mut first = Raw::connect(fx.addr);
    first.send(&post_raw(CHAT, &chat_body(BIG)));
    until("launched", || launched(&fx));
    let mut second = Raw::connect(fx.addr);
    second.send(&post_raw(CHAT, &chat_body(BIG)));
    until("queued", || fx.server.admission().counts().waiting == 1);
    // The first ends while the second is still connected, so the second starts; then its client leaves. Its stream
    // is open and nothing is written to it until text comes: the slot is released when the generation ends.
    assert!(fx.backend.generations()[0].finish_held_ok(1, true));
    until("second launched", || fx.backend.generations().len() == 2);
    drop((first, second));
    assert_closed_once(&fx);
    fx.finish();
}
