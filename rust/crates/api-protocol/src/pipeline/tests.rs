//! The pipeline as a whole: what a request's output turns into, and that ordinary output does not depend on how the
//! text is cut.

use std::iter::once;
use std::sync::mpsc::channel;
use std::thread::spawn;
use std::time::Duration;

use super::*;
use crate::json::JsonValue;

const CALL_F: &str = "<tool_call>\n<function=f>\n</function>\n</tool_call>";
const SCHEMA: &str = r#"{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}"#;

fn tool(name: &str, schema: Option<&str>) -> ToolDefinition {
    ToolDefinition {
        name: name.to_string(),
        description: None,
        parameters: schema.map(|text| JsonValue::parse(text).unwrap()),
    }
}

fn tools(choice: ToolChoice, parallel: bool) -> Option<ToolOptions> {
    Some(ToolOptions {
        tools: vec![tool("f", None), tool("get_weather", Some(SCHEMA))],
        choice,
        parallel,
    })
}

fn options(reasoning: bool, stops: &[&str], tools: Option<ToolOptions>) -> PipelineOptions {
    PipelineOptions {
        reasoning,
        stops: stops.iter().map(|stop| (*stop).to_string()).collect(),
        tools,
    }
}

fn reasoning(text: &str) -> Piece {
    Piece::Reasoning(text.to_string())
}

fn text(text: &str) -> Piece {
    Piece::Text(text.to_string())
}

fn call(name: &str, arguments: &str) -> Piece {
    Piece::ToolCall(ToolCall {
        name: name.to_string(),
        arguments: arguments.to_string(),
    })
}

/// What a run of a pipeline produced, with adjacent pieces of one kind joined.
#[derive(Debug, PartialEq)]
struct Outcome {
    pieces: Vec<Piece>,
    error: Option<String>,
    matched_stop: Option<String>,
    tool_calls: usize,
    reasoning_open: bool,
    ended: bool,
}

fn add(pieces: &mut Vec<Piece>, piece: Piece) {
    match (pieces.last_mut(), piece) {
        (Some(Piece::Reasoning(last)), Piece::Reasoning(next)) => last.push_str(&next),
        (Some(Piece::Text(last)), Piece::Text(next)) => last.push_str(&next),
        (_, piece) => pieces.push(piece),
    }
}

/// Feeds the chunks in order, then finishes (at the stop token, or truncated), stopping at the first error.
fn run(options: &PipelineOptions, chunks: &[&str], stop_token: bool) -> Outcome {
    let mut pipeline = TextPipeline::new(options);
    let mut pieces = Vec::new();
    let mut error = None;
    for chunk in chunks {
        match pipeline.push(chunk) {
            Ok(settled) => settled.into_iter().for_each(|piece| add(&mut pieces, piece)),
            Err(PipelineError::MalformedToolCall { message, emitted }) => {
                emitted.into_iter().for_each(|piece| add(&mut pieces, piece));
                error = Some(message);
                break;
            }
        }
    }
    if error.is_none() {
        let finished = if stop_token {
            pipeline.finish()
        } else {
            pipeline.finish_truncated()
        };
        match finished {
            Ok(settled) => settled.into_iter().for_each(|piece| add(&mut pieces, piece)),
            Err(PipelineError::MalformedToolCall { message, emitted }) => {
                emitted.into_iter().for_each(|piece| add(&mut pieces, piece));
                error = Some(message);
            }
        }
    }
    Outcome {
        pieces,
        error,
        matched_stop: pipeline.matched_stop().map(str::to_string),
        tool_calls: pipeline.tool_calls(),
        reasoning_open: pipeline.reasoning_open(),
        ended: pipeline.ended(),
    }
}

struct Case {
    name: &'static str,
    options: PipelineOptions,
    input: String,
    stop_token: bool,
}

fn case(name: &'static str, options: PipelineOptions, input: &str, stop_token: bool) -> Case {
    Case {
        name,
        options,
        input: input.to_string(),
        stop_token,
    }
}

fn weather_call(city: &str) -> String {
    format!("<tool_call>\n<function=get_weather>\n<parameter=city>\n{city}\n</parameter>\n</function>\n</tool_call>")
}

fn cases() -> Vec<Case> {
    let two_calls = format!(
        "Let me check.\n\n{}\n{}\n{CALL_F}",
        weather_call("Line one\nline two"),
        weather_call("Paris")
    );
    vec![
        case(
            "reasoning split",
            options(true, &[], None),
            " \nFirst <b>step</b>, then </thin k.\n\n</think>\n\n Answer </think> stays.",
            true,
        ),
        case(
            "stop in the middle",
            options(false, &["END"], None),
            "hello END tail",
            true,
        ),
        case(
            "earliest of two stops",
            options(false, &["zz", "\n\n"], None),
            "one\n\ntwo zz",
            true,
        ),
        case(
            "held stop prefix at the end",
            options(false, &["STOP"], None),
            "xST",
            true,
        ),
        case(
            "plain text with tools",
            options(false, &[], tools(ToolChoice::Auto, true)),
            "It is cold <tool",
            true,
        ),
        case(
            "content then call",
            options(false, &[], tools(ToolChoice::Auto, true)),
            &format!("It is cold {CALL_F}"),
            true,
        ),
        case(
            "several calls",
            options(false, &[], tools(ToolChoice::Auto, true)),
            &two_calls,
            true,
        ),
        case(
            "no parallel calls",
            options(false, &[], tools(ToolChoice::Auto, false)),
            &format!("{CALL_F}\n{CALL_F} more"),
            true,
        ),
        case(
            "reasoning then a call",
            options(true, &[], tools(ToolChoice::Auto, true)),
            &format!("Need the weather.\n</think>\n\n{}", weather_call("Paris")),
            true,
        ),
        case(
            "stop applies to the answer only",
            options(true, &["STOP"], None),
            "I will say STOP later.</think>\n\nGo STOP now",
            true,
        ),
        case(
            "stop before a call",
            options(false, &["END"], tools(ToolChoice::Auto, true)),
            &format!("Done. END {CALL_F} trailing text"),
            true,
        ),
        case(
            "stop matching the call tag",
            options(false, &["<tool_call>"], tools(ToolChoice::Auto, true)),
            &format!("Before {}", weather_call("Paris")),
            true,
        ),
        case(
            "held tag prefix and stop at the end",
            options(false, &["<tool_"], tools(ToolChoice::Auto, true)),
            "Answer <tool_",
            true,
        ),
        case(
            "delimiters in a value",
            options(false, &[], tools(ToolChoice::Auto, true)),
            &weather_call("before <parameter=x>\n2 </paramete\n</tool_call> after"),
            true,
        ),
        case(
            "unknown function",
            options(false, &[], tools(ToolChoice::Auto, true)),
            "<tool_call>\n<function=g>\n</function>\n</tool_call>tail",
            true,
        ),
        case(
            "text after a call",
            options(false, &[], tools(ToolChoice::Auto, true)),
            &format!("{CALL_F}\nAnd more text."),
            true,
        ),
        case(
            "unfinished call",
            options(false, &[], tools(ToolChoice::Auto, true)),
            "<tool_call>\n<function=f>\n<parameter=n>\n",
            true,
        ),
        case(
            "truncated call",
            options(false, &[], tools(ToolChoice::Auto, true)),
            &format!("{CALL_F}\n<tool_call>\n<function=f>\n<param"),
            false,
        ),
        case(
            "required without a call",
            options(false, &[], tools(ToolChoice::Required, true)),
            "No call.",
            true,
        ),
        case(
            "required, truncated",
            options(false, &[], tools(ToolChoice::Required, true)),
            "No call.",
            false,
        ),
        case(
            "named function",
            options(false, &[], tools(ToolChoice::Function("f".to_string()), true)),
            &weather_call("Paris"),
            true,
        ),
        case(
            "tools none",
            options(false, &[], tools(ToolChoice::None, true)),
            &format!("text {CALL_F}"),
            true,
        ),
        case(
            "reasoning never closed",
            options(true, &[], None),
            "<think>\nUnfinished  \n</thi",
            true,
        ),
        case(
            "multi-byte text",
            options(true, &["日本語", "STOP 😀"], tools(ToolChoice::Auto, true)),
            "思考 é😀 \n</think>\n\nこんにちは 日本 ok 日本語 tail",
            true,
        ),
        case(
            "multi-byte text and a call",
            options(false, &["😀😀"], tools(ToolChoice::Auto, true)),
            &format!("é😀 {}\u{3000}", weather_call("München 北京 😀")),
            true,
        ),
    ]
}

/// A hard cut of one text into chunks at every character boundary of the text, as two chunks.
fn two_way_splits(input: &str) -> Vec<(&str, &str)> {
    input
        .char_indices()
        .map(|(index, _)| index)
        .chain(once(input.len()))
        .map(|index| input.split_at(index))
        .collect()
}

#[test]
fn the_pieces_do_not_depend_on_where_the_text_is_cut() {
    for case in cases() {
        let whole = run(&case.options, &[&case.input], case.stop_token);
        for (head, tail) in two_way_splits(&case.input) {
            let split = run(&case.options, &[head, tail], case.stop_token);
            assert_eq!(split, whole, "{}: cut at {}", case.name, head.len());
        }
        let characters: Vec<String> = case.input.chars().map(String::from).collect();
        let chunks: Vec<&str> = characters.iter().map(String::as_str).collect();
        assert_eq!(
            run(&case.options, &chunks, case.stop_token),
            whole,
            "{}: one character at a time",
            case.name
        );
    }
}

#[test]
fn the_pieces_do_not_depend_on_where_the_text_is_cut_in_three() {
    for case in cases() {
        let whole = run(&case.options, &[&case.input], case.stop_token);
        let boundaries: Vec<usize> = case
            .input
            .char_indices()
            .map(|(index, _)| index)
            .chain([case.input.len()])
            .collect();
        for &first in boundaries.iter().step_by(3) {
            for &second in boundaries.iter().filter(|&&index| index >= first).step_by(5) {
                let chunks = [&case.input[..first], &case.input[first..second], &case.input[second..]];
                assert_eq!(
                    run(&case.options, &chunks, case.stop_token),
                    whole,
                    "{}: cuts {first}, {second}",
                    case.name
                );
            }
        }
    }
}

fn run_case(name: &str) -> Outcome {
    let case = cases().into_iter().find(|case| case.name == name).unwrap();
    run(&case.options, &[&case.input], case.stop_token)
}

#[test]
fn reasoning_and_answer_are_split_and_trimmed() {
    let outcome = run_case("reasoning split");
    assert_eq!(
        outcome.pieces,
        [
            reasoning("First <b>step</b>, then </thin k."),
            text("Answer </think> stays.")
        ]
    );
    assert!(!outcome.reasoning_open);
}

#[test]
fn a_stop_sequence_ends_the_text_and_is_reported() {
    let outcome = run_case("stop in the middle");
    assert_eq!(outcome.pieces, [text("hello ")]);
    assert_eq!(outcome.matched_stop.as_deref(), Some("END"));
    assert!(outcome.ended);
    assert_eq!(run_case("earliest of two stops").matched_stop.as_deref(), Some("\n\n"));
}

#[test]
fn held_text_is_released_at_the_end() {
    assert_eq!(run_case("held stop prefix at the end").pieces, [text("xST")]);
    assert_eq!(run_case("plain text with tools").pieces, [text("It is cold <tool")]);
}

#[test]
fn stops_apply_to_the_answer_and_not_to_the_reasoning() {
    let outcome = run_case("stop applies to the answer only");
    assert_eq!(outcome.pieces, [reasoning("I will say STOP later."), text("Go ")]);
    assert_eq!(outcome.matched_stop.as_deref(), Some("STOP"));
}

#[test]
fn calls_arrive_in_order_with_their_content() {
    let outcome = run_case("several calls");
    assert_eq!(
        outcome.pieces,
        [
            text("Let me check."),
            call("get_weather", "{\"city\":\"Line one\\nline two\"}"),
            call("get_weather", "{\"city\":\"Paris\"}"),
            call("f", "{}"),
        ]
    );
    assert_eq!(outcome.tool_calls, 3);
    assert_eq!(
        run_case("content then call").pieces,
        [text("It is cold"), call("f", "{}")]
    );
}

#[test]
fn reasoning_then_a_call() {
    let outcome = run_case("reasoning then a call");
    assert_eq!(
        outcome.pieces,
        [
            reasoning("Need the weather."),
            call("get_weather", "{\"city\":\"Paris\"}")
        ]
    );
}

#[test]
fn without_parallel_calls_the_first_call_is_the_last_piece() {
    let outcome = run_case("no parallel calls");
    assert_eq!(outcome.pieces, [call("f", "{}")]);
    assert!(outcome.ended);
}

#[test]
fn a_stop_sequence_before_a_call_prevents_the_call() {
    let outcome = run_case("stop before a call");
    assert_eq!(outcome.pieces, [text("Done. ")]);
    assert_eq!(outcome.tool_calls, 0);
    let outcome = run_case("stop matching the call tag");
    assert_eq!(outcome.pieces, [text("Before ")]);
    assert_eq!(outcome.matched_stop.as_deref(), Some("<tool_call>"));
}

#[test]
fn a_held_tag_prefix_released_at_the_end_still_honors_stop_sequences() {
    let outcome = run_case("held tag prefix and stop at the end");
    assert_eq!(outcome.pieces, [text("Answer ")]);
    assert_eq!(outcome.matched_stop.as_deref(), Some("<tool_"));
}

#[test]
fn delimiters_in_a_value_are_kept() {
    let outcome = run_case("delimiters in a value");
    let [Piece::ToolCall(found)] = outcome.pieces.as_slice() else {
        panic!("{:?}", outcome.pieces)
    };
    let arguments = JsonValue::parse(&found.arguments).unwrap();
    assert_eq!(
        arguments.get("city"),
        Some(&JsonValue::String(
            "before <parameter=x>\n2 </paramete\n</tool_call> after".to_string()
        ))
    );
}

#[test]
fn malformed_output_fails_with_the_reason() {
    let expected = [
        (
            "unknown function",
            "the model called a function that was not offered: g",
        ),
        ("text after a call", "the model produced text after a tool call"),
        ("unfinished call", "generation ended inside a tool call"),
        (
            "required without a call",
            "the model did not make the required function call",
        ),
    ];
    for (name, message) in expected {
        assert_eq!(run_case(name).error.as_deref(), Some(message), "{name}");
    }
}

#[test]
fn truncated_output_drops_an_unfinished_call_and_a_missing_required_call_is_no_error() {
    let truncated = run_case("truncated call");
    assert_eq!(truncated.error, None);
    assert_eq!(truncated.pieces, [call("f", "{}")]);
    let required = run_case("required, truncated");
    assert_eq!(required.error, None);
    assert_eq!(required.pieces, [text("No call.")]);
}

#[test]
fn pieces_settled_before_a_failure_come_with_it_and_later_text_is_ignored() {
    let mut pipeline = TextPipeline::new(&options(false, &[], tools(ToolChoice::Auto, true)));
    let error = pipeline
        .push("Hi <tool_call>\n<function=g>\n</function>\n</tool_call>")
        .unwrap_err();
    let PipelineError::MalformedToolCall { emitted, .. } = &error;
    assert_eq!(emitted, &[text("Hi")]);
    assert_eq!(error.message(), "the model called a function that was not offered: g");
    assert_eq!(pipeline.push("more").unwrap(), []);
    assert_eq!(pipeline.finish().unwrap_err().message(), error.message());
    assert!(pipeline.ended());
}

#[test]
fn only_a_named_function_may_be_called_and_none_offers_nothing() {
    let named = run_case("named function");
    assert_eq!(
        named.error.as_deref(),
        Some("the model called a function that was not offered: get_weather")
    );
    assert_eq!(run_case("tools none").pieces, [text(&format!("text {CALL_F}"))]);
}

#[test]
fn multi_byte_text_comes_out_whole() {
    let outcome = run_case("multi-byte text");
    assert_eq!(outcome.pieces, [reasoning("思考 é😀"), text("こんにちは 日本 ok ")]);
    assert_eq!(outcome.matched_stop.as_deref(), Some("日本語"));
    let outcome = run_case("multi-byte text and a call");
    assert_eq!(
        outcome.pieces,
        [text("é😀"), call("get_weather", "{\"city\":\"München 北京 😀\"}")]
    );
}

#[test]
fn unfinished_reasoning_is_released_trimmed_and_reported_open() {
    let outcome = run_case("reasoning never closed");
    assert_eq!(outcome.pieces, [reasoning("<think>\nUnfinished  \n</thi")]);
    assert!(outcome.reasoning_open);
}

/// Runs `work` on its own thread and fails the test when it has not returned in time, so a pipeline that loops
/// forever fails fast instead of hanging the suite.
#[allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test fixture: a watchdog thread and a channel make a hang fail fast"
)]
fn within_deadline<T: Send + 'static>(work: impl FnOnce() -> T + Send + 'static) -> T {
    let (sender, receiver) = channel();
    spawn(move || {
        let _ = sender.send(work());
    });
    receiver
        .recv_timeout(Duration::from_secs(20))
        .expect("the pipeline did not return")
}

#[test]
fn an_unclosed_reasoning_marker_cannot_hang_finish() {
    let inputs = [
        "<think>\nno end in sight",
        "<think>\nstill thinking </think",
        "<think>\n",
        "<think>",
        "</thi",
        "",
        "<think>\n<think>\n</think",
    ];
    for input in inputs {
        let (outcome, closed) = within_deadline(move || {
            let mut pipeline = TextPipeline::new(&options(true, &["stop"], tools(ToolChoice::Required, true)));
            let mut pieces = Vec::new();
            for chunk in input.split_inclusive(char::is_whitespace) {
                pieces.extend(pipeline.push(chunk).unwrap());
            }
            let finished = pipeline.finish();
            (finished.map(|rest| (pieces, rest)), pipeline.reasoning_closed())
        });
        // The required call never came, so finishing at the stop token reports it; the reasoning is still open.
        let error = outcome.unwrap_err();
        assert_eq!(
            error.message(),
            "the model did not make the required function call",
            "{input:?}"
        );
        assert!(!closed, "{input:?}");
    }
    let (pieces, open) = within_deadline(|| {
        let mut pipeline = TextPipeline::new(&options(true, &[], None));
        let mut pieces = pipeline.push("<think>\nno end in sight  \n").unwrap();
        pieces.extend(pipeline.finish().unwrap());
        (pieces, pipeline.reasoning_open())
    });
    assert_eq!(pieces, [reasoning("<think>\nno end in sight")]);
    assert!(open);
}
