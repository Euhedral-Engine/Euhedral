//! Splits decoded model output into assistant text and the checkpoint's `<tool_call>` blocks.

use std::collections::BTreeSet;

use crate::ToolDefinition;
#[cfg(test)]
use crate::json::JsonInteger;
use crate::json::{JsonValue, cut_pair_marker, put};
use crate::pipeline::text_space::{held_tag_prefix, is_block_space, trailing_space_start};

const CALL_START: &str = "<tool_call>";
const CALL_END: &str = "</tool_call>";
const FUNCTION_START: &str = "<function=";
const FUNCTION_END: &str = "</function>";
const PARAMETER_START: &str = "<parameter=";
const PARAMETER_END: &str = "</parameter>";

/// A complete call: the function's name and its arguments as a compact JSON object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: String,
}

/// Parsed output in generation order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolEvent {
    Content(String),
    Call(ToolCall),
}

/// The model's output claims to be a tool call but is not one this request can accept; the message says why.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MalformedToolCall(pub String);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Content,
    Call,
    BetweenCalls,
    /// The single call a request without parallel calls allows is complete; everything after it is discarded.
    Done,
}

/// The format is the inverse of the template's assistant rendering:
/// `<tool_call>\n<function=NAME>\n<parameter=KEY>\nVALUE\n</parameter>\n...</function>\n</tool_call>`.
/// A block is parsed only once its closing tag has arrived, so every emitted call is complete and its arguments are
/// a valid JSON object. Text that could still begin a `<tool_call>` tag, and whitespace that could still precede
/// one, is held back until the next chunk decides it.
#[derive(Debug)]
pub struct ToolCallParser {
    callable: Vec<ToolDefinition>,
    parallel: bool,
    /// A required or named choice: the grammar forces a call, so an answer without one is malformed.
    requires_call: bool,
    buffer: String,
    state: State,
    calls: usize,
}

impl ToolCallParser {
    pub fn new(callable: Vec<ToolDefinition>, parallel: bool, requires_call: bool) -> Self {
        Self {
            callable,
            parallel,
            requires_call,
            buffer: String::new(),
            state: State::Content,
            calls: 0,
        }
    }

    /// The number of complete calls so far.
    pub fn calls(&self) -> usize {
        self.calls
    }

    /// Takes the next chunk and appends what it settles to `out`; events before a failure stay in `out`.
    pub fn accept(&mut self, chunk: &str, out: &mut Vec<ToolEvent>) -> Result<(), MalformedToolCall> {
        self.buffer.push_str(chunk);
        self.drain(out)
    }

    /// Releases held-back text once generation has ended. After a stop token an unfinished call is malformed; after
    /// an exhausted budget it is truncated output and is dropped.
    pub fn finish(&mut self, stop_token_reached: bool, out: &mut Vec<ToolEvent>) -> Result<(), MalformedToolCall> {
        self.drain(out)?;
        if self.state == State::Content {
            if self.requires_call && self.calls == 0 && stop_token_reached {
                return Err(malformed("the model did not make the required function call"));
            }
            if !self.buffer.is_empty() {
                out.push(ToolEvent::Content(self.buffer.clone()));
            }
        } else if stop_token_reached && (self.state == State::Call || !self.buffer.is_empty()) {
            return Err(malformed("generation ended inside a tool call"));
        }
        self.buffer.clear();
        Ok(())
    }

    fn drain(&mut self, out: &mut Vec<ToolEvent>) -> Result<(), MalformedToolCall> {
        loop {
            match self.state {
                State::Content => {
                    let Some(start) = self.buffer.find(CALL_START) else {
                        let limit = self.buffer.len() - held_tag_prefix(&self.buffer, CALL_START);
                        let safe = trailing_space_start(&self.buffer, limit);
                        if safe > 0 {
                            out.push(ToolEvent::Content(self.buffer[..safe].to_string()));
                        }
                        self.buffer.drain(..safe);
                        return Ok(());
                    };
                    // The template renders content|trim before calls, so trailing space before one is dropped.
                    let end = trailing_space_start(&self.buffer, start);
                    if end > 0 {
                        out.push(ToolEvent::Content(self.buffer[..end].to_string()));
                    }
                    self.buffer.drain(..start + CALL_START.len());
                    self.state = State::Call;
                }
                State::Call => {
                    let Some(end) = self.call_end() else {
                        return Ok(());
                    };
                    let block = self.buffer[..end].to_string();
                    self.buffer.drain(..end + CALL_END.len());
                    self.parse_block(&block, out)?;
                    self.state = if self.parallel {
                        State::BetweenCalls
                    } else {
                        State::Done
                    };
                }
                State::BetweenCalls => {
                    let text = self.buffer.len() - self.buffer.trim_start_matches(is_block_space).len();
                    self.buffer.drain(..text);
                    if self.buffer.is_empty() {
                        return Ok(());
                    }
                    if self.buffer.starts_with(CALL_START) {
                        self.buffer.drain(..CALL_START.len());
                        self.state = State::Call;
                    } else if CALL_START.starts_with(self.buffer.as_str()) {
                        return Ok(());
                    } else {
                        return Err(malformed("the model produced text after a tool call"));
                    }
                }
                State::Done => {
                    self.buffer.clear();
                    return Ok(());
                }
            }
        }
    }

    /// The template closes the function before the call. A `</tool_call>` inside a parameter value is ignored.
    fn call_end(&self) -> Option<usize> {
        let mut from = 0;
        while let Some(found) = self.buffer[from..].find(CALL_END) {
            let end = from + found;
            // Without a function opening before it, this cannot be parameter text; reject the closed block now.
            if !self.buffer[..end].contains(FUNCTION_START) {
                return Some(end);
            }
            let before = self.buffer[..end].trim_end_matches(is_block_space);
            if before.ends_with(FUNCTION_END) {
                return Some(end);
            }
            from = end + CALL_END.len();
        }
        None
    }

    fn parse_block(&mut self, block: &str, out: &mut Vec<ToolEvent>) -> Result<(), MalformedToolCall> {
        let mut position = skip_space(block, 0);
        if !block[position..].starts_with(FUNCTION_START) {
            return Err(malformed("a tool call has no <function=...> tag"));
        }
        let Some(name_end) = block[position..].find('>').map(|offset| position + offset) else {
            return Err(malformed("a tool call has an unterminated <function=...> tag"));
        };
        let name = &block[position + FUNCTION_START.len()..name_end];
        let Some(tool) = self.callable.iter().find(|tool| tool.name == name) else {
            return Err(malformed(&format!(
                "the model called a function that was not offered: {name}"
            )));
        };
        position = name_end + 1;
        let mut arguments: Vec<(String, JsonValue)> = Vec::new();
        loop {
            position = skip_space(block, position);
            if block[position..].starts_with(FUNCTION_END) {
                if skip_space(block, position + FUNCTION_END.len()) != block.len() {
                    return Err(malformed("a tool call has text after </function>"));
                }
                break;
            }
            if !block[position..].starts_with(PARAMETER_START) {
                return Err(malformed(&format!(
                    "a tool call for {name} has text outside its parameters"
                )));
            }
            let Some(key_end) = block[position..].find('>').map(|offset| position + offset) else {
                return Err(malformed(&format!(
                    "a tool call for {name} has an unterminated <parameter=...> tag"
                )));
            };
            let key = &block[position + PARAMETER_START.len()..key_end];
            let Some(value_end) = parameter_end(block, key_end + 1) else {
                return Err(malformed(&format!("parameter {key} of {name} is not closed")));
            };
            if arguments.iter().any(|(existing, _)| existing == key) {
                return Err(malformed(&format!("parameter {key} of {name} was repeated")));
            }
            if !tool.accepts_parameter(key) {
                return Err(malformed(&format!("{name} has no parameter {key}")));
            }
            let raw = strip_one_newline(&block[key_end + 1..value_end]);
            put(&mut arguments, key.to_string(), argument_value(tool, key, raw)?);
            position = value_end + PARAMETER_END.len();
        }
        for required in tool.required_parameters() {
            if !arguments.iter().any(|(existing, _)| *existing == required) {
                return Err(malformed(&format!(
                    "required parameter {required} of {name} is missing"
                )));
            }
        }
        self.calls += 1;
        out.push(ToolEvent::Call(ToolCall {
            name: name.to_string(),
            arguments: JsonValue::Object(arguments).to_compact_string(),
        }));
        Ok(())
    }
}

/// The first `</parameter>` followed by another parameter or `</function>`; values may contain the tag.
fn parameter_end(block: &str, from: usize) -> Option<usize> {
    let mut search = from;
    while let Some(found) = block[search..].find(PARAMETER_END) {
        let end = search + found;
        let next = skip_space(block, end + PARAMETER_END.len());
        if block[next..].starts_with(PARAMETER_START) || block[next..].starts_with(FUNCTION_END) {
            return Some(end);
        }
        search = end + 1;
    }
    None
}

/// Inverts the template: strings render verbatim, every other value as JSON. The schema's declared types decide
/// between the two readings; an untyped parameter takes the JSON reading when it parses.
fn argument_value(tool: &ToolDefinition, key: &str, raw: &str) -> Result<JsonValue, MalformedToolCall> {
    let types = tool.parameter_types(key);
    let text = || JsonValue::String(raw.to_string());
    if types.len() == 1 && types.contains("string") {
        return Ok(text());
    }
    let Some(parsed) = JsonValue::parse_keeping_unpaired(raw) else {
        if types.is_empty() || types.contains("string") {
            return Ok(text());
        }
        // Qwen writes Python literals at times; the reference parsers read them for these types.
        let literal = raw.trim_matches(is_block_space);
        if types.contains("boolean") && literal.eq_ignore_ascii_case("true") {
            return Ok(JsonValue::Bool(true));
        }
        if types.contains("boolean") && literal.eq_ignore_ascii_case("false") {
            return Ok(JsonValue::Bool(false));
        }
        if types.contains("null") && literal == "None" {
            return Ok(JsonValue::Null);
        }
        return Err(malformed(&format!(
            "parameter {key} of {} is not valid JSON of type {}: {}",
            tool.name,
            type_list(&types),
            excerpt(raw)
        )));
    };
    // Out-of-range literals parse as infinities, which no JSON arguments string can carry.
    if !parsed.is_finite() {
        return Err(malformed(&format!(
            "parameter {key} of {} has a number outside the double range",
            tool.name
        )));
    }
    let kind = parsed.type_name();
    // A JSON string literal is never how the template renders a string, so keep the model's exact text.
    if kind == "string" && (types.is_empty() || types.contains("string")) {
        return Ok(text());
    }
    if types.is_empty() || types.contains(kind) || (kind == "integer" && types.contains("number")) {
        return Ok(parsed);
    }
    if types.contains("string") {
        return Ok(text());
    }
    Err(malformed(&format!(
        "parameter {key} of {} is not of type {}: {}",
        tool.name,
        type_list(&types),
        excerpt(raw)
    )))
}

/// The declared types as a bracketed list, in alphabetical order.
fn type_list(types: &BTreeSet<String>) -> String {
    let names: Vec<&str> = types.iter().map(String::as_str).collect();
    format!("[{}]", names.join(", "))
}

/// The start of a value, for an error message: at most 80 UTF-16 code units, then `...`. A character cut in half by
/// the limit leaves its high surrogate alone at the end; a message cannot hold one, so a stand-in
/// ([`cut_pair_marker`]) takes its place and the reply writes it back as the unit.
fn excerpt(raw: &str) -> String {
    const LIMIT: usize = 80;
    if raw.encode_utf16().count() <= LIMIT {
        return format!("'{raw}'");
    }
    let mut cut = String::new();
    let mut units = 0;
    for value in raw.chars() {
        let width = value.len_utf16();
        if units + width > LIMIT {
            if units < LIMIT {
                let mut pair = [0; 2];
                value.encode_utf16(&mut pair);
                cut.push(cut_pair_marker(pair[0]));
            }
            break;
        }
        cut.push(value);
        units += width;
    }
    format!("'{cut}...'")
}

fn strip_one_newline(value: &str) -> &str {
    let value = value.strip_prefix('\n').unwrap_or(value);
    value.strip_suffix('\n').unwrap_or(value)
}

fn skip_space(text: &str, position: usize) -> usize {
    text.len() - text[position..].trim_start_matches(is_block_space).len()
}

fn malformed(message: &str) -> MalformedToolCall {
    MalformedToolCall(message.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::{ApiError, Surface};
    use crate::openai::ToolCall as ReplyToolCall;
    use crate::wire::ToJson;

    fn tool(parameters: Option<&str>) -> ToolDefinition {
        tool_definition("f", parameters)
    }

    fn tool_definition(name: &str, parameters: Option<&str>) -> ToolDefinition {
        ToolDefinition {
            name: name.to_string(),
            description: None,
            parameters: parameters.map(|text| JsonValue::parse(text).unwrap()),
        }
    }

    #[test]
    fn an_excerpt_counts_utf16_units_and_cuts_after_eighty() {
        assert_eq!(excerpt(&"a".repeat(80)), format!("'{}'", "a".repeat(80)));
        assert_eq!(excerpt(&"a".repeat(81)), format!("'{}...'", "a".repeat(80)));
        // 40 emoji are 80 units and fit; 50 emoji are 100 units and are cut after the 40th.
        assert_eq!(excerpt(&"😀".repeat(40)), format!("'{}'", "😀".repeat(40)));
        assert_eq!(excerpt(&"😀".repeat(50)), format!("'{}...'", "😀".repeat(40)));
        // A pair cut in half by the limit leaves its high surrogate alone, which the stand-in marks.
        assert_eq!(
            excerpt(&format!("a{}", "😀".repeat(50))),
            format!("'a{}{}...'", "😀".repeat(39), cut_pair_marker(0xd83d))
        );
        assert_eq!(excerpt("é"), "'é'");
    }

    /// A parameter that is not of its declared type, with an emoji straddling the 80th unit of its value.
    fn cut_pair_failure() -> MalformedToolCall {
        let schema = r#"{"type":"object","properties":{"n":{"type":"integer"}}}"#;
        let value = format!("{}{}", "a".repeat(79), "😀".repeat(5));
        arguments(schema, &["n", &value]).unwrap_err()
    }

    #[test]
    fn the_excerpt_of_a_cut_pair_is_written_as_the_unpaired_surrogate() {
        let error = ApiError::invalid_tool_call(&cut_pair_failure().0);
        let body = String::from_utf8(error.body(Surface::OpenAi)).unwrap();
        assert!(
            body.contains(r#"'{}\uD83D...'"#.replace("{}", &"a".repeat(79)).as_str()),
            "{body}"
        );
        let body = String::from_utf8(error.body(Surface::Anthropic)).unwrap();
        assert!(body.contains("\\uD83D...'"), "{body}");
        // An excerpt that ends on a whole character has no escape.
        let schema = r#"{"type":"object","properties":{"n":{"type":"integer"}}}"#;
        let whole = arguments(schema, &["n", &"😀".repeat(50)]).unwrap_err();
        let body = String::from_utf8(ApiError::invalid_tool_call(&whole.0).body(Surface::OpenAi)).unwrap();
        assert!(!body.contains("\\u"), "{body}");
    }

    #[test]
    fn an_unpaired_surrogate_in_an_argument_is_written_as_its_escape() {
        let schema = r#"{"type":"object","properties":{"v":{}}}"#;
        let arguments = arguments(schema, &["v", r#"["\ud800x", "\ud83d\ude00", "\udc00", "\\ud800"]"#]).unwrap();
        assert_eq!(arguments, r#"{"v":["\uD800x","😀","\uDC00","\\ud800"]}"#, "{arguments}");
        // Written again as the string value it is in a reply, the escape stays an escape of the unit and the
        // text of an escape stays text.
        let call = ReplyToolCall::function("c", "f", arguments);
        assert_eq!(
            call.to_json_string(),
            r#"{"id":"c","type":"function","function":{"name":"f","arguments":"{\"v\":[\"\uD800x\",\"😀\",\"\uDC00\",\"\\\\ud800\"]}"}}"#
        );
    }

    fn parser(parameters: Option<&str>, parallel: bool, requires_call: bool) -> ToolCallParser {
        ToolCallParser::new(vec![tool(parameters)], parallel, requires_call)
    }

    fn contents(events: &[ToolEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| {
                if let ToolEvent::Content(text) = event {
                    Some(text.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    fn arguments_of(events: &[ToolEvent]) -> Vec<&str> {
        events
            .iter()
            .filter_map(|event| {
                if let ToolEvent::Call(call) = event {
                    Some(call.arguments.as_str())
                } else {
                    None
                }
            })
            .collect()
    }

    /// One call to `f` with the given parameters, parsed under `schema`; returns its arguments.
    fn arguments(schema: &str, key_values: &[&str]) -> Result<String, MalformedToolCall> {
        let mut block = String::from("<tool_call>\n<function=f>\n");
        for pair in key_values.chunks(2) {
            block.push_str(&format!("<parameter={}>\n{}\n</parameter>\n", pair[0], pair[1]));
        }
        block.push_str("</function>\n</tool_call>");
        let mut parser = parser(Some(schema), true, false);
        let mut events = Vec::new();
        parser.accept(&block, &mut events)?;
        parser.finish(true, &mut events)?;
        assert!(contents(&events).is_empty());
        assert_eq!(events.len(), 1);
        Ok(arguments_of(&events)[0].to_string())
    }

    /// Each parameter value is read back the way the template wrote it: strings verbatim, other JSON as JSON.
    #[test]
    fn parameter_values_follow_the_declared_schema_types() {
        let schema = r#"{"type":"object","properties":{
              "s":{"type":"string"},
              "i":{"type":"integer"},
              "n":{"type":"number"},
              "b":{"type":"boolean"},
              "a":{"type":"array"},
              "o":{"type":"object"},
              "sn":{"type":["string","null"]},
              "e":{"enum":["1","2"]},
              "any":{"anyOf":[{"type":"integer"},{"type":"string"}]},
              "free":{}
            }}"#;
        assert_eq!(
            arguments(
                schema,
                &[
                    "s",
                    "42",
                    "i",
                    "42",
                    "n",
                    "3",
                    "b",
                    "false",
                    "a",
                    "[1, \"x\"]",
                    "o",
                    "{\"k\": null}",
                    "sn",
                    "null",
                    "e",
                    "1",
                    "any",
                    "forty",
                    "free",
                    "plain words"
                ]
            )
            .unwrap(),
            "{\"s\":\"42\",\"i\":42,\"n\":3,\"b\":false,\"a\":[1,\"x\"],\"o\":{\"k\":null},\"sn\":null,\"e\":\"1\",\"any\":\"forty\",\"free\":\"plain words\"}"
        );
        // A quoted value keeps its quotes: the template never JSON-encodes strings.
        assert_eq!(
            arguments(schema, &["sn", "\"quoted\"", "free", "\"q\""]).unwrap(),
            "{\"sn\":\"\\\"quoted\\\"\",\"free\":\"\\\"q\\\"\"}"
        );
        // Untyped values take the JSON reading when it parses.
        assert_eq!(arguments(schema, &["free", "[1, 2]"]).unwrap(), "{\"free\":[1,2]}");
        // Python literals for booleans read as booleans; anything else of the wrong type names the value.
        assert_eq!(arguments(schema, &["b", "True"]).unwrap(), "{\"b\":true}");
        assert_eq!(arguments(schema, &["b", " FALSE "]).unwrap(), "{\"b\":false}");
        let refused = arguments(schema, &["b", "yes"]).unwrap_err();
        assert_eq!(refused.0, "parameter b of f is not valid JSON of type [boolean]: 'yes'");
    }

    #[test]
    fn literal_closing_call_tag_inside_a_string_argument_does_not_end_the_call() {
        assert_eq!(
            arguments(
                r#"{"type":"object","properties":{"s":{"type":"string"}}}"#,
                &["s", "before </tool_call> after"]
            )
            .unwrap(),
            "{\"s\":\"before </tool_call> after\"}"
        );
    }

    #[test]
    fn values_that_are_not_the_declared_type_are_malformed() {
        let schema = r#"{"type":"object","properties":{"i":{"type":"integer"},"b":{"type":"boolean"},
            "o":{"type":"object"},"n":{"type":"number"},"free":{}}}"#;
        let invalid: [[&str; 2]; 9] = [
            ["i", "3 4"],
            ["n", "1e400"],
            ["free", "[1e400]"],
            ["i", "3.5"],
            ["i", "\"3\""],
            ["i", ""],
            ["b", "yes"],
            ["o", "[1]"],
            ["o", "{\"k\": 1} trailing"],
        ];
        for pair in invalid {
            assert!(arguments(schema, &pair).is_err(), "{pair:?}");
        }
    }

    #[test]
    fn a_required_choice_without_a_call_is_malformed_once_the_model_stops() {
        let mut parser = parser(None, true, true);
        let mut events = Vec::new();
        parser.accept("No call.", &mut events).unwrap();
        let error = parser.finish(true, &mut events).unwrap_err();
        assert_eq!(error.0, "the model did not make the required function call");
        let mut truncated = self::parser(None, true, true);
        let mut events = Vec::new();
        truncated.accept("No call.", &mut events).unwrap();
        truncated.finish(false, &mut events).unwrap();
        assert_eq!(contents(&events), ["No call."]);
    }

    #[test]
    fn answer_text_streams_as_it_arrives_and_only_a_tag_prefix_is_held_back() {
        let mut parser = parser(None, true, false);
        let mut events = Vec::new();
        parser.accept("It is", &mut events).unwrap();
        parser.accept(" cold <tool", &mut events).unwrap();
        assert_eq!(contents(&events), ["It is", " cold"]);
        parser
            .accept("_call>\n<function=f>\n</function>\n</tool_call>", &mut events)
            .unwrap();
        parser.finish(true, &mut events).unwrap();
        assert_eq!(contents(&events), ["It is", " cold"]);
        assert_eq!(arguments_of(&events), ["{}"]);
    }

    #[test]
    fn without_parallel_calls_text_after_the_first_call_is_discarded() {
        let mut parser = parser(None, false, false);
        let mut events = Vec::new();
        let call = "<tool_call>\n<function=f>\n</function>\n</tool_call>";
        parser.accept(&format!("{call}\n{call} more"), &mut events).unwrap();
        parser.finish(true, &mut events).unwrap();
        assert_eq!(arguments_of(&events), ["{}"]);
        assert!(contents(&events).is_empty());
    }

    #[test]
    fn malformed_blocks_say_why() {
        let cases = [
            (
                "<tool_call>\nnot a function\n</tool_call>",
                "a tool call has no <function=...> tag",
            ),
            (
                "<tool_call>\n<function=g>\n</function>\n</tool_call>",
                "the model called a function that was not offered: g",
            ),
            (
                "<tool_call>\n<function=f>\n</function> x </function>\n</tool_call>",
                "a tool call has text after </function>",
            ),
            (
                "<tool_call>\n<function=f>\ntext\n</function>\n</tool_call>",
                "a tool call for f has text outside its parameters",
            ),
            (
                "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n<parameter=a>\n2\n</parameter>\n</function>\n</tool_call>",
                "parameter a of f was repeated",
            ),
            (
                "<tool_call>\n<function=f>\n<parameter=z>\n1\n</parameter>\n</function>\n</tool_call>",
                "f has no parameter z",
            ),
            (
                "<tool_call>\n<function=f>\n</function>\n</tool_call>\ntrailing",
                "the model produced text after a tool call",
            ),
        ];
        let schema = r#"{"type":"object","properties":{"a":{}},"additionalProperties":false}"#;
        for (output, message) in cases {
            let mut parser = parser(Some(schema), true, false);
            let mut events = Vec::new();
            let result = parser.accept(output, &mut events);
            assert_eq!(result.unwrap_err().0, message, "{output}");
        }
        let required = r#"{"type":"object","properties":{"a":{}},"required":["a"]}"#;
        let mut parser = parser(Some(required), true, false);
        let error = parser
            .accept("<tool_call>\n<function=f>\n</function>\n</tool_call>", &mut Vec::new())
            .unwrap_err();
        assert_eq!(error.0, "required parameter a of f is missing");
    }

    #[test]
    fn an_unfinished_call_is_malformed_after_a_stop_token_and_dropped_after_an_exhausted_budget() {
        let output = "<tool_call>\n<function=f>\n<parameter=n>\n";
        let mut stopped = parser(None, true, false);
        let mut events = Vec::new();
        stopped.accept(output, &mut events).unwrap();
        assert_eq!(
            stopped.finish(true, &mut events).unwrap_err().0,
            "generation ended inside a tool call"
        );
        let mut cut = parser(None, true, false);
        let mut events = Vec::new();
        cut.accept(output, &mut events).unwrap();
        cut.finish(false, &mut events).unwrap();
        assert!(events.is_empty());
    }

    #[test]
    fn whitespace_before_a_call_is_not_content() {
        let mut parser = parser(None, true, false);
        let mut events = Vec::new();
        parser.accept(" \n<tool", &mut events).unwrap();
        parser
            .accept("_call>\n<function=f>\n</function>\n</tool_call>", &mut events)
            .unwrap();
        parser.finish(true, &mut events).unwrap();
        assert!(contents(&events).is_empty());
        assert_eq!(arguments_of(&events), ["{}"]);
    }

    #[test]
    fn delimiter_looking_text_in_a_value_is_kept_except_the_closing_call_tag() {
        let schema = r#"{"type":"object","properties":{"s":{"type":"string"},"x":{"type":"integer"}},"required":["s"],"additionalProperties":false}"#;
        let dangerous = "before <parameter=x>\n2 </paramete\n</tool_call> after";
        let arguments = arguments(schema, &["s", dangerous, "x", "5"]).unwrap();
        let parsed = JsonValue::parse(&arguments).unwrap();
        assert_eq!(parsed.get("s"), Some(&JsonValue::String(dangerous.to_string())));
        assert_eq!(parsed.get("x"), Some(&JsonValue::Integer(JsonInteger::from(5_i128))));
    }
}
