//! OpenAI Responses: the request, the `response` object and the `response.*` stream events.
//!
//! A request body is read as JSON and then field by field: the fields the server understands are typed (scalars are
//! read as [`crate::coerce`] describes), the rest are kept in `other` in the order they arrived, and fields that take
//! several shapes stay [`JsonValue`]s. A `null` is the same as an absent field. Replies are [`ToJson`]: their members
//! are written in declaration order.

pub use crate::openai::{Model, ModelList};

use crate::coerce;
use crate::error::ApiError;
use crate::json::{JsonValue, put};
use crate::wire::{
    Members, Reader, ToJson, arguments_text, int, list, members, message_text, object, optional_text, push, push_some,
    text,
};

/// A Responses request. Input items and the fields that take several shapes stay untyped JSON so the mapper can
/// report precise errors.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ResponsesRequest {
    pub model: Option<String>,
    pub input: Option<JsonValue>,
    pub instructions: Option<String>,
    pub tools: Option<JsonValue>,
    pub tool_choice: Option<JsonValue>,
    pub parallel_tool_calls: Option<bool>,
    pub max_output_tokens: Option<i32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub stream: Option<bool>,
    pub text: Option<JsonValue>,
    pub reasoning: Option<JsonValue>,
    pub include: Option<JsonValue>,
    /// Every other top-level field.
    pub other: Members,
}

/// A text part of an output message.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputText {
    pub kind: &'static str,
    pub text: String,
    pub annotations: Vec<JsonValue>,
}

impl OutputText {
    pub fn new(text: impl Into<String>) -> OutputText {
        OutputText {
            kind: "output_text",
            text: text.into(),
            annotations: Vec::new(),
        }
    }
}

/// A text part of a reasoning item.
#[derive(Clone, Debug, PartialEq)]
pub struct ReasoningText {
    pub kind: &'static str,
    pub text: String,
}

impl ReasoningText {
    pub fn new(text: impl Into<String>) -> ReasoningText {
        ReasoningText {
            kind: "reasoning_text",
            text: text.into(),
        }
    }
}

/// A content part carried by a part event.
#[derive(Clone, Debug, PartialEq)]
pub enum ContentPart {
    OutputText(OutputText),
    ReasoningText(ReasoningText),
}

/// A reasoning item; `encrypted_content` is present only when the request asked for it.
#[derive(Clone, Debug, PartialEq)]
pub struct ReasoningItem {
    pub id: String,
    pub kind: &'static str,
    pub summary: Vec<JsonValue>,
    pub content: Vec<ReasoningText>,
    pub encrypted_content: Option<String>,
}

impl ReasoningItem {
    /// An item with no text yet.
    pub fn new(id: impl Into<String>) -> ReasoningItem {
        ReasoningItem {
            id: id.into(),
            kind: "reasoning",
            summary: Vec::new(),
            content: Vec::new(),
            encrypted_content: None,
        }
    }
}

/// An assistant message item.
#[derive(Clone, Debug, PartialEq)]
pub struct MessageItem {
    pub id: String,
    pub kind: &'static str,
    pub status: String,
    pub role: &'static str,
    pub content: Vec<OutputText>,
}

impl MessageItem {
    /// An `in_progress` message with no text yet.
    pub fn new(id: impl Into<String>) -> MessageItem {
        MessageItem {
            id: id.into(),
            kind: "message",
            status: "in_progress".to_owned(),
            role: "assistant",
            content: Vec::new(),
        }
    }
}

/// A function call item; `arguments` is a JSON object as text.
#[derive(Clone, Debug, PartialEq)]
pub struct FunctionCallItem {
    pub id: String,
    pub kind: &'static str,
    pub status: String,
    pub call_id: String,
    pub name: String,
    pub arguments: String,
}

impl FunctionCallItem {
    pub fn new(
        id: impl Into<String>,
        status: impl Into<String>,
        call_id: impl Into<String>,
        name: impl Into<String>,
        arguments: impl Into<String>,
    ) -> FunctionCallItem {
        FunctionCallItem {
            id: id.into(),
            kind: "function_call",
            status: status.into(),
            call_id: call_id.into(),
            name: name.into(),
            arguments: arguments.into(),
        }
    }
}

/// An item of a response's output.
#[derive(Clone, Debug, PartialEq)]
pub enum OutputItem {
    Reasoning(ReasoningItem),
    Message(MessageItem),
    FunctionCall(FunctionCallItem),
}

/// Why a response is incomplete.
#[derive(Clone, Debug, PartialEq)]
pub struct IncompleteDetails {
    pub reason: String,
}

/// Token accounting of a response.
#[derive(Clone, Debug, PartialEq)]
pub struct Usage {
    pub input_tokens: i32,
    pub input_tokens_details: InputTokensDetails,
    pub output_tokens: i32,
    pub output_tokens_details: OutputTokensDetails,
    pub total_tokens: i32,
}

/// The prompt tokens restored from the prefix cache.
#[derive(Clone, Debug, PartialEq)]
pub struct InputTokensDetails {
    pub cached_tokens: i32,
}

/// The reasoning tokens among the output tokens.
#[derive(Clone, Debug, PartialEq)]
pub struct OutputTokensDetails {
    pub reasoning_tokens: i32,
}

impl Usage {
    /// The usage of a generation; no reasoning counts as zero reasoning tokens.
    pub fn of(prompt_tokens: i32, cached_tokens: i32, completion_tokens: i32, reasoning_tokens: Option<i32>) -> Usage {
        Usage {
            input_tokens: prompt_tokens,
            input_tokens_details: InputTokensDetails { cached_tokens },
            output_tokens: completion_tokens,
            output_tokens_details: OutputTokensDetails {
                reasoning_tokens: reasoning_tokens.unwrap_or(0),
            },
            total_tokens: prompt_tokens + completion_tokens,
        }
    }
}

/// The `response` object. `error`, `incomplete_details` and `usage` are explicit nulls while absent; `echo` holds the
/// request settings the response repeats, between `output` and `usage`.
#[derive(Clone, Debug, PartialEq)]
pub struct Response {
    pub id: String,
    pub object: &'static str,
    pub created_at: i64,
    pub status: String,
    pub error: Option<JsonValue>,
    pub incomplete_details: Option<IncompleteDetails>,
    pub model: String,
    pub output: Vec<OutputItem>,
    pub echo: Members,
    pub usage: Option<Usage>,
}

impl Response {
    /// A response with the given status and output and no error, incomplete details or usage.
    pub fn new(
        id: impl Into<String>,
        created_at: i64,
        status: impl Into<String>,
        model: impl Into<String>,
        output: Vec<OutputItem>,
        echo: Members,
    ) -> Response {
        Response {
            id: id.into(),
            object: "response",
            created_at,
            status: status.into(),
            error: None,
            incomplete_details: None,
            model: model.into(),
            output,
            echo,
            usage: None,
        }
    }
}

/// The fields of a stream event after its `type` and `sequence_number`.
#[derive(Clone, Debug, PartialEq)]
pub enum EventBody {
    /// `response.created`, `response.in_progress`, `response.completed`, `response.incomplete`.
    Response { response: Response },
    /// `response.output_item.added` and `response.output_item.done`.
    OutputItem { output_index: i32, item: OutputItem },
    /// `response.content_part.added` and `response.content_part.done`.
    ContentPart {
        item_id: String,
        output_index: i32,
        content_index: i32,
        part: ContentPart,
    },
    /// `response.output_text.delta` and `response.reasoning_text.delta`.
    TextDelta {
        item_id: String,
        output_index: i32,
        content_index: i32,
        delta: String,
    },
    /// `response.output_text.done` and `response.reasoning_text.done`.
    TextDone {
        item_id: String,
        output_index: i32,
        content_index: i32,
        text: String,
    },
    /// `response.function_call_arguments.delta`.
    ArgumentsDelta {
        item_id: String,
        output_index: i32,
        delta: String,
    },
    /// `response.function_call_arguments.done`.
    ArgumentsDone {
        item_id: String,
        output_index: i32,
        arguments: String,
    },
    /// `error`, an in-band failure after the headers are committed.
    Error {
        code: Option<String>,
        message: String,
        param: Option<String>,
    },
}

impl EventBody {
    /// The in-band event body of a failure.
    pub fn error(failure: &ApiError) -> EventBody {
        EventBody::Error {
            code: failure.code.map(str::to_owned),
            message: failure.message.clone(),
            param: failure.param.clone(),
        }
    }
}

/// One stream event: its `type` (also the SSE event name), its position in the stream and its body.
#[derive(Clone, Debug, PartialEq)]
pub struct ResponseEvent {
    pub kind: String,
    pub sequence_number: u64,
    pub body: EventBody,
}

impl ResponseEvent {
    pub fn new(kind: impl Into<String>, sequence_number: u64, body: EventBody) -> ResponseEvent {
        ResponseEvent {
            kind: kind.into(),
            sequence_number,
            body,
        }
    }
}

impl ResponsesRequest {
    /// Reads a request body; any body that is not a JSON object of the documented shape is an unparsable body.
    pub fn parse(body: &str) -> Result<ResponsesRequest, ApiError> {
        let mut reader = Reader::parse(body)?;
        let mut request = ResponsesRequest {
            model: reader
                .take("model")
                .map(|value| coerce::string(&value))
                .transpose()?
                .flatten(),
            input: reader.take("input"),
            instructions: reader
                .take("instructions")
                .map(|value| coerce::string(&value))
                .transpose()?
                .flatten(),
            tools: reader.take("tools"),
            tool_choice: reader.take("tool_choice"),
            parallel_tool_calls: reader
                .take("parallel_tool_calls")
                .map(|value| coerce::boolean(&value))
                .transpose()?
                .flatten(),
            max_output_tokens: reader
                .take("max_output_tokens")
                .map(|value| coerce::integer(&value))
                .transpose()?
                .flatten(),
            temperature: reader
                .take("temperature")
                .map(|value| coerce::double(&value))
                .transpose()?
                .flatten(),
            top_p: reader
                .take("top_p")
                .map(|value| coerce::double(&value))
                .transpose()?
                .flatten(),
            stream: reader
                .take("stream")
                .map(|value| coerce::boolean(&value))
                .transpose()?
                .flatten(),
            text: reader.take("text"),
            reasoning: reader.take("reasoning"),
            include: reader.take("include"),
            other: Vec::new(),
        };
        request.other = reader.rest();
        Ok(request)
    }
}

impl ToJson for OutputText {
    fn to_json(&self) -> JsonValue {
        object([
            ("type", text(self.kind)),
            ("text", text(&self.text)),
            ("annotations", JsonValue::Array(self.annotations.clone())),
        ])
    }
}

impl ToJson for ReasoningText {
    fn to_json(&self) -> JsonValue {
        object([("type", text(self.kind)), ("text", text(&self.text))])
    }
}

impl ToJson for ContentPart {
    fn to_json(&self) -> JsonValue {
        match self {
            ContentPart::OutputText(part) => part.to_json(),
            ContentPart::ReasoningText(part) => part.to_json(),
        }
    }
}

impl ToJson for ReasoningItem {
    fn to_json(&self) -> JsonValue {
        let mut fields = Vec::new();
        push(&mut fields, "id", text(&self.id));
        push(&mut fields, "type", text(self.kind));
        push(&mut fields, "summary", JsonValue::Array(self.summary.clone()));
        push(&mut fields, "content", list(&self.content));
        push_some(
            &mut fields,
            "encrypted_content",
            self.encrypted_content.as_deref().map(text),
        );
        members(fields)
    }
}

impl ToJson for MessageItem {
    fn to_json(&self) -> JsonValue {
        object([
            ("id", text(&self.id)),
            ("type", text(self.kind)),
            ("status", text(&self.status)),
            ("role", text(self.role)),
            ("content", list(&self.content)),
        ])
    }
}

impl ToJson for FunctionCallItem {
    fn to_json(&self) -> JsonValue {
        object([
            ("id", text(&self.id)),
            ("type", text(self.kind)),
            ("status", text(&self.status)),
            ("call_id", text(&self.call_id)),
            ("name", text(&self.name)),
            ("arguments", arguments_text(&self.arguments)),
        ])
    }
}

impl ToJson for OutputItem {
    fn to_json(&self) -> JsonValue {
        match self {
            OutputItem::Reasoning(item) => item.to_json(),
            OutputItem::Message(item) => item.to_json(),
            OutputItem::FunctionCall(item) => item.to_json(),
        }
    }
}

impl ToJson for Usage {
    fn to_json(&self) -> JsonValue {
        object([
            ("input_tokens", int(self.input_tokens)),
            (
                "input_tokens_details",
                object([("cached_tokens", int(self.input_tokens_details.cached_tokens))]),
            ),
            ("output_tokens", int(self.output_tokens)),
            (
                "output_tokens_details",
                object([("reasoning_tokens", int(self.output_tokens_details.reasoning_tokens))]),
            ),
            ("total_tokens", int(self.total_tokens)),
        ])
    }
}

impl ToJson for Response {
    /// The members in order; an echo member whose key is already present replaces that member's value in place, and
    /// `usage` comes last unless the echo already holds it.
    fn to_json(&self) -> JsonValue {
        let mut fields: Members = Vec::new();
        push(&mut fields, "id", text(&self.id));
        push(&mut fields, "object", text(self.object));
        push(&mut fields, "created_at", int(self.created_at));
        push(&mut fields, "status", text(&self.status));
        push(&mut fields, "error", self.error.clone().unwrap_or(JsonValue::Null));
        push(
            &mut fields,
            "incomplete_details",
            self.incomplete_details
                .as_ref()
                .map_or(JsonValue::Null, |details| object([("reason", text(&details.reason))])),
        );
        push(&mut fields, "model", text(&self.model));
        push(&mut fields, "output", list(&self.output));
        for (key, value) in &self.echo {
            put(&mut fields, key.clone(), value.clone());
        }
        put(
            &mut fields,
            "usage".to_owned(),
            self.usage.as_ref().map_or(JsonValue::Null, ToJson::to_json),
        );
        members(fields)
    }
}

impl ToJson for ResponseEvent {
    /// `type` and `sequence_number`, then the body's members; a body member of the same name is not written twice.
    fn to_json(&self) -> JsonValue {
        let mut fields: Members = Vec::new();
        push(&mut fields, "type", text(&self.kind));
        push(&mut fields, "sequence_number", int(self.sequence_number));
        let body = match &self.body {
            EventBody::Response { response } => vec![("response", response.to_json())],
            EventBody::OutputItem { output_index, item } => {
                vec![("output_index", int(*output_index)), ("item", item.to_json())]
            }
            EventBody::ContentPart {
                item_id,
                output_index,
                content_index,
                part,
            } => vec![
                ("item_id", text(item_id)),
                ("output_index", int(*output_index)),
                ("content_index", int(*content_index)),
                ("part", part.to_json()),
            ],
            EventBody::TextDelta {
                item_id,
                output_index,
                content_index,
                delta,
            } => vec![
                ("item_id", text(item_id)),
                ("output_index", int(*output_index)),
                ("content_index", int(*content_index)),
                ("delta", text(delta)),
            ],
            EventBody::TextDone {
                item_id,
                output_index,
                content_index,
                text: done,
            } => vec![
                ("item_id", text(item_id)),
                ("output_index", int(*output_index)),
                ("content_index", int(*content_index)),
                ("text", text(done)),
            ],
            EventBody::ArgumentsDelta {
                item_id,
                output_index,
                delta,
            } => {
                vec![
                    ("item_id", text(item_id)),
                    ("output_index", int(*output_index)),
                    ("delta", arguments_text(delta)),
                ]
            }
            EventBody::ArgumentsDone {
                item_id,
                output_index,
                arguments,
            } => vec![
                ("item_id", text(item_id)),
                ("output_index", int(*output_index)),
                ("arguments", arguments_text(arguments)),
            ],
            EventBody::Error { code, message, param } => vec![
                ("code", optional_text(code.as_deref())),
                ("message", message_text(message)),
                ("param", optional_text(param.as_deref())),
            ],
        };
        for (key, value) in body {
            put(&mut fields, key.to_owned(), value);
        }
        members(fields)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(text: &str) -> JsonValue {
        JsonValue::parse(text).expect("valid json")
    }

    fn echo() -> Members {
        match json(r#"{"instructions":null,"tool_choice":"auto","tools":[],"temperature":1.0}"#) {
            JsonValue::Object(members) => members,
            _ => unreachable!(),
        }
    }

    #[test]
    fn a_request_with_every_known_field_is_read() {
        let request = ResponsesRequest::parse(
            r#"{"model":"m","input":[{"type":"message","role":"user","content":[{"type":"input_text","text":"hi"}]}],
            "instructions":"be brief","tools":[{"type":"function","name":"f","parameters":{"b":1,"a":2}}],"tool_choice":"auto",
            "parallel_tool_calls":true,"max_output_tokens":9,"temperature":0.1,"top_p":0.7,"stream":false,
            "text":{"format":{"type":"text"}},"reasoning":{"effort":"low"},"include":["reasoning.encrypted_content"],
            "metadata":{"k":"v"},"store":false}"#,
        )
        .expect("reads");
        assert_eq!(request.instructions.as_deref(), Some("be brief"));
        assert_eq!(request.max_output_tokens, Some(9));
        assert_eq!(
            request.include.expect("include").to_compact_string(),
            r#"["reasoning.encrypted_content"]"#
        );
        assert_eq!(
            request.tools.expect("tools").to_compact_string(),
            r#"[{"type":"function","name":"f","parameters":{"b":1,"a":2}}]"#
        );
        let names: Vec<&str> = request.other.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["metadata", "store"]);
    }

    #[test]
    fn the_input_may_be_a_string_and_scalars_are_coerced() {
        let request = ResponsesRequest::parse(r#"{"input":"hi","max_output_tokens":"12","instructions":5,"stream":1}"#)
            .expect("reads");
        assert_eq!(request.input, Some(json("\"hi\"")));
        assert_eq!(request.max_output_tokens, Some(12));
        assert_eq!(request.instructions.as_deref(), Some("5"));
        assert_eq!(request.stream, Some(true));
        for body in [r#"{"instructions":[]}"#, r#"{"max_output_tokens":1e10}"#, "[]", "{"] {
            assert_eq!(
                ResponsesRequest::parse(body),
                Err(ApiError::unparsable_body()),
                "{body}"
            );
        }
    }

    #[test]
    fn a_completed_response_has_its_items_echo_and_usage_in_order() {
        let mut reasoning = ReasoningItem::new("rs_1");
        reasoning.content.push(ReasoningText::new("think"));
        reasoning.encrypted_content = Some("enc".into());
        let mut message = MessageItem::new("msg_1");
        message.status = "completed".into();
        message.content.push(OutputText::new("hi"));
        let call = FunctionCallItem::new("fc_1", "completed", "call_1", "f", r#"{"a":1}"#);
        let mut response = Response::new(
            "resp_1",
            7,
            "incomplete",
            "m",
            vec![
                OutputItem::Reasoning(reasoning),
                OutputItem::Message(message),
                OutputItem::FunctionCall(call),
            ],
            echo(),
        );
        response.incomplete_details = Some(IncompleteDetails {
            reason: "max_output_tokens".into(),
        });
        response.usage = Some(Usage::of(10, 4, 3, Some(2)));
        assert_eq!(
            response.to_json_string(),
            r#"{"id":"resp_1","object":"response","created_at":7,"status":"incomplete","error":null,"incomplete_details":{"reason":"max_output_tokens"},"model":"m","output":[{"id":"rs_1","type":"reasoning","summary":[],"content":[{"type":"reasoning_text","text":"think"}],"encrypted_content":"enc"},{"id":"msg_1","type":"message","status":"completed","role":"assistant","content":[{"type":"output_text","text":"hi","annotations":[]}]},{"id":"fc_1","type":"function_call","status":"completed","call_id":"call_1","name":"f","arguments":"{\"a\":1}"}],"instructions":null,"tool_choice":"auto","tools":[],"temperature":1.0,"usage":{"input_tokens":10,"input_tokens_details":{"cached_tokens":4},"output_tokens":3,"output_tokens_details":{"reasoning_tokens":2},"total_tokens":13}}"#
        );
    }

    #[test]
    fn echoed_numbers_are_written_as_doubles_and_echo_keys_replace_in_place() {
        let echo = match json(
            r#"{"temperature":1e-4,"top_p":25000000.0,"seed":123456789012345678901234567890,"model":"other","metadata":{"k":"\u001f"}}"#,
        ) {
            JsonValue::Object(members) => members,
            _ => unreachable!(),
        };
        let response = Response::new("resp_1", 7, "in_progress", "m", Vec::new(), echo);
        assert_eq!(
            response.to_json_string(),
            r#"{"id":"resp_1","object":"response","created_at":7,"status":"in_progress","error":null,"incomplete_details":null,"model":"other","output":[],"temperature":1.0E-4,"top_p":2.5E7,"seed":123456789012345678901234567890,"metadata":{"k":"\u001F"},"usage":null}"#
        );
    }

    #[test]
    fn an_opening_response_has_no_output_and_null_usage() {
        let response = Response::new("resp_1", 7, "in_progress", "m", Vec::new(), Vec::new());
        assert_eq!(
            response.to_json_string(),
            r#"{"id":"resp_1","object":"response","created_at":7,"status":"in_progress","error":null,"incomplete_details":null,"model":"m","output":[],"usage":null}"#
        );
    }

    #[test]
    fn every_event_kind_has_its_payload() {
        let message = MessageItem::new("msg_1");
        let events = [
            (
                ResponseEvent::new(
                    "response.created",
                    0,
                    EventBody::Response {
                        response: Response::new("resp_1", 7, "in_progress", "m", Vec::new(), Vec::new()),
                    },
                ),
                r#"{"type":"response.created","sequence_number":0,"response":{"id":"resp_1","object":"response","created_at":7,"status":"in_progress","error":null,"incomplete_details":null,"model":"m","output":[],"usage":null}}"#,
            ),
            (
                ResponseEvent::new(
                    "response.output_item.added",
                    2,
                    EventBody::OutputItem {
                        output_index: 0,
                        item: OutputItem::Message(message),
                    },
                ),
                r#"{"type":"response.output_item.added","sequence_number":2,"output_index":0,"item":{"id":"msg_1","type":"message","status":"in_progress","role":"assistant","content":[]}}"#,
            ),
            (
                ResponseEvent::new(
                    "response.content_part.added",
                    3,
                    EventBody::ContentPart {
                        item_id: "msg_1".into(),
                        output_index: 0,
                        content_index: 0,
                        part: ContentPart::OutputText(OutputText::new("")),
                    },
                ),
                r#"{"type":"response.content_part.added","sequence_number":3,"item_id":"msg_1","output_index":0,"content_index":0,"part":{"type":"output_text","text":"","annotations":[]}}"#,
            ),
            (
                ResponseEvent::new(
                    "response.content_part.done",
                    4,
                    EventBody::ContentPart {
                        item_id: "rs_1".into(),
                        output_index: 0,
                        content_index: 0,
                        part: ContentPart::ReasoningText(ReasoningText::new("r")),
                    },
                ),
                r#"{"type":"response.content_part.done","sequence_number":4,"item_id":"rs_1","output_index":0,"content_index":0,"part":{"type":"reasoning_text","text":"r"}}"#,
            ),
            (
                ResponseEvent::new(
                    "response.output_text.delta",
                    5,
                    EventBody::TextDelta {
                        item_id: "msg_1".into(),
                        output_index: 0,
                        content_index: 0,
                        delta: "h\u{1b}\u{b}\u{1f}".into(),
                    },
                ),
                r#"{"type":"response.output_text.delta","sequence_number":5,"item_id":"msg_1","output_index":0,"content_index":0,"delta":"h\u001B\u000B\u001F"}"#,
            ),
            (
                ResponseEvent::new(
                    "response.output_text.done",
                    6,
                    EventBody::TextDone {
                        item_id: "msg_1".into(),
                        output_index: 0,
                        content_index: 0,
                        text: "hi".into(),
                    },
                ),
                r#"{"type":"response.output_text.done","sequence_number":6,"item_id":"msg_1","output_index":0,"content_index":0,"text":"hi"}"#,
            ),
            (
                ResponseEvent::new(
                    "response.function_call_arguments.delta",
                    7,
                    EventBody::ArgumentsDelta {
                        item_id: "fc_1".into(),
                        output_index: 1,
                        delta: "{}".into(),
                    },
                ),
                r#"{"type":"response.function_call_arguments.delta","sequence_number":7,"item_id":"fc_1","output_index":1,"delta":"{}"}"#,
            ),
            (
                ResponseEvent::new(
                    "response.function_call_arguments.done",
                    8,
                    EventBody::ArgumentsDone {
                        item_id: "fc_1".into(),
                        output_index: 1,
                        arguments: "{}".into(),
                    },
                ),
                r#"{"type":"response.function_call_arguments.done","sequence_number":8,"item_id":"fc_1","output_index":1,"arguments":"{}"}"#,
            ),
            (
                ResponseEvent::new("error", 9, EventBody::error(&ApiError::timeout())),
                r#"{"type":"error","sequence_number":9,"code":"timeout","message":"The request timed out before generation completed.","param":null}"#,
            ),
        ];
        for (event, payload) in events {
            assert_eq!(event.to_json_string(), payload);
        }
    }
}
