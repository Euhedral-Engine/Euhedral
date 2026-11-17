//! Anthropic Messages: the request, the `message` reply, the stream event payloads, the token count and the model
//! list.
//!
//! Requests accept every top-level field: the ones the server understands are typed, the rest are kept in `other` in
//! the order they arrived. A field of the wrong type fails to deserialise; a `null` is the same as an absent field.
//! Replies serialise their fields in the order the declarations give, `type` first where there is one.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

pub use crate::error::{AnthropicErrorBody, AnthropicErrorEnvelope};

/// A Messages request (`POST /v1/messages`, and `/v1/messages/count_tokens` without `max_tokens`). Messages and the
/// fields that take several shapes stay untyped JSON so the mapper can report precise errors.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct MessagesRequest {
    pub model: Option<String>,
    pub messages: Option<Vec<Value>>,
    pub system: Option<Value>,
    pub max_tokens: Option<i32>,
    pub stop_sequences: Option<Value>,
    pub stream: Option<bool>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i32>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub thinking: Option<Value>,
    pub output_config: Option<Value>,
    pub output_format: Option<Value>,
    /// Every other top-level field.
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

/// Token accounting of a message: `input_tokens` counts the prompt tokens that were prefilled and
/// `cache_read_input_tokens` those restored from the prefix cache.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Usage {
    pub input_tokens: i32,
    pub cache_creation_input_tokens: i32,
    pub cache_read_input_tokens: i32,
    pub output_tokens: i32,
}

impl Usage {
    /// The usage of a generation of `prompt_tokens` (cached ones included) and `output_tokens`; nothing is ever
    /// written to a cache on request.
    pub fn of(prompt_tokens: i32, cached_tokens: i32, output_tokens: i32) -> Usage {
        Usage {
            input_tokens: prompt_tokens - cached_tokens,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: cached_tokens,
            output_tokens,
        }
    }
}

/// A content block of a message, tagged by `type`.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ContentBlock {
    Thinking { thinking: String, signature: String },
    Text { text: String },
    ToolUse { id: String, name: String, input: Value },
}

/// An assistant `message`; `stop_reason` and `stop_sequence` are explicit nulls while unknown.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Message {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub role: &'static str,
    pub model: String,
    pub content: Vec<ContentBlock>,
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
    pub usage: Usage,
}

impl Message {
    /// An assistant message with the given content and no stop information yet.
    pub fn new(id: String, model: String, content: Vec<ContentBlock>, usage: Usage) -> Message {
        Message {
            id,
            kind: "message",
            role: "assistant",
            model,
            content,
            stop_reason: None,
            stop_sequence: None,
            usage,
        }
    }
}

/// An increment of the open content block, tagged by `type`.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum BlockDelta {
    ThinkingDelta { thinking: String },
    TextDelta { text: String },
    InputJsonDelta { partial_json: String },
    SignatureDelta { signature: String },
}

/// The stop information of a stream's `message_delta`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct MessageDeltaBody {
    pub stop_reason: Option<String>,
    pub stop_sequence: Option<String>,
}

/// One event of a message stream; the payload's `type` is the event name.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum StreamEvent {
    MessageStart {
        message: Message,
    },
    ContentBlockStart {
        index: i32,
        content_block: ContentBlock,
    },
    ContentBlockDelta {
        index: i32,
        delta: BlockDelta,
    },
    ContentBlockStop {
        index: i32,
    },
    MessageDelta {
        delta: MessageDeltaBody,
        usage: Usage,
    },
    MessageStop,
    Ping,
    /// An in-band failure, after the headers are committed.
    Error {
        error: AnthropicErrorBody,
    },
}

impl StreamEvent {
    /// The SSE event name, equal to the payload's `type`.
    pub fn name(&self) -> &'static str {
        match self {
            StreamEvent::MessageStart { .. } => "message_start",
            StreamEvent::ContentBlockStart { .. } => "content_block_start",
            StreamEvent::ContentBlockDelta { .. } => "content_block_delta",
            StreamEvent::ContentBlockStop { .. } => "content_block_stop",
            StreamEvent::MessageDelta { .. } => "message_delta",
            StreamEvent::MessageStop => "message_stop",
            StreamEvent::Ping => "ping",
            StreamEvent::Error { .. } => "error",
        }
    }
}

/// The reply of `/v1/messages/count_tokens`.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct TokenCount {
    pub input_tokens: i32,
}

/// The `/v1/models` listing in Anthropic's form.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelList {
    pub data: Vec<Model>,
    pub has_more: bool,
    pub first_id: String,
    pub last_id: String,
}

/// One served model in Anthropic's form.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Model {
    #[serde(rename = "type")]
    pub kind: &'static str,
    pub id: String,
    pub display_name: String,
    pub created_at: String,
}

impl Model {
    /// The model `id`, created at `created` (seconds since the epoch).
    pub fn of(id: &str, created: i64) -> Model {
        Model {
            kind: "model",
            id: id.to_owned(),
            display_name: id.to_owned(),
            created_at: instant(created),
        }
    }
}

impl ModelList {
    /// A listing of the one served model.
    pub fn of(model: Model) -> ModelList {
        ModelList {
            has_more: false,
            first_id: model.id.clone(),
            last_id: model.id.clone(),
            data: vec![model],
        }
    }
}

/// An ISO-8601 UTC instant of whole seconds, such as `2024-05-17T08:50:00Z`.
fn instant(seconds: i64) -> String {
    let days = seconds.div_euclid(86_400);
    let rest = seconds.rem_euclid(86_400);
    // Civil date from days since 1970-01-01 (proleptic Gregorian).
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era = (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = day_of_year - (153 * month_index + 2) / 5 + 1;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    };
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    format!(
        "{year:04}-{month:02}-{day:02}T{:02}:{:02}:{:02}Z",
        rest / 3_600,
        rest % 3_600 / 60,
        rest % 60
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn text<T: Serialize>(value: &T) -> String {
        serde_json::to_string(value).expect("serialises")
    }

    #[test]
    fn a_request_with_every_known_field_deserialises() {
        let request: MessagesRequest = serde_json::from_str(
            r#"{"model":"m","max_tokens":64,"system":[{"type":"text","text":"s"}],
            "messages":[{"role":"user","content":"hi"},{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"f","input":{"b":1,"a":2}}]}],
            "stop_sequences":["x"],"stream":true,"temperature":0.2,"top_p":0.8,"top_k":3,
            "tools":[{"name":"f","input_schema":{"type":"object"}}],"tool_choice":{"type":"auto"},
            "thinking":{"type":"enabled","budget_tokens":10},"output_config":{"effort":"low"},"output_format":null,
            "metadata":{"user_id":"u"},"service_tier":"auto"}"#,
        )
        .expect("deserialises");
        assert_eq!(request.model.as_deref(), Some("m"));
        assert_eq!(request.max_tokens, Some(64));
        assert_eq!(request.messages.as_ref().map(Vec::len), Some(2));
        assert_eq!(
            text(&request.messages.expect("messages")[1]),
            r#"{"role":"assistant","content":[{"type":"tool_use","id":"t","name":"f","input":{"b":1,"a":2}}]}"#
        );
        assert_eq!((request.stream, request.top_k), (Some(true), Some(3)));
        assert!(request.output_format.is_none());
        assert_eq!(request.other.keys().collect::<Vec<_>>(), ["metadata", "service_tier"]);
    }

    #[test]
    fn a_count_tokens_request_needs_no_max_tokens() {
        let request: MessagesRequest =
            serde_json::from_str(r#"{"model":"m","messages":[{"role":"user","content":"hi"}]}"#).expect("deserialises");
        assert_eq!(request.max_tokens, None);
        assert!(serde_json::from_str::<MessagesRequest>(r#"{"max_tokens":"many"}"#).is_err());
    }

    #[test]
    fn a_message_serialises_its_blocks_in_order_with_explicit_null_stop_fields() {
        let mut message = Message::new(
            "msg_1".into(),
            "m".into(),
            vec![
                ContentBlock::Thinking {
                    thinking: "t".into(),
                    signature: "euhedral-x".into(),
                },
                ContentBlock::Text { text: "hi".into() },
                ContentBlock::ToolUse {
                    id: "toolu_1".into(),
                    name: "f".into(),
                    input: json!({"b": 1, "a": 2}),
                },
            ],
            Usage::of(10, 4, 3),
        );
        assert_eq!(
            text(&message),
            r#"{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[{"type":"thinking","thinking":"t","signature":"euhedral-x"},{"type":"text","text":"hi"},{"type":"tool_use","id":"toolu_1","name":"f","input":{"b":1,"a":2}}],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":6,"cache_creation_input_tokens":0,"cache_read_input_tokens":4,"output_tokens":3}}"#
        );
        message.stop_reason = Some("stop_sequence".into());
        message.stop_sequence = Some("x".into());
        assert!(text(&message).contains(r#""stop_reason":"stop_sequence","stop_sequence":"x""#));
    }

    #[test]
    fn every_stream_event_has_its_payload_and_name() {
        let message = Message::new("msg_1".into(), "m".into(), Vec::new(), Usage::of(5, 1, 0));
        let events = [
            (
                StreamEvent::MessageStart { message },
                "message_start",
                r#"{"type":"message_start","message":{"id":"msg_1","type":"message","role":"assistant","model":"m","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":4,"cache_creation_input_tokens":0,"cache_read_input_tokens":1,"output_tokens":0}}}"#,
            ),
            (
                StreamEvent::ContentBlockStart {
                    index: 0,
                    content_block: ContentBlock::Thinking {
                        thinking: String::new(),
                        signature: String::new(),
                    },
                },
                "content_block_start",
                r#"{"type":"content_block_start","index":0,"content_block":{"type":"thinking","thinking":"","signature":""}}"#,
            ),
            (
                StreamEvent::ContentBlockStart {
                    index: 1,
                    content_block: ContentBlock::Text { text: String::new() },
                },
                "content_block_start",
                r#"{"type":"content_block_start","index":1,"content_block":{"type":"text","text":""}}"#,
            ),
            (
                StreamEvent::ContentBlockStart {
                    index: 2,
                    content_block: ContentBlock::ToolUse {
                        id: "toolu_1".into(),
                        name: "f".into(),
                        input: json!({}),
                    },
                },
                "content_block_start",
                r#"{"type":"content_block_start","index":2,"content_block":{"type":"tool_use","id":"toolu_1","name":"f","input":{}}}"#,
            ),
            (
                StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: BlockDelta::ThinkingDelta { thinking: "r".into() },
                },
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"thinking_delta","thinking":"r"}}"#,
            ),
            (
                StreamEvent::ContentBlockDelta {
                    index: 1,
                    delta: BlockDelta::TextDelta { text: "t".into() },
                },
                "content_block_delta",
                r#"{"type":"content_block_delta","index":1,"delta":{"type":"text_delta","text":"t"}}"#,
            ),
            (
                StreamEvent::ContentBlockDelta {
                    index: 2,
                    delta: BlockDelta::InputJsonDelta {
                        partial_json: "{}".into(),
                    },
                },
                "content_block_delta",
                r#"{"type":"content_block_delta","index":2,"delta":{"type":"input_json_delta","partial_json":"{}"}}"#,
            ),
            (
                StreamEvent::ContentBlockDelta {
                    index: 0,
                    delta: BlockDelta::SignatureDelta { signature: "s".into() },
                },
                "content_block_delta",
                r#"{"type":"content_block_delta","index":0,"delta":{"type":"signature_delta","signature":"s"}}"#,
            ),
            (
                StreamEvent::ContentBlockStop { index: 2 },
                "content_block_stop",
                r#"{"type":"content_block_stop","index":2}"#,
            ),
            (
                StreamEvent::MessageDelta {
                    delta: MessageDeltaBody {
                        stop_reason: Some("end_turn".into()),
                        stop_sequence: None,
                    },
                    usage: Usage::of(5, 1, 7),
                },
                "message_delta",
                r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":4,"cache_creation_input_tokens":0,"cache_read_input_tokens":1,"output_tokens":7}}"#,
            ),
            (StreamEvent::MessageStop, "message_stop", r#"{"type":"message_stop"}"#),
            (StreamEvent::Ping, "ping", r#"{"type":"ping"}"#),
        ];
        for (event, name, payload) in events {
            assert_eq!(event.name(), name);
            assert_eq!(text(&event), payload);
        }
    }

    #[test]
    fn an_in_band_error_is_the_error_object() {
        let failure = crate::ApiError::timeout();
        let event = StreamEvent::Error {
            error: failure.anthropic_envelope().error,
        };
        assert_eq!(event.name(), "error");
        assert_eq!(
            text(&event),
            r#"{"type":"error","error":{"type":"overloaded_error","message":"The request timed out before generation completed."}}"#
        );
        assert_eq!(
            text(&event).as_bytes(),
            failure.body(crate::Surface::Anthropic).as_slice()
        );
    }

    #[test]
    fn the_token_count_and_model_list_have_their_shapes() {
        assert_eq!(text(&TokenCount { input_tokens: 12 }), r#"{"input_tokens":12}"#);
        assert_eq!(
            text(&ModelList::of(Model::of("m", 1_715_935_800))),
            r#"{"data":[{"type":"model","id":"m","display_name":"m","created_at":"2024-05-17T08:50:00Z"}],"has_more":false,"first_id":"m","last_id":"m"}"#
        );
    }

    #[test]
    fn instants_are_calendar_correct() {
        assert_eq!(instant(0), "1970-01-01T00:00:00Z");
        assert_eq!(instant(951_782_400), "2000-02-29T00:00:00Z");
        assert_eq!(instant(1_767_225_599), "2025-12-31T23:59:59Z");
        assert_eq!(instant(-1), "1969-12-31T23:59:59Z");
    }
}
