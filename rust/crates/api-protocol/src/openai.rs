//! OpenAI Chat Completions: the request, the `chat.completion` reply, the `chat.completion.chunk` stream payloads and
//! the model list.
//!
//! Requests accept every top-level field: the ones the server understands are typed, the rest are kept in `other` in
//! the order they arrived, for the mapper to accept or reject. A field of the wrong type fails to deserialise; a
//! `null` is the same as an absent field. Replies serialise their fields in the order the declarations give.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

/// A Chat Completions request. `stop`, `tools`, `tool_choice`, `reasoning_effort`, `chat_template_kwargs` and
/// `response_format` stay untyped JSON so the mapper can report precise errors and render tool schemas in their
/// original key order.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    pub messages: Option<Vec<ChatMessage>>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub max_tokens: Option<i32>,
    pub max_completion_tokens: Option<i32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i32>,
    pub seed: Option<i64>,
    pub stop: Option<Value>,
    pub tools: Option<Value>,
    pub tool_choice: Option<Value>,
    pub parallel_tool_calls: Option<bool>,
    pub reasoning_effort: Option<Value>,
    pub chat_template_kwargs: Option<Value>,
    pub response_format: Option<Value>,
    pub return_token_ids: Option<bool>,
    /// Every other top-level field.
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

/// One request message. `content` is a string, a list of content parts, or absent; `tool_calls`, `tool_call_id` and
/// `reasoning_content` stay untyped JSON for validation.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct ChatMessage {
    pub role: Option<String>,
    pub content: Option<Value>,
    pub name: Option<String>,
    pub tool_calls: Option<Value>,
    pub tool_call_id: Option<Value>,
    pub reasoning_content: Option<Value>,
    /// Every other field of the message.
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

/// The `stream_options` of a request.
#[derive(Clone, Debug, Default, PartialEq, Deserialize, Serialize)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
    /// Every other field of the options.
    #[serde(flatten)]
    pub other: Map<String, Value>,
}

/// A function call the assistant made. `arguments` is a JSON object serialised as a string.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFunction,
}

/// The named function and its arguments of a [`ToolCall`].
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

impl ToolCall {
    /// A call of the function `name` with `arguments` (a JSON object as text).
    pub fn function(id: impl Into<String>, name: impl Into<String>, arguments: impl Into<String>) -> ToolCall {
        ToolCall {
            id: id.into(),
            kind: "function".to_owned(),
            function: ToolCallFunction {
                name: name.into(),
                arguments: arguments.into(),
            },
        }
    }
}

/// Token accounting of one completion. Absent details are left out.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_tokens_details: Option<PromptDetails>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completion_tokens_details: Option<CompletionDetails>,
}

/// The prompt tokens restored from the prefix cache (part of `prompt_tokens`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct PromptDetails {
    pub cached_tokens: i32,
}

/// The reasoning tokens (part of `completion_tokens`).
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct CompletionDetails {
    pub reasoning_tokens: i32,
}

impl Usage {
    /// The usage of a completion; `reasoning_tokens` of `None` leaves out the completion details.
    pub fn of(prompt_tokens: i32, cached_tokens: i32, completion_tokens: i32, reasoning_tokens: Option<i32>) -> Usage {
        Usage {
            prompt_tokens,
            completion_tokens,
            total_tokens: prompt_tokens + completion_tokens,
            prompt_tokens_details: Some(PromptDetails { cached_tokens }),
            completion_tokens_details: reasoning_tokens.map(|reasoning_tokens| CompletionDetails { reasoning_tokens }),
        }
    }
}

/// The non-streaming `chat.completion` object with exactly one assistant choice.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub prompt_token_ids: Option<Vec<i32>>,
}

/// A choice of a [`ChatCompletionResponse`]; `logprobs` and `finish_reason` are explicit nulls when absent.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Choice {
    pub index: i32,
    pub message: AssistantMessage,
    pub logprobs: Option<Value>,
    pub finish_reason: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub token_ids: Option<Vec<i32>>,
}

/// The assistant message of a choice. `content` is null when the assistant only called tools and `refusal` is always
/// null; `reasoning_content` and `tool_calls` are left out when empty.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct AssistantMessage {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    pub refusal: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
}

/// The parts of a finished generation that make a [`ChatCompletionResponse`].
#[derive(Clone, Debug, PartialEq)]
pub struct Completion {
    pub id: String,
    pub created: i64,
    pub model: String,
    pub reasoning: String,
    pub content: String,
    pub tool_calls: Vec<ToolCall>,
    pub finish_reason: String,
    pub usage: Usage,
    pub prompt_token_ids: Option<Vec<i32>>,
    pub token_ids: Option<Vec<i32>>,
}

impl ChatCompletionResponse {
    /// The reply for a finished generation: no reasoning leaves `reasoning_content` out, and an assistant that called
    /// tools has `tool_calls` and, when it wrote nothing, a null `content`.
    pub fn of(completion: Completion) -> ChatCompletionResponse {
        let reasoning_content = (!completion.reasoning.is_empty()).then_some(completion.reasoning);
        let message = if completion.tool_calls.is_empty() {
            AssistantMessage {
                role: "assistant".to_owned(),
                content: Some(completion.content),
                reasoning_content,
                refusal: None,
                tool_calls: None,
            }
        } else {
            AssistantMessage {
                role: "assistant".to_owned(),
                content: (!completion.content.is_empty()).then_some(completion.content),
                reasoning_content,
                refusal: None,
                tool_calls: Some(completion.tool_calls),
            }
        };
        ChatCompletionResponse {
            id: completion.id,
            object: "chat.completion".to_owned(),
            created: completion.created,
            model: completion.model,
            choices: vec![Choice {
                index: 0,
                message,
                logprobs: None,
                finish_reason: Some(completion.finish_reason),
                token_ids: completion.token_ids,
            }],
            usage: completion.usage,
            prompt_token_ids: completion.prompt_token_ids,
        }
    }
}

/// A streaming `chat.completion.chunk`. `usage` appears only on the optional final usage chunk.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub usage: Option<Usage>,
}

/// A choice of a chunk; `logprobs` and `finish_reason` are explicit nulls when absent.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ChunkChoice {
    pub index: i32,
    pub delta: Delta,
    pub logprobs: Option<Value>,
    pub finish_reason: Option<String>,
}

/// The increment of a chunk; absent parts are left out. `reasoning_content` precedes any `content`.
#[derive(Clone, Debug, Default, PartialEq, Serialize)]
pub struct Delta {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub role: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reasoning_content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Streaming form of [`ToolCall`]; `index` identifies the call across chunks.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ToolCallDelta {
    pub index: i32,
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolCallFunction,
}

impl ChatCompletionChunk {
    fn choice(id: &str, created: i64, model: &str, delta: Delta, finish_reason: Option<&str>) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: id.to_owned(),
            object: "chat.completion.chunk".to_owned(),
            created,
            model: model.to_owned(),
            choices: vec![ChunkChoice {
                index: 0,
                delta,
                logprobs: None,
                finish_reason: finish_reason.map(str::to_owned),
            }],
            usage: None,
        }
    }

    /// The opening chunk: the assistant role with empty content.
    pub fn role(id: &str, created: i64, model: &str) -> ChatCompletionChunk {
        let delta = Delta {
            role: Some("assistant".to_owned()),
            content: Some(String::new()),
            ..Delta::default()
        };
        ChatCompletionChunk::choice(id, created, model, delta, None)
    }

    /// A piece of the answer text.
    pub fn content(id: &str, created: i64, model: &str, text: &str) -> ChatCompletionChunk {
        let delta = Delta {
            content: Some(text.to_owned()),
            ..Delta::default()
        };
        ChatCompletionChunk::choice(id, created, model, delta, None)
    }

    /// A piece of the reasoning text.
    pub fn reasoning(id: &str, created: i64, model: &str, text: &str) -> ChatCompletionChunk {
        let delta = Delta {
            reasoning_content: Some(text.to_owned()),
            ..Delta::default()
        };
        ChatCompletionChunk::choice(id, created, model, delta, None)
    }

    /// One complete call per chunk: `id`, `type`, `name` and the full `arguments` arrive together.
    pub fn tool_call(id: &str, created: i64, model: &str, index: i32, call: ToolCall) -> ChatCompletionChunk {
        let call = ToolCallDelta {
            index,
            id: call.id,
            kind: call.kind,
            function: call.function,
        };
        let delta = Delta {
            tool_calls: Some(vec![call]),
            ..Delta::default()
        };
        ChatCompletionChunk::choice(id, created, model, delta, None)
    }

    /// The chunk that carries the finish reason.
    pub fn finish(id: &str, created: i64, model: &str, finish_reason: &str) -> ChatCompletionChunk {
        ChatCompletionChunk::choice(id, created, model, Delta::default(), Some(finish_reason))
    }

    /// The final usage chunk, with no choices.
    pub fn usage(id: &str, created: i64, model: &str, usage: Usage) -> ChatCompletionChunk {
        ChatCompletionChunk {
            id: id.to_owned(),
            object: "chat.completion.chunk".to_owned(),
            created,
            model: model.to_owned(),
            choices: Vec::new(),
            usage: Some(usage),
        }
    }
}

/// The data of the event that ends a chat completion stream.
pub const STREAM_DONE: &str = "[DONE]";

/// The `/v1/models` listing; only the public model id is exposed.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<Model>,
}

/// One served model.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Model {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub owned_by: String,
}

impl Model {
    /// The model `id`, created at `created` (seconds since the epoch).
    pub fn of(id: &str, created: i64) -> Model {
        Model {
            id: id.to_owned(),
            object: "model".to_owned(),
            created,
            owned_by: "euhedral".to_owned(),
        }
    }
}

impl ModelList {
    /// A listing of the one served model.
    pub fn of(model: Model) -> ModelList {
        ModelList {
            object: "list".to_owned(),
            data: vec![model],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json<T: Serialize>(value: &T) -> String {
        serde_json::to_string(value).expect("serialises")
    }

    #[test]
    fn a_request_with_every_known_field_deserialises() {
        let request: ChatCompletionRequest = serde_json::from_str(
            r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":[{"type":"text","text":"hi"}],"name":"bob"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}],"reasoning_content":"r"},
            {"role":"tool","tool_call_id":"c","content":"x"}],
            "stream":true,"stream_options":{"include_usage":true,"extra":1},"max_tokens":5,"max_completion_tokens":6,"temperature":0.5,
            "top_p":0.9,"top_k":4,"seed":123456789012,"stop":["a","b"],"tools":[{"type":"function","function":{"name":"f","parameters":{"z":1,"a":2}}}],
            "tool_choice":"auto","parallel_tool_calls":false,"reasoning_effort":"low","chat_template_kwargs":{"enable_thinking":false},
            "response_format":{"type":"json_object"},"return_token_ids":true,"n":2,"user":"u"}"#,
        )
        .expect("deserialises");
        assert_eq!(request.model.as_deref(), Some("m"));
        let messages = request.messages.expect("messages");
        assert_eq!(messages.len(), 4);
        assert_eq!(messages[1].name.as_deref(), Some("bob"));
        assert!(messages[2].content.is_none());
        assert!(messages[2].tool_calls.is_some());
        assert_eq!(messages[3].tool_call_id, Some(Value::String("c".into())));
        let options = request.stream_options.expect("options");
        assert_eq!(options.include_usage, Some(true));
        assert!(options.other.contains_key("extra"));
        assert_eq!(request.stream, Some(true));
        assert_eq!(
            (request.max_tokens, request.max_completion_tokens, request.top_k),
            (Some(5), Some(6), Some(4))
        );
        assert_eq!(
            (request.temperature, request.top_p, request.seed),
            (Some(0.5), Some(0.9), Some(123456789012))
        );
        assert_eq!(json(&request.stop.expect("stop")), r#"["a","b"]"#);
        assert_eq!(
            json(&request.tools.expect("tools")),
            r#"[{"type":"function","function":{"name":"f","parameters":{"z":1,"a":2}}}]"#
        );
        assert_eq!(request.parallel_tool_calls, Some(false));
        assert_eq!(request.return_token_ids, Some(true));
        assert_eq!(request.other.keys().collect::<Vec<_>>(), ["n", "user"]);
    }

    #[test]
    fn an_empty_object_and_explicit_nulls_are_the_same() {
        let empty: ChatCompletionRequest = serde_json::from_str("{}").expect("empty");
        let nulls: ChatCompletionRequest =
            serde_json::from_str(r#"{"model":null,"messages":null,"stream":null,"stop":null,"tools":null}"#)
                .expect("nulls");
        assert_eq!(empty, nulls);
        assert!(empty.other.is_empty());
    }

    #[test]
    fn a_field_of_the_wrong_type_is_rejected() {
        assert!(serde_json::from_str::<ChatCompletionRequest>(r#"{"stream":"yes"}"#).is_err());
        assert!(serde_json::from_str::<ChatCompletionRequest>(r#"{"messages":"hi"}"#).is_err());
        assert!(serde_json::from_str::<ChatCompletionRequest>("[]").is_err());
    }

    fn completion() -> Completion {
        Completion {
            id: "chatcmpl-1".into(),
            created: 7,
            model: "m".into(),
            reasoning: String::new(),
            content: "hello".into(),
            tool_calls: Vec::new(),
            finish_reason: "stop".into(),
            usage: Usage::of(10, 4, 3, None),
            prompt_token_ids: None,
            token_ids: Some(vec![1, 2]),
        }
    }

    #[test]
    fn a_plain_reply_has_null_logprobs_and_refusal() {
        assert_eq!(
            json(&ChatCompletionResponse::of(completion())),
            r#"{"id":"chatcmpl-1","object":"chat.completion","created":7,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":"hello","refusal":null},"logprobs":null,"finish_reason":"stop","token_ids":[1,2]}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":4}}}"#
        );
    }

    #[test]
    fn a_tool_call_reply_has_null_content_and_reasoning_when_present() {
        let mut parts = completion();
        parts.content = String::new();
        parts.reasoning = "think".into();
        parts.tool_calls = vec![ToolCall::function("call_1", "f", r#"{"a":1}"#)];
        parts.finish_reason = "tool_calls".into();
        parts.usage = Usage::of(10, 0, 3, Some(2));
        parts.prompt_token_ids = Some(vec![5, 6]);
        assert_eq!(
            json(&ChatCompletionResponse::of(parts)),
            r#"{"id":"chatcmpl-1","object":"chat.completion","created":7,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_content":"think","refusal":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]},"logprobs":null,"finish_reason":"tool_calls","token_ids":[1,2]}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens_details":{"reasoning_tokens":2}},"prompt_token_ids":[5,6]}"#
        );
    }

    #[test]
    fn every_chunk_kind_has_its_payload() {
        let (id, model) = ("c1", "m");
        let chunks = [
            ChatCompletionChunk::role(id, 7, model),
            ChatCompletionChunk::reasoning(id, 7, model, "r"),
            ChatCompletionChunk::content(id, 7, model, "t"),
            ChatCompletionChunk::tool_call(id, 7, model, 1, ToolCall::function("call_1", "f", "{}")),
            ChatCompletionChunk::finish(id, 7, model, "stop"),
            ChatCompletionChunk::usage(id, 7, model, Usage::of(2, 1, 3, Some(1))),
        ];
        let head = r#"{"id":"c1","object":"chat.completion.chunk","created":7,"model":"m","choices":["#;
        let expected = [
            r#"{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}"#,
            r#"{"index":0,"delta":{"reasoning_content":"r"},"logprobs":null,"finish_reason":null}]}"#,
            r#"{"index":0,"delta":{"content":"t"},"logprobs":null,"finish_reason":null}]}"#,
            r#"{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},"logprobs":null,"finish_reason":null}]}"#,
            r#"{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}"#,
            r#"],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5,"prompt_tokens_details":{"cached_tokens":1},"completion_tokens_details":{"reasoning_tokens":1}}}"#,
        ];
        for (chunk, tail) in chunks.iter().zip(expected) {
            assert_eq!(json(chunk), format!("{head}{tail}"));
        }
    }

    #[test]
    fn the_model_list_and_model_have_the_listing_shape() {
        let model = Model::of("m", 9);
        assert_eq!(
            json(&model),
            r#"{"id":"m","object":"model","created":9,"owned_by":"euhedral"}"#
        );
        assert_eq!(
            json(&ModelList::of(model)),
            r#"{"object":"list","data":[{"id":"m","object":"model","created":9,"owned_by":"euhedral"}]}"#
        );
    }
}
