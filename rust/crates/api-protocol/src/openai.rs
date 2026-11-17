//! OpenAI Chat Completions: the request, the `chat.completion` reply, the `chat.completion.chunk` stream payloads and
//! the model list.
//!
//! A request body is read as JSON and then field by field: the fields the server understands are typed (scalars are
//! read as [`crate::coerce`] describes), the rest are kept in `other` in the order they arrived, and fields that take
//! several shapes stay [`JsonValue`]s. A `null` is the same as an absent field. Replies are [`ToJson`]: their members
//! are written in declaration order, with explicit nulls only where stated.

use crate::coerce;
use crate::error::ApiError;
use crate::json::JsonValue;
use crate::wire::{
    Members, Reader, ToJson, int, int_list, list, members, object, optional_text, push, push_some, text,
};

/// A Chat Completions request. `stop`, `tools`, `tool_choice`, `reasoning_effort`, `chat_template_kwargs` and
/// `response_format` stay untyped so the mapper can report precise errors and render tool schemas in their original
/// key order.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatCompletionRequest {
    pub model: Option<String>,
    /// The messages; a `null` element is kept as `None`.
    pub messages: Option<Vec<Option<ChatMessage>>>,
    pub stream: Option<bool>,
    pub stream_options: Option<StreamOptions>,
    pub max_tokens: Option<i32>,
    pub max_completion_tokens: Option<i32>,
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<i32>,
    pub seed: Option<i64>,
    pub stop: Option<JsonValue>,
    pub tools: Option<JsonValue>,
    pub tool_choice: Option<JsonValue>,
    pub parallel_tool_calls: Option<bool>,
    pub reasoning_effort: Option<JsonValue>,
    pub chat_template_kwargs: Option<JsonValue>,
    pub response_format: Option<JsonValue>,
    pub return_token_ids: Option<bool>,
    /// Every other top-level field, in arrival order; nulls included.
    pub other: Members,
}

impl ChatCompletionRequest {
    /// Reads a request body; any body that is not a JSON object of the documented shape is an unparsable body.
    pub fn parse(body: &str) -> Result<ChatCompletionRequest, ApiError> {
        let mut reader = Reader::parse(body)?;
        let mut request = ChatCompletionRequest {
            model: reader
                .take("model")
                .map(|value| coerce::string(&value))
                .transpose()?
                .flatten(),
            messages: match reader.take("messages") {
                Some(value) => {
                    let items = coerce::array(&value)?.unwrap_or_default();
                    Some(items.iter().map(message_or_null).collect::<Result<Vec<_>, _>>()?)
                }
                None => None,
            },
            stream: reader
                .take("stream")
                .map(|value| coerce::boolean(&value))
                .transpose()?
                .flatten(),
            stream_options: reader.take("stream_options").map(StreamOptions::read).transpose()?,
            max_tokens: reader
                .take("max_tokens")
                .map(|value| coerce::integer(&value))
                .transpose()?
                .flatten(),
            max_completion_tokens: reader
                .take("max_completion_tokens")
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
            top_k: reader
                .take("top_k")
                .map(|value| coerce::integer(&value))
                .transpose()?
                .flatten(),
            seed: reader
                .take("seed")
                .map(|value| coerce::long(&value))
                .transpose()?
                .flatten(),
            stop: reader.take("stop"),
            tools: reader.take("tools"),
            tool_choice: reader.take("tool_choice"),
            parallel_tool_calls: reader
                .take("parallel_tool_calls")
                .map(|value| coerce::boolean(&value))
                .transpose()?
                .flatten(),
            reasoning_effort: reader.take("reasoning_effort"),
            chat_template_kwargs: reader.take("chat_template_kwargs"),
            response_format: reader.take("response_format"),
            return_token_ids: reader
                .take("return_token_ids")
                .map(|value| coerce::boolean(&value))
                .transpose()?
                .flatten(),
            other: Vec::new(),
        };
        request.other = reader.rest();
        Ok(request)
    }
}

fn message_or_null(value: &JsonValue) -> Result<Option<ChatMessage>, ApiError> {
    match value {
        JsonValue::Null => Ok(None),
        other => ChatMessage::read(other.clone()).map(Some),
    }
}

/// One request message. `content` is a string, a list of content parts, or absent; `tool_calls`, `tool_call_id` and
/// `reasoning_content` stay untyped for validation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct ChatMessage {
    pub role: Option<String>,
    pub content: Option<JsonValue>,
    pub name: Option<String>,
    pub tool_calls: Option<JsonValue>,
    pub tool_call_id: Option<JsonValue>,
    pub reasoning_content: Option<JsonValue>,
    /// Every other field of the message, in arrival order; nulls included.
    pub other: Members,
}

impl ChatMessage {
    fn read(value: JsonValue) -> Result<ChatMessage, ApiError> {
        let mut reader = Reader::of(value)?;
        let mut message = ChatMessage {
            role: reader
                .take("role")
                .map(|value| coerce::string(&value))
                .transpose()?
                .flatten(),
            content: reader.take("content"),
            name: reader
                .take("name")
                .map(|value| coerce::string(&value))
                .transpose()?
                .flatten(),
            tool_calls: reader.take("tool_calls"),
            tool_call_id: reader.take("tool_call_id"),
            reasoning_content: reader.take("reasoning_content"),
            other: Vec::new(),
        };
        message.other = reader.rest();
        Ok(message)
    }
}

/// The `stream_options` of a request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct StreamOptions {
    pub include_usage: Option<bool>,
    /// Every other field of the options, in arrival order; nulls included.
    pub other: Members,
}

impl StreamOptions {
    fn read(value: JsonValue) -> Result<StreamOptions, ApiError> {
        let mut reader = Reader::of(value)?;
        let include_usage = reader
            .take("include_usage")
            .map(|value| coerce::boolean(&value))
            .transpose()?
            .flatten();
        Ok(StreamOptions {
            include_usage,
            other: reader.rest(),
        })
    }
}

/// A function call the assistant made. `arguments` is a JSON object serialised as a string.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCall {
    pub id: String,
    pub kind: String,
    pub function: ToolCallFunction,
}

/// The named function and its arguments of a [`ToolCall`].
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallFunction {
    pub name: String,
    pub arguments: String,
}

impl ToJson for ToolCallFunction {
    fn to_json(&self) -> JsonValue {
        object([("name", text(&self.name)), ("arguments", text(&self.arguments))])
    }
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

impl ToJson for ToolCall {
    fn to_json(&self) -> JsonValue {
        object([
            ("id", text(&self.id)),
            ("type", text(&self.kind)),
            ("function", self.function.to_json()),
        ])
    }
}

/// Token accounting of one completion. Absent details are left out.
#[derive(Clone, Debug, PartialEq)]
pub struct Usage {
    pub prompt_tokens: i32,
    pub completion_tokens: i32,
    pub total_tokens: i32,
    pub prompt_tokens_details: Option<PromptDetails>,
    pub completion_tokens_details: Option<CompletionDetails>,
}

/// The prompt tokens restored from the prefix cache (part of `prompt_tokens`).
#[derive(Clone, Debug, PartialEq)]
pub struct PromptDetails {
    pub cached_tokens: i32,
}

/// The reasoning tokens (part of `completion_tokens`).
#[derive(Clone, Debug, PartialEq)]
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

impl ToJson for Usage {
    fn to_json(&self) -> JsonValue {
        let mut fields = Vec::new();
        push(&mut fields, "prompt_tokens", int(self.prompt_tokens));
        push(&mut fields, "completion_tokens", int(self.completion_tokens));
        push(&mut fields, "total_tokens", int(self.total_tokens));
        push_some(
            &mut fields,
            "prompt_tokens_details",
            self.prompt_tokens_details
                .as_ref()
                .map(|details| object([("cached_tokens", int(details.cached_tokens))])),
        );
        push_some(
            &mut fields,
            "completion_tokens_details",
            self.completion_tokens_details
                .as_ref()
                .map(|details| object([("reasoning_tokens", int(details.reasoning_tokens))])),
        );
        members(fields)
    }
}

/// The non-streaming `chat.completion` object with exactly one assistant choice.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatCompletionResponse {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
    pub prompt_token_ids: Option<Vec<i32>>,
}

/// A choice of a [`ChatCompletionResponse`]; `logprobs` and `finish_reason` are explicit nulls when absent.
#[derive(Clone, Debug, PartialEq)]
pub struct Choice {
    pub index: i32,
    pub message: AssistantMessage,
    pub logprobs: Option<JsonValue>,
    pub finish_reason: Option<String>,
    pub token_ids: Option<Vec<i32>>,
}

/// The assistant message of a choice. `content` is null when the assistant only called tools and `refusal` is always
/// null; `reasoning_content` and `tool_calls` are left out when absent.
#[derive(Clone, Debug, PartialEq)]
pub struct AssistantMessage {
    pub role: String,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub refusal: Option<String>,
    pub tool_calls: Option<Vec<ToolCall>>,
}

impl ToJson for AssistantMessage {
    fn to_json(&self) -> JsonValue {
        let mut fields = Vec::new();
        push(&mut fields, "role", text(&self.role));
        push(&mut fields, "content", optional_text(self.content.as_deref()));
        push_some(
            &mut fields,
            "reasoning_content",
            self.reasoning_content.as_deref().map(text),
        );
        push(&mut fields, "refusal", optional_text(self.refusal.as_deref()));
        push_some(&mut fields, "tool_calls", self.tool_calls.as_deref().map(list));
        members(fields)
    }
}

impl ToJson for Choice {
    fn to_json(&self) -> JsonValue {
        let mut fields = Vec::new();
        push(&mut fields, "index", int(self.index));
        push(&mut fields, "message", self.message.to_json());
        push(
            &mut fields,
            "logprobs",
            self.logprobs.clone().unwrap_or(JsonValue::Null),
        );
        push(
            &mut fields,
            "finish_reason",
            optional_text(self.finish_reason.as_deref()),
        );
        push_some(&mut fields, "token_ids", self.token_ids.as_deref().map(int_list));
        members(fields)
    }
}

impl ToJson for ChatCompletionResponse {
    fn to_json(&self) -> JsonValue {
        let mut fields = Vec::new();
        push(&mut fields, "id", text(&self.id));
        push(&mut fields, "object", text(&self.object));
        push(&mut fields, "created", int(self.created));
        push(&mut fields, "model", text(&self.model));
        push(&mut fields, "choices", list(&self.choices));
        push(&mut fields, "usage", self.usage.to_json());
        push_some(
            &mut fields,
            "prompt_token_ids",
            self.prompt_token_ids.as_deref().map(int_list),
        );
        members(fields)
    }
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
#[derive(Clone, Debug, PartialEq)]
pub struct ChatCompletionChunk {
    pub id: String,
    pub object: String,
    pub created: i64,
    pub model: String,
    pub choices: Vec<ChunkChoice>,
    pub usage: Option<Usage>,
}

/// A choice of a chunk; `logprobs` and `finish_reason` are explicit nulls when absent.
#[derive(Clone, Debug, PartialEq)]
pub struct ChunkChoice {
    pub index: i32,
    pub delta: Delta,
    pub logprobs: Option<JsonValue>,
    pub finish_reason: Option<String>,
}

/// The increment of a chunk; absent parts are left out. `reasoning_content` precedes any `content`.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Delta {
    pub role: Option<String>,
    pub content: Option<String>,
    pub reasoning_content: Option<String>,
    pub tool_calls: Option<Vec<ToolCallDelta>>,
}

/// Streaming form of [`ToolCall`]; `index` identifies the call across chunks.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolCallDelta {
    pub index: i32,
    pub id: String,
    pub kind: String,
    pub function: ToolCallFunction,
}

impl ToJson for ToolCallDelta {
    fn to_json(&self) -> JsonValue {
        object([
            ("index", int(self.index)),
            ("id", text(&self.id)),
            ("type", text(&self.kind)),
            ("function", self.function.to_json()),
        ])
    }
}

impl ToJson for Delta {
    fn to_json(&self) -> JsonValue {
        let mut fields = Vec::new();
        push_some(&mut fields, "role", self.role.as_deref().map(text));
        push_some(&mut fields, "content", self.content.as_deref().map(text));
        push_some(
            &mut fields,
            "reasoning_content",
            self.reasoning_content.as_deref().map(text),
        );
        push_some(&mut fields, "tool_calls", self.tool_calls.as_deref().map(list));
        members(fields)
    }
}

impl ToJson for ChunkChoice {
    fn to_json(&self) -> JsonValue {
        object([
            ("index", int(self.index)),
            ("delta", self.delta.to_json()),
            ("logprobs", self.logprobs.clone().unwrap_or(JsonValue::Null)),
            ("finish_reason", optional_text(self.finish_reason.as_deref())),
        ])
    }
}

impl ToJson for ChatCompletionChunk {
    fn to_json(&self) -> JsonValue {
        let mut fields = Vec::new();
        push(&mut fields, "id", text(&self.id));
        push(&mut fields, "object", text(&self.object));
        push(&mut fields, "created", int(self.created));
        push(&mut fields, "model", text(&self.model));
        push(&mut fields, "choices", list(&self.choices));
        push_some(&mut fields, "usage", self.usage.as_ref().map(ToJson::to_json));
        members(fields)
    }
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
#[derive(Clone, Debug, PartialEq)]
pub struct ModelList {
    pub object: String,
    pub data: Vec<Model>,
}

/// One served model.
#[derive(Clone, Debug, PartialEq)]
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

impl ToJson for Model {
    fn to_json(&self) -> JsonValue {
        object([
            ("id", text(&self.id)),
            ("object", text(&self.object)),
            ("created", int(self.created)),
            ("owned_by", text(&self.owned_by)),
        ])
    }
}

impl ToJson for ModelList {
    fn to_json(&self) -> JsonValue {
        object([("object", text(&self.object)), ("data", list(&self.data))])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn json(text: &str) -> JsonValue {
        JsonValue::parse(text).expect("valid json")
    }

    #[test]
    fn a_request_with_every_known_field_is_read() {
        let request = ChatCompletionRequest::parse(
            r#"{"model":"m","messages":[{"role":"system","content":"s"},{"role":"user","content":[{"type":"text","text":"hi"}],"name":"bob"},
            {"role":"assistant","content":null,"tool_calls":[{"id":"c","type":"function","function":{"name":"f","arguments":"{}"}}],"reasoning_content":"r"},
            {"role":"tool","tool_call_id":"c","content":"x","extra":null}, null],
            "stream":true,"stream_options":{"include_usage":true,"extra":1},"max_tokens":5,"max_completion_tokens":6,"temperature":0.5,
            "top_p":0.9,"top_k":4,"seed":123456789012,"stop":["a","b"],"tools":[{"type":"function","function":{"name":"f","parameters":{"z":1,"a":2}}}],
            "tool_choice":"auto","parallel_tool_calls":false,"reasoning_effort":"low","chat_template_kwargs":{"enable_thinking":false},
            "response_format":{"type":"json_object"},"return_token_ids":true,"n":2,"user":"u","metadata":null}"#,
        )
        .expect("reads");
        assert_eq!(request.model.as_deref(), Some("m"));
        let messages = request.messages.expect("messages");
        assert_eq!(messages.len(), 5);
        assert!(messages[4].is_none());
        let user = messages[1].as_ref().expect("user");
        assert_eq!(user.name.as_deref(), Some("bob"));
        assert!(messages[2].as_ref().expect("assistant").content.is_none());
        assert!(messages[2].as_ref().expect("assistant").tool_calls.is_some());
        let tool = messages[3].as_ref().expect("tool");
        assert_eq!(tool.tool_call_id, Some(json("\"c\"")));
        assert_eq!(tool.other, vec![("extra".to_owned(), JsonValue::Null)]);
        let options = request.stream_options.expect("options");
        assert_eq!(options.include_usage, Some(true));
        assert_eq!(options.other, vec![("extra".to_owned(), json("1"))]);
        assert_eq!(request.stream, Some(true));
        assert_eq!(
            (request.max_tokens, request.max_completion_tokens, request.top_k),
            (Some(5), Some(6), Some(4))
        );
        assert_eq!(
            (request.temperature, request.top_p, request.seed),
            (Some(0.5), Some(0.9), Some(123456789012))
        );
        assert_eq!(request.stop.expect("stop").to_compact_string(), r#"["a","b"]"#);
        assert_eq!(
            request.tools.expect("tools").to_compact_string(),
            r#"[{"type":"function","function":{"name":"f","parameters":{"z":1,"a":2}}}]"#
        );
        assert_eq!(request.parallel_tool_calls, Some(false));
        assert_eq!(request.return_token_ids, Some(true));
        let names: Vec<&str> = request.other.iter().map(|(name, _)| name.as_str()).collect();
        assert_eq!(names, ["n", "user", "metadata"]);
    }

    #[test]
    fn an_empty_object_and_explicit_nulls_are_the_same() {
        let empty = ChatCompletionRequest::parse("{}").expect("empty");
        let nulls =
            ChatCompletionRequest::parse(r#"{"model":null,"messages":null,"stream":null,"stop":null,"tools":null}"#)
                .expect("nulls");
        assert_eq!(empty, nulls);
        assert!(empty.other.is_empty());
    }

    #[test]
    fn scalars_are_coerced_and_the_last_duplicate_wins() {
        let request = ChatCompletionRequest::parse(
            r#"{"model":123,"max_tokens":"7","temperature":" 0.7 ","top_p":1e400,"stream":"TRUE","top_k":5.9,"seed":"","stream":0,"n":1,"n":2}"#,
        )
        .expect("reads");
        assert_eq!(request.model.as_deref(), Some("123"));
        assert_eq!(request.max_tokens, Some(7));
        assert_eq!(request.temperature, Some(0.7));
        assert_eq!(request.top_p, Some(f64::INFINITY));
        assert_eq!(request.top_k, Some(5));
        assert_eq!(request.seed, None);
        assert_eq!(request.stream, Some(false));
        assert_eq!(request.other, vec![("n".to_owned(), json("2"))]);
    }

    #[test]
    fn a_body_of_the_wrong_shape_is_unparsable() {
        for body in [
            r#"{"stream":"yes"}"#,
            r#"{"messages":"hi"}"#,
            r#"{"messages":{"role":"user"}}"#,
            r#"{"messages":["hi"]}"#,
            r#"{"stream_options":[]}"#,
            r#"{"max_tokens":2147483648}"#,
            r#"{"model":[]}"#,
            "[]",
            "null",
            "",
            r#"{"model":"m"} x"#,
            r#"{"model":"m""#,
        ] {
            let error = ChatCompletionRequest::parse(body).expect_err(body);
            assert_eq!(error, ApiError::unparsable_body(), "{body}");
        }
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
            ChatCompletionResponse::of(completion()).to_json_string(),
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
            ChatCompletionResponse::of(parts).to_json_string(),
            r#"{"id":"chatcmpl-1","object":"chat.completion","created":7,"model":"m","choices":[{"index":0,"message":{"role":"assistant","content":null,"reasoning_content":"think","refusal":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"f","arguments":"{\"a\":1}"}}]},"logprobs":null,"finish_reason":"tool_calls","token_ids":[1,2]}],"usage":{"prompt_tokens":10,"completion_tokens":3,"total_tokens":13,"prompt_tokens_details":{"cached_tokens":0},"completion_tokens_details":{"reasoning_tokens":2}},"prompt_token_ids":[5,6]}"#
        );
    }

    #[test]
    fn every_chunk_kind_has_its_payload() {
        let (id, model) = ("c1", "m");
        let head = r#"{"id":"c1","object":"chat.completion.chunk","created":7,"model":"m","choices":["#;
        let cases = [
            (
                ChatCompletionChunk::role(id, 7, model),
                r#"{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}"#,
            ),
            (
                ChatCompletionChunk::reasoning(id, 7, model, "r"),
                r#"{"index":0,"delta":{"reasoning_content":"r"},"logprobs":null,"finish_reason":null}]}"#,
            ),
            (
                ChatCompletionChunk::content(id, 7, model, "t"),
                r#"{"index":0,"delta":{"content":"t"},"logprobs":null,"finish_reason":null}]}"#,
            ),
            (
                ChatCompletionChunk::tool_call(id, 7, model, 1, ToolCall::function("call_1", "f", "{}")),
                r#"{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_1","type":"function","function":{"name":"f","arguments":"{}"}}]},"logprobs":null,"finish_reason":null}]}"#,
            ),
            (
                ChatCompletionChunk::finish(id, 7, model, "stop"),
                r#"{"index":0,"delta":{},"logprobs":null,"finish_reason":"stop"}]}"#,
            ),
            (
                ChatCompletionChunk::usage(id, 7, model, Usage::of(2, 1, 3, Some(1))),
                r#"],"usage":{"prompt_tokens":2,"completion_tokens":3,"total_tokens":5,"prompt_tokens_details":{"cached_tokens":1},"completion_tokens_details":{"reasoning_tokens":1}}}"#,
            ),
        ];
        for (chunk, tail) in cases {
            assert_eq!(chunk.to_json_string(), format!("{head}{tail}"));
        }
    }

    #[test]
    fn control_characters_in_a_text_delta_are_written_with_capital_hex() {
        assert_eq!(
            ChatCompletionChunk::content("c1", 7, "m", "a\u{1b}b\u{b}c\u{1f}d\"\\\n").to_json_string(),
            r#"{"id":"c1","object":"chat.completion.chunk","created":7,"model":"m","choices":[{"index":0,"delta":{"content":"a\u001Bb\u000Bc\u001Fd\"\\\n"},"logprobs":null,"finish_reason":null}]}"#
        );
    }

    #[test]
    fn the_model_list_and_model_have_the_listing_shape() {
        let model = Model::of("m", 9);
        assert_eq!(
            model.to_json_string(),
            r#"{"id":"m","object":"model","created":9,"owned_by":"euhedral"}"#
        );
        assert_eq!(
            ModelList::of(model).to_json_string(),
            r#"{"object":"list","data":[{"id":"m","object":"model","created":9,"owned_by":"euhedral"}]}"#
        );
    }
}
