//! Chat Completions request to [`GenerationRequest`]: every validation of the request, in the order a client sees
//! them.
//!
//! Fields fall into four classes: supported; metadata that cannot change generation and is ignored; behavioural
//! features accepted only at their neutral value; and everything else, which is rejected.

use std::collections::HashSet;
use std::slice;

use crate::error::ApiError;
use crate::generation::{Defaults, GenerationRequest, GenerationSettings, Sampling};
use crate::json::JsonValue;
use crate::openai::{ChatCompletionRequest, ChatMessage};
use crate::reasoning;
use crate::response_format::ResponseFormat;
use crate::served_model::require_served_model;
use crate::template::{AssistantMessage, ContentPart, Message, Role, ToolCall};
use crate::tool_calling::{ToolCalling, check_keys, is_function_name, is_parameter_name, member};
use crate::tool_results::ToolResults;

/// The most stop sequences a Chat Completions request may carry.
const MAX_STOP_SEQUENCES: usize = 4;

/// A mapped Chat Completions request: the generation to run, and the one thing only this surface asks of the
/// reply.
#[derive(Clone, Debug, PartialEq)]
pub struct ChatRequest {
    pub generation: GenerationRequest,
    /// The streamed reply ends with a usage chunk.
    pub include_usage: bool,
}

/// Accepted and ignored: these never influence the generated tokens.
const IGNORED_METADATA: [&str; 7] = [
    "user",
    "metadata",
    "store",
    "service_tier",
    "safety_identifier",
    "prompt_cache_key",
    "prompt_cache_retention",
];

/// Validates everything the request states and maps it. The first failing check decides the error.
pub fn map_chat(mut request: ChatCompletionRequest, defaults: &Defaults) -> Result<ChatRequest, ApiError> {
    check_other_fields(&request.other, top_level_neutral, "")?;
    let stream = request.stream == Some(true);
    let include_usage = stream_options_include_usage(&request, stream)?;
    let return_token_ids = request.return_token_ids == Some(true);
    if return_token_ids && stream {
        return Err(ApiError::invalid_request(
            "'return_token_ids' is available only without streaming.",
            Some("return_token_ids"),
        ));
    }
    let model = require_served_model(
        request.model.as_deref(),
        &defaults.model,
        "You must provide a model parameter.",
    )?;
    let tools = ToolCalling::from_request(
        request.tools.as_ref(),
        request.tool_choice.as_ref(),
        request.parallel_tool_calls,
    )?;
    let thinking = reasoning::thinking(request.reasoning_effort.as_ref(), request.chat_template_kwargs.as_ref())?;
    let format = ResponseFormat::from_request(request.response_format.as_ref())?;
    if format.is_json() && request.stop.is_some() {
        return Err(ApiError::invalid_request(
            "'stop' cannot be combined with a JSON 'response_format': a stop inside the JSON would end it invalid.",
            Some("stop"),
        ));
    }
    let messages = messages(request.messages.take().unwrap_or_default())?;
    let stops = stops(request.stop.as_ref())?;
    let sampling = sampling(&request, defaults)?;
    let max_tokens = max_tokens(&request)?;
    let max_tokens_param = if request.max_tokens.is_some() {
        "max_tokens"
    } else {
        "max_completion_tokens"
    };
    let settings = GenerationSettings {
        model,
        thinking,
        reasoning_budget: None,
        format,
        stops,
        sampling,
        max_tokens,
        max_tokens_param: max_tokens_param.to_owned(),
        stream,
        return_token_ids,
    };
    Ok(ChatRequest {
        generation: GenerationRequest::new(messages, tools, settings),
        include_usage,
    })
}

fn stream_options_include_usage(request: &ChatCompletionRequest, stream: bool) -> Result<bool, ApiError> {
    let Some(options) = &request.stream_options else {
        return Ok(false);
    };
    if !stream {
        return Err(ApiError::invalid_request(
            "The 'stream_options' parameter is only allowed when 'stream' is enabled.",
            Some("stream_options"),
        ));
    }
    for (key, _) in &options.other {
        // Obfuscation padding only guards against network side channels; omitting it changes no content.
        if key == "include_obfuscation" {
            continue;
        }
        return Err(ApiError::unrecognized_argument(&format!("stream_options.{key}")));
    }
    Ok(options.include_usage == Some(true))
}

fn messages(messages: Vec<Option<ChatMessage>>) -> Result<Vec<Message>, ApiError> {
    if messages.is_empty() {
        return Err(ApiError::invalid_request(
            "'messages' must contain at least one message.",
            Some("messages"),
        ));
    }
    let mut turns = Vec::with_capacity(messages.len());
    let mut pending: Option<ToolResults> = None;
    for (index, message) in messages.into_iter().enumerate() {
        let path = format!("messages[{index}]");
        let Some(message) = message else {
            return Err(ApiError::invalid_request("Message must be an object.", Some(&path)));
        };
        check_other_fields(&message.other, message_neutral, &format!("{path}."))?;
        let role = role(message.role.as_deref(), &path)?;
        if role != Role::Assistant && message.tool_calls.is_some() {
            return Err(ApiError::invalid_request(
                "Only assistant messages may contain 'tool_calls'.",
                Some(&format!("{path}.tool_calls")),
            ));
        }
        if role != Role::Tool && message.tool_call_id.is_some() {
            return Err(ApiError::invalid_request(
                "Only tool messages may contain 'tool_call_id'.",
                Some(&format!("{path}.tool_call_id")),
            ));
        }
        if role != Role::Assistant && message.reasoning_content.is_some() {
            return Err(ApiError::invalid_request(
                "Only assistant messages may contain 'reasoning_content'.",
                Some(&format!("{path}.reasoning_content")),
            ));
        }
        if message
            .reasoning_content
            .as_ref()
            .is_some_and(|value| !matches!(value, JsonValue::String(_)))
        {
            return Err(ApiError::invalid_request(
                "'reasoning_content' must be a string.",
                Some(&format!("{path}.reasoning_content")),
            ));
        }
        let content = content(message.content.as_ref(), role, &path)?;
        if role == Role::Tool {
            let Some(results) = pending.as_mut() else {
                return Err(ApiError::invalid_request(
                    "Messages with role 'tool' must be a response to a preceding message with 'tool_calls'.",
                    Some(&format!("{path}.role")),
                ));
            };
            let id = match &message.tool_call_id {
                Some(JsonValue::String(id)) => Some(id.as_str()),
                _ => None,
            };
            results.answer(id, content, &format!("{path}.tool_call_id"))?;
            continue;
        }
        if let Some(results) = &pending {
            results.close_into(&mut turns)?;
        }
        let mut ids = Vec::new();
        let tool_calls = tool_calls(message.tool_calls.as_ref(), &format!("{path}.tool_calls"), &mut ids)?;
        let has_calls = !tool_calls.is_empty();
        let turn = match role {
            Role::System => Message::System(vec![ContentPart::Text(content)]),
            Role::User => Message::User(vec![ContentPart::Text(content)]),
            _ => {
                let reasoning = match message.reasoning_content {
                    Some(JsonValue::String(text)) => text,
                    _ => String::new(),
                };
                Message::Assistant(AssistantMessage {
                    content: vec![ContentPart::Text(content)],
                    tool_calls,
                    reasoning,
                })
            }
        };
        turns.push(turn);
        pending = has_calls.then(|| ToolResults::new(ids, format!("{path}.tool_calls"), "tool_call_id"));
    }
    if let Some(results) = &pending {
        results.close_into(&mut turns)?;
    }
    Ok(turns)
}

/// Parses OpenAI assistant `tool_calls`, whose `arguments` are JSON-object strings, adding each call's id to `ids`.
fn tool_calls(value: Option<&JsonValue>, path: &str, ids: &mut Vec<String>) -> Result<Vec<ToolCall>, ApiError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let JsonValue::Array(calls) = value else {
        return Err(ApiError::invalid_request("'tool_calls' must be an array.", Some(path)));
    };
    let mut parsed = Vec::with_capacity(calls.len());
    let mut seen = HashSet::new();
    for (index, call) in calls.iter().enumerate() {
        let call_path = format!("{path}[{index}]");
        if !matches!(call, JsonValue::Object(_)) {
            return Err(ApiError::invalid_request(
                "Each tool call must be an object.",
                Some(&call_path),
            ));
        }
        check_keys(call, &["id", "type", "function"], &call_path)?;
        let id = match member(call, "id") {
            Some(JsonValue::String(id)) if !id.is_empty() => id,
            _ => {
                return Err(ApiError::invalid_request(
                    "Tool call 'id' is required.",
                    Some(&format!("{call_path}.id")),
                ));
            }
        };
        if !seen.insert(id.as_str()) {
            return Err(ApiError::invalid_request(
                "Tool call IDs must be unique within a message.",
                Some(&format!("{call_path}.id")),
            ));
        }
        ids.push(id.clone());
        let Some(JsonValue::String(kind)) = member(call, "type") else {
            return Err(ApiError::invalid_request(
                "Tool call 'type' is required.",
                Some(&format!("{call_path}.type")),
            ));
        };
        if kind != "function" {
            return Err(ApiError::unsupported_parameter(&format!("{call_path}.type")));
        }
        let Some(function @ JsonValue::Object(_)) = call.get("function") else {
            return Err(ApiError::invalid_request(
                "Tool call 'function' is required.",
                Some(&format!("{call_path}.function")),
            ));
        };
        let function_path = format!("{call_path}.function");
        check_keys(function, &["name", "arguments"], &function_path)?;
        let name = match member(function, "name") {
            Some(JsonValue::String(name)) if is_function_name(name) => name.clone(),
            _ => {
                return Err(ApiError::invalid_request(
                    "Function names must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.",
                    Some(&format!("{function_path}.name")),
                ));
            }
        };
        let arguments = arguments(member(function, "arguments"), &function_path)?;
        parsed.push(ToolCall { name, arguments });
    }
    Ok(parsed)
}

/// The template iterates the parsed object; it renders nothing for the empty string, as for `{}`.
fn arguments(value: Option<&JsonValue>, function_path: &str) -> Result<Vec<(String, JsonValue)>, ApiError> {
    let path = format!("{function_path}.arguments");
    let Some(JsonValue::String(text)) = value else {
        return Err(ApiError::invalid_request(
            "Tool call 'arguments' must be a JSON string.",
            Some(&path),
        ));
    };
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let Some(parsed) = JsonValue::parse(text) else {
        return Err(ApiError::invalid_request(
            "Tool call 'arguments' must be valid JSON.",
            Some(&path),
        ));
    };
    let JsonValue::Object(members) = parsed else {
        return Err(ApiError::invalid_request(
            "Tool call 'arguments' must encode a JSON object.",
            Some(&path),
        ));
    };
    if members.iter().any(|(key, _)| !is_parameter_name(key)) {
        return Err(ApiError::invalid_request(
            "Argument names must be non-empty and free of '<', '>' and line breaks.",
            Some(&path),
        ));
    }
    Ok(members)
}

fn role(role: Option<&str>, path: &str) -> Result<Role, ApiError> {
    let Some(role) = role else {
        return Err(ApiError::invalid_request(
            "Message role is required.",
            Some(&format!("{path}.role")),
        ));
    };
    match role {
        // OpenAI's newer name for instructions formerly sent as system messages.
        "system" | "developer" => Ok(Role::System),
        "user" => Ok(Role::User),
        "assistant" => Ok(Role::Assistant),
        "tool" => Ok(Role::Tool),
        // Legacy function calling predates tool_call ids and is not implemented.
        "function" => Err(ApiError::unsupported_parameter(&format!("{path}.role"))),
        other => Err(ApiError::invalid_request(
            format!("Invalid message role '{other}'."),
            Some(&format!("{path}.role")),
        )),
    }
}

/// Mirrors the template's `render_content` for text: strings pass through, text parts concatenate.
fn content(content: Option<&JsonValue>, role: Role, path: &str) -> Result<String, ApiError> {
    let content_path = format!("{path}.content");
    let content = match content {
        None => {
            if role == Role::Assistant {
                return Ok(String::new());
            }
            return Err(ApiError::invalid_request(
                "Message content is required.",
                Some(&content_path),
            ));
        }
        Some(JsonValue::String(text)) => return Ok(text.clone()),
        Some(JsonValue::Array(parts)) => parts,
        Some(_) => {
            return Err(ApiError::invalid_request(
                "Message content must be a string or an array of content parts.",
                Some(&content_path),
            ));
        }
    };
    let mut text = String::new();
    for (index, part) in content.iter().enumerate() {
        let part_path = format!("{content_path}[{index}]");
        let JsonValue::Object(members) = part else {
            return Err(ApiError::invalid_request(
                "Content part must be an object.",
                Some(&part_path),
            ));
        };
        match member(part, "type") {
            Some(JsonValue::String(kind)) if kind == "text" => {}
            Some(JsonValue::String(kind)) => {
                return Err(ApiError::unsupported_parameter(&format!("{part_path}.type={kind}")));
            }
            _ => {
                return Err(ApiError::invalid_request(
                    "Content part type is required.",
                    Some(&format!("{part_path}.type")),
                ));
            }
        }
        let Some(JsonValue::String(part_text)) = member(part, "text") else {
            return Err(ApiError::invalid_request(
                "Text content part requires 'text'.",
                Some(&format!("{part_path}.text")),
            ));
        };
        if let Some((key, _)) = members.iter().find(|(key, _)| key != "type" && key != "text") {
            return Err(ApiError::unrecognized_argument(&format!("{part_path}.{key}")));
        }
        text.push_str(part_text);
    }
    Ok(text)
}

/// The requested completion budget, or `None` for the server's default; the planner checks it against the context.
fn max_tokens(request: &ChatCompletionRequest) -> Result<Option<i32>, ApiError> {
    let mut requested = request.max_completion_tokens;
    if let Some(max_tokens) = request.max_tokens {
        if requested.is_some_and(|other| other != max_tokens) {
            return Err(ApiError::invalid_request(
                "'max_tokens' and 'max_completion_tokens' conflict; send only one.",
                Some("max_tokens"),
            ));
        }
        requested = Some(max_tokens);
    }
    Ok(requested)
}

/// Omitted fields take the defaults; temperature or top_p of 0 is greedy.
fn sampling(request: &ChatCompletionRequest, defaults: &Defaults) -> Result<Sampling, ApiError> {
    let mut temperature = defaults.temperature;
    if let Some(value) = request.temperature {
        if !(0.0..=2.0).contains(&value) {
            return Err(ApiError::invalid_request(
                "'temperature' must be between 0 and 2.",
                Some("temperature"),
            ));
        }
        temperature = value as f32;
    }
    let mut top_p = defaults.top_p;
    if let Some(value) = request.top_p {
        if !(0.0..=1.0).contains(&value) {
            return Err(ApiError::invalid_request(
                "'top_p' must be between 0 and 1.",
                Some("top_p"),
            ));
        }
        top_p = value as f32;
    }
    let greedy = temperature == 0.0 || top_p == 0.0;
    let mut top_k = defaults.top_k;
    if let Some(value) = request.top_k {
        // As vLLM spells it: 0 or -1 turns top-k off.
        if value < -1 {
            return Err(ApiError::invalid_request(
                "'top_k' must be at least 1, or 0 or -1 for no limit.",
                Some("top_k"),
            ));
        }
        top_k = value.max(0);
    }
    Ok(Sampling {
        temperature,
        top_k,
        top_p: if greedy { 1.0 } else { top_p },
        seed: request.seed,
        greedy,
    })
}

fn stops(stop: Option<&JsonValue>) -> Result<Vec<String>, ApiError> {
    let Some(stop) = stop else {
        return Ok(Vec::new());
    };
    let values = match stop {
        JsonValue::Array(values) => values.as_slice(),
        single @ JsonValue::String(_) => slice::from_ref(single),
        _ => {
            return Err(ApiError::invalid_request(
                "'stop' must be a string or an array of strings.",
                Some("stop"),
            ));
        }
    };
    if values.len() > MAX_STOP_SEQUENCES {
        return Err(ApiError::invalid_request(
            format!("'stop' accepts at most {MAX_STOP_SEQUENCES} sequences."),
            Some("stop"),
        ));
    }
    values
        .iter()
        .map(|value| match value {
            JsonValue::String(text) if !text.is_empty() => Ok(text.clone()),
            _ => Err(ApiError::invalid_request(
                "Each stop sequence must be a non-empty string.",
                Some("stop"),
            )),
        })
        .collect()
}

fn check_other_fields(
    fields: &[(String, JsonValue)],
    neutral: fn(&str, &JsonValue) -> Option<bool>,
    prefix: &str,
) -> Result<(), ApiError> {
    for (name, value) in fields {
        if prefix.is_empty() && IGNORED_METADATA.contains(&name.as_str()) {
            continue;
        }
        let Some(is_neutral) = neutral(name, value) else {
            return Err(ApiError::unrecognized_argument(&format!("{prefix}{name}")));
        };
        if *value != JsonValue::Null && !is_neutral {
            return Err(ApiError::unsupported_parameter(&format!("{prefix}{name}")));
        }
    }
    Ok(())
}

/// Behavioural features that are not implemented are accepted only when the value requests no behaviour; `None`
/// for a field that is not one of them.
fn top_level_neutral(name: &str, value: &JsonValue) -> Option<bool> {
    Some(match name {
        "n" | "repeat_penalty" | "repetition_penalty" => number(value) == Some(1.0),
        "logprobs" => *value == JsonValue::Bool(false),
        "top_logprobs" | "frequency_penalty" | "presence_penalty" | "min_p" => number(value) == Some(0.0),
        "logit_bias" => matches!(value, JsonValue::Object(members) if members.is_empty()),
        "functions" => matches!(value, JsonValue::Array(items) if items.is_empty()),
        "function_call" => matches!(value, JsonValue::String(text) if text == "none"),
        "modalities" => matches!(value, JsonValue::Array(items)
            if matches!(items.as_slice(), [JsonValue::String(text)] if text == "text")),
        "audio" | "prediction" | "web_search_options" | "verbosity" => false,
        _ => return None,
    })
}

fn message_neutral(name: &str, _value: &JsonValue) -> Option<bool> {
    matches!(name, "refusal" | "function_call" | "audio").then_some(false)
}

/// The value as a double when it is a JSON number.
fn number(value: &JsonValue) -> Option<f64> {
    match value {
        JsonValue::Integer(digits) => digits.as_str().parse().ok(),
        JsonValue::Float(number) => Some(*number),
        _ => None,
    }
}

#[cfg(test)]
mod tests;
