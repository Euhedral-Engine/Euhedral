//! Responses request to [`GenerationRequest`]: every validation of the request, in the order a client sees them.
//!
//! The server keeps no state between requests: a conversation is sent whole as `input`, `previous_response_id` and
//! `conversation` are refused, and `store` has no effect. Reasoning the server returned comes back as `reasoning`
//! items, by their `content` or, with `include: ["reasoning.encrypted_content"]`, their opaque `encrypted_content`.

use std::collections::HashSet;

use crate::error::ApiError;
use crate::generation::{Defaults, GenerationRequest, GenerationSettings, Sampling};
use crate::json::JsonValue;
use crate::pipeline::ToolChoice;
use crate::reasoning;
use crate::response_format::ResponseFormat;
use crate::responses::ResponsesRequest;
use crate::served_model::require_served_model;
use crate::template::{AssistantMessage, ContentPart, Effort, Message, TemplateOptions, ToolCall};
use crate::tool_calling::{FunctionTool, ToolCalling, check_keys, is_function_name, is_parameter_name, parameters};
use crate::tool_definition::ToolDefinition;
use crate::tool_results::ToolResults;

/// The prefix of the opaque reasoning token: its UTF-8 text in URL-safe base64, not encrypted.
pub const ENCRYPTED_PREFIX: &str = "euhedral:";

/// Accepted and ignored: these never influence the generated tokens.
const IGNORED: [&str; 7] = [
    "user",
    "metadata",
    "store",
    "service_tier",
    "safety_identifier",
    "prompt_cache_key",
    "truncation",
];

/// Accepted only as null (and `background` as false).
const REFUSED: [&str; 6] = [
    "previous_response_id",
    "conversation",
    "background",
    "prompt",
    "max_tool_calls",
    "top_logprobs",
];

/// The request parameter that carries the completion budget.
const MAX_TOKENS_PARAM: &str = "max_output_tokens";

const FUNCTION_NAME_MESSAGE: &str = "Function names must be 1-64 characters of a-z, A-Z, 0-9, underscores, and dashes.";

/// A mapped Responses request: the generation to run, and what only this surface asks of the reply.
#[derive(Clone, Debug, PartialEq)]
pub struct MappedResponses {
    pub generation: GenerationRequest,
    /// The request asked for `reasoning.encrypted_content`: reasoning items carry the opaque token.
    pub include_encrypted_content: bool,
    /// The settings every response repeats, in the order they are written.
    pub echo: JsonValue,
}

/// Validates everything the request states and maps it. The first failing check decides the error.
pub fn map_responses(request: ResponsesRequest, defaults: &Defaults) -> Result<MappedResponses, ApiError> {
    check_other_fields(&request.other)?;
    let model = require_served_model(
        request.model.as_deref(),
        &defaults.model,
        "You must provide a model parameter.",
    )?;
    let include_encrypted_content = include(present(request.include.as_ref()))?;
    let tools = tools(
        present(request.tools.as_ref()),
        present(request.tool_choice.as_ref()),
        request.parallel_tool_calls,
    )?;
    let effort = reasoning_options(present(request.reasoning.as_ref()))?;
    let thinking = reasoning::thinking(effort, None)?;
    let format = format(present(request.text.as_ref()))?;
    let messages = turns(request.instructions.as_deref(), present(request.input.as_ref()))?;
    let sampling = sampling(&request, defaults)?;
    let echo = echo(&request, &tools, &thinking);
    let settings = GenerationSettings {
        model,
        thinking,
        reasoning_budget: None,
        format,
        stops: Vec::new(),
        sampling,
        max_tokens: request.max_output_tokens,
        max_tokens_param: MAX_TOKENS_PARAM.to_owned(),
        stream: request.stream == Some(true),
        return_token_ids: false,
    };
    Ok(MappedResponses {
        generation: GenerationRequest::new(messages, tools, settings),
        include_encrypted_content,
        echo,
    })
}

/// The request's settings as a response repeats them; absent ones take their defaults.
fn echo(request: &ResponsesRequest, tools: &ToolCalling, thinking: &TemplateOptions) -> JsonValue {
    let raw = |value: Option<&JsonValue>, default: JsonValue| present(value).cloned().unwrap_or(default);
    let text = |value: &str| JsonValue::String(value.to_owned());
    let effort = match thinking.effort {
        None => "none",
        Some(Effort::Low) => "low",
        Some(Effort::Medium) => "medium",
        Some(Effort::XHigh) => "xhigh",
    };
    let metadata = request
        .other
        .iter()
        .find(|(name, _)| name == "metadata")
        .map_or_else(|| JsonValue::Object(Vec::new()), |(_, value)| value.clone());
    let default_text = JsonValue::Object(vec![(
        "format".to_owned(),
        JsonValue::Object(vec![("type".to_owned(), text("text"))]),
    )]);
    JsonValue::Object(vec![
        (
            "instructions".to_owned(),
            request.instructions.as_deref().map_or(JsonValue::Null, text),
        ),
        (
            "max_output_tokens".to_owned(),
            request
                .max_output_tokens
                .map_or(JsonValue::Null, |tokens| JsonValue::Integer(i128::from(tokens).into())),
        ),
        ("parallel_tool_calls".to_owned(), JsonValue::Bool(tools.parallel)),
        (
            "temperature".to_owned(),
            request.temperature.map_or(JsonValue::Null, JsonValue::Float),
        ),
        (
            "top_p".to_owned(),
            request.top_p.map_or(JsonValue::Null, JsonValue::Float),
        ),
        (
            "tool_choice".to_owned(),
            raw(request.tool_choice.as_ref(), text("auto")),
        ),
        (
            "tools".to_owned(),
            raw(request.tools.as_ref(), JsonValue::Array(Vec::new())),
        ),
        ("text".to_owned(), raw(request.text.as_ref(), default_text)),
        (
            "reasoning".to_owned(),
            JsonValue::Object(vec![
                ("effort".to_owned(), text(effort)),
                ("summary".to_owned(), JsonValue::Null),
            ]),
        ),
        ("store".to_owned(), JsonValue::Bool(false)),
        ("metadata".to_owned(), metadata),
    ])
}

/// A JSON null is the same as an absent field.
pub(crate) fn present(value: Option<&JsonValue>) -> Option<&JsonValue> {
    value.filter(|value| **value != JsonValue::Null)
}

fn check_other_fields(fields: &[(String, JsonValue)]) -> Result<(), ApiError> {
    for (name, value) in fields {
        let name = name.as_str();
        if name == "truncation" && *value != JsonValue::Null && *value != JsonValue::String("disabled".to_owned()) {
            return Err(ApiError::unsupported_parameter("truncation"));
        }
        if name == "background" && *value == JsonValue::Bool(false) {
            continue;
        }
        if IGNORED.contains(&name) {
            continue;
        }
        if REFUSED.contains(&name) {
            if *value == JsonValue::Null {
                continue;
            }
            return Err(ApiError::unsupported_parameter(name));
        }
        return Err(ApiError::unrecognized_argument(name));
    }
    Ok(())
}

// Input.

/// The assistant turn being collected: reasoning, text and calls, in that order in the prompt.
struct AssistantTurn {
    path: String,
    reasoning: String,
    text: String,
    calls: Vec<ToolCall>,
    ids: Vec<String>,
}

impl AssistantTurn {
    fn new(path: &str) -> AssistantTurn {
        AssistantTurn {
            path: path.to_owned(),
            reasoning: String::new(),
            text: String::new(),
            calls: Vec::new(),
            ids: Vec::new(),
        }
    }

    /// Adds the turn; returns the results its calls are owed, if it has calls.
    fn close_into(self, turns: &mut Vec<Message>) -> Option<ToolResults> {
        let owed = (!self.calls.is_empty()).then(|| ToolResults::new(self.ids, self.path, "call_id"));
        turns.push(Message::Assistant(AssistantMessage {
            content: vec![ContentPart::Text(self.text)],
            tool_calls: self.calls,
            reasoning: self.reasoning,
        }));
        owed
    }
}

fn turns(instructions: Option<&str>, input: Option<&JsonValue>) -> Result<Vec<Message>, ApiError> {
    let Some(input) = input else {
        return Err(ApiError::invalid_request("'input' is required.", Some("input")));
    };
    let single;
    let items: &[JsonValue] = match input {
        JsonValue::String(text) => {
            single = [JsonValue::Object(vec![
                ("type".to_owned(), JsonValue::String("message".to_owned())),
                ("role".to_owned(), JsonValue::String("user".to_owned())),
                ("content".to_owned(), JsonValue::String(text.clone())),
            ])];
            &single
        }
        JsonValue::Array(items) => items,
        _ => {
            return Err(ApiError::invalid_request(
                "'input' must be a string or a list of items.",
                Some("input"),
            ));
        }
    };
    if items.is_empty() {
        return Err(ApiError::invalid_request("'input' must not be empty.", Some("input")));
    }
    let mut turns = Vec::new();
    let mut system = instructions.unwrap_or_default().to_owned();
    let mut leading = true;
    let mut assistant: Option<AssistantTurn> = None;
    let mut pending: Option<ToolResults> = None;
    for (index, item) in items.iter().enumerate() {
        let path = format!("input[{index}]");
        if !matches!(item, JsonValue::Object(_)) {
            return Err(ApiError::invalid_request(
                "Each input item must be an object.",
                Some(&path),
            ));
        }
        let kind = match item.get("type") {
            Some(JsonValue::String(name)) => name.as_str(),
            _ if item.get("role").is_some() => "message",
            _ => {
                return Err(ApiError::invalid_request(
                    "Input item 'type' is required.",
                    Some(&format!("{path}.type")),
                ));
            }
        };
        let role = match item.get("role") {
            Some(JsonValue::String(name)) if kind == "message" => Some(name.as_str()),
            _ => None,
        };
        let assistant_item = kind == "reasoning" || kind == "function_call" || role == Some("assistant");
        if !assistant_item && let Some(turn) = assistant.take() {
            pending = turn.close_into(&mut turns);
        }
        if kind != "function_call_output"
            && !assistant_item
            && let Some(results) = pending.take()
        {
            results.close_into(&mut turns)?;
        }
        match kind {
            "message" => {
                check_keys(item, &["type", "role", "content", "id", "status"], &path)?;
                let Some(role) = role else {
                    return Err(ApiError::invalid_request(
                        "Message 'role' is required.",
                        Some(&format!("{path}.role")),
                    ));
                };
                let text = content(item.get("content"), &format!("{path}.content"))?;
                match role {
                    "system" | "developer" => {
                        if !leading {
                            return Err(ApiError::invalid_request(
                                "System and developer messages must come before the conversation.",
                                Some(&path),
                            ));
                        }
                        if !system.is_empty() {
                            system.push_str("\n\n");
                        }
                        system.push_str(&text);
                    }
                    "user" => turns.push(Message::user(text)),
                    "assistant" => {
                        if let Some(results) = pending.take() {
                            results.close_into(&mut turns)?;
                        }
                        assistant
                            .get_or_insert_with(|| AssistantTurn::new(&path))
                            .text
                            .push_str(&text);
                    }
                    _ => {
                        return Err(ApiError::invalid_request(
                            format!("Invalid message role '{role}'."),
                            Some(&format!("{path}.role")),
                        ));
                    }
                }
            }
            "reasoning" => {
                check_keys(
                    item,
                    &["type", "id", "summary", "content", "encrypted_content", "status"],
                    &path,
                )?;
                if let Some(results) = pending.take() {
                    results.close_into(&mut turns)?;
                }
                let turn = assistant.get_or_insert_with(|| AssistantTurn::new(&path));
                turn.reasoning.push_str(&reasoning_text(item, &path)?);
            }
            "function_call" => {
                check_keys(item, &["type", "id", "call_id", "name", "arguments", "status"], &path)?;
                if let Some(results) = pending.take() {
                    results.close_into(&mut turns)?;
                }
                let turn = assistant.get_or_insert_with(|| AssistantTurn::new(&path));
                let Some(JsonValue::String(id)) = item.get("call_id") else {
                    return Err(ApiError::invalid_request(
                        "'call_id' is required.",
                        Some(&format!("{path}.call_id")),
                    ));
                };
                if id.is_empty() {
                    return Err(ApiError::invalid_request(
                        "'call_id' is required.",
                        Some(&format!("{path}.call_id")),
                    ));
                }
                if turn.ids.contains(id) {
                    return Err(ApiError::invalid_request(
                        "Call IDs must be unique within a turn.",
                        Some(&format!("{path}.call_id")),
                    ));
                }
                let Some(JsonValue::String(name)) = item.get("name").filter(|name| match name {
                    JsonValue::String(name) => is_function_name(name),
                    _ => false,
                }) else {
                    return Err(ApiError::invalid_request(
                        FUNCTION_NAME_MESSAGE,
                        Some(&format!("{path}.name")),
                    ));
                };
                turn.ids.push(id.clone());
                turn.calls.push(ToolCall {
                    name: name.clone(),
                    arguments: arguments(item.get("arguments"), &path)?,
                });
            }
            "function_call_output" => {
                check_keys(item, &["type", "id", "call_id", "output", "status"], &path)?;
                let Some(results) = pending.as_mut() else {
                    return Err(ApiError::invalid_request(
                        "A function_call_output must answer a function_call of the preceding turn.",
                        Some(&path),
                    ));
                };
                let output = content(item.get("output"), &format!("{path}.output"))?;
                let id = match item.get("call_id") {
                    Some(JsonValue::String(id)) => Some(id.as_str()),
                    _ => None,
                };
                results.answer(id, output, &format!("{path}.call_id"))?;
            }
            other => {
                return Err(ApiError::unsupported_parameter(&format!("{path}.type={other}")));
            }
        }
        if kind != "message" || !matches!(role, Some("system" | "developer")) {
            leading = false;
        }
    }
    if let Some(turn) = assistant.take() {
        pending = turn.close_into(&mut turns);
    }
    if let Some(results) = pending.take() {
        results.close_into(&mut turns)?;
    }
    if !system.is_empty() {
        turns.insert(0, Message::system(system));
    }
    Ok(turns)
}

/// A message's or output's text: a string, or `input_text`/`output_text` parts concatenated.
fn content(content: Option<&JsonValue>, path: &str) -> Result<String, ApiError> {
    let parts = match content {
        None | Some(JsonValue::Null) => return Ok(String::new()),
        Some(JsonValue::String(text)) => return Ok(text.clone()),
        Some(JsonValue::Array(parts)) => parts,
        Some(_) => {
            return Err(ApiError::invalid_request(
                "Content must be a string or a list of content parts.",
                Some(path),
            ));
        }
    };
    let mut text = String::new();
    for (index, part) in parts.iter().enumerate() {
        let part_path = format!("{path}[{index}]");
        if !matches!(part, JsonValue::Object(_)) {
            return Err(ApiError::invalid_request(
                "Content part must be an object.",
                Some(&part_path),
            ));
        }
        let kind = part.get("type");
        if !matches!(kind, Some(JsonValue::String(name)) if name == "input_text" || name == "output_text") {
            return Err(ApiError::unsupported_parameter(&format!(
                "{part_path}.type={}",
                kind.map_or_else(|| "null".to_owned(), JsonValue::param_text)
            )));
        }
        check_keys(part, &["type", "text", "annotations", "logprobs"], &part_path)?;
        if matches!(part.get("annotations"), Some(JsonValue::Array(notes)) if !notes.is_empty()) {
            return Err(ApiError::unsupported_parameter(&format!("{part_path}.annotations")));
        }
        let Some(JsonValue::String(part_text)) = part.get("text") else {
            return Err(ApiError::invalid_request(
                "Text content part requires 'text'.",
                Some(&format!("{part_path}.text")),
            ));
        };
        text.push_str(part_text);
    }
    Ok(text)
}

/// The reasoning a `reasoning` item carries: its `reasoning_text` content, else its opaque token, else its summary.
fn reasoning_text(item: &JsonValue, path: &str) -> Result<String, ApiError> {
    if let Some(JsonValue::Array(parts)) = item.get("content")
        && !parts.is_empty()
    {
        let mut text = String::new();
        for part in parts {
            let is_text = matches!(part.get("type"), Some(JsonValue::String(kind)) if kind == "reasoning_text");
            let Some(JsonValue::String(part_text)) = part.get("text").filter(|_| is_text) else {
                return Err(ApiError::invalid_request(
                    "Reasoning content must be reasoning_text parts.",
                    Some(&format!("{path}.content")),
                ));
            };
            text.push_str(part_text);
        }
        return Ok(text);
    }
    if let Some(JsonValue::String(token)) = item.get("encrypted_content") {
        let param = format!("{path}.encrypted_content");
        let Some(encoded) = token.strip_prefix(ENCRYPTED_PREFIX) else {
            return Err(ApiError::invalid_request(
                "'encrypted_content' was not issued by this server.",
                Some(&param),
            ));
        };
        let Some(bytes) = decode_url_base64(encoded) else {
            return Err(ApiError::invalid_request(
                "'encrypted_content' is malformed.",
                Some(&param),
            ));
        };
        return Ok(String::from_utf8_lossy(&bytes).into_owned());
    }
    let mut text = String::new();
    if let Some(JsonValue::Array(parts)) = item.get("summary") {
        for part in parts {
            if let Some(JsonValue::String(part_text)) = part.get("text") {
                text.push_str(part_text);
            }
        }
    }
    Ok(text)
}

/// The opaque token of a reasoning text, which a client sends back in a later request.
pub fn encrypt(reasoning: &str) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut token = String::from(ENCRYPTED_PREFIX);
    for chunk in reasoning.as_bytes().chunks(3) {
        let bits = chunk.iter().enumerate().fold(0u32, |bits, (index, byte)| {
            bits | (u32::from(*byte) << (16 - 8 * index))
        });
        for index in 0..=chunk.len() {
            token.push(char::from(ALPHABET[((bits >> (18 - 6 * index)) & 0x3f) as usize]));
        }
    }
    token
}

/// Decodes the URL-safe base64 alphabet as the wire format's decoder does: padding is optional, but when present it must close
/// the last group exactly; any other character, or a last group of one character, is refused.
fn decode_url_base64(text: &str) -> Option<Vec<u8>> {
    let bytes = text.as_bytes();
    let mut decoded = Vec::with_capacity(bytes.len() / 4 * 3 + 2);
    let mut bits = 0u32;
    let mut shift: i32 = 18;
    let mut position = 0;
    while position < bytes.len() {
        let byte = bytes[position];
        position += 1;
        let value = match byte {
            b'A'..=b'Z' => byte - b'A',
            b'a'..=b'z' => byte - b'a' + 26,
            b'0'..=b'9' => byte - b'0' + 52,
            b'-' => 62,
            b'_' => 63,
            b'=' => {
                if shift == 18 || (shift == 6 && (position >= bytes.len() || bytes[position] != b'=')) {
                    return None;
                }
                if shift == 6 {
                    position += 1;
                }
                break;
            }
            _ => return None,
        };
        bits |= u32::from(value) << shift;
        shift -= 6;
        if shift < 0 {
            decoded.extend_from_slice(&[(bits >> 16) as u8, (bits >> 8) as u8, bits as u8]);
            shift = 18;
            bits = 0;
        }
    }
    match shift {
        6 => decoded.push((bits >> 16) as u8),
        0 => decoded.extend_from_slice(&[(bits >> 16) as u8, (bits >> 8) as u8]),
        12 => return None,
        _ => {}
    }
    (position >= bytes.len()).then_some(decoded)
}

/// A call's arguments: the members of the JSON object the string holds; the template renders nothing for the empty
/// string, as for `{}`.
fn arguments(value: Option<&JsonValue>, path: &str) -> Result<Vec<(String, JsonValue)>, ApiError> {
    let param = format!("{path}.arguments");
    let Some(JsonValue::String(text)) = value else {
        return Err(ApiError::invalid_request(
            "'arguments' must be a JSON string.",
            Some(&param),
        ));
    };
    if text.is_empty() {
        return Ok(Vec::new());
    }
    let Some(parsed) = JsonValue::parse(text) else {
        return Err(ApiError::invalid_request(
            "'arguments' must be valid JSON.",
            Some(&param),
        ));
    };
    let JsonValue::Object(members) = parsed else {
        return Err(ApiError::invalid_request(
            "'arguments' must encode a JSON object.",
            Some(&param),
        ));
    };
    if members.iter().any(|(key, _)| !is_parameter_name(key)) {
        return Err(ApiError::invalid_request(
            "Argument names must be non-empty and free of '<', '>' and line breaks.",
            Some(&param),
        ));
    }
    Ok(members)
}

// Options.

fn include(value: Option<&JsonValue>) -> Result<bool, ApiError> {
    let Some(value) = value else {
        return Ok(false);
    };
    let JsonValue::Array(entries) = value else {
        return Err(ApiError::invalid_request("'include' must be a list.", Some("include")));
    };
    let mut encrypted = false;
    for entry in entries {
        if matches!(entry, JsonValue::String(name) if name == "reasoning.encrypted_content") {
            encrypted = true;
        } else {
            return Err(ApiError::unsupported_parameter(&format!(
                "include={}",
                entry.param_text()
            )));
        }
    }
    Ok(encrypted)
}

fn tools(
    value: Option<&JsonValue>,
    choice: Option<&JsonValue>,
    parallel_tool_calls: Option<bool>,
) -> Result<ToolCalling, ApiError> {
    let mut functions: Vec<FunctionTool> = Vec::new();
    if let Some(value) = value {
        let JsonValue::Array(list) = value else {
            return Err(ApiError::invalid_request("'tools' must be a list.", Some("tools")));
        };
        let mut names = HashSet::new();
        for (index, tool) in list.iter().enumerate() {
            let path = format!("tools[{index}]");
            if !matches!(tool, JsonValue::Object(_)) {
                return Err(ApiError::invalid_request("Each tool must be an object.", Some(&path)));
            }
            if !matches!(tool.get("type"), Some(JsonValue::String(kind)) if kind == "function") {
                return Err(ApiError::unsupported_parameter(&format!(
                    "{path}.type={}",
                    tool.get("type")
                        .map_or_else(|| "null".to_owned(), JsonValue::param_text)
                )));
            }
            check_keys(tool, &["type", "name", "description", "parameters", "strict"], &path)?;
            let name = match tool.get("name") {
                Some(JsonValue::String(name)) if is_function_name(name) => name.clone(),
                _ => {
                    return Err(ApiError::invalid_request(
                        FUNCTION_NAME_MESSAGE,
                        Some(&format!("{path}.name")),
                    ));
                }
            };
            if !names.insert(name.clone()) {
                return Err(ApiError::invalid_request(
                    "Tool names must be unique.",
                    Some(&format!("{path}.name")),
                ));
            }
            let description = match present(tool.get("description")) {
                None => None,
                Some(JsonValue::String(text)) => Some(text.clone()),
                Some(_) => {
                    return Err(ApiError::invalid_request(
                        "'description' must be a string.",
                        Some(&format!("{path}.description")),
                    ));
                }
            };
            let strict = match present(tool.get("strict")) {
                None => None,
                Some(JsonValue::Bool(strict)) => Some(*strict),
                Some(_) => {
                    return Err(ApiError::invalid_request(
                        "'strict' must be a boolean.",
                        Some(&format!("{path}.strict")),
                    ));
                }
            };
            let parameters_param = format!("{path}.parameters");
            let parameters = parameters(present(tool.get("parameters")), &parameters_param)?.cloned();
            // In this API function tools are strict unless they say otherwise.
            functions.push(FunctionTool {
                definition: ToolDefinition {
                    name,
                    description,
                    parameters,
                },
                strict: strict != Some(false),
                parameters_param,
            });
        }
    }
    let parallel = parallel_tool_calls != Some(false);
    if functions.is_empty() {
        return match choice {
            None => Ok(no_tools(parallel)),
            Some(JsonValue::String(mode)) if mode == "none" || mode == "auto" => Ok(no_tools(parallel)),
            Some(_) => Err(ApiError::invalid_request(
                "'tool_choice' is only allowed when 'tools' are specified.",
                Some("tool_choice"),
            )),
        };
    }
    let choice = match choice {
        None => ToolChoice::Auto,
        Some(JsonValue::String(mode)) => match mode.as_str() {
            "auto" => ToolChoice::Auto,
            "none" => ToolChoice::None,
            "required" => ToolChoice::Required,
            _ => return Err(bad_choice()),
        },
        Some(named @ JsonValue::Object(_)) => {
            if !matches!(named.get("type"), Some(JsonValue::String(kind)) if kind == "function") {
                return Err(ApiError::unsupported_parameter(&format!(
                    "tool_choice.type={}",
                    named
                        .get("type")
                        .map_or_else(|| "null".to_owned(), JsonValue::param_text)
                )));
            }
            check_keys(named, &["type", "name"], "tool_choice")?;
            let Some(JsonValue::String(name)) = named.get("name").filter(|name| match name {
                JsonValue::String(name) => functions.iter().any(|tool| tool.name() == name),
                _ => false,
            }) else {
                return Err(ApiError::invalid_request(
                    "'tool_choice.name' must name one of the tools.",
                    Some("tool_choice.name"),
                ));
            };
            ToolChoice::Function(name.clone())
        }
        Some(_) => return Err(bad_choice()),
    };
    Ok(ToolCalling {
        tools: functions,
        choice,
        parallel,
    })
}

fn no_tools(parallel: bool) -> ToolCalling {
    ToolCalling {
        tools: Vec::new(),
        choice: ToolChoice::None,
        parallel,
    }
}

fn bad_choice() -> ApiError {
    ApiError::invalid_request(
        "'tool_choice' must be 'none', 'auto', 'required', or a function choice.",
        Some("tool_choice"),
    )
}

/// `reasoning`: `effort` selects the template's level; `summary` is accepted, but no summary is generated (the
/// reasoning itself is returned as `reasoning_text`). Returns the effort.
fn reasoning_options(value: Option<&JsonValue>) -> Result<Option<&JsonValue>, ApiError> {
    let Some(options) = value else {
        return Ok(None);
    };
    if !matches!(options, JsonValue::Object(_)) {
        return Err(ApiError::invalid_request(
            "'reasoning' must be an object.",
            Some("reasoning"),
        ));
    }
    check_keys(options, &["effort", "summary", "generate_summary"], "reasoning")?;
    for key in ["summary", "generate_summary"] {
        if let Some(summary) = present(options.get(key))
            && !matches!(summary, JsonValue::String(name) if matches!(name.as_str(), "auto" | "concise" | "detailed"))
        {
            return Err(ApiError::invalid_request(
                format!("'reasoning.{key}' must be auto, concise or detailed."),
                Some(&format!("reasoning.{key}")),
            ));
        }
    }
    Ok(options.get("effort"))
}

fn format(text: Option<&JsonValue>) -> Result<ResponseFormat, ApiError> {
    let Some(options) = text else {
        return Ok(ResponseFormat::Text);
    };
    if !matches!(options, JsonValue::Object(_)) {
        return Err(ApiError::invalid_request("'text' must be an object.", Some("text")));
    }
    check_keys(options, &["format", "verbosity"], "text")?;
    if present(options.get("verbosity")).is_some() {
        return Err(ApiError::unsupported_parameter("text.verbosity"));
    }
    let Some(definition) = present(options.get("format")) else {
        return Ok(ResponseFormat::Text);
    };
    let Some(JsonValue::String(kind)) = definition.get("type") else {
        return Err(ApiError::invalid_request(
            "'text.format.type' is required.",
            Some("text.format.type"),
        ));
    };
    match kind.as_str() {
        "text" => {
            check_keys(definition, &["type"], "text.format")?;
            Ok(ResponseFormat::Text)
        }
        "json_object" => {
            check_keys(definition, &["type"], "text.format")?;
            Ok(ResponseFormat::JsonObject)
        }
        "json_schema" => {
            // The flattened Responses form: the json_schema object's fields sit in the format itself.
            check_keys(
                definition,
                &["type", "name", "schema", "strict", "description"],
                "text.format",
            )?;
            let fields = match definition {
                JsonValue::Object(members) => members.iter().filter(|(key, _)| key != "type").cloned().collect(),
                _ => Vec::new(),
            };
            let chat = JsonValue::Object(vec![
                ("type".to_owned(), JsonValue::String("json_schema".to_owned())),
                ("json_schema".to_owned(), JsonValue::Object(fields)),
            ]);
            Ok(match ResponseFormat::from_request(Some(&chat))? {
                ResponseFormat::JsonSchema { name, schema, .. } => ResponseFormat::JsonSchema {
                    name,
                    schema,
                    schema_param: "text.format.schema".to_owned(),
                },
                other => other,
            })
        }
        other => Err(ApiError::unsupported_parameter(&format!("text.format.type={other}"))),
    }
}

/// Omitted fields take the defaults; temperature or top_p of 0 is greedy.
fn sampling(request: &ResponsesRequest, defaults: &Defaults) -> Result<Sampling, ApiError> {
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
    Ok(Sampling {
        temperature,
        top_k: defaults.top_k,
        top_p: if greedy { 1.0 } else { top_p },
        seed: None,
        greedy,
    })
}

#[cfg(test)]
mod tests;
