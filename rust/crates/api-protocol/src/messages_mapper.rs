//! Anthropic Messages request to [`GenerationRequest`]: every validation of the request, in the order a client sees
//! them.
//!
//! A count-tokens request is the same request without `max_tokens` and `stream`; [`map_count_tokens`] maps it to
//! the same value with no completion budget.

use std::borrow::Cow;
use std::collections::HashSet;

use crate::anthropic::MessagesRequest;
use crate::error::ApiError;
use crate::generation::{Defaults, GenerationRequest, GenerationSettings, Sampling};
use crate::json::JsonValue;
use crate::pipeline::ToolChoice;
use crate::response_format::ResponseFormat;
use crate::responses_mapper::present;
use crate::served_model::require_served_model;
use crate::template::{AssistantMessage, ContentPart, Effort, Message, TemplateOptions, ToolCall};
use crate::tool_calling::{FunctionTool, ToolCalling, check_keys, is_parameter_name, member, parameters};
use crate::tool_definition::ToolDefinition;
use crate::tool_results::ToolResults;

/// The most stop sequences a Messages request may carry.
const MAX_STOP_SEQUENCES: usize = 32;

/// Validates a `POST /v1/messages` request and maps it. The first failing check decides the error.
pub fn map_messages(request: MessagesRequest, defaults: &Defaults) -> Result<GenerationRequest, ApiError> {
    map(request, defaults, true)
}

/// Validates a `POST /v1/messages/count_tokens` request, which states no `max_tokens` and no `stream`, and maps it to
/// the request whose prompt is counted.
pub fn map_count_tokens(request: MessagesRequest, defaults: &Defaults) -> Result<GenerationRequest, ApiError> {
    map(request, defaults, false)
}

fn map(request: MessagesRequest, defaults: &Defaults, generating: bool) -> Result<GenerationRequest, ApiError> {
    for (name, _) in &request.other {
        match name.as_str() {
            "metadata" | "service_tier" => {}
            "mcp_servers" | "container" | "context_management" => {
                return Err(ApiError::unsupported_parameter(name));
            }
            _ => return Err(ApiError::unrecognized_argument(name)),
        }
    }
    let model = require_served_model(request.model.as_deref(), &defaults.model, "model: Field required")?;
    let max_tokens = request.max_tokens;
    if generating {
        match max_tokens {
            None => {
                return Err(ApiError::invalid_request(
                    "max_tokens: Field required",
                    Some("max_tokens"),
                ));
            }
            Some(value) if value < 1 => {
                return Err(ApiError::invalid_request(
                    "max_tokens: must be at least 1",
                    Some("max_tokens"),
                ));
            }
            Some(_) => {}
        }
    } else if max_tokens.is_some() {
        return Err(ApiError::unrecognized_argument("max_tokens"));
    }
    let stream = request.stream == Some(true);
    if !generating && request.stream.is_some() {
        return Err(ApiError::unrecognized_argument("stream"));
    }

    let tools = tools(present(request.tools.as_ref()), present(request.tool_choice.as_ref()))?;
    let (thinking, reasoning_budget) = thinking(
        present(request.thinking.as_ref()),
        present(request.output_config.as_ref()),
        max_tokens,
        generating,
    )?;
    let format = format(
        present(request.output_config.as_ref()),
        present(request.output_format.as_ref()),
    )?;
    let stops = stops(present(request.stop_sequences.as_ref()))?;
    if format.is_json() && !stops.is_empty() {
        return Err(ApiError::invalid_request(
            "stop_sequences cannot be combined with a JSON output format: a stop inside the JSON would end it \
             invalid.",
            Some("stop_sequences"),
        ));
    }
    let messages = turns(present(request.system.as_ref()), request.messages.as_deref())?;
    let sampling = sampling(&request, defaults)?;
    let settings = GenerationSettings {
        model,
        thinking,
        reasoning_budget,
        format,
        stops,
        sampling,
        max_tokens,
        max_tokens_param: "max_tokens".to_owned(),
        stream,
        return_token_ids: false,
    };
    Ok(GenerationRequest::new(messages, tools, settings))
}

/// Anthropic's tool-name rule: 1-128 of `a-z`, `A-Z`, `0-9`, `_` and `-`.
fn is_tool_name(name: &str) -> bool {
    (1..=128).contains(&name.len())
        && name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || byte == b'_' || byte == b'-')
}

fn invalid(message: impl Into<String>, param: &str) -> ApiError {
    ApiError::invalid_request(message, Some(param))
}

// Messages.

fn turns(system: Option<&JsonValue>, messages: Option<&[JsonValue]>) -> Result<Vec<Message>, ApiError> {
    let mut turns = Vec::new();
    if let Some(system) = system {
        let text = match system {
            JsonValue::String(plain) => plain.clone(),
            other => text_blocks(other, "system")?,
        };
        turns.push(Message::system(text));
    }
    let messages = messages.unwrap_or_default();
    if messages.is_empty() {
        return Err(invalid("messages: at least one message is required", "messages"));
    }
    let mut pending: Option<ToolResults> = None;
    for (index, message) in messages.iter().enumerate() {
        let path = format!("messages[{index}]");
        if !matches!(message, JsonValue::Object(_)) {
            return Err(invalid(format!("{path}: must be an object"), &path));
        }
        check_keys(message, &["role", "content"], &path)?;
        let Some(content) = member(message, "content") else {
            return Err(invalid(
                format!("{path}.content: Field required"),
                &format!("{path}.content"),
            ));
        };
        let blocks = blocks(content, &format!("{path}.content"))?;
        match member(message, "role") {
            Some(JsonValue::String(role)) if role == "user" => {
                let mut results = Vec::new();
                let mut text = String::new();
                for (block, item) in blocks.iter().enumerate() {
                    let block_path = format!("{path}.content[{block}]");
                    match block_type(item, &block_path)? {
                        "text" => text.push_str(block_text(item, &block_path)?),
                        "tool_result" => {
                            if !text.is_empty() {
                                return Err(invalid(
                                    format!("{block_path}: tool_result blocks must come before text in a user message"),
                                    &block_path,
                                ));
                            }
                            results.push(tool_result(item, &block_path)?);
                        }
                        _ => return Err(unsupported_block(item, &block_path)),
                    }
                }
                let answers = !results.is_empty();
                if !answers {
                    if let Some(open) = pending.take() {
                        open.close_into(&mut turns)?;
                    }
                } else {
                    let Some(mut open) = pending.take() else {
                        return Err(invalid(
                            format!(
                                "{path}: tool_result blocks must answer tool_use blocks of the preceding assistant \
                                 message"
                            ),
                            &path,
                        ));
                    };
                    for result in results {
                        open.answer(result.id.as_deref(), result.content, &result.path)?;
                    }
                    open.close_into(&mut turns)?;
                }
                if !text.is_empty() || !answers {
                    turns.push(Message::user(text));
                }
            }
            Some(JsonValue::String(role)) if role == "assistant" => {
                if let Some(open) = pending.take() {
                    open.close_into(&mut turns)?;
                }
                if index == messages.len() - 1 {
                    return Err(ApiError::unsupported_parameter(&format!(
                        "{path} (a final assistant message to continue)"
                    )));
                }
                let mut reasoning = String::new();
                let mut text = String::new();
                let mut calls = Vec::new();
                let mut ids: Vec<String> = Vec::new();
                for (block, item) in blocks.iter().enumerate() {
                    let block_path = format!("{path}.content[{block}]");
                    match block_type(item, &block_path)? {
                        "thinking" => {
                            check_keys(item, &["type", "thinking", "signature", "cache_control"], &block_path)?;
                            let Some(JsonValue::String(thought)) = member(item, "thinking") else {
                                return Err(invalid(
                                    format!("{block_path}.thinking: Field required"),
                                    &format!("{block_path}.thinking"),
                                ));
                            };
                            reasoning.push_str(thought);
                        }
                        "text" => text.push_str(block_text(item, &block_path)?),
                        "tool_use" => {
                            let call = tool_use(item, &block_path, &ids)?;
                            ids.push(call.0);
                            calls.push(call.1);
                        }
                        _ => return Err(unsupported_block(item, &block_path)),
                    }
                }
                pending = (!calls.is_empty()).then(|| ToolResults::new(ids, format!("{path}.content"), "tool_use_id"));
                turns.push(Message::Assistant(AssistantMessage {
                    content: vec![ContentPart::Text(text)],
                    tool_calls: calls,
                    reasoning,
                }));
            }
            Some(JsonValue::String(_)) => {
                return Err(invalid(
                    format!("{path}.role: must be 'user' or 'assistant'"),
                    &format!("{path}.role"),
                ));
            }
            _ => {
                return Err(invalid(format!("{path}.role: Field required"), &format!("{path}.role")));
            }
        }
    }
    if let Some(open) = pending {
        open.close_into(&mut turns)?;
    }
    Ok(turns)
}

struct PendingResult {
    id: Option<String>,
    content: String,
    path: String,
}

fn tool_result(block: &JsonValue, path: &str) -> Result<PendingResult, ApiError> {
    check_keys(
        block,
        &["type", "tool_use_id", "content", "is_error", "cache_control"],
        path,
    )?;
    if member(block, "is_error").is_some_and(|value| !matches!(value, JsonValue::Bool(_))) {
        return Err(invalid(
            format!("{path}.is_error: must be a boolean"),
            &format!("{path}.is_error"),
        ));
    }
    let content = match member(block, "content") {
        None => String::new(),
        Some(JsonValue::String(plain)) => plain.clone(),
        Some(other) => text_blocks(other, &format!("{path}.content"))?,
    };
    let id = match member(block, "tool_use_id") {
        Some(JsonValue::String(id)) => Some(id.clone()),
        _ => None,
    };
    Ok(PendingResult {
        id,
        content,
        path: format!("{path}.tool_use_id"),
    })
}

/// An assistant `tool_use` block: the id and the call.
fn tool_use(block: &JsonValue, path: &str, ids: &[String]) -> Result<(String, ToolCall), ApiError> {
    check_keys(block, &["type", "id", "name", "input", "cache_control"], path)?;
    let id = match member(block, "id") {
        Some(JsonValue::String(id)) if !id.is_empty() => id,
        _ => return Err(invalid(format!("{path}.id: Field required"), &format!("{path}.id"))),
    };
    if ids.contains(id) {
        return Err(invalid(
            format!("{path}.id: tool_use IDs must be unique"),
            &format!("{path}.id"),
        ));
    }
    let name = match member(block, "name") {
        Some(JsonValue::String(name)) if is_tool_name(name) => name,
        _ => {
            return Err(invalid(
                format!("{path}.name: must match ^[a-zA-Z0-9_-]{{1,128}}$"),
                &format!("{path}.name"),
            ));
        }
    };
    let Some(JsonValue::Object(input)) = member(block, "input") else {
        return Err(invalid(
            format!("{path}.input: must be an object"),
            &format!("{path}.input"),
        ));
    };
    if input.iter().any(|(key, _)| !is_parameter_name(key)) {
        return Err(invalid(
            format!("{path}.input: argument names must be non-empty and free of '<', '>' and line breaks"),
            &format!("{path}.input"),
        ));
    }
    Ok((
        id.clone(),
        ToolCall {
            name: name.clone(),
            arguments: input.clone(),
        },
    ))
}

/// The text of an array of text blocks, concatenated as the template concatenates text parts.
fn text_blocks(value: &JsonValue, path: &str) -> Result<String, ApiError> {
    let mut text = String::new();
    for (index, block) in blocks(value, path)?.iter().enumerate() {
        let block_path = format!("{path}[{index}]");
        if block_type(block, &block_path)? != "text" {
            return Err(unsupported_block(block, &block_path));
        }
        text.push_str(block_text(block, &block_path)?);
    }
    Ok(text)
}

/// The content blocks of a string (one text block) or an array of objects.
fn blocks<'a>(value: &'a JsonValue, path: &str) -> Result<Vec<Cow<'a, JsonValue>>, ApiError> {
    match value {
        JsonValue::String(text) => Ok(vec![Cow::Owned(JsonValue::Object(vec![
            ("type".to_owned(), JsonValue::String("text".to_owned())),
            ("text".to_owned(), JsonValue::String(text.clone())),
        ]))]),
        JsonValue::Array(items) => items
            .iter()
            .enumerate()
            .map(|(index, item)| {
                if matches!(item, JsonValue::Object(_)) {
                    Ok(Cow::Borrowed(item))
                } else {
                    let item_path = format!("{path}[{index}]");
                    Err(invalid(format!("{item_path}: must be an object"), &item_path))
                }
            })
            .collect(),
        _ => Err(invalid(
            format!("{path}: must be a string or a list of content blocks"),
            path,
        )),
    }
}

fn block_type<'a>(block: &'a JsonValue, path: &str) -> Result<&'a str, ApiError> {
    match member(block, "type") {
        Some(JsonValue::String(kind)) => Ok(kind),
        _ => Err(invalid(format!("{path}.type: Field required"), &format!("{path}.type"))),
    }
}

fn block_text<'a>(block: &'a JsonValue, path: &str) -> Result<&'a str, ApiError> {
    check_keys(block, &["type", "text", "cache_control", "citations"], path)?;
    match member(block, "citations") {
        None => {}
        Some(JsonValue::Array(citations)) if citations.is_empty() => {}
        Some(_) => return Err(ApiError::unsupported_parameter(&format!("{path}.citations"))),
    }
    match member(block, "text") {
        Some(JsonValue::String(text)) => Ok(text),
        _ => Err(invalid(format!("{path}.text: Field required"), &format!("{path}.text"))),
    }
}

fn unsupported_block(block: &JsonValue, path: &str) -> ApiError {
    ApiError::unsupported_parameter(&format!(
        "{path}.type={}",
        block
            .get("type")
            .map_or_else(|| "null".to_owned(), JsonValue::param_text)
    ))
}

// Tools.

fn tools(value: Option<&JsonValue>, choice: Option<&JsonValue>) -> Result<ToolCalling, ApiError> {
    let mut functions: Vec<FunctionTool> = Vec::new();
    if let Some(value) = value {
        let JsonValue::Array(list) = value else {
            return Err(invalid("tools: must be a list", "tools"));
        };
        let mut names = HashSet::new();
        for (index, tool) in list.iter().enumerate() {
            let path = format!("tools[{index}]");
            if !matches!(tool, JsonValue::Object(_)) {
                return Err(invalid(format!("{path}: must be an object"), &path));
            }
            if let Some(kind) = member(tool, "type")
                && *kind != JsonValue::String("custom".to_owned())
            {
                return Err(ApiError::unsupported_parameter(&format!(
                    "{path}.type={}",
                    kind.param_text()
                )));
            }
            check_keys(
                tool,
                &["type", "name", "description", "input_schema", "cache_control", "strict"],
                &path,
            )?;
            let name = match member(tool, "name") {
                Some(JsonValue::String(name)) if is_tool_name(name) => name,
                _ => {
                    return Err(invalid(
                        format!("{path}.name: must match ^[a-zA-Z0-9_-]{{1,128}}$"),
                        &format!("{path}.name"),
                    ));
                }
            };
            if !names.insert(name.as_str()) {
                return Err(invalid(
                    format!("{path}.name: tool names must be unique"),
                    &format!("{path}.name"),
                ));
            }
            let description = match member(tool, "description") {
                None => None,
                Some(JsonValue::String(text)) => Some(text.clone()),
                Some(_) => {
                    return Err(invalid(
                        format!("{path}.description: must be a string"),
                        &format!("{path}.description"),
                    ));
                }
            };
            let strict = match member(tool, "strict") {
                None => false,
                Some(JsonValue::Bool(strict)) => *strict,
                Some(_) => {
                    return Err(invalid(
                        format!("{path}.strict: must be a boolean"),
                        &format!("{path}.strict"),
                    ));
                }
            };
            let schema_path = format!("{path}.input_schema");
            let Some(schema) = member(tool, "input_schema") else {
                return Err(invalid(format!("{schema_path}: Field required"), &schema_path));
            };
            let parameters = parameters(Some(schema), &schema_path)?.cloned();
            functions.push(FunctionTool {
                definition: ToolDefinition {
                    name: name.clone(),
                    description,
                    parameters,
                },
                strict,
                parameters_param: schema_path,
            });
        }
    }
    let calling = |functions: Vec<FunctionTool>, choice: ToolChoice, parallel: bool| ToolCalling {
        tools: functions,
        choice,
        parallel,
    };
    let Some(selected) = choice else {
        return Ok(if functions.is_empty() {
            ToolCalling::disabled()
        } else {
            calling(functions, ToolChoice::Auto, true)
        });
    };
    if !matches!(selected, JsonValue::Object(_)) {
        return Err(invalid("tool_choice: must be an object", "tool_choice"));
    }
    check_keys(selected, &["type", "name", "disable_parallel_tool_use"], "tool_choice")?;
    if member(selected, "disable_parallel_tool_use").is_some_and(|value| !matches!(value, JsonValue::Bool(_))) {
        return Err(invalid(
            "tool_choice.disable_parallel_tool_use: must be a boolean",
            "tool_choice.disable_parallel_tool_use",
        ));
    }
    let parallel = member(selected, "disable_parallel_tool_use") != Some(&JsonValue::Bool(true));
    let kind = match member(selected, "type") {
        Some(JsonValue::String(kind)) => Some(kind.as_str()),
        _ => None,
    };
    if kind == Some("none") {
        return Ok(calling(functions, ToolChoice::None, parallel));
    }
    if functions.is_empty() {
        return Err(invalid("tool_choice: requires tools", "tool_choice"));
    }
    match kind {
        Some("auto") => Ok(calling(functions, ToolChoice::Auto, parallel)),
        Some("any") => Ok(calling(functions, ToolChoice::Required, parallel)),
        Some("tool") => {
            let name = match member(selected, "name") {
                Some(JsonValue::String(name)) if functions.iter().any(|tool| tool.name() == name) => name.clone(),
                _ => {
                    return Err(invalid(
                        "tool_choice.name: must name one of the tools",
                        "tool_choice.name",
                    ));
                }
            };
            Ok(calling(functions, ToolChoice::Function(name), parallel))
        }
        None => Err(invalid("tool_choice.type: Field required", "tool_choice.type")),
        Some(_) => Err(invalid(
            "tool_choice.type: must be auto, any, tool or none",
            "tool_choice.type",
        )),
    }
}

// Thinking and output.

/// Extended thinking is opt-in in this API: without `thinking` the model answers directly. Returns the template's
/// thinking variables and the cap on reasoning tokens.
fn thinking(
    value: Option<&JsonValue>,
    output_config: Option<&JsonValue>,
    max_tokens: Option<i32>,
    generating: bool,
) -> Result<(TemplateOptions, Option<u32>), ApiError> {
    let effort = effort(output_config)?;
    let needs_thinking = || {
        invalid(
            "output_config.effort: selects how much the model thinks, which needs thinking enabled",
            "output_config.effort",
        )
    };
    let Some(value) = value else {
        if effort.is_some() {
            return Err(needs_thinking());
        }
        return Ok((TemplateOptions::THINKING_DISABLED, None));
    };
    if !matches!(value, JsonValue::Object(_)) {
        return Err(invalid("thinking: must be an object", "thinking"));
    }
    let level = effort.unwrap_or(Effort::XHigh);
    let on = TemplateOptions {
        effort: Some(level),
        preserve_thinking: true,
    };
    match member(value, "type") {
        Some(JsonValue::String(kind)) if kind == "disabled" => {
            check_keys(value, &["type"], "thinking")?;
            if effort.is_some() {
                return Err(needs_thinking());
            }
            Ok((TemplateOptions::THINKING_DISABLED, None))
        }
        Some(JsonValue::String(kind)) if kind == "adaptive" => {
            check_keys(value, &["type"], "thinking")?;
            Ok((on, None))
        }
        Some(JsonValue::String(kind)) if kind == "enabled" => {
            check_keys(value, &["type", "budget_tokens"], "thinking")?;
            let budget = match member(value, "budget_tokens") {
                Some(JsonValue::Integer(digits)) => digits.as_str().parse::<i32>().ok().filter(|budget| *budget >= 1),
                _ => None,
            };
            let Some(budget) = budget else {
                return Err(invalid(
                    "thinking.budget_tokens: must be a positive integer",
                    "thinking.budget_tokens",
                ));
            };
            if generating && max_tokens.is_some_and(|max_tokens| budget >= max_tokens) {
                return Err(invalid(
                    "thinking.budget_tokens: must be less than max_tokens",
                    "thinking.budget_tokens",
                ));
            }
            Ok((on, Some(budget as u32)))
        }
        None => Err(invalid("thinking.type: Field required", "thinking.type")),
        Some(JsonValue::String(_)) => Err(invalid(
            "thinking.type: must be enabled, adaptive or disabled",
            "thinking.type",
        )),
        // A type that is no string reads as absent.
        Some(_) => Err(invalid("thinking.type: Field required", "thinking.type")),
    }
}

/// `output_config.effort`: `low` and `medium` are the template's; `high` and `max` its highest, `xhigh`.
fn effort(output_config: Option<&JsonValue>) -> Result<Option<Effort>, ApiError> {
    let Some(config @ JsonValue::Object(_)) = output_config else {
        return Ok(None);
    };
    let Some(effort) = member(config, "effort") else {
        return Ok(None);
    };
    match effort {
        JsonValue::String(name) if name == "low" => Ok(Some(Effort::Low)),
        JsonValue::String(name) if name == "medium" => Ok(Some(Effort::Medium)),
        JsonValue::String(name) if name == "high" || name == "max" => Ok(Some(Effort::XHigh)),
        _ => Err(invalid(
            "output_config.effort: must be low, medium, high or max",
            "output_config.effort",
        )),
    }
}

fn format(output_config: Option<&JsonValue>, legacy: Option<&JsonValue>) -> Result<ResponseFormat, ApiError> {
    let mut format = None;
    let mut path = "";
    if let Some(config) = output_config {
        if !matches!(config, JsonValue::Object(_)) {
            return Err(invalid("output_config: must be an object", "output_config"));
        }
        check_keys(config, &["effort", "format"], "output_config")?;
        format = member(config, "format");
        path = "output_config.format";
    }
    if let Some(legacy) = legacy {
        if format.is_some() {
            return Err(invalid(
                "output_format: conflicts with output_config.format; send only one",
                "output_format",
            ));
        }
        format = Some(legacy);
        path = "output_format";
    }
    let Some(definition) = format else {
        return Ok(ResponseFormat::Text);
    };
    if !matches!(definition, JsonValue::Object(_))
        || member(definition, "type") != Some(&JsonValue::String("json_schema".to_owned()))
    {
        return Err(invalid(format!("{path}: type must be json_schema"), path));
    }
    check_keys(definition, &["type", "schema"], path)?;
    let schema_path = format!("{path}.schema");
    let Some(schema @ JsonValue::Object(_)) = member(definition, "schema") else {
        return Err(invalid(
            format!("{schema_path}: must be a JSON Schema object"),
            &schema_path,
        ));
    };
    if schema.get("x-guidance").is_some() {
        return Err(ApiError::unsupported_parameter(&format!("{schema_path}.x-guidance")));
    }
    Ok(ResponseFormat::JsonSchema {
        name: None,
        schema: schema.clone(),
        schema_param: schema_path,
    })
}

fn stops(value: Option<&JsonValue>) -> Result<Vec<String>, ApiError> {
    let Some(value) = value else {
        return Ok(Vec::new());
    };
    let JsonValue::Array(list) = value else {
        return Err(invalid("stop_sequences: must be a list of strings", "stop_sequences"));
    };
    if list.len() > MAX_STOP_SEQUENCES {
        return Err(invalid(
            format!("stop_sequences: at most {MAX_STOP_SEQUENCES} sequences"),
            "stop_sequences",
        ));
    }
    list.iter()
        .map(|stop| match stop {
            JsonValue::String(text) if !text.is_empty() => Ok(text.clone()),
            _ => Err(invalid(
                "stop_sequences: each sequence must be a non-empty string",
                "stop_sequences",
            )),
        })
        .collect()
}

/// Omitted values take the defaults; a temperature of 0 is greedy.
fn sampling(request: &MessagesRequest, defaults: &Defaults) -> Result<Sampling, ApiError> {
    let mut temperature = defaults.temperature;
    if let Some(value) = request.temperature {
        if !(0.0..=1.0).contains(&value) {
            return Err(invalid("temperature: must be between 0 and 1", "temperature"));
        }
        temperature = value as f32;
    }
    let mut top_p = defaults.top_p;
    if let Some(value) = request.top_p {
        if !(value > 0.0 && value <= 1.0) {
            return Err(invalid("top_p: must be greater than 0 and at most 1", "top_p"));
        }
        top_p = value as f32;
    }
    let mut top_k = defaults.top_k;
    if let Some(value) = request.top_k {
        if value < 1 {
            return Err(invalid("top_k: must be at least 1", "top_k"));
        }
        top_k = value;
    }
    let greedy = temperature == 0.0;
    Ok(Sampling {
        temperature,
        top_k,
        top_p: if greedy { 1.0 } else { top_p },
        seed: None,
        greedy,
    })
}

#[cfg(test)]
mod tests;
