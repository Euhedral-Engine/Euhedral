//! A prompts file: one JSON record a line, `{"id", "messages": [{"role", "content", "tool_calls"}], "tools"}`, rendered
//! with the chat template at its default thinking settings and tokenized with no start or end token.
//!
//! A message's `content` is text (absent: empty), its role one of `system`, `user`, `assistant` and `tool` in any
//! case. An assistant message's `tool_calls` are OpenAI's: `function.name` and `function.arguments`, a JSON object
//! or its text. A tool is any JSON object; the template prints it as written, keys in the order of the record. The `id` is a string or an integer.

use euhedral_inference_api_protocol::json::JsonValue;
use euhedral_inference_api_protocol::template::{
    AssistantMessage, ContentPart, Message, TemplateOptions, ToolCall, render_tool_objects,
};
use euhedral_inference_tokenizer::Tokenizer;

use super::Prompt;

/// The prompts in `text`, rendered and tokenized by `tokenizer`; `source` names the file in a refusal.
pub fn read(text: &str, source: &str, tokenizer: &Tokenizer) -> Result<Vec<Prompt>, String> {
    let mut prompts = Vec::new();
    for (index, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let at = |why: String| format!("{source} line {}: {why}", index + 1);
        let (id, messages, tools) = parse(line).map_err(at)?;
        let rendered = render_tool_objects(&messages, &tools, &TemplateOptions::default())
            .map_err(|error| at(error.to_string()))?;
        let tokens = tokenizer
            .encode(&rendered)
            .map_err(|error| at(format!("prompt {id}: {error}")))?;
        prompts.push(Prompt { id, tokens });
    }
    Ok(prompts)
}

/// One record: its id, its messages and its tools, each tool as the record writes it.
pub fn parse(line: &str) -> Result<(String, Vec<Message>, Vec<JsonValue>), String> {
    let record = JsonValue::parse(line).ok_or("not a JSON document")?;
    let id = match record.get("id") {
        Some(JsonValue::String(id)) if !id.is_empty() => id.clone(),
        Some(JsonValue::Integer(number)) => number.as_str().to_owned(),
        _ => return Err("the record needs an id, a string or an integer".to_owned()),
    };
    let mut messages = Vec::new();
    for message in items(record.get("messages")) {
        messages.push(message_of(message).map_err(|why| format!("prompt {id}: {why}"))?);
    }
    let mut tools = Vec::new();
    for tool in items(record.get("tools")) {
        if !matches!(tool, JsonValue::Object(_)) {
            return Err(format!("prompt {id}: a tool is not a JSON object"));
        }
        tools.push(tool.clone());
    }
    Ok((id, messages, tools))
}

/// The elements of an array, none for anything else.
fn items(value: Option<&JsonValue>) -> &[JsonValue] {
    match value {
        Some(JsonValue::Array(items)) => items,
        _ => &[],
    }
}

fn text_of(value: Option<&JsonValue>) -> String {
    match value {
        Some(JsonValue::String(text)) => text.clone(),
        _ => String::new(),
    }
}

fn message_of(message: &JsonValue) -> Result<Message, String> {
    let role = text_of(message.get("role")).to_ascii_lowercase();
    let content = vec![ContentPart::Text(text_of(message.get("content")))];
    match role.as_str() {
        "system" => Ok(Message::System(content)),
        "user" => Ok(Message::User(content)),
        "tool" => Ok(Message::Tool(content)),
        "assistant" => {
            let tool_calls = items(message.get("tool_calls"))
                .iter()
                .map(call_of)
                .collect::<Result<_, _>>()?;
            Ok(Message::Assistant(AssistantMessage {
                content,
                tool_calls,
                reasoning: String::new(),
            }))
        }
        other => Err(format!(
            "a message has the role \"{other}\"; the roles are system, user, assistant and tool"
        )),
    }
}

fn call_of(call: &JsonValue) -> Result<ToolCall, String> {
    let function = call.get("function").ok_or("a tool call has no function")?;
    let name = text_of(function.get("name"));
    let arguments = match function.get("arguments") {
        None | Some(JsonValue::Null) => JsonValue::Object(Vec::new()),
        Some(JsonValue::String(text)) => {
            JsonValue::parse(text).ok_or_else(|| format!("the arguments of the call to {name} are not JSON"))?
        }
        Some(other) => other.clone(),
    };
    match arguments {
        JsonValue::Object(arguments) => Ok(ToolCall { name, arguments }),
        _ => Err(format!("the arguments of the call to {name} are not an object")),
    }
}
