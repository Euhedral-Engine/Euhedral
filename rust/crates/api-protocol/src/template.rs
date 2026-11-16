//! The chat template of the Qwen checkpoint, rendered with the generation prompt appended.
//!
//! The rendering covers the template's text branches: system, user and assistant text, the thinking controls
//! (`enable_thinking`, `reasoning_effort`, `preserve_thinking`) and replayed reasoning, the tool-definition system
//! block, assistant `<tool_call>` XML and grouped `<tool_response>` turns. Text that comes from a client has the
//! spellings of control tokens defused, so it cannot be encoded as one.

mod escape;
mod json;

use std::error::Error;
use std::fmt::{self, Display, Formatter};

use escape::{escape_control_tokens, trim};
pub use json::JsonValue;

const IM_START: &str = "<|im_start|>";
const IM_END: &str = "<|im_end|>";
const THINK_START: &str = "<think>";
const THINK_END: &str = "</think>";
const TOOL_CALL_START: &str = "<tool_call>";
const TOOL_CALL_END: &str = "</tool_call>";
const TOOL_RESPONSE_START: &str = "<tool_response>";
const TOOL_RESPONSE_END: &str = "</tool_response>";

/// The format instructions that follow the tool definitions.
const TOOL_INSTRUCTIONS: &str = "\n\nIf you choose to call a function ONLY reply in the following format with NO suffix:\n\n\
<tool_call>\n<function=example_function_name>\n<parameter=example_parameter_1>\nvalue_1\n\
</parameter>\n<parameter=example_parameter_2>\nThis is the value for the second parameter\n\
that can span\nmultiple lines\n</parameter>\n</function>\n</tool_call>\n\n<IMPORTANT>\n\
Reminder:\n\
- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n\
- Required parameters MUST be specified\n\
- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n\
- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n\
</IMPORTANT>";

const XHIGH_INSTRUCTIONS: &str = "Reasoning effort is set to xhigh. Please think carefully through the task, \
validate key assumptions, consider plausible alternatives, and prioritize correctness, consistency, and clarity in \
the final answer.";
const LOW_INSTRUCTIONS: &str = "Reasoning effort is set to low. Keep your thinking brief and focused, moving \
directly to the conclusion without unnecessary elaboration.";

/// The speaker of a message.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// One piece of a message's content.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ContentPart {
    Text(String),
}

/// A function call of an assistant message: `arguments` are the parsed JSON object's members, in order.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Vec<(String, JsonValue)>,
}

/// One turn. Only assistant messages carry `tool_calls`, rendered after the content in list order, and
/// `reasoning`, the replayed reasoning of an earlier response (empty when there is none).
#[derive(Debug, Clone, PartialEq)]
pub struct Message {
    pub role: Role,
    pub content: Vec<ContentPart>,
    pub tool_calls: Vec<ToolCall>,
    pub reasoning: String,
}

impl Message {
    /// A message with one text part and no calls or reasoning.
    pub fn text(role: Role, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text(text.into())],
            tool_calls: Vec::new(),
            reasoning: String::new(),
        }
    }

    /// The text parts joined in order.
    fn joined_text(&self) -> String {
        self.content
            .iter()
            .map(|ContentPart::Text(text)| text.as_str())
            .collect()
    }
}

/// A function tool offered to the model. `description` and `parameters` are left out of the prompt when `None`.
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    pub name: String,
    pub description: Option<String>,
    pub parameters: Option<JsonValue>,
}

impl ToolDefinition {
    /// The object the template serializes for the tool: `{"type": "function", "function": {...}}`.
    fn to_json(&self) -> JsonValue {
        let mut function = vec![("name".to_owned(), JsonValue::String(self.name.clone()))];
        if let Some(description) = &self.description {
            function.push(("description".to_owned(), JsonValue::String(description.clone())));
        }
        if let Some(parameters) = &self.parameters {
            function.push(("parameters".to_owned(), parameters.clone()));
        }
        JsonValue::Object(vec![
            ("type".to_owned(), JsonValue::String("function".to_owned())),
            ("function".to_owned(), JsonValue::Object(function)),
        ])
    }
}

/// What to render: the messages and the tools the model may call (none renders the template's no-tools branch).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct Conversation {
    pub messages: Vec<Message>,
    pub tools: Vec<ToolDefinition>,
}

/// The template's `reasoning_effort` values. Only `XHigh` and `Low` add system instructions.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Effort {
    Low,
    Medium,
    XHigh,
}

impl Effort {
    fn instructions(self) -> &'static str {
        match self {
            Self::Low => LOW_INSTRUCTIONS,
            Self::Medium => "",
            Self::XHigh => XHIGH_INSTRUCTIONS,
        }
    }
}

/// The template's thinking variables. `effort` `None` is `enable_thinking=false`; otherwise thinking is on with
/// that `reasoning_effort`. `preserve_thinking` false drops the think block of assistant messages before the last
/// user query.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TemplateOptions {
    pub effort: Option<Effort>,
    pub preserve_thinking: bool,
}

impl TemplateOptions {
    /// Thinking off, as `enable_thinking=false`.
    pub const THINKING_DISABLED: Self = Self {
        effort: None,
        preserve_thinking: true,
    };
}

impl Default for TemplateOptions {
    /// What the template renders when the caller sets none of its thinking variables.
    fn default() -> Self {
        Self {
            effort: Some(Effort::XHigh),
            preserve_thinking: true,
        }
    }
}

/// A conversation the template refuses; the messages are the template's own.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    /// The conversation has no messages.
    NoMessages,
    /// No user message that is not a wrapped tool response.
    NoUserQuery,
    /// A system message after the first position.
    SystemNotFirst,
    /// A message other than an assistant message carries tool calls.
    ToolCallsOnNonAssistant,
    /// A message other than an assistant message carries reasoning.
    ReasoningOnNonAssistant,
}

impl Display for TemplateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NoMessages => "No messages provided.",
            Self::NoUserQuery => "No user query found in messages.",
            Self::SystemNotFirst => "System message must be at the beginning.",
            Self::ToolCallsOnNonAssistant => "only assistant turns carry tool calls",
            Self::ReasoningOnNonAssistant => "only assistant turns carry reasoning",
        })
    }
}

impl Error for TemplateError {}

/// Renders the conversation followed by the assistant generation prompt.
pub fn render(conversation: &Conversation, options: &TemplateOptions) -> Result<String, TemplateError> {
    let messages = &conversation.messages;
    for message in messages {
        if !message.tool_calls.is_empty() && message.role != Role::Assistant {
            return Err(TemplateError::ToolCallsOnNonAssistant);
        }
        if !message.reasoning.is_empty() && message.role != Role::Assistant {
            return Err(TemplateError::ReasoningOnNonAssistant);
        }
    }
    let Some(first) = messages.first() else {
        return Err(TemplateError::NoMessages);
    };
    let texts: Vec<String> = messages.iter().map(Message::joined_text).collect();
    let mut prompt = String::new();
    let instructions = options.effort.map_or("", Effort::instructions);
    let system = if first.role == Role::System {
        trim(&texts[0])
    } else {
        ""
    };
    if !conversation.tools.is_empty() {
        prompt.push_str(IM_START);
        prompt.push_str("system\n");
        if !instructions.is_empty() {
            prompt.push_str(instructions);
            prompt.push_str("\n\n");
        }
        prompt.push_str("# Tools\n\nYou have access to the following functions:\n\n<tools>");
        for tool in &conversation.tools {
            prompt.push('\n');
            prompt.push_str(&escape_control_tokens(&tool.to_json().to_python_json()));
        }
        prompt.push_str("\n</tools>");
        prompt.push_str(TOOL_INSTRUCTIONS);
        if !system.is_empty() {
            prompt.push_str("\n\n");
            prompt.push_str(system);
        }
        prompt.push_str(IM_END);
        prompt.push('\n');
    } else if !system.is_empty() || !instructions.is_empty() {
        prompt.push_str(IM_START);
        prompt.push_str("system\n");
        prompt.push_str(instructions);
        if !system.is_empty() && !instructions.is_empty() {
            prompt.push_str("\n\n");
        }
        prompt.push_str(system);
        prompt.push_str(IM_END);
        prompt.push('\n');
    }
    let last_query = last_user_query(messages, &texts).ok_or(TemplateError::NoUserQuery)?;
    for (index, message) in messages.iter().enumerate() {
        let content = trim(&texts[index]);
        match message.role {
            Role::System => {
                if index != 0 {
                    return Err(TemplateError::SystemNotFirst);
                }
            }
            Role::User => {
                prompt.push_str(IM_START);
                prompt.push_str("user\n");
                prompt.push_str(&escape_control_tokens(content));
                prompt.push_str(IM_END);
                prompt.push('\n');
            }
            Role::Assistant => {
                prompt.push_str(IM_START);
                prompt.push_str("assistant\n");
                if options.preserve_thinking || index > last_query {
                    prompt.push_str(THINK_START);
                    prompt.push('\n');
                    prompt.push_str(&escape_control_tokens(trim(&message.reasoning)));
                    prompt.push('\n');
                    prompt.push_str(THINK_END);
                    prompt.push_str("\n\n");
                }
                prompt.push_str(&escape_control_tokens(content));
                append_tool_calls(&message.tool_calls, !content.is_empty(), &mut prompt);
                prompt.push_str(IM_END);
                prompt.push('\n');
            }
            // Consecutive tool results share one user turn, opened only after a non-tool message.
            Role::Tool => {
                if index > 0 && messages[index - 1].role != Role::Tool {
                    prompt.push_str(IM_START);
                    prompt.push_str("user");
                }
                prompt.push('\n');
                prompt.push_str(TOOL_RESPONSE_START);
                prompt.push('\n');
                prompt.push_str(&escape_control_tokens(content));
                prompt.push('\n');
                prompt.push_str(TOOL_RESPONSE_END);
                if messages.get(index + 1).is_none_or(|next| next.role != Role::Tool) {
                    prompt.push_str(IM_END);
                    prompt.push('\n');
                }
            }
        }
    }
    prompt.push_str(IM_START);
    prompt.push_str("assistant\n");
    prompt.push_str(THINK_START);
    prompt.push('\n');
    if options.effort.is_none() {
        prompt.push('\n');
        prompt.push_str(THINK_END);
        prompt.push_str("\n\n");
    }
    Ok(prompt)
}

/// Appends the `<tool_call>` blocks: string arguments verbatim, every other value as JSON.
fn append_tool_calls(calls: &[ToolCall], after_content: bool, prompt: &mut String) {
    for (index, call) in calls.iter().enumerate() {
        if index > 0 || after_content {
            prompt.push_str(if index == 0 { "\n\n" } else { "\n" });
        }
        prompt.push_str(TOOL_CALL_START);
        prompt.push_str("\n<function=");
        prompt.push_str(&escape_control_tokens(&call.name));
        prompt.push_str(">\n");
        for (key, value) in &call.arguments {
            prompt.push_str("<parameter=");
            prompt.push_str(&escape_control_tokens(key));
            prompt.push_str(">\n");
            match value {
                JsonValue::String(text) => prompt.push_str(&escape_control_tokens(text)),
                other => prompt.push_str(&escape_control_tokens(&other.to_python_json())),
            }
            prompt.push_str("\n</parameter>\n");
        }
        prompt.push_str("</function>\n");
        prompt.push_str(TOOL_CALL_END);
    }
}

/// The index of the last user message that is not a wrapped tool response.
fn last_user_query(messages: &[Message], texts: &[String]) -> Option<usize> {
    messages.iter().enumerate().rev().find_map(|(index, message)| {
        if message.role != Role::User {
            return None;
        }
        let content = trim(&texts[index]);
        let wrapped = content.starts_with(TOOL_RESPONSE_START) && content.ends_with(TOOL_RESPONSE_END);
        (!wrapped).then_some(index)
    })
}
