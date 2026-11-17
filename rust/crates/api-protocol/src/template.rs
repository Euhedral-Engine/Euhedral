//! The chat template of the Qwen checkpoint, rendered with the generation prompt appended.
//!
//! The rendering covers the template's text branches: system, user and assistant text, the thinking controls
//! (`enable_thinking`, `reasoning_effort`, `preserve_thinking`) and replayed reasoning, the tool-definition system
//! block, assistant `<tool_call>` XML and grouped `<tool_response>` turns. Text that comes from a client has the
//! spellings of control tokens defused, so it cannot be encoded as one.

mod escape;

use std::error::Error;
use std::fmt::{self, Display, Formatter};

pub use crate::json::{JsonInteger, JsonValue};
pub use crate::tool_definition::ToolDefinition;
use escape::{escape_control_tokens, trim};

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

/// The control tokens the rendering emits; the tokenizer must encode each as one token.
pub const CONTROL_TOKENS: [&str; 8] = [
    IM_START,
    IM_END,
    THINK_START,
    THINK_END,
    TOOL_CALL_START,
    TOOL_CALL_END,
    TOOL_RESPONSE_START,
    TOOL_RESPONSE_END,
];

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

/// One turn. Only assistant messages carry tool calls and replayed reasoning.
#[derive(Debug, Clone, PartialEq)]
pub enum Message {
    System(Vec<ContentPart>),
    User(Vec<ContentPart>),
    Assistant(AssistantMessage),
    /// The result of a tool call.
    Tool(Vec<ContentPart>),
}

/// An assistant turn: its content, then `tool_calls` rendered after it in list order, and `reasoning`, the replayed
/// reasoning of an earlier response (empty when there is none).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct AssistantMessage {
    pub content: Vec<ContentPart>,
    pub tool_calls: Vec<ToolCall>,
    pub reasoning: String,
}

impl Message {
    /// A system message of one text part.
    pub fn system(text: impl Into<String>) -> Self {
        Self::System(vec![ContentPart::Text(text.into())])
    }

    /// A user message of one text part.
    pub fn user(text: impl Into<String>) -> Self {
        Self::User(vec![ContentPart::Text(text.into())])
    }

    /// A tool-result message of one text part.
    pub fn tool(text: impl Into<String>) -> Self {
        Self::Tool(vec![ContentPart::Text(text.into())])
    }

    /// An assistant message of one text part, without calls or reasoning.
    pub fn assistant(text: impl Into<String>) -> Self {
        Self::Assistant(AssistantMessage {
            content: vec![ContentPart::Text(text.into())],
            ..AssistantMessage::default()
        })
    }

    /// The speaker.
    pub fn role(&self) -> Role {
        match self {
            Self::System(_) => Role::System,
            Self::User(_) => Role::User,
            Self::Assistant(_) => Role::Assistant,
            Self::Tool(_) => Role::Tool,
        }
    }

    /// The text parts joined in order.
    fn joined_text(&self) -> String {
        let parts = match self {
            Self::System(parts) | Self::User(parts) | Self::Tool(parts) => parts,
            Self::Assistant(assistant) => &assistant.content,
        };
        parts.iter().map(|ContentPart::Text(text)| text.as_str()).collect()
    }
}

/// What to render: the messages and the tools the model may call (none renders the template's no-tools branch).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TemplateInput {
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

/// An input the template refuses; the messages are the template's own. All three are errors of the client's
/// conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplateError {
    /// The input has no messages.
    NoMessages,
    /// No user message that is not a wrapped tool response.
    NoUserQuery,
    /// A system message after the first position.
    SystemNotFirst,
}

impl Display for TemplateError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.write_str(match self {
            Self::NoMessages => "No messages provided.",
            Self::NoUserQuery => "No user query found in messages.",
            Self::SystemNotFirst => "System message must be at the beginning.",
        })
    }
}

impl Error for TemplateError {}

/// Renders the conversation followed by the assistant generation prompt.
pub fn render(conversation: &TemplateInput, options: &TemplateOptions) -> Result<String, TemplateError> {
    let messages = &conversation.messages;
    let Some(first) = messages.first() else {
        return Err(TemplateError::NoMessages);
    };
    let texts: Vec<String> = messages.iter().map(Message::joined_text).collect();
    let mut prompt = String::new();
    let instructions = options.effort.map_or("", Effort::instructions);
    let system = if first.role() == Role::System {
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
        match message {
            Message::System(_) => {
                if index != 0 {
                    return Err(TemplateError::SystemNotFirst);
                }
            }
            Message::User(_) => {
                prompt.push_str(IM_START);
                prompt.push_str("user\n");
                prompt.push_str(&escape_control_tokens(content));
                prompt.push_str(IM_END);
                prompt.push('\n');
            }
            Message::Assistant(assistant) => {
                prompt.push_str(IM_START);
                prompt.push_str("assistant\n");
                if options.preserve_thinking || index > last_query {
                    prompt.push_str(THINK_START);
                    prompt.push('\n');
                    prompt.push_str(&escape_control_tokens(trim(&assistant.reasoning)));
                    prompt.push('\n');
                    prompt.push_str(THINK_END);
                    prompt.push_str("\n\n");
                }
                prompt.push_str(&escape_control_tokens(content));
                append_tool_calls(&assistant.tool_calls, !content.is_empty(), &mut prompt);
                prompt.push_str(IM_END);
                prompt.push('\n');
            }
            // Consecutive tool results share one user turn, opened only after a non-tool message.
            Message::Tool(_) => {
                if index > 0 && messages[index - 1].role() != Role::Tool {
                    prompt.push_str(IM_START);
                    prompt.push_str("user");
                }
                prompt.push('\n');
                prompt.push_str(TOOL_RESPONSE_START);
                prompt.push('\n');
                prompt.push_str(&escape_control_tokens(content));
                prompt.push('\n');
                prompt.push_str(TOOL_RESPONSE_END);
                if messages.get(index + 1).is_none_or(|next| next.role() != Role::Tool) {
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
        if message.role() != Role::User {
            return None;
        }
        let content = trim(&texts[index]);
        let wrapped = content.starts_with(TOOL_RESPONSE_START) && content.ends_with(TOOL_RESPONSE_END);
        (!wrapped).then_some(index)
    })
}

/// A template source that is not the one the rendering reproduces.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TemplateSourceError {
    fragment: String,
}

impl TemplateSourceError {
    /// The fragment of template source that is missing.
    pub fn fragment(&self) -> &str {
        &self.fragment
    }
}

impl Display for TemplateSourceError {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "unsupported checkpoint chat template; the Qwen formatter no longer matches fragment: {}",
            self.fragment
        )
    }
}

impl Error for TemplateSourceError {}

/// Verifies that `source`, the checkpoint's template, still contains every fragment of template source whose
/// behaviour `render` reproduces; CRLF line endings are read as LF. An artifact whose template changed is refused
/// here instead of being rendered in the old format.
pub fn check_template_source(source: &str) -> Result<(), TemplateSourceError> {
    let normalized = source.replace("\r\n", "\n");
    // The instructions as the template's string literals spell them (`\n` escapes).
    let tool_instructions = TOOL_INSTRUCTIONS.replace('\n', "\\n");
    let fragments = [
        "{%- if tools and tools is iterable and tools is not mapping %}".to_owned(),
        "{{- \"# Tools\\n\\nYou have access to the following functions:\\n\\n<tools>\" }}".to_owned(),
        "{{- \"\\n\" }}\n        {{- tool | tojson }}".to_owned(),
        "{{- \"\\n</tools>\" }}".to_owned(),
        format!("{{{{- '{tool_instructions}' }}}}"),
        "{{- '\\n\\n' + content }}".to_owned(),
        "{%- if messages[0].role == 'system' %}".to_owned(),
        "render_content(messages[0].content, false, true)|trim".to_owned(),
        "'<|im_start|>system\\n' + (reasoning_instructions + '\\n\\n' if reasoning_instructions else '')  + content + '<|im_end|>\\n'".to_owned(),
        "raise_exception('No user query found in messages.')".to_owned(),
        "render_content(message.content, true)|trim".to_owned(),
        "raise_exception('System message must be at the beginning.')".to_owned(),
        "'<|im_start|>' + message.role + '\\n' + content + '<|im_end|>' + '\\n'".to_owned(),
        "'<|im_start|>' + message.role + '\\n<think>\\n' + reasoning_content + '\\n</think>\\n\\n' + content".to_owned(),
        "{%- if message.tool_calls and message.tool_calls is iterable and message.tool_calls is not mapping %}".to_owned(),
        "{{- '\\n\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}".to_owned(),
        "{{- '<tool_call>\\n<function=' + tool_call.name + '>\\n' }}".to_owned(),
        "{{- '\\n<tool_call>\\n<function=' + tool_call.name + '>\\n' }}".to_owned(),
        "{%- for args_name, args_value in tool_call.arguments|items %}".to_owned(),
        "{{- '<parameter=' + args_name + '>\\n' }}".to_owned(),
        "{%- set args_value = args_value | string if args_value is string else args_value | tojson | safe %}".to_owned(),
        "{{- '\\n</parameter>\\n' }}".to_owned(),
        "{{- '</function>\\n</tool_call>' }}".to_owned(),
        "{%- if loop.previtem and loop.previtem.role != \"tool\" %}".to_owned(),
        "{{- '<|im_start|>user' }}".to_owned(),
        "{{- '\\n<tool_response>\\n' }}".to_owned(),
        "{{- '\\n</tool_response>' }}".to_owned(),
        "{%- if not loop.last and loop.nextitem.role != \"tool\" %}".to_owned(),
        "{%- if enable_thinking is undefined or enable_thinking is true %}".to_owned(),
        "{%- set resolved_reasoning_effort = reasoning_effort|default('xhigh') %}".to_owned(),
        "{%- if resolved_reasoning_effort not in ('xhigh', 'medium', 'low') %}".to_owned(),
        "{%- if resolved_reasoning_effort == 'xhigh' %}".to_owned(),
        format!("{{%- set reasoning_instructions = '{XHIGH_INSTRUCTIONS}' %}}"),
        "{%- elif resolved_reasoning_effort == 'low' %}".to_owned(),
        format!("{{%- set reasoning_instructions = '{LOW_INSTRUCTIONS}' %}}"),
        "{{- reasoning_instructions + '\\n\\n' }}".to_owned(),
        "{{- '<|im_start|>system\\n' + reasoning_instructions + '<|im_end|>\\n' }}".to_owned(),
        "{%- if message.reasoning_content is string %}".to_owned(),
        "{%- set reasoning_content = reasoning_content|trim %}".to_owned(),
        "{%- if preserve_thinking is undefined or preserve_thinking is true or loop.index0 > ns.last_query_index %}".to_owned(),
        "{{- '<|im_start|>' + message.role + '\\n' + content }}".to_owned(),
        "{%- if enable_thinking is defined and enable_thinking is false %}".to_owned(),
        "{{- '<think>\\n' }}".to_owned(),
        "{{- '<|im_start|>assistant\\n' }}".to_owned(),
        "{{- '<think>\\n\\n</think>\\n\\n' }}".to_owned(),
    ];
    match fragments
        .into_iter()
        .find(|fragment| !normalized.contains(fragment.as_str()))
    {
        Some(fragment) => Err(TemplateSourceError { fragment }),
        None => Ok(()),
    }
}
