//! The reply of a finished generation: what the accumulated answer becomes on each surface. Pure functions of the
//! plan and the answer; nothing here touches a connection, a slot or a clock.

use euhedral_inference_api_protocol::anthropic::{ContentBlock, Message, Usage as AnthropicUsage};
use euhedral_inference_api_protocol::json::JsonValue;
use euhedral_inference_api_protocol::openai::Usage as ChatUsage;
use euhedral_inference_api_protocol::openai::{ChatCompletionResponse, Completion, ToolCall as ChatToolCall};
use euhedral_inference_api_protocol::pipeline::{Piece, ToolCall};
use euhedral_inference_api_protocol::responses::{
    FunctionCallItem, IncompleteDetails, MessageItem, OutputItem, OutputText, ReasoningItem, ReasoningText, Response,
    Usage as ResponsesUsage,
};
use euhedral_inference_api_protocol::responses_mapper::encrypt;
use euhedral_inference_api_protocol::{ApiError, ToJson};
use sha2::{Digest, Sha256};

use crate::backend::GenerationResult;
use crate::metrics::FinishReason;
use crate::planning::{Endpoint, Extras, Planned, new_id};

/// A generation that ended with an answer to give: what the pipeline settled and how the backend ended.
#[derive(Debug)]
pub struct Completed {
    /// The settled pieces, in generation order.
    pub parts: Vec<Piece>,
    pub reason: FinishReason,
    /// The stop string that matched, when one did.
    pub stop_sequence: Option<String>,
    pub result: GenerationResult,
    /// Tool calls that went out in the stream as they came, so that `parts` does not hold them.
    pub streamed_calls: usize,
}

impl Completed {
    fn reasoning(&self) -> String {
        self.parts
            .iter()
            .filter_map(|part| match part {
                Piece::Reasoning(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    fn text(&self) -> String {
        self.parts
            .iter()
            .filter_map(|part| match part {
                Piece::Text(text) => Some(text.as_str()),
                _ => None,
            })
            .collect()
    }

    /// The number of tool calls in the answer.
    pub fn calls(&self) -> usize {
        self.streamed_calls
            + self
                .parts
                .iter()
                .filter(|part| matches!(part, Piece::ToolCall(_)))
                .count()
    }
}

pub(super) fn count(value: usize) -> i32 {
    i32::try_from(value).unwrap_or(i32::MAX)
}

/// The body of the reply to a finished generation, or the error that replaces it: a structured answer that is not
/// one JSON document.
pub fn body(planned: &Planned, model: &str, done: &Completed) -> Result<Vec<u8>, ApiError> {
    if planned.request.format.is_json() && done.result.stop_token_reached && done.calls() == 0 {
        let answer = done.text();
        if !answer.trim().is_empty() && JsonValue::parse_distinct(&answer).is_none() {
            return Err(ApiError::invalid_structured_output());
        }
    }
    let text = match planned.endpoint {
        Endpoint::ChatCompletions => chat(planned, model, done).to_json_string(),
        Endpoint::Messages => message(planned, model, done).to_json_string(),
        Endpoint::Responses => response(planned, model, done).to_json_string(),
    };
    Ok(text.into_bytes())
}

fn chat(planned: &Planned, model: &str, done: &Completed) -> ChatCompletionResponse {
    let result = &done.result;
    let tool_calls = done
        .parts
        .iter()
        .filter_map(|part| match part {
            Piece::ToolCall(call) => Some(ChatToolCall::function(
                format!("call_{}", new_id()),
                call.name.clone(),
                call.arguments.clone(),
            )),
            _ => None,
        })
        .collect();
    let ids = |tokens: &[u32]| tokens.iter().map(|id| i32::try_from(*id).unwrap_or(i32::MAX)).collect();
    let finish_reason = match done.reason {
        FinishReason::Length => "length",
        FinishReason::ToolCalls => "tool_calls",
        FinishReason::End | FinishReason::StopSequence => "stop",
    };
    ChatCompletionResponse::of(Completion {
        id: format!("chatcmpl-{}", planned.id),
        created: i64::try_from(planned.created).unwrap_or(i64::MAX),
        model: model.to_owned(),
        reasoning: done.reasoning(),
        content: done.text(),
        tool_calls,
        finish_reason: finish_reason.to_owned(),
        usage: ChatUsage::of(
            count(planned.prompt.token_count()),
            count(result.cached_prompt_tokens),
            count(result.completion_tokens),
            planned.output.reasoning.then(|| count(result.reasoning_tokens)),
        ),
        prompt_token_ids: planned.request.return_token_ids.then(|| ids(&planned.prompt.token_ids)),
        token_ids: planned.request.return_token_ids.then(|| ids(&result.token_ids)),
    })
}

/// The input of a tool-use block: the call's arguments read back, an unpaired surrogate included.
fn tool_input(arguments: &str) -> JsonValue {
    JsonValue::parse_keeping_unpaired(arguments).unwrap_or(JsonValue::Object(Vec::new()))
}

/// The opaque token naming the thinking it accompanies.
pub(super) fn signature(thinking: &str) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let digest = Sha256::digest(thinking.as_bytes());
    let mut token = String::from("euhedral-");
    for chunk in digest.chunks(3) {
        let bits = chunk.iter().fold(0u32, |all, byte| (all << 8) | u32::from(*byte)) << (8 * (3 - chunk.len()));
        for position in 0..=chunk.len() {
            let index = usize::try_from((bits >> (18 - 6 * position)) & 63).unwrap_or(0);
            token.push(char::from(ALPHABET[index]));
        }
    }
    token
}

fn message(planned: &Planned, model: &str, done: &Completed) -> Message {
    let result = &done.result;
    let mut content = Vec::new();
    let thinking = done.reasoning();
    if !thinking.is_empty() {
        content.push(ContentBlock::Thinking {
            signature: signature(&thinking),
            thinking,
        });
    }
    let text = done.text();
    if !text.is_empty() {
        content.push(ContentBlock::Text { text });
    }
    for part in &done.parts {
        if let Piece::ToolCall(ToolCall { name, arguments }) = part {
            content.push(ContentBlock::ToolUse {
                id: format!("toolu_{}", new_id()),
                name: name.clone(),
                input: tool_input(arguments),
            });
        }
    }
    let mut message = Message::new(
        format!("msg_{}", planned.id),
        model.to_owned(),
        content,
        AnthropicUsage::of(
            count(planned.prompt.token_count()),
            count(result.cached_prompt_tokens),
            count(result.completion_tokens),
        ),
    );
    message.stop_reason = Some(
        match done.reason {
            FinishReason::End => "end_turn",
            FinishReason::StopSequence => "stop_sequence",
            FinishReason::Length => "max_tokens",
            FinishReason::ToolCalls => "tool_use",
        }
        .to_owned(),
    );
    message.stop_sequence = done.stop_sequence.clone();
    message
}

/// The item of the Responses output that is still taking text.
enum Open {
    Reasoning(String),
    Message(String),
}

fn close(open: Option<Open>, encrypted: bool, items: &mut Vec<OutputItem>) {
    match open {
        Some(Open::Reasoning(text)) => {
            let mut item = ReasoningItem::new(format!("rs_{}", new_id()));
            item.encrypted_content = encrypted.then(|| encrypt(&text));
            item.content = vec![ReasoningText::new(text)];
            items.push(OutputItem::Reasoning(item));
        }
        Some(Open::Message(text)) => {
            let mut item = MessageItem::new(format!("msg_{}", new_id()));
            "completed".clone_into(&mut item.status);
            item.content = vec![OutputText::new(text)];
            items.push(OutputItem::Message(item));
        }
        None => {}
    }
}

/// Whether the reasoning is returned encrypted, and the request settings a Responses reply repeats.
pub(super) fn responses_extras(planned: &Planned) -> (bool, Vec<(String, JsonValue)>) {
    match &planned.extras {
        Extras::Responses {
            include_encrypted_content,
            echo: JsonValue::Object(members),
        } => (*include_encrypted_content, members.clone()),
        Extras::Responses {
            include_encrypted_content,
            ..
        } => (*include_encrypted_content, Vec::new()),
        _ => (false, Vec::new()),
    }
}

/// The message a stream opens with: no content yet, and the prompt tokens split by what the prefix cache restored.
pub(super) fn opening_message(id: &str, model: &str, prompt_tokens: i32, cached: i32) -> Message {
    Message::new(
        id.to_owned(),
        model.to_owned(),
        Vec::new(),
        AnthropicUsage::of(prompt_tokens, cached, 0),
    )
}

fn response(planned: &Planned, model: &str, done: &Completed) -> Response {
    let result = &done.result;
    let (encrypted, echo) = responses_extras(planned);
    let mut items = Vec::new();
    let mut open: Option<Open> = None;
    for part in &done.parts {
        match part {
            Piece::Reasoning(text) => match &mut open {
                Some(Open::Reasoning(held) | Open::Message(held)) => held.push_str(text),
                None => open = Some(Open::Reasoning(text.clone())),
            },
            Piece::Text(text) => {
                if matches!(open, Some(Open::Reasoning(_))) {
                    close(open.take(), encrypted, &mut items);
                }
                match &mut open {
                    Some(Open::Reasoning(held) | Open::Message(held)) => held.push_str(text),
                    None => open = Some(Open::Message(text.clone())),
                }
            }
            Piece::ToolCall(call) => {
                close(open.take(), encrypted, &mut items);
                items.push(OutputItem::FunctionCall(FunctionCallItem::new(
                    format!("fc_{}", new_id()),
                    "completed",
                    format!("call_{}", new_id()),
                    call.name.clone(),
                    call.arguments.clone(),
                )));
            }
        }
    }
    close(open.take(), encrypted, &mut items);
    let length = done.reason == FinishReason::Length;
    let mut response = Response::new(
        format!("resp_{}", planned.id),
        i64::try_from(planned.created).unwrap_or(i64::MAX),
        if length { "incomplete" } else { "completed" },
        model,
        items,
        echo,
    );
    if length {
        response.incomplete_details = Some(IncompleteDetails {
            reason: "max_output_tokens".to_owned(),
        });
    }
    response.usage = Some(ResponsesUsage::of(
        count(planned.prompt.token_count()),
        count(result.cached_prompt_tokens),
        count(result.completion_tokens),
        planned.output.reasoning.then(|| count(result.reasoning_tokens)),
    ));
    response
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tool_input_keeps_an_unpaired_surrogate_as_its_escape() {
        let input = tool_input(r#"{"a":["\uD800x","\\uD800"],"b":1}"#);
        assert_eq!(input.to_compact_string(), r#"{"a":["\uD800x","\\uD800"],"b":1}"#);
        assert_eq!(tool_input("not json"), JsonValue::Object(Vec::new()));
    }

    #[test]
    fn a_tool_input_keeps_an_unpaired_surrogate_in_a_key_as_its_escape() {
        let input = tool_input(r#"{"v":{"\uD800":1,"a":2}}"#);
        assert_eq!(input.to_compact_string(), r#"{"v":{"\uD800":1,"a":2}}"#);
    }

    #[test]
    fn a_signature_is_the_url_safe_unpadded_digest() {
        // SHA-256 of the empty text is e3b0c442 98fc1c14 9afbf4c8 996fb924 27ae41e4 649b934c a495991b 7852b855.
        assert_eq!(signature(""), "euhedral-47DEQpj8HBSa-_TImW-5JCeuQeRkm5NMpJWZG3hSuFU");
    }
}
