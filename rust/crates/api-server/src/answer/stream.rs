//! The events of a streamed answer: what the pieces of the generated text become on each surface, as the bytes of a
//! server-sent event stream. The encoder is plain state of the owner; it appends to a buffer and touches nothing else.
//!
//! A data event is `data:<json>`, a named one is `event:<name>` and then `data:<json>`, a comment is `:<text>`; each
//! ends with an empty line, and nothing follows a colon but the value.

use std::mem::take;

use euhedral_inference_api_protocol::anthropic::{
    BlockDelta, ContentBlock, MessageDeltaBody, StreamEvent, Usage as AnthropicUsage,
};
use euhedral_inference_api_protocol::json::JsonValue;
use euhedral_inference_api_protocol::openai::{
    ChatCompletionChunk, STREAM_DONE, ToolCall as ChatToolCall, Usage as ChatUsage,
};
use euhedral_inference_api_protocol::pipeline::Piece;
use euhedral_inference_api_protocol::responses::{
    ContentPart, EventBody, FunctionCallItem, IncompleteDetails, MessageItem, OutputItem, OutputText, ReasoningItem,
    ReasoningText, Response, ResponseEvent, Usage as ResponsesUsage,
};
use euhedral_inference_api_protocol::responses_mapper::encrypt;
use euhedral_inference_api_protocol::{ApiError, ToJson};
use euhedral_net::Response as Head;

use super::reply::{count, opening_message, responses_extras, signature};
use crate::backend::GenerationResult;
use crate::metrics::FinishReason;
use crate::planning::{Endpoint, Extras, Planned, new_id};

/// The comment a surface that has no ping event writes while the prompt is prefilled.
const KEEP_ALIVE: &str = "keep-alive";

fn data(out: &mut Vec<u8>, json: &str) {
    out.extend_from_slice(b"data:");
    out.extend_from_slice(json.as_bytes());
    out.extend_from_slice(b"\n\n");
}

fn named(out: &mut Vec<u8>, name: &str, json: &str) {
    out.extend_from_slice(b"event:");
    out.extend_from_slice(name.as_bytes());
    out.extend_from_slice(b"\ndata:");
    out.extend_from_slice(json.as_bytes());
    out.extend_from_slice(b"\n\n");
}

fn comment(out: &mut Vec<u8>, text: &str) {
    out.push(b':');
    out.extend_from_slice(text.as_bytes());
    out.extend_from_slice(b"\n\n");
}

/// The head of a stream: the status and the headers, with no length; the transport frames the body.
pub(super) fn head() -> Head {
    Head::with_status(200)
        .header("Content-Type", "text/event-stream;charset=UTF-8")
        .header("Cache-Control", "no-cache")
        .header("X-Accel-Buffering", "no")
}

/// The block of an Anthropic stream that is taking text.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Block {
    Thinking,
    Text,
}

/// The item of a Responses stream that is taking text.
struct Item {
    id: String,
    reasoning: bool,
    text: String,
}

/// One request's events. Pieces go in as the pipeline settles them and bytes come out.
pub(super) struct Encoder {
    endpoint: Endpoint,
    id: String,
    created: i64,
    model: String,
    prompt_tokens: i32,
    reasoning_usage: bool,
    include_usage: bool,
    encrypted: bool,
    echo: Vec<(String, JsonValue)>,
    started: bool,
    sequence: u64,
    calls: usize,
    /// Anthropic: the next block's index and the open block.
    next_block: i32,
    block: Option<Block>,
    thinking: String,
    /// Responses: the finished items and the open one.
    items: Vec<OutputItem>,
    item: Option<Item>,
    /// The answer text, kept only to check a structured answer.
    answer: Option<String>,
}

impl Encoder {
    pub(super) fn new(planned: &Planned, model: &str) -> Encoder {
        let (encrypted, echo) = responses_extras(planned);
        let include_usage = matches!(
            planned.extras,
            Extras::Chat {
                include_usage: true,
                ..
            }
        );
        let created = i64::try_from(planned.created).unwrap_or(i64::MAX);
        let id = match planned.endpoint {
            Endpoint::ChatCompletions => format!("chatcmpl-{}", planned.id),
            Endpoint::Messages => format!("msg_{}", planned.id),
            Endpoint::Responses => format!("resp_{}", planned.id),
        };
        Encoder {
            endpoint: planned.endpoint,
            id,
            created,
            model: model.to_owned(),
            prompt_tokens: count(planned.prompt.token_count()),
            reasoning_usage: planned.output.reasoning,
            include_usage,
            encrypted,
            echo,
            started: false,
            sequence: 0,
            calls: 0,
            next_block: 0,
            block: None,
            thinking: String::new(),
            items: Vec::new(),
            item: None,
            answer: planned.request.format.is_json().then(String::new),
        }
    }

    /// Whether the opening events were written.
    pub(super) fn started(&self) -> bool {
        self.started
    }

    /// The tool calls so far.
    pub(super) fn calls(&self) -> usize {
        self.calls
    }

    /// The answer text, when a structured answer is being kept.
    pub(super) fn answer(&self) -> Option<&str> {
        self.answer.as_deref()
    }

    /// What is written while the prompt is prefilled and nothing else was: a ping or a comment.
    pub(super) fn keep_alive(&mut self, out: &mut Vec<u8>) {
        match self.endpoint {
            Endpoint::Messages => Self::event(out, &StreamEvent::Ping),
            Endpoint::ChatCompletions | Endpoint::Responses => comment(out, KEEP_ALIVE),
        }
    }

    fn event(out: &mut Vec<u8>, event: &StreamEvent) {
        named(out, event.name(), &event.to_json_string());
    }

    fn chunk(out: &mut Vec<u8>, chunk: &ChatCompletionChunk) {
        data(out, &chunk.to_json_string());
    }

    fn sent(&mut self, out: &mut Vec<u8>, kind: &str, body: EventBody) {
        let event = ResponseEvent::new(kind, self.sequence, body);
        self.sequence += 1;
        named(out, kind, &event.to_json_string());
    }

    fn response(&self, status: &str, output: Vec<OutputItem>) -> Response {
        Response::new(
            self.id.clone(),
            self.created,
            status,
            self.model.clone(),
            output,
            self.echo.clone(),
        )
    }

    /// The opening events, once; `cached` prompt tokens were restored from the prefix cache.
    pub(super) fn start(&mut self, cached: usize, out: &mut Vec<u8>) {
        if self.started {
            return;
        }
        self.started = true;
        match self.endpoint {
            Endpoint::ChatCompletions => {
                Self::chunk(out, &ChatCompletionChunk::role(&self.id, self.created, &self.model));
            }
            Endpoint::Messages => {
                let message = opening_message(&self.id, &self.model, self.prompt_tokens, count(cached));
                Self::event(out, &StreamEvent::MessageStart { message });
            }
            Endpoint::Responses => {
                for kind in ["response.created", "response.in_progress"] {
                    let response = self.response("in_progress", Vec::new());
                    self.sent(out, kind, EventBody::Response { response });
                }
            }
        }
    }

    /// The events of one settled piece.
    pub(super) fn piece(&mut self, piece: &Piece, out: &mut Vec<u8>) {
        match piece {
            Piece::Reasoning(text) if !text.is_empty() => self.reasoning(text, out),
            Piece::Text(text) if !text.is_empty() => {
                if let Some(answer) = &mut self.answer {
                    answer.push_str(text);
                }
                self.text(text, out);
            }
            Piece::Reasoning(_) | Piece::Text(_) => {}
            Piece::ToolCall(call) => {
                self.call(&call.name, &call.arguments, out);
                self.calls += 1;
            }
        }
    }

    fn reasoning(&mut self, text: &str, out: &mut Vec<u8>) {
        match self.endpoint {
            Endpoint::ChatCompletions => {
                Self::chunk(
                    out,
                    &ChatCompletionChunk::reasoning(&self.id, self.created, &self.model, text),
                );
            }
            Endpoint::Messages => {
                if self.block != Some(Block::Thinking) {
                    self.close_block(out);
                    let content_block = ContentBlock::Thinking {
                        thinking: String::new(),
                        signature: String::new(),
                    };
                    self.open_block(Block::Thinking, content_block, out);
                }
                self.thinking.push_str(text);
                let delta = BlockDelta::ThinkingDelta {
                    thinking: text.to_owned(),
                };
                Self::event(
                    out,
                    &StreamEvent::ContentBlockDelta {
                        index: self.next_block - 1,
                        delta,
                    },
                );
            }
            Endpoint::Responses => {
                if self.item.is_none() {
                    self.open_item(true, out);
                }
                self.item_delta("response.reasoning_text.delta", text, out);
            }
        }
    }

    fn text(&mut self, text: &str, out: &mut Vec<u8>) {
        match self.endpoint {
            Endpoint::ChatCompletions => {
                Self::chunk(
                    out,
                    &ChatCompletionChunk::content(&self.id, self.created, &self.model, text),
                );
            }
            Endpoint::Messages => {
                if self.block != Some(Block::Text) {
                    self.close_block(out);
                    let content_block = ContentBlock::Text { text: String::new() };
                    self.open_block(Block::Text, content_block, out);
                }
                let delta = BlockDelta::TextDelta { text: text.to_owned() };
                Self::event(
                    out,
                    &StreamEvent::ContentBlockDelta {
                        index: self.next_block - 1,
                        delta,
                    },
                );
            }
            Endpoint::Responses => {
                if self.item.as_ref().is_some_and(|item| item.reasoning) {
                    self.close_item(out);
                }
                if self.item.is_none() {
                    self.open_item(false, out);
                }
                self.item_delta("response.output_text.delta", text, out);
            }
        }
    }

    fn call(&mut self, name: &str, arguments: &str, out: &mut Vec<u8>) {
        match self.endpoint {
            Endpoint::ChatCompletions => {
                let call = ChatToolCall::function(format!("call_{}", new_id()), name, arguments);
                let index = i32::try_from(self.calls).unwrap_or(i32::MAX);
                Self::chunk(
                    out,
                    &ChatCompletionChunk::tool_call(&self.id, self.created, &self.model, index, call),
                );
            }
            Endpoint::Messages => {
                self.close_block(out);
                let index = self.next_block;
                self.next_block += 1;
                let content_block = ContentBlock::ToolUse {
                    id: format!("toolu_{}", new_id()),
                    name: name.to_owned(),
                    input: JsonValue::Object(Vec::new()),
                };
                Self::event(out, &StreamEvent::ContentBlockStart { index, content_block });
                let delta = BlockDelta::InputJsonDelta {
                    partial_json: arguments.to_owned(),
                };
                Self::event(out, &StreamEvent::ContentBlockDelta { index, delta });
                Self::event(out, &StreamEvent::ContentBlockStop { index });
            }
            Endpoint::Responses => {
                self.close_item(out);
                let output_index = i32::try_from(self.items.len()).unwrap_or(i32::MAX);
                let id = format!("fc_{}", new_id());
                let call_id = format!("call_{}", new_id());
                let added = FunctionCallItem::new(id.clone(), "in_progress", call_id.clone(), name, "");
                self.sent(
                    out,
                    "response.output_item.added",
                    EventBody::OutputItem {
                        output_index,
                        item: OutputItem::FunctionCall(added),
                    },
                );
                self.sent(
                    out,
                    "response.function_call_arguments.delta",
                    EventBody::ArgumentsDelta {
                        item_id: id.clone(),
                        output_index,
                        delta: arguments.to_owned(),
                    },
                );
                self.sent(
                    out,
                    "response.function_call_arguments.done",
                    EventBody::ArgumentsDone {
                        item_id: id.clone(),
                        output_index,
                        arguments: arguments.to_owned(),
                    },
                );
                let done = OutputItem::FunctionCall(FunctionCallItem::new(id, "completed", call_id, name, arguments));
                self.items.push(done.clone());
                self.sent(
                    out,
                    "response.output_item.done",
                    EventBody::OutputItem {
                        output_index,
                        item: done,
                    },
                );
            }
        }
    }

    fn open_block(&mut self, block: Block, content_block: ContentBlock, out: &mut Vec<u8>) {
        let index = self.next_block;
        self.next_block += 1;
        self.block = Some(block);
        Self::event(out, &StreamEvent::ContentBlockStart { index, content_block });
    }

    /// Closes the open block of an Anthropic stream; a thinking block is signed first.
    fn close_block(&mut self, out: &mut Vec<u8>) {
        let Some(block) = self.block.take() else {
            return;
        };
        let index = self.next_block - 1;
        if block == Block::Thinking {
            let delta = BlockDelta::SignatureDelta {
                signature: signature(&self.thinking),
            };
            Self::event(out, &StreamEvent::ContentBlockDelta { index, delta });
            self.thinking.clear();
        }
        Self::event(out, &StreamEvent::ContentBlockStop { index });
    }

    fn open_item(&mut self, reasoning: bool, out: &mut Vec<u8>) {
        let output_index = i32::try_from(self.items.len()).unwrap_or(i32::MAX);
        let (id, item, part) = if reasoning {
            let id = format!("rs_{}", new_id());
            (
                id.clone(),
                OutputItem::Reasoning(ReasoningItem::new(id)),
                ContentPart::ReasoningText(ReasoningText::new("")),
            )
        } else {
            let id = format!("msg_{}", new_id());
            (
                id.clone(),
                OutputItem::Message(MessageItem::new(id)),
                ContentPart::OutputText(OutputText::new("")),
            )
        };
        self.sent(
            out,
            "response.output_item.added",
            EventBody::OutputItem { output_index, item },
        );
        self.sent(
            out,
            "response.content_part.added",
            EventBody::ContentPart {
                item_id: id.clone(),
                output_index,
                content_index: 0,
                part,
            },
        );
        self.item = Some(Item {
            id,
            reasoning,
            text: String::new(),
        });
    }

    fn item_delta(&mut self, kind: &str, delta: &str, out: &mut Vec<u8>) {
        let output_index = i32::try_from(self.items.len()).unwrap_or(i32::MAX);
        let Some(item) = &mut self.item else {
            return;
        };
        item.text.push_str(delta);
        let body = EventBody::TextDelta {
            item_id: item.id.clone(),
            output_index,
            content_index: 0,
            delta: delta.to_owned(),
        };
        self.sent(out, kind, body);
    }

    fn close_item(&mut self, out: &mut Vec<u8>) {
        let Some(Item { id, reasoning, text }) = self.item.take() else {
            return;
        };
        let output_index = i32::try_from(self.items.len()).unwrap_or(i32::MAX);
        let (done_kind, part, item) = if reasoning {
            let mut item = ReasoningItem::new(id.clone());
            item.encrypted_content = self.encrypted.then(|| encrypt(&text));
            item.content = vec![ReasoningText::new(text.clone())];
            (
                "response.reasoning_text.done",
                ContentPart::ReasoningText(ReasoningText::new(text.clone())),
                OutputItem::Reasoning(item),
            )
        } else {
            let mut item = MessageItem::new(id.clone());
            "completed".clone_into(&mut item.status);
            item.content = vec![OutputText::new(text.clone())];
            (
                "response.output_text.done",
                ContentPart::OutputText(OutputText::new(text.clone())),
                OutputItem::Message(item),
            )
        };
        self.sent(
            out,
            done_kind,
            EventBody::TextDone {
                item_id: id.clone(),
                output_index,
                content_index: 0,
                text,
            },
        );
        self.sent(
            out,
            "response.content_part.done",
            EventBody::ContentPart {
                item_id: id,
                output_index,
                content_index: 0,
                part,
            },
        );
        self.items.push(item.clone());
        self.sent(
            out,
            "response.output_item.done",
            EventBody::OutputItem { output_index, item },
        );
    }

    /// The closing events of an answer that ended as it should, with the usage the generation reports.
    pub(super) fn finish(
        &mut self,
        reason: FinishReason,
        stop_sequence: Option<&str>,
        result: &GenerationResult,
        out: &mut Vec<u8>,
    ) {
        let cached = count(result.cached_prompt_tokens);
        let completion = count(result.completion_tokens);
        match self.endpoint {
            Endpoint::ChatCompletions => {
                let finish_reason = match reason {
                    FinishReason::Length => "length",
                    FinishReason::ToolCalls => "tool_calls",
                    FinishReason::End | FinishReason::StopSequence => "stop",
                };
                Self::chunk(
                    out,
                    &ChatCompletionChunk::finish(&self.id, self.created, &self.model, finish_reason),
                );
                if self.include_usage {
                    let usage = ChatUsage::of(
                        self.prompt_tokens,
                        cached,
                        completion,
                        self.reasoning_usage.then(|| count(result.reasoning_tokens)),
                    );
                    Self::chunk(
                        out,
                        &ChatCompletionChunk::usage(&self.id, self.created, &self.model, usage),
                    );
                }
                data(out, STREAM_DONE);
            }
            Endpoint::Messages => {
                self.close_block(out);
                let stop_reason = match reason {
                    FinishReason::End => "end_turn",
                    FinishReason::StopSequence => "stop_sequence",
                    FinishReason::Length => "max_tokens",
                    FinishReason::ToolCalls => "tool_use",
                };
                Self::event(
                    out,
                    &StreamEvent::MessageDelta {
                        delta: MessageDeltaBody {
                            stop_reason: Some(stop_reason.to_owned()),
                            stop_sequence: stop_sequence.map(str::to_owned),
                        },
                        usage: AnthropicUsage::of(self.prompt_tokens, cached, completion),
                    },
                );
                Self::event(out, &StreamEvent::MessageStop);
            }
            Endpoint::Responses => {
                self.close_item(out);
                let length = reason == FinishReason::Length;
                let status = if length { "incomplete" } else { "completed" };
                let items = take(&mut self.items);
                let mut response = self.response(status, items);
                if length {
                    response.incomplete_details = Some(IncompleteDetails {
                        reason: "max_output_tokens".to_owned(),
                    });
                }
                response.usage = Some(ResponsesUsage::of(
                    self.prompt_tokens,
                    cached,
                    completion,
                    self.reasoning_usage.then(|| count(result.reasoning_tokens)),
                ));
                let kind = format!("response.{status}");
                self.sent(out, &kind, EventBody::Response { response });
            }
        }
    }

    /// The in-stream error of a failure after the head went out.
    pub(super) fn error(&mut self, error: &ApiError, out: &mut Vec<u8>) {
        match self.endpoint {
            Endpoint::ChatCompletions => data(out, &error.openai_envelope().to_json_string()),
            Endpoint::Messages => Self::event(
                out,
                &StreamEvent::Error {
                    error: error.anthropic_envelope().error,
                },
            ),
            Endpoint::Responses => self.sent(out, "error", EventBody::error(error)),
        }
    }
}
