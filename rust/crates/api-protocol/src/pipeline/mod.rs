//! The text pipeline of one generation: the decoded text, as it arrives in pieces of any size, becomes reasoning,
//! answer text and tool calls. Three stages run in order: [`ReasoningSplitter`] separates the think block, then
//! [`StopSequenceFilter`] applies the request's `stop` strings to the answer only, then [`ToolCallParser`] reads
//! the answer's `<tool_call>` blocks. Each stage is usable alone; [`TextPipeline`] composes them.

mod function_tool;
mod json_value;
pub mod python_json;
mod reasoning_splitter;
mod stop_sequence_filter;
mod text_space;
mod tool_call_parser;

#[cfg(test)]
mod tests;

pub use function_tool::FunctionTool;
pub use json_value::JsonValue;
pub use reasoning_splitter::{ReasoningSplitter, Split, THINK_END};
pub use stop_sequence_filter::StopSequenceFilter;
pub use tool_call_parser::{MalformedToolCall, ToolCall, ToolCallParser, ToolEvent};

use thiserror::Error;

/// One settled part of the output, in generation order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Piece {
    /// Text of the think block.
    Reasoning(String),
    /// Assistant answer text.
    Text(String),
    /// A complete tool call.
    ToolCall(ToolCall),
}

/// How the model may call the request's functions.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ToolChoice {
    /// The definitions are not offered: the output is plain text.
    None,
    /// Sampling is free; calls are split from the text.
    Auto,
    /// Sampling is constrained to calls; an answer without one is malformed.
    Required,
    /// As `Required`, restricted to the named function.
    Function(String),
}

/// The request's tools and how they may be called.
#[derive(Clone, Debug, PartialEq)]
pub struct ToolOptions {
    pub tools: Vec<FunctionTool>,
    pub choice: ToolChoice,
    /// False ends the response after the first complete call.
    pub parallel: bool,
}

impl ToolOptions {
    /// Functions the model may call in this response.
    fn callable(&self) -> Vec<FunctionTool> {
        match &self.choice {
            ToolChoice::None => Vec::new(),
            ToolChoice::Function(name) => self.tools.iter().filter(|tool| tool.name == *name).cloned().collect(),
            ToolChoice::Auto | ToolChoice::Required => self.tools.clone(),
        }
    }
}

/// What the pipeline does for one request.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct PipelineOptions {
    /// The output opens inside the think block and is split at `</think>`.
    pub reasoning: bool,
    /// The request's `stop` strings.
    pub stops: Vec<String>,
    /// The request's tools; `None` when it offers none, which leaves the answer as plain text.
    pub tools: Option<ToolOptions>,
}

/// The model's output cannot be accepted.
#[derive(Debug, Error)]
pub enum PipelineError {
    /// The output claims to be a tool call but is not one the request can accept. `emitted` holds the pieces
    /// settled by the same call before the failure was found; the pipeline accepts no text after it.
    #[error("{message}")]
    MalformedToolCall { message: String, emitted: Vec<Piece> },
}

impl PipelineError {
    /// The message of the failure.
    pub fn message(&self) -> &str {
        match self {
            PipelineError::MalformedToolCall { message, .. } => message,
        }
    }
}

/// Turns the decoded text of one generation into pieces. Text may be cut anywhere between characters: the pieces
/// concatenate to the same reasoning, text and calls however it is cut. Text that could still be part of a marker,
/// a stop string or a call is held until it is decided, and [`finish`](Self::finish) releases what remains.
#[derive(Debug)]
pub struct TextPipeline {
    splitter: Option<ReasoningSplitter>,
    stop: StopSequenceFilter,
    parser: Option<ToolCallParser>,
    parallel: bool,
    failure: Option<String>,
    call_limit_reached: bool,
    finished: bool,
}

impl TextPipeline {
    pub fn new(options: &PipelineOptions) -> Self {
        let parser = options
            .tools
            .as_ref()
            .filter(|tools| tools.choice != ToolChoice::None)
            .map(|tools| {
                let requires_call = matches!(tools.choice, ToolChoice::Required | ToolChoice::Function(_));
                ToolCallParser::new(tools.callable(), tools.parallel, requires_call)
            });
        Self {
            splitter: options.reasoning.then(ReasoningSplitter::new),
            stop: StopSequenceFilter::new(&options.stops),
            parser,
            parallel: options.tools.as_ref().is_none_or(|tools| tools.parallel),
            failure: None,
            call_limit_reached: false,
            finished: false,
        }
    }

    /// Takes the next text and returns the pieces it settles. After a stop sequence matched or a call failed, text
    /// is ignored and nothing is returned.
    pub fn push(&mut self, text: &str) -> Result<Vec<Piece>, PipelineError> {
        let mut pieces = Vec::new();
        if self.failure.is_some() || self.stop.matched() {
            return Ok(pieces);
        }
        match self.push_into(text, &mut pieces) {
            Ok(()) => Ok(pieces),
            Err(failure) => Err(self.fail(failure, pieces)),
        }
    }

    /// The generation ended at its stop token: releases what is held. Unfinished reasoning is released trimmed;
    /// an unfinished call, or no call under a required choice, is malformed.
    pub fn finish(&mut self) -> Result<Vec<Piece>, PipelineError> {
        self.finish_with(true)
    }

    /// The generation ended because its token budget ran out: as [`finish`](Self::finish), except that an
    /// unfinished call is dropped and a required call that never came is not an error.
    pub fn finish_truncated(&mut self) -> Result<Vec<Piece>, PipelineError> {
        self.finish_with(false)
    }

    /// True once `</think>` has been generated; false before it, and always without reasoning.
    pub fn reasoning_closed(&self) -> bool {
        self.splitter.as_ref().is_some_and(ReasoningSplitter::answering)
    }

    /// True while the output is inside a think block that has not been closed.
    pub fn reasoning_open(&self) -> bool {
        self.splitter.as_ref().is_some_and(|splitter| !splitter.answering())
    }

    /// True once a stop sequence matched; the rest of the generation is discarded.
    pub fn stop_matched(&self) -> bool {
        self.stop.matched()
    }

    /// The stop string that matched.
    pub fn matched_stop(&self) -> Option<&str> {
        self.stop.matched_stop()
    }

    /// The number of complete tool calls so far.
    pub fn tool_calls(&self) -> usize {
        self.parser.as_ref().map_or(0, ToolCallParser::calls)
    }

    /// True once the one call a request without parallel calls allows is complete.
    pub fn call_limit_reached(&self) -> bool {
        self.call_limit_reached
    }

    /// True when decoding can end: a stop sequence matched, the call limit was reached or a call failed.
    pub fn ended(&self) -> bool {
        self.stop.matched() || self.call_limit_reached || self.failure.is_some()
    }

    fn fail(&mut self, failure: MalformedToolCall, emitted: Vec<Piece>) -> PipelineError {
        self.failure = Some(failure.0.clone());
        PipelineError::MalformedToolCall {
            message: failure.0,
            emitted,
        }
    }

    fn push_into(&mut self, text: &str, pieces: &mut Vec<Piece>) -> Result<(), MalformedToolCall> {
        let Some(splitter) = self.splitter.as_mut() else {
            return self.answer(text, pieces);
        };
        let mut splits = Vec::new();
        splitter.accept(text, &mut splits);
        for split in splits {
            match split {
                Split::Reasoning(text) => pieces.push(Piece::Reasoning(text)),
                Split::Answer(text) => self.answer(&text, pieces)?,
            }
        }
        Ok(())
    }

    /// Answer text: the stop strings see it first, then the call parser.
    fn answer(&mut self, text: &str, pieces: &mut Vec<Piece>) -> Result<(), MalformedToolCall> {
        if self.failure.is_some() || self.stop.matched() {
            return Ok(());
        }
        let visible = self.stop.accept(text);
        self.route(&visible, pieces)
    }

    fn route(&mut self, visible: &str, pieces: &mut Vec<Piece>) -> Result<(), MalformedToolCall> {
        let Some(parser) = self.parser.as_mut() else {
            if !visible.is_empty() {
                pieces.push(Piece::Text(visible.to_string()));
            }
            return Ok(());
        };
        let mut events = Vec::new();
        let result = parser.accept(visible, &mut events);
        self.collect(events, pieces);
        result
    }

    fn collect(&mut self, events: Vec<ToolEvent>, pieces: &mut Vec<Piece>) {
        for event in events {
            match event {
                ToolEvent::Content(text) => pieces.push(Piece::Text(text)),
                ToolEvent::Call(call) => {
                    if !self.parallel {
                        self.call_limit_reached = true;
                    }
                    pieces.push(Piece::ToolCall(call));
                }
            }
        }
    }

    fn finish_with(&mut self, stop_token_reached: bool) -> Result<Vec<Piece>, PipelineError> {
        if let Some(message) = &self.failure {
            return Err(PipelineError::MalformedToolCall {
                message: message.clone(),
                emitted: Vec::new(),
            });
        }
        let mut pieces = Vec::new();
        if self.finished {
            return Ok(pieces);
        }
        self.finished = true;
        match self.finish_into(stop_token_reached, &mut pieces) {
            Ok(()) => Ok(pieces),
            Err(failure) => Err(self.fail(failure, pieces)),
        }
    }

    fn finish_into(&mut self, stop_token_reached: bool, pieces: &mut Vec<Piece>) -> Result<(), MalformedToolCall> {
        if let Some(splitter) = self.splitter.as_mut() {
            let mut splits = Vec::new();
            splitter.finish(&mut splits);
            for split in splits {
                if let Split::Reasoning(text) = split {
                    pieces.push(Piece::Reasoning(text));
                }
            }
        }
        if self.call_limit_reached {
            return Ok(());
        }
        let remaining = self.stop.finish();
        self.route(&remaining, pieces)?;
        let stop_token = stop_token_reached && !self.stop.matched();
        let Some(parser) = self.parser.as_mut() else {
            return Ok(());
        };
        let mut events = Vec::new();
        let result = parser.finish(stop_token, &mut events);
        self.collect(events, pieces);
        result
    }
}
