//! A request as every API surface states it once validated: what the rest of the server needs to generate.
//!
//! Each surface maps its own request format into a [`GenerationRequest`]; everything after it (rendering, grammar,
//! the generation, prefix reuse, the reply encoding) is shared.

use crate::response_format::ResponseFormat;
use crate::template::{TemplateInput, TemplateOptions};
use crate::tool_calling::ToolCalling;

/// The most stop sequences a request may carry.
pub const MAX_STOP_SEQUENCES: usize = 4;

/// What the server states for every request: the served model and the values a request that omits a field takes.
#[derive(Clone, Debug, PartialEq)]
pub struct Defaults {
    /// The one model the server answers to.
    pub model: String,
    pub temperature: f32,
    pub top_p: f32,
    /// Top-k; 0 is no limit.
    pub top_k: i32,
    /// The completion budget of a request that states none, before the context caps it.
    pub max_tokens: i32,
}

/// How tokens are drawn. Temperature or top-p of 0 is greedy, which also fixes `top_p` at 1.
#[derive(Clone, Debug, PartialEq)]
pub struct Sampling {
    pub temperature: f32,
    /// Top-k; 0 is no limit.
    pub top_k: i32,
    pub top_p: f32,
    /// The request's seed; `None` asks the caller to draw one.
    pub seed: Option<i64>,
    pub greedy: bool,
}

/// A validated request: the turns to render and how to answer them.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerationRequest {
    /// The model the request named.
    pub model: String,
    /// The messages, and the tools the prompt offers (none when the choice is `none`).
    pub template: TemplateInput,
    /// The thinking variables of the template.
    pub thinking: TemplateOptions,
    /// Caps the reasoning tokens; `None` is no cap.
    pub reasoning_budget: Option<u32>,
    pub tools: ToolCalling,
    pub format: ResponseFormat,
    pub stops: Vec<String>,
    pub sampling: Sampling,
    /// The requested completion budget; `None` takes the server's default. Checking it against the context is the
    /// planner's.
    pub max_tokens: Option<i32>,
    /// The request field that carries `max_tokens`, for errors.
    pub max_tokens_param: String,
    pub stream: bool,
    /// Asks for the prompt's and the completion's token ids.
    pub return_token_ids: bool,
    /// The streamed reply ends with a usage chunk (Chat Completions only).
    pub include_usage: bool,
}

impl GenerationRequest {
    /// True when the output opens inside the model's think block.
    pub fn reasoning(&self) -> bool {
        self.thinking.effort.is_some()
    }
}
