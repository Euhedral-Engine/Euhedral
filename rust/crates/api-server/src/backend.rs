//! What the generating routes need of the engine: the narrow view of the loaded model.
//!
//! Nothing here waits. A call that has to do work on the engine's workers returns at once and reports through a
//! continuation ([`Next`]) or a [`TextListener`]; both run on a worker of the engine's lattice, one call at a time and
//! in the order of the events, and must not block: hand the event on and return.
//!
//! A continuation or a listener's `finished` runs exactly once on every path: with the result when the work was done,
//! with an error when the work could not be offered to the lattice (it is closing or refused the frame), never
//! dropped unrun and never twice.

use std::error::Error;
use std::fmt;
use std::sync::Arc;

/// The rest of a request, run with the outcome of one piece of work. It runs exactly once, on a worker, and must not
/// wait; if the work cannot be offered to the lattice it runs with an error instead.
pub type Next<T> = Box<dyn FnOnce(T) + Send>;

/// Why a backend could not do what it was asked.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum BackendError {
    /// The engine is closing or closed; the text says why.
    Unavailable(String),
    /// A grammar or JSON Schema cannot be enforced while sampling; the text names the problem.
    Unenforceable(String),
    /// The generation failed; the text is for the log, never for a client.
    Failed(String),
}

impl fmt::Display for BackendError {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BackendError::Unavailable(reason) => write!(out, "the engine is unavailable: {reason}"),
            BackendError::Unenforceable(reason) => write!(out, "cannot be enforced: {reason}"),
            BackendError::Failed(reason) => write!(out, "the generation failed: {reason}"),
        }
    }
}

impl Error for BackendError {}

/// A rendered prompt and its token ids, encoded once, exactly as a new generation encodes a first prompt.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedPrompt {
    pub text: String,
    pub token_ids: Vec<u32>,
}

impl EncodedPrompt {
    /// The number of tokens.
    pub fn token_count(&self) -> usize {
        self.token_ids.len()
    }
}

/// The sampling of one generation. A temperature of zero or `greedy` selects the most likely token; a `top_k` of zero
/// and a `top_p` of one disable those filters.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GenerationConfig {
    pub temperature: f32,
    pub top_k: u32,
    pub top_p: f32,
    pub seed: u64,
    pub greedy: bool,
}

impl GenerationConfig {
    /// Greedy decoding.
    pub fn greedy(seed: u64) -> GenerationConfig {
        GenerationConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 1.0,
            seed,
            greedy: true,
        }
    }
}

/// How a generation's output is shaped. With `reasoning` the prompt opened the model's think block, so the output is
/// reasoning up to `</think>`, at most `reasoning_budget` tokens (`None`: no cap), and the answer after it. A
/// `grammar` (llguidance Lark) constrains the answer; a backend enforces both while sampling.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct OutputSpec {
    pub reasoning: bool,
    pub reasoning_budget: Option<u32>,
    pub grammar: Option<String>,
}

impl OutputSpec {
    /// Plain text: no reasoning, no grammar.
    pub fn text() -> OutputSpec {
        OutputSpec {
            reasoning: false,
            reasoning_budget: None,
            grammar: None,
        }
    }
}

/// How a generation ended. `completion_tokens` counts every sampled token, including a terminating stop token;
/// `reasoning_tokens` those before `</think>`; `cached_prompt_tokens` of the prompt were restored from the prefix cache
/// instead of prefilled. `token_ids` are the sampled tokens, terminator included (empty when a backend does not
/// report them).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct GenerationResult {
    pub completion_tokens: usize,
    pub stop_token_reached: bool,
    pub reasoning_tokens: usize,
    pub cached_prompt_tokens: usize,
    pub token_ids: Vec<u32>,
}

/// Where a generation's events go. Every method runs on a worker, one call at a time and in order, and must not wait.
pub trait TextListener: Send {
    /// Newly decoded, non-empty text.
    fn text(&mut self, text: &str);

    /// A prefill quantum retired, before the next is admitted.
    fn prefilled(&mut self) {}

    /// The generation ended: the last call, made after the last text, and made exactly once on every path (a refused
    /// start and a cancellation included).
    fn finished(&mut self, result: Result<GenerationResult, BackendError>);
}

/// One request's sequence state, opened by [`Backend::open_generation`]. The caller closes it on every path, after
/// the generation has ended.
pub trait Generation: Send + Sync {
    /// Starts prefill and decode on the workers and returns at once; `listener` hears the rest.
    fn generate(&self, prompt: EncodedPrompt, max_new_tokens: usize, listener: Box<dyn TextListener>);

    /// Asks the generation to stop at its next quantum. Callable from anywhere, the listener included.
    fn cancel(&self);

    /// Whether cancellation was requested by any party, engine shutdown included.
    fn is_cancelled(&self) -> bool;

    /// Prompt tokens restored from the prefix cache; known once the generation produced its first text.
    fn cached_prompt_tokens(&self) -> usize {
        0
    }

    /// Releases the sequence at once, without waiting. The caller makes this call only when no generation is running:
    /// after the listener's `finished` has run, or when `generate` was never called. A caller that cancels waits for
    /// `finished` (by continuation, not by blocking) before it closes; a close while a generation runs is a misuse
    /// the backend need not honour.
    fn close(&self);
}

/// The API layer's narrow view of the loaded model.
pub trait Backend: Send + Sync + 'static {
    /// The model's identity as the engine knows it.
    fn model_id(&self) -> &str;

    /// False once engine shutdown has begun; no further generation is admitted.
    fn is_available(&self) -> bool;

    /// The longest sequence in tokens, prompt plus completion.
    fn context_length(&self) -> usize;

    /// How many generations may run at once: the sequences the engine holds. The service queues the rest.
    fn max_active_generations(&self) -> usize {
        1
    }

    /// Encodes a rendered prompt on the workers, so that the plan counts its tokens and the generation runs them: a
    /// prompt is tokenized once. `then` runs on a worker with the encoding.
    fn encode_prompt(&self, prompt: String, then: Next<Result<EncodedPrompt, BackendError>>);

    /// Counts the tokens of a rendered prompt on the workers, without generating.
    fn count_tokens(&self, prompt: String, then: Next<Result<usize, BackendError>>) {
        self.encode_prompt(
            prompt,
            Box::new(move |encoded| then(encoded.map(|encoded| encoded.token_count()))),
        );
    }

    /// Compiles an answer grammar without generating.
    fn check_grammar(&self, grammar: &str) -> Result<(), BackendError>;

    /// Compiles a JSON Schema on its own, so a refusal names the schema's problem.
    fn check_json_schema(&self, schema: &str) -> Result<(), BackendError>;

    /// Opens one request-owned generation, or says the engine is closing or closed.
    fn open_generation(
        &self,
        config: GenerationConfig,
        output: OutputSpec,
    ) -> Result<Arc<dyn Generation>, BackendError>;
}
