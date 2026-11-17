//! From the body of a request to a planned generation, as a chain of small frames.
//!
//! The chain is parse, map, grammar, template, encode and plan. Each step is a frame of its own, with its own routing
//! hash, published to the lattice by the step before it; none runs inline in another, so a large body, a large schema
//! or a long prompt never holds a worker for the whole chain, and the steps of different requests interleave. The
//! first frame is returned to the caller, who publishes it. An error at any step ends the chain: the request's outcome
//! is called once with it, and the caller answers in the format of the surface the request addressed. The outcome of
//! a chain is called exactly once on every path, a frame the lattice refuses included.
//!
//! The chain checks in the order a client sees the errors: the body (as UTF-8 and as the surface's JSON), the mapping
//! of the surface, the answer's grammar, the template, and last, once the prompt's tokens are known, the completion
//! budget against the context. Counting tokens (`count_tokens`) runs the same chain without the grammar and the plan:
//! it ends with the number of the prompt's tokens.
//!
//! The outcome is guarded: if a step panics, or a backend drops its continuation without calling it, the guard answers
//! `ApiError::server_error()` when the chain's state is dropped, so a request never goes unanswered.
//!
//! The prompt is encoded in one step, by [`Backend::encode_prompt`]: the backend takes the rendered prompt whole, so
//! a prompt cannot be cut at message boundaries into several encodings from here.

use std::collections::hash_map::RandomState;
use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};
use std::hash::BuildHasher;
use std::str::from_utf8;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use euhedral_core::frames::Frame;
use euhedral_core::ingest::QueueIngestSink;
use euhedral_inference_api_protocol::anthropic::MessagesRequest;
use euhedral_inference_api_protocol::generation::{Defaults, GenerationRequest, Sampling};
use euhedral_inference_api_protocol::json::JsonValue;
use euhedral_inference_api_protocol::openai::ChatCompletionRequest;
use euhedral_inference_api_protocol::output_grammar::{GrammarCheck, OutputGrammar};
use euhedral_inference_api_protocol::responses::ResponsesRequest;
use euhedral_inference_api_protocol::template::render;
use euhedral_inference_api_protocol::{ApiError, map_chat, map_count_tokens, map_messages, map_responses};

use crate::backend::{Backend, BackendError, EncodedPrompt, GenerationConfig, OutputSpec};
use crate::frames::once_or_refused;

/// What a request that states nothing takes, and the server's budget for the completion: settings of the server.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RequestDefaults {
    pub temperature: f32,
    /// Top-k; 0 is no limit.
    pub top_k: i32,
    pub top_p: f32,
    /// The completion budget of a request that states none, before the context caps it.
    pub max_tokens: i32,
}

/// A default [`RequestDefaults`] cannot hold.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DefaultsError {
    /// The completion budget is below 1.
    MaxTokens,
    /// The temperature is negative or not a number.
    Temperature,
    /// The top-k is negative.
    TopK,
    /// The top-p is not in (0, 1].
    TopP,
}

impl Display for DefaultsError {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.write_str(match self {
            DefaultsError::MaxTokens => "the default max_tokens must be at least 1",
            DefaultsError::Temperature => "the default temperature must not be negative",
            DefaultsError::TopK => "the default top_k must not be negative",
            DefaultsError::TopP => "the default top_p must be greater than 0 and at most 1",
        })
    }
}

impl Error for DefaultsError {}

impl RequestDefaults {
    /// Checks the settings, as the server does when it loads them.
    pub fn validate(&self) -> Result<(), DefaultsError> {
        if self.max_tokens < 1 {
            Err(DefaultsError::MaxTokens)
        } else if self.temperature.is_nan() || self.temperature < 0.0 {
            Err(DefaultsError::Temperature)
        } else if self.top_k < 0 {
            Err(DefaultsError::TopK)
        } else if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            Err(DefaultsError::TopP)
        } else {
            Ok(())
        }
    }
}

impl Default for RequestDefaults {
    fn default() -> RequestDefaults {
        RequestDefaults {
            temperature: 1.0,
            top_k: 20,
            top_p: 0.95,
            max_tokens: 4096,
        }
    }
}

/// Where the frames of a chain go: the lattice's sink of the server. A frame the lattice refuses is dropped, which
/// tells its work that it never ran.
pub trait Publisher: Send + Sync + 'static {
    fn publish(&self, frame: Box<dyn Frame>);
}

impl Publisher for QueueIngestSink {
    fn publish(&self, frame: Box<dyn Frame>) {
        drop(self.offer(frame));
    }
}

/// The generating endpoint a request addressed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Endpoint {
    ChatCompletions,
    Messages,
    Responses,
}

/// What only one surface asks of its reply.
#[derive(Clone, Debug, PartialEq)]
pub enum Extras {
    /// The streamed reply ends with a usage chunk.
    Chat {
        include_usage: bool,
    },
    Messages,
    Responses {
        /// Reasoning items carry the opaque `encrypted_content` token.
        include_encrypted_content: bool,
        /// The settings every response repeats, in the order they are written.
        echo: JsonValue,
    },
}

/// A request ready to be admitted: everything the generation and its reply need, fixed.
#[derive(Clone, Debug, PartialEq)]
pub struct Planned {
    /// Names the request in logs; the surfaces derive their public ids from it. 32 lower-case hex digits.
    pub id: String,
    pub endpoint: Endpoint,
    /// When the request was mapped, in seconds since the epoch: what the OpenAI and Responses replies call `created`.
    pub created: u64,
    /// The mapped request: the stops, the tools, the response format, `stream`, `return_token_ids`.
    pub request: GenerationRequest,
    pub extras: Extras,
    /// The rendered prompt and its tokens. The prompt's tokens plus `max_new_tokens` never exceed the context.
    pub prompt: EncodedPrompt,
    pub max_new_tokens: usize,
    /// The sampling the backend opens its generation with; the seed is the request's or one drawn for it.
    pub config: GenerationConfig,
    /// How the output is shaped: reasoning, its budget and the answer's grammar (`None` for free text).
    pub output: OutputSpec,
}

/// Called once with what a chain came to: the plan or the error that ended it.
pub type Outcome<T> = Box<dyn FnOnce(Result<T, ApiError>) + Send>;

/// The surface's request, read from the body.
enum Parsed {
    Chat(ChatCompletionRequest),
    Messages(MessagesRequest),
    Responses(ResponsesRequest),
    Count(MessagesRequest),
}

/// The request as mapped, with what its reply needs of the surface.
struct Mapped {
    created: u64,
    request: GenerationRequest,
    endpoint: Endpoint,
    extras: Extras,
}

/// An outcome that is called once: with the result it is given, or, if it is dropped before (a step panicked, a
/// backend let go of its continuation), with the server's error.
struct Guard<T>(Option<Outcome<T>>);

impl<T> Guard<T> {
    fn finish(mut self, result: Result<T, ApiError>) {
        if let Some(outcome) = self.0.take() {
            outcome(result);
        }
    }
}

impl<T> Drop for Guard<T> {
    fn drop(&mut self) {
        if let Some(outcome) = self.0.take() {
            outcome(Err(ApiError::server_error()));
        }
    }
}

/// How the chain ends.
enum Finish {
    Plan(Guard<Planned>),
    Count(Guard<usize>),
}

impl Finish {
    fn fail(self, error: ApiError) {
        match self {
            Finish::Plan(guard) => guard.finish(Err(error)),
            Finish::Count(guard) => guard.finish(Err(error)),
        }
    }
}

/// Seconds since the epoch now.
fn now_seconds() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |since| since.as_secs())
}

/// A frame of the chain: `work` runs on a worker with the chain's end, or, if the lattice refuses the frame, the end
/// is told the server is shutting down.
fn step(finish: Finish, work: impl FnOnce(Finish) + Send + 'static) -> Box<dyn Frame> {
    once_or_refused(move |ran| {
        if ran {
            work(finish);
        } else {
            finish.fail(ApiError::shutting_down());
        }
    })
}

/// The backend's grammar and schema checks, with its refusals worded as the API words them.
struct BackendChecks<'a>(&'a dyn Backend);

impl GrammarCheck for BackendChecks<'_> {
    fn check_grammar(&self, grammar: &str) -> Result<(), ApiError> {
        self.0.check_grammar(grammar).map_err(|error| match error {
            BackendError::Unenforceable(reason) => {
                ApiError::invalid_request(format!("The requested output cannot be enforced: {reason}"), None)
            }
            other => backend_failure(other),
        })
    }

    fn check_json_schema(&self, schema: &JsonValue, param: &str) -> Result<(), ApiError> {
        self.0
            .check_json_schema(&OutputGrammar::schema_text(schema))
            .map_err(|error| match error {
                BackendError::Unenforceable(reason) => {
                    ApiError::unenforceable_schema(format!("The JSON Schema cannot be enforced: {reason}"), Some(param))
                }
                other => backend_failure(other),
            })
    }
}

/// What the client is told of a backend failure that is not a refused grammar.
fn backend_failure(error: BackendError) -> ApiError {
    match error {
        BackendError::Unavailable(reason) => ApiError::engine_unavailable(&reason),
        BackendError::Unenforceable(_) | BackendError::Failed(_) => ApiError::server_error(),
    }
}

struct Inner {
    backend: Arc<dyn Backend>,
    publisher: Arc<dyn Publisher>,
    defaults: Defaults,
    max_tokens: i32,
}

/// Plans requests as chains of frames on the lattice of the publisher.
#[derive(Clone)]
pub struct Planner {
    inner: Arc<Inner>,
}

impl Debug for Planner {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("Planner").finish_non_exhaustive()
    }
}

impl Planner {
    /// A planner over `backend` whose frames go to `publisher`, or the setting `defaults` cannot hold.
    pub fn new(
        backend: Arc<dyn Backend>,
        defaults: RequestDefaults,
        publisher: Arc<dyn Publisher>,
    ) -> Result<Planner, DefaultsError> {
        defaults.validate()?;
        let mapper_defaults = Defaults {
            model: backend.model_id().to_owned(),
            temperature: defaults.temperature,
            top_p: defaults.top_p,
            top_k: defaults.top_k,
            max_tokens: defaults.max_tokens,
        };
        Ok(Planner {
            inner: Arc::new(Inner {
                backend,
                publisher,
                defaults: mapper_defaults,
                max_tokens: defaults.max_tokens,
            }),
        })
    }

    /// The chain of a generating request: returns its first frame, which the caller publishes. `outcome` is called
    /// once, with the plan or the error that ended the chain.
    pub fn plan(&self, endpoint: Endpoint, body: Vec<u8>, outcome: Outcome<Planned>) -> Box<dyn Frame> {
        let kind = match endpoint {
            Endpoint::ChatCompletions => Kind::Chat,
            Endpoint::Messages => Kind::Messages,
            Endpoint::Responses => Kind::Responses,
        };
        self.parse(kind, body, Finish::Plan(Guard(Some(outcome))))
    }

    /// The chain of a `count_tokens` request: its first frame; `outcome` is called once with the prompt's token count.
    pub fn count_tokens(&self, body: Vec<u8>, outcome: Outcome<usize>) -> Box<dyn Frame> {
        self.parse(Kind::Count, body, Finish::Count(Guard(Some(outcome))))
    }

    fn publish(&self, frame: Box<dyn Frame>) {
        self.inner.publisher.publish(frame);
    }

    fn parse(&self, kind: Kind, body: Vec<u8>, finish: Finish) -> Box<dyn Frame> {
        let planner = self.clone();
        step(finish, move |finish| {
            let parsed = from_utf8(&body)
                .map_err(|_| ApiError::unparsable_body())
                // One leading byte order mark is skipped, as the JSON readers of the surfaces' own servers do.
                .map(|text| text.strip_prefix('\u{feff}').unwrap_or(text))
                .and_then(|text| match kind {
                    Kind::Chat => ChatCompletionRequest::parse(text).map(Parsed::Chat),
                    Kind::Messages => MessagesRequest::parse(text).map(Parsed::Messages),
                    Kind::Responses => ResponsesRequest::parse(text).map(Parsed::Responses),
                    Kind::Count => MessagesRequest::parse(text).map(Parsed::Count),
                });
            match parsed {
                Ok(parsed) => planner.publish(planner.map(parsed, finish)),
                Err(error) => finish.fail(error),
            }
        })
    }

    fn map(&self, parsed: Parsed, finish: Finish) -> Box<dyn Frame> {
        let planner = self.clone();
        step(finish, move |finish| {
            let defaults = &planner.inner.defaults;
            let created = now_seconds();
            let mapped = match parsed {
                Parsed::Chat(request) => map_chat(request, defaults).map(|chat| Mapped {
                    created,
                    request: chat.generation,
                    endpoint: Endpoint::ChatCompletions,
                    extras: Extras::Chat {
                        include_usage: chat.include_usage,
                    },
                }),
                Parsed::Messages(request) => map_messages(request, defaults).map(|request| Mapped {
                    created,
                    request,
                    endpoint: Endpoint::Messages,
                    extras: Extras::Messages,
                }),
                Parsed::Responses(request) => map_responses(request, defaults).map(|responses| Mapped {
                    created,
                    request: responses.generation,
                    endpoint: Endpoint::Responses,
                    extras: Extras::Responses {
                        include_encrypted_content: responses.include_encrypted_content,
                        echo: responses.echo,
                    },
                }),
                Parsed::Count(request) => map_count_tokens(request, defaults).map(|request| Mapped {
                    created,
                    request,
                    endpoint: Endpoint::Messages,
                    extras: Extras::Messages,
                }),
            };
            match mapped {
                Ok(mapped) => {
                    let next = match finish {
                        Finish::Plan(_) => planner.grammar(mapped, finish),
                        Finish::Count(_) => planner.template(mapped, None, finish),
                    };
                    planner.publish(next);
                }
                Err(error) => finish.fail(error),
            }
        })
    }

    fn grammar(&self, mapped: Mapped, finish: Finish) -> Box<dyn Frame> {
        let planner = self.clone();
        step(finish, move |finish| {
            let request = &mapped.request;
            let checks = BackendChecks(planner.inner.backend.as_ref());
            match OutputGrammar::for_request(&request.tools, &request.format, request.reasoning(), &checks) {
                Ok(grammar) => planner.publish(planner.template(mapped, grammar, finish)),
                Err(error) => finish.fail(error),
            }
        })
    }

    fn template(&self, mapped: Mapped, grammar: Option<String>, finish: Finish) -> Box<dyn Frame> {
        let planner = self.clone();
        step(finish, move |finish| {
            let prompt = match render(&mapped.request.template, &mapped.request.thinking) {
                Ok(prompt) => prompt,
                Err(error) => return finish.fail(ApiError::from(error)),
            };
            planner.encode(mapped, grammar, prompt, finish);
        })
    }

    /// Hands the prompt to the backend's encoding frames; the next frame of the chain is published from the
    /// encoding's continuation.
    fn encode(&self, mapped: Mapped, grammar: Option<String>, prompt: String, finish: Finish) {
        let backend = self.inner.backend.clone();
        match finish {
            Finish::Count(guard) => backend.count_tokens(
                prompt,
                Box::new(move |counted| guard.finish(counted.map_err(backend_failure))),
            ),
            Finish::Plan(guard) => {
                let planner = self.clone();
                backend.encode_prompt(
                    prompt,
                    Box::new(move |encoded| match encoded {
                        Ok(encoded) => planner.publish(planner.plan_step(mapped, grammar, encoded, guard)),
                        Err(error) => guard.finish(Err(backend_failure(error))),
                    }),
                );
            }
        }
    }

    fn plan_step(
        &self,
        mapped: Mapped,
        grammar: Option<String>,
        encoded: EncodedPrompt,
        guard: Guard<Planned>,
    ) -> Box<dyn Frame> {
        let planner = self.clone();
        once_or_refused(move |ran| {
            if ran {
                guard.finish(planner.planned(mapped, grammar, encoded));
            } else {
                guard.finish(Err(ApiError::shutting_down()));
            }
        })
    }

    /// The completion budget checked against the context, and the plan.
    fn planned(&self, mapped: Mapped, grammar: Option<String>, prompt: EncodedPrompt) -> Result<Planned, ApiError> {
        let Mapped {
            created,
            request,
            endpoint,
            extras,
        } = mapped;
        let max_new_tokens = self.max_new_tokens(&request, prompt.token_count())?;
        let config = sampling_config(&request.sampling);
        let output = OutputSpec {
            reasoning: request.reasoning(),
            reasoning_budget: request.reasoning_budget,
            grammar,
        };
        Ok(Planned {
            id: new_id(),
            endpoint,
            created,
            request,
            extras,
            prompt,
            max_new_tokens,
            config,
            output,
        })
    }

    fn max_new_tokens(&self, request: &GenerationRequest, prompt_tokens: usize) -> Result<usize, ApiError> {
        let context = self.inner.backend.context_length() as i64;
        let prompt_tokens = prompt_tokens as i64;
        let remaining = context - prompt_tokens;
        let param = request.max_tokens_param.as_str();
        if let Some(requested) = request.max_tokens {
            let requested = i64::from(requested);
            if requested < 1 {
                return Err(ApiError::invalid_request(
                    format!("'{param}' must be at least 1."),
                    Some(param),
                ));
            }
            if requested > remaining {
                return Err(ApiError::context_length_exceeded(
                    format!(
                        "This model's maximum context length is {context} tokens. However, you requested {} tokens \
                         ({prompt_tokens} in the messages, {requested} in the completion).",
                        prompt_tokens + requested
                    ),
                    Some(param),
                ));
            }
            return Ok(requested as usize);
        }
        if remaining < 1 {
            return Err(ApiError::context_length_exceeded(
                format!(
                    "This model's maximum context length is {context} tokens, but the messages use {prompt_tokens} \
                     tokens."
                ),
                Some("messages"),
            ));
        }
        Ok(i64::from(self.inner.max_tokens).min(remaining) as usize)
    }
}

/// Which surface's body a chain reads.
#[derive(Clone, Copy)]
enum Kind {
    Chat,
    Messages,
    Responses,
    Count,
}

/// The sampling of a request as the backend takes it; a request without a seed gets one drawn.
fn sampling_config(sampling: &Sampling) -> GenerationConfig {
    GenerationConfig {
        temperature: sampling.temperature,
        top_k: sampling.top_k.max(0) as u32,
        top_p: sampling.top_p,
        seed: sampling.seed.map_or_else(random, |seed| seed as u64),
        greedy: sampling.greedy,
    }
}

/// A fresh 64-bit value: the process's random hash keys over a counter, never repeated within a process.
fn random() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    RandomState::new().hash_one(COUNTER.fetch_add(1, Ordering::Relaxed))
}

/// A request id: 32 lower-case hex digits.
fn new_id() -> String {
    format!("{:016x}{:016x}", random(), random())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_id_is_32_hex_digits_and_differs() {
        let (first, second) = (new_id(), new_id());
        assert_eq!(first.len(), 32);
        assert!(first.bytes().all(|byte| byte.is_ascii_hexdigit()));
        assert_ne!(first, second);
    }

    #[test]
    fn a_negative_top_k_is_no_limit() {
        let sampling = Sampling {
            temperature: 0.0,
            top_k: -1,
            top_p: 1.0,
            seed: Some(-1),
            greedy: true,
        };
        let config = sampling_config(&sampling);
        assert_eq!((config.top_k, config.seed), (0, u64::MAX));
    }
}
