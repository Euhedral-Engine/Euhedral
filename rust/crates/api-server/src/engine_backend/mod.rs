//! The backend over the engine: a planned request becomes a generation on the engine's one sequence, and its tokens
//! reach the answer module's listener as text.
//!
//! The engine and the tokenizer are reached through two small traits, [`Engine`] and [`Codec`], so the adapter is
//! tested without a device. [`start`] builds the real ones.
//!
//! Nothing here waits or owns a thread: prompts are encoded as frames on the engine's lattice, and a generation's
//! events arrive on the engine's workers and go straight on.

mod relay;
mod start;

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_inference_api_protocol::ApiError;
use euhedral_inference_api_protocol::generation::GenerationRequest;
use euhedral_inference_engine::generation::{
    AdmissionError, CancelHandle, GenerationRequest as EngineRequest, Speculation,
};
use euhedral_inference_engine::sampling::SamplingConfig;

use crate::backend::{
    Backend, BackendError, EncodedPrompt, EngineReport, Generation, GenerationConfig, Next, OutputSpec, TextListener,
};
use crate::frames::once_or_refused;
use crate::planning::Endpoint;
use crate::refusals::refusal;
use relay::{Hand, Relay, Shape};
pub use start::{EngineHost, EngineStarting, StartError, begin_start, start};

/// What the backend needs of the engine.
pub trait Engine: Send + Sync + 'static {
    /// Offers a generation. An `Err` means the request was refused and its listener dropped untold.
    fn generate(&self, request: EngineRequest, cancel: CancelHandle) -> Result<(), AdmissionError>;

    /// The longest sequence in tokens, prompt plus completion.
    fn context_length(&self) -> usize;

    /// How many generations the engine runs at once.
    fn sequences(&self) -> usize;

    /// What a generation speculates with: the artifact's draft layer when it has one.
    fn speculation(&self) -> Speculation;

    /// The engine's state for a scrape.
    fn report(&self) -> EngineReport;
}

/// The decoded text of one generation, token by token.
pub trait TextStream: Send {
    /// Appends a token and returns the complete text it finishes, possibly empty.
    fn push(&mut self, id: u32) -> Result<String, String>;

    /// Ends the stream and returns what is held back.
    fn finish(self: Box<Self>) -> String;
}

/// What the backend needs of the tokenizer.
pub trait Codec: Send + Sync + 'static {
    /// The tokens of a rendered prompt, with the model's special tokens.
    fn encode(&self, text: &str) -> Result<Vec<u32>, String>;

    /// A decoder for one generation.
    fn stream(&self) -> Box<dyn TextStream>;

    /// The tokens that end a generation.
    fn stop_tokens(&self) -> Vec<u32>;

    /// The token that closes the model's think block, when it has one.
    fn think_end(&self) -> Option<u32>;
}

/// The [`Backend`] over an engine and its tokenizer.
pub struct EngineBackend {
    engine: Arc<dyn Engine>,
    codec: Arc<dyn Codec>,
    sink: Arc<QueueIngestSink>,
    model_id: String,
    context_length: usize,
    available: Arc<AtomicBool>,
}

impl Debug for EngineBackend {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("EngineBackend")
            .field("model_id", &self.model_id)
            .field("available", &self.available)
            .finish_non_exhaustive()
    }
}

impl EngineBackend {
    /// A backend whose frames run on the lattice `sink` is attached to.
    pub fn new(
        engine: Arc<dyn Engine>,
        codec: Arc<dyn Codec>,
        sink: Arc<QueueIngestSink>,
        model_id: String,
    ) -> EngineBackend {
        let context_length = engine.context_length();
        EngineBackend {
            engine,
            codec,
            sink,
            model_id,
            context_length,
            available: Arc::new(AtomicBool::new(true)),
        }
    }

    /// Stops admitting generations and refuses the frames offered from now on (their continuations are told): the lattice
    /// is about to close.
    pub fn close(&self) {
        self.available.store(false, Ordering::Release);
        self.sink.complete();
    }
}

impl Backend for EngineBackend {
    fn model_id(&self) -> &str {
        &self.model_id
    }

    fn is_available(&self) -> bool {
        self.available.load(Ordering::Acquire)
    }

    fn context_length(&self) -> usize {
        self.context_length
    }

    fn max_active_generations(&self) -> usize {
        self.engine.sequences().max(1)
    }

    fn engine_report(&self) -> EngineReport {
        self.engine.report()
    }

    fn refusal(&self, endpoint: Endpoint, request: &GenerationRequest) -> Option<ApiError> {
        refusal(endpoint, request)
    }

    fn encode_prompt(&self, prompt: String, then: Next<Result<EncodedPrompt, BackendError>>) {
        let codec = Arc::clone(&self.codec);
        let frame = once_or_refused(move |ran| {
            if !ran {
                return then(Err(BackendError::Unavailable("the engine is shutting down".to_owned())));
            }
            let encoded = codec
                .encode(&prompt)
                .map(|token_ids| EncodedPrompt {
                    text: prompt,
                    token_ids,
                })
                .map_err(BackendError::Failed);
            then(encoded);
        });
        // A refused frame is dropped, which tells `then`.
        drop(self.sink.offer(frame));
    }

    fn check_grammar(&self, _grammar: &str) -> Result<(), BackendError> {
        // The engine has no grammar constraint, and a request that needs one is refused before this is asked.
        Ok(())
    }

    fn check_json_schema(&self, _schema: &str) -> Result<(), BackendError> {
        Ok(())
    }

    fn open_generation(
        &self,
        config: GenerationConfig,
        output: OutputSpec,
    ) -> Result<Arc<dyn Generation>, BackendError> {
        if !self.is_available() {
            return Err(BackendError::Unavailable(
                "inference engine is shutting down".to_owned(),
            ));
        }
        if output.grammar.is_some() || output.reasoning_budget.is_some() {
            return Err(BackendError::Failed(
                "the engine cannot constrain the output".to_owned(),
            ));
        }
        if output.reasoning && self.codec.think_end().is_none() {
            return Err(BackendError::Failed(
                "the model has no think block to read reasoning from".to_owned(),
            ));
        }
        let sampling = if config.greedy {
            SamplingConfig::greedy(config.seed)
        } else {
            SamplingConfig::new(config.temperature, config.top_k, config.top_p, config.seed, false)
                .map_err(|error| BackendError::Failed(error.to_string()))?
        };
        Ok(Arc::new(EngineGeneration {
            engine: Arc::clone(&self.engine),
            codec: Arc::clone(&self.codec),
            sampling,
            reasoning: output.reasoning,
            cancel: CancelHandle::default(),
            available: Arc::clone(&self.available),
            started: AtomicBool::new(false),
        }))
    }
}

/// The error a listener hears for a request the engine refused.
fn refused(error: AdmissionError) -> BackendError {
    match error {
        AdmissionError::Engine(reason) => BackendError::Unavailable(reason),
        other => BackendError::Failed(other.to_string()),
    }
}

struct EngineGeneration {
    engine: Arc<dyn Engine>,
    codec: Arc<dyn Codec>,
    sampling: SamplingConfig,
    reasoning: bool,
    cancel: CancelHandle,
    available: Arc<AtomicBool>,
    started: AtomicBool,
}

impl Generation for EngineGeneration {
    fn generate(&self, prompt: EncodedPrompt, max_new_tokens: usize, mut listener: Box<dyn TextListener>) {
        if self.started.swap(true, Ordering::AcqRel) {
            return listener.finished(Err(BackendError::Failed(
                "a generation runs once per sequence".to_owned(),
            )));
        }
        let hand = Hand::new();
        let relay = Relay::new(
            listener,
            self.codec.stream(),
            Shape::of(self.codec.as_ref(), self.reasoning),
            self.cancel.clone(),
            Arc::clone(&hand),
        );
        let request = EngineRequest {
            prompt: prompt.token_ids,
            max_tokens: u32::try_from(max_new_tokens).unwrap_or(u32::MAX),
            stop_tokens: self.codec.stop_tokens(),
            speculation: self.engine.speculation(),
            sampling: self.sampling,
            listener: Box::new(relay),
        };
        let offered = self.engine.generate(request, self.cancel.clone());
        if offered.is_ok() {
            hand.accepted.store(true, Ordering::SeqCst);
        }
        // Whatever the engine dropped untold comes back here; a listener it kept comes back to nobody.
        if let Some(mut listener) = hand.returned.take() {
            let error = match offered {
                Err(error) => refused(error),
                Ok(()) => BackendError::Failed("the engine dropped the generation".to_owned()),
            };
            listener.finished(Err(error));
        }
    }

    fn cancel(&self) {
        self.cancel.cancel();
    }

    fn is_cancelled(&self) -> bool {
        self.cancel.is_cancelled() || !self.available.load(Ordering::Acquire)
    }

    fn close(&self) {
        // The engine's sequence is free once the generation has ended; there is nothing to release.
    }
}
