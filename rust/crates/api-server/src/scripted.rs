//! A backend that follows a script, for the tests of this crate and of the crates built on it.
//!
//! It follows the engine's asynchronous contract without a device: prompts are encoded and generations run as frames on
//! the lattice, text is emitted between steps (cancellation stops the next step), and each generation records its
//! inputs and lifecycle so a test can assert cancellation and cleanup.
#![allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "test fixture: the records a test reads back from its own thread sit behind a lock"
)]

use std::fmt::{self, Debug, Formatter};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, PoisonError, Weak};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::ingest::{IngestSink, QueueIngestSink};

use crate::backend::{
    Backend, BackendError, EncodedPrompt, EngineReport, Generation, GenerationConfig, GenerationResult, Next,
    OutputSpec, TextListener,
};
use crate::frames::once_or_refused;

/// The model id a scripted backend reports.
pub const MODEL_ID: &str = "euhedral-test-model";
/// Scripted prompt tokens are `PROMPT_ID_BASE + position`, completion tokens `COMPLETION_ID_BASE + position`.
pub const PROMPT_ID_BASE: u32 = 100;
pub const COMPLETION_ID_BASE: u32 = 5000;

/// What the next generations emit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Script {
    /// One chunk per token, then a stop token when `stop_token` and budget remain.
    Tokens { chunks: Vec<String>, stop_token: bool },
    /// `tok0 tok1 ...` until cancelled or the budget is used.
    Endless,
}

impl Script {
    /// The chunks `Hello`, `, `, `world` and a stop token.
    pub fn hello() -> Script {
        Script::Tokens {
            chunks: ["Hello", ", ", "world"].map(str::to_owned).to_vec(),
            stop_token: true,
        }
    }

    fn chunk(&self, index: usize) -> Option<String> {
        match self {
            Script::Tokens { chunks, .. } => chunks.get(index).cloned(),
            Script::Endless => Some(format!("tok{index} ")),
        }
    }

    fn stop_token(&self) -> bool {
        matches!(self, Script::Tokens { stop_token: true, .. })
    }
}

/// The scripted backend. Its work runs on the lattice it was made for.
pub struct ScriptedBackend {
    sink: Arc<QueueIngestSink>,
    available: AtomicBool,
    context_length: AtomicUsize,
    script: Mutex<Script>,
    generations: Mutex<Vec<Arc<ScriptedGeneration>>>,
    encoded: AtomicUsize,
    report: Mutex<EngineReport>,
}

impl Debug for ScriptedBackend {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("ScriptedBackend")
            .field("available", &self.available)
            .finish_non_exhaustive()
    }
}

impl ScriptedBackend {
    /// A backend whose frames run on `lattice`. It is available, has a context of 4096 tokens and follows
    /// [`Script::hello`].
    pub fn on(lattice: &Arc<ControlPlaneLattice>) -> Arc<ScriptedBackend> {
        let sink = Arc::new(QueueIngestSink::new());
        let attached = lattice.add_upstream(sink.delegate());
        assert!(attached.is_ok(), "the lattice takes the backend's sink");
        Arc::new(ScriptedBackend {
            sink,
            available: AtomicBool::new(true),
            context_length: AtomicUsize::new(4096),
            script: Mutex::new(Script::hello()),
            generations: Mutex::new(Vec::new()),
            encoded: AtomicUsize::new(0),
            report: Mutex::new(EngineReport::default()),
        })
    }

    /// Makes the backend available or not (engine shutdown).
    pub fn set_available(&self, available: bool) {
        self.available.store(available, Ordering::Release);
    }

    /// Sets what [`Backend::engine_report`] returns. The speculative depth is read when the server starts.
    pub fn set_engine_report(&self, report: EngineReport) {
        *self.report.lock().unwrap_or_else(PoisonError::into_inner) = report;
    }

    /// Sets the script of the generations opened from now on.
    pub fn set_script(&self, script: Script) {
        *self.script.lock().unwrap_or_else(PoisonError::into_inner) = script;
    }

    /// The generations opened so far.
    pub fn generations(&self) -> Vec<Arc<ScriptedGeneration>> {
        self.generations.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    /// How many prompts were encoded.
    pub fn encoded(&self) -> usize {
        self.encoded.load(Ordering::Acquire)
    }

    /// Releases the backend's sink; call before the lattice closes.
    pub fn release(&self) {
        self.sink.complete_gracefully();
    }

    /// Makes the lattice refuse every frame the backend offers from now on, as a lattice that is closing does.
    pub fn refuse_work(&self) {
        self.sink.complete();
    }

    fn offer(&self, work: impl FnOnce(bool) + Send + 'static) {
        // A refused frame (the lattice is closing) is dropped, and its work is called with `false`.
        drop(self.sink.offer(once_or_refused(work)));
    }

    /// The tokens of a scripted prompt: one per character.
    pub fn prompt_tokens(prompt: &str) -> Vec<u32> {
        (0..prompt.chars().count())
            .map(|position| PROMPT_ID_BASE + position as u32)
            .collect()
    }
}

impl Backend for ScriptedBackend {
    fn model_id(&self) -> &str {
        MODEL_ID
    }

    fn is_available(&self) -> bool {
        self.available.load(Ordering::Acquire)
    }

    fn context_length(&self) -> usize {
        self.context_length.load(Ordering::Acquire)
    }

    fn engine_report(&self) -> EngineReport {
        self.report.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }

    fn encode_prompt(&self, prompt: String, then: Next<Result<EncodedPrompt, BackendError>>) {
        self.encoded.fetch_add(1, Ordering::AcqRel);
        self.offer(move |ran| {
            if !ran {
                return then(Err(refused()));
            }
            let token_ids = ScriptedBackend::prompt_tokens(&prompt);
            then(Ok(EncodedPrompt {
                text: prompt,
                token_ids,
            }));
        });
    }

    fn check_grammar(&self, _grammar: &str) -> Result<(), BackendError> {
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
        let script = self.script.lock().unwrap_or_else(PoisonError::into_inner).clone();
        let generation = Arc::new_cyclic(|me| ScriptedGeneration {
            me: me.clone(),
            config,
            output,
            script,
            sink: self.sink.clone(),
            cancelled: AtomicBool::new(false),
            emitted: AtomicUsize::new(0),
            close_count: AtomicUsize::new(0),
            started: Mutex::new(None),
        });
        self.generations
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .push(generation.clone());
        Ok(generation)
    }
}

/// One scripted generation and what it recorded.
pub struct ScriptedGeneration {
    pub config: GenerationConfig,
    pub output: OutputSpec,
    me: Weak<ScriptedGeneration>,
    script: Script,
    sink: Arc<QueueIngestSink>,
    cancelled: AtomicBool,
    emitted: AtomicUsize,
    close_count: AtomicUsize,
    /// The prompt text and the token budget `generate` was called with.
    started: Mutex<Option<(String, usize)>>,
}

impl Debug for ScriptedGeneration {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("ScriptedGeneration")
            .field("emitted", &self.emitted())
            .finish_non_exhaustive()
    }
}

impl ScriptedGeneration {
    /// How many chunks were emitted.
    pub fn emitted(&self) -> usize {
        self.emitted.load(Ordering::Acquire)
    }

    /// How many times `close` was called.
    pub fn close_count(&self) -> usize {
        self.close_count.load(Ordering::Acquire)
    }

    /// The prompt text and the token budget of the call to `generate`, once it was made.
    pub fn started(&self) -> Option<(String, usize)> {
        self.started.lock().unwrap_or_else(PoisonError::into_inner).clone()
    }
}

/// What a backend says when the lattice would not run its work.
fn refused() -> BackendError {
    BackendError::Unavailable("the lattice refused the work".to_owned())
}

/// The state one step hands to the next.
struct Run {
    generation: Arc<ScriptedGeneration>,
    listener: Box<dyn TextListener>,
    index: usize,
    max_new_tokens: usize,
}

impl Run {
    /// Emits the next chunk or ends, and offers the following step.
    fn step(mut self) {
        let generation = self.generation.clone();
        let cancelled = generation.is_cancelled();
        if cancelled || self.index == self.max_new_tokens {
            return self.finish(false);
        }
        let Some(chunk) = generation.script.chunk(self.index) else {
            let stop = generation.script.stop_token() && !cancelled;
            return self.finish(stop);
        };
        generation.emitted.fetch_add(1, Ordering::AcqRel);
        self.index += 1;
        self.listener.text(&chunk);
        drop(generation.sink.offer(once_or_refused(
            move |ran| if ran { self.step() } else { self.refuse() },
        )));
    }

    /// The lattice would not run the next step: the listener hears it failed.
    fn refuse(mut self) {
        self.listener.finished(Err(refused()));
    }

    fn finish(mut self, stop_token: bool) {
        let completion_tokens = self.index + usize::from(stop_token);
        let token_ids = (0..completion_tokens)
            .map(|position| COMPLETION_ID_BASE + position as u32)
            .collect();
        self.listener.finished(Ok(GenerationResult {
            completion_tokens,
            stop_token_reached: stop_token,
            token_ids,
            ..GenerationResult::default()
        }));
    }
}

impl Generation for ScriptedGeneration {
    fn generate(&self, prompt: EncodedPrompt, max_new_tokens: usize, listener: Box<dyn TextListener>) {
        *self.started.lock().unwrap_or_else(PoisonError::into_inner) = Some((prompt.text, max_new_tokens));
        // The step frames hold the generation until the last one.
        let Some(generation) = self.me.upgrade() else {
            let mut listener = listener;
            return listener.finished(Err(refused()));
        };
        let run = Run {
            generation,
            listener,
            index: 0,
            max_new_tokens,
        };
        drop(
            self.sink
                .offer(once_or_refused(move |ran| if ran { run.step() } else { run.refuse() })),
        );
    }

    fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn close(&self) {
        self.close_count.fetch_add(1, Ordering::AcqRel);
    }
}
