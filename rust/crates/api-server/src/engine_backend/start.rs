//! Starting the engine: the artifact is checked and its tokenizer read first, so a setup problem is reported before
//! the card is touched, then the engine's own start runs as frames and hands the runtime over when it is ready.

use std::error::Error;
use std::fmt::{self, Debug, Display, Formatter};
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Weak};
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::ingest::{IngestSink, QueueIngestSink};
use euhedral_inference_api_protocol::output_grammar::{end_ids, token_bytes};
use euhedral_inference_api_protocol::template::{CONTROL_TOKENS, check_template_source};
use euhedral_inference_engine::error::EngineError;
use euhedral_inference_engine::generation::{
    AdmissionError, CancelHandle, GenerationRequest as EngineRequest, MAX_VERIFIED, Speculation,
};
use euhedral_inference_engine::prefix::PrefixCacheStats;
use euhedral_inference_engine::runtime::{EngineRuntime, ReadyHandoff, RuntimeOwner};
use euhedral_inference_settings::{Settings, variable_name};
use euhedral_inference_tokenizer::{TokenStream, Tokenizer, TokenizerSection, chat_template, read_artifact_section};

use super::{Codec, Engine, EngineBackend, TextStream};
use crate::backend::{EngineReport, PrefixCacheReport};

/// The settings key that names the artifact.
const ARTIFACT_KEY: &str = "model.artifact";

/// Why the engine did not start; the text is for the operator, in the words of the startup checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StartError {
    /// The artifact is not there or cannot be read.
    Setup(String),
    /// The artifact's tokenizer could not be read.
    Tokenizer(String),
    /// The artifact's chat template, or the control tokens it renders, are not ones the engine supports.
    Template(String),
    /// The CUDA library or the driver it needs is not usable.
    Driver(String),
    /// The engine did not load.
    Engine(String),
}

impl Display for StartError {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        match self {
            StartError::Setup(reason) => write!(out, "{reason}"),
            StartError::Tokenizer(reason) => write!(out, "Reading the artifact's tokenizer failed: {reason}"),
            StartError::Template(reason) => write!(out, "The artifact's chat template cannot be used: {reason}"),
            StartError::Driver(reason) => {
                write!(
                    out,
                    "the NVIDIA driver is required, with an NVIDIA Blackwell GPU: {reason}; install the driver \
                     (nvidia-smi lists the GPU when it works)"
                )
            }
            StartError::Engine(reason) => write!(out, "Loading the inference engine failed: {reason}"),
        }
    }
}

impl Error for StartError {}

impl From<EngineError> for StartError {
    fn from(error: EngineError) -> StartError {
        // Only nothing loading (the library, its runtime or the driver) is the driver probe's; any other call failing
        // is the engine's.
        if error.loads_nothing() {
            StartError::Driver(error.to_string())
        } else {
            StartError::Engine(error.to_string())
        }
    }
}

/// The engine behind the traits: the runtime, held weakly so that [`EngineHost::shutdown`] holds the last handle.
struct RuntimeEngine {
    runtime: Weak<EngineRuntime>,
    artifact: Option<String>,
    context_length: usize,
    vocabulary_size: usize,
}

impl RuntimeEngine {
    fn runtime(&self) -> Option<Arc<EngineRuntime>> {
        self.runtime.upgrade()
    }
}

impl Engine for RuntimeEngine {
    fn generate(&self, request: EngineRequest, cancel: CancelHandle) -> Result<(), AdmissionError> {
        match self.runtime() {
            Some(runtime) => runtime.generate_cancellable(request, cancel),
            None => Err(AdmissionError::Engine("the engine is closed".to_owned())),
        }
    }

    fn context_length(&self) -> usize {
        self.context_length
    }

    fn vocabulary_size(&self) -> usize {
        self.vocabulary_size
    }

    fn sequences(&self) -> usize {
        self.runtime().map_or(1, |runtime| runtime.sequences())
    }

    fn speculation(&self) -> Speculation {
        match self.runtime() {
            Some(runtime) if runtime.plan().draft_depth > 0 => Speculation::Artifact,
            _ => Speculation::Off,
        }
    }

    fn report(&self) -> EngineReport {
        let Some(runtime) = self.runtime() else {
            return EngineReport::default();
        };
        let depth = runtime.plan().draft_depth.min(MAX_VERIFIED);
        EngineReport {
            artifact: self.artifact.clone(),
            speculation: if depth > 0 { "mtp" } else { "none" }.to_owned(),
            speculative_depth: depth,
            workers: runtime.lattice().active_workers(),
            allocated_device_bytes: runtime.report().allocated_bytes,
            free_device_bytes: runtime.library().free_bytes().unwrap_or(0),
            host_backed_weight_bytes: runtime.plan().location_bytes().host,
            pinned_pool_bytes: runtime.pinned_pool().held_bytes(),
            pinned_pool_overflows: 0,
            prefix_cache: runtime.prefix_cache_stats().map(prefix_report),
        }
    }
}

impl TextStream for TokenStream {
    fn push(&mut self, id: u32) -> Result<String, String> {
        TokenStream::push(self, id).map_err(|error| error.to_string())
    }

    fn finish(self: Box<Self>) -> String {
        TokenStream::finish(*self)
    }
}

impl Codec for Tokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode_with_model_special_tokens(text)
            .map_err(|error| error.to_string())
    }

    fn stream(&self) -> Box<dyn TextStream> {
        Box::new(self.decode_stream())
    }

    fn stop_tokens(&self) -> Vec<u32> {
        self.eos_ids().to_vec()
    }

    fn think_end(&self) -> Option<u32> {
        self.control_token_id("</think>")
    }

    fn token_bytes(&self, size: usize) -> Vec<Vec<u8>> {
        token_bytes(self, size)
    }

    fn end_ids(&self) -> Vec<u32> {
        end_ids(self)
    }
}

#[allow(
    clippy::disallowed_methods,
    reason = "startup: the artifact is looked at once, before the engine starts"
)]
fn artifact_problem(path: &Path) -> Option<String> {
    let name = format!("{ARTIFACT_KEY} ({})", variable_name(ARTIFACT_KEY));
    match path.metadata() {
        Err(_) => Some(format!("{name}: no file at {}", path.display())),
        Ok(metadata) if !metadata.is_file() => Some(format!("{name}: no file at {}", path.display())),
        Ok(_) => match File::open(path) {
            Ok(_) => None,
            Err(_) => Some(format!("{name}: {} is not readable", path.display())),
        },
    }
}

/// An engine that is loading. It holds the engine's owner: finish it with [`wait`](EngineStarting::wait) or poll it
/// with [`ready`](EngineStarting::ready), on the thread that started it.
pub struct EngineStarting {
    pending: Pending,
    tokenizer: Tokenizer,
    settings: Settings,
}

/// The owner and the hand-off of a start that has not ended. Dropped with its start still pending, it abandons the
/// start: a runtime that arrives later is freed by its arrival, and one that is already there is freed through the
/// owner.
struct Pending {
    owner: Option<RuntimeOwner>,
    ready: ReadyHandoff,
}

impl Pending {
    /// The owner, for a start that ended (or that this takes over).
    fn owner(&mut self) -> Option<RuntimeOwner> {
        self.owner.take()
    }

    /// Ends the start without serving it. Nothing waits here beyond what the owner's drop does.
    fn abandon(&mut self) {
        let Some(owner) = self.owner.take() else { return };
        match self.ready.abandon() {
            Some(Ok(runtime)) => {
                if let Err(error) = owner.shutdown(runtime) {
                    log::error!("stopping the engine that did not start failed: {error}");
                }
            }
            Some(Err(_)) | None => drop(owner),
        }
    }
}

impl Drop for Pending {
    fn drop(&mut self) {
        self.abandon();
    }
}

/// Checks the artifact, reads its tokenizer and starts the engine of `settings` on the CUDA library at `library`.
/// Returns at once; the engine loads on its own lattice. Call it from the thread that will shut the engine down, which
/// is not a worker of any lattice. `close_timeout` is the longest the lattice's close waits for the workers (the server
/// passes `server.shutdown-timeout`); `None` keeps the default.
pub fn begin_start(
    settings: &Settings,
    library: &Path,
    close_timeout: Option<Duration>,
) -> Result<EngineStarting, StartError> {
    if let Some(problem) = artifact_problem(&settings.model.artifact) {
        return Err(StartError::Setup(problem));
    }
    let section =
        read_artifact_section(&settings.model.artifact).map_err(|error| StartError::Tokenizer(error.to_string()))?;
    let tokenizer = checked_tokenizer(&section)?;
    let (owner, ready) = EngineRuntime::start_with(settings, library, close_timeout);
    Ok(EngineStarting {
        pending: Pending {
            owner: Some(owner),
            ready,
        },
        tokenizer,
        settings: settings.clone(),
    })
}

/// The tokenizer of `section`, once its chat template passes the fragment check and every control token the
/// rendering emits encodes as one added token.
fn checked_tokenizer(section: &TokenizerSection) -> Result<Tokenizer, StartError> {
    let tokenizer = Tokenizer::from_section(section).map_err(|error| StartError::Tokenizer(error.to_string()))?;
    let source = chat_template(section).map_err(|error| StartError::Template(error.to_string()))?;
    check_template_source(source).map_err(|error| StartError::Template(error.to_string()))?;
    for token in CONTROL_TOKENS {
        let id = tokenizer.control_token_id(token);
        let encoded = tokenizer
            .encode(token)
            .map_err(|error| StartError::Template(format!("{token} does not encode: {error}")))?;
        if id.is_none() || encoded.as_slice() != id.as_slice() {
            return Err(StartError::Template(format!(
                "the tokenizer does not encode {token} as one added token"
            )));
        }
    }
    Ok(tokenizer)
}

impl Debug for EngineStarting {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("EngineStarting").finish_non_exhaustive()
    }
}

impl Debug for EngineHost {
    fn fmt(&self, out: &mut Formatter<'_>) -> fmt::Result {
        out.debug_struct("EngineHost")
            .field("backend", &self.backend)
            .finish_non_exhaustive()
    }
}

impl EngineStarting {
    /// The loaded engine, or this still loading (`Err`), without waiting.
    pub fn ready(self) -> Result<Result<EngineHost, StartError>, Box<EngineStarting>> {
        match self.pending.ready.take() {
            Some(loaded) => Ok(self.host(loaded)),
            None => Err(Box::new(self)),
        }
    }

    /// Waits up to `limit` for the engine, on the caller's own thread: the one wait of a start. An engine that is not
    /// ready by then is stopped.
    pub fn wait(mut self, limit: Duration) -> Result<EngineHost, StartError> {
        match self.pending.ready.wait(limit) {
            Ok(loaded) => self.host(loaded),
            Err(_) => {
                // It may have finished just now: what is loaded is freed through the owner. Otherwise the start is
                // marked abandoned, so that a runtime arriving later is freed by its arrival, and the owner's drop
                // closes the lattice.
                self.pending.abandon();
                Err(StartError::Engine(format!(
                    "the engine was not ready within {} s",
                    limit.as_secs()
                )))
            }
        }
    }

    fn host(mut self, loaded: Result<Arc<EngineRuntime>, EngineError>) -> Result<EngineHost, StartError> {
        let runtime = match loaded {
            Ok(runtime) => runtime,
            // Nothing was loaded; the owner's drop closes what started.
            Err(error) => return Err(StartError::from(error)),
        };
        let Some(owner) = self.pending.owner() else {
            return Err(StartError::Engine("the engine's owner is gone".to_owned()));
        };
        let plan = runtime.plan();
        let context_length = plan.max_context_tokens.min(runtime.config().max_position_embeddings);
        let sink = Arc::new(QueueIngestSink::new());
        if let Err(error) = runtime.lattice().add_upstream(sink.delegate()) {
            if let Err(stop) = owner.shutdown(runtime) {
                log::error!("stopping the engine that did not start failed: {stop}");
            }
            return Err(StartError::Engine(format!("the lattice takes no backend: {error:?}")));
        }
        let engine = Arc::new(RuntimeEngine {
            runtime: Arc::downgrade(&runtime),
            artifact: self
                .settings
                .model
                .artifact
                .file_name()
                .map(|name| name.to_string_lossy().into_owned()),
            context_length,
            vocabulary_size: runtime.config().vocab_size,
        });
        let backend = Arc::new(
            EngineBackend::new(engine, Arc::new(self.tokenizer), sink, self.settings.model.name.clone())
                .map_err(StartError::Engine)?,
        );
        Ok(EngineHost {
            owner,
            runtime,
            backend,
        })
    }
}

/// A running engine: its backend, its lattice (the server runs on it too) and the means to stop it.
pub struct EngineHost {
    owner: RuntimeOwner,
    runtime: Arc<EngineRuntime>,
    backend: Arc<EngineBackend>,
}

impl EngineHost {
    /// The backend the server answers with.
    pub fn backend(&self) -> Arc<EngineBackend> {
        Arc::clone(&self.backend)
    }

    /// The runtime itself, for what the backend does not offer (a direct generation, a snapshot). Do not keep the
    /// handle: [`shutdown`](EngineHost::shutdown) needs the last one.
    pub fn runtime(&self) -> &Arc<EngineRuntime> {
        &self.runtime
    }

    /// The engine's lattice: the server's frames run here.
    pub fn lattice(&self) -> &Arc<ControlPlaneLattice> {
        self.runtime.lattice()
    }

    /// Stops the engine within `window`: the backend admits nothing more, then the lattice closes and the device
    /// memory is freed. What it could not finish in the window is reported (and not waited for): the error says what
    /// stays. Call it on the thread that started the engine, after the server has stopped and no generation runs.
    pub fn shutdown(self, window: Duration) -> Result<(), StartError> {
        self.backend.close();
        self.owner
            .shutdown_within(self.runtime, window)
            .map_err(StartError::from)
    }
}

/// Starts the engine and waits up to `limit` for it: [`begin_start`] and [`EngineStarting::wait`].
pub fn start(
    settings: &Settings,
    library: &Path,
    limit: Duration,
    close_timeout: Option<Duration>,
) -> Result<EngineHost, StartError> {
    begin_start(settings, library, close_timeout)?.wait(limit)
}

/// The engine's counters as the metrics page reports them.
fn prefix_report(stats: PrefixCacheStats) -> PrefixCacheReport {
    PrefixCacheReport {
        lookups: stats.lookups,
        hits: stats.hits,
        restored_tokens: stats.restored_tokens,
        captures: stats.captures,
        skipped: stats.skipped,
        failed: stats.failed,
        evictions: stats.evictions,
        restores: stats.restores,
        capture_nanos: stats.capture_nanos,
        restore_nanos: stats.restore_nanos,
        nodes: stats.nodes,
        used_bytes: stats.used_bytes,
        capacity_bytes: stats.capacity_bytes,
    }
}

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::path::PathBuf;

    use euhedral_inference_engine::runtime::Teardown;
    use euhedral_inference_engine::{Handoff, lattice_config};
    use euhedral_inference_settings::ModelSettings;

    use euhedral_inference_edrl::{FOOTER_BYTES, TOKENIZER_ENTRIES, TokenizerFooter};
    use serde_json::{Value, json};

    use super::*;

    #[test]
    fn the_engines_prefix_counters_all_reach_the_report() {
        let stats = PrefixCacheStats {
            lookups: 1,
            hits: 2,
            restored_tokens: 3,
            captures: 4,
            skipped: 5,
            failed: 6,
            evictions: 7,
            restores: 8,
            capture_nanos: 9,
            restore_nanos: 10,
            nodes: 11,
            used_bytes: 12,
            capacity_bytes: 13,
        };
        let report = prefix_report(stats);
        let fields = [
            report.lookups,
            report.hits,
            report.restored_tokens,
            report.captures,
            report.skipped,
            report.failed,
            report.evictions,
            report.restores,
            report.capture_nanos,
            report.restore_nanos,
            report.nodes,
            report.used_bytes,
            report.capacity_bytes,
        ];
        assert_eq!(fields, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]);
    }

    const CHECKPOINT_TEMPLATE: &str = include_str!("../../../api-protocol/tests/data/qwen-chat-template.jinja");

    /// A byte-level tokenizer with the given added tokens (ids from 256 up), the first being the EOS.
    fn tokenizer_json(added: &[&str]) -> Value {
        let mut next = 256;
        let characters: Vec<char> = (0..=255u32)
            .map(|byte| {
                if matches!(byte, 33..=126 | 161..=172 | 174..=255) {
                    char::from_u32(byte).expect("a character")
                } else {
                    next += 1;
                    char::from_u32(next - 1).expect("a character")
                }
            })
            .collect();
        let vocabulary: serde_json::Map<String, Value> = characters
            .iter()
            .enumerate()
            .map(|(byte, character)| (character.to_string(), json!(byte)))
            .collect();
        let added: Vec<Value> = added
            .iter()
            .enumerate()
            .map(|(index, content)| {
                json!({"id": 256 + index, "content": content, "single_word": false, "lstrip": false,
                       "rstrip": false, "normalized": false, "special": true})
            })
            .collect();
        json!({
            "version": "1.0",
            "added_tokens": added,
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
                {"type": "Split", "pattern": {"Regex": r"\p{L}+|\p{N}|\s+|[^\s\p{L}\p{N}]+"},
                 "behavior": "Isolated", "invert": false},
                {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
            ]},
            "post_processor": null,
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
            "model": {"type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
                      "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false,
                      "ignore_merges": false, "vocab": vocabulary, "merges": []},
        })
    }

    /// A section whose added tokens are `added` (`<|im_end|>` first, as the EOS) and whose template is `template`.
    fn section(added: &[&str], template: &str) -> TokenizerSection {
        let contents = [
            tokenizer_json(added).to_string(),
            json!({"eos_token": "<|im_end|>", "bos_token": null, "add_bos_token": false}).to_string(),
            template.to_owned(),
            json!({"bos_token_id": 256, "eos_token_id": [256]}).to_string(),
        ];
        let mut body = 4i32.to_be_bytes().to_vec();
        for (name, data) in TOKENIZER_ENTRIES.iter().zip(&contents) {
            body.extend(i32::try_from(name.len()).expect("short").to_be_bytes());
            body.extend(name.as_bytes());
            body.extend(i32::try_from(data.len()).expect("short").to_be_bytes());
            body.extend(data.as_bytes());
        }
        let mut footer = i64::try_from(body.len()).expect("short").to_be_bytes().to_vec();
        footer.extend(crc32fast::hash(&body).to_be_bytes());
        footer.extend(1i32.to_be_bytes());
        footer.extend(b"EDRLTOKN");
        let footer: [u8; FOOTER_BYTES] = footer.try_into().expect("the footer's size");
        let footer = TokenizerFooter::parse(&footer).expect("a footer").expect("the magic");
        TokenizerSection::parse(body, &footer).expect("a section")
    }

    fn control_tokens() -> Vec<&'static str> {
        let mut tokens = vec!["<|im_end|>"];
        tokens.extend(CONTROL_TOKENS.iter().filter(|token| **token != "<|im_end|>"));
        tokens
    }

    /// The real artifact (`euhedral.qwen.rco-artifact`) passes the startup checks; skips without the key.
    #[test]
    #[allow(
        clippy::print_stdout,
        reason = "a test prints its skip line when the machine has no artifact"
    )]
    fn the_real_artifact_passes_the_startup_checks() {
        let Some(path) = test_support::require_property("euhedral.qwen.rco-artifact") else {
            return;
        };
        let path = PathBuf::from(path);
        if !path.is_file() {
            println!("skipped: euhedral.qwen.rco-artifact names no file");
            return;
        }
        let section = read_artifact_section(&path).expect("the artifact's tokenizer section");
        if let Err(error) = checked_tokenizer(&section) {
            panic!("the real artifact is refused: {error}");
        }
    }

    #[test]
    fn a_section_with_the_checkpoint_template_and_all_control_tokens_starts() {
        assert!(checked_tokenizer(&section(&control_tokens(), CHECKPOINT_TEMPLATE)).is_ok());
    }

    #[test]
    fn a_template_that_fails_the_fragment_check_is_refused() {
        for template in ["{{ messages }}", ""] {
            let error = checked_tokenizer(&section(&control_tokens(), template)).expect_err("refused");
            assert!(matches!(error, StartError::Template(_)), "{error}");
        }
    }

    #[test]
    fn a_control_token_that_is_missing_is_refused() {
        let tokens: Vec<&str> = control_tokens()
            .into_iter()
            .filter(|token| *token != "<think>")
            .collect();
        let error = checked_tokenizer(&section(&tokens, CHECKPOINT_TEMPLATE)).expect_err("refused");
        assert!(error.to_string().contains("<think>"), "{error}");
    }

    #[test]
    fn an_error_that_is_not_nothing_loading_is_the_engines() {
        // The three load errors and another call failing are classified by the engine's `loads_nothing`, whose tests
        // build the CUDA errors; this crate may not.
        let error = StartError::from(EngineError::Runtime("no lattice".to_owned()));
        assert_eq!(error, StartError::Engine("no lattice".to_owned()));
    }

    #[test]
    fn a_start_dropped_without_waiting_is_abandoned() {
        let settings = Settings::with_model(ModelSettings {
            artifact: PathBuf::from("m.edrl"),
            name: "n".to_owned(),
        });
        let lattice = ControlPlaneLattice::new(lattice_config(&settings)).expect("the lattice starts");
        let owner = RuntimeOwner::over(Teardown::new(lattice, Vec::new()));
        let ready = Handoff::new();
        drop(Pending {
            owner: Some(owner),
            ready: Arc::clone(&ready),
        });
        // An arrival after the drop finds the start abandoned and is released by its sender, once.
        let released = Cell::new(0);
        ready.deliver(Err(EngineError::Runtime("late".to_owned())), |_| {
            released.set(released.get() + 1)
        });
        assert_eq!(released.get(), 1);
    }

    #[test]
    fn a_driver_problem_carries_the_install_hint() {
        let text = StartError::Driver("no library".to_owned()).to_string();
        assert!(text.starts_with("the NVIDIA driver is required, with an NVIDIA Blackwell GPU: no library; install"));
        assert!(text.ends_with("(nvidia-smi lists the GPU when it works)"));
    }
}
