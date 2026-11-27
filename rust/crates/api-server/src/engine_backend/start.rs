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
use euhedral_inference_engine::flash::serve::FlashRuntime;
use euhedral_inference_engine::flash::start::{ModelFamily, family_at as family_of_file, family_name};
use euhedral_inference_engine::flash::startup::{FlashModel, FlashReady, start as start_flash};
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
        let pool = runtime.pinned_pool().gauges();
        EngineReport {
            artifact: self.artifact.clone(),
            family: family_name(ModelFamily::DenseV2).to_owned(),
            speculation: if depth > 0 { "mtp" } else { "none" }.to_owned(),
            speculative_depth: depth,
            workers: runtime.lattice().active_workers(),
            allocated_device_bytes: runtime.report().allocated_bytes,
            free_device_bytes: runtime.library().free_bytes().unwrap_or(0),
            host_backed_weight_bytes: runtime.plan().location_bytes().host,
            pinned_pool_bytes: pool.held_bytes,
            pinned_pool_overflows: pool.overflows,
            prefix_cache: runtime.prefix_cache_stats().map(prefix_report),
        }
    }
}

/// The Flash-Next engine behind the traits: the runtime, held weakly so that [`EngineHost::shutdown`] holds the last
/// handle. One sequence, no drafter.
struct FlashServing {
    runtime: Weak<FlashRuntime>,
    artifact: Option<String>,
    context_length: usize,
    vocabulary_size: usize,
}

impl Engine for FlashServing {
    fn generate(&self, request: EngineRequest, cancel: CancelHandle) -> Result<(), AdmissionError> {
        match self.runtime.upgrade() {
            Some(runtime) => runtime.generate(request, cancel),
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
        1
    }

    fn speculation(&self) -> Speculation {
        Speculation::Off
    }

    fn report(&self) -> EngineReport {
        let Some(runtime) = self.runtime.upgrade() else {
            return EngineReport::default();
        };
        let pool = runtime.pinned_pool().gauges();
        EngineReport {
            artifact: self.artifact.clone(),
            family: family_name(ModelFamily::FlashNextV3).to_owned(),
            speculation: "none".to_owned(),
            speculative_depth: 0,
            workers: runtime.lattice().active_workers(),
            allocated_device_bytes: runtime.allocated_device_bytes(),
            free_device_bytes: runtime.library().free_bytes().unwrap_or(0),
            host_backed_weight_bytes: runtime.host_backed_weight_bytes(),
            pinned_pool_bytes: pool.held_bytes,
            pinned_pool_overflows: pool.overflows,
            prefix_cache: None,
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

/// The family of the artifact at `path`, from its first bytes: the dispatch of the start. An artifact that is not a
/// family this engine serves is refused here, naming what the file is, before the card is touched.
fn family_at(path: &Path) -> Result<ModelFamily, StartError> {
    family_of_file(path)
        .map_err(|error| StartError::Setup(format!("{ARTIFACT_KEY} ({}): {error}", variable_name(ARTIFACT_KEY))))
}

/// An engine that is loading. It holds the engine's owner: finish it with [`wait`](EngineStarting::wait) or poll it
/// with [`ready`](EngineStarting::ready), on the thread that started it.
pub struct EngineStarting {
    pending: Pending,
    tokenizer: Tokenizer,
    settings: Settings,
}

/// The start of one family that has not ended.
enum Pending {
    Dense(DensePending),
    Flash(FlashPending),
}

/// What a start delivered.
enum Arrival {
    Dense(Result<Arc<EngineRuntime>, EngineError>),
    Flash(Result<Box<FlashModel>, EngineError>),
}

impl Pending {
    /// The delivery, if it is there.
    fn take(&self) -> Option<Arrival> {
        match self {
            Pending::Dense(start) => start.ready.take().map(Arrival::Dense),
            Pending::Flash(start) => start.ready.take().map(|loaded| Arrival::Flash(loaded.map(Box::new))),
        }
    }

    /// The delivery within `limit`.
    fn wait(&self, limit: Duration) -> Option<Arrival> {
        match self {
            Pending::Dense(start) => start.ready.wait(limit).ok().map(Arrival::Dense),
            Pending::Flash(start) => start
                .ready
                .wait(limit)
                .ok()
                .map(|loaded| Arrival::Flash(loaded.map(Box::new))),
        }
    }

    /// The owner, for a start that ended (or that this takes over).
    fn owner(&mut self) -> Option<RuntimeOwner> {
        match self {
            Pending::Dense(start) => start.owner.take(),
            Pending::Flash(start) => start.owner.take(),
        }
    }

    fn abandon(&mut self) {
        match self {
            Pending::Dense(start) => start.abandon(),
            Pending::Flash(start) => start.abandon(),
        }
    }
}

/// The owner and the hand-off of a dense start that has not ended. Dropped with its start still pending, it abandons
/// the start: a runtime that arrives later is freed by its arrival, and one that is already there is freed through the
/// owner.
struct DensePending {
    owner: Option<RuntimeOwner>,
    ready: ReadyHandoff,
}

impl DensePending {
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

impl Drop for DensePending {
    fn drop(&mut self) {
        self.abandon();
    }
}

/// [`DensePending`] for a Flash-Next start: a model that is already there is freed through the owner, one that
/// arrives later is dropped by its arrival (its device memory stays until exit, as the dense arrival's does when it
/// cannot free).
struct FlashPending {
    owner: Option<RuntimeOwner>,
    ready: FlashReady,
}

impl FlashPending {
    fn abandon(&mut self) {
        let Some(owner) = self.owner.take() else { return };
        match self.ready.abandon() {
            Some(Ok(model)) => model.shutdown(owner, true),
            Some(Err(_)) | None => drop(owner),
        }
    }
}

impl Drop for FlashPending {
    fn drop(&mut self) {
        self.abandon();
    }
}

/// Checks the artifact, reads its tokenizer and starts the engine of `settings` on the CUDA library at `library`: the
/// engine of the artifact's family. Returns at once; the engine loads on its own lattice. Call it from the thread that
/// will shut the engine down, which is not a worker of any lattice. `close_timeout` is the longest the lattice's close
/// waits for the workers (the server passes `server.shutdown-timeout`); `None` keeps the default.
pub fn begin_start(
    settings: &Settings,
    library: &Path,
    close_timeout: Option<Duration>,
) -> Result<EngineStarting, StartError> {
    if let Some(problem) = artifact_problem(&settings.model.artifact) {
        return Err(StartError::Setup(problem));
    }
    let family = family_at(&settings.model.artifact)?;
    let section =
        read_artifact_section(&settings.model.artifact).map_err(|error| StartError::Tokenizer(error.to_string()))?;
    let tokenizer = checked_tokenizer(&section)?;
    let pending = match family {
        ModelFamily::DenseV2 => {
            let (owner, ready) = EngineRuntime::start_with(settings, library, close_timeout);
            Pending::Dense(DensePending {
                owner: Some(owner),
                ready,
            })
        }
        ModelFamily::FlashNextV3 => {
            let (owner, ready) = start_flash(settings, library, close_timeout);
            Pending::Flash(FlashPending {
                owner: Some(owner),
                ready,
            })
        }
    };
    Ok(EngineStarting {
        pending,
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
        match self.pending.take() {
            Some(arrival) => Ok(self.host(arrival)),
            None => Err(Box::new(self)),
        }
    }

    /// Waits up to `limit` for the engine, on the caller's own thread: the one wait of a start. An engine that is not
    /// ready by then is stopped.
    pub fn wait(mut self, limit: Duration) -> Result<EngineHost, StartError> {
        match self.pending.wait(limit) {
            Some(arrival) => self.host(arrival),
            None => {
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

    fn host(self, arrival: Arrival) -> Result<EngineHost, StartError> {
        match arrival {
            Arrival::Dense(loaded) => self.dense_host(loaded),
            Arrival::Flash(loaded) => self.flash_host(loaded),
        }
    }

    /// The artifact's file name, which the engine reports.
    fn artifact_name(&self) -> Option<String> {
        self.settings
            .model
            .artifact
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
    }

    fn dense_host(mut self, loaded: Result<Arc<EngineRuntime>, EngineError>) -> Result<EngineHost, StartError> {
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
            artifact: self.artifact_name(),
            context_length,
            vocabulary_size: runtime.config().vocab_size,
        });
        let backend = Arc::new(
            EngineBackend::new(engine, Arc::new(self.tokenizer), sink, self.settings.model.name.clone())
                .map_err(StartError::Engine)?,
        );
        Ok(EngineHost {
            held: Held::Dense { owner, runtime },
            backend,
        })
    }

    fn flash_host(mut self, loaded: Result<Box<FlashModel>, EngineError>) -> Result<EngineHost, StartError> {
        let model = match loaded {
            Ok(model) => model,
            // Nothing was loaded; the owner's drop closes what started.
            Err(error) => return Err(StartError::from(error)),
        };
        let Some(owner) = self.pending.owner() else {
            return Err(StartError::Engine("the engine's owner is gone".to_owned()));
        };
        let (runtime, owner) = FlashRuntime::over(*model, owner)?;
        let runtime = Arc::new(runtime);
        let sink = Arc::new(QueueIngestSink::new());
        if let Err(error) = runtime.lattice().add_upstream(sink.delegate()) {
            // Nothing runs on an engine that did not start: no window need bound its stop.
            if let Err(stop) = FlashRuntime::shutdown(runtime, owner, Duration::MAX) {
                log::error!("stopping the engine that did not start failed: {stop}");
            }
            return Err(StartError::Engine(format!("the lattice takes no backend: {error:?}")));
        }
        let engine = Arc::new(FlashServing {
            runtime: Arc::downgrade(&runtime),
            artifact: self.artifact_name(),
            context_length: runtime.context_tokens(),
            vocabulary_size: runtime.vocabulary_size(),
        });
        let backend = EngineBackend::new(engine, Arc::new(self.tokenizer), sink, self.settings.model.name.clone());
        match backend {
            Ok(backend) => Ok(EngineHost {
                held: Held::Flash { owner, runtime },
                backend: Arc::new(backend),
            }),
            Err(error) => {
                if let Err(stop) = FlashRuntime::shutdown(runtime, owner, Duration::MAX) {
                    log::error!("stopping the engine that did not start failed: {stop}");
                }
                Err(StartError::Engine(error))
            }
        }
    }
}

/// What a running engine is held by, per family.
enum Held {
    Dense {
        owner: RuntimeOwner,
        runtime: Arc<EngineRuntime>,
    },
    Flash {
        owner: RuntimeOwner,
        runtime: Arc<FlashRuntime>,
    },
}

/// A running engine: its backend, its lattice (the server runs on it too) and the means to stop it.
pub struct EngineHost {
    held: Held,
    backend: Arc<EngineBackend>,
}

impl EngineHost {
    /// The backend the server answers with.
    pub fn backend(&self) -> Arc<EngineBackend> {
        Arc::clone(&self.backend)
    }

    /// The dense runtime itself, for what the backend does not offer (a direct generation, a snapshot); `None` for a
    /// Flash-Next engine. Do not keep the handle: [`shutdown`](EngineHost::shutdown) needs the last one.
    pub fn runtime(&self) -> Option<&Arc<EngineRuntime>> {
        match &self.held {
            Held::Dense { runtime, .. } => Some(runtime),
            Held::Flash { .. } => None,
        }
    }

    /// The engine's lattice: the server's frames run here.
    pub fn lattice(&self) -> &Arc<ControlPlaneLattice> {
        match &self.held {
            Held::Dense { runtime, .. } => runtime.lattice(),
            Held::Flash { runtime, .. } => runtime.lattice(),
        }
    }

    /// Stops the engine within `window`: the backend admits nothing more, then the lattice closes and the device
    /// memory is freed. What it could not finish in the window is reported (and not waited for): the error says what
    /// stays. Call it on the thread that started the engine, after the server has stopped and no generation runs.
    pub fn shutdown(self, window: Duration) -> Result<(), StartError> {
        self.backend.close();
        match self.held {
            Held::Dense { owner, runtime } => owner.shutdown_within(runtime, window).map_err(StartError::from),
            Held::Flash { owner, runtime } => FlashRuntime::shutdown(runtime, owner, window).map_err(StartError::from),
        }
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
    use std::io::Write;
    use std::path::PathBuf;

    use euhedral_inference_engine::runtime::Teardown;
    use euhedral_inference_engine::{Handoff, lattice_config};
    use euhedral_inference_settings::ModelSettings;

    use euhedral_inference_edrl::{FOOTER_BYTES, TOKENIZER_ENTRIES, TokenizerFooter};
    use serde_json::{Value, json};
    use tempfile::NamedTempFile;

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
        drop(DensePending {
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

    /// A temporary file holding `bytes`, removed when dropped.
    fn artifact_with(bytes: &[u8]) -> NamedTempFile {
        let mut file = NamedTempFile::new().expect("a temporary file");
        file.write_all(bytes).expect("the file is written");
        file
    }

    /// The first bytes of an artifact: the magic, the container version, the architecture.
    fn head(version: u32, architecture: u32) -> Vec<u8> {
        let mut bytes = 0x5157_454Eu32.to_be_bytes().to_vec();
        bytes.extend(version.to_be_bytes());
        bytes.extend(architecture.to_be_bytes());
        bytes
    }

    #[test]
    fn the_start_dispatches_on_the_family_the_artifact_names() {
        let dense = artifact_with(&head(2, 0));
        let flash = artifact_with(&head(3, 1));
        assert_eq!(family_at(dense.path()).expect("dense"), ModelFamily::DenseV2);
        assert_eq!(family_at(flash.path()).expect("flash"), ModelFamily::FlashNextV3);
    }

    #[test]
    fn an_artifact_of_a_family_this_engine_does_not_serve_is_refused_naming_it() {
        let unknown = artifact_with(&head(3, 9));
        let refused = family_at(unknown.path())
            .expect_err("an unknown architecture")
            .to_string();
        assert!(refused.contains("architecture 9"), "{refused}");
        assert!(refused.contains(ARTIFACT_KEY), "{refused}");
        let older = artifact_with(&head(1, 0));
        let refused = family_at(older.path()).expect_err("an unknown version").to_string();
        assert!(refused.contains("version 1"), "{refused}");
    }

    #[test]
    fn an_artifact_too_short_to_name_its_family_is_refused() {
        let short = artifact_with(&[0x51, 0x57]);
        let refused = family_at(short.path()).expect_err("a short file").to_string();
        assert!(refused.contains("cannot be read"), "{refused}");
    }

    #[test]
    fn a_flash_engine_whose_runtime_is_gone_refuses_requests_and_reports_nothing() {
        let engine = FlashServing {
            runtime: Weak::new(),
            artifact: None,
            context_length: 8,
            vocabulary_size: 4,
        };
        assert_eq!(engine.sequences(), 1);
        assert_eq!(engine.speculation(), Speculation::Off);
        assert_eq!(engine.report(), EngineReport::default());
    }
}

/// Proof runs on the real Flash-Next artifact through the server's engine path: [`begin_start`] dispatches on the
/// artifact's family, and the request carries the stop tokens the backend's codec gives (the test builds the rest itself).
#[cfg(test)]
#[allow(clippy::print_stdout, reason = "a proof run reports what it generated")]
mod flash_device {
    use std::mem;
    use std::path::PathBuf;
    use std::time::Instant;

    use euhedral_inference_engine::Handoff;
    use euhedral_inference_engine::generation::{GenerationListener, GenerationOutcome, QuantumEvent};
    use euhedral_inference_engine::sampling::SamplingConfig;
    use euhedral_inference_settings::{Auto, ByteSize, ModelSettings};

    use super::*;

    const WAIT: Duration = Duration::from_secs(40 * 60);
    const GENERATION: Duration = Duration::from_secs(15 * 60);

    /// `<|im_start|>user\nWhat is 17 times 23?<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n`, as the
    /// artifact's tokenizer encodes it.
    const PROMPT: [u32; 22] = [
        248_045, 846, 198, 3710, 369, 220, 16, 22, 2942, 220, 17, 18, 30, 248_046, 198, 248_045, 74_455, 198, 248_068,
        271, 248_069, 271,
    ];

    struct Recorder {
        tokens: Vec<u32>,
        done: Arc<Handoff<(Vec<u32>, GenerationOutcome)>>,
    }

    impl GenerationListener for Recorder {
        fn prefill_done(&mut self, _at: Instant) {}

        fn token(&mut self, id: u32, _at: Instant) {
            self.tokens.push(id);
        }

        fn quantum(&mut self, _event: QuantumEvent) {}

        fn finished(&mut self, outcome: GenerationOutcome) {
            self.done.send((mem::take(&mut self.tokens), outcome));
        }
    }

    fn run(engine: &dyn Engine, prompt: &[u32], wanted: u32, stops: &[u32]) -> (Vec<u32>, GenerationOutcome) {
        let done = Handoff::new();
        let request = EngineRequest {
            prompt: prompt.to_vec(),
            max_tokens: wanted,
            stop_tokens: stops.to_vec(),
            speculation: engine.speculation(),
            sampling: SamplingConfig::greedy(0),
            constraint: None,
            listener: Box::new(Recorder {
                tokens: Vec::new(),
                done: Arc::clone(&done),
            }),
        };
        engine
            .generate(request, CancelHandle::default())
            .expect("the engine takes the generation");
        done.wait(GENERATION).expect("the generation never finished")
    }

    #[test]
    #[ignore = "needs a GPU, the native library and the Flash-Next artifact"]
    fn the_server_serves_the_flash_next_artifact_with_the_same_greedy_tokens_every_time() {
        let Some(library) = test_support::require_property("euhedral.cuda.library") else {
            return;
        };
        let Some(artifact) = test_support::require_property("euhedral.qwen4.artifact") else {
            return;
        };
        let mut settings = Settings::with_model(ModelSettings {
            artifact: PathBuf::from(artifact),
            name: "flash".to_owned(),
        });
        settings.context.max_tokens = 4096;
        settings.memory.host_limit = Auto::Value(ByteSize::mib(24_000));
        let host = begin_start(&settings, Path::new(&library), None)
            .expect("the start begins")
            .wait(WAIT)
            .expect("the engine starts");
        assert!(
            host.runtime().is_none(),
            "the start did not dispatch to the Flash-Next engine"
        );
        let engine = Arc::clone(&host.backend.engine);
        let stops = host.backend.codec.stop_tokens();
        assert!(
            stops.contains(&248_044) && stops.contains(&248_046),
            "the model's end tokens: {stops:?}"
        );
        assert_eq!(engine.report().family, "flash-next");

        let (first, outcome) = run(engine.as_ref(), &PROMPT, 32, &stops);
        println!("first: {first:?} {:?}", outcome.reason);
        let (second, _) = run(engine.as_ref(), &PROMPT, 32, &stops);
        assert!(!first.is_empty(), "no token was generated");
        assert_eq!(first, second, "the same prompt generated different tokens");

        host.shutdown(Duration::from_secs(120)).expect("the engine stops");
    }
}
