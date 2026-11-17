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
use euhedral_inference_engine::error::EngineError;
use euhedral_inference_engine::generation::{
    AdmissionError, CancelHandle, GenerationRequest as EngineRequest, MAX_VERIFIED, Speculation,
};
use euhedral_inference_engine::runtime::{EngineRuntime, ReadyHandoff, RuntimeOwner};
use euhedral_inference_settings::{Settings, variable_name};
use euhedral_inference_tokenizer::{TokenStream, Tokenizer, read_artifact_tokenizer};

use super::{Codec, Engine, EngineBackend, TextStream};
use crate::backend::EngineReport;

/// The settings key that names the artifact.
const ARTIFACT_KEY: &str = "model.artifact";

/// Why the engine did not start; the text is for the operator, in the words of the startup checks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum StartError {
    /// The artifact is not there or cannot be read.
    Setup(String),
    /// The artifact's tokenizer could not be read.
    Tokenizer(String),
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

    fn sequences(&self) -> usize {
        // The engine runs one generation at a time, whatever `batching.max-sequences` asks for.
        1
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
    let tokenizer =
        read_artifact_tokenizer(&settings.model.artifact).map_err(|error| StartError::Tokenizer(error.to_string()))?;
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
        });
        let backend = Arc::new(EngineBackend::new(
            engine,
            Arc::new(self.tokenizer),
            sink,
            self.settings.model.name.clone(),
        ));
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

#[cfg(test)]
mod tests {
    use std::cell::Cell;
    use std::path::PathBuf;

    use euhedral_inference_engine::runtime::Teardown;
    use euhedral_inference_engine::{Handoff, lattice_config};
    use euhedral_inference_settings::ModelSettings;

    use super::*;

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
