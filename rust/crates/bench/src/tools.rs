//! What the quality commands share: the engine settings of a tool run, and the few blocking file calls of the
//! binary's main thread.

use std::fmt::Display;
use std::fs::OpenOptions;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_inference_engine::lattice_config;
use euhedral_inference_quantization::kld::{KldError, Reference, open_records};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings, load_file, process_env};
use euhedral_io::{AsyncIo, OwnedFile};

use crate::await_value;
use crate::config::ConfigError;

fn invalid<T>(message: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError::Invalid(message.into()))
}

/// The engine settings of a quality run: the settings file `config` names (or the defaults), with the artifact, a
/// context that holds `context_tokens` tokens and no prefix cache, which a fresh sequence per run does not use.
pub fn tool_settings(artifact: &Path, config: Option<&Path>, context_tokens: usize) -> Result<Settings, ConfigError> {
    let model = ModelSettings {
        artifact: artifact.to_path_buf(),
        name: artifact
            .file_stem()
            .map_or_else(|| "model".to_string(), |stem| stem.to_string_lossy().into_owned()),
    };
    let mut settings = match config {
        None => Settings::with_model(model.clone()),
        Some(path) => {
            load_file(path, &process_env).or_else(|errors| invalid(format!("{}: {errors}", path.display())))?
        }
    };
    settings.model = model;
    settings.context.max_tokens = u32::try_from(context_tokens.next_multiple_of(512))
        .or_else(|_| invalid(format!("a context of {context_tokens} tokens is too long")))?;
    settings.prefix_cache.size = ByteSize::mib(0);
    Ok(settings)
}

/// Appends `text` to the file at `path`, creating it.
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: the summary is appended once per reference, on the main thread, between runs"
)]
pub fn append_text(path: &Path, text: &str) -> Result<(), ConfigError> {
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .or_else(|error| invalid(format!("{}: {error}", path.display())))?;
    file.write_all(text.as_bytes())
        .or_else(|error| invalid(format!("{}: {error}", path.display())))
}

/// Reads the references at `paths` (the headers and tokens, not the records) through `AsyncIo`, on a lattice of its
/// own that lives only for the reads: a run's context, which its engine's settings need, is known only once the
/// headers are read. Waits for the reads: called by the binary's main thread only.
pub fn read_references(paths: &[PathBuf]) -> Result<Vec<Reference>, KldError> {
    let io_error = |what: &str, error: &dyn Display| KldError::Io(format!("{what}: {error}"));
    let lattice = ControlPlaneLattice::new(lattice_config(&Settings::with_model(ModelSettings::default())))
        .map_err(|error| io_error("the lattice for the reference files would not start", &error))?;
    let io = AsyncIo::open().map_err(|error| io_error("the asynchronous file I/O would not open", &error))?;
    lattice
        .add_upstream(Arc::clone(&io) as Arc<dyn LatticeSource>)
        .map_err(|error| io_error("the lattice would not take the asynchronous file I/O", &error))?;
    let pending: Vec<_> = paths.iter().map(|path| Reference::read(&io, path)).collect();
    let read = pending.iter().map(|handoff| await_value(handoff)).collect();
    io.complete();
    lattice.close();
    read
}

/// Opens the record files at `paths` through `io`, the `AsyncIo` that will read them (a file belongs to the one that
/// opened it). Waits for the opens: called by the binary's main thread only.
pub fn open_record_files(io: &Arc<AsyncIo>, paths: &[PathBuf]) -> Result<Vec<Arc<OwnedFile>>, KldError> {
    let pending: Vec<_> = paths.iter().map(|path| open_records(io, path)).collect();
    pending
        .iter()
        .map(|handoff| await_value(handoff).map(Arc::new))
        .collect()
}
