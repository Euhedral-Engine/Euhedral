//! What the quality commands share: the engine settings of a tool run, and the few blocking file calls of the
//! binary's main thread.

use std::fs::OpenOptions;
use std::io::Write;
use std::path::Path;

use euhedral_inference_settings::{ByteSize, ModelSettings, Settings, load_file, process_env};

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
