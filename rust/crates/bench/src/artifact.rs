//! The tokenizer the artifact packages, read from its tokenizer section: the same tokenizer the engine's prompts are
//! encoded with, so the benchmark needs no checkpoint directory.

use std::path::Path;

use euhedral_inference_tokenizer::{Tokenizer, read_artifact_tokenizer};

use crate::config::ConfigError;

/// Reads the tokenizer of the artifact at `path`.
pub fn load_tokenizer(path: &Path) -> Result<Tokenizer, ConfigError> {
    read_artifact_tokenizer(path).map_err(|error| ConfigError::Invalid(format!("{}: {error}", path.display())))
}
