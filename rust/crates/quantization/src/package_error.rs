//! Why packaging a tokenizer failed.

use std::io;
use std::path::PathBuf;

use euhedral_inference_edrl::EdrlError;

/// A failed `package-tokenizer`: which step, and why.
#[derive(Debug, thiserror::Error)]
pub enum PackageError {
    /// A file of the checkpoint or the artifact could not be read, written or moved.
    #[error("{what}: {source}")]
    Io {
        what: &'static str,
        #[source]
        source: io::Error,
    },
    /// A file of the checkpoint is not UTF-8 text.
    #[error("{file} is not UTF-8 text")]
    NotText { file: &'static str },
    /// A file of the checkpoint is not JSON.
    #[error("{file} is not valid JSON: {source}")]
    Json {
        file: &'static str,
        #[source]
        source: serde_json::Error,
    },
    /// The checkpoint has neither `chat_template.jinja` nor a `chat_template` string in `tokenizer_config.json`.
    #[error(
        "the checkpoint has no chat template: no chat_template.jinja and no chat_template string in tokenizer_config.json"
    )]
    NoChatTemplate,
    /// The encoded section does not read back as a section.
    #[error("the encoded tokenizer section does not read back")]
    Section,
    /// The tokenizer does not load from the checkpoint's files.
    #[error("the tokenizer does not load: {0}")]
    Tokenizer(String),
    /// The chat template is not the one the server renders.
    #[error("the chat template is not the one the server renders: {0}")]
    Template(String),
    /// The artifact, or the section, is not valid.
    #[error("the artifact is not valid: {0}")]
    Artifact(#[from] EdrlError),
    /// The artifact is too short or too large for its format.
    #[error("the artifact is not valid: {0}")]
    Format(&'static str),
    /// A step of the run was dropped before it ran: the I/O refused it or the host lake closed.
    #[error("the run ended before it finished: a step was dropped")]
    Lost,
    /// An artifact written in place was cut back to its content and the section did not land: it has no tokenizer
    /// section now, and running the command again writes it.
    #[error("the artifact was cut and has no tokenizer section now; run the command again to write it ({0})")]
    Cut(Box<PackageError>),
    /// The run failed and the temporary output could not be removed: the file is left.
    #[error("{cause}; the temporary output {} could not be removed and was left ({removal})", part.display())]
    PartLeft {
        part: PathBuf,
        removal: io::Error,
        cause: Box<PackageError>,
    },
    /// The host lake would not take a frame of the run.
    #[error("the host lake refused work: {0}")]
    Lake(String),
}

impl PackageError {
    /// An I/O failure of the step `what`.
    pub fn io(what: &'static str) -> impl FnOnce(io::Error) -> PackageError {
        move |source| PackageError::Io { what, source }
    }
}
