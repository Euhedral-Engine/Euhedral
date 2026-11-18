/// Why a tokenizer could not be built, or a token could not be decoded.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    /// The artifact's tokenizer section lacks one of its files.
    #[error("the tokenizer section has no {file}")]
    MissingFile { file: &'static str },
    /// A file of the section is not the JSON it must be.
    #[error("{file} is not valid: {source}")]
    Json {
        file: &'static str,
        source: serde_json::Error,
    },
    /// The tokenizer is not the byte-level BPE with NFC normalization that Qwen checkpoints use, or its metadata is
    /// inconsistent.
    #[error("unsupported tokenizer: {0}")]
    Unsupported(String),
    /// The `tokenizers` library refused the tokenizer or a text.
    #[error("tokenizer library: {0}")]
    Library(String),
    /// A token ID that the vocabulary does not hold.
    #[error("unknown token ID {0}")]
    UnknownToken(u32),
}

impl TokenizerError {
    pub(crate) fn unsupported(message: impl Into<String>) -> TokenizerError {
        TokenizerError::Unsupported(message.into())
    }
}
