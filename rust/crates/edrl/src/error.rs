use thiserror::Error;

/// Why a region of an artifact was refused.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum EdrlError {
    /// The bytes ended before `what` was complete.
    #[error("artifact is truncated while reading {what}")]
    Truncated { what: &'static str },
    /// The first four bytes are not the EDRL magic.
    #[error("unsupported artifact magic {found:#010x}")]
    BadMagic { found: u32 },
    /// Only container version 2 (tensor descriptors carry a persistent weight layout) is supported.
    #[error("unsupported artifact version {found}: only compact version 2 artifacts are supported")]
    UnsupportedVersion { found: u32 },
    /// A region is malformed: `what` is the first violated rule.
    #[error("{what}")]
    Invalid { what: String },
    /// A tensor descriptor names a format/layout/shape combination the engine cannot size.
    #[error("unsupported compact tensor '{name}': {reason}")]
    UnsupportedTensor { name: String, reason: String },
    /// A tensor's byte size does not match its shape, format and layout.
    #[error("compact tensor byte size does not match its shape, format, and layout: {name} was {bytes}")]
    SizeMismatch { name: String, bytes: u64 },
    /// Two tensor payloads share bytes.
    #[error("tensor data ranges overlap: {first} and {second}")]
    Overlap { first: String, second: String },
    /// Two tensors share a name.
    #[error("tensor name appears twice: {name}")]
    DuplicateName { name: String },
    /// The tokenizer section's footer names an unsupported version.
    #[error(
        "unsupported tokenizer section version {found}; running package-tokenizer on the artifact again replaces the section"
    )]
    TokenizerVersion { found: u32 },
    /// The tokenizer section's recorded CRC-32 differs from the body's.
    #[error("the tokenizer section fails its CRC-32")]
    TokenizerCrc,
}

impl EdrlError {
    pub(crate) fn invalid(what: impl Into<String>) -> Self {
        Self::Invalid { what: what.into() }
    }
}
