//! The reference logits file of `llama-perplexity --kl-divergence-base` (the layout the Java scorer and reference
//! writer share):
//!
//! ```text
//! "_logits_"                                  8 bytes
//! i32 context, i32 vocabulary, i32 chunks     little endian
//! chunks * context i32 token ids              chunk after chunk
//! per chunk, records of the chunk's second half (context - 1 - context / 2 of them), one per forced token:
//!     f32 scale, f32 minimum, then 2 * ceil(vocabulary / 2) u16 codes  (a code per token, padded to an even count)
//! ```
//!
//! A record holds the reference's log-probabilities over a 16-nat window: `log p = scale * code + minimum`, computed in
//! float64. Record `k` of a chunk is the reference's prediction of the token at position `context / 2 + 1 + k`.

use std::fs::File;
use std::io::{self, Read};
use std::path::Path;

use euhedral_graph::OwnedFile;
use thiserror::Error;

/// The first bytes of the file.
pub const MAGIC: &[u8; 8] = b"_logits_";
/// The magic and the three counts.
pub const HEADER_BYTES: usize = 20;
/// The window of log-probabilities a reference row holds: entries at or below it are the floor, and carry no mass.
pub const FLOOR: f64 = -16.0;

/// Why a file is not a reference.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum KldError {
    #[error("not a llama.cpp logits file: it does not start with \"_logits_\"")]
    NotReference,
    #[error("the file ends inside {what}: {have} of {need} bytes")]
    Truncated {
        what: &'static str,
        have: usize,
        need: usize,
    },
    #[error("{0}")]
    Invalid(String),
    #[error("{0}")]
    Io(String),
}

/// A reference: its dimensions and the tokens of its chunks. The records stay in the file.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reference {
    pub context: usize,
    pub vocabulary: usize,
    /// `tokens[chunk]` holds `context` token ids.
    pub tokens: Vec<Vec<u32>>,
}

impl Reference {
    /// The bytes before the first record for a file of `chunks` chunks of `context` tokens.
    pub fn records_start(context: usize, chunks: usize) -> usize {
        HEADER_BYTES + 4 * chunks * context
    }

    /// The dimensions in a header: context, vocabulary, chunks.
    pub fn parse_header(bytes: &[u8]) -> Result<(usize, usize, usize), KldError> {
        if bytes.len() < HEADER_BYTES {
            return Err(KldError::Truncated {
                what: "the header",
                have: bytes.len(),
                need: HEADER_BYTES,
            });
        }
        if &bytes[..8] != MAGIC {
            return Err(KldError::NotReference);
        }
        let count = |at: usize| {
            let value = i32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]]);
            usize::try_from(value).map_err(|_| KldError::Invalid(format!("a negative count {value} in the header")))
        };
        let (context, vocabulary, chunks) = (count(8)?, count(12)?, count(16)?);
        if context < 2 || vocabulary == 0 || chunks == 0 {
            return Err(KldError::Invalid(format!(
                "the header names {chunks} chunks of {context} tokens over a vocabulary of {vocabulary}"
            )));
        }
        Ok((context, vocabulary, chunks))
    }

    /// The reference whose header and tokens are `bytes` (the file's first [`Reference::records_start`] bytes).
    pub fn parse(bytes: &[u8]) -> Result<Reference, KldError> {
        let (context, vocabulary, chunks) = Self::parse_header(bytes)?;
        let start = Self::records_start(context, chunks);
        if bytes.len() < start {
            return Err(KldError::Truncated {
                what: "the chunks' tokens",
                have: bytes.len(),
                need: start,
            });
        }
        let tokens = (0..chunks)
            .map(|chunk| {
                (0..context)
                    .map(|index| {
                        let at = HEADER_BYTES + 4 * (chunk * context + index);
                        u32::from_le_bytes([bytes[at], bytes[at + 1], bytes[at + 2], bytes[at + 3]])
                    })
                    .collect()
            })
            .collect();
        Ok(Reference {
            context,
            vocabulary,
            tokens,
        })
    }

    /// The chunks.
    pub fn chunks(&self) -> usize {
        self.tokens.len()
    }

    /// Tokens of a chunk the model is given before the forced ones: `context / 2 + 1`.
    pub fn prefix_len(&self) -> usize {
        self.context / 2 + 1
    }

    /// Forced tokens, and records, of a chunk: `context - 1 - context / 2`.
    pub fn records(&self) -> usize {
        self.context - 1 - self.context / 2
    }

    /// Bytes of one record: two f32 and a u16 code per token, padded to an even count of codes.
    pub fn row_bytes(&self) -> usize {
        8 + 2 * (2 * self.vocabulary.div_ceil(2))
    }

    /// The file offset of record `step` of `chunk`.
    pub fn row_offset(&self, chunk: usize, step: usize) -> u64 {
        Self::records_start(self.context, self.chunks()) as u64
            + self.row_bytes() as u64 * (chunk * self.records() + step) as u64
    }

    /// The prefix of `chunk`.
    pub fn prefix(&self, chunk: usize) -> &[u32] {
        &self.tokens[chunk][..self.prefix_len()]
    }

    /// The forced tokens of `chunk`: the targets of its records.
    pub fn forced(&self, chunk: usize) -> &[u32] {
        &self.tokens[chunk][self.prefix_len()..self.prefix_len() + self.records()]
    }

    /// Opens the file at `path`: the reference, and the file for `AsyncIo` to read the records from. The bytes before
    /// the records are read here; the records are not.
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, blocking I/O: the header and the tokens are read once, on the main thread, before the run"
    )]
    pub fn open(path: &Path) -> Result<(Reference, OwnedFile), KldError> {
        let io = |what: &str, error: io::Error| KldError::Io(format!("{}: {what}: {error}", path.display()));
        let mut file = File::open(path).map_err(|error| io("cannot open", error))?;
        let length = file
            .metadata()
            .map_err(|error| io("cannot read the size", error))?
            .len();
        let mut header = [0u8; HEADER_BYTES];
        file.read_exact(&mut header).map_err(|error| match error.kind() {
            io::ErrorKind::UnexpectedEof => KldError::Truncated {
                what: "the header",
                have: length as usize,
                need: HEADER_BYTES,
            },
            _ => io("cannot read the header", error),
        })?;
        let (context, _, chunks) = Self::parse_header(&header)?;
        let start = Self::records_start(context, chunks);
        let mut head = header.to_vec();
        head.resize(start, 0);
        file.read_exact(&mut head[HEADER_BYTES..])
            .map_err(|error| match error.kind() {
                io::ErrorKind::UnexpectedEof => KldError::Truncated {
                    what: "the chunks' tokens",
                    have: length as usize,
                    need: start,
                },
                _ => io("cannot read the chunks' tokens", error),
            })?;
        let reference = Self::parse(&head)?;
        let need = start as u64 + reference.row_bytes() as u64 * (chunks * reference.records()) as u64;
        if length < need {
            return Err(KldError::Truncated {
                what: "the records",
                have: length as usize,
                need: need as usize,
            });
        }
        let records = records_file(path, file).map_err(|error| io("cannot open for asynchronous reads", error))?;
        Ok((reference, records))
    }
}

/// The file `AsyncIo` reads the records from: the same file on Unix.
#[cfg(unix)]
fn records_file(_path: &Path, file: File) -> io::Result<OwnedFile> {
    Ok(OwnedFile::from(file))
}

/// The file `AsyncIo` reads the records from: on Windows a second handle opened for overlapped I/O, which `AsyncIo`
/// requires and the synchronous header reads cannot use.
#[cfg(windows)]
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: opened once, on the main thread, before the run"
)]
fn records_file(path: &Path, _file: File) -> io::Result<OwnedFile> {
    use std::fs::OpenOptions;
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
    let file = OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_OVERLAPPED)
        .open(path)?;
    Ok(OwnedFile::from(file))
}

/// One record: the reference's log-probabilities of a step.
#[derive(Clone, Copy, Debug)]
pub struct Row<'a> {
    scale: f64,
    minimum: f64,
    codes: &'a [u8],
}

impl<'a> Row<'a> {
    /// The record in `bytes` (a whole record of `vocabulary` codes).
    pub fn new(bytes: &'a [u8], vocabulary: usize) -> Row<'a> {
        let word = |at: usize| {
            f64::from(f32::from_le_bytes([
                bytes[at],
                bytes[at + 1],
                bytes[at + 2],
                bytes[at + 3],
            ]))
        };
        Row {
            scale: word(0),
            minimum: word(4),
            codes: &bytes[8..8 + 2 * vocabulary],
        }
    }

    /// The log-probability of `token`.
    pub fn log_p(&self, token: usize) -> f64 {
        let code = u16::from_le_bytes([self.codes[2 * token], self.codes[2 * token + 1]]);
        self.scale * f64::from(code) + self.minimum
    }
}

/// A reference file's bytes, for tests: `tokens[chunk]` and a record per forced token from `row`.
pub fn encode(
    context: usize,
    vocabulary: usize,
    tokens: &[Vec<u32>],
    mut row: impl FnMut(usize, usize) -> (f32, f32, Vec<u16>),
) -> Vec<u8> {
    let mut bytes = MAGIC.to_vec();
    for count in [context, vocabulary, tokens.len()] {
        bytes.extend_from_slice(&(count as i32).to_le_bytes());
    }
    for chunk in tokens {
        for token in chunk {
            bytes.extend_from_slice(&(*token as i32).to_le_bytes());
        }
    }
    let records = context - 1 - context / 2;
    for chunk in 0..tokens.len() {
        for step in 0..records {
            let (scale, minimum, mut codes) = row(chunk, step);
            codes.resize(2 * vocabulary.div_ceil(2), 0);
            bytes.extend_from_slice(&scale.to_le_bytes());
            bytes.extend_from_slice(&minimum.to_le_bytes());
            for code in codes {
                bytes.extend_from_slice(&code.to_le_bytes());
            }
        }
    }
    bytes
}
