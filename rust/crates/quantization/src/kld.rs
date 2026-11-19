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
use std::io;
use std::path::Path;
use std::sync::Arc;

use euhedral_inference_engine::Handoff;
use euhedral_io::{AsyncIo, IoDoneOwned, OpenSpec, OwnedFile};
use thiserror::Error;

use crate::rows::{Kind, frame, raw_file};

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

    /// Starts opening the file at `path`. The hand-off is set with the reference and the file `AsyncIo` reads the
    /// records from, or with why not. The header and the tokens are read, and the length of the file checked against
    /// the records it must hold; the records are not read. Nothing waits here: the caller waits on the hand-off, from
    /// outside the lattice.
    pub fn open(io: &Arc<AsyncIo>, path: &Path) -> OpenedHandoff {
        let handoff = Handoff::new();
        let delivery = Delivery::new(&handoff, path);
        let next = Arc::clone(io);
        let then = Box::new(move |opened: io::Result<OwnedFile>| {
            frame(Kind::Reading, 0, move || match opened {
                Ok(file) => read_header(&next, file, delivery),
                Err(error) => delivery.fail("cannot open", error),
            })
        });
        if let Err(error) = io.open_path(path, OpenSpec::default(), then) {
            // The continuation was dropped on the way out, and its delivery reported that: replace it.
            let _ = handoff.take();
            handoff.send(Err(KldError::Io(format!(
                "{}: cannot start opening: {error}",
                path.display()
            ))));
        }
        handoff
    }
}

/// What opening a reference gives.
pub type Opened = Result<(Reference, OwnedFile), KldError>;

/// The hand-off of an [`Opened`].
pub type OpenedHandoff = Arc<Handoff<Opened>>;

/// The way back to the caller of [`Reference::open`]. A delivery dropped without a result reports that, so the caller
/// never waits for a read that was lost.
struct Delivery {
    handoff: OpenedHandoff,
    sent: bool,
    path: String,
}

impl Delivery {
    fn new(handoff: &OpenedHandoff, path: &Path) -> Delivery {
        Delivery {
            handoff: Arc::clone(handoff),
            sent: false,
            path: path.display().to_string(),
        }
    }

    fn handoff(&self) -> OpenedHandoff {
        Arc::clone(&self.handoff)
    }

    fn send(mut self, result: Opened) {
        self.sent = true;
        self.handoff.send(result);
    }

    fn fail(self, what: &str, error: io::Error) {
        let message = format!("{}: {what}: {error}", self.path);
        self.send(Err(KldError::Io(message)));
    }
}

impl Drop for Delivery {
    fn drop(&mut self) {
        if !self.sent {
            self.handoff
                .send(Err(KldError::Io(format!("{}: the read was lost", self.path))));
        }
    }
}

/// Reads `wanted` bytes at `offset` of `file`; `next` gets them in a frame of its own. A file that ends first is
/// [`KldError::Truncated`] for `what`, with `before` bytes already known to be there.
fn read_part(
    io: &Arc<AsyncIo>,
    file: OwnedFile,
    offset: u64,
    wanted: usize,
    what: &'static str,
    delivery: Delivery,
    next: impl FnOnce(OwnedFile, Vec<u8>, Delivery) + Send + 'static,
) {
    let before = offset as usize;
    let raw = raw_file(&file);
    let lost = delivery.handoff();
    let path = delivery.path.clone();
    let then = Box::new(move |done: IoDoneOwned<Vec<u8>>| {
        frame(Kind::Reading, offset as usize, move || match (done.result, done.buf) {
            (Ok(read), Some(bytes)) if read == wanted => next(file, bytes, delivery),
            (Ok(read), _) => delivery.send(Err(KldError::Truncated {
                what,
                have: before + read,
                need: before + wanted,
            })),
            (Err(error), _) if error.kind() == io::ErrorKind::UnexpectedEof => {
                delivery.send(Err(KldError::Truncated {
                    what,
                    have: before,
                    need: before + wanted,
                }))
            }
            (Err(error), _) => delivery.fail(&format!("cannot read {what}"), error),
        })
    });
    // The file is held by the continuation, which lives until the read ends.
    if let Err(error) = io.read_into(raw, offset, vec![0u8; wanted], then) {
        // The continuation was dropped on the way out, and its delivery reported that: replace it.
        let _ = lost.take();
        lost.send(Err(KldError::Io(format!(
            "{path}: cannot start reading {what}: {error}"
        ))));
    }
}

/// The first step: the header says how many tokens follow.
fn read_header(io: &Arc<AsyncIo>, file: OwnedFile, delivery: Delivery) {
    let next = Arc::clone(io);
    read_part(
        io,
        file,
        0,
        HEADER_BYTES,
        "the header",
        delivery,
        move |file, header, delivery| match Reference::parse_header(&header) {
            Ok((context, _, chunks)) => {
                read_tokens(&next, file, header, Reference::records_start(context, chunks), delivery)
            }
            Err(error) => delivery.send(Err(error)),
        },
    );
}

/// The second step: the tokens, and then the length check.
fn read_tokens(io: &Arc<AsyncIo>, file: OwnedFile, mut head: Vec<u8>, start: usize, delivery: Delivery) {
    read_part(
        io,
        file,
        HEADER_BYTES as u64,
        start - HEADER_BYTES,
        "the chunks' tokens",
        delivery,
        move |file, tokens, delivery| {
            head.extend_from_slice(&tokens);
            let reference = match Reference::parse(&head) {
                Ok(reference) => reference,
                Err(error) => return delivery.send(Err(error)),
            };
            let need = start as u64 + reference.row_bytes() as u64 * (reference.chunks() * reference.records()) as u64;
            let (length, file) = file_len(file);
            match length {
                Ok(length) if length >= need => delivery.send(Ok((reference, file))),
                Ok(length) => delivery.send(Err(KldError::Truncated {
                    what: "the records",
                    have: length as usize,
                    need: need as usize,
                })),
                Err(error) => delivery.fail("cannot read the size", error),
            }
        },
    );
}

/// The length of `file`, and the file back.
fn file_len(file: OwnedFile) -> (io::Result<u64>, OwnedFile) {
    let file = File::from(file);
    (file.metadata().map(|metadata| metadata.len()), OwnedFile::from(file))
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
