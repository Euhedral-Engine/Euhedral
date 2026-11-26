//! The MTP drafter's training corpus: a directory of four append-only files.
//!
//! - `tokens.bin`: every sequence's tokens (int32, little endian), the prompt then the answer, the last token the one
//!   the base model chose but never ran;
//! - `labels.bin`: for each token of `tokens.bin`, the base model's greedy choice at its position (the token itself in
//!   the prompt): what a draft of that position should be;
//! - `hidden.bin`: the post-final-norm hidden (BF16, `width` values) of every row the base model ran: all of a
//!   sequence's tokens but the last;
//! - `index.jsonl`: one line per sequence, written after its rows: its id, first token and first hidden row, token
//!   count, prompt length and whether its answer was sampled (otherwise greedy, its labels its tokens).
//!
//! The format is the one the Java tools write and read. A corpus whose labels were not kept is greedy throughout:
//! opening it for writing makes its labels its tokens. A corpus left with rows beyond its index (a stopped run) is cut
//! back to the index.
//!
//! The writer's calls are blocking file calls. They are made from one frame of the host lake at a time, as the calls
//! of [`durable`](crate::durable) are, and end when `AsyncIo` has the offset write, the durability barrier and the
//! truncation they need; only the bodies of [`Writer::open`], [`Writer::append`] and [`entries`] change then.

#![allow(
    clippy::disallowed_methods,
    reason = "blocking I/O with no AsyncIo form yet, confined to this module: startup reads and opens, and one append at a time as a host lake frame whose completion continues the chain"
)]

use std::collections::HashSet;
use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
#[cfg(unix)]
use std::os::unix::fs::FileExt;
#[cfg(windows)]
use std::os::windows::fs::FileExt;
use std::path::Path;

use serde_json::Value;

const TOKENS: &str = "tokens.bin";
const LABELS: &str = "labels.bin";
const HIDDEN: &str = "hidden.bin";
const INDEX: &str = "index.jsonl";
/// Bytes of a token or a label in its file.
const TOKEN_BYTES: u64 = 4;
/// Bytes of a hidden value (BF16).
const VALUE_BYTES: u64 = 2;

/// One sequence of the corpus, as its index line states it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub id: String,
    /// Index of its first token in `tokens.bin`.
    pub token: u64,
    /// Index of its first hidden row in `hidden.bin`.
    pub row: u64,
    /// Its tokens, the prompt's and the answer's.
    pub length: u64,
    /// How many of them are the prompt.
    pub prompt: u64,
    pub sampled: bool,
}

/// A sequence to append: its tokens, their labels and the hidden row of every token but the last.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Sequence {
    pub id: String,
    pub tokens: Vec<u32>,
    /// The greedy choice at each position of `tokens`.
    pub labels: Vec<u32>,
    /// How many of `tokens` are the prompt.
    pub prompt: usize,
    pub sampled: bool,
    /// BF16 values, little endian, `width` of them for each row, `tokens.len() - 1` rows.
    pub hidden: Vec<u8>,
}

fn invalid(message: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

/// The index of the corpus in `directory`: empty when it has none.
pub fn entries(directory: &Path) -> io::Result<Vec<Entry>> {
    let text = match fs::read_to_string(directory.join(INDEX)) {
        Ok(text) => text,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(error) => return Err(error),
    };
    let mut entries = Vec::new();
    for (number, line) in text.lines().enumerate() {
        if line.trim().is_empty() {
            continue;
        }
        let node: Value =
            serde_json::from_str(line).map_err(|error| invalid(format!("{INDEX} line {number}: {error}")))?;
        let number_of = |key: &str| node.get(key).and_then(Value::as_u64).unwrap_or(0);
        entries.push(Entry {
            id: node.get("id").and_then(Value::as_str).unwrap_or_default().to_owned(),
            token: number_of("token"),
            row: number_of("row"),
            length: number_of("length"),
            prompt: number_of("prompt"),
            sampled: node.get("sampled").and_then(Value::as_bool).unwrap_or(false),
        });
    }
    Ok(entries)
}

/// The id as a JSON string body: the quote and the backslash escaped as the Java tools do, and the control
/// characters too, which they leave as they are and so write a line no reader accepts.
fn escape(id: &str) -> String {
    let mut text = String::with_capacity(id.len());
    for c in id.chars() {
        match c {
            '\\' => text.push_str("\\\\"),
            '"' => text.push_str("\\\""),
            c if u32::from(c) < 0x20 => text.push_str(&format!("\\u{:04x}", u32::from(c))),
            c => text.push(c),
        }
    }
    text
}

/// The index line of a sequence, with its newline.
fn index_line(entry: &Entry) -> String {
    format!(
        "{{\"id\":\"{}\",\"token\":{},\"row\":{},\"length\":{},\"prompt\":{},\"sampled\":{}}}\n",
        escape(&entry.id),
        entry.token,
        entry.row,
        entry.length,
        entry.prompt,
        entry.sampled
    )
}

#[cfg(unix)]
fn write_all_at(file: &File, bytes: &[u8], position: u64) -> io::Result<()> {
    file.write_all_at(bytes, position)
}

#[cfg(windows)]
fn write_all_at(file: &File, mut bytes: &[u8], mut position: u64) -> io::Result<()> {
    while !bytes.is_empty() {
        let written = file.seek_write(bytes, position)?;
        if written == 0 {
            return Err(io::ErrorKind::WriteZero.into());
        }
        bytes = &bytes[written..];
        position += written as u64;
    }
    Ok(())
}

#[cfg(unix)]
fn read_exact_at(file: &File, bytes: &mut [u8], position: u64) -> io::Result<()> {
    file.read_exact_at(bytes, position)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut bytes: &mut [u8], mut position: u64) -> io::Result<()> {
    while !bytes.is_empty() {
        let read = file.seek_read(bytes, position)?;
        if read == 0 {
            return Err(io::ErrorKind::UnexpectedEof.into());
        }
        bytes = &mut bytes[read..];
        position += read as u64;
    }
    Ok(())
}

/// Cuts `file` back to `length` bytes when it is longer; a shorter file is left as it is.
fn cut_back(file: &File, length: u64) -> io::Result<()> {
    if file.metadata()?.len() > length {
        file.set_len(length)?;
    }
    Ok(())
}

fn token_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// Appends sequences to a corpus, from one owner.
#[derive(Debug)]
pub struct Writer {
    tokens: File,
    labels: File,
    hidden: File,
    index: File,
    /// Hidden values in a row: 0 when the corpus is opened only to read its ids.
    width: usize,
    ids: HashSet<String>,
    token_count: u64,
    row_count: u64,
}

impl Writer {
    /// Opens the corpus in `directory` (made if it is not there) for rows of `width` hidden values (0: only to read
    /// its ids), cutting it back to its index.
    pub fn open(directory: &Path, width: usize) -> io::Result<Writer> {
        fs::create_dir_all(directory)?;
        let (mut ids, mut token_count, mut row_count) = (HashSet::new(), 0, 0);
        for entry in entries(directory)? {
            token_count = token_count.max(entry.token + entry.length);
            row_count = row_count.max((entry.row + entry.length).saturating_sub(1));
            ids.insert(entry.id);
        }
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);
        let tokens = options.open(directory.join(TOKENS))?;
        let labels = options.open(directory.join(LABELS))?;
        let hidden = options.open(directory.join(HIDDEN))?;
        let index = OpenOptions::new()
            .create(true)
            .append(true)
            .open(directory.join(INDEX))?;
        cut_back(&tokens, TOKEN_BYTES * token_count)?;
        cut_back(&labels, TOKEN_BYTES * token_count)?;
        if width > 0 {
            cut_back(&hidden, VALUE_BYTES * width as u64 * row_count)?;
        }
        // Sequences written before labels were kept were answered greedily: their labels are their tokens.
        let kept = labels.metadata()?.len();
        let wanted = TOKEN_BYTES * token_count;
        if kept < wanted {
            let mut missing =
                vec![0; usize::try_from(wanted - kept).map_err(|_| invalid("the corpus is too large".into()))?];
            read_exact_at(&tokens, &mut missing, kept)?;
            write_all_at(&labels, &missing, kept)?;
        }
        Ok(Writer {
            tokens,
            labels,
            hidden,
            index,
            width,
            ids,
            token_count,
            row_count,
        })
    }

    /// The ids already in the corpus.
    pub fn ids(&self) -> &HashSet<String> {
        &self.ids
    }

    /// Appends a sequence and makes it durable: its tokens, labels and hidden rows first, then its index line, so
    /// that an index line only ever names rows that are there.
    pub fn append(&mut self, sequence: &Sequence) -> io::Result<()> {
        let length = sequence.tokens.len();
        if length == 0 || sequence.labels.len() != length || sequence.prompt > length {
            return Err(invalid(format!(
                "sequence {}: {length} tokens, {} labels, a prompt of {}",
                sequence.id,
                sequence.labels.len(),
                sequence.prompt
            )));
        }
        let rows = length - 1;
        if sequence.hidden.len() != rows * self.width * VALUE_BYTES as usize {
            return Err(invalid(format!(
                "sequence {}: {} hidden bytes, {rows} rows of {} values need {}",
                sequence.id,
                sequence.hidden.len(),
                self.width,
                rows * self.width * VALUE_BYTES as usize
            )));
        }
        write_all_at(
            &self.tokens,
            &token_bytes(&sequence.tokens),
            TOKEN_BYTES * self.token_count,
        )?;
        write_all_at(
            &self.labels,
            &token_bytes(&sequence.labels),
            TOKEN_BYTES * self.token_count,
        )?;
        write_all_at(
            &self.hidden,
            &sequence.hidden,
            VALUE_BYTES * self.width as u64 * self.row_count,
        )?;
        self.tokens.sync_data()?;
        self.labels.sync_data()?;
        self.hidden.sync_data()?;
        let entry = Entry {
            id: sequence.id.clone(),
            token: self.token_count,
            row: self.row_count,
            length: length as u64,
            prompt: sequence.prompt as u64,
            sampled: sequence.sampled,
        };
        self.index.write_all(index_line(&entry).as_bytes())?;
        self.index.sync_data()?;
        self.ids.insert(sequence.id.clone());
        self.token_count += length as u64;
        self.row_count += rows as u64;
        Ok(())
    }
}
