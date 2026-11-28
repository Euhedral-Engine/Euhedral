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
//! The format is line-oriented JSON. A corpus whose labels were not kept is greedy throughout:
//! opening it for writing makes its labels its tokens. A corpus left with rows beyond its index (a stopped run) is cut
//! back to the index.
//!
//! Appends go through `AsyncIo` (the offset writes) and [`durable`](crate::durable) (the durability barrier), one
//! append at a time, each step the completion of the one before; nothing waits. Opening a corpus is the one blocking
//! part: it reads the index, cuts the files back and opens them, on the thread that calls it, before a run starts
//! (see [`startup`]).

use std::collections::{HashSet, VecDeque};
use std::fs::File;
use std::io::{self, Error as IoError};
use std::path::Path;
use std::sync::Arc;

use euhedral_graph::Lake;
use euhedral_inference_engine::Handoff;
use euhedral_io::{AsyncIo, IoBuf, IoDone, RawFile};

use crate::durable::{self, SharedFile};
use crate::step::{Kind, step};

pub use startup::entries;

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

/// The id as a JSON string body: the quote and the backslash escaped, and the control
/// characters too.
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

fn token_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

#[cfg(unix)]
fn raw_file(file: &File) -> RawFile {
    use std::os::fd::AsRawFd;

    file.as_raw_fd()
}

#[cfg(windows)]
fn raw_file(file: &File) -> RawFile {
    use std::os::windows::io::AsRawHandle;

    file.as_raw_handle()
}

/// What follows an append: the writer, back, and whether the sequence is in the corpus and durable.
pub type Appended = Box<dyn FnOnce(Writer, io::Result<()>) + Send>;

/// Appends sequences to a corpus, from one owner.
#[derive(Debug)]
pub struct Writer {
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
    /// `tokens.bin`, `labels.bin`, `hidden.bin` and `index.jsonl`, in that order.
    files: [SharedFile; 4],
    /// Hidden values in a row: 0 when the corpus is opened only to read its ids.
    width: usize,
    ids: HashSet<String>,
    token_count: u64,
    row_count: u64,
    index_len: u64,
}

/// One step of an append.
enum Op {
    Write { file: usize, offset: u64, bytes: Vec<u8> },
    Sync { file: usize },
}

/// An append under way: the steps left, and what the corpus holds once they are done.
struct Append {
    writer: Writer,
    ops: VecDeque<Op>,
    id: String,
    tokens: u64,
    rows: u64,
    index_len: u64,
    then: Appended,
}

impl Writer {
    /// Opens the corpus in `directory` (made if it is not there) for rows of `width` hidden values (0: only to read
    /// its ids), cutting it back to its index. Appends go through `io`, their durability barriers through `host`.
    pub fn open(directory: &Path, width: usize, io: Arc<AsyncIo>, host: Arc<Lake>) -> io::Result<Writer> {
        let opened = startup::open(directory, width)?;
        Ok(Writer {
            io,
            host,
            files: opened.files,
            width,
            ids: opened.ids,
            token_count: opened.token_count,
            row_count: opened.row_count,
            index_len: opened.index_len,
        })
    }

    /// The ids already in the corpus.
    pub fn ids(&self) -> &HashSet<String> {
        &self.ids
    }

    /// Appends a sequence and makes it durable: its tokens, labels and hidden rows first, then its index line, so
    /// that an index line only ever names rows that are there. `then` is called once, from a frame, with the writer
    /// back; a sequence refused before anything is written calls it at once. `then` never runs if the host lake or
    /// the `AsyncIo` closes first.
    pub fn append(self, sequence: Sequence, then: Appended) {
        let length = sequence.tokens.len();
        if length == 0 || sequence.labels.len() != length || sequence.prompt > length {
            let why = invalid(format!(
                "sequence {}: {length} tokens, {} labels, a prompt of {}",
                sequence.id,
                sequence.labels.len(),
                sequence.prompt
            ));
            return then(self, Err(why));
        }
        let rows = length as u64 - 1;
        let hidden_bytes = rows * self.width as u64 * VALUE_BYTES;
        if sequence.hidden.len() as u64 != hidden_bytes {
            let why = invalid(format!(
                "sequence {}: {} hidden bytes, {rows} rows of {} values need {hidden_bytes}",
                sequence.id,
                sequence.hidden.len(),
                self.width,
            ));
            return then(self, Err(why));
        }
        let entry = Entry {
            id: sequence.id.clone(),
            token: self.token_count,
            row: self.row_count,
            length: length as u64,
            prompt: sequence.prompt as u64,
            sampled: sequence.sampled,
        };
        let line = index_line(&entry).into_bytes();
        let index_len = self.index_len + line.len() as u64;
        let writes = [
            (0, TOKEN_BYTES * self.token_count, token_bytes(&sequence.tokens)),
            (1, TOKEN_BYTES * self.token_count, token_bytes(&sequence.labels)),
            (2, VALUE_BYTES * self.width as u64 * self.row_count, sequence.hidden),
        ];
        let mut ops = VecDeque::new();
        for (file, offset, bytes) in writes {
            // A sequence of one token ran no row: it has no hidden bytes, and a write needs some.
            if !bytes.is_empty() {
                ops.push_back(Op::Write { file, offset, bytes });
            }
        }
        ops.extend((0..3).map(|file| Op::Sync { file }));
        ops.push_back(Op::Write {
            file: 3,
            offset: self.index_len,
            bytes: line,
        });
        ops.push_back(Op::Sync { file: 3 });
        Box::new(Append {
            tokens: entry.token + entry.length,
            rows: entry.row + rows,
            id: sequence.id,
            index_len,
            writer: self,
            ops,
            then,
        })
        .advance();
    }
}

impl Append {
    /// Starts the next step, whose completion is the step after it; the end calls `then`.
    fn advance(mut self: Box<Self>) {
        let Some(op) = self.ops.pop_front() else {
            return self.finish(Ok(()));
        };
        let cell = Handoff::new();
        match op {
            Op::Write { file, offset, bytes } => {
                let (io, raw) = (Arc::clone(&self.writer.io), raw_file(&self.writer.files[file]));
                let filled = bytes.len();
                cell.send(self);
                let started = Arc::clone(&cell);
                let submitted = io.write(
                    raw,
                    offset,
                    IoBuf { bytes, filled },
                    Box::new(move |written: IoDone| {
                        step(Kind::Corpus, move || {
                            let Some(append) = cell.take() else { return };
                            match written.result {
                                Ok(moved) if moved == filled => append.advance(),
                                Ok(moved) => append.finish(Err(IoError::new(
                                    io::ErrorKind::WriteZero,
                                    format!("a write moved {moved} of {filled} bytes"),
                                ))),
                                Err(error) => append.finish(Err(error)),
                            }
                        })
                    }),
                );
                if let Err(error) = submitted
                    && let Some(append) = started.take()
                {
                    append.finish(Err(error));
                }
            }
            Op::Sync { file } => {
                let (host, shared) = (Arc::clone(&self.writer.host), Arc::clone(&self.writer.files[file]));
                cell.send(self);
                let started = Arc::clone(&cell);
                let submitted = durable::sync(
                    &host,
                    &shared,
                    Box::new(move |result| {
                        step(Kind::Corpus, move || {
                            let Some(append) = cell.take() else { return };
                            match result {
                                Ok(()) => append.advance(),
                                Err(error) => append.finish(Err(error)),
                            }
                        })
                    }),
                );
                if submitted.is_err()
                    && let Some(append) = started.take()
                {
                    append.finish(Err(IoError::other("the host lake is closed")));
                }
            }
        }
    }

    /// The append ended: the corpus counts the sequence only when it is all there.
    fn finish(self: Box<Self>, result: io::Result<()>) {
        let Append {
            mut writer,
            id,
            tokens,
            rows,
            index_len,
            then,
            ..
        } = *self;
        if result.is_ok() {
            writer.ids.insert(id);
            (writer.token_count, writer.row_count, writer.index_len) = (tokens, rows, index_len);
        }
        then(writer, result);
    }
}

/// Opening a corpus: blocking file calls, made by the thread that opens it before a run starts, never from a frame of a
/// lattice. They are confined here; a run's appends use none.
mod startup {
    #![allow(
        clippy::disallowed_methods,
        reason = "blocking I/O on the calling thread before a run starts, not on a lattice: reading the index, cutting the files back, opening them"
    )]

    use std::collections::HashSet;
    use std::fs::{self, File, OpenOptions};
    use std::io;
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;
    #[cfg(windows)]
    use std::os::windows::fs::{FileExt, OpenOptionsExt};
    use std::path::Path;
    use std::sync::Arc;

    use serde_json::Value;

    use super::{Entry, HIDDEN, INDEX, LABELS, TOKEN_BYTES, TOKENS, VALUE_BYTES, invalid};
    use crate::durable::SharedFile;

    /// What opening a corpus found, and its files.
    pub(super) struct Opened {
        pub files: [SharedFile; 4],
        pub ids: HashSet<String>,
        pub token_count: u64,
        pub row_count: u64,
        pub index_len: u64,
    }

    /// The valid lines of an index, and how many bytes they take. A last line that is torn (cut short by a stopped
    /// run: no newline, or not a record) is not one of them; a line that is not a record before the last is an error.
    fn scan(text: &str) -> io::Result<(Vec<Entry>, u64)> {
        let mut entries = Vec::new();
        let mut valid = 0;
        let mut lines = text.split_inclusive('\n').peekable();
        let mut number = 0;
        while let Some(line) = lines.next() {
            number += 1;
            let last = lines.peek().is_none();
            let parsed = line
                .strip_suffix('\n')
                .ok_or_else(|| invalid(format!("{INDEX} line {number}: no end of line")))
                .and_then(|body| {
                    serde_json::from_str::<Value>(body)
                        .map_err(|error| invalid(format!("{INDEX} line {number}: {error}")))
                });
            match parsed {
                Ok(node) => {
                    let number_of = |key: &str| node.get(key).and_then(Value::as_u64).unwrap_or(0);
                    entries.push(Entry {
                        id: node.get("id").and_then(Value::as_str).unwrap_or_default().to_owned(),
                        token: number_of("token"),
                        row: number_of("row"),
                        length: number_of("length"),
                        prompt: number_of("prompt"),
                        sampled: node.get("sampled").and_then(Value::as_bool).unwrap_or(false),
                    });
                    valid += line.len() as u64;
                }
                Err(_) if last => break,
                Err(error) => return Err(error),
            }
        }
        Ok((entries, valid))
    }

    fn read_index(directory: &Path) -> io::Result<String> {
        match fs::read_to_string(directory.join(INDEX)) {
            Ok(text) => Ok(text),
            Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(String::new()),
            Err(error) => Err(error),
        }
    }

    /// The index of the corpus in `directory`: empty when it has none. A torn last line is not in it.
    pub fn entries(directory: &Path) -> io::Result<Vec<Entry>> {
        Ok(scan(&read_index(directory)?)?.0)
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

    /// The files of a corpus for the asynchronous writes: on Windows opened for overlapped I/O, as `AsyncIo` needs.
    fn open_shared(path: &Path) -> io::Result<SharedFile> {
        let mut options = OpenOptions::new();
        options.create(true).read(true).write(true);
        #[cfg(windows)]
        options.custom_flags(0x4000_0000);
        Ok(Arc::new(options.open(path)?))
    }

    /// Cuts the corpus in `directory` back to its index (and the index to its last whole line), makes the labels of
    /// a corpus without them its tokens, and opens its files.
    pub(super) fn open(directory: &Path, width: usize) -> io::Result<Opened> {
        fs::create_dir_all(directory)?;
        let text = read_index(directory)?;
        let (entries, index_len) = scan(&text)?;
        let (mut ids, mut token_count, mut row_count) = (HashSet::new(), 0, 0);
        for entry in entries {
            token_count = token_count.max(entry.token + entry.length);
            row_count = row_count.max((entry.row + entry.length).saturating_sub(1));
            ids.insert(entry.id);
        }
        {
            let mut options = OpenOptions::new();
            options.create(true).read(true).write(true);
            let tokens = options.open(directory.join(TOKENS))?;
            let labels = options.open(directory.join(LABELS))?;
            let hidden = options.open(directory.join(HIDDEN))?;
            let index = options.open(directory.join(INDEX))?;
            cut_back(&index, index_len)?;
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
        }
        let files = [TOKENS, LABELS, HIDDEN, INDEX].map(|name| open_shared(&directory.join(name)));
        let [tokens, labels, hidden, index] = files;
        Ok(Opened {
            files: [tokens?, labels?, hidden?, index?],
            ids,
            token_count,
            row_count,
            index_len,
        })
    }
}
