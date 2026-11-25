//! One safetensors file. Tensors are registered with [`SafetensorsWriter::add`] and written, in insertion order, by
//! [`SafetensorsWriter::finish`]. The header (`{"__metadata__":{"format":"pt"}, name: {dtype, shape, data_offsets}}`,
//! padded with spaces to a multiple of 8 bytes) needs every tensor's size, so the data is pulled from each tensor's
//! source in chunks only when the file is written; a tensor never has to be resident whole. The file is written as
//! `<name>.part` and renamed once complete, so an interrupted run never leaves a truncated file under the final name.

use std::collections::HashSet;
use std::ffi::OsString;
use std::fmt::{Debug, Formatter, Result as FmtResult, Write as _};
use std::io::Error as IoError;
use std::mem::take;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use euhedral_graph::{Lake, UnitGuard};
use euhedral_inference_engine::Handoff;
use euhedral_io::{AsyncIo, Create, IoBuf, IoDone, OpenSpec, OwnedFile};

use super::io::{Next, file_length, open_read, raw_file, read_exact, submit_error};
use super::{CheckpointError, Done};
use crate::durable;
use crate::step::{Kind, idle, step};

/// The size of the buffer the tensors are copied to the file through: 16 MiB.
pub const DEFAULT_BUFFER_BYTES: usize = 16 << 20;

/// The element sizes of the data types a writer takes.
const ELEMENT_BYTES: [(&str, u64); 8] = [
    ("I16", 2),
    ("F16", 2),
    ("BF16", 2),
    ("F32", 4),
    ("I32", 4),
    ("I64", 8),
    ("U8", 1),
    ("I8", 1),
];

/// Supplies a tensor's bytes, one chunk at a time: `source(offset, buffer, done)` must fill `buffer`, whose length is
/// the chunk's, with the bytes `[offset, offset + buffer.len())` of the tensor and give it back through `done`, once,
/// from a frame (not from the reap of an `AsyncIo`). It may do so before it returns or later, after reads of its own.
/// It must not wait. `Err` started nothing and never calls `done`. The next chunk is asked for only after the
/// previous one has been written, so a tensor is never resident whole. Chunks start at multiples of the writer's
/// buffer size, which is a multiple of 8.
pub type Source = Arc<dyn Fn(u64, Vec<u8>, Done<Vec<u8>>) -> Result<(), CheckpointError> + Send + Sync>;

struct Entry {
    name: String,
    /// What error messages call the data: `tensor <name>`, or the file.
    label: String,
    dtype: String,
    shape: Vec<u64>,
    bytes: u64,
    source: Source,
}

/// The tensors of one file, and where it goes.
pub struct SafetensorsWriter {
    file: PathBuf,
    buffer_bytes: usize,
    entries: Vec<Entry>,
    names: HashSet<String>,
    data_bytes: u64,
}

impl Debug for SafetensorsWriter {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("SafetensorsWriter")
            .field("file", &self.file)
            .field("tensors", &self.entries.len())
            .field("data_bytes", &self.data_bytes)
            .finish_non_exhaustive()
    }
}

impl SafetensorsWriter {
    /// A writer of `file`, copying through a buffer of [`DEFAULT_BUFFER_BYTES`].
    pub fn new(file: &Path) -> Self {
        Self {
            file: file.to_path_buf(),
            buffer_bytes: DEFAULT_BUFFER_BYTES,
            entries: Vec::new(),
            names: HashSet::new(),
            data_bytes: 0,
        }
    }

    /// A writer of `file` that copies the tensors through a buffer of `buffer_bytes` bytes. Each tensor is asked for in
    /// chunks that start at multiples of `buffer_bytes` (the last one may be shorter), and the size must be a multiple
    /// of 8 so that no chunk splits an element.
    pub fn with_buffer(file: &Path, buffer_bytes: usize) -> Result<Self, CheckpointError> {
        if buffer_bytes == 0 || !buffer_bytes.is_multiple_of(8) {
            return Err(CheckpointError::Invalid(format!(
                "the buffer must be a positive multiple of 8 bytes, so a chunk never splits an element: {buffer_bytes}"
            )));
        }
        Ok(Self {
            buffer_bytes,
            ..Self::new(file)
        })
    }

    /// Adds a tensor held in `data` (little-endian, row-major).
    pub fn add(&mut self, name: &str, dtype: &str, shape: &[u64], data: Vec<u8>) -> Result<(), CheckpointError> {
        let bytes = data.len() as u64;
        self.add_source(name, dtype, shape, bytes, bytes_source(data))
    }

    /// Adds a tensor of `bytes` bytes that `source` produces on demand.
    pub fn add_source(
        &mut self,
        name: &str,
        dtype: &str,
        shape: &[u64],
        bytes: u64,
        source: Source,
    ) -> Result<(), CheckpointError> {
        if name == "__metadata__" {
            return Err(CheckpointError::Invalid(
                "__metadata__ is the header's own entry, not a tensor name".to_owned(),
            ));
        }
        let Some((_, width)) = ELEMENT_BYTES.iter().find(|(known, _)| *known == dtype) else {
            return Err(CheckpointError::Invalid(format!("{name}: unsupported dtype {dtype}")));
        };
        let implied = shape
            .iter()
            .try_fold(*width, |size, dimension| size.checked_mul(*dimension))
            .ok_or_else(|| CheckpointError::Invalid(format!("{name}: the shape is too large")))?;
        if implied != bytes {
            return Err(CheckpointError::Invalid(format!(
                "{name}: shape implies {implied} bytes, got {bytes}"
            )));
        }
        if !self.names.insert(name.to_owned()) {
            return Err(CheckpointError::Invalid(format!("duplicate tensor {name}")));
        }
        self.entries.push(Entry {
            name: name.to_owned(),
            label: format!("tensor {name}"),
            dtype: dtype.to_owned(),
            shape: shape.to_vec(),
            bytes,
            source,
        });
        self.data_bytes += bytes;
        Ok(())
    }

    /// The total size of the tensor data added so far.
    pub fn data_bytes(&self) -> u64 {
        self.data_bytes
    }

    /// The JSON header, padded with spaces to a multiple of 8 bytes (the 8-byte length that precedes it in the file is
    /// not part of it).
    pub fn header(&self) -> Vec<u8> {
        let mut json = String::from("{\"__metadata__\":{\"format\":\"pt\"}");
        let mut offset = 0u64;
        for entry in &self.entries {
            json.push(',');
            quote(&mut json, &entry.name);
            let _ = write!(json, ":{{\"dtype\":\"{}\",\"shape\":[", entry.dtype);
            for (index, dimension) in entry.shape.iter().enumerate() {
                if index > 0 {
                    json.push(',');
                }
                let _ = write!(json, "{dimension}");
            }
            let _ = write!(json, "],\"data_offsets\":[{},{}]}}", offset, offset + entry.bytes);
            offset += entry.bytes;
        }
        json.push('}');
        let mut bytes = json.into_bytes();
        bytes.resize(bytes.len().div_ceil(8) * 8, b' ');
        bytes
    }

    /// Writes the file: `then` gets the result once it is complete under its final name, or has failed and left no
    /// `.part` file. The durability and removal calls run as frames of `host`, which is held admitted until the end.
    /// If `host` closes before the data is made durable, or before a failed file can be removed, the error says that
    /// the `.part` file was left. `Err` started nothing.
    pub fn finish(self, io: &Arc<AsyncIo>, host: &Arc<Lake>, then: Done<()>) -> Result<(), CheckpointError> {
        let json = self.header();
        let mut head = Vec::with_capacity(8 + json.len());
        head.extend_from_slice(&(json.len() as u64).to_le_bytes());
        head.extend_from_slice(&json);
        self.finish_with(head, io, host, then)
    }

    /// The write chain: `head` first, then the tensors' data.
    fn finish_with(
        self,
        head: Vec<u8>,
        io: &Arc<AsyncIo>,
        host: &Arc<Lake>,
        then: Done<()>,
    ) -> Result<(), CheckpointError> {
        let unit = host.admit().map_err(|_| CheckpointError::LakeClosed)?;
        let mut part: OsString = self.file.clone().into_os_string();
        part.push(".part");
        let part = PathBuf::from(part);
        let name = self.file_name();
        let largest = self.entries.iter().map(|entry| entry.bytes).max().unwrap_or(0);
        let buffer = usize::try_from(largest).map_or(self.buffer_bytes, |largest| largest.min(self.buffer_bytes));
        let spec = OpenSpec {
            read: false,
            write: true,
            create: Create::Truncate,
            ..OpenSpec::default()
        };
        let writing = Writing {
            io: Arc::clone(io),
            host: Arc::clone(host),
            unit,
            part: part.clone(),
            target: self.file,
            entries: self.entries,
            buffer_bytes: self.buffer_bytes,
            entry: 0,
            at: 0,
            offset: 0,
            buffer: vec![0; buffer],
            then,
        };
        let context = part.display().to_string();
        io.open_path(
            &part,
            spec,
            Box::new(move |opened| {
                step(Kind::Resolve, move || match opened {
                    Ok(file) => writing.begin(file, head),
                    Err(error) => writing.end(Err(CheckpointError::io(format!("cannot create {context}"), error))),
                })
            }),
        )
        .map_err(|error| submit_error(&name, error))
    }

    fn file_name(&self) -> String {
        self.file.display().to_string()
    }
}

/// A JSON string: quotes, backslashes and control characters escaped, everything else as it is.
fn quote(out: &mut String, text: &str) {
    out.push('"');
    for c in text.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\u{8}' => out.push_str("\\b"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\u{c}' => out.push_str("\\f"),
            '\r' => out.push_str("\\r"),
            c if c < ' ' => {
                let _ = write!(out, "\\u{:04X}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

/// The chain that writes the header and then the tensors' chunks, one write at a time.
struct Writing {
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
    /// Keeps the host lake admitted until the chain has ended.
    unit: UnitGuard,
    part: PathBuf,
    target: PathBuf,
    entries: Vec<Entry>,
    buffer_bytes: usize,
    /// The tensor being written, and how far into it.
    entry: usize,
    at: u64,
    /// Where the next write goes in the file.
    offset: u64,
    buffer: Vec<u8>,
    then: Done<()>,
}

/// An open `.part` file and the chain's state.
struct Step {
    file: OwnedFile,
    writing: Writing,
}

impl Writing {
    /// Writes the header, then the data.
    fn begin(self, file: OwnedFile, head: Vec<u8>) {
        let filled = head.len();
        if filled == 0 {
            // An empty file has nothing to write first.
            Step { file, writing: self }.next();
            return;
        }
        Step { file, writing: self }.submit(head, filled, false);
    }

    /// Hands the result over and lets the lake go.
    fn end(self, result: Result<(), CheckpointError>) {
        let Writing { then, unit, .. } = self;
        then(result);
        drop(unit);
    }

    /// Ends the chain with `error`, once the `.part` file is removed (or cannot be: the lake is closed).
    fn fail(self, error: CheckpointError) {
        let Writing {
            host, unit, part, then, ..
        } = self;
        let pending = Handoff::new();
        pending.send(error);
        let (reported, held) = (Arc::clone(&pending), Arc::clone(&then));
        let removed = durable::remove(
            &host,
            &part,
            Box::new(move |removal| {
                if let Some(error) = reported.take() {
                    held(Err(match removal {
                        Ok(()) => error,
                        Err(source) => CheckpointError::io(format!("{error}; the .part file was not removed"), source),
                    }));
                }
                step(Kind::Release, move || drop(unit))
            }),
        );
        if removed.is_err()
            && let Some(error) = pending.take()
        {
            // The lake is closed: the continuation was dropped with the refusal, and nothing can remove the file.
            then(Err(CheckpointError::io(
                format!("{error}; the .part file was not removed"),
                IoError::other("the host lake is closed"),
            )));
        }
    }
}

impl Step {
    /// Submits one write of `bytes[..filled]`; the chunk buffer comes back with it when `chunk` is set.
    fn submit(self, bytes: Vec<u8>, filled: usize, chunk: bool) {
        let Step { file, writing } = self;
        let raw = raw_file(&file);
        let offset = writing.offset;
        let io = Arc::clone(&writing.io);
        let cell = Handoff::new();
        cell.send(Step { file, writing });
        let started = Arc::clone(&cell);
        let submitted = io.write(
            raw,
            offset,
            IoBuf { bytes, filled },
            Box::new(move |done: IoDone| {
                step(Kind::Resolve, move || {
                    let Some(Step { file, mut writing }) = cell.take() else {
                        return;
                    };
                    writing.offset += filled as u64;
                    if chunk {
                        writing.buffer = done.buf.bytes;
                    }
                    match done.result {
                        Ok(moved) if moved == filled => Step { file, writing }.next(),
                        Ok(moved) => {
                            drop(file);
                            writing.fail(CheckpointError::Format(format!(
                                "a write moved {moved} of {filled} bytes"
                            )));
                        }
                        Err(error) => {
                            drop(file);
                            let context = format!("cannot write {}", writing.part.display());
                            writing.fail(CheckpointError::io(context, error));
                        }
                    }
                })
            }),
        );
        if let Err(error) = submitted
            && let Some(Step { file, writing }) = started.take()
        {
            drop(file);
            let context = writing.part.display().to_string();
            writing.fail(submit_error(&context, error));
        }
    }

    /// Asks the tensor's source for the next chunk and writes it, or finishes the file when no data is left.
    fn next(self) {
        let Step { file, mut writing } = self;
        while writing.entry < writing.entries.len() && writing.at >= writing.entries[writing.entry].bytes {
            writing.entry += 1;
            writing.at = 0;
        }
        if writing.entry == writing.entries.len() {
            writing.finish(file);
            return;
        }
        let entry = &writing.entries[writing.entry];
        let n = (writing.buffer_bytes as u64).min(entry.bytes - writing.at) as usize;
        let source = Arc::clone(&entry.source);
        let name = entry.label.clone();
        let named = name.clone();
        let at = writing.at;
        let mut buffer = take(&mut writing.buffer);
        buffer.resize(n, 0);
        let cell = Handoff::new();
        cell.send(Step { file, writing });
        let waiting = Arc::clone(&cell);
        let filled: Done<Vec<u8>> = Arc::new(move |result| {
            let Some(Step { file, mut writing }) = waiting.take() else {
                return;
            };
            match result {
                Ok(bytes) if bytes.len() == n => {
                    writing.at += n as u64;
                    Step { file, writing }.submit(bytes, n, true);
                }
                Ok(bytes) => {
                    drop(file);
                    writing.fail(CheckpointError::Invalid(format!(
                        "the source of {name} gave {} bytes for a chunk of {n}",
                        bytes.len()
                    )));
                }
                Err(error) => {
                    drop(file);
                    writing.fail(CheckpointError::Invalid(format!("cannot read {name}: {error}")));
                }
            }
        });
        if let Err(error) = source(at, buffer, filled)
            && let Some(Step { file, writing }) = cell.take()
        {
            drop(file);
            writing.fail(CheckpointError::Invalid(format!("cannot read {named}: {error}")));
        }
    }
}

impl Writing {
    /// Makes the data durable, closes the file and renames it to its final name.
    fn finish(self, file: OwnedFile) {
        let shared = durable::share(file);
        let host = Arc::clone(&self.host);
        let cell = Handoff::new();
        cell.send(self);
        let waiting = Arc::clone(&cell);
        // The call gives up its reference before the completion runs, so the file is closed before it is renamed (a
        // file that is open cannot be renamed on Windows).
        let started = durable::sync_and_close(
            &host,
            shared,
            Box::new(move |synced| {
                let Some(writing) = waiting.take() else {
                    return idle();
                };
                match synced {
                    Ok(()) => writing.rename(),
                    Err(error) => {
                        let context = format!("cannot sync {}", writing.part.display());
                        writing.fail(CheckpointError::io(context, error));
                    }
                }
                idle()
            }),
        );
        if started.is_err()
            && let Some(writing) = cell.take()
        {
            let context = format!(
                "cannot sync {}: the .part file is complete and was left",
                writing.part.display()
            );
            writing.end(Err(CheckpointError::io(
                context,
                IoError::other("the host lake is closed"),
            )));
        }
    }

    /// Renames the finished `.part` file to its final name.
    fn rename(self) {
        let io = Arc::clone(&self.io);
        let (part, target) = (self.part.clone(), self.target.clone());
        let cell = Handoff::new();
        cell.send(self);
        let waiting = Arc::clone(&cell);
        let renamed = io.rename(
            &part,
            &target,
            Box::new(move |result| {
                step(Kind::Resolve, move || {
                    let Some(writing) = waiting.take() else {
                        return;
                    };
                    match result {
                        Ok(()) => writing.end(Ok(())),
                        Err(error) => {
                            let context = format!("cannot rename {}", writing.part.display());
                            writing.fail(CheckpointError::io(context, error));
                        }
                    }
                })
            }),
        );
        if let Err(error) = renamed
            && let Some(writing) = cell.take()
        {
            let context = writing.part.display().to_string();
            writing.fail(submit_error(&context, error));
        }
    }
}

/// A [`Source`] over bytes held in memory.
pub fn bytes_source(data: Vec<u8>) -> Source {
    Arc::new(move |offset, mut chunk, done| {
        let range = usize::try_from(offset)
            .ok()
            .and_then(|from| Some(from..from.checked_add(chunk.len())?))
            .and_then(|range| data.get(range))
            .ok_or_else(|| CheckpointError::Invalid("a chunk beyond the data".to_owned()))?;
        chunk.copy_from_slice(range);
        done(Ok(chunk));
        Ok(())
    })
}

/// Writes the `size` bytes that `source` produces as the whole of `file`: as `<file>.part`, made durable and renamed
/// once complete, with nothing left behind on failure, and with no more than `buffer_bytes` (a positive multiple of 8)
/// in memory at a time. `then` and the host lake are as for [`SafetensorsWriter::finish`]. `Err` started nothing.
pub fn write_file(
    file: &Path,
    size: u64,
    source: Source,
    buffer_bytes: usize,
    io: &Arc<AsyncIo>,
    host: &Arc<Lake>,
    then: Done<()>,
) -> Result<(), CheckpointError> {
    let target = Target {
        file,
        label: file.display().to_string(),
        size,
    };
    write_target(target, source, buffer_bytes, io, host, then)
}

/// The file a write makes, the name its read errors give for where the bytes come from, and its size.
struct Target<'a> {
    file: &'a Path,
    label: String,
    size: u64,
}

fn write_target(
    target: Target<'_>,
    source: Source,
    buffer_bytes: usize,
    io: &Arc<AsyncIo>,
    host: &Arc<Lake>,
    then: Done<()>,
) -> Result<(), CheckpointError> {
    let mut writer = SafetensorsWriter::with_buffer(target.file, buffer_bytes)?;
    writer.entries.push(Entry {
        name: target.file.display().to_string(),
        label: target.label,
        dtype: String::new(),
        shape: Vec::new(),
        bytes: target.size,
        source,
    });
    writer.finish_with(Vec::new(), io, host, then)
}

/// Copies `from` to `to` the way [`write_file`] writes a file, a buffer at a time, so a file of any size can be
/// copied. `Err` started nothing.
pub(crate) fn copy_file(
    from: &Path,
    to: &Path,
    buffer_bytes: usize,
    io: &Arc<AsyncIo>,
    host: &Arc<Lake>,
    then: Done<()>,
) -> Result<(), CheckpointError> {
    let (io_next, host, to, label) = (
        Arc::clone(io),
        Arc::clone(host),
        to.to_path_buf(),
        from.display().to_string(),
    );
    let handed = Handoff::new();
    handed.send(then);
    let waiting = Arc::clone(&handed);
    let opened = open_read(
        io,
        from,
        Box::new(move |opened| {
            let Some(then) = waiting.take() else { return };
            let file = match opened {
                Ok(file) => Arc::new(file),
                Err(error) => return then(Err(error)),
            };
            let sizing = Arc::clone(&file);
            let cell = Handoff::new();
            cell.send((then, io_next.clone()));
            let ready = Arc::clone(&cell);
            let sized = file_length(
                &io_next,
                sizing,
                label.clone(),
                Box::new(move |length| {
                    let Some((then, io)) = ready.take() else { return };
                    let size = match length {
                        Ok(size) => size,
                        Err(error) => return then(Err(error)),
                    };
                    let reader = Arc::clone(&io);
                    let context = label.clone();
                    let source: Source = Arc::new(move |offset, chunk, done| {
                        let next: Next<Vec<u8>> = Box::new(move |bytes| done(bytes));
                        read_exact(&reader, Arc::clone(&file), offset, chunk.len(), context.clone(), next)
                    });
                    let report = Arc::clone(&then);
                    let target = Target {
                        file: &to,
                        label: label.clone(),
                        size,
                    };
                    if let Err(error) = write_target(target, source, buffer_bytes, &io, &host, then) {
                        report(Err(error));
                    }
                }),
            );
            if let Err(error) = sized
                && let Some((then, _)) = cell.take()
            {
                then(Err(error));
            }
        }),
    );
    if let Err(error) = opened {
        // Nothing started: the continuation was dropped with the refusal.
        drop(handed.take());
        return Err(error);
    }
    Ok(())
}
