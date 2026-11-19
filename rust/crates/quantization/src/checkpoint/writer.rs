//! One safetensors file. Tensors are registered with [`SafetensorsWriter::add`] and written, in insertion order, by
//! [`SafetensorsWriter::finish`]. The header (`{"__metadata__":{"format":"pt"}, name: {dtype, shape, data_offsets}}`,
//! padded with spaces to a multiple of 8 bytes) needs every tensor's size, so the data is pulled from each tensor's
//! source in chunks only when the file is written; a tensor never has to be resident whole. The file is written as
//! `<name>.part` and renamed once complete, so an interrupted run never leaves a truncated file under the final name.

use std::collections::HashSet;
use std::ffi::OsString;
use std::fmt::Write as _;
use std::fs::{self, File};
use std::io::Error as IoError;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use euhedral_io::{AsyncIo, Create, IoBuf, IoDone, OpenSpec, OwnedFile};

use super::io::{raw_file, submit_error, work};
use super::{CheckpointError, Done, report};

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

/// Supplies a tensor's bytes: fills `destination` with the bytes `[offset, offset + destination.len())`. It runs in a
/// frame, so it must not wait.
pub type Source = Box<dyn FnMut(u64, &mut [u8]) -> Result<(), IoError> + Send>;

struct Entry {
    name: String,
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

impl std::fmt::Debug for SafetensorsWriter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
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

    /// A writer of `file` that copies the tensors through a buffer of `buffer_bytes` bytes.
    pub fn with_buffer(file: &Path, buffer_bytes: usize) -> Result<Self, CheckpointError> {
        if buffer_bytes < 1 {
            return Err(CheckpointError::Invalid(format!("buffer must hold a byte: {buffer_bytes}")));
        }
        Ok(Self {
            buffer_bytes,
            ..Self::new(file)
        })
    }

    /// Adds a tensor held in `data` (little-endian, row-major).
    pub fn add(&mut self, name: &str, dtype: &str, shape: &[u64], data: Vec<u8>) -> Result<(), CheckpointError> {
        let bytes = data.len() as u64;
        self.add_source(
            name,
            dtype,
            shape,
            bytes,
            Box::new(move |offset, destination| {
                let from = usize::try_from(offset).map_err(|_| IoError::other("offset beyond memory"))?;
                destination.copy_from_slice(&data[from..from + destination.len()]);
                Ok(())
            }),
        )
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
    /// `.part` file. `Err` started nothing.
    pub fn finish(self, io: &Arc<AsyncIo>, then: Done<()>) -> Result<(), CheckpointError> {
        let mut part: OsString = self.file.clone().into_os_string();
        part.push(".part");
        let part = PathBuf::from(part);
        let name = self.file_name();
        let largest = self.entries.iter().map(|entry| entry.bytes).max().unwrap_or(0);
        let buffer = usize::try_from(largest).map_or(self.buffer_bytes, |largest| largest.min(self.buffer_bytes));
        let json = self.header();
        let mut head = Vec::with_capacity(8 + json.len());
        head.extend_from_slice(&(json.len() as u64).to_le_bytes());
        head.extend_from_slice(&json);
        let spec = OpenSpec {
            read: false,
            write: true,
            create: Create::Truncate,
            ..OpenSpec::default()
        };
        let writing = Writing {
            io: Arc::clone(io),
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
                work(move || match opened {
                    Ok(file) => writing.begin(file, head),
                    Err(error) => (writing.then)(Err(CheckpointError::io(format!("cannot create {context}"), error))),
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

/// Removes a `.part` file. `AsyncIo` has no unlink, so this is one short call on the frame that found the failure.
fn discard(part: &Path) {
    let _ = fs::remove_file(part);
}

impl Writing {
    /// Writes the header, then the data.
    fn begin(self, file: OwnedFile, head: Vec<u8>) {
        let filled = head.len();
        Step { file, writing: self }.submit(head, filled, false);
    }

    /// Ends the chain with `error`, leaving no `.part` file.
    fn fail(self, file: OwnedFile, error: CheckpointError) {
        drop(file);
        discard(&self.part);
        (self.then)(Err(error));
    }
}

impl Step {
    /// Submits one write of `bytes[..filled]`; the chunk buffer comes back with it when `chunk` is set.
    fn submit(self, bytes: Vec<u8>, filled: usize, chunk: bool) {
        let Step { file, writing } = self;
        let raw = raw_file(&file);
        let offset = writing.offset;
        let io = Arc::clone(&writing.io);
        let then = Arc::clone(&writing.then);
        let part = writing.part.clone();
        let submitted = io.write(
            raw,
            offset,
            IoBuf { bytes, filled },
            Box::new(move |done: IoDone| {
                work(move || {
                    let mut writing = writing;
                    writing.offset += filled as u64;
                    if chunk {
                        writing.buffer = done.buf.bytes;
                    }
                    match done.result {
                        Ok(moved) if moved == filled => Step { file, writing }.next(),
                        Ok(moved) => writing.fail(
                            file,
                            CheckpointError::Format(format!("a write moved {moved} of {filled} bytes")),
                        ),
                        Err(error) => writing.fail(file, CheckpointError::io("cannot write the safetensors file", error)),
                    }
                })
            }),
        );
        if let Err(error) = submitted {
            discard(&part);
            report(&then, submit_error("the safetensors file", error));
        }
    }

    /// Fills the buffer with the next chunk and writes it, or finishes the file when no data is left.
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
        let entry = &mut writing.entries[writing.entry];
        let n = (writing.buffer_bytes as u64).min(entry.bytes - writing.at) as usize;
        let mut buffer = std::mem::take(&mut writing.buffer);
        if buffer.len() < n {
            buffer.resize(n, 0);
        }
        if let Err(error) = (entry.source)(writing.at, &mut buffer[..n]) {
            let context = format!("cannot read tensor {}", entry.name);
            writing.fail(file, CheckpointError::io(context, error));
            return;
        }
        writing.at += n as u64;
        Step { file, writing }.submit(buffer, n, true);
    }
}

impl Writing {
    /// Makes the data durable, closes the file and renames it to its final name.
    fn finish(self, file: OwnedFile) {
        // `AsyncIo` has no fsync yet: the data is made durable by one call on a duplicate of the handle, on this frame.
        let synced = file.try_clone().map(File::from).and_then(|duplicate| duplicate.sync_data());
        drop(file);
        if let Err(error) = synced {
            let context = format!("cannot sync {}", self.part.display());
            let Writing { part, then, .. } = self;
            discard(&part);
            then(Err(CheckpointError::io(context, error)));
            return;
        }
        let Writing {
            io, part, target, then, ..
        } = self;
        let failed = Arc::clone(&then);
        let from = part.clone();
        let cleanup = part.clone();
        let renamed = io.rename(
            &part,
            &target,
            Box::new(move |result| {
                work(move || match result {
                    Ok(()) => then(Ok(())),
                    Err(error) => {
                        discard(&cleanup);
                        then(Err(CheckpointError::io(format!("cannot rename {}", cleanup.display()), error)));
                    }
                })
            }),
        );
        if let Err(error) = renamed {
            discard(&from);
            report(&failed, submit_error("the safetensors file", error));
        }
    }
}
