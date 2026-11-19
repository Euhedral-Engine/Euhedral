//! The assembled artifact as a file: the header, the metadata and the table, each payload copied from its source in
//! chunks, then the tokenizer section at the end of the content. The file is written as `<name>.part`, made durable
//! and renamed once complete, so an interrupted run never leaves a truncated artifact under the final name.
//!
//! The file is sparse where the alignment leaves gaps (they read as zeros). No payload is held whole: each chunk is
//! read into a buffer, written, and the buffer is read into again.

use std::ffi::OsString;
use std::fs::File;
use std::io::{Error as IoError, ErrorKind};
use std::mem::take;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use euhedral_graph::{Lake, UnitGuard};
use euhedral_inference_edrl::{
    ArtifactPlan, EdrlError, ObjectSpec, PAYLOAD_ALIGNMENT, TOKENIZER_ENTRIES, TokenizerTexts, encode_metadata,
    encode_tokenizer_section,
};
use euhedral_inference_engine::Handoff;
use euhedral_io::{AsyncIo, Create, IoBuf, IoDone, OpenSpec, OwnedFile};
use same_file::Handle;

use super::chunks::read_chunk;
use super::view::View;
use super::{AssembleError, Done};
use crate::checkpoint::CheckpointError;
use crate::checkpoint::io::{raw_file, submit_error};
use crate::durable;
use crate::step::{Kind, idle, step};

/// One payload to copy.
struct Job {
    name: String,
    file: Arc<OwnedFile>,
    from: u64,
    to: u64,
    bytes: u64,
}

/// Writes `view` as an artifact file at `out`, copying payloads in chunks of at most `copy_buffer_bytes`. `done` gets
/// the result once the file is complete under its final name, or has failed and left no `.part` file. The durability
/// and removal calls run as frames of `host`, which is held admitted until the end. If `host` closes before the data
/// is made durable, or before a failed file can be removed, the error says that the `.part` file was left. A source
/// that is the `.part` file itself (under any name) is refused and left as it is. `Err` started nothing.
pub fn write(
    io: &Arc<AsyncIo>,
    host: &Arc<Lake>,
    view: &View,
    out: &Path,
    copy_buffer_bytes: usize,
    done: Done<()>,
) -> Result<(), AssembleError> {
    if copy_buffer_bytes == 0 {
        return Err(AssembleError::invalid("the copy buffer must be at least one byte"));
    }
    let named = |source: EdrlError| AssembleError::Artifact {
        path: out.display().to_string(),
        source,
    };
    let metadata = encode_metadata(&view.metadata).map_err(named)?;
    let specs: Vec<ObjectSpec> = view
        .pieces
        .iter()
        .map(|piece| ObjectSpec {
            name: piece.entry.name().to_owned(),
            shape: piece.entry.shape().to_vec(),
            dtype: piece.entry.dtype(),
            format: piece.entry.format(),
            layout: piece.entry.layout(),
            bytes: piece.entry.bytes(),
        })
        .collect();
    let plan = ArtifactPlan::new(metadata.len() as u64, &specs, PAYLOAD_ALIGNMENT).map_err(named)?;
    let table = plan.table().map_err(named)?;
    let mut head = Vec::with_capacity(plan.header().encode().len() + metadata.len() + table.len());
    head.extend_from_slice(&plan.header().encode());
    head.extend_from_slice(&metadata);
    head.extend_from_slice(&table);
    let tokenizer = view
        .tokenizer
        .as_ref()
        .ok_or_else(|| AssembleError::invalid("the base carries no tokenizer, which an artifact is written with"))?;
    let texts: Vec<&str> = TOKENIZER_ENTRIES
        .iter()
        .map(|name| {
            tokenizer
                .text(name)
                .ok_or_else(|| AssembleError::invalid(format!("the tokenizer section has no {name}")))
        })
        .collect::<Result<_, _>>()?;
    let tail = encode_tokenizer_section(&TokenizerTexts {
        tokenizer: texts[0],
        tokenizer_config: texts[1],
        chat_template: texts[2],
        special_tokens: texts[3],
    })
    .map_err(named)?;
    let jobs: Vec<Job> = view
        .pieces
        .iter()
        .zip(plan.entries())
        .map(|(piece, placed)| Job {
            name: piece.entry.name().to_owned(),
            file: Arc::clone(&piece.file),
            from: piece.entry.offset(),
            to: placed.offset(),
            bytes: piece.entry.bytes(),
        })
        .collect();
    let largest = jobs.iter().map(|job| job.bytes).max().unwrap_or(0);
    let buffer = usize::try_from(largest).map_or(copy_buffer_bytes, |largest| largest.min(copy_buffer_bytes));
    let unit = host.admit().map_err(|_| CheckpointError::LakeClosed)?;
    let mut part: OsString = out.as_os_str().to_owned();
    part.push(".part");
    let part = PathBuf::from(part);
    // Opened without truncating: a source may be another name of this file, which the run must not cut. Read as well,
    // so that the identity of the file can be queried on every platform.
    let spec = OpenSpec {
        read: true,
        write: true,
        create: Create::IfMissing,
        ..OpenSpec::default()
    };
    let writing = Writing {
        io: Arc::clone(io),
        host: Arc::clone(host),
        unit,
        part: part.clone(),
        target: out.to_path_buf(),
        file: None,
        jobs,
        job: 0,
        at: 0,
        chunk: copy_buffer_bytes,
        buffer: Vec::with_capacity(buffer),
        tail: Some((plan.content_end(), tail)),
        done,
    };
    let context = part.display().to_string();
    io.open_path(
        &part,
        spec,
        Box::new(move |opened| {
            step(Kind::Resolve, move || match opened {
                Ok(file) => writing.start(file, head),
                Err(error) => writing.end(Err(
                    CheckpointError::io(format!("cannot create {context}"), error).into()
                )),
            })
        }),
    )
    .map_err(|error| submit_error(&out.display().to_string(), error).into())
}

/// The chain that writes the head, then each payload's chunks, then the tokenizer.
struct Writing {
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
    /// Keeps the host lake admitted until the chain has ended.
    unit: UnitGuard,
    part: PathBuf,
    target: PathBuf,
    /// The `.part` file while it is open.
    file: Option<OwnedFile>,
    jobs: Vec<Job>,
    job: usize,
    at: u64,
    chunk: usize,
    buffer: Vec<u8>,
    /// The tokenizer section and where it goes.
    tail: Option<(u64, Vec<u8>)>,
    done: Done<()>,
}

impl Writing {
    /// The `.part` file is open: it is refused if a source is the same file, else cut to empty and filled.
    fn start(mut self, file: OwnedFile, head: Vec<u8>) {
        match self.is_a_source(&file) {
            Ok(false) => {}
            Ok(true) => {
                // The file is a source, not this run's: it is left as it is.
                let refusal = format!(
                    "{} is a source and the temporary name of the output: the run would overwrite its own source",
                    self.part.display()
                );
                return self.end(Err(AssembleError::invalid(refusal)));
            }
            Err(error) => {
                let context = format!("cannot compare {} with the sources", self.part.display());
                return self.fail_with(file, CheckpointError::io(context, error).into());
            }
        }
        let Ok(cut) = file.try_clone().map(durable::share) else {
            let error = IoError::other("the handle cannot be duplicated");
            return self.fail_with(file, CheckpointError::io("cannot empty the .part file", error).into());
        };
        self.file = Some(file);
        let cell = Handoff::new();
        let host = Arc::clone(&self.host);
        cell.send(self);
        let waiting = Arc::clone(&cell);
        let started = durable::truncate_and_close(
            &host,
            cut,
            0,
            Box::new(move |emptied| {
                if let Some(writing) = waiting.take() {
                    match emptied {
                        Ok(()) => writing.begin(head),
                        Err(error) => {
                            let context = format!("cannot empty {}", writing.part.display());
                            writing.fail(CheckpointError::io(context, error).into());
                        }
                    }
                }
                idle()
            }),
        );
        if started.is_err()
            && let Some(writing) = cell.take()
        {
            let context = format!("cannot empty {}: the host lake is closed", writing.part.display());
            writing.end(Err(CheckpointError::io(context, IoError::from(ErrorKind::Other)).into()));
        }
    }

    /// Whether `file` is one of the sources.
    fn is_a_source(&self, file: &OwnedFile) -> Result<bool, IoError> {
        let target = Handle::from_file(File::from(file.try_clone()?))?;
        for job in &self.jobs {
            if Handle::from_file(File::from(job.file.try_clone()?))? == target {
                return Ok(true);
            }
        }
        Ok(false)
    }

    /// Ends the chain with `error` for a file that is not stored yet.
    fn fail_with(mut self, file: OwnedFile, error: AssembleError) {
        self.file = Some(file);
        self.fail(error);
    }

    /// Writes the head, then the data.
    fn begin(self, head: Vec<u8>) {
        let filled = head.len();
        self.put(head, filled, 0, false);
    }

    /// Hands the result over and lets the lake go.
    fn end(self, result: Result<(), AssembleError>) {
        let Writing { done, unit, .. } = self;
        done(result);
        drop(unit);
    }

    /// Ends the chain with `error`, once the `.part` file is removed (or cannot be: the lake is closed).
    fn fail(mut self, error: AssembleError) {
        drop(self.file.take());
        let Writing {
            host, unit, part, done, ..
        } = self;
        let pending = Handoff::new();
        pending.send(error);
        let (reported, held) = (Arc::clone(&pending), Arc::clone(&done));
        let removed = durable::remove(
            &host,
            &part,
            Box::new(move |removal| {
                if let Some(error) = reported.take() {
                    held(Err(match removal {
                        Ok(()) => error,
                        Err(source) => {
                            CheckpointError::io(format!("{error}; the .part file was not removed"), source).into()
                        }
                    }));
                }
                step(Kind::Release, move || drop(unit))
            }),
        );
        if removed.is_err()
            && let Some(error) = pending.take()
        {
            done(Err(CheckpointError::io(
                format!("{error}; the .part file was not removed"),
                IoError::other("the host lake is closed"),
            )
            .into()));
        }
    }

    /// Submits one write of `bytes[..filled]` at `offset`; the buffer comes back with it when `chunk` is set.
    fn put(self, bytes: Vec<u8>, filled: usize, offset: u64, chunk: bool) {
        let Some(raw) = self.file.as_ref().map(raw_file) else {
            return self.fail(AssembleError::invalid("the file is not open"));
        };
        let io = Arc::clone(&self.io);
        let cell = Handoff::new();
        cell.send(self);
        let started = Arc::clone(&cell);
        let submitted = io.write(
            raw,
            offset,
            IoBuf { bytes, filled },
            Box::new(move |written: IoDone| {
                step(Kind::Resolve, move || {
                    let Some(mut writing) = cell.take() else {
                        return;
                    };
                    if chunk {
                        writing.buffer = written.buf.bytes;
                    }
                    match written.result {
                        Ok(moved) if moved == filled => writing.next(),
                        Ok(moved) => writing
                            .fail(CheckpointError::Format(format!("a write moved {moved} of {filled} bytes")).into()),
                        Err(error) => writing.fail(CheckpointError::io("cannot write the artifact file", error).into()),
                    }
                })
            }),
        );
        if let Err(error) = submitted
            && let Some(writing) = started.take()
        {
            writing.fail(submit_error("the artifact file", error).into());
        }
    }

    /// Copies the next chunk, or writes the tokenizer, or finishes the file.
    fn next(mut self) {
        while self.job < self.jobs.len() && self.at >= self.jobs[self.job].bytes {
            self.job += 1;
            self.at = 0;
        }
        if self.job == self.jobs.len() {
            return match self.tail.take() {
                Some((offset, bytes)) => {
                    let filled = bytes.len();
                    self.put(bytes, filled, offset, false);
                }
                None => self.finish(),
            };
        }
        let job = &self.jobs[self.job];
        let n = usize::try_from(job.bytes - self.at).map_or(self.chunk, |left| left.min(self.chunk));
        let (file, from, to) = (Arc::clone(&job.file), job.from + self.at, job.to + self.at);
        let context = format!("{} (copying)", job.name);
        let (io, buffer) = (Arc::clone(&self.io), take(&mut self.buffer));
        let cell = Handoff::new();
        cell.send(self);
        let waiting = Arc::clone(&cell);
        let started = read_chunk(
            &io,
            file,
            from,
            n,
            buffer,
            context,
            Box::new(move |read| {
                let Some(mut writing) = waiting.take() else {
                    return;
                };
                match read {
                    Ok(bytes) => {
                        writing.at += n as u64;
                        writing.put(bytes, n, to, true);
                    }
                    Err(error) => writing.fail(error),
                }
            }),
        );
        if let Err(error) = started
            && let Some(writing) = cell.take()
        {
            writing.fail(error);
        }
    }

    /// Makes the data durable, closes the file and renames it to its final name.
    fn finish(mut self) {
        let Some(file) = self.file.take() else {
            return self.fail(AssembleError::invalid("the file is not open"));
        };
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
                        writing.fail(CheckpointError::io(context, error).into());
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
            )
            .into()));
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
                            writing.fail(CheckpointError::io(context, error).into());
                        }
                    }
                })
            }),
        );
        if let Err(error) = renamed
            && let Some(writing) = cell.take()
        {
            writing.fail(submit_error("the artifact file", error).into());
        }
    }
}
