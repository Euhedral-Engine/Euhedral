//! A checkpoint directory that reads as a source checkpoint with some 1-D tensors replaced.
//!
//! Every file of the source is linked, the replacements (BF16) go into a new shard, and the index's weight map sends
//! the replaced names there. The readers that open the shards the index names ([`Safetensors`](super::safetensors))
//! open them in name order and take a tensor from the last shard that holds it, so the shard is named to sort after
//! every shard the source's index names: [`SHARD`], or when the source has a shard that sorts at or after that (a
//! name such as `weights-…`, or the shard of an earlier overlay), that last shard's name followed by `.` and
//! [`SHARD`]. The replacements then shadow the source's copies whatever the source's shard names are, and an overlay
//! of an overlay shadows the first one's.
//!
//! Nothing here blocks the caller: the index is read through `AsyncIo`, the links are made by a frame of the host lake,
//! and the two files are written as [`write_file`] and [`SafetensorsWriter`] write them, whole or not at all. The shard
//! is complete before the index that names it is replaced.

use std::collections::BTreeMap;
use std::fs;
use std::io::Result as IoResult;
use std::path::{self, Path, PathBuf};
use std::sync::Arc;

use euhedral_graph::Lake;
use euhedral_inference_engine::Handoff;
use euhedral_io::AsyncIo;
use serde_json::Value;

use super::io::{Closing, read_file};
use super::writer::{SafetensorsWriter, bytes_source, write_file};
use super::{CheckpointError, Done};
use crate::step::{Kind, step};

/// The file the replacements are written to, unless a shard of the source sorts at or after it.
pub const SHARD: &str = "trained.safetensors";
const INDEX: &str = "model.safetensors.index.json";
const BUFFER_BYTES: usize = 1 << 20;

/// What the overlay's steps share.
struct Overlaying {
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
    source: PathBuf,
    target: PathBuf,
    tensors: BTreeMap<String, Vec<f32>>,
    /// The name of the shard of the replacements; known once the source's index was read.
    shard: String,
    then: Closing<()>,
}

/// Makes `target` read as `source` with `tensors` (name to values, stored as BF16) replacing the source's tensors of
/// those names. A name the source's index does not list is an error, found before anything is written. `then` is called
/// once, from a frame, with the result; `Err` here started nothing.
pub fn write(
    io: &Arc<AsyncIo>,
    host: &Arc<Lake>,
    source: &Path,
    target: &Path,
    tensors: BTreeMap<String, Vec<f32>>,
    then: Done<()>,
) -> Result<(), CheckpointError> {
    // The host lake takes frames only while a unit is admitted: this one is held until the result is delivered.
    let unit = Handoff::new();
    unit.send(host.admit().map_err(|_| CheckpointError::LakeClosed)?);
    let then: Done<()> = Arc::new(move |result| {
        drop(unit.take());
        then(result);
    });
    let overlaying = Overlaying {
        io: Arc::clone(io),
        host: Arc::clone(host),
        source: source.to_path_buf(),
        target: target.to_path_buf(),
        tensors,
        shard: SHARD.to_owned(),
        then: Closing::new(then),
    };
    let cell = Handoff::new();
    cell.send(overlaying);
    let held = Arc::clone(&cell);
    read_file(
        io,
        &source.join(INDEX),
        Arc::new(move |index| {
            let Some(overlaying) = cell.take() else { return };
            let mut overlaying = overlaying;
            match index.and_then(|bytes| overlaying.reindexed(&bytes)) {
                Ok(index) => overlaying.link(index),
                Err(error) => overlaying.then.end(Err(error)),
            }
        }),
    )
    .inspect_err(|_| {
        // Nothing started: the refusal is the caller's, and the continuation is not called.
        if let Some(overlaying) = held.take() {
            drop(overlaying.then.release());
        }
    })
}

impl Overlaying {
    /// The source's index with the replaced names sent to the shard, whose name is chosen to sort after the source's.
    fn reindexed(&mut self, bytes: &[u8]) -> Result<Vec<u8>, CheckpointError> {
        let mut index: Value = serde_json::from_slice(bytes)
            .map_err(|error| CheckpointError::Format(format!("{INDEX} is not JSON: {error}")))?;
        let map = index
            .get_mut("weight_map")
            .and_then(Value::as_object_mut)
            .ok_or_else(|| CheckpointError::Format(format!("{INDEX} has no weight_map")))?;
        let mut shard = SHARD.to_owned();
        for source in map.values().filter_map(Value::as_str) {
            if source >= shard.as_str() {
                shard = format!("{source}.{SHARD}");
            }
        }
        for name in self.tensors.keys() {
            let Some(entry) = map.get_mut(name) else {
                return Err(CheckpointError::Invalid(format!(
                    "{name} is not a tensor of {}",
                    self.source.display()
                )));
            };
            *entry = Value::String(shard.clone());
        }
        self.shard = shard;
        serde_json::to_vec_pretty(&index).map_err(|error| CheckpointError::Format(format!("the index: {error}")))
    }

    /// Links the source's files into the target, on the host lake, then writes the shard.
    fn link(self, index: Vec<u8>) {
        let host = Arc::clone(&self.host);
        let cell = Handoff::new();
        cell.send((self, index));
        let held = Arc::clone(&cell);
        let frame = step(Kind::Durable, move || {
            let Some((overlaying, index)) = held.take() else { return };
            match link_files(&overlaying.source, &overlaying.target, &overlaying.shard) {
                Ok(()) => overlaying.shard(index),
                Err(error) => overlaying.then.end(Err(error)),
            }
        });
        // A frame that is accepted and later discarded unrun drops the overlay with it, and the drop ends the call.
        if host.offer(frame).is_err()
            && let Some((overlaying, _)) = cell.take()
        {
            overlaying.then.end(Err(CheckpointError::LakeClosed));
        }
    }

    /// Writes the replacements, then the index that names them.
    fn shard(self, index: Vec<u8>) {
        let Overlaying {
            io,
            host,
            target,
            tensors,
            shard,
            then,
            ..
        } = self;
        let Some(then) = then.release() else { return };
        let mut writer = SafetensorsWriter::new(&target.join(&shard));
        for (name, values) in &tensors {
            if let Err(error) = writer.add(name, "BF16", &[values.len() as u64], bf16_bytes(values)) {
                return then(Err(error));
            }
        }
        let cell = Handoff::new();
        cell.send((index, Arc::clone(&io), Arc::clone(&host), target));
        let failed = Arc::clone(&then);
        let next = Arc::clone(&then);
        let started = writer.finish(
            &io,
            &host,
            Arc::new(move |written| {
                let Some((index, io, host, target)) = cell.take() else {
                    return;
                };
                if let Err(error) = written {
                    return next(Err(error));
                }
                let size = index.len() as u64;
                if let Err(error) = write_file(
                    &target.join(INDEX),
                    size,
                    bytes_source(index),
                    BUFFER_BYTES,
                    &io,
                    &host,
                    Arc::clone(&next),
                ) {
                    next(Err(error));
                }
            }),
        );
        if let Err(error) = started {
            failed(Err(error));
        }
    }
}

/// Links every file of `source` (but its index and a file named `shard`) into `target`, which is made if it is not there. A name
/// `target` already has is left alone.
#[expect(
    clippy::disallowed_methods,
    reason = "no-wait, blocking directory calls with no AsyncIo form: run as a host lake frame"
)]
fn link_files(source: &Path, target: &Path, shard: &str) -> Result<(), CheckpointError> {
    let context = |what: &str, path: &Path, error| CheckpointError::io(format!("{what} {}", path.display()), error);
    fs::create_dir_all(target).map_err(|error| context("creating", target, error))?;
    for entry in fs::read_dir(source).map_err(|error| context("listing", source, error))? {
        let entry = entry.map_err(|error| context("listing", source, error))?;
        let name = entry.file_name();
        if name == INDEX || name == shard {
            continue;
        }
        let link = target.join(&name);
        if fs::symlink_metadata(&link).is_ok() {
            continue;
        }
        let file = path::absolute(entry.path()).map_err(|error| context("resolving", &entry.path(), error))?;
        symlink(&file, &link).map_err(|error| context("linking", &link, error))?;
    }
    Ok(())
}

#[cfg(unix)]
use std::os::unix::fs as unix_fs;
#[cfg(windows)]
use std::os::windows::fs as windows_fs;

#[cfg(unix)]
fn symlink(file: &Path, link: &Path) -> IoResult<()> {
    unix_fs::symlink(file, link)
}

#[cfg(windows)]
fn symlink(file: &Path, link: &Path) -> IoResult<()> {
    if file.is_dir() {
        windows_fs::symlink_dir(file, link)
    } else {
        windows_fs::symlink_file(file, link)
    }
}

/// `values` as little-endian BF16, each rounded to nearest even; a NaN stays a NaN.
fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| {
            let bits = value.to_bits();
            let rounded = if value.is_nan() {
                (bits >> 16) | 0x0040
            } else {
                bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16
            };
            (rounded as u16).to_le_bytes()
        })
        .collect()
}
