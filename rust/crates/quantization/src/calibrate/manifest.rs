//! What an output directory holds, and of which run: the manifest names the run (its resume key), lists the files the
//! run saved and says how much of the model it rounded, so a consumer takes the codes only when they are all there.
//! A run starts in an empty directory or in its own, and refuses any other: nothing a run did not write is removed.

use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::sync::Arc;

use euhedral_inference_engine::Handoff;
use serde::{Deserialize, Serialize};

use super::CalibrateError;
use super::driver::Then;

/// The manifest's name in the output directory.
pub const MANIFEST_FILE: &str = "calibration.json";
/// The resume file's name in the output directory.
pub const RESUME_FILE: &str = "hidden.bin";
/// The suffix of a file being written, before it is renamed over its name.
const TEMPORARY_SUFFIX: &str = ".tmp";
/// The most names a refusal lists.
const MOST_NAMED: usize = 5;

/// How far a run of one key got.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Manifest {
    /// The resume key of the run that wrote the files.
    pub key: String,
    /// Layers whose objects are all written.
    pub layers_done: usize,
    /// Layers of the model.
    pub layers_total: usize,
    /// Whether the output head and the draft head are written.
    pub heads_done: bool,
    /// Whether the MTP block's objects are written.
    pub mtp_done: bool,
    /// The state files the run saved, each listed once its save sealed.
    pub files: Vec<String>,
}

impl Manifest {
    /// The manifest of a run of `key` that has written nothing.
    pub fn new(key: &str, layers_total: usize) -> Manifest {
        Manifest {
            key: key.to_owned(),
            layers_done: 0,
            layers_total,
            heads_done: false,
            mtp_done: false,
            files: Vec::new(),
        }
    }

    /// The same run after `layers_done` layers.
    pub fn after(&self, layers_done: usize) -> Manifest {
        Manifest {
            layers_done: layers_done.min(self.layers_total),
            ..self.clone()
        }
    }

    /// Whether every object of the model is written.
    pub fn is_complete(&self) -> bool {
        self.layers_done == self.layers_total && self.heads_done && self.mtp_done
    }

    /// The manifest as the file holds it.
    pub fn to_bytes(&self) -> Result<Vec<u8>, CalibrateError> {
        serde_json::to_vec_pretty(self).map_err(|error| CalibrateError::failed("the manifest", error))
    }

    /// The manifest a file holds.
    pub fn parse(bytes: &[u8]) -> Result<Manifest, CalibrateError> {
        serde_json::from_slice(bytes).map_err(|error| CalibrateError::failed("the manifest", error))
    }
}

/// Names a run's key by its parts: the parts of `old` and `new` (joined with `|`) that differ, by the names of the
/// fields, and the layers of the model when they differ.
pub fn differences(old: &Manifest, new: &Manifest, fields: &[&str]) -> Vec<String> {
    let (before, after): (Vec<&str>, Vec<&str>) = (old.key.split('|').collect(), new.key.split('|').collect());
    let mut differing: Vec<String> = if before.len() == after.len() && after.len() == fields.len() {
        fields
            .iter()
            .zip(before.iter().zip(&after))
            .filter(|(_, (before, after))| before != after)
            .map(|(name, _)| (*name).to_owned())
            .collect()
    } else if old.key != new.key {
        vec!["the key".to_owned()]
    } else {
        Vec::new()
    };
    if old.layers_total != new.layers_total {
        differing.push("the model's layers".to_owned());
    }
    differing
}

/// What `inspect` decided for an output directory; `apply` does it.
#[derive(Debug)]
pub struct Prepared {
    out: PathBuf,
    manifest: Manifest,
    /// Whether the directory holds this run's manifest already.
    resumed: bool,
    /// The files the manifest of another run listed, which `--fresh` removes.
    clear: Vec<PathBuf>,
}

impl Prepared {
    /// The manifest the run starts from: the directory's, or one of a run that has written nothing.
    pub fn manifest(&self) -> &Manifest {
        &self.manifest
    }

    /// Makes the directory the run's: removes the files of the manifest `--fresh` replaces and writes the manifest of
    /// a run that has written nothing (to a temporary file, renamed over the name). A resumed directory is left as it
    /// is. Call it only once the run has opened everything it needs.
    ///
    /// Blocking: the run calls it before any of its work starts.
    #[allow(
        clippy::disallowed_methods,
        reason = "a run's setup, before the lattice has any work: it removes and writes files on the calling thread"
    )]
    pub fn apply(self) -> Result<Manifest, CalibrateError> {
        let named = |path: &Path, error| CalibrateError::failed(&path.display().to_string(), error);
        fs::create_dir_all(&self.out).map_err(|error| named(&self.out, error))?;
        for path in &self.clear {
            match fs::remove_file(path) {
                Err(error) if error.kind() != ErrorKind::NotFound => return Err(named(path, error)),
                _ => {}
            }
        }
        if !self.resumed {
            let path = self.out.join(MANIFEST_FILE);
            let mut temporary = path.clone().into_os_string();
            temporary.push(TEMPORARY_SUFFIX);
            let temporary = PathBuf::from(temporary);
            let bytes = self.manifest.to_bytes()?;
            let write = fs::File::create(&temporary)
                .and_then(|mut file| {
                    file.write_all(&bytes)?;
                    file.sync_all()
                })
                .and_then(|()| fs::rename(&temporary, &path));
            write.map_err(|error| named(&path, error))?;
        }
        Ok(self.manifest)
    }
}

fn refuse(out: &Path, why: String) -> CalibrateError {
    CalibrateError::Invalid(format!("{}: {why}", out.display()))
}

fn named(names: &[String]) -> String {
    let shown: Vec<&str> = names.iter().take(MOST_NAMED).map(String::as_str).collect();
    let more = names.len().saturating_sub(MOST_NAMED);
    if more > 0 {
        format!("{} and {more} more", shown.join(", "))
    } else {
        shown.join(", ")
    }
}

/// Decides what a run of `key` does with the directory `out`, and changes nothing there:
/// - an empty (or missing) directory: the run starts;
/// - this run's manifest (the same key and model) whose listed files all exist: the run resumes;
/// - the manifest of another key: refused, naming the key's fields (`fields`, in the key's order) that differ, unless
///   `fresh`, which removes the files that manifest lists and its manifest, and nothing else;
/// - anything else (files and no manifest, a manifest that is not a calibration's or cannot be read, listed files
///   missing): refused, naming what is there.
///
/// Files beside a calibration's manifest that it does not list (a note, the resume file, a state file of the layer a
/// stop interrupted) are the run's neighbours: they are never removed, and a run writes over its own.
///
/// Blocking: the run calls it before any of its work starts.
#[allow(
    clippy::disallowed_methods,
    reason = "a run's setup, before the lattice has any work: it reads the directory on the calling thread"
)]
pub fn inspect(
    out: &Path,
    key: &str,
    layers_total: usize,
    fields: &[&str],
    fresh: bool,
) -> Result<Prepared, CalibrateError> {
    let wanted = Manifest::new(key, layers_total);
    let start = |resumed, manifest: Manifest, clear| Prepared {
        out: out.to_path_buf(),
        manifest,
        resumed,
        clear,
    };
    let entries: Vec<String> = match fs::read_dir(out) {
        Ok(entries) => entries
            .map(|entry| {
                entry
                    .map(|entry| entry.file_name().to_string_lossy().into_owned())
                    .map_err(|error| CalibrateError::failed(&out.display().to_string(), error))
            })
            .collect::<Result<_, _>>()?,
        Err(error) if error.kind() == ErrorKind::NotFound => Vec::new(),
        Err(error) => return Err(CalibrateError::failed(&out.display().to_string(), error)),
    };
    if entries.is_empty() {
        return Ok(start(false, wanted, Vec::new()));
    }
    let path = out.join(MANIFEST_FILE);
    let bytes = match fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == ErrorKind::NotFound => {
            let mut entries = entries;
            entries.sort();
            return Err(refuse(
                out,
                format!(
                    "holds files and no {MANIFEST_FILE}: {}; use an empty directory",
                    named(&entries)
                ),
            ));
        }
        Err(error) => return Err(CalibrateError::failed(&path.display().to_string(), error)),
    };
    let existing = Manifest::parse(&bytes).map_err(|error| {
        refuse(
            out,
            format!("its {MANIFEST_FILE} is not a calibration's manifest or cannot be read ({error})"),
        )
    })?;
    let listed: Vec<PathBuf> = existing.files.iter().map(|name| out.join(name)).collect();
    if existing.key == key && existing.layers_total == layers_total {
        let missing: Vec<String> = existing
            .files
            .iter()
            .filter(|name| !out.join(name).is_file())
            .cloned()
            .collect();
        if !missing.is_empty() {
            return Err(refuse(
                out,
                format!(
                    "its {MANIFEST_FILE} lists files that are not there: {}",
                    named(&missing)
                ),
            ));
        }
        return Ok(start(true, existing, Vec::new()));
    }
    if !fresh {
        return Err(refuse(
            out,
            format!(
                "holds the output of another run (it differs in {}); pass --fresh to remove the files its {MANIFEST_FILE} lists, or use another directory",
                named(&differences(&existing, &wanted, fields))
            ),
        ));
    }
    Ok(start(false, wanted, listed))
}

/// The manifest of a run in progress, which the layers update serially: a layer's files are listed once saved, and the
/// layer is counted only when its manifest is on disk.
#[derive(Debug)]
pub struct Ledger {
    current: Handoff<Manifest>,
}

impl Ledger {
    /// A ledger that starts from `manifest`.
    pub fn new(manifest: Manifest) -> Ledger {
        let current = Handoff::empty();
        current.send(manifest);
        Ledger { current }
    }

    /// Lists files whose saves have sealed.
    pub fn list(&self, names: impl IntoIterator<Item = String>) {
        if let Some(mut manifest) = self.current.take() {
            for name in names {
                if !manifest.files.contains(&name) {
                    manifest.files.push(name);
                }
            }
            self.current.send(manifest);
        }
    }

    /// The manifest as the ledger holds it.
    pub fn manifest(&self) -> Option<Manifest> {
        let manifest = self.current.take()?;
        self.current.send(manifest.clone());
        Some(manifest)
    }

    /// Counts the layers `done` once `result`, the end of the layer's work and its saves, is Ok: `write` saves the
    /// manifest's bytes and gives its result to the callback; only an Ok save changes what the ledger holds. An Err
    /// `result` writes nothing and goes on to `then`.
    pub fn record<W>(self: &Arc<Self>, done: usize, result: Result<(), CalibrateError>, write: W, then: Then<()>)
    where
        W: FnOnce(Vec<u8>, Then<()>),
    {
        if let Err(error) = result {
            return then(Err(error));
        }
        let Some(held) = self.current.take() else {
            return then(Err(CalibrateError::Failed("the manifest is in use".to_owned())));
        };
        let next = held.after(done);
        let bytes = match next.to_bytes() {
            Ok(bytes) => bytes,
            Err(error) => {
                self.current.send(held);
                return then(Err(error));
            }
        };
        let ledger = Arc::clone(self);
        write(
            bytes,
            Box::new(move |saved| {
                ledger.current.send(if saved.is_ok() { next } else { held });
                then(saved);
            }),
        );
    }
}
