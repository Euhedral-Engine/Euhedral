//! What an output directory holds, and of which run: the manifest names the run (its resume key), lists the files the
//! run saved (the resume file among them, once a checkpoint sealed) and says how much of the model it rounded, so a
//! consumer takes the codes only when they are all there. It is the one record of the directory: a run resumes only
//! from what it lists, counts only layers whose files it lists, and `--fresh` removes only what it lists. A run starts
//! in an empty directory or in its own, and refuses any other.

use std::fs;
use std::io::{ErrorKind, Write};
use std::path::{Component, Path, PathBuf};
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
/// Characters a listed name may not hold: the separators of every platform, and Windows' stream and drive mark.
const NOT_IN_NAMES: [char; 4] = ['/', '\\', ':', '\0'];

/// The parts of a run that are written after its layers, each sealed and listed on its own.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Part {
    /// The MTP block: its input projection and its layer's linears.
    Mtp,
    /// The output head.
    OutputHead,
    /// The draft head; the heads are written once it is.
    DraftHead,
}

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

    /// The manifest a file holds. Every listed name must be a plain file name of the directory (see [`plain`]): a
    /// manifest that names anything else is refused, so nothing outside the directory is ever read or removed.
    pub fn parse(bytes: &[u8]) -> Result<Manifest, CalibrateError> {
        let manifest: Manifest =
            serde_json::from_slice(bytes).map_err(|error| CalibrateError::failed("the manifest", error))?;
        if let Some(name) = manifest.files.iter().find(|name| !plain(name)) {
            return Err(CalibrateError::Invalid(format!(
                "the manifest lists {name:?}, which is not a file name of the directory"
            )));
        }
        Ok(manifest)
    }

    /// Whether the run's resume file is one of the files it saved.
    pub fn lists_resume_file(&self) -> bool {
        self.files.iter().any(|name| name == RESUME_FILE)
    }
}

/// Whether `name` names a file directly in the directory, on every platform: one normal path component, no separator,
/// not `.` or `..`, not absolute or prefixed, and not the manifest itself.
pub fn plain(name: &str) -> bool {
    let mut components = Path::new(name).components();
    !name.contains(NOT_IN_NAMES)
        && name != MANIFEST_FILE
        && matches!(components.next(), Some(Component::Normal(_)))
        && components.next().is_none()
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
    /// The files the manifest `--fresh` replaces listed, which `apply` removes.
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
/// - a calibration's manifest and `fresh`: the files that manifest lists are removed (those that are there), and the
///   run starts over with a manifest that lists nothing, whatever key the old one had;
/// - this run's manifest (the same key and model) whose listed files all exist: the run resumes;
/// - this run's manifest with listed files missing: refused, naming them and `--fresh`;
/// - the manifest of another key: refused, naming the key's fields (`fields`, in the key's order) that differ and
///   `--fresh`;
/// - anything else (files and no manifest, a manifest that is not a calibration's, cannot be read or lists a name that
///   is not a plain file name): refused, naming what is there, even with `fresh`.
///
/// Files beside a calibration's manifest that it does not list (a note, a state file of the layer a stop
/// interrupted, a resume file no checkpoint of the run sealed) are never removed; a run writes over its own.
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
    if fresh {
        let listed = existing.files.iter().map(|name| out.join(name)).collect();
        return Ok(start(false, wanted, listed));
    }
    if existing.key != key || existing.layers_total != layers_total {
        return Err(refuse(
            out,
            format!(
                "holds the output of another run (it differs in {}); pass --fresh to remove the files its {MANIFEST_FILE} lists, or use another directory",
                named(&differences(&existing, &wanted, fields))
            ),
        ));
    }
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
                "its {MANIFEST_FILE} lists files that are not there: {}; pass --fresh to remove the files it lists and start over",
                named(&missing)
            ),
        ));
    }
    Ok(start(true, existing, Vec::new()))
}

/// The manifest of a run in progress, which the layers and the checkpoints update serially: a layer's files are listed
/// once saved, and the layer is counted only when its manifest is on disk and every layer before it is counted, so
/// the count is always the layers from the first whose files are all listed. The resume file is listed once a
/// checkpoint sealed it, and that listing is saved before the run goes on.
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

    /// Lists files whose saves have sealed. False, listing nothing, while a save of the manifest is in flight: the
    /// caller fails its layer, which then is not counted.
    pub fn list(&self, names: impl IntoIterator<Item = String>) -> bool {
        let Some(mut manifest) = self.current.take() else {
            return false;
        };
        for name in names {
            if !manifest.files.contains(&name) {
                manifest.files.push(name);
            }
        }
        self.current.send(manifest);
        true
    }

    /// The manifest as the ledger holds it.
    pub fn manifest(&self) -> Option<Manifest> {
        let manifest = self.current.take()?;
        self.current.send(manifest.clone());
        Some(manifest)
    }

    /// Counts layer `layer` once `result`, the end of the layer's work and its saves, is Ok: `write` saves the
    /// manifest's bytes and gives its result to the callback; only an Ok save changes what the ledger holds. An Err
    /// `result` writes nothing and goes on to `then`. A layer past the first one not counted is refused, writing
    /// nothing: counting it would count the layers between, whose files the manifest does not list. A layer already
    /// counted (a resumed run redoing the layers after its checkpoint) leaves the count as it is.
    pub fn record<W>(self: &Arc<Self>, layer: usize, result: Result<(), CalibrateError>, write: W, then: Then<()>)
    where
        W: FnOnce(Vec<u8>, Then<()>),
    {
        if let Err(error) = result {
            return then(Err(error));
        }
        self.commit(
            |held| {
                if layer > held.layers_done {
                    return Err(CalibrateError::Failed(format!(
                        "layer {layer} cannot be counted: the manifest counts {} layers",
                        held.layers_done
                    )));
                }
                Ok(held.after(held.layers_done.max(layer + 1)))
            },
            write,
            then,
        );
    }

    /// Lists the resume file once `result`, the end of a checkpoint, is Ok, and saves the manifest through `write` as
    /// [`Ledger::record`] does: the run resumes from the checkpoint only once the manifest on disk lists it. An Err
    /// `result` writes nothing and goes on to `then`.
    pub fn checkpointed<W>(self: &Arc<Self>, result: Result<(), CalibrateError>, write: W, then: Then<()>)
    where
        W: FnOnce(Vec<u8>, Then<()>),
    {
        if let Err(error) = result {
            return then(Err(error));
        }
        self.commit(
            |held| {
                let mut next = held.clone();
                if !next.lists_resume_file() {
                    next.files.push(RESUME_FILE.to_owned());
                }
                Ok(next)
            },
            write,
            then,
        );
    }

    /// Lists the files of `part`, which are saved, and marks the part written, saving the manifest through `write` as
    /// [`Ledger::record`] does: a part counts only once its files are listed on disk.
    pub fn sealed<W>(self: &Arc<Self>, part: Part, files: Vec<String>, write: W, then: Then<()>)
    where
        W: FnOnce(Vec<u8>, Then<()>),
    {
        self.commit(
            |held| {
                let mut next = held.clone();
                for name in files {
                    if !next.files.contains(&name) {
                        next.files.push(name);
                    }
                }
                match part {
                    Part::Mtp => next.mtp_done = true,
                    Part::OutputHead => {}
                    Part::DraftHead => next.heads_done = true,
                }
                Ok(next)
            },
            write,
            then,
        );
    }

    /// Unlists the resume file, saving the manifest through `write`: the caller removes the file only once this is
    /// Ok, so the manifest never lists a file that is gone.
    pub fn resume_file_dropped<W>(self: &Arc<Self>, write: W, then: Then<()>)
    where
        W: FnOnce(Vec<u8>, Then<()>),
    {
        self.commit(
            |held| {
                let mut next = held.clone();
                next.files.retain(|name| name != RESUME_FILE);
                Ok(next)
            },
            write,
            then,
        );
    }

    /// Saves the manifest `change` makes of the one held; the ledger holds the new one only once the save is Ok.
    fn commit<C, W>(self: &Arc<Self>, change: C, write: W, then: Then<()>)
    where
        C: FnOnce(&Manifest) -> Result<Manifest, CalibrateError>,
        W: FnOnce(Vec<u8>, Then<()>),
    {
        let Some(held) = self.current.take() else {
            return then(Err(CalibrateError::Failed("the manifest is in use".to_owned())));
        };
        let (bytes, next) = match change(&held).and_then(|next| next.to_bytes().map(|bytes| (bytes, next))) {
            Ok(changed) => changed,
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
