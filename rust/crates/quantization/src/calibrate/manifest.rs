//! What an output directory holds, and of which run: the manifest names the run (its resume key) and says how much of
//! the model it rounded, so a consumer takes the codes only when they are all there, and a run of another key does not
//! leave its files among the new run's.

use std::fs;
use std::io::{self, ErrorKind};
use std::path::Path;

use serde::{Deserialize, Serialize};

use super::CalibrateError;

/// The manifest's name in the output directory.
pub const MANIFEST_FILE: &str = "calibration.json";
/// The resume file's name in the output directory.
pub const RESUME_FILE: &str = "hidden.bin";
/// The suffix of a state file.
const STATE_SUFFIX: &str = ".npz";
/// The suffix of a file being written, before it is renamed over its name.
const TEMPORARY_SUFFIX: &str = ".tmp";

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

/// Whether `name` is a file a run writes into the output directory.
fn is_output(name: &str) -> bool {
    let name = name.strip_suffix(TEMPORARY_SUFFIX).unwrap_or(name);
    name.ends_with(STATE_SUFFIX) || name == RESUME_FILE || name == MANIFEST_FILE
}

/// Makes `out` the directory of the run of `key`: when the manifest there is of this key the files stay, so the run
/// resumes; else (another key, no manifest, a manifest that cannot be read) every file a run writes is removed, so the
/// old run's later layers do not stay among the new run's. Writes the manifest of a run that has done nothing when it
/// removed the files. Returns the manifest in force.
///
/// Blocking: the run calls it before any of its work starts.
#[allow(
    clippy::disallowed_methods,
    reason = "a run's setup, before the lattice has any work: it reads and removes files on the calling thread"
)]
pub fn prepare(out: &Path, key: &str, layers_total: usize) -> Result<Manifest, CalibrateError> {
    fs::create_dir_all(out).map_err(|error| CalibrateError::failed(&out.display().to_string(), error))?;
    let path = out.join(MANIFEST_FILE);
    let existing = match fs::read(&path) {
        Ok(bytes) => Manifest::parse(&bytes).ok(),
        Err(error) if error.kind() == ErrorKind::NotFound => None,
        Err(error) => return Err(CalibrateError::failed(&path.display().to_string(), error)),
    };
    if let Some(existing) = existing
        && existing.key == key
        && existing.layers_total == layers_total
    {
        return Ok(existing);
    }
    clear(out).map_err(|error| CalibrateError::failed(&out.display().to_string(), error))?;
    let fresh = Manifest::new(key, layers_total);
    fs::write(&path, fresh.to_bytes()?).map_err(|error| CalibrateError::failed(&path.display().to_string(), error))?;
    Ok(fresh)
}

#[allow(
    clippy::disallowed_methods,
    reason = "a run's setup, before the lattice has any work: it removes files on the calling thread"
)]
fn clear(out: &Path) -> io::Result<()> {
    for entry in fs::read_dir(out)? {
        let entry = entry?;
        if entry.file_type()?.is_file() && entry.file_name().to_str().is_some_and(is_output) {
            fs::remove_file(entry.path())?;
        }
    }
    Ok(())
}
