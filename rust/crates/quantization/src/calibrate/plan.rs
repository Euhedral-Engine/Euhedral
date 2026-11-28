//! What a calibration covers and how it resumes.

use crate::resume::Header;
use std::cmp::Reverse;
use std::fs;
use std::io;
use std::path::Path;

use super::CalibrateError;
use super::manifest::{MANIFEST_FILE, Manifest, RESUME_FILE};

use sha2::{Digest, Sha256};

/// Layers between two checkpoints of the hidden states.
pub const CHECKPOINT_EVERY: usize = 8;

/// The header property naming the run a resume file belongs to.
pub const KEY_PROPERTY: &str = "key";
/// The header property counting the layers a resume file's states are the output of.
pub const LAYERS_PROPERTY: &str = "layers";

/// The layers a run covers and when it checkpoints.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Plan {
    /// The layers of the model the run rounds (`--layers`).
    pub layers: usize,
    /// Layers between two checkpoints; 0 never checkpoints.
    pub checkpoint_every: usize,
    /// Stops after this many layers, leaving the checkpoint and skipping the heads; 0 runs on to the end.
    pub stop_after: usize,
}

impl Plan {
    /// A plan of `layers` layers.
    pub fn new(layers: usize) -> Plan {
        Plan {
            layers,
            checkpoint_every: CHECKPOINT_EVERY,
            stop_after: 0,
        }
    }

    /// One past the last layer the run rounds.
    pub fn stop(&self) -> usize {
        if self.stop_after > 0 {
            self.layers.min(self.stop_after)
        } else {
            self.layers
        }
    }

    /// Whether the run ends before the model's layers do: then it skips the heads and keeps its checkpoint.
    pub fn stops_early(&self) -> bool {
        self.stop() < self.layers
    }

    /// Whether the states after `done` layers are checkpointed: every `checkpoint_every` layers, but not at the end.
    pub fn checkpoint_due(&self, done: usize) -> bool {
        self.checkpoint_every > 0 && done.is_multiple_of(self.checkpoint_every) && done < self.layers
    }
}

/// A digest of the calibration tokens' contents, or of the draft ids', in the resume key: the same file name over
/// other values is another run.
pub fn values_digest(values: &[i32]) -> String {
    let mut hash = Sha256::new();
    for value in values {
        hash.update(value.to_le_bytes());
    }
    format!("{:x}", hash.finalize())
}

/// A digest of what a checkpoint directory holds, in the resume key: a checkpoint rewritten at the same path is
/// another run. Every regular file counts by name; a `.json` file (the config, the index, the generation config) by
/// its bytes; any other file (the shards) by its size alone, so the shards' contents are never read: a rebuilt shard
/// of the same size is not told apart, which is the price of not reading tens of gigabytes at every start.
#[allow(
    clippy::disallowed_methods,
    reason = "a binary's startup: the directory is read on its main thread before the run starts"
)]
pub fn model_identity(directory: &Path) -> io::Result<String> {
    let mut files = Vec::new();
    for entry in fs::read_dir(directory)? {
        let entry = entry?;
        // `metadata` follows a link, as in a Hugging Face cache snapshot, where every file is one.
        if entry.path().metadata().is_ok_and(|metadata| metadata.is_file()) {
            files.push(entry.file_name().to_string_lossy().into_owned());
        }
    }
    files.sort();
    let mut hash = Sha256::new();
    for name in files {
        let path = directory.join(&name);
        hash.update((name.len() as u64).to_le_bytes());
        hash.update(name.as_bytes());
        if name.ends_with(".json") {
            let bytes = fs::read(&path)?;
            hash.update(b"b");
            hash.update((bytes.len() as u64).to_le_bytes());
            hash.update(bytes);
        } else {
            hash.update(b"s");
            hash.update(fs::metadata(&path)?.len().to_le_bytes());
        }
    }
    Ok(format!("{:x}", hash.finalize()))
}

/// What names a run: a resume file of another run is not resumed. The key holds the tokens file, the sequences, the
/// length, the recipe, act-order and damp, the model and the NVFP4 search: a file of a run that rounded
/// differently must not be resumed either.
pub fn resume_key(parts: &[&str]) -> String {
    parts.join("|")
}

/// The layer a run of `layers` layers resumes at, from its resume file's `header` and the directory's `manifest`:
/// - the manifest does not list the resume file (a new manifest, or no checkpoint of the run sealed): 0, whatever
///   the file holds;
/// - the listed file names another key than the manifest's: refused, since the two records of the directory disagree;
/// - else the layers the header says are done, when they are at most the run's layers and the layers the manifest
///   counts; otherwise (states past what the manifest counts, or an unreadable count) 0. The states are the output
///   of the header's layers, so a run never resumes at any other layer.
pub fn resume_at(header: &Header, manifest: &Manifest, layers: usize) -> Result<usize, CalibrateError> {
    if !manifest.lists_resume_file() {
        return Ok(0);
    }
    if header.get(KEY_PROPERTY) != Some(manifest.key.as_str()) {
        return Err(CalibrateError::Invalid(format!(
            "{RESUME_FILE} holds the states of another run than the one its {MANIFEST_FILE} names; pass --fresh to \
             remove the files it lists and start over"
        )));
    }
    Ok(match header.get(LAYERS_PROPERTY).map(str::parse::<usize>) {
        Some(Ok(done)) if done <= layers && done <= manifest.layers_done => done,
        _ => 0,
    })
}

/// The header of the resume file after `done` layers.
pub fn resume_header(key: &str, done: usize) -> Header {
    let mut header = Header::new();
    header.set(KEY_PROPERTY, key);
    header.set(LAYERS_PROPERTY, done.to_string());
    header
}

/// The order a layer's linears are rounded in: the largest `rows * columns` first, ties in their own order. The
/// result is indices into `shapes`.
pub fn round_order(shapes: &[(String, (u64, u64))]) -> Vec<usize> {
    let mut order: Vec<usize> = (0..shapes.len()).collect();
    order.sort_by_key(|&index| Reverse(shapes[index].1.0.saturating_mul(shapes[index].1.1)));
    order
}

#[cfg(test)]
#[allow(clippy::disallowed_methods, reason = "tests write their fixtures on the test thread")]
mod tests {
    use super::*;
    use tempfile::{TempDir, tempdir};

    fn model(files: &[(&str, &str)]) -> TempDir {
        let directory = tempdir().unwrap();
        for (name, text) in files {
            fs::write(directory.path().join(name), text).unwrap();
        }
        directory
    }

    fn identity(directory: &TempDir) -> String {
        model_identity(directory.path()).unwrap()
    }

    const FILES: [(&str, &str); 3] = [
        ("config.json", "{\"layers\": 64}"),
        ("model.safetensors.index.json", "{}"),
        ("model-00001.safetensors", "0123456789"),
    ];

    #[test]
    fn the_same_files_give_the_same_identity_wherever_they_are() {
        assert_eq!(identity(&model(&FILES)), identity(&model(&FILES)));
    }

    #[test]
    fn a_changed_config_byte_a_changed_shard_size_and_an_added_or_removed_file_change_it() {
        let base = identity(&model(&FILES));
        let mut config = FILES;
        config[0].1 = "{\"layers\": 65}";
        assert_ne!(base, identity(&model(&config)), "a config byte");
        let mut shard = FILES;
        shard[2].1 = "01234567890";
        assert_ne!(base, identity(&model(&shard)), "a shard's size");
        assert_ne!(base, identity(&model(&FILES[..2])), "a removed shard");
        let mut more = FILES.to_vec();
        more.push(("model-00002.safetensors", ""));
        assert_ne!(base, identity(&model(&more)), "an added shard");
    }

    #[test]
    fn a_shard_is_told_by_its_size_not_its_bytes() {
        let base = model(&FILES);
        let other = model(&FILES);
        fs::write(other.path().join("model-00001.safetensors"), "abcdefghij").unwrap();
        assert_eq!(identity(&base), identity(&other));
    }

    #[cfg(unix)]
    #[test]
    fn a_shard_that_cannot_be_read_still_has_an_identity() {
        use std::os::unix::fs::PermissionsExt;
        let directory = model(&FILES);
        let base = identity(&directory);
        let shard = directory.path().join("model-00001.safetensors");
        fs::set_permissions(&shard, fs::Permissions::from_mode(0o0)).unwrap();
        assert_eq!(identity(&directory), base);
    }

    #[test]
    fn a_missing_directory_is_an_error() {
        let directory = tempdir().unwrap();
        assert!(model_identity(&directory.path().join("absent")).is_err());
    }
}
