//! What a calibration covers and how it resumes.

use crate::resume::Header;
use std::cmp::Reverse;

use super::CalibrateError;
use super::manifest::{MANIFEST_FILE, Manifest, RESUME_FILE};

use sha2::{Digest, Sha256};

/// Layers between two checkpoints of the hidden states, as Java's calibration.
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
    /// A plan of `layers` layers, checkpointing as Java does.
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

/// What names a run: a resume file of another run is not resumed. Java's key (the tokens file, the sequences, the
/// length, the recipe, act-order and damp) with the model and the NVFP4 search added: a file of a run that rounded
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
