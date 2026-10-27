//! The teacher-forced report: the engine is fed the first `prefix` tokens of the benchmark corpus document, then decodes
//! the next `steps` tokens of it one row at a time, and the logits row of each step is written in the Java quality
//! tool's format (`tools/compare_teacher_forced.py` reads it):
//!
//! ```text
//! i32 steps, i32 vocabulary                      (little endian)
//! per step: i32 forced token, vocabulary BF16 logits (little endian)
//! ```

use std::fs;
use std::mem;
use std::path::Path;
use std::sync::Arc;

use euhedral_graph::{Latch, LatchSetter};
use euhedral_inference_engine::EngineRuntime;
use euhedral_inference_engine::generation::{FinishReason, GenerationOutcome, LogitsSink};
use euhedral_inference_tokenizer::Tokenizer;

use crate::config::ConfigError;
use crate::prompt::chat;

fn invalid<T>(message: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError::Invalid(message.into()))
}

/// The prefix and the forced tokens: tokens `0..prefix` and `prefix..prefix + steps` of the corpus document.
pub fn split_document(tokenizer: &Tokenizer, prefix: usize, steps: usize) -> Result<(Vec<u32>, Vec<u32>), ConfigError> {
    let document = tokenizer
        .encode(chat::DOCUMENT)
        .or_else(|error| invalid(format!("cannot tokenize the corpus document: {error}")))?;
    if prefix == 0 || steps == 0 {
        return invalid("--prefix and --steps must be positive");
    }
    if document.len() < prefix + steps {
        return invalid(format!(
            "the corpus document has {} tokens, fewer than prefix {prefix} + steps {steps}",
            document.len()
        ));
    }
    Ok((document[..prefix].to_vec(), document[prefix..prefix + steps].to_vec()))
}

/// Collects the rows of one run in the report's format.
struct Report {
    bytes: Vec<u8>,
    row_bytes: usize,
    steps: usize,
    done: Option<LatchSetter<(Vec<u8>, GenerationOutcome)>>,
}

impl LogitsSink for Report {
    fn row(&mut self, step: usize, forced: u32, logits: &[u8]) {
        if step != self.steps || logits.len() != self.row_bytes {
            // A row out of order or of another width ends the report short; `finished` says so.
            self.steps = usize::MAX;
            return;
        }
        self.steps += 1;
        self.bytes.extend_from_slice(&(forced as i32).to_le_bytes());
        self.bytes.extend_from_slice(logits);
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        if let Some(done) = self.done.take() {
            done.set((mem::take(&mut self.bytes), outcome));
        }
    }
}

/// Runs the teacher-forced decode of `forced` after `prefix` on `runtime` and returns the report's bytes. Waits for the
/// run: called by the binary's main thread only.
pub fn teacher_forced_report(
    runtime: &Arc<EngineRuntime>,
    prefix: Vec<u32>,
    forced: Vec<u32>,
) -> Result<Vec<u8>, ConfigError> {
    let vocabulary = runtime.config().vocab_size;
    let steps = forced.len();
    let mut bytes = Vec::with_capacity(8 + steps * (4 + 2 * vocabulary));
    bytes.extend_from_slice(&(steps as i32).to_le_bytes());
    bytes.extend_from_slice(&(vocabulary as i32).to_le_bytes());
    let (done, waiting) = Latch::new();
    runtime
        .teacher_forced(
            prefix,
            forced,
            Box::new(Report {
                bytes,
                row_bytes: 2 * vocabulary,
                steps: 0,
                done: Some(done),
            }),
        )
        .or_else(|error| invalid(format!("the run was refused: {error}")))?;
    let (bytes, outcome) = waiting.wait();
    if outcome.reason != FinishReason::Length || outcome.generated != steps {
        return invalid(format!(
            "the run ended with {:?} after {} of {steps} steps",
            outcome.reason, outcome.generated
        ));
    }
    Ok(bytes)
}

/// Writes the report.
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: the report is written once, on the main thread, after the run"
)]
pub fn write_report(path: &Path, bytes: &[u8]) -> Result<(), ConfigError> {
    if let Some(parent) = path.parent().filter(|parent| !parent.as_os_str().is_empty()) {
        fs::create_dir_all(parent).or_else(|error| invalid(format!("{}: {error}", parent.display())))?;
    }
    fs::write(path, bytes).or_else(|error| invalid(format!("{}: {error}", path.display())))
}
