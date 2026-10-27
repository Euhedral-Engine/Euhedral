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

/// What a run delivered: the report's bytes, the rows taken, the first fault in the rows' delivery, the engine's outcome.
type Delivered = (Vec<u8>, usize, Option<String>, GenerationOutcome);

/// Collects the rows of one run in the report's format. A row out of order or of another width is the run's fault: it and
/// every later row are left out, and the fault is handed on with the rows taken, so the run is refused, never written short.
struct Report {
    bytes: Vec<u8>,
    row_bytes: usize,
    steps: usize,
    fault: Option<String>,
    done: Option<LatchSetter<Delivered>>,
}

impl Report {
    fn new(bytes: Vec<u8>, row_bytes: usize, done: LatchSetter<Delivered>) -> Report {
        Report {
            bytes,
            row_bytes,
            steps: 0,
            fault: None,
            done: Some(done),
        }
    }
}

impl LogitsSink for Report {
    fn row(&mut self, step: usize, forced: u32, logits: &[u8]) {
        if self.fault.is_some() {
            return;
        }
        if step != self.steps {
            self.fault = Some(format!("row {step} arrived when row {} was next", self.steps));
        } else if logits.len() != self.row_bytes {
            self.fault = Some(format!(
                "row {step} has {} bytes, a row of the vocabulary has {}",
                logits.len(),
                self.row_bytes
            ));
        } else {
            self.steps += 1;
            self.bytes.extend_from_slice(&(forced as i32).to_le_bytes());
            self.bytes.extend_from_slice(logits);
        }
    }

    fn finished(&mut self, outcome: GenerationOutcome) {
        if let Some(done) = self.done.take() {
            done.set((mem::take(&mut self.bytes), self.steps, self.fault.take(), outcome));
        }
    }
}

/// The report's bytes of a delivered run, or the reason it is refused: a fault in the rows, rows missing, or an engine
/// outcome that is not a full run of `steps` steps.
fn accept(delivered: Delivered, steps: usize) -> Result<Vec<u8>, ConfigError> {
    let (bytes, taken, fault, outcome) = delivered;
    if let Some(fault) = fault {
        return invalid(format!(
            "the report is refused, {fault} (after {taken} of {steps} rows)"
        ));
    }
    if taken != steps {
        return invalid(format!("the report is refused: {taken} of {steps} rows arrived"));
    }
    if outcome.reason != FinishReason::Length || outcome.generated != steps {
        return invalid(format!(
            "the run ended with {:?} after {} of {steps} steps",
            outcome.reason, outcome.generated
        ));
    }
    Ok(bytes)
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
        .teacher_forced(prefix, forced, Box::new(Report::new(bytes, 2 * vocabulary, done)))
        .or_else(|error| invalid(format!("the run was refused: {error}")))?;
    accept(waiting.wait(), steps)
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    fn outcome(generated: usize) -> GenerationOutcome {
        GenerationOutcome {
            reason: FinishReason::Length,
            prompt_tokens: 4,
            generated,
            elapsed: Duration::ZERO,
        }
    }

    /// Drives a report with `rows` as `(step, logits bytes)` and returns what `teacher_forced_report` would decide.
    fn run(rows: &[(usize, usize)], steps: usize, generated: usize) -> Result<Vec<u8>, ConfigError> {
        let (done, waiting) = Latch::new();
        let mut report = Report::new(Vec::new(), 4, done);
        for (step, width) in rows {
            report.row(*step, 7, &vec![0u8; *width]);
        }
        report.finished(outcome(generated));
        accept(waiting.wait(), steps)
    }

    fn message(result: Result<Vec<u8>, ConfigError>) -> String {
        let ConfigError::Invalid(message) = result.expect_err("the report is refused");
        message
    }

    #[test]
    fn a_complete_report_is_accepted() {
        let bytes = run(&[(0, 4), (1, 4), (2, 4)], 3, 3).unwrap();
        assert_eq!(bytes.len(), 3 * (4 + 4));
    }

    #[test]
    fn a_row_of_the_wrong_width_refuses_the_report_naming_the_row() {
        // The engine counted all three steps; the report holds two rows.
        let text = message(run(&[(0, 4), (1, 6), (2, 4)], 3, 3));
        assert!(
            text.contains("row 1 has 6 bytes") && text.contains("after 1 of 3 rows"),
            "{text}"
        );
    }

    #[test]
    fn a_row_out_of_order_refuses_the_report_naming_the_row() {
        let text = message(run(&[(0, 4), (2, 4), (1, 4)], 3, 3));
        assert!(text.contains("row 2 arrived when row 1 was next"), "{text}");
    }

    #[test]
    fn missing_rows_and_a_short_run_refuse_the_report() {
        assert!(message(run(&[(0, 4), (1, 4)], 3, 3)).contains("2 of 3 rows arrived"));
        assert!(message(run(&[(0, 4), (1, 4), (2, 4)], 3, 2)).contains("after 2 of 3 steps"));
    }
}
