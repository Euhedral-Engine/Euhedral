//! The result row: schema `euhedral-inference.benchmark-result` version 1, with the field names, the field order and the
//! null-for-absent convention of the Java benchmark's `BenchmarkResult`, so one reader (and `tools/rust_gate.py`) takes
//! rows of both engines. Timings are integer nanoseconds and rates are tokens per second; a value is `null` when the
//! work it describes did not complete.
//!
//! A Rust row differs from a Java row in values only: `implementation` is `rust`, and `engine` (free-form in the schema)
//! describes this engine.

use std::collections::BTreeMap;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};

use crate::config::ConfigError;

pub const SCHEMA: &str = "euhedral-inference.benchmark-result";
pub const DOCUMENT_SCHEMA: &str = "euhedral-inference.benchmark-results";
pub const SCHEMA_VERSION: u32 = 1;
/// The `implementation` of a row of this engine.
pub const IMPLEMENTATION: &str = "rust";
pub const TOOL: &str = "euhedral-inference-benchmark";
pub const SUCCESS: &str = "success";
pub const INELIGIBLE: &str = "ineligible";
pub const FAILED: &str = "failed";

/// One benchmark row.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Row {
    pub schema: String,
    pub schema_version: u32,
    pub implementation: String,
    pub provenance: Provenance,
    pub run_id: String,
    pub fork: Fork,
    pub recorded_at: String,
    pub scenario: ScenarioRecord,
    pub warmup: bool,
    pub iteration: usize,
    pub status: String,
    pub status_reason: Option<String>,
    pub work: Work,
    pub timings: Option<Timings>,
    pub throughput: Option<Throughput>,
    pub engine: Value,
    pub gpu_memory: Option<GpuMemory>,
}

/// `kind` is `measured` for harness rows and `imported` for external rows.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Provenance {
    pub kind: String,
    pub tool: String,
    pub command_line: String,
    pub source: Option<String>,
    pub source_sha256: Option<String>,
    pub note: Option<String>,
}

/// `id` is the externally supplied fork identifier (`source` `external`), or null (`source` `unspecified`).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Fork {
    pub id: Option<String>,
    pub source: String,
    pub pid: Option<u64>,
    /// The process start, as the field name says in Java (`jvmStartedAt`); this engine has no JVM, the field is the
    /// process's start.
    pub jvm_started_at: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ScenarioRecord {
    pub name: String,
    pub kind: String,
    pub target_prompt_tokens: usize,
    pub requested_new_tokens: usize,
    pub prompt_generator: String,
    pub prompt_seed: i64,
    pub prompt_sha256: String,
}

/// Actual completed work. `generated_tokens` counts every returned token, a terminator included;
/// `decode_sampled_tokens` those from decode (tokens 2..N). `final_commit_quanta` is always 0 here: this engine ends a
/// request after its last token is selected, with no commit-only quantum.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Work {
    pub prompt_tokens: Option<usize>,
    pub prefill_quanta: Option<usize>,
    pub prefill_tokens: Option<usize>,
    pub generated_tokens: Option<usize>,
    pub decode_sampled_tokens: Option<usize>,
    pub final_commit_quanta: Option<usize>,
    pub eos_observed: Option<bool>,
    pub speculative: Option<Speculative>,
    pub output_sha256: Option<String>,
    pub prefix_restored_tokens: Option<usize>,
}

/// Speculative decoding: verifications, `accepted_drafts[a]` = verifications that accepted `a` drafts, the tokens they
/// committed, and the time spent verifying and drafting (ns).
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Speculative {
    pub verifications: usize,
    pub accepted_drafts: Vec<u64>,
    pub committed_tokens: usize,
    pub verify_nanos: i64,
    pub draft_nanos: i64,
    pub rejections_in_shortlist: usize,
    pub rejections_outside_shortlist: usize,
    pub draft_phases: BTreeMap<String, DraftPhase>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DraftPhase {
    pub quanta: u64,
    pub nanos: i64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Timings {
    pub tokenization: Option<i64>,
    pub prefill: Option<i64>,
    pub first_token_sample: Option<i64>,
    pub time_to_first_token: Option<i64>,
    pub decode: Option<i64>,
    pub decode_quanta_sum: Option<i64>,
    pub final_commit: Option<i64>,
    pub time_to_last_token: Option<i64>,
    pub end_to_end: Option<i64>,
    pub prefix_restore: Option<i64>,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct Throughput {
    pub prefill_tokens_per_second: Option<f64>,
    pub decode_tokens_per_second: Option<f64>,
    pub end_to_end_output_tokens_per_second: Option<f64>,
}

/// Device free/total bytes around the iteration, observed outside timing, and the engine's own allocations.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GpuMemory {
    pub before_free_bytes: Option<u64>,
    pub after_free_bytes: Option<u64>,
    pub total_bytes: Option<u64>,
    pub allocated_before_bytes: Option<u64>,
    pub peak_allocated_bytes: Option<u64>,
    pub allocated_after_bytes: Option<u64>,
    pub retained_workspace_bytes: Option<u64>,
}

impl Row {
    /// The row as one JSON line (no newline).
    pub fn to_json(&self) -> String {
        // A row holds only strings, numbers, booleans, nulls and a JSON value: it always serializes.
        serde_json::to_string(self).unwrap_or_else(|error| format!("{{\"serializationError\":\"{error}\"}}"))
    }
}

/// The SHA-256 of the returned token ids, as decimal numbers joined by commas, in lower-case hex: equal across runs and
/// engines that must generate the same tokens.
pub fn output_sha256(tokens: &[u32]) -> String {
    let text = tokens.iter().map(u32::to_string).collect::<Vec<_>>().join(",");
    format!("{:x}", Sha256::digest(text.as_bytes()))
}

/// The calendar time of `seconds` since the Unix epoch, UTC: (year, month, day, hour, minute, second).
pub fn civil(seconds: u64) -> (i64, u32, u32, u32, u32, u32) {
    let days = (seconds / 86_400) as i64;
    let rest = seconds % 86_400;
    // Days since 0000-03-01 (Howard Hinnant's civil-from-days).
    let shifted = days + 719_468;
    let era = shifted.div_euclid(146_097);
    let day_of_era = shifted.rem_euclid(146_097);
    let year_of_era = (day_of_era - day_of_era / 1_460 + day_of_era / 36_524 - day_of_era / 146_096) / 365;
    let day_of_year = day_of_era - (365 * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_index = (5 * day_of_year + 2) / 153;
    let day = (day_of_year - (153 * month_index + 2) / 5 + 1) as u32;
    let month = if month_index < 10 {
        month_index + 3
    } else {
        month_index - 9
    } as u32;
    let year = year_of_era + era * 400 + i64::from(month <= 2);
    (
        year,
        month,
        day,
        (rest / 3_600) as u32,
        (rest % 3_600 / 60) as u32,
        (rest % 60) as u32,
    )
}

/// `time` as Java's `Instant.toString()` prints it: seconds precision, with the fraction in groups of three digits only
/// when it is not zero.
pub fn iso_instant(time: SystemTime) -> String {
    let since = time.duration_since(UNIX_EPOCH).unwrap_or_default();
    let (year, month, day, hour, minute, second) = civil(since.as_secs());
    let mut text = format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}");
    let nanos = since.subsec_nanos();
    if nanos != 0 {
        let digits = if nanos.is_multiple_of(1_000_000) {
            format!("{:03}", nanos / 1_000_000)
        } else if nanos.is_multiple_of(1_000) {
            format!("{:06}", nanos / 1_000)
        } else {
            format!("{nanos:09}")
        };
        text.push('.');
        text.push_str(&digits);
    }
    text.push('Z');
    text
}

/// Writes the rows of one run: JSONL as they are produced (flushed per row, so a stopped run keeps what it measured), or
/// one JSON document written when the run closes.
#[derive(Debug)]
pub struct ResultWriter {
    output: PathBuf,
    json: bool,
    lines: Option<BufWriter<File>>,
    rows: Vec<Row>,
}

impl ResultWriter {
    /// Rejects an existing output unless overwrite or append was asked for, and an output that cannot be written.
    pub fn check_output(output: &Path, overwrite: bool, append: bool) -> Result<(), ConfigError> {
        if output.is_dir() {
            return Err(ConfigError::Invalid(format!(
                "output is a directory: {}",
                output.display()
            )));
        }
        if output.exists() && !overwrite && !append {
            return Err(ConfigError::Invalid(format!(
                "output exists; set overwrite or append: {}",
                output.display()
            )));
        }
        Ok(())
    }

    /// Opens the output, creating its directory.
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, blocking I/O: the result file is written by the main thread between iterations, outside any timing"
    )]
    pub fn create(output: &Path, json: bool, overwrite: bool, append: bool) -> Result<ResultWriter, ConfigError> {
        ResultWriter::check_output(output, overwrite, append)?;
        let io = |error: io::Error| ConfigError::Invalid(format!("{}: {error}", output.display()));
        if let Some(parent) = output.parent().filter(|parent| !parent.as_os_str().is_empty()) {
            fs::create_dir_all(parent).map_err(io)?;
        }
        let lines = if json {
            None
        } else {
            let mut options = fs::OpenOptions::new();
            options.create(true).write(true);
            if append {
                options.append(true);
            } else {
                options.truncate(true);
            }
            Some(BufWriter::new(options.open(output).map_err(io)?))
        };
        Ok(ResultWriter {
            output: output.to_path_buf(),
            json,
            lines,
            rows: Vec::new(),
        })
    }

    /// Records a row; a JSONL row reaches the file now.
    pub fn write(&mut self, row: Row) -> Result<(), ConfigError> {
        if let Some(lines) = &mut self.lines {
            writeln!(lines, "{}", row.to_json())
                .and_then(|()| lines.flush())
                .map_err(|error| ConfigError::Invalid(format!("{}: {error}", self.output.display())))?;
        }
        self.rows.push(row);
        Ok(())
    }

    /// The rows written so far.
    pub fn rows(&self) -> &[Row] {
        &self.rows
    }

    /// Finishes the output: writes the document of a `.json` output.
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, blocking I/O: the document is written once, on the main thread, after the last iteration"
    )]
    pub fn close(self) -> Result<(), ConfigError> {
        if !self.json {
            return Ok(());
        }
        let document = serde_json::json!({
            "schema": DOCUMENT_SCHEMA,
            "schemaVersion": SCHEMA_VERSION,
            "results": self.rows,
        });
        let text = serde_json::to_string_pretty(&document).map_err(|error| ConfigError::Invalid(error.to_string()))?;
        fs::write(&self.output, text + "\n")
            .map_err(|error| ConfigError::Invalid(format!("{}: {error}", self.output.display())))
    }
}

/// The median of `values` (the mean of the middle two for an even count), or `None` for none.
pub fn median(mut values: Vec<f64>) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    values.sort_by(f64::total_cmp);
    let middle = values.len() / 2;
    Some(if values.len() % 2 == 1 {
        values[middle]
    } else {
        (values[middle - 1] + values[middle]) / 2.0
    })
}

/// A short summary: medians over the measured successful rows of each scenario, in absolute figures. Warmup rows are
/// left out; ineligible and failed rows are counted.
pub fn summary(rows: &[Row]) -> String {
    let mut groups: Vec<(String, Vec<&Row>)> = Vec::new();
    for row in rows.iter().filter(|row| !row.warmup) {
        let artifact = row.engine["configuration"]["artifact"].as_str().unwrap_or("-");
        let key = format!("{} | {} | {}", row.implementation, artifact, row.scenario.name);
        match groups.iter_mut().find(|(name, _)| *name == key) {
            Some((_, group)) => group.push(row),
            None => groups.push((key, vec![row])),
        }
    }
    const LABEL: &str = "implementation | artifact | scenario";
    let width = groups
        .iter()
        .map(|(key, _)| key.len())
        .max()
        .unwrap_or(0)
        .max(LABEL.len());
    let mut out = format!(
        "{LABEL:<width$} {:>4} {:>4} {:>4} {:>12} {:>10} {:>12} {:>10}\n",
        "ok", "inel", "fail", "prefill t/s", "ttft ms", "decode t/s", "e2e ms"
    );
    for (key, group) in &groups {
        let ok: Vec<&&Row> = group.iter().filter(|row| row.status == SUCCESS).collect();
        let count = |status: &str| group.iter().filter(|row| row.status == status).count();
        let show = |value: Option<f64>| value.map_or("-".to_string(), |value| format!("{value:.1}"));
        let of = |metric: &dyn Fn(&Row) -> Option<f64>| show(median(ok.iter().filter_map(|row| metric(row)).collect()));
        out.push_str(&format!(
            "{key:<width$} {:>4} {:>4} {:>4} {:>12} {:>10} {:>12} {:>10}\n",
            ok.len(),
            count(INELIGIBLE),
            count(FAILED),
            of(&|row| row.throughput.and_then(|t| t.prefill_tokens_per_second)),
            of(&|row| row.timings.and_then(|t| t.time_to_first_token).map(|n| n as f64 * 1e-6)),
            of(&|row| row.throughput.and_then(|t| t.decode_tokens_per_second)),
            of(&|row| row.timings.and_then(|t| t.end_to_end).map(|n| n as f64 * 1e-6)),
        ));
    }
    out
}
