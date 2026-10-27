//! Teacher-forced scoring against a llama.cpp reference (`docs/evaluation/QUALITY.md`): the Java scorer's scheme on this
//! engine.
//!
//! Each chunk of N reference tokens is forced on a fresh sequence: a prefill of N / 2 + 1 tokens, then the forced
//! tokens one row at a time, so that step k's logits row predicts the token the reference's record k does. The rows
//! the engine decodes one at a time are the rows its verification of up to 8 tokens produces (row-exact), so the scheme
//! needs no row count. Every row is scored on the CPU, in a frame of the `host` lake, against the reference record a read
//! has fetched ([`crate::rows`]): five values per step, in the order the `.values` file has them.

use std::sync::Arc;
use std::time::{Instant, SystemTime};

use euhedral_graph::OwnedFile;
use euhedral_inference_engine::EngineRuntime;
use euhedral_inference_engine::generation::FinishReason;

use crate::config::ConfigError;
use crate::kld::{FLOOR, Reference, Row};
use crate::result::iso_instant;
use crate::rows::{self, ReferenceRows, RowInput, RowTask, RunValues};

/// Values per step: the candidate's NLL of the forced token, the reference's, KL(reference || candidate), top-1
/// agreement, and the difference in probability of the forced token.
pub const FIELDS: usize = 5;

/// A BF16 logit.
fn logit(row: &[u8], token: usize) -> f64 {
    f64::from(f32::from_bits(
        u32::from(u16::from_le_bytes([row[2 * token], row[2 * token + 1]])) << 16,
    ))
}

/// The five values of one step: `logits` are the candidate's BF16 logits of the step, `reference` the reference's
/// record, `target` the forced token. KL runs over the reference's entries above the floor; top-1 takes the first
/// maximum on both sides.
pub fn score_row(logits: &[u8], reference: &Row<'_>, vocabulary: usize, target: usize) -> [f64; FIELDS] {
    let (mut q_max, mut q_arg) = (f64::MIN, usize::MAX);
    let (mut p_max, mut p_arg) = (f64::MIN, usize::MAX);
    for token in 0..vocabulary {
        let q = logit(logits, token);
        if q > q_max {
            (q_max, q_arg) = (q, token);
        }
        let p = reference.log_p(token);
        if p > p_max {
            (p_max, p_arg) = (p, token);
        }
    }
    let mut sum = 0.0;
    for token in 0..vocabulary {
        sum += (logit(logits, token) - q_max).exp();
    }
    let log_z = q_max + sum.ln();
    let mut kl = 0.0;
    for token in 0..vocabulary {
        let p = reference.log_p(token);
        if p > FLOOR {
            kl += p.exp() * (p - (logit(logits, token) - log_z));
        }
    }
    let q_target = logit(logits, target) - log_z;
    let p_target = reference.log_p(target);
    [
        -q_target,
        -p_target,
        kl,
        f64::from(u8::from(q_arg == p_arg)),
        q_target.exp() - p_target.exp(),
    ]
}

/// The scoring of one chunk as a task of [`rows`].
struct ChunkTask {
    vocabulary: usize,
    rows: ReferenceRows,
}

impl RowTask for ChunkTask {
    fn fields(&self) -> usize {
        FIELDS
    }

    fn reference(&self) -> Option<&ReferenceRows> {
        Some(&self.rows)
    }

    fn compute(&self, input: &RowInput, out: &mut [f64]) {
        let row = Row::new(&input.reference, self.vocabulary);
        out.copy_from_slice(&score_row(&input.logits, &row, self.vocabulary, input.forced as usize));
    }
}

/// One reference scored: the values of every chunk, `records * FIELDS` each.
#[derive(Debug)]
pub struct Scored {
    pub values: Vec<Vec<f64>>,
    pub seconds: f64,
}

fn finished(values: RunValues) -> Result<Vec<f64>, ConfigError> {
    values.map_err(|why| ConfigError::Invalid(format!("scoring failed: {why}")))
}

/// Scores every chunk of `reference` (whose records `file` holds) on `runtime`: a chunk is forced while the previous
/// one's last rows are still being scored. Waits for the runs: called by the binary's main thread only.
pub fn score(
    runtime: &Arc<EngineRuntime>,
    reference: &Reference,
    file: &Arc<OwnedFile>,
) -> Result<Scored, ConfigError> {
    let vocabulary = runtime.config().vocab_size;
    if vocabulary != reference.vocabulary {
        return Err(ConfigError::Invalid(format!(
            "the reference has a vocabulary of {}, the artifact {vocabulary}",
            reference.vocabulary
        )));
    }
    let started = Instant::now();
    let records = reference.records();
    let mut values = Vec::with_capacity(reference.chunks());
    let mut previous = None;
    for chunk in 0..reference.chunks() {
        let task = Arc::new(ChunkTask {
            vocabulary,
            rows: ReferenceRows {
                file: Arc::clone(file),
                first_offset: reference.row_offset(chunk, 0),
                row_bytes: reference.row_bytes(),
            },
        });
        let (engine, scored) = rows::start(
            runtime,
            reference.prefix(chunk).to_vec(),
            reference.forced(chunk).to_vec(),
            task,
        )?;
        let outcome = engine.wait();
        if outcome.reason != FinishReason::Length || outcome.generated != records {
            let _ = scored.wait();
            return Err(ConfigError::Invalid(format!(
                "chunk {chunk} ended with {:?} after {} of {records} steps",
                outcome.reason, outcome.generated
            )));
        }
        // The engine is free: the next chunk starts while this one's last rows are scored. The chunk before it is
        // waited for first, so that no more than two chunks of rows are ever held.
        if let Some(before) = previous.replace(scored) {
            values.push(finished(before.wait())?);
        }
    }
    if let Some(last) = previous {
        values.push(finished(last.wait())?);
    }
    Ok(Scored {
        values,
        seconds: started.elapsed().as_secs_f64(),
    })
}

/// The `.values` file of a run: three little-endian i32 (chunks, scored tokens per chunk, values per token), then the
/// values as little-endian float64, chunk after chunk.
pub fn values_bytes(values: &[Vec<f64>], steps: usize) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(12 + 8 * values.len() * steps * FIELDS);
    for count in [values.len(), steps, FIELDS] {
        bytes.extend_from_slice(&(count as i32).to_le_bytes());
    }
    for chunk in values {
        for value in chunk {
            bytes.extend_from_slice(&value.to_le_bytes());
        }
    }
    bytes
}

/// The mean of `values[from..from + count]`.
pub fn mean(values: &[f64], from: usize, count: usize) -> f64 {
    values[from..from + count].iter().sum::<f64>() / count as f64
}

/// Chunk-level over several chunks (tokens within a document are correlated), token-level over one.
pub fn standard_error(per_chunk: &[f64], per_token: &[f64]) -> f64 {
    let values = if per_chunk.len() > 1 { per_chunk } else { per_token };
    let average = mean(values, 0, values.len());
    let sum: f64 = values.iter().map(|value| (value - average) * (value - average)).sum();
    (sum / (values.len() as f64 - 1.0)).sqrt() / (values.len() as f64).sqrt()
}

/// NumPy's default (linear) percentile of sorted values.
pub fn percentile(sorted: &[f64], q: f64) -> f64 {
    let position = q / 100.0 * (sorted.len() - 1) as f64;
    let below = position.floor() as usize;
    let above = (sorted.len() - 1).min(below + 1);
    sorted[below] + (position - below as f64) * (sorted[above] - sorted[below])
}

/// The summary of one reference, in the Java scorer's text: a comment line, the chunk line, the column titles, the
/// reference's own mean NLL and perplexity, and the artifact's row (mean NLL, perplexity, dNLL and its standard error, KL
/// and its standard error, top-1 agreement, then the KL median, p95, p99 and maximum, the top-1 standard error, and the
/// mean, RMS, p1 and p99 of the probability difference).
pub fn summary_lines(
    name: &str,
    reference_path: &str,
    artifact_path: &str,
    reference: &Reference,
    values: &[Vec<f64>],
    when: SystemTime,
) -> String {
    let chunks = values.len();
    let steps = reference.records();
    let n = chunks * steps;
    let field = |index: usize| -> Vec<f64> {
        values
            .iter()
            .flat_map(|chunk| (0..steps).map(move |step| chunk[FIELDS * step + index]))
            .collect()
    };
    let (nll, reference_nll, kl, agree, dp) = (field(0), field(1), field(2), field(3), field(4));
    let difference: Vec<f64> = nll.iter().zip(&reference_nll).map(|(a, b)| a - b).collect();
    let per_chunk = |data: &[f64]| -> Vec<f64> { (0..chunks).map(|c| mean(data, c * steps, steps)).collect() };
    let chunk_difference: Vec<f64> = (0..chunks)
        .map(|c| mean(&nll, c * steps, steps) - mean(&reference_nll, c * steps, steps))
        .collect();
    let (chunk_kl, chunk_agree) = (per_chunk(&kl), per_chunk(&agree));
    let mut sorted_kl = kl.clone();
    let mut sorted_dp = dp.clone();
    sorted_kl.sort_by(f64::total_cmp);
    sorted_dp.sort_by(f64::total_cmp);
    let rms = (dp.iter().map(|value| value * value).sum::<f64>() / n as f64).sqrt();
    let mean_nll = mean(&nll, 0, n);
    let reference_mean = mean(&reference_nll, 0, n);
    let mut text = format!(
        "# {name} {} artifact {artifact_path} reference {reference_path}\n",
        iso_instant(when)
    );
    text += &format!(
        "{chunks} chunk(s) of {}, {steps} forced tokens each (positions {}..{}); reference {reference_path}\n",
        reference.context,
        reference.context / 2 + 1,
        reference.context - 1
    );
    text += &format!(
        "{:<50} {:>9} {:>8} {:>9} {:>8} {:>11} {:>9} {:>7} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}\n",
        "report",
        "mean NLL",
        "ppl",
        "dNLL",
        "+-SE",
        "KL(ref||c)",
        "+-SE",
        "top-1",
        "KL med",
        "KL p95",
        "KL p99",
        "KL max",
        "top1 SE",
        "dp mean",
        "dp RMS",
        "dp p1",
        "dp p99"
    );
    text += &format!(
        "{:<50} {:>9.5} {:>8.4}\n",
        reference_path,
        reference_mean,
        reference_mean.exp()
    );
    text += &format!(
        "{:<50} {:>9.5} {:>8.4} {:>+9.5} {:>8.5} {:>11.6} {:>9.6} {:>7.4} {:>8.5} {:>8.5} {:>8.5} {:>8.4} {:>8.5} {:>+8.5} {:>8.5} {:>+8.5} {:>+8.5}\n",
        name,
        mean_nll,
        mean_nll.exp(),
        mean(&difference, 0, n),
        standard_error(&chunk_difference, &difference),
        mean(&kl, 0, n),
        standard_error(&chunk_kl, &kl),
        mean(&agree, 0, n),
        percentile(&sorted_kl, 50.0),
        percentile(&sorted_kl, 95.0),
        percentile(&sorted_kl, 99.0),
        sorted_kl[n - 1],
        standard_error(&chunk_agree, &agree),
        mean(&dp, 0, n),
        rms,
        percentile(&sorted_dp, 1.0),
        percentile(&sorted_dp, 99.0),
    );
    text
}

/// The pooled aggregates of a run: mean KL, top-1 agreement and mean dNLL, for reports that print them in one line.
pub fn aggregates(values: &[Vec<f64>], steps: usize) -> (f64, f64, f64) {
    let n = values.len() * steps;
    let (mut kl, mut agree, mut difference) = (0.0, 0.0, 0.0);
    for chunk in values {
        for step in 0..steps {
            kl += chunk[FIELDS * step + 2];
            agree += chunk[FIELDS * step + 3];
            difference += chunk[FIELDS * step] - chunk[FIELDS * step + 1];
        }
    }
    (kl / n as f64, agree / n as f64, difference / n as f64)
}
