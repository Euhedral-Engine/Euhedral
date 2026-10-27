//! Long-horizon drift of a candidate numeric route against the all-GPU route (`RelaxedNumericsDriftCudaIntegrationTest`):
//! the same forced tokens are decoded twice, and each step compares the second run's final hidden state and logits row
//! with the first's.
//!
//! The first run is kept in memory ([`RowStore`]); the second compares each row with the first's in a frame of the
//! `host` lake ([`crate::rows`]). Per step: the relative error of the final hidden state (before the final norm), the
//! KL divergence of the first run's distribution from the second's, whether the top-1 token agrees, the first run's
//! logit lead of its top-1 over the second's when they differ, and whether both rows are bitwise equal.

use std::sync::{Arc, OnceLock};

use euhedral_inference_engine::EngineRuntime;
use euhedral_inference_engine::generation::FinishReason;

use crate::config::ConfigError;
use crate::rows::{self, RowInput, RowTask};

/// Steps of a window of the hidden-error peak.
pub const WINDOW: usize = 32;
/// Steps before the hidden error is judged for growth: it rises from nothing to its level within about 150 positions.
pub const BURN_IN: usize = 128;
/// A top-1 flip whose lead is at or below this is a near-tie.
pub const NEAR_TIE: f64 = 0.25;
/// Mean KL at or above which a prefix fails.
pub const MAX_MEAN_KL: f64 = 1.0e-2;
/// Share of steps that may flip top-1 with a lead above [`NEAR_TIE`].
pub const MAX_DECIDED_FLIPS: f64 = 0.02;
/// Median peak window hidden error at or above which the candidate fails.
pub const MAX_PEAK: f64 = 0.12;
/// Median late/early hidden error above which the candidate fails.
pub const MAX_GROWTH: f64 = 1.5;

/// The candidate route the second run takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Candidate {
    /// The engine as configured (host-backed layers computed by the CPU): bounded error, not equality.
    HostTier,
    /// The first run's configuration again: the runs must agree bitwise.
    Exact,
}

/// The values of one step, in the order a [`Compare`] stores them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StepMetrics {
    pub hidden_relative: f64,
    pub kl: f64,
    pub top_agrees: bool,
    pub flip_lead: f64,
    pub bitwise: bool,
}

const METRICS: usize = 5;

impl StepMetrics {
    fn from_values(values: &[f64]) -> StepMetrics {
        StepMetrics {
            hidden_relative: values[0],
            kl: values[1],
            top_agrees: values[2] != 0.0,
            flip_lead: values[3],
            bitwise: values[4] != 0.0,
        }
    }
}

fn bf16(bytes: &[u8], index: usize) -> f64 {
    f64::from(f32::from_bits(
        u32::from(u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]])) << 16,
    ))
}

/// The first maximum of a BF16 row: its index and value.
fn top(row: &[u8]) -> (usize, f64) {
    let (mut best, mut at) = (f64::NEG_INFINITY, 0);
    for index in 0..row.len() / 2 {
        let value = bf16(row, index);
        if value > best {
            (best, at) = (value, index);
        }
    }
    (at, best)
}

/// `log Z` of a BF16 row of maximum `max`.
fn log_z(row: &[u8], max: f64) -> f64 {
    max + (0..row.len() / 2)
        .map(|index| (bf16(row, index) - max).exp())
        .sum::<f64>()
        .ln()
}

/// One step's comparison of the first run's rows (`a`, the oracle) with the second's (`b`).
pub fn compare_step(a_logits: &[u8], b_logits: &[u8], a_hidden: &[u8], b_hidden: &[u8]) -> StepMetrics {
    let (mut norm, mut difference) = (0.0, 0.0);
    for index in 0..a_hidden.len() / 2 {
        let (x, y) = (bf16(a_hidden, index), bf16(b_hidden, index));
        norm += x * x;
        difference += (x - y) * (x - y);
    }
    let ((top_a, max_a), (top_b, max_b)) = (top(a_logits), top(b_logits));
    let (log_z_a, log_z_b) = (log_z(a_logits, max_a), log_z(b_logits, max_b));
    let mut kl = 0.0;
    for index in 0..a_logits.len() / 2 {
        let log_p = bf16(a_logits, index) - log_z_a;
        kl += log_p.exp() * (log_p - (bf16(b_logits, index) - log_z_b));
    }
    StepMetrics {
        hidden_relative: (difference / norm).sqrt(),
        kl: kl.max(0.0),
        top_agrees: top_a == top_b,
        flip_lead: max_a - bf16(a_logits, top_b),
        bitwise: a_logits == b_logits && a_hidden == b_hidden,
    }
}

/// The rows of the first run, kept for the second to compare with.
#[derive(Debug)]
pub struct RowStore {
    rows: Vec<OnceLock<(Vec<u8>, Vec<u8>)>>,
}

impl RowStore {
    /// A store for a run of `steps` steps.
    pub fn new(steps: usize) -> RowStore {
        RowStore {
            rows: (0..steps).map(|_| OnceLock::new()).collect(),
        }
    }

    /// Whether every step's row arrived.
    pub fn is_complete(&self) -> bool {
        self.rows.iter().all(|row| row.get().is_some())
    }
}

impl RowTask for RowStore {
    fn fields(&self) -> usize {
        0
    }

    fn wants_hidden(&self) -> bool {
        true
    }

    fn compute(&self, input: &RowInput, _out: &mut [f64]) {
        let _ = self.rows[input.step].set((input.logits.clone(), input.hidden.clone()));
    }
}

/// The second run's task: compares each row with the store's.
struct Compare {
    store: Arc<RowStore>,
}

impl RowTask for Compare {
    fn fields(&self) -> usize {
        METRICS
    }

    fn wants_hidden(&self) -> bool {
        true
    }

    fn compute(&self, input: &RowInput, out: &mut [f64]) {
        let Some((logits, hidden)) = self.store.rows[input.step].get() else {
            return;
        };
        let metrics = compare_step(logits, &input.logits, hidden, &input.hidden);
        out.copy_from_slice(&[
            metrics.hidden_relative,
            metrics.kl,
            f64::from(u8::from(metrics.top_agrees)),
            metrics.flip_lead,
            f64::from(u8::from(metrics.bitwise)),
        ]);
    }
}

fn run(
    runtime: &Arc<EngineRuntime>,
    prefix: Vec<u32>,
    forced: Vec<u32>,
    task: Arc<dyn RowTask>,
) -> Result<Vec<f64>, ConfigError> {
    let steps = forced.len();
    let (engine, values) = rows::start(runtime, prefix, forced, task)?;
    let outcome = engine.wait();
    let values = values.wait();
    if outcome.reason != FinishReason::Length || outcome.generated != steps {
        return Err(ConfigError::Invalid(format!(
            "the run ended with {:?} after {} of {steps} steps",
            outcome.reason, outcome.generated
        )));
    }
    values.map_err(|why| ConfigError::Invalid(format!("the run failed: {why}")))
}

/// Runs the first sequence on `runtime`, keeping its rows in `store` (which holds `forced.len()` steps). Waits for the
/// run: called by the binary's main thread only.
pub fn collect(
    runtime: &Arc<EngineRuntime>,
    prefix: Vec<u32>,
    forced: Vec<u32>,
    store: &Arc<RowStore>,
) -> Result<(), ConfigError> {
    run(runtime, prefix, forced, Arc::clone(store) as Arc<dyn RowTask>)?;
    if store.is_complete() {
        Ok(())
    } else {
        Err(ConfigError::Invalid(
            "the first run delivered fewer rows than it has steps".to_string(),
        ))
    }
}

/// Runs the second sequence on `runtime` and compares each step with the first's rows in `store`; the metrics of steps
/// `1..` (step 0 is the prefill's row, which both sequences compute alike). Waits for the run: called by the binary's
/// main thread only.
pub fn compare(
    runtime: &Arc<EngineRuntime>,
    prefix: Vec<u32>,
    forced: Vec<u32>,
    store: &Arc<RowStore>,
) -> Result<Vec<StepMetrics>, ConfigError> {
    let task = Arc::new(Compare {
        store: Arc::clone(store),
    });
    let values = run(runtime, prefix, forced, task)?;
    Ok(values.chunks(METRICS).skip(1).map(StepMetrics::from_values).collect())
}

/// What one starting position came to.
#[derive(Clone, Debug, PartialEq)]
pub struct PrefixSummary {
    pub steps: usize,
    pub mean_kl: f64,
    pub top_agreements: usize,
    /// Steps whose top-1 flipped with a lead above [`NEAR_TIE`].
    pub decided_flips: usize,
    /// The largest mean hidden error of a window of [`WINDOW`] steps.
    pub peak: f64,
    /// Late over early mean hidden error after the burn-in.
    pub growth: f64,
    pub bitwise: bool,
}

fn average(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0usize), |(sum, count), value| (sum + value, count + 1));
    sum / count as f64
}

/// The summary of `metrics`, the steps of one starting position.
pub fn summarize(metrics: &[StepMetrics]) -> PrefixSummary {
    let peak = metrics
        .chunks(WINDOW)
        .map(|window| average(window.iter().map(|step| step.hidden_relative)))
        .fold(0.0, f64::max);
    let burn_in = BURN_IN.min(metrics.len() / 2);
    let settled = &metrics[burn_in..];
    let half = settled.len() / 2;
    let early = average(settled[..half].iter().map(|step| step.hidden_relative));
    let late = average(settled[half..].iter().map(|step| step.hidden_relative));
    PrefixSummary {
        steps: metrics.len(),
        mean_kl: average(metrics.iter().map(|step| step.kl)),
        top_agreements: metrics.iter().filter(|step| step.top_agrees).count(),
        decided_flips: metrics
            .iter()
            .filter(|step| !step.top_agrees && step.flip_lead > NEAR_TIE)
            .count(),
        peak,
        growth: late / (early + 1.0e-4),
        bitwise: metrics.iter().all(|step| step.bitwise),
    }
}

fn median(values: &[f64]) -> f64 {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let n = sorted.len();
    if n % 2 == 1 {
        sorted[n / 2]
    } else {
        (sorted[n / 2 - 1] + sorted[n / 2]) / 2.0
    }
}

/// The thresholds `summaries` (one per starting position) breach, as sentences; none means the candidate passes. The
/// hidden-state bounds apply to the median over the starting positions: a wrong kernel raises every position. `Exact`
/// must also agree bitwise at every step.
pub fn breaches(summaries: &[PrefixSummary], candidate: Candidate) -> Vec<String> {
    let mut breaches = Vec::new();
    for (index, summary) in summaries.iter().enumerate() {
        if summary.mean_kl >= MAX_MEAN_KL {
            breaches.push(format!(
                "position {index}: mean KL {:.3e} is at or above {MAX_MEAN_KL:.0e}",
                summary.mean_kl
            ));
        }
        if summary.decided_flips as f64 > MAX_DECIDED_FLIPS * summary.steps as f64 {
            breaches.push(format!(
                "position {index}: {} of {} steps flip top-1 with a lead above {NEAR_TIE}, more than {:.0}%",
                summary.decided_flips,
                summary.steps,
                100.0 * MAX_DECIDED_FLIPS
            ));
        }
        if candidate == Candidate::Exact && !summary.bitwise {
            breaches.push(format!(
                "position {index}: the exact candidate differs from the first run"
            ));
        }
    }
    let peak = median(&summaries.iter().map(|summary| summary.peak).collect::<Vec<_>>());
    let growth = median(&summaries.iter().map(|summary| summary.growth).collect::<Vec<_>>());
    if peak >= MAX_PEAK {
        breaches.push(format!(
            "median peak window hidden error {peak:.4} is at or above {MAX_PEAK}"
        ));
    }
    if growth > MAX_GROWTH {
        breaches.push(format!("median hidden error growth {growth:.3} is above {MAX_GROWTH}"));
    }
    breaches
}

/// The report of a drift run: one line per starting position, then the medians.
pub fn report(summaries: &[PrefixSummary], prefixes: &[usize], candidate: Candidate) -> String {
    let mut text = format!("candidate {candidate:?}\n");
    for (summary, prefix) in summaries.iter().zip(prefixes) {
        text += &format!(
            "prefix {prefix}: peak window hidden error {:.4}, late/early {:.3}, mean KL {:.2e}, top-1 {}/{}, decided flips {}, bitwise {}\n",
            summary.peak,
            summary.growth,
            summary.mean_kl,
            summary.top_agreements,
            summary.steps,
            summary.decided_flips,
            summary.bitwise
        );
    }
    text += &format!(
        "median peak {:.4}, median growth {:.3}\n",
        median(&summaries.iter().map(|summary| summary.peak).collect::<Vec<_>>()),
        median(&summaries.iter().map(|summary| summary.growth).collect::<Vec<_>>())
    );
    text
}
