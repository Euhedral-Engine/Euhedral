//! The `run` configuration: the benchmark's JSON file (`rust/resources/bench/configs/*.json`): its field names,
//! defaults and checks. Unknown fields are rejected, except `tokenizer`, which older files carry and
//! the Rust benchmark ignores (it reads the tokenizer from the artifact).
//!
//! `generation` is `greedy` (the default; `temperature`, `topK` and `topP` are refused) or `sample` (temperature 0.7,
//! topK 20, topP 0.8 unless given); the seed is a signed 64-bit number whose bits seed the sampler.
//!
//! A file that asks for something this engine does not do is refused with the reason, never run with the request
//! dropped, because the two engines' rows would then measure different work:
//! - `prefixCacheBytes` above 0: the engine always prefills the whole prompt;
//!
//! `cpus` and `excludeCpus` / `excludeCores` select the processors the engine's workers may use; they are resolved
//! against the host's topology (see [`crate::cpus`]) and passed as `host.cpus`.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use euhedral_inference_engine::sampling::{SamplingConfig, SamplingError};
use euhedral_inference_quantization::rows::RowsError;
use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::cpus::{CpuSelection, CpuTopology, ResolvedCpus};
use crate::prompt::PromptCorpus;
use crate::prompt::words::DEFAULT_SEED;
use crate::result::civil;

/// The longest sequence the engine keeps device memory for when the file does not say.
pub const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 32768;
/// The tokens of each sequence's attention cache kept on the GPU when the file does not say.
pub const DEFAULT_KV_DEVICE_MINIMUM_TOKENS: u32 = 65536;

/// The tokens a prefix scenario's request has after its shared prompt.
pub const PREFIX_TAIL_TOKENS: usize = 16;

/// Why a configuration or a command line was refused.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ConfigError {
    #[error("{0}")]
    Invalid(String),
}

impl From<RowsError> for ConfigError {
    fn from(error: RowsError) -> Self {
        ConfigError::Invalid(error.to_string())
    }
}

fn invalid<T>(message: impl Into<String>) -> Result<T, ConfigError> {
    Err(ConfigError::Invalid(message.into()))
}

/// The kind of a scenario.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ScenarioKind {
    /// Prompt prefill only (`maxNewTokens = 0`).
    Prefill,
    /// Prefill plus the first token (`maxNewTokens = 1`).
    FirstToken,
    /// Prompt to the Nth selected token.
    PromptToN,
    /// Sustained decode after a short prompt; eligible only if all N tokens are produced.
    SustainedDecode,
    /// Prefill plus the first token of a prompt, asked for twice in a row: cold, then repeated, with the prefix cache
    /// on (`maxNewTokens = 1` each).
    PrefixRepeat,
}

impl ScenarioKind {
    /// The kind's name in scenario specifications and result rows.
    pub fn label(self) -> &'static str {
        match self {
            ScenarioKind::Prefill => "prefill",
            ScenarioKind::FirstToken => "first-token",
            ScenarioKind::PromptToN => "prompt-to-n",
            ScenarioKind::SustainedDecode => "decode",
            ScenarioKind::PrefixRepeat => "prefix",
        }
    }
}

/// One workload: a prompt of `target_prompt_tokens` tokens followed by at most `requested_new_tokens` generated tokens
/// in a fresh sequence. Specification syntax: `prefill:P`, `first-token:P`, `prompt-to-n:P:N`, `decode:P:N`,
/// `prefix:P`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Scenario {
    pub kind: ScenarioKind,
    pub target_prompt_tokens: usize,
    pub requested_new_tokens: usize,
}

impl Scenario {
    /// The default suite.
    pub fn default_suite() -> Vec<Scenario> {
        [
            (ScenarioKind::Prefill, 32, 0),
            (ScenarioKind::Prefill, 256, 0),
            (ScenarioKind::Prefill, 1024, 0),
            (ScenarioKind::Prefill, 4096, 0),
            (ScenarioKind::FirstToken, 1024, 1),
            (ScenarioKind::PromptToN, 1024, 64),
            (ScenarioKind::SustainedDecode, 32, 256),
        ]
        .into_iter()
        .map(|(kind, target_prompt_tokens, requested_new_tokens)| Scenario {
            kind,
            target_prompt_tokens,
            requested_new_tokens,
        })
        .collect()
    }

    /// A scenario, checked on construction.
    pub fn new(
        kind: ScenarioKind,
        target_prompt_tokens: usize,
        requested_new_tokens: usize,
    ) -> Result<Scenario, ConfigError> {
        if target_prompt_tokens == 0 {
            return invalid("scenario prompt tokens must be positive");
        }
        match kind {
            ScenarioKind::Prefill if requested_new_tokens != 0 => {
                return invalid("prefill scenarios generate no tokens");
            }
            ScenarioKind::FirstToken if requested_new_tokens != 1 => {
                return invalid("first-token scenarios generate one");
            }
            ScenarioKind::PrefixRepeat if requested_new_tokens != 1 => {
                return invalid("prefix scenarios generate one");
            }
            ScenarioKind::PromptToN | ScenarioKind::SustainedDecode if requested_new_tokens < 2 => {
                return invalid(format!("{} scenarios need at least two new tokens", kind.label()));
            }
            _ => {}
        }
        Ok(Scenario {
            kind,
            target_prompt_tokens,
            requested_new_tokens,
        })
    }

    /// The tokens a request carries after the scenario's `target_prompt_tokens`: a prefix scenario's prompt is the
    /// shared prompt and a short tail, so that the checkpoint at the end of the shared part is one a match can restore
    /// (a match always leaves a token to prefill). Zero for every other kind.
    pub fn tail_tokens(&self) -> usize {
        if self.kind == ScenarioKind::PrefixRepeat {
            PREFIX_TAIL_TOKENS
        } else {
            0
        }
    }

    /// Whether a prefix scenario's shared prompt ends on a checkpoint of a cache that stores one every `interval`
    /// tokens: a restore can only be at a checkpoint, so a repeat restores the whole shared prompt only then.
    pub fn check_checkpoint(&self, interval: usize) -> Result<(), ConfigError> {
        if self.kind == ScenarioKind::PrefixRepeat
            && interval != 0
            && !self.target_prompt_tokens.is_multiple_of(interval)
        {
            return invalid(format!(
                "{}: the prompt is {} tokens, which is not a multiple of the checkpoint interval of {interval} tokens \
                 (prefix-cache.checkpoint-tokens): no checkpoint ends it, so a repeat cannot restore it",
                self.spec(),
                self.target_prompt_tokens
            ));
        }
        Ok(())
    }

    /// The name rows carry: `prefill-32`, `decode-32-256`.
    pub fn name(&self) -> String {
        match self.kind {
            ScenarioKind::Prefill | ScenarioKind::FirstToken | ScenarioKind::PrefixRepeat => {
                format!("{}-{}", self.kind.label(), self.target_prompt_tokens)
            }
            ScenarioKind::PromptToN | ScenarioKind::SustainedDecode => {
                format!(
                    "{}-{}-{}",
                    self.kind.label(),
                    self.target_prompt_tokens,
                    self.requested_new_tokens
                )
            }
        }
    }

    /// The specification that parses back to this scenario.
    pub fn spec(&self) -> String {
        match self.kind {
            ScenarioKind::Prefill | ScenarioKind::FirstToken | ScenarioKind::PrefixRepeat => {
                format!("{}:{}", self.kind.label(), self.target_prompt_tokens)
            }
            ScenarioKind::PromptToN | ScenarioKind::SustainedDecode => {
                format!(
                    "{}:{}:{}",
                    self.kind.label(),
                    self.target_prompt_tokens,
                    self.requested_new_tokens
                )
            }
        }
    }

    /// Parses `prefill:P`, `first-token:P`, `prompt-to-n:P:N`, `decode:P:N` or `prefix:P`.
    pub fn parse(text: &str) -> Result<Scenario, ConfigError> {
        let lowered = text.trim().to_lowercase();
        let parts: Vec<&str> = lowered.split(':').collect();
        let number = |part: &str| {
            part.parse::<usize>()
                .or_else(|_| invalid(format!("invalid scenario token count: {text}")))
        };
        let kinds = [
            ScenarioKind::Prefill,
            ScenarioKind::FirstToken,
            ScenarioKind::PromptToN,
            ScenarioKind::SustainedDecode,
            ScenarioKind::PrefixRepeat,
        ];
        let Some(kind) = kinds.into_iter().find(|kind| kind.label() == parts[0]) else {
            return invalid(format!("unknown scenario kind: {text}"));
        };
        match kind {
            ScenarioKind::Prefill | ScenarioKind::FirstToken | ScenarioKind::PrefixRepeat => {
                if parts.len() != 2 {
                    return invalid(format!("invalid scenario: {text}"));
                }
                let new = usize::from(kind != ScenarioKind::Prefill);
                Scenario::new(kind, number(parts[1])?, new)
            }
            ScenarioKind::PromptToN | ScenarioKind::SustainedDecode => {
                if parts.len() != 3 {
                    return invalid(format!("invalid scenario: {text}"));
                }
                Scenario::new(kind, number(parts[1])?, number(parts[2])?)
            }
        }
    }
}

/// Whether the engine drafts tokens with the artifact's draft layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SpeculationMode {
    /// `artifact`: the strategy the artifact selects.
    Artifact,
    /// `off`: one token per quantum.
    Off,
}

impl SpeculationMode {
    pub fn label(self) -> &'static str {
        match self {
            SpeculationMode::Artifact => "artifact",
            SpeculationMode::Off => "off",
        }
    }
}

/// The validated configuration of a `run`.
#[derive(Clone, Debug, PartialEq)]
pub struct Options {
    pub artifact: PathBuf,
    pub cuda_library: PathBuf,
    pub max_context_tokens: u32,
    pub kv_device_minimum_tokens: u32,
    pub scenarios: Vec<Scenario>,
    pub warmup: usize,
    pub iterations: usize,
    pub sampling: SamplingConfig,
    pub prompt_seed: i64,
    pub output: PathBuf,
    pub overwrite: bool,
    pub append: bool,
    pub gpu_memory: bool,
    pub shutdown_timeout_seconds: u64,
    pub prompt_corpus: PromptCorpus,
    pub speculation: SpeculationMode,
    /// A settings file (`euhedral.yaml`) the engine's tuning is read from; the fields above choose the model and context.
    pub engine_config: Option<PathBuf>,
    /// The worker CPUs the run selects (`cpus`), before the exclusions; see [`Options::worker_cpus`].
    pub cpus: CpuSelection,
    pub exclude_cpus: Vec<u32>,
    pub exclude_cores: Vec<u32>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct Raw {
    artifact: PathBuf,
    cuda_library: PathBuf,
    /// Accepted and ignored: the Rust benchmark reads the tokenizer from the artifact.
    #[allow(dead_code, reason = "named so that the file is accepted; the value is not used")]
    tokenizer: Option<Value>,
    max_context_tokens: Option<i64>,
    prefix_cache_bytes: Option<i64>,
    cpus: Option<String>,
    exclude_cpus: Option<Vec<i64>>,
    exclude_cores: Option<Vec<i64>>,
    scenarios: Option<Vec<String>>,
    warmup: Option<i64>,
    iterations: Option<i64>,
    generation: Option<RawGeneration>,
    prompt_seed: Option<i64>,
    output: Option<PathBuf>,
    overwrite: Option<bool>,
    append: Option<bool>,
    gpu_memory: Option<bool>,
    shutdown_timeout_seconds: Option<i64>,
    prompt_corpus: Option<String>,
    speculation: Option<String>,
    kv_device_minimum_tokens: Option<i64>,
    engine_config: Option<PathBuf>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields, rename_all = "camelCase")]
struct RawGeneration {
    mode: Option<String>,
    seed: Option<i64>,
    temperature: Option<f64>,
    top_k: Option<i64>,
    top_p: Option<f64>,
}

impl RawGeneration {
    /// The sampling the section asks for, checked on construction.
    fn sampling(&self) -> Result<SamplingConfig, ConfigError> {
        let seed = self.seed.unwrap_or(1) as u64;
        let refused = |error: SamplingError| ConfigError::Invalid(error.to_string());
        match self.mode.as_deref().unwrap_or("greedy") {
            "greedy" => {
                if self.temperature.is_some() || self.top_k.is_some() || self.top_p.is_some() {
                    return invalid("temperature, topK, and topP require mode \"sample\"");
                }
                Ok(SamplingConfig::greedy(seed))
            }
            "sample" => {
                let top_k = self.top_k.unwrap_or(20);
                let top_k = match u32::try_from(top_k) {
                    Ok(top_k) if top_k <= i32::MAX as u32 => top_k,
                    _ if top_k < 0 => return invalid("topK must not be negative"),
                    _ => return invalid("topK is too large"),
                };
                let temperature = self.temperature.unwrap_or(0.7) as f32;
                let top_p = self.top_p.unwrap_or(0.8) as f32;
                SamplingConfig::new(temperature, top_k, top_p, seed, false).map_err(refused)
            }
            _ => invalid("generation.mode must be greedy or sample"),
        }
    }
}

impl Options {
    /// Reads and validates the configuration file at `path`. `now` names the default output file.
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, blocking I/O: the configuration is read once, on the main thread, before the engine starts"
    )]
    pub fn load(path: &Path, now: SystemTime) -> Result<Options, ConfigError> {
        let text = fs::read_to_string(path).or_else(|error| invalid(format!("{}: {error}", path.display())))?;
        Options::parse(&text, now)
            .map_err(|ConfigError::Invalid(message)| ConfigError::Invalid(format!("{}: {message}", path.display())))
    }

    /// Validates the configuration `text`.
    pub fn parse(text: &str, now: SystemTime) -> Result<Options, ConfigError> {
        let raw: Raw = serde_json::from_str(text).or_else(|error| invalid(error.to_string()))?;

        let max_context_tokens = positive(raw.max_context_tokens, DEFAULT_MAX_CONTEXT_TOKENS, "maxContextTokens")?;
        let prefix_cache_bytes = raw.prefix_cache_bytes.unwrap_or(0);
        if prefix_cache_bytes < 0 {
            return invalid("prefixCacheBytes must not be negative");
        }
        if prefix_cache_bytes > 0 {
            return invalid(
                "prefixCacheBytes must be 0: the prefix cache is off unless the run has prefix scenarios, which turn it \
                 on at the size of the engine settings",
            );
        }
        let cpus_text = raw.cpus.as_deref().map_or("all", str::trim);
        if cpus_text.is_empty() {
            return invalid("cpus must not be blank");
        }
        let cpus = CpuSelection::parse(cpus_text)?;
        let ids = |name: &str, ids: &Option<Vec<i64>>| -> Result<Vec<u32>, ConfigError> {
            ids.iter()
                .flatten()
                .map(|id| u32::try_from(*id).or_else(|_| invalid(format!("{name} must not be negative"))))
                .collect()
        };
        let exclude_cpus = ids("excludeCpus", &raw.exclude_cpus)?;
        let exclude_cores = ids("excludeCores", &raw.exclude_cores)?;
        let scenarios = match &raw.scenarios {
            None => Scenario::default_suite(),
            Some(specs) => specs
                .iter()
                .map(|spec| Scenario::parse(spec))
                .collect::<Result<_, _>>()?,
        };
        if scenarios.is_empty() {
            return invalid("no scenarios selected");
        }
        let prefix = scenarios
            .iter()
            .any(|scenario| scenario.kind == ScenarioKind::PrefixRepeat);
        if prefix
            && scenarios
                .iter()
                .any(|scenario| scenario.kind != ScenarioKind::PrefixRepeat)
        {
            return invalid(
                "a prefix scenario runs with the prefix cache on, which changes the work of any other scenario: run \
                 it alone",
            );
        }
        let warmup = non_negative(raw.warmup, 1, "warmup")?;
        let iterations = positive(raw.iterations, 3, "iterations")? as usize;
        let sampling = match &raw.generation {
            Some(generation) => generation.sampling()?,
            None => SamplingConfig::greedy(1),
        };
        let prompt_corpus = match raw.prompt_corpus.as_deref().map(str::trim).unwrap_or("words") {
            "words" => PromptCorpus::Words,
            "chat" => PromptCorpus::Chat,
            _ => return invalid("promptCorpus must be \"words\" or \"chat\""),
        };
        let speculation = match raw.speculation.as_deref().map(str::trim).unwrap_or("artifact") {
            "artifact" => SpeculationMode::Artifact,
            "off" => SpeculationMode::Off,
            _ => return invalid("speculation must be \"artifact\" or \"off\""),
        };
        let kv_device_minimum_tokens = positive(
            raw.kv_device_minimum_tokens,
            DEFAULT_KV_DEVICE_MINIMUM_TOKENS,
            "kvDeviceMinimumTokens",
        )?;
        let shutdown_timeout_seconds = positive(raw.shutdown_timeout_seconds, 10, "shutdownTimeoutSeconds")?;
        let overwrite = raw.overwrite.unwrap_or(false);
        let append = raw.append.unwrap_or(false);
        if overwrite && append {
            return invalid("choose either overwrite or append");
        }
        let output = raw.output.unwrap_or_else(|| default_output(now));
        if append && output.to_string_lossy().ends_with(".json") {
            return invalid("append requires JSONL output");
        }
        Ok(Options {
            artifact: raw.artifact,
            cuda_library: raw.cuda_library,
            max_context_tokens,
            kv_device_minimum_tokens,
            scenarios,
            warmup,
            iterations,
            sampling,
            prompt_seed: raw.prompt_seed.unwrap_or(DEFAULT_SEED as i64),
            output,
            overwrite,
            append,
            gpu_memory: raw.gpu_memory.unwrap_or(false),
            shutdown_timeout_seconds: u64::from(shutdown_timeout_seconds),
            prompt_corpus,
            speculation,
            engine_config: raw.engine_config,
            cpus,
            exclude_cpus,
            exclude_cores,
        })
    }

    /// The worker CPUs the run asks for on `topology`: the selection less the exclusions (all the available CPUs for
    /// `all`). Rejects an empty result and a CPU the process may not use.
    pub fn worker_cpus(&self, topology: &CpuTopology) -> Result<ResolvedCpus, ConfigError> {
        topology.resolve(&self.cpus, &self.exclude_cpus, &self.exclude_cores)
    }

    /// Whether the run needs the engine's prefix cache: it has prefix scenarios (and then only those). The cache is
    /// on at the size of the engine settings; any other run keeps it off.
    pub fn uses_prefix_cache(&self) -> bool {
        self.scenarios
            .iter()
            .any(|scenario| scenario.kind == ScenarioKind::PrefixRepeat)
    }

    /// Whether the output is one JSON document (a `.json` path) and not JSONL.
    pub fn json_output(&self) -> bool {
        self.output.to_string_lossy().ends_with(".json")
    }
}

fn positive(value: Option<i64>, default: u32, name: &str) -> Result<u32, ConfigError> {
    match value {
        None => Ok(default),
        Some(value) if value > 0 && value <= i64::from(u32::MAX) => Ok(value as u32),
        Some(_) => invalid(format!("{name} must be positive")),
    }
}

fn non_negative(value: Option<i64>, default: usize, name: &str) -> Result<usize, ConfigError> {
    match value {
        None => Ok(default),
        Some(value) if value >= 0 => Ok(value as usize),
        Some(_) => invalid(format!("{name} must not be negative")),
    }
}

/// `benchmark-results/euhedral-rust-<UTC time>.jsonl`.
fn default_output(now: SystemTime) -> PathBuf {
    let seconds = now.duration_since(UNIX_EPOCH).map_or(0, |elapsed| elapsed.as_secs());
    let (year, month, day, hour, minute, second) = civil(seconds);
    PathBuf::from(format!(
        "benchmark-results/euhedral-rust-{year:04}{month:02}{day:02}T{hour:02}{minute:02}{second:02}Z.jsonl"
    ))
}
