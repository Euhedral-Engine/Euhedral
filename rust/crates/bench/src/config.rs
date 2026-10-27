//! The `run` configuration: the Java benchmark's JSON file (`benchmark/configs/*.json`), read with the same field names,
//! defaults and checks. Unknown fields are rejected, as Java rejects them, except `tokenizer`, which older files carry and
//! the Rust benchmark ignores (it reads the tokenizer from the artifact).
//!
//! A file that asks for something this engine does not do is refused with the reason, never run with the request
//! dropped, because the two engines' rows would then measure different work:
//! - `prefixCacheBytes` above 0: the engine always prefills the whole prompt;
//! - `cpus` other than `all`, or any `excludeCpus` / `excludeCores`: the engine uses every processor and never restricts
//!   its workers;
//! - `generation.mode` `sample`: the engine selects greedily.

use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;
use serde_json::Value;
use thiserror::Error;

use crate::prompt::PromptCorpus;
use crate::prompt::words::DEFAULT_SEED;
use crate::result::civil;

/// The longest sequence the engine keeps device memory for when the file does not say.
pub const DEFAULT_MAX_CONTEXT_TOKENS: u32 = 32768;
/// The tokens of each sequence's attention cache kept on the GPU when the file does not say.
pub const DEFAULT_KV_DEVICE_MINIMUM_TOKENS: u32 = 65536;

/// Why a configuration or a command line was refused.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum ConfigError {
    #[error("{0}")]
    Invalid(String),
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
}

impl ScenarioKind {
    /// The kind's name in scenario specifications and result rows.
    pub fn label(self) -> &'static str {
        match self {
            ScenarioKind::Prefill => "prefill",
            ScenarioKind::FirstToken => "first-token",
            ScenarioKind::PromptToN => "prompt-to-n",
            ScenarioKind::SustainedDecode => "decode",
        }
    }
}

/// One workload: a prompt of `target_prompt_tokens` tokens followed by at most `requested_new_tokens` generated tokens
/// in a fresh sequence. Specification syntax: `prefill:P`, `first-token:P`, `prompt-to-n:P:N`, `decode:P:N`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Scenario {
    pub kind: ScenarioKind,
    pub target_prompt_tokens: usize,
    pub requested_new_tokens: usize,
}

impl Scenario {
    /// The Java benchmark's default suite.
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

    /// A scenario, checked as Java's constructor checks it.
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

    /// The name rows carry: `prefill-32`, `decode-32-256`.
    pub fn name(&self) -> String {
        match self.kind {
            ScenarioKind::Prefill | ScenarioKind::FirstToken => {
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
            ScenarioKind::Prefill | ScenarioKind::FirstToken => {
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

    /// Parses `prefill:P`, `first-token:P`, `prompt-to-n:P:N` or `decode:P:N`.
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
        ];
        let Some(kind) = kinds.into_iter().find(|kind| kind.label() == parts[0]) else {
            return invalid(format!("unknown scenario kind: {text}"));
        };
        match kind {
            ScenarioKind::Prefill | ScenarioKind::FirstToken => {
                if parts.len() != 2 {
                    return invalid(format!("invalid scenario: {text}"));
                }
                let new = usize::from(kind == ScenarioKind::FirstToken);
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
    pub kv_device_minimum_tokens: Option<u32>,
    pub scenarios: Vec<Scenario>,
    pub warmup: usize,
    pub iterations: usize,
    pub generation_seed: i64,
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
                "prefixCacheBytes must be 0: the Rust engine always prefills the whole prompt, so a prefix cache would \
                 make the engines run different work",
            );
        }
        let cpus = raw.cpus.as_deref().map_or("all", str::trim);
        if cpus.is_empty() {
            return invalid("cpus must not be blank");
        }
        if cpus != "all" {
            return invalid(format!(
                "cpus must be \"all\" (got \"{cpus}\"): the Rust engine uses every processor and never restricts its workers"
            ));
        }
        for (name, ids) in [("excludeCpus", &raw.exclude_cpus), ("excludeCores", &raw.exclude_cores)] {
            if ids.iter().flatten().any(|id| *id < 0) {
                return invalid(format!("{name} must not be negative"));
            }
            if ids.as_ref().is_some_and(|ids| !ids.is_empty()) {
                return invalid(format!(
                    "{name} must be empty: the Rust engine uses every processor and never restricts its workers"
                ));
            }
        }
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
        let warmup = non_negative(raw.warmup, 1, "warmup")?;
        let iterations = positive(raw.iterations, 3, "iterations")? as usize;
        let mut generation_seed = 1;
        if let Some(generation) = &raw.generation {
            let mode = generation.mode.as_deref().unwrap_or("greedy");
            match mode {
                "greedy" => {
                    if generation.temperature.is_some() || generation.top_k.is_some() || generation.top_p.is_some() {
                        return invalid("temperature, topK, and topP require mode \"sample\"");
                    }
                }
                "sample" => {
                    return invalid("generation.mode \"sample\" is not supported: the Rust engine selects greedily");
                }
                _ => return invalid("generation.mode must be greedy or sample"),
            }
            generation_seed = generation.seed.unwrap_or(1);
        }
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
        let kv_device_minimum_tokens = match raw.kv_device_minimum_tokens {
            None => None,
            Some(value) if value > 0 && value <= i64::from(u32::MAX) => Some(value as u32),
            Some(_) => return invalid("kvDeviceMinimumTokens must be positive"),
        };
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
            generation_seed,
            prompt_seed: raw.prompt_seed.unwrap_or(DEFAULT_SEED as i64),
            output,
            overwrite,
            append,
            gpu_memory: raw.gpu_memory.unwrap_or(false),
            shutdown_timeout_seconds: u64::from(shutdown_timeout_seconds),
            prompt_corpus,
            speculation,
            engine_config: raw.engine_config,
        })
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
