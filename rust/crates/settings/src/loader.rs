//! Reading the file, applying the environment, and validating every key.
//!
//! The file is flattened to dotted keys, a variable named `EUHEDRAL_` plus the upper-cased key path replaces or supplies
//! each key, and every key's text is parsed by its entry in [`leaves`]. Every problem found is reported together.

use std::collections::{HashMap, HashSet};
use std::error::Error;
use std::fmt;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use serde_yaml_ng::{Mapping, Value};

use crate::model::{Auto, CpuList, HostLayerCompute, KvFormat, MIN_TENSOR_CORE_ROWS, ModelSettings, Settings};
use crate::retired;
use crate::sizes::{ByteSize, parse_duration};

const PREFIX: &str = "EUHEDRAL_";

/// The largest count a key accepts: the same bound the Java engine puts on its `int` settings, so one file serves both.
const COUNT_MAX: i64 = i32::MAX as i64;

/// One problem with the settings, naming the key and, when a variable supplied the value, the variable.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SettingsError {
    key: Option<String>,
    variable: Option<String>,
    message: String,
}

impl SettingsError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            key: None,
            variable: None,
            message: message.into(),
        }
    }

    fn for_key(key: &str, source: &HashMap<String, String>, detail: impl fmt::Display) -> Self {
        let variable = source.get(key).cloned();
        let named = variable
            .as_ref()
            .map_or_else(|| key.to_owned(), |variable| format!("{key} ({variable})"));
        Self {
            key: Some(key.to_owned()),
            variable,
            message: format!("{named}: {detail}"),
        }
    }

    /// The key at fault, such as `server.port`, when the problem belongs to one.
    pub fn key(&self) -> Option<&str> {
        self.key.as_deref()
    }

    /// The environment variable that supplied the bad value, when one did.
    pub fn variable(&self) -> Option<&str> {
        self.variable.as_deref()
    }

    /// The whole message, which names the key and the variable.
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for SettingsError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl Error for SettingsError {}

/// Every problem found in the settings, never empty.
///
/// A duplicate key in the file is reported by the YAML parser, which names the key's last segment (`port`, not
/// `server.port`) with its line and column.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SettingsErrors(pub Vec<SettingsError>);

impl fmt::Display for SettingsErrors {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid settings")?;
        for (index, problem) in self.0.iter().enumerate() {
            write!(f, "{} {problem}", if index == 0 { ":" } else { ";" })?;
        }
        Ok(())
    }
}

impl Error for SettingsErrors {}

impl From<SettingsError> for SettingsErrors {
    fn from(problem: SettingsError) -> Self {
        Self(vec![problem])
    }
}

/// How a leaf key may be left out or written `auto`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// Must be given.
    Required,
    /// Has a default; `auto` is not accepted.
    Fixed,
    /// Has a default of `auto`, which the engine resolves at run time.
    Optional,
}

type Apply = fn(&mut Settings, &str) -> Result<(), String>;

/// How one leaf key reads its text.
struct Leaf {
    key: &'static str,
    kind: Kind,
    apply: Apply,
}

/// The variable that overrides a key: `EUHEDRAL_` plus the key path upper-cased, with `.` and `-` written as `_`.
pub fn variable_name(key: &str) -> String {
    let path: String = key
        .chars()
        .map(|c| {
            if c == '.' || c == '-' {
                '_'
            } else {
                c.to_ascii_uppercase()
            }
        })
        .collect();
    format!("{PREFIX}{path}")
}

/// Every leaf key path, such as `memory.weight-streaming.copy-streams`, in the order of the example file.
pub fn known_keys() -> Vec<&'static str> {
    leaves().iter().map(|leaf| leaf.key).collect()
}

/// Loads the file (if any), applies the environment, validates, and builds the settings.
pub(crate) fn load_optional(
    file: Option<&Path>,
    env: &dyn Fn(&str) -> Option<String>,
) -> Result<Settings, SettingsErrors> {
    let leaves = leaves();
    let mut problems = Vec::new();
    let mut raw = HashMap::new();
    let mut source = HashMap::new();
    if let Some(file) = file {
        read_file(file, &leaves, &mut raw, &mut problems);
    }
    let mut blank = HashSet::new();
    apply_environment(env, &leaves, &mut raw, &mut source, &mut blank, &mut problems);

    let mut settings = Settings::with_model(ModelSettings::default());
    for leaf in &leaves {
        let Some(text) = raw.get(leaf.key) else {
            if leaf.kind == Kind::Required && !blank.contains(leaf.key) {
                problems.push(SettingsError {
                    key: Some(leaf.key.to_owned()),
                    variable: Some(variable_name(leaf.key)),
                    message: format!(
                        "{} is required (set it in the file or as {})",
                        leaf.key,
                        variable_name(leaf.key)
                    ),
                });
            }
            continue;
        };
        let text = text.trim();
        if text.eq_ignore_ascii_case("auto") && leaf.kind != Kind::Optional {
            problems.push(SettingsError::for_key(
                leaf.key,
                &source,
                "auto is not accepted, give a value",
            ));
            continue;
        }
        if let Err(detail) = (leaf.apply)(&mut settings, text) {
            problems.push(SettingsError::for_key(leaf.key, &source, detail));
        }
    }
    if problems.is_empty() {
        Ok(settings)
    } else {
        Err(SettingsErrors(problems))
    }
}

// The configuration file is read once at startup, before the lattice exists, and has no non-blocking form.
#[allow(clippy::disallowed_methods)]
fn read_file(file: &Path, leaves: &[Leaf], raw: &mut HashMap<String, String>, problems: &mut Vec<SettingsError>) {
    if !fs::metadata(file).is_ok_and(|meta| meta.is_file()) {
        problems.push(SettingsError::new(format!(
            "configuration file not found: {}",
            file.display()
        )));
        return;
    }
    let text = match fs::read_to_string(file) {
        Ok(text) => text,
        Err(error) => {
            problems.push(SettingsError::new(format!(
                "configuration file {} could not be read: {error}",
                file.display()
            )));
            return;
        }
    };
    // A duplicate key is a parse error of the mapping, so it is reported here with the key named.
    let root: Value = match serde_yaml_ng::from_str(&text) {
        Ok(root) => root,
        Err(error) => {
            problems.push(SettingsError::new(format!(
                "configuration file {} is not valid YAML: {error}",
                file.display()
            )));
            return;
        }
    };
    match root {
        Value::Null => {}
        Value::Mapping(mapping) => walk("", &mapping, leaves, raw, problems),
        _ => problems.push(SettingsError::new(format!(
            "configuration file {} must hold a mapping of settings",
            file.display()
        ))),
    }
}

fn walk(
    prefix: &str,
    mapping: &Mapping,
    leaves: &[Leaf],
    raw: &mut HashMap<String, String>,
    problems: &mut Vec<SettingsError>,
) {
    for (name, value) in mapping {
        let Some(name) = name.as_str() else {
            problems.push(SettingsError::new(format!(
                "a key under {prefix:?} is not text: {name:?}"
            )));
            continue;
        };
        let key = if prefix.is_empty() {
            name.to_owned()
        } else {
            format!("{prefix}.{name}")
        };
        let section_prefix = format!("{key}.");
        let is_leaf = leaves.iter().any(|leaf| leaf.key == key);
        let is_section = !is_leaf && leaves.iter().any(|leaf| leaf.key.starts_with(&section_prefix));
        let problem = |detail: &str| SettingsError {
            key: Some(key.clone()),
            variable: None,
            message: format!("{key} {detail}"),
        };
        if !is_leaf && !is_section {
            let message = match retired::key_message(&key) {
                Some(message) => format!("{key} is retired: {message}"),
                None => format!("unknown setting {key}"),
            };
            problems.push(SettingsError {
                key: Some(key),
                variable: None,
                message,
            });
        } else if value.is_null() {
            problems.push(problem("has no value"));
        } else if is_section {
            match value {
                Value::Mapping(inner) => walk(&key, inner, leaves, raw, problems),
                _ => problems.push(problem("is a section, not a value")),
            }
        } else {
            match value {
                Value::String(text) => drop(raw.insert(key, text.clone())),
                Value::Number(number) => drop(raw.insert(key, number.to_string())),
                Value::Bool(flag) => drop(raw.insert(key, flag.to_string())),
                _ => problems.push(problem("must be a single value")),
            }
        }
    }
}

fn apply_environment(
    env: &dyn Fn(&str) -> Option<String>,
    leaves: &[Leaf],
    raw: &mut HashMap<String, String>,
    source: &mut HashMap<String, String>,
    blank: &mut HashSet<&'static str>,
    problems: &mut Vec<SettingsError>,
) {
    for variable in retired::variables() {
        if env(variable).is_some() {
            let message = retired::message(variable).unwrap_or_default();
            problems.push(SettingsError {
                key: None,
                variable: Some(variable.to_owned()),
                message: format!("{variable} is retired: {message}"),
            });
        }
    }
    for leaf in leaves {
        let mut variable = variable_name(leaf.key);
        let mut value = env(&variable);
        if value.is_none() && leaf.key == "server.port" {
            value = env("PORT");
            variable = "PORT".to_owned();
        }
        let Some(value) = value else { continue };
        if value.trim().is_empty() {
            problems.push(SettingsError {
                key: Some(leaf.key.to_owned()),
                variable: Some(variable.clone()),
                message: format!("{variable} is empty"),
            });
            blank.insert(leaf.key);
            continue;
        }
        raw.insert(leaf.key.to_owned(), value);
        source.insert(leaf.key.to_owned(), variable);
    }
}

// Typed reads of a key's text.

fn int<T: TryFrom<i64>>(text: &str, min: i64, max: i64) -> Result<T, String> {
    let value: i64 = text.parse().map_err(|_| format!("not a whole number: {text}"))?;
    let range = || format!("must be {min} to {max}: {text}");
    if value < min || value > max {
        return Err(range());
    }
    T::try_from(value).map_err(|_| range())
}

fn decimal(text: &str) -> Result<f32, String> {
    let value: f32 = text.parse().map_err(|_| format!("not a number: {text}"))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err(format!("not a finite number: {text}"))
    }
}

fn temperature(text: &str) -> Result<f32, String> {
    let value = decimal(text)?;
    if value < 0.0 {
        Err(format!("must be 0 or more: {text}"))
    } else {
        Ok(value)
    }
}

fn top_p(text: &str) -> Result<f32, String> {
    let value = decimal(text)?;
    if value > 0.0 && value <= 1.0 {
        Ok(value)
    } else {
        Err(format!("must be above 0 and at most 1: {text}"))
    }
}

fn size(text: &str, min: u64) -> Result<ByteSize, String> {
    let value = ByteSize::parse(text)?;
    if value.bytes() < min {
        Err(format!("must be at least {min} bytes: {text}"))
    } else {
        Ok(value)
    }
}

fn duration(text: &str, positive: bool) -> Result<Duration, String> {
    let value = parse_duration(text)?;
    if positive && value.is_zero() {
        Err(format!("must be positive: {text}"))
    } else {
        Ok(value)
    }
}

fn name(text: &str) -> Result<String, String> {
    if text.is_empty() {
        Err("must not be blank".to_owned())
    } else {
        Ok(text.to_owned())
    }
}

fn kv_format(text: &str) -> Result<KvFormat, String> {
    match text.to_ascii_lowercase().as_str() {
        "nvfp4" => Ok(KvFormat::Nvfp4),
        "bf16" => Ok(KvFormat::Bf16),
        _ => Err(format!("must be nvfp4, bf16 or auto: {text}")),
    }
}

fn host_layer_compute(text: &str) -> Result<HostLayerCompute, String> {
    match text.to_ascii_lowercase().as_str() {
        "cpu" => Ok(HostLayerCompute::Cpu),
        "stream" => Ok(HostLayerCompute::Stream),
        _ => Err(format!("must be auto, cpu or stream: {text}")),
    }
}

fn checkpoint(text: &str) -> Result<u32, String> {
    let value: u32 = int(text, 1, COUNT_MAX)?;
    if value.is_multiple_of(512) {
        Ok(value)
    } else {
        Err(format!("must be a positive multiple of 512: {text}"))
    }
}

/// The value of an optional key: `auto`, or the text parsed by `parse`.
fn auto_or<T>(text: &str, parse: impl FnOnce(&str) -> Result<T, String>) -> Result<Auto<T>, String> {
    if text.eq_ignore_ascii_case("auto") {
        Ok(Auto::Auto)
    } else {
        parse(text).map(Auto::Value)
    }
}

/// Stores the parsed value of a key.
macro_rules! assign {
    ($place:expr, $value:expr) => {{
        $place = $value?;
        Ok(())
    }};
}

fn required(key: &'static str, apply: Apply) -> Leaf {
    Leaf {
        key,
        kind: Kind::Required,
        apply,
    }
}

fn fixed(key: &'static str, apply: Apply) -> Leaf {
    Leaf {
        key,
        kind: Kind::Fixed,
        apply,
    }
}

fn optional(key: &'static str, apply: Apply) -> Leaf {
    Leaf {
        key,
        kind: Kind::Optional,
        apply,
    }
}

/// Every leaf key, in the order of the example file.
fn leaves() -> Vec<Leaf> {
    vec![
        required("model.artifact", |s, t| {
            assign!(s.model.artifact, Ok::<PathBuf, String>(PathBuf::from(t)))
        }),
        required("model.name", |s, t| assign!(s.model.name, name(t))),
        fixed("sampling.temperature", |s, t| {
            assign!(s.sampling.temperature, temperature(t))
        }),
        fixed("sampling.top-k", |s, t| assign!(s.sampling.top_k, int(t, 0, COUNT_MAX))),
        fixed("sampling.top-p", |s, t| assign!(s.sampling.top_p, top_p(t))),
        fixed("server.port", |s, t| assign!(s.server.port, int(t, 1, 65535))),
        fixed("server.shutdown-timeout", |s, t| {
            assign!(s.server.shutdown_timeout, duration(t, true))
        }),
        fixed("server.requests.default-max-tokens", |s, t| {
            assign!(s.server.requests.default_max_tokens, int(t, 1, COUNT_MAX))
        }),
        fixed("server.requests.max-queued", |s, t| {
            assign!(s.server.requests.max_queued, int(t, 0, COUNT_MAX))
        }),
        fixed("server.requests.max-body-size", |s, t| {
            assign!(s.server.requests.max_body_size, size(t, 1))
        }),
        fixed("server.requests.timeout", |s, t| {
            assign!(s.server.requests.timeout, duration(t, true))
        }),
        fixed("context.max-tokens", |s, t| {
            assign!(s.context.max_tokens, int(t, 1, COUNT_MAX))
        }),
        optional("context.kv-cache-format", |s, t| {
            assign!(s.context.kv_cache_format, auto_or(t, kv_format))
        }),
        fixed("context.kv-cache-on-gpu-tokens", |s, t| {
            assign!(s.context.kv_cache_on_gpu_tokens, int(t, 0, COUNT_MAX))
        }),
        fixed("prefix-cache.size", |s, t| assign!(s.prefix_cache.size, size(t, 0))),
        fixed("prefix-cache.checkpoint-tokens", |s, t| {
            assign!(s.prefix_cache.checkpoint_tokens, checkpoint(t))
        }),
        fixed("batching.max-sequences", |s, t| {
            assign!(s.batching.max_sequences, int(t, 0, COUNT_MAX))
        }),
        optional("batching.max-sequences-per-pass", |s, t| {
            assign!(s.batching.max_sequences_per_pass, auto_or(t, |t| int(t, 1, COUNT_MAX)))
        }),
        optional("batching.max-tokens-per-pass", |s, t| {
            assign!(s.batching.max_tokens_per_pass, auto_or(t, |t| int(t, 1, COUNT_MAX)))
        }),
        fixed("batching.prefill-chunk-tokens", |s, t| {
            assign!(s.batching.prefill_chunk_tokens, int(t, 1, COUNT_MAX))
        }),
        optional("memory.host-limit", |s, t| {
            assign!(s.memory.host_limit, auto_or(t, |t| size(t, 0)))
        }),
        optional("memory.host-weights-limit", |s, t| {
            assign!(s.memory.host_weights_limit, auto_or(t, |t| size(t, 1)))
        }),
        optional("memory.gpu-limit", |s, t| {
            assign!(s.memory.gpu_limit, auto_or(t, |t| size(t, 1)))
        }),
        fixed("memory.gpu-reserve", |s, t| assign!(s.memory.gpu_reserve, size(t, 0))),
        optional("memory.pinned-buffer-pool", |s, t| {
            assign!(s.memory.pinned_buffer_pool, auto_or(t, |t| size(t, 0)))
        }),
        fixed("memory.weight-streaming.concurrent-reads", |s, t| {
            assign!(s.memory.weight_streaming.concurrent_reads, int(t, 1, COUNT_MAX))
        }),
        fixed("memory.weight-streaming.read-split", |s, t| {
            assign!(s.memory.weight_streaming.read_split, int(t, 1, COUNT_MAX))
        }),
        fixed("memory.weight-streaming.copy-streams", |s, t| {
            assign!(s.memory.weight_streaming.copy_streams, int(t, 1, COUNT_MAX))
        }),
        fixed("memory.weight-streaming.gpu-streams", |s, t| {
            assign!(s.memory.weight_streaming.gpu_streams, int(t, 1, COUNT_MAX))
        }),
        optional("memory.host-layers.compute", |s, t| {
            assign!(s.memory.host_layers.compute, auto_or(t, host_layer_compute))
        }),
        optional("memory.host-layers.gemv-rows", |s, t| {
            assign!(s.memory.host_layers.gemv_rows, auto_or(t, |t| int(t, 1, COUNT_MAX)))
        }),
        optional("memory.host-layers.prefill-slots", |s, t| {
            assign!(s.memory.host_layers.prefill_slots, auto_or(t, |t| int(t, 1, COUNT_MAX)))
        }),
        optional("host.cpus", |s, t| assign!(s.host.cpus, auto_or(t, CpuList::parse))),
        fixed("host.task-parallelism", |s, t| {
            assign!(s.host.task_parallelism, int(t, 1, COUNT_MAX))
        }),
        fixed("host.idle-release", |s, t| {
            assign!(s.host.idle_release, duration(t, false))
        }),
        optional("host.native-cache", |s, t| {
            assign!(s.host.native_cache, auto_or(t, |t| Ok(PathBuf::from(t))))
        }),
        fixed("kernels.trellis.tensor-core-max-tokens", |s, t| {
            assign!(
                s.kernels.trellis.tensor_core_max_tokens,
                int(t, i64::from(MIN_TENSOR_CORE_ROWS), COUNT_MAX)
            )
        }),
        optional("kernels.trellis.target-blocks", |s, t| {
            assign!(s.kernels.trellis.target_blocks, auto_or(t, |t| int(t, 1, COUNT_MAX)))
        }),
        fixed("kernels.trellis.whole-k-min-tokens", |s, t| {
            assign!(s.kernels.trellis.whole_k_min_tokens, int(t, 1, COUNT_MAX))
        }),
        fixed("kernels.trellis.wide-block-min-tokens", |s, t| {
            assign!(s.kernels.trellis.wide_block_min_tokens, int(t, 1, COUNT_MAX))
        }),
    ]
}
