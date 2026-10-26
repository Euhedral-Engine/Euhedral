//! Environment variables (and file keys) that once configured the server and no longer do. Setting one fails
//! startup, so a stale environment or configuration file is caught instead of silently ignored.

use crate::loader::variable_name;

/// Why a variable no longer configures anything.
enum Retirement {
    /// Moved to a settings key.
    Renamed(&'static str),
    /// Moved to a settings key that takes a size with a unit instead of a bare number of MiB.
    Resized(&'static str),
    /// The native files travel with the program.
    Bundled,
    /// The artifact carries its tokenizer.
    Tokenizer,
    /// The experiment's winner is built in.
    Settled,
}

use Retirement::{Bundled, Renamed, Resized, Settled, Tokenizer};

const RETIRED: &[(&str, Retirement)] = &[
    ("EUHEDRAL_INFERENCE_ARTIFACT_PATH", Renamed("model.artifact")),
    ("EUHEDRAL_INFERENCE_TOKENIZER_DIRECTORY", Tokenizer),
    ("EUHEDRAL_MODEL_TOKENIZER", Tokenizer),
    ("EUHEDRAL_INFERENCE_MODEL_ID", Renamed("model.name")),
    ("EUHEDRAL_INFERENCE_WORKER_CPUS", Renamed("host.cpus")),
    ("EUHEDRAL_INFERENCE_MAX_CONTEXT_TOKENS", Renamed("context.max-tokens")),
    (
        "EUHEDRAL_INFERENCE_SHUTDOWN_TIMEOUT",
        Renamed("server.shutdown-timeout"),
    ),
    ("EUHEDRAL_INFERENCE_PREFIX_CACHE_BYTES", Renamed("prefix-cache.size")),
    (
        "EUHEDRAL_INFERENCE_PREFIX_CACHE_CHECKPOINT_TOKENS",
        Renamed("prefix-cache.checkpoint-tokens"),
    ),
    ("EUHEDRAL_INFERENCE_KV_CACHE", Renamed("context.kv-cache-format")),
    ("EUHEDRAL_INFERENCE_MAX_SEQUENCES", Renamed("batching.max-sequences")),
    (
        "EUHEDRAL_INFERENCE_KV_DEVICE_MINIMUM_TOKENS",
        Renamed("context.kv-cache-on-gpu-tokens"),
    ),
    (
        "EUHEDRAL_API_DEFAULT_MAX_TOKENS",
        Renamed("server.requests.default-max-tokens"),
    ),
    (
        "EUHEDRAL_API_MAX_QUEUED_GENERATIONS",
        Renamed("server.requests.max-queued"),
    ),
    (
        "EUHEDRAL_API_MAX_REQUEST_BYTES",
        Renamed("server.requests.max-body-size"),
    ),
    ("EUHEDRAL_API_REQUEST_TIMEOUT", Renamed("server.requests.timeout")),
    ("EUHEDRAL_HOST_MEMORY_MIB", Resized("memory.host-limit")),
    ("EUHEDRAL_GPU_SYSTEM_RESERVE_MIB", Resized("memory.gpu-reserve")),
    ("EUHEDRAL_LAKE_SINKS", Renamed("host.task-parallelism")),
    (
        "EUHEDRAL_QWEN4_READS",
        Renamed("memory.weight-streaming.concurrent-reads"),
    ),
    (
        "EUHEDRAL_QWEN4_READ_PARTS",
        Renamed("memory.weight-streaming.read-split"),
    ),
    (
        "EUHEDRAL_QWEN4_COPY_STREAMS",
        Renamed("memory.weight-streaming.copy-streams"),
    ),
    ("EUHEDRAL_QWEN4_LANES", Renamed("memory.weight-streaming.gpu-streams")),
    ("EUHEDRAL_INFERENCE_CUDA_LIBRARY_PATH", Bundled),
    ("EUHEDRAL_CUDA_INCLUDE_DIR", Bundled),
    ("EUHEDRAL_LAKE_PARTITIONS", Settled),
    ("EUHEDRAL_CUDA_GRAPHS", Settled),
    ("EUHEDRAL_CUDA_KV_CODEC", Settled),
    ("EUHEDRAL_QWEN4_PREFETCH", Settled),
    ("EUHEDRAL_QWEN4_AHEAD", Settled),
    ("EUHEDRAL_QWEN4_LINEAR_TC", Settled),
    ("EUHEDRAL_QWEN4_LINEAR_SPLIT", Settled),
    ("EUHEDRAL_QWEN4_TIER_POLICY", Settled),
    ("EUHEDRAL_QWEN4_GPU_POLICY", Settled),
    ("EUHEDRAL_QWEN4_TIER_PRELOAD", Settled),
    ("EUHEDRAL_QWEN4_PIN_TIER", Settled),
    ("EUHEDRAL_QWEN4_TIER_ADMISSION", Settled),
    ("EUHEDRAL_QWEN4_GROUP", Settled),
    ("EUHEDRAL_QWEN4_DIRECT_READS", Settled),
];

fn renamed(key: &str) -> String {
    format!("renamed to {key} ({})", variable_name(key))
}

/// What to do about a retired variable, or `None` when the variable is not retired.
pub fn message(variable: &str) -> Option<String> {
    let (_, retirement) = RETIRED.iter().find(|(name, _)| *name == variable)?;
    Some(match retirement {
        Renamed(key) => renamed(key),
        Resized(key) => format!("{}; give a size with a unit such as 24GiB", renamed(key)),
        Bundled => "removed: the program carries its native files".to_owned(),
        Tokenizer => "removed: the artifact carries its tokenizer".to_owned(),
        Settled => "removed: the setting is fixed".to_owned(),
    })
}

/// What to do about a retired key of the file, such as `model.tokenizer`, or `None` when the key is not retired.
pub fn key_message(key: &str) -> Option<String> {
    message(&variable_name(key))
}

/// Every retired variable name.
pub fn variables() -> impl Iterator<Item = &'static str> {
    RETIRED.iter().map(|(name, _)| *name)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loader::known_keys;

    #[test]
    fn every_retired_message_is_removed_or_names_a_key() {
        let keys = known_keys();
        for variable in variables() {
            let message = message(variable).expect("listed variable has a message");
            let names_key = keys
                .iter()
                .any(|key| message.starts_with(&format!("renamed to {key} (")));
            assert!(message.starts_with("removed") || names_key, "{variable}: {message}");
        }
        assert!(message("EUHEDRAL_MODEL_ARTIFACT").is_none());
        assert!(message("EUHEDRAL_HOST_MEMORY_MIB").unwrap().contains("with a unit"));
    }

    #[test]
    fn settled_switches_are_removed() {
        for variable in [
            "EUHEDRAL_CUDA_GRAPHS",
            "EUHEDRAL_QWEN4_DIRECT_READS",
            "EUHEDRAL_LAKE_PARTITIONS",
        ] {
            assert_eq!(message(variable).as_deref(), Some("removed: the setting is fixed"));
        }
    }

    #[test]
    fn a_retired_key_reads_through_its_variable() {
        assert_eq!(
            key_message("model.tokenizer").as_deref(),
            Some("removed: the artifact carries its tokenizer")
        );
        assert_eq!(key_message("model.name"), None);
    }
}
