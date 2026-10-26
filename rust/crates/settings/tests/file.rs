//! Reading the YAML file: values, `auto`, and every kind of mistake in it.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: writes fixture files in tests"
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Duration;

use euhedral_inference_settings::{
    Auto, ByteSize, CpuList, HostLayerCompute, KvFormat, ModelSettings, Settings, SettingsErrors, load_file,
};

const MODEL: &str = "model:\n  artifact: /m/a.edrl\n  name: q\n";

fn model() -> ModelSettings {
    ModelSettings {
        artifact: PathBuf::from("/m/a.edrl"),
        name: "q".to_owned(),
    }
}

fn no_env(_: &str) -> Option<String> {
    None
}

fn load(yaml: &str) -> Result<Settings, SettingsErrors> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    fs::write(&path, yaml).unwrap();
    load_file(&path, &no_env)
}

fn problems(yaml: &str) -> Vec<String> {
    load(yaml)
        .expect_err("the file is invalid")
        .0
        .iter()
        .map(|problem| problem.message().to_owned())
        .collect()
}

#[test]
fn minimal_file_gives_defaults() {
    assert_eq!(load(MODEL), Ok(Settings::with_model(model())));
}

#[test]
fn missing_required_keys_are_named() {
    let message = problems("").join("|");
    assert!(
        message.contains("model.artifact is required (set it in the file or as EUHEDRAL_MODEL_ARTIFACT)"),
        "{message}"
    );
    assert!(message.contains("model.name"), "{message}");
    assert!(!message.contains("model.tokenizer"), "{message}");
}

#[test]
fn unknown_key_is_error() {
    assert_eq!(
        problems(&format!("{MODEL}memory:\n  gpu-limt: 1GiB\n")),
        ["unknown setting memory.gpu-limt"]
    );
    assert_eq!(problems(&format!("{MODEL}surprise: 1\n")), ["unknown setting surprise"]);
}

#[test]
fn retired_key_says_why() {
    let yaml = MODEL.replace("  name: q\n", "  tokenizer: /m/tok\n  name: q\n");
    assert_eq!(
        problems(&yaml),
        ["model.tokenizer is retired: removed: the artifact carries its tokenizer"]
    );
}

#[test]
fn duplicate_key_is_error() {
    let message = problems(&format!("{MODEL}server:\n  port: 1\n  port: 2\n")).join("|");
    assert!(message.contains("duplicate entry with key \"port\""), "{message}");
}

#[test]
fn value_in_place_of_a_section_and_back() {
    assert_eq!(
        problems(&format!("{MODEL}server: 1\n")),
        ["server is a section, not a value"]
    );
    assert_eq!(
        problems(&format!("{MODEL}server:\n  port: [1]\n")),
        ["server.port must be a single value"]
    );
    assert_eq!(
        problems(&format!("{MODEL}batching:\n  max-sequences:\n")),
        ["batching.max-sequences has no value"]
    );
}

#[test]
fn file_must_be_a_mapping_and_must_exist() {
    assert!(problems("- a\n- b\n")[0].contains("must hold a mapping"));
    assert!(problems("a: [")[0].contains("is not valid YAML"));
    let missing = load_file(Path::new("/nope/euhedral.yaml"), &no_env).unwrap_err();
    assert_eq!(
        missing.0[0].message(),
        "configuration file not found: /nope/euhedral.yaml"
    );
}

#[test]
fn values_parse() {
    let settings = load(&format!(
        "{MODEL}context:\n  kv-cache-format: bf16\nmemory:\n  gpu-limit: 1GiB\n  host-limit: 0\nhost:\n  cpus: 2-5,8\n  idle-release: 0\n  max-batch-size: 2\n\
         server:\n  requests:\n    timeout: 90s\n    max-body-size: 2MiB\n"
    ))
    .unwrap();
    assert_eq!(settings.context.kv_cache_format, Auto::Value(KvFormat::Bf16));
    assert_eq!(settings.memory.gpu_limit, Auto::Value(ByteSize::gib(1)));
    assert_eq!(settings.memory.host_limit, Auto::Value(ByteSize(0)));
    assert_eq!(
        settings.host.cpus.value().unwrap().ids().collect::<Vec<_>>(),
        [2, 3, 4, 5, 8]
    );
    assert_eq!(settings.host.idle_release, Duration::ZERO);
    assert_eq!(settings.host.max_batch_size, Auto::Value(2));
    assert_eq!(settings.server.requests.timeout, Duration::from_secs(90));
    assert_eq!(settings.server.requests.max_body_size, ByteSize::mib(2));
}

#[test]
fn sizes_and_durations_parse() {
    let settings = load(&format!(
        "{MODEL}memory:\n  gpu-reserve: 700MiB\nserver:\n  shutdown-timeout: 1s\n  requests:\n    timeout: 30m\n"
    ))
    .unwrap();
    assert_eq!(settings.memory.gpu_reserve, ByteSize(734_003_200));
    assert_eq!(settings.server.shutdown_timeout, Duration::from_secs(1));
    assert_eq!(settings.server.requests.timeout, Duration::from_secs(1800));
}

#[test]
fn sampling_is_checked() {
    assert_eq!(
        problems(&format!("{MODEL}sampling:\n  top-p: 1.5\n")),
        ["sampling.top-p: must be above 0 and at most 1: 1.5"]
    );
    assert_eq!(
        problems(&format!("{MODEL}sampling:\n  top-k: -2\n")),
        ["sampling.top-k: must be 0 to 2147483647: -2"]
    );
    assert!(problems(&format!("{MODEL}sampling:\n  top-p: NaN\n"))[0].contains("sampling.top-p"));
    let settings = load(&format!(
        "{MODEL}sampling:\n  temperature: 0.7\n  top-k: 0\n  top-p: 1\n"
    ))
    .unwrap();
    assert_eq!(
        (
            settings.sampling.temperature,
            settings.sampling.top_k,
            settings.sampling.top_p
        ),
        (0.7, 0, 1.0)
    );
}

#[test]
fn auto_only_on_optional_keys() {
    assert_eq!(
        problems(&format!("{MODEL}server:\n  port: auto\n")),
        ["server.port: auto is not accepted, give a value"]
    );
    assert!(problems("model:\n  artifact: auto\n  name: q\n")[0].contains("model.artifact"));
    let settings = load(&format!(
        "{MODEL}batching:\n  max-sequences-per-pass: auto\nmemory:\n  gpu-limit: AUTO\n  host-layers:\n    compute: auto\n"
    ))
    .unwrap();
    assert!(settings.batching.max_sequences_per_pass.is_auto());
    assert!(settings.memory.gpu_limit.is_auto());
}

#[test]
fn host_layers_keys() {
    let settings = load(&format!(
        "{MODEL}memory:\n  host-layers:\n    compute: stream\n    gemv-rows: 64\n    prefill-slots: 3\n"
    ))
    .unwrap();
    assert_eq!(
        settings.memory.host_layers.compute,
        Auto::Value(HostLayerCompute::Stream)
    );
    assert_eq!(settings.memory.host_layers.gemv_rows, Auto::Value(64));
    assert_eq!(settings.memory.host_layers.prefill_slots, Auto::Value(3));
    let cpu = load(&format!("{MODEL}memory:\n  host-layers:\n    compute: CPU\n")).unwrap();
    assert_eq!(cpu.memory.host_layers.compute, Auto::Value(HostLayerCompute::Cpu));

    assert_eq!(
        problems(&format!("{MODEL}memory:\n  host-layers:\n    compute: gpu\n")),
        ["memory.host-layers.compute: must be auto, cpu or stream: gpu"]
    );
    assert_eq!(
        problems(&format!(
            "{MODEL}memory:\n  host-layers:\n    gemv-rows: 0\n    prefill-slots: -1\n"
        )),
        [
            "memory.host-layers.gemv-rows: must be 1 to 2147483647: 0",
            "memory.host-layers.prefill-slots: must be 1 to 2147483647: -1"
        ]
    );
}

#[test]
fn limits_and_formats() {
    assert!(problems(&format!("{MODEL}memory:\n  gpu-limit: 0\n"))[0].contains("memory.gpu-limit"));
    assert!(problems(&format!("{MODEL}server:\n  shutdown-timeout: 0s\n"))[0].contains("server.shutdown-timeout"));
    assert!(problems(&format!("{MODEL}server:\n  shutdown-timeout: 1d\n"))[0].contains("server.shutdown-timeout"));
    assert!(problems(&format!("{MODEL}prefix-cache:\n  checkpoint-tokens: 1000\n"))[0].contains("multiple of 512"));
    assert!(problems(&format!("{MODEL}kernels:\n  trellis:\n    tensor-core-max-tokens: 7\n"))[0].contains("8 to"));
    assert!(problems(&format!("{MODEL}context:\n  kv-cache-format: fp8\n"))[0].contains("nvfp4, bf16 or auto"));
    assert_eq!(
        load(&format!("{MODEL}server:\n  requests:\n    max-queued: 0\n"))
            .unwrap()
            .server
            .requests
            .max_queued,
        0
    );
    assert_eq!(
        load(&format!("{MODEL}prefix-cache:\n  size: 0\n"))
            .unwrap()
            .prefix_cache
            .size,
        ByteSize(0)
    );
}

#[test]
fn cpu_lists() {
    let list = CpuList::parse("2-5,8").unwrap();
    assert_eq!(list.ranges(), [2..=5, 8..=8]);
    for text in ["5-2", "", "1,,2", "a", "1-", "-1", "1 - 2", "2-5-6"] {
        assert!(CpuList::parse(text).is_err(), "{text:?}");
    }
    assert!(problems(&format!("{MODEL}host:\n  cpus: 5-2\n"))[0].contains("host.cpus"));
}

#[test]
fn all_errors_reported_together() {
    let found = problems(&format!(
        "{MODEL}server:\n  port: abc\nbatching:\n  max-sequences: -1\nmemory:\n  nope: 1\n"
    ));
    assert_eq!(found.len(), 3, "{found:?}");
    assert_eq!(found[0], "unknown setting memory.nope");
    assert!(found[1].starts_with("server.port: "), "{found:?}");
    assert!(found[2].starts_with("batching.max-sequences: "), "{found:?}");
}
