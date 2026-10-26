//! The environment: overrides, the `PORT` alias, retired variables, and finding the configuration file.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: writes fixture files in tests"
)]

use std::collections::HashMap;
use std::ffi::OsString;
use std::fs;
use std::path::PathBuf;

use euhedral_inference_settings::{
    Auto, ByteSize, ModelSettings, Settings, SettingsErrors, load, load_file, locate, variable_name,
};

const MODEL: &str = "model:\n  artifact: /m/a.edrl\n  name: q\n";

fn model() -> ModelSettings {
    ModelSettings {
        artifact: PathBuf::from("/m/a.edrl"),
        name: "q".to_owned(),
    }
}

/// An environment of the given variables.
fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<String> + use<> {
    let vars: HashMap<String, String> = pairs.iter().map(|(k, v)| ((*k).to_owned(), (*v).to_owned())).collect();
    move |name| vars.get(name).cloned()
}

/// Loads `MODEL` from a file with the variables set.
fn load_with(yaml: &str, pairs: &[(&str, &str)]) -> Result<Settings, SettingsErrors> {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    fs::write(&path, yaml).unwrap();
    load_file(&path, &env(pairs))
}

fn problems(pairs: &[(&str, &str)]) -> Vec<String> {
    load_with(MODEL, pairs)
        .expect_err("invalid")
        .0
        .iter()
        .map(|p| p.message().to_owned())
        .collect()
}

#[test]
fn variable_names() {
    assert_eq!(variable_name("model.artifact"), "EUHEDRAL_MODEL_ARTIFACT");
    assert_eq!(
        variable_name("kernels.trellis.target-blocks"),
        "EUHEDRAL_KERNELS_TRELLIS_TARGET_BLOCKS"
    );
    assert_eq!(
        variable_name("memory.host-layers.gemv-rows"),
        "EUHEDRAL_MEMORY_HOST_LAYERS_GEMV_ROWS"
    );
}

#[test]
fn env_override_dash_keys() {
    let settings = load_with(MODEL, &[("EUHEDRAL_MEMORY_GPU_RESERVE", "1GiB")]).unwrap();
    assert_eq!(settings.memory.gpu_reserve, ByteSize::gib(1));
    let settings = load_with(MODEL, &[("EUHEDRAL_MEMORY_HOST_LAYERS_GEMV_ROWS", "128")]).unwrap();
    assert_eq!(settings.memory.host_layers.gemv_rows, Auto::Value(128));
}

#[test]
fn variable_overrides_file_and_creates_missing_sections() {
    let settings = load_with(
        &format!("{MODEL}server:\n  port: 8000\n"),
        &[("EUHEDRAL_SERVER_PORT", "8001")],
    )
    .unwrap();
    assert_eq!(settings.server.port, 8001);
    let settings = load_with(MODEL, &[("EUHEDRAL_KERNELS_TRELLIS_TARGET_BLOCKS", "280")]).unwrap();
    assert_eq!(settings.kernels.trellis.target_blocks, Auto::Value(280));
}

#[test]
fn variable_auto_clears_file_value() {
    let settings = load_with(
        &format!("{MODEL}memory:\n  gpu-limit: 1GiB\n"),
        &[("EUHEDRAL_MEMORY_GPU_LIMIT", "auto")],
    )
    .unwrap();
    assert!(settings.memory.gpu_limit.is_auto());
}

#[test]
fn variables_supply_required_keys() {
    let vars = env(&[("EUHEDRAL_MODEL_ARTIFACT", "/m/a.edrl"), ("EUHEDRAL_MODEL_NAME", "q")]);
    let settings = load(&[], &vars).unwrap();
    assert_eq!(settings, Settings::with_model(model()));
}

#[test]
fn empty_env_is_error() {
    assert_eq!(
        problems(&[("EUHEDRAL_SERVER_PORT", "")]),
        ["EUHEDRAL_SERVER_PORT is empty"]
    );
    assert_eq!(problems(&[("PORT", "  ")]), ["PORT is empty"]);
}

#[test]
fn blank_required_variable_reports_only_empty() {
    let found = load_with("model:\n  name: q\n", &[("EUHEDRAL_MODEL_ARTIFACT", " ")]).unwrap_err();
    assert_eq!(found.0.len(), 1, "{found}");
    assert_eq!(found.0[0].message(), "EUHEDRAL_MODEL_ARTIFACT is empty");
    assert_eq!(found.0[0].key(), Some("model.artifact"));
    assert_eq!(found.0[0].variable(), Some("EUHEDRAL_MODEL_ARTIFACT"));
}

#[test]
fn retired_env_is_error() {
    let found = problems(&[("EUHEDRAL_CUDA_GRAPHS", "0")]);
    assert_eq!(
        found,
        ["EUHEDRAL_CUDA_GRAPHS is retired: removed: the setting is fixed"]
    );
    let found = problems(&[("EUHEDRAL_INFERENCE_ARTIFACT_PATH", "/x")]);
    assert_eq!(
        found,
        ["EUHEDRAL_INFERENCE_ARTIFACT_PATH is retired: renamed to model.artifact (EUHEDRAL_MODEL_ARTIFACT)"]
    );
    let found = problems(&[("EUHEDRAL_HOST_MEMORY_MIB", "")]);
    assert!(found[0].contains("give a size with a unit"), "{found:?}");
}

#[test]
fn model_tokenizer_variable_is_retired() {
    let vars = env(&[
        ("EUHEDRAL_MODEL_ARTIFACT", "/m/a.edrl"),
        ("EUHEDRAL_MODEL_TOKENIZER", "/m/tok"),
        ("EUHEDRAL_MODEL_NAME", "q"),
    ]);
    let found = load(&[], &vars).unwrap_err();
    assert_eq!(found.0.len(), 1);
    assert_eq!(
        found.0[0].message(),
        "EUHEDRAL_MODEL_TOKENIZER is retired: removed: the artifact carries its tokenizer"
    );
}

#[test]
fn other_variables_are_ignored() {
    let settings = load_with(
        MODEL,
        &[
            ("EUHEDRAL_QWEN4_TRACE", "1"),
            ("EUHEDRAL_DEPLOY_HOST", "h"),
            ("HOME", "/h"),
        ],
    )
    .unwrap();
    assert_eq!(settings, Settings::with_model(model()));
}

#[test]
fn port_env_sets_server_port() {
    assert_eq!(load_with(MODEL, &[("PORT", "9000")]).unwrap().server.port, 9000);
    let both = load_with(MODEL, &[("PORT", "9000"), ("EUHEDRAL_SERVER_PORT", "9001")]).unwrap();
    assert_eq!(both.server.port, 9001);
}

#[test]
fn errors_name_key_and_variable() {
    let found = load_with(MODEL, &[("EUHEDRAL_SERVER_PORT", "x")]).unwrap_err();
    assert_eq!(
        found.0[0].message(),
        "server.port (EUHEDRAL_SERVER_PORT): not a whole number: x"
    );
    assert_eq!(found.0[0].key(), Some("server.port"));
    assert_eq!(found.0[0].variable(), Some("EUHEDRAL_SERVER_PORT"));
    let found = load_with(MODEL, &[("PORT", "70000")]).unwrap_err();
    assert_eq!(found.0[0].message(), "server.port (PORT): must be 1 to 65535: 70000");
    assert_eq!(found.0[0].variable(), Some("PORT"));
}

#[test]
fn locate_forms() {
    let args = |list: &[&str]| list.iter().map(OsString::from).collect::<Vec<_>>();
    let none = env(&[]);
    assert_eq!(locate(&args(&["--config", "/a"]), &none), Ok(Some(PathBuf::from("/a"))));
    assert_eq!(
        locate(&args(&["x", "--config=/a"]), &none),
        Ok(Some(PathBuf::from("/a")))
    );
    assert_eq!(
        locate(&[], &env(&[("EUHEDRAL_CONFIG", "/b")])),
        Ok(Some(PathBuf::from("/b")))
    );
    assert_eq!(
        locate(&args(&["--config", "/a"]), &env(&[("EUHEDRAL_CONFIG", "/b")])),
        Ok(Some(PathBuf::from("/a")))
    );
    assert_eq!(locate(&[], &none), Ok(None));
    assert_eq!(locate(&[], &env(&[("EUHEDRAL_CONFIG", " ")])), Ok(None));
    for bad in [&["--config"][..], &["--config="][..], &["--config", " "][..]] {
        let found = locate(&args(bad), &none).unwrap_err();
        assert_eq!(found.0[0].message(), "--config needs a file path");
    }
}

#[test]
fn load_reads_the_located_file_then_the_environment() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("c.yaml");
    fs::write(&path, format!("{MODEL}server:\n  port: 8000\n")).unwrap();
    let args = vec![OsString::from("--config"), path.clone().into_os_string()];
    assert_eq!(load(&args, &env(&[])).unwrap().server.port, 8000);
    let path_text = path.to_str().unwrap();
    let vars = env(&[("EUHEDRAL_CONFIG", path_text), ("PORT", "8002")]);
    assert_eq!(load(&[], &vars).unwrap().server.port, 8002);
    let missing = load(&[OsString::from("--config=/nope/c.yaml")], &env(&[])).unwrap_err();
    assert_eq!(missing.0[0].message(), "configuration file not found: /nope/c.yaml");
}
