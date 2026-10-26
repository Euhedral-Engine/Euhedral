//! `config/euhedral.example.yaml` is the master example: it sets every key, and its values are the defaults.

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;

use euhedral_inference_settings::{
    Auto, HostSettings, KernelSettings, MemorySettings, ModelSettings, Settings, known_keys, load_file,
};

fn example() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../config/euhedral.example.yaml")
}

fn no_env(_: &str) -> Option<String> {
    None
}

/// The dotted key paths the file sets, read from its indentation (two spaces per level).
// Reading a fixture file in a test.
#[allow(clippy::disallowed_methods)]
fn keys_in_file() -> BTreeSet<String> {
    let text = fs::read_to_string(example()).expect("example file");
    let mut stack: Vec<String> = Vec::new();
    let mut keys = BTreeSet::new();
    for line in text.lines() {
        let content = line.trim_start();
        if content.is_empty() || content.starts_with('#') {
            continue;
        }
        let depth = (line.len() - content.len()) / 2;
        stack.truncate(depth);
        stack.push(content.split(':').next().unwrap().to_owned());
        keys.insert(stack.join("."));
    }
    keys
}

#[test]
fn example_file_parses_every_key() {
    let settings = load_file(&example(), &no_env).expect("the example loads");
    assert_eq!(settings.model.artifact, PathBuf::from("/models/qwen3_8_27b.edrl"));
    assert_eq!(settings.model.name, "qwen3.8-27b");
    assert_eq!(settings.memory.weight_streaming.copy_streams, 4);

    let in_file = keys_in_file();
    for key in known_keys() {
        assert!(in_file.contains(key), "the example lacks {key}");
    }
}

#[test]
fn example_equals_defaults() {
    let settings = load_file(&example(), &no_env).expect("the example loads");
    let expected = Settings {
        model: settings.model.clone(),
        memory: MemorySettings::default(),
        host: HostSettings::default(),
        kernels: KernelSettings::default(),
        ..Settings::with_model(ModelSettings::default())
    };
    assert_eq!(settings, expected);
    assert!(settings.memory.host_layers.compute.is_auto());
    assert_eq!(settings.memory.host_layers.gemv_rows, Auto::Auto);
    assert_eq!(settings.memory.host_layers.prefill_slots, Auto::Auto);
}
