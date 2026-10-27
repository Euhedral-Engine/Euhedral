//! The configuration: every Java benchmark configuration file parses, and what this engine cannot do is refused with
//! the reason.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: a test reads the repository's configuration files"
)]

use std::fs;
use std::path::PathBuf;
use std::time::{Duration, UNIX_EPOCH};

use euhedral_inference_bench::config::{ConfigError, Options, Scenario, ScenarioKind, SpeculationMode};
use euhedral_inference_bench::prompt::PromptCorpus;

fn configs_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../benchmark/configs")
}

fn parse(text: &str) -> Result<Options, String> {
    Options::parse(text, UNIX_EPOCH + Duration::from_secs(1_790_424_000))
        .map_err(|ConfigError::Invalid(message)| message)
}

fn with(extra: &str) -> String {
    format!(r#"{{"artifact": "/a.edrl", "cudaLibrary": "/lib.so"{extra}}}"#)
}

#[test]
fn parses_java_benchmark_configs() {
    let mut seen = Vec::new();
    for entry in fs::read_dir(configs_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let result = Options::load(&path, UNIX_EPOCH);
        if name == "custom-cpus.json" {
            // The one file that restricts the workers: refused, naming the setting.
            let ConfigError::Invalid(message) = result.expect_err("custom-cpus.json restricts the workers");
            assert!(message.contains("cpus must be \"all\""), "{message}");
            assert!(message.contains(&name), "the message names the file: {message}");
        } else {
            let options = result.unwrap_or_else(|error| panic!("{name}: {error}"));
            assert!(!options.scenarios.is_empty(), "{name}");
        }
        seen.push(name);
    }
    seen.sort();
    assert_eq!(
        seen,
        [
            "baseline.json",
            "context.json",
            "custom-cpus.json",
            "forks.json",
            "smoke.json"
        ],
        "the repository's configuration files"
    );

    // Values of two of them, as Java reads them.
    let smoke = Options::load(&configs_dir().join("smoke.json"), UNIX_EPOCH).unwrap();
    let names: Vec<String> = smoke.scenarios.iter().map(Scenario::name).collect();
    assert_eq!(names, ["prefill-32", "first-token-32", "decode-32-32"]);
    assert_eq!((smoke.warmup, smoke.iterations, smoke.gpu_memory), (1, 1, true));
    assert_eq!(smoke.output, PathBuf::from("benchmark-results/smoke.jsonl"));
    let context = Options::load(&configs_dir().join("context.json"), UNIX_EPOCH).unwrap();
    assert_eq!(context.prompt_corpus, PromptCorpus::Chat);
    assert_eq!(context.scenarios[2].spec(), "prompt-to-n:32000:128");
    let forks = Options::load(&configs_dir().join("forks.json"), UNIX_EPOCH).unwrap();
    assert!(forks.append);
}

#[test]
fn defaults_are_javas() {
    let options = parse(&with("")).unwrap();
    assert_eq!(options.max_context_tokens, 32768);
    assert_eq!((options.warmup, options.iterations), (1, 3));
    assert_eq!(options.prompt_corpus, PromptCorpus::Words);
    assert_eq!(options.speculation, SpeculationMode::Artifact);
    assert_eq!(options.prompt_seed, 20_260_925);
    assert_eq!(options.scenarios, Scenario::default_suite());
    assert_eq!(options.scenarios.len(), 7);
    assert!(!options.overwrite && !options.append && !options.gpu_memory);
    // The default output is timestamped, in UTC (1790424000 is 2026-09-26T12:00:00Z).
    assert_eq!(
        options.output,
        PathBuf::from("benchmark-results/euhedral-rust-20260926T120000Z.jsonl")
    );
}

#[test]
fn scenarios_parse_and_print_as_java() {
    for (spec, name) in [
        ("prefill:32", "prefill-32"),
        ("first-token:1024", "first-token-1024"),
        ("prompt-to-n:1024:64", "prompt-to-n-1024-64"),
        ("decode:32:256", "decode-32-256"),
    ] {
        let scenario = Scenario::parse(spec).unwrap();
        assert_eq!((scenario.spec().as_str(), scenario.name().as_str()), (spec, name));
    }
    assert_eq!(
        Scenario::parse(" DECODE:8:2 ").unwrap().kind,
        ScenarioKind::SustainedDecode
    );
    for bad in [
        "prefill",
        "prefill:0",
        "prefill:8:1",
        "first-token:8:2",
        "decode:8",
        "decode:8:1",
        "decode:x:4",
        "sleep:8",
    ] {
        assert!(Scenario::parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn what_the_engine_cannot_do_is_refused_with_the_reason() {
    let message = parse(&with(r#", "prefixCacheBytes": 1073741824"#)).unwrap_err();
    assert!(
        message.contains("prefixCacheBytes must be 0") && message.contains("prefills the whole prompt"),
        "{message}"
    );
    assert!(parse(&with(r#", "prefixCacheBytes": 0"#)).is_ok());

    for cpus in ["one-per-core", "performance", "performance-one-per-core", "2-5,8"] {
        let message = parse(&with(&format!(r#", "cpus": "{cpus}""#))).unwrap_err();
        assert!(
            message.contains("cpus must be \"all\"") && message.contains(cpus),
            "{message}"
        );
    }
    assert!(parse(&with(r#", "cpus": " all ""#)).is_ok());
    for key in ["excludeCpus", "excludeCores"] {
        let message = parse(&with(&format!(r#", "{key}": [0]"#))).unwrap_err();
        assert!(
            message.contains(key) && message.contains("never restricts"),
            "{message}"
        );
        assert!(parse(&with(&format!(r#", "{key}": []"#))).is_ok());
    }

    let message = parse(&with(r#", "generation": {"mode": "sample"}"#)).unwrap_err();
    assert!(
        message.contains("sample") && message.contains("selects greedily"),
        "{message}"
    );
    let message = parse(&with(r#", "generation": {"temperature": 0.7}"#)).unwrap_err();
    assert!(
        message.contains("temperature, topK, and topP require mode"),
        "{message}"
    );
    assert_eq!(
        parse(&with(r#", "generation": {"mode": "greedy", "seed": 7}"#))
            .unwrap()
            .generation_seed,
        7
    );
}

#[test]
fn the_tokenizer_field_is_accepted_and_ignored_and_other_unknown_fields_are_not() {
    assert!(parse(&with(r#", "tokenizer": "/path/to/checkpoint""#)).is_ok());
    let message = parse(&with(r#", "warmupp": 1"#)).unwrap_err();
    assert!(message.contains("unknown field `warmupp`"), "{message}");
    let message = parse(r#"{"cudaLibrary": "/lib.so"}"#).unwrap_err();
    assert!(message.contains("missing field `artifact`"), "{message}");
}

#[test]
fn values_are_checked_as_java_checks_them() {
    for (extra, expected) in [
        (r#", "iterations": 0"#, "iterations must be positive"),
        (r#", "warmup": -1"#, "warmup must not be negative"),
        (r#", "maxContextTokens": 0"#, "maxContextTokens must be positive"),
        (
            r#", "kvDeviceMinimumTokens": 0"#,
            "kvDeviceMinimumTokens must be positive",
        ),
        (
            r#", "shutdownTimeoutSeconds": 0"#,
            "shutdownTimeoutSeconds must be positive",
        ),
        (r#", "promptCorpus": "poems""#, "promptCorpus must be"),
        (r#", "speculation": "mtp""#, "speculation must be"),
        (r#", "scenarios": []"#, "no scenarios selected"),
        (r#", "scenarios": ["decode:8:1"]"#, "at least two new tokens"),
        (
            r#", "overwrite": true, "append": true"#,
            "choose either overwrite or append",
        ),
        (
            r#", "append": true, "output": "r.json""#,
            "append requires JSONL output",
        ),
        (r#", "cpus": " ""#, "cpus must not be blank"),
    ] {
        let message = parse(&with(extra)).unwrap_err();
        assert!(message.contains(expected), "{extra}: {message}");
    }
    assert!(
        parse(&with(r#", "output": "r.json", "overwrite": true"#))
            .unwrap()
            .json_output()
    );
}
