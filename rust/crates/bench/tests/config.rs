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
use euhedral_inference_bench::cpus::CpuSelection;
use euhedral_inference_bench::prompt::PromptCorpus;
use euhedral_inference_bench::run::generation;
use euhedral_inference_engine::sampling::SamplingConfig;

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
    let mut seen = 0;
    for entry in fs::read_dir(configs_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|extension| extension != "json") {
            continue;
        }
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        let result = Options::load(&path, UNIX_EPOCH);
        let options = result.unwrap_or_else(|error| panic!("{name}: {error}"));
        assert!(!options.scenarios.is_empty(), "{name}");
        if name == "custom-cpus.json" {
            assert_eq!(options.cpus, CpuSelection::PerformanceOnePerCore);
            assert_eq!(options.exclude_cores, [0]);
        }
        seen += 1;
    }
    assert!(seen > 0, "the repository has configuration files");

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

    // A file without `output` writes a timestamped, UTC-named file (1790424000 is 2026-09-26T12:00:00Z).
    assert_eq!(
        parse(&with("")).unwrap().output,
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
        ("prefix:2048", "prefix-2048"),
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
        "prefix",
        "prefix:0",
        "prefix:2048:4",
        "sleep:8",
    ] {
        assert!(Scenario::parse(bad).is_err(), "{bad}");
    }
}

#[test]
fn what_the_engine_cannot_do_is_refused_with_the_reason() {
    let message = parse(&with(r#", "prefixCacheBytes": 1073741824"#)).unwrap_err();
    assert!(
        message.contains("prefixCacheBytes must be 0") && message.contains("which turn it on"),
        "{message}"
    );
    assert!(parse(&with(r#", "prefixCacheBytes": 0"#)).is_ok());

    for cpus in [
        "one-per-core",
        "performance",
        "performance-one-per-core",
        "2-5,8",
        " all ",
    ] {
        assert!(parse(&with(&format!(r#", "cpus": "{cpus}""#))).is_ok(), "{cpus}");
    }
    for cpus in ["fast", "5-2"] {
        let message = parse(&with(&format!(r#", "cpus": "{cpus}""#))).unwrap_err();
        assert!(message.contains("cpus must be") && message.contains(cpus), "{message}");
    }
    for key in ["excludeCpus", "excludeCores"] {
        assert!(parse(&with(&format!(r#", "{key}": [0, 3]"#))).is_ok());
        assert!(parse(&with(&format!(r#", "{key}": []"#))).is_ok());
        let message = parse(&with(&format!(r#", "{key}": [-1]"#))).unwrap_err();
        assert!(
            message.contains(key) && message.contains("must not be negative"),
            "{message}"
        );
    }

    assert_eq!(
        parse(&with(r#", "generation": {"mode": "greedy", "seed": 7}"#))
            .unwrap()
            .sampling
            .seed(),
        7
    );
}

fn sampling(generation: &str) -> Result<SamplingConfig, String> {
    parse(&with(&format!(r#", "generation": {generation}"#))).map(|options| options.sampling)
}

#[test]
fn sample_mode_takes_the_defaults_and_each_field() {
    let defaults = sampling(r#"{"mode": "sample"}"#).unwrap();
    assert_eq!(defaults, SamplingConfig::new(0.7, 20, 0.8, 1, false).unwrap());
    let all = sampling(r#"{"mode": "sample", "seed": 9, "temperature": 0.3, "topK": 5, "topP": 0.5}"#).unwrap();
    assert_eq!(all, SamplingConfig::new(0.3, 5, 0.5, 9, false).unwrap());
    assert_eq!(sampling(r#"{"mode": "sample", "topK": 0}"#).unwrap().top_k(), 0);
    assert_eq!(sampling(r#"{"mode": "sample", "topP": 1}"#).unwrap().top_p(), 1.0);
    assert_eq!(sampling(r#"{"seed": 3}"#).unwrap(), SamplingConfig::greedy(3));
    assert_eq!(parse(&with("")).unwrap().sampling, SamplingConfig::greedy(1));
}

#[test]
fn a_negative_seed_keeps_its_bits() {
    let negative = sampling(r#"{"mode": "sample", "seed": -1}"#).unwrap();
    assert_eq!(negative.seed(), u64::MAX);
    assert_eq!(sampling(r#"{"seed": -2}"#).unwrap().seed(), u64::MAX - 1);
}

#[test]
fn sampling_values_are_refused_as_java_refuses_them() {
    for (generation, expected) in [
        (r#"{"mode": "sample", "topK": -1}"#, "topK must not be negative"),
        (
            r#"{"mode": "sample", "topP": 0}"#,
            "top_p must be finite and in the range (0, 1]",
        ),
        (r#"{"mode": "sample", "topP": 1.5}"#, "top_p must be finite"),
        (
            r#"{"mode": "sample", "temperature": 1e300}"#,
            "temperature must be finite",
        ),
        (r#"{"mode": "nucleus"}"#, "generation.mode must be greedy or sample"),
        (r#"{"temperature": 0.7}"#, "temperature, topK, and topP require mode"),
        (
            r#"{"mode": "greedy", "topK": 3}"#,
            "temperature, topK, and topP require mode",
        ),
        (
            r#"{"mode": "greedy", "topP": 0.5}"#,
            "temperature, topK, and topP require mode",
        ),
    ] {
        let message = sampling(generation).unwrap_err();
        assert!(message.contains(expected), "{generation}: {message}");
    }
}

#[test]
fn the_generation_block_holds_the_settings_sent() {
    let sample = sampling(r#"{"mode": "sample", "seed": -1, "temperature": 0.7, "topK": 20, "topP": 0.8}"#).unwrap();
    assert_eq!(
        generation(&sample).to_string(),
        r#"{"temperature":0.7,"topK":20,"topP":0.8,"seed":-1,"greedy":false}"#
    );
    assert_eq!(
        generation(&SamplingConfig::greedy(7)).to_string(),
        r#"{"temperature":0.0,"topK":0,"topP":1.0,"seed":7,"greedy":true}"#
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

#[test]
fn a_prefix_scenario_turns_the_prefix_cache_on_for_the_run_and_no_other_does() {
    let options = parse(&with(r#", "scenarios": ["prefix:2048"]"#)).unwrap();
    assert!(options.uses_prefix_cache());
    assert_eq!(options.scenarios[0].kind, ScenarioKind::PrefixRepeat);
    assert_eq!(options.scenarios[0].requested_new_tokens, 1);
    assert!(!parse(&with("")).unwrap().uses_prefix_cache());
    assert!(
        !parse(&with(r#", "scenarios": ["decode:512:512", "first-token:2048"]"#))
            .unwrap()
            .uses_prefix_cache()
    );
}

#[test]
fn a_prefix_scenario_is_refused_beside_any_other() {
    let message = parse(&with(r#", "scenarios": ["prefix:2048", "decode:512:512"]"#)).unwrap_err();
    assert!(message.contains("run it alone"), "{message}");
    assert!(parse(&with(r#", "scenarios": ["prefix:2048", "prefix:4096"]"#)).is_ok());
}

#[test]
fn a_prefix_prompt_must_end_on_a_checkpoint() {
    let prefix = Scenario::parse("prefix:2048").unwrap();
    assert!(prefix.check_checkpoint(2048).is_ok());
    assert!(prefix.check_checkpoint(512).is_ok());
    let message = prefix.check_checkpoint(1536).unwrap_err().to_string();
    assert!(
        message.contains("not a multiple of the checkpoint interval"),
        "{message}"
    );
    assert!(Scenario::parse("decode:100:4").unwrap().check_checkpoint(2048).is_ok());
}
