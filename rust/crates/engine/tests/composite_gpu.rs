//! A model loaded from a composite: its projections taken from another artifact file, each payload read from the file
//! its entry belongs to. The artifact comes from `euhedral.qwen.rco-artifact`, the library from `euhedral.cuda.library`.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(clippy::print_stdout, reason = "a skipped test says so")]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O and waiting calls: a test reads files and waits on hand-offs"
)]

mod common;

use std::fs;
use std::io;
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::Closer;
use euhedral_inference_edrl::WeightFormat;
use euhedral_inference_engine::generation::{
    GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
};
use euhedral_inference_engine::model::is_companion;
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::{Composite, EngineRuntime, Handoff, Replacement};
use euhedral_inference_settings::{ModelSettings, Settings};

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);

struct Machine {
    artifact: PathBuf,
    library: PathBuf,
    goldens: PathBuf,
}

fn machine() -> Option<Machine> {
    let library = PathBuf::from(test_support::require_property("euhedral.cuda.library")?);
    let artifact = PathBuf::from(test_support::require_property("euhedral.qwen.rco-artifact")?);
    let goldens = PathBuf::from(test_support::require_property("euhedral.rust.goldens")?);
    (artifact.is_file() && goldens.is_dir()).then_some(Machine {
        artifact,
        library,
        goldens,
    })
}

fn settings(artifact: &Path) -> Settings {
    let mut settings = Settings::with_model(ModelSettings {
        artifact: artifact.to_path_buf(),
        name: "test".to_string(),
    });
    settings.context.max_tokens = 4096;
    settings.kernels.graphs.captures_per_view = 8;
    settings
}

struct Collector {
    tokens: Vec<u32>,
    done: Arc<Handoff<Vec<u32>>>,
}

impl GenerationListener for Collector {
    fn prefill_done(&mut self, _at: Instant) {}

    fn token(&mut self, id: u32, _at: Instant) {
        self.tokens.push(id);
    }

    fn quantum(&mut self, _event: QuantumEvent) {}

    fn finished(&mut self, _outcome: GenerationOutcome) {
        self.done.send(mem::take(&mut self.tokens));
    }
}

/// The greedy golden: the prompt and the tokens that follow it.
fn greedy_golden(machine: &Machine) -> (Vec<u32>, Vec<u32>) {
    let text = fs::read_to_string(machine.goldens.join("greedy-rco.txt")).unwrap();
    let ids = |line: &str, tag: &str| -> Vec<u32> {
        line.strip_prefix(tag)
            .unwrap()
            .split_whitespace()
            .map(|id| id.parse().unwrap())
            .collect()
    };
    let mut lines = text.lines();
    (
        ids(lines.next().unwrap(), "prompt "),
        ids(lines.next().unwrap(), "generated "),
    )
}

#[cfg(unix)]
use std::os::unix::fs::symlink;

#[cfg(unix)]
fn link(target: &Path, at: &Path) -> io::Result<()> {
    symlink(target, at)
}

#[cfg(not(unix))]
fn link(_target: &Path, _at: &Path) -> io::Result<()> {
    Err(io::ErrorKind::Unsupported.into())
}

/// Every quantized projection of the layers: the objects a composite can take.
fn choosable(artifact: &Path) -> Vec<String> {
    let (_, table) = common::open_artifact(artifact);
    table
        .entries()
        .iter()
        .filter(|entry| entry.name().starts_with("text/layers/") && !is_companion(entry.name()))
        .filter(|entry| matches!(entry.format(), WeightFormat::Nvfp4 | WeightFormat::Trellis))
        .map(|entry| entry.name().to_string())
        .collect()
}

#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact and the Java references"]
fn a_composite_of_the_artifact_with_itself_generates_the_artifacts_tokens() {
    let Some(machine) = machine() else { return };
    let (prompt, golden) = greedy_golden(&machine);
    let objects = choosable(&machine.artifact);
    assert!(!objects.is_empty());

    // The source is another path to the same bytes, so every projection is read through the source's file.
    let scratch = tempfile::tempdir().unwrap();
    let source = scratch.path().join("source.edrl");
    if link(&machine.artifact, &source).is_err() {
        println!("skipped: this platform makes no link to the artifact");
        return;
    }
    let composite = Composite {
        replacements: vec![Replacement { source, objects }],
    };

    let (owner, ready) =
        EngineRuntime::start_composite(&settings(&machine.artifact), &machine.library, None, composite);
    let runtime = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect("the engine starts");
    let closer = Closer::new(owner, &runtime);
    let done = Handoff::new();
    runtime
        .generate(GenerationRequest {
            prompt,
            max_tokens: 128,
            stop_tokens: Vec::new(),
            speculation: Speculation::Off,
            sampling: SamplingConfig::greedy(0),
            constraint: None,
            listener: Box::new(Collector {
                tokens: Vec::new(),
                done: Arc::clone(&done),
            }),
        })
        .expect("the request is admitted");
    let tokens = done.wait(WAIT).expect("the generation reports");
    assert_eq!(tokens, golden);
    closer.shutdown(runtime).expect("the engine stops");
}

#[test]
#[ignore = "needs a GPU, the native library, the RCO artifact and the Java references"]
fn a_source_that_cannot_be_opened_stops_the_start_naming_it() {
    let Some(machine) = machine() else { return };
    let composite = Composite {
        replacements: vec![Replacement {
            source: PathBuf::from("no-such-source.edrl"),
            objects: choosable(&machine.artifact).into_iter().take(1).collect(),
        }],
    };
    let (owner, ready) =
        EngineRuntime::start_composite(&settings(&machine.artifact), &machine.library, None, composite);
    let error = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect_err("a missing source stops the start");
    assert!(error.to_string().contains("no-such-source.edrl"), "{error}");
    drop(owner);
}
