//! A model loaded from a composite: projections taken from another artifact file, each payload read from the file its
//! entry belongs to. The artifact comes from `euhedral.qwen.rco-artifact`, the library from `euhedral.cuda.library`.

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

use std::fs::{self, File};
use std::io::{Read, Seek, SeekFrom, Write};
use std::mem;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use common::Closer;
use euhedral_inference_edrl::{
    ArtifactPlan, HEADER_BYTES, Header, ObjectSpec, PAYLOAD_ALIGNMENT, TensorEntry, WeightFormat,
};
use euhedral_inference_engine::generation::{
    GenerationListener, GenerationOutcome, GenerationRequest, QuantumEvent, Speculation,
};
use euhedral_inference_engine::model::{TRELLIS_COMPANIONS, is_companion};
use euhedral_inference_engine::sampling::SamplingConfig;
use euhedral_inference_engine::{Composite, EngineRuntime, Handoff, Replacement};
use euhedral_inference_settings::{ModelSettings, Settings};

/// The longest a device test waits for one hand-off; past it the test fails.
const WAIT: Duration = Duration::from_secs(20 * 60);

/// A prompt of ids every vocabulary of the model holds.
const PROMPT: [u32; 6] = [785, 6722, 315, 9625, 374, 264];

struct Machine {
    artifact: PathBuf,
    library: PathBuf,
}

fn machine() -> Option<Machine> {
    let library = PathBuf::from(test_support::require_property("euhedral.cuda.library")?);
    let artifact = PathBuf::from(test_support::require_property("euhedral.qwen.rco-artifact")?);
    artifact.is_file().then_some(Machine { artifact, library })
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

/// The greedy tokens the engine over `artifact` (and `composite`) generates after the prompt.
fn generate(machine: &Machine, composite: Composite) -> Vec<u32> {
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
            prompt: PROMPT.to_vec(),
            max_tokens: 32,
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
    closer.shutdown(runtime).expect("the engine stops");
    tokens
}

/// The index of the artifact: its header, the metadata bytes and its table.
fn index(artifact: &Path) -> (Header, Vec<u8>, Vec<TensorEntry>) {
    let (_, table) = common::open_artifact(artifact);
    let mut file = File::open(artifact).unwrap();
    let mut header = [0u8; HEADER_BYTES];
    file.read_exact(&mut header).unwrap();
    let header = Header::parse(&header).unwrap();
    let range = header.metadata_range();
    let mut metadata = vec![0u8; usize::try_from(range.end - range.start).unwrap()];
    file.seek(SeekFrom::Start(range.start)).unwrap();
    file.read_exact(&mut metadata).unwrap();
    (header, metadata, table.entries().to_vec())
}

/// Two trellis projections of different layers: the objects a composite takes.
fn two_projections(entries: &[TensorEntry]) -> [String; 2] {
    let layer = |name: &str| name.split('/').nth(2).map(str::to_string);
    let mut chosen: Vec<String> = Vec::new();
    for entry in entries {
        let name = entry.name();
        if name.starts_with("text/layers/")
            && !is_companion(name)
            && entry.format() == WeightFormat::Trellis
            && chosen.iter().all(|taken| layer(taken) != layer(name))
        {
            chosen.push(name.to_string());
        }
        if chosen.len() == 2 {
            break;
        }
    }
    let [first, second] = <[String; 2]>::try_from(chosen).expect("the artifact has trellis projections in two layers");
    [first, second]
}

/// Writes an artifact at `path` that holds `objects` and their companions, with the payloads of `base`, in the
/// reverse of the base's order and at other offsets, so only a read from this file finds them. With `corrupt`, a block
/// inside the first object's payload is zeros.
fn write_source(base: &Path, path: &Path, objects: &[String; 2], corrupt: bool) {
    let (header, metadata, entries) = index(base);
    let mut specs = Vec::new();
    let mut from = Vec::new();
    for object in objects.iter().rev() {
        let names = [object.clone()]
            .into_iter()
            .chain(TRELLIS_COMPANIONS.map(|suffix| format!("{object}{suffix}")));
        for name in names {
            let entry = entries.iter().find(|entry| entry.name() == name).unwrap();
            specs.push(ObjectSpec {
                name,
                shape: entry.shape().to_vec(),
                dtype: entry.dtype(),
                format: entry.format(),
                layout: entry.layout(),
                bytes: entry.bytes(),
            });
            from.push(entry.clone());
        }
    }
    let plan = ArtifactPlan::new(header.metadata_size(), &specs, PAYLOAD_ALIGNMENT).unwrap();
    let mut out = File::create(path).unwrap();
    out.write_all(&plan.header().encode()).unwrap();
    out.write_all(&metadata).unwrap();
    out.write_all(&plan.table().unwrap()).unwrap();
    let mut source = File::open(base).unwrap();
    let mut at = out.stream_position().unwrap();
    for (placed, original) in plan.entries().iter().zip(&from) {
        out.write_all(&vec![0u8; usize::try_from(placed.offset() - at).unwrap()])
            .unwrap();
        let mut payload = vec![0u8; usize::try_from(original.bytes()).unwrap()];
        source.seek(SeekFrom::Start(original.offset())).unwrap();
        source.read_exact(&mut payload).unwrap();
        if corrupt && placed.name() == specs[0].name {
            let middle = payload.len() / 2;
            let end = middle + 4096.min(payload.len() - middle);
            payload[middle..end].fill(0);
        }
        out.write_all(&payload).unwrap();
        at = placed.offset() + original.bytes();
    }
    out.write_all(&vec![0u8; usize::try_from(PAYLOAD_ALIGNMENT).unwrap()])
        .unwrap();
    out.flush().unwrap();
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_composite_reads_each_object_from_its_own_file_and_generates_the_bases_tokens() {
    let Some(machine) = machine() else { return };
    let (_, _, entries) = index(&machine.artifact);
    let objects = two_projections(&entries);
    let scratch = PathBuf::from(env!("CARGO_TARGET_TMPDIR")).join("composite-weights");
    fs::create_dir_all(&scratch).unwrap();
    let source = scratch.join("source.edrl");
    write_source(&machine.artifact, &source, &objects, false);

    // The source holds the payloads at other offsets than the base: a read of the base's file at the source's offset
    // finds other bytes, and the tokens differ.
    let expected = generate(&machine, Composite::default());
    assert_eq!(expected.len(), 32);
    let composite = Composite {
        replacements: vec![Replacement {
            source: source.clone(),
            objects: objects.to_vec(),
        }],
        draft_depth: None,
    };
    let tokens = generate(&machine, composite);
    fs::remove_file(&source).unwrap();
    assert_eq!(tokens, expected);

    // The same source with a block of one payload zeroed: the engine reads the source's payload, so its tokens differ.
    // Without this, a composite that were ignored would pass the comparison above.
    let damaged = scratch.join("damaged.edrl");
    write_source(&machine.artifact, &damaged, &objects, true);
    let composite = Composite {
        replacements: vec![Replacement {
            source: damaged.clone(),
            objects: objects.to_vec(),
        }],
        draft_depth: None,
    };
    let changed = generate(&machine, composite);
    fs::remove_file(&damaged).unwrap();
    assert_ne!(changed, expected);
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn a_source_that_cannot_be_opened_stops_the_start_naming_it() {
    let Some(machine) = machine() else { return };
    let (_, _, entries) = index(&machine.artifact);
    let composite = Composite {
        replacements: vec![Replacement {
            source: PathBuf::from("no-such-source.edrl"),
            objects: two_projections(&entries).into_iter().take(1).collect(),
        }],
        draft_depth: None,
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
