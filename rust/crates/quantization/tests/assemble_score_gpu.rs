//! Device test of assembled artifacts scored on the card: a view taken wholly from one source scores as that source,
//! and the composite an allocation makes scores as the file written for it. `#[ignore]`d like the crate's other device
//! tests: `cargo test -p euhedral-inference-quantization --test assemble_score_gpu -- --ignored --nocapture`.
//!
//! Keys (environment `EUHEDRAL_TEST_<KEY>` first, then `~/.gradle/gradle.properties`; a missing key fails the test):
//! `euhedral.cuda.library`, `euhedral.qwen.trellis30-artifact` (the base), `euhedral.qwen.nvfp4-artifact` and
//! `euhedral.quantization.score-reference`. The written artifact is as large as the base: it goes under the
//! temporary directory (`TMPDIR`), which must hold it, and is removed when the test ends, a failure included.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test file: it waits on the hand-offs of the runs it starts and stops on a failed unwrap"
)]

mod checkpoint_support;

use std::collections::BTreeMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use checkpoint_support::World;
use euhedral_inference_engine::{Composite, EngineRuntime, Handoff, Replacement};
use euhedral_inference_quantization::assemble::{AssembleError, Done, SourceSet, view, write};
use euhedral_inference_quantization::kld::{self, Reference};
use euhedral_inference_quantization::mixed::{Allocation, QuantOption};
use euhedral_inference_quantization::reference::Forcer;
use euhedral_inference_quantization::score::device::Geometry;
use euhedral_inference_quantization::score::summary::FIELDS;
use euhedral_inference_quantization::score::{Scorer, Source, score};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings, SpeculationMode};

const WAIT: Duration = Duration::from_secs(20 * 60);
const ROWS: usize = 4;
const BATCH: usize = 2;
const DRAFT_DEPTH: u8 = 3;
const BASE: QuantOption = QuantOption::Trellis3_0;
const OTHER: QuantOption = QuantOption::Nvfp4;
/// The field of a row's values that holds its KL.
const KL: usize = 2;
const HEADER_BYTES: usize = 20;

fn property(key: &str) -> String {
    test_support::gradle_property(key).unwrap_or_else(|| panic!("set the property {key}: a device test needs it"))
}

/// The values of every chunk, one run of the engine over `artifact` and `composite`.
fn scored(artifact: &Path, composite: Composite, library: &Path, reference: &Path) -> Vec<Vec<f64>> {
    let mut header = [0u8; HEADER_BYTES];
    File::open(reference).unwrap().read_exact(&mut header).unwrap();
    let (context, _, _) = Reference::parse_header(&header).unwrap();
    let mut settings = Settings::with_model(ModelSettings {
        artifact: artifact.to_path_buf(),
        name: "assembled".to_owned(),
    });
    settings.context.max_tokens = u32::try_from(context.next_multiple_of(512)).unwrap();
    settings.prefix_cache.size = ByteSize::mib(0);
    settings.speculation.mode = SpeculationMode::None;
    settings.batching.max_sequences = u32::try_from(BATCH).unwrap();
    settings.batching.forced_rows = u32::try_from(ROWS).unwrap();
    let (owner, ready) = EngineRuntime::start_composite(&settings, library, None, composite);
    let runtime = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect("the engine starts");
    let io = runtime.io();
    let header = Reference::read(io, reference)
        .wait(WAIT)
        .expect("the header is read")
        .unwrap();
    let records = kld::open_records(io, reference)
        .wait(WAIT)
        .expect("the records open")
        .unwrap();
    let geometry = Geometry {
        vocabulary: runtime.config().vocab_size,
        record_bytes: header.row_bytes(),
        group: BATCH * ROWS,
    };
    let scorer = Scorer::open(&runtime, geometry).expect("the scorer opens");
    let forcer: Arc<dyn Forcer> = Arc::clone(&runtime) as Arc<dyn Forcer>;
    let source = Source {
        forcer,
        io: Arc::clone(io),
        scorer: Arc::clone(&scorer),
        reference: header,
        records,
        rows: ROWS,
    };
    let values = score(source, BATCH)
        .wait(WAIT)
        .expect("the score ends")
        .expect("the score succeeds");
    scorer.close();
    owner.shutdown(runtime).expect("the engine shuts down");
    values
}

/// The mean KL of each chunk.
fn chunk_kl(values: &[Vec<f64>]) -> Vec<f64> {
    values
        .iter()
        .map(|chunk| {
            let rows = chunk.len() / FIELDS;
            chunk.iter().skip(KL).step_by(FIELDS).sum::<f64>() / rows as f64
        })
        .collect()
}

/// Same chunks, same values to the bit, all five fields of every row.
fn assert_same(left: &[Vec<f64>], right: &[Vec<f64>], what: &str) {
    assert!(!left.is_empty(), "{what}: nothing was scored");
    assert_eq!(left.len(), right.len(), "{what}: the chunk counts differ");
    for (chunk, (a, b)) in left.iter().zip(right).enumerate() {
        let same = a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same, "{what}: chunk {chunk} differs");
    }
}

fn call<T: Send + 'static>(start: impl FnOnce(Done<T>) -> Result<(), AssembleError>) -> Result<T, AssembleError> {
    let handoff = Handoff::new();
    let sender = Arc::clone(&handoff);
    start(Arc::new(move |result| sender.send(result)))?;
    handoff.wait(WAIT).expect("the callback never ran")
}

/// Taking every projection from the base's own source scores as the base, and the composite that takes the
/// `mlp/down` projections from the other source scores as the artifact written for the same allocation.
#[test]
#[ignore = "needs a GPU, the engine library, two source artifacts and a reference"]
fn a_view_of_one_source_scores_as_it_and_a_composite_scores_as_the_file_written_for_it() {
    let library = PathBuf::from(property("euhedral.cuda.library"));
    let base_path = PathBuf::from(property("euhedral.qwen.trellis30-artifact"));
    let other_path = PathBuf::from(property("euhedral.qwen.nvfp4-artifact"));
    let reference = PathBuf::from(property("euhedral.quantization.score-reference"));

    let world = World::start();
    let artifacts = BTreeMap::from([(BASE, base_path.clone()), (OTHER, other_path.clone())]);
    let sources = call(|done| SourceSet::open(&world.io, &artifacts, BASE, 8 << 20, done)).unwrap();
    let every = || sources.projections().iter().cloned();
    let down: Vec<String> = every().filter(|name| name.ends_with("/mlp/down")).collect();
    assert!(!down.is_empty(), "the sources hold no mlp/down projection");

    let source = scored(&base_path, Composite::default(), &library, &reference);
    let pure = Composite {
        replacements: vec![Replacement {
            source: base_path.clone(),
            objects: every().collect(),
        }],
        draft_depth: None,
    };
    let identity = scored(&base_path, pure, &library, &reference);
    assert_same(
        &source,
        &identity,
        "the view of the base's own objects against the base",
    );
    assert_eq!(chunk_kl(&source), chunk_kl(&identity));

    let mixed = Composite {
        replacements: vec![Replacement {
            source: other_path,
            objects: down.clone(),
        }],
        draft_depth: Some(DRAFT_DEPTH),
    };
    let composite = scored(&base_path, mixed, &library, &reference);

    // The directory and the file it holds go when the guard drops, on a failed assertion too.
    let directory = tempfile::Builder::new().prefix("assembled-").tempdir().unwrap();
    let out = directory.path().join("mixed.edrl");
    let allocation = Allocation::new(down.into_iter().map(|name| (name, OTHER)).collect());
    let assembled = view(&sources, &allocation, Some(DRAFT_DEPTH)).unwrap();
    call(|done| write(&world.io, &world.host, &assembled, &out, 8 << 20, done)).unwrap();
    let written = scored(&out, Composite::default(), &library, &reference);
    assert_same(&composite, &written, "the composite against the file written for it");
    assert_eq!(chunk_kl(&composite), chunk_kl(&written));
    drop(directory);
    world.finish();
}
