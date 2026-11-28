//! Device test of the reference writer: the engine forces two chunks of a text. `#[ignore]`d like the crate's other
//! device tests: `cargo test -p euhedral-inference-quantization --test reference_gpu -- --ignored --nocapture`.
//!
//! The engine library is the `euhedral.cuda.library` property and the model the `euhedral.qwen.rco-artifact` one
//! (environment `EUHEDRAL_TEST_<KEY>` first); a missing key fails the test.
#![cfg(target_os = "linux")]
#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting call: a test thread waits on the hand-off for the run it started, and reads its files"
)]

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use euhedral_inference_engine::EngineRuntime;
use euhedral_inference_quantization::kld::Reference;
use euhedral_inference_quantization::reference::{Forcer, Options, Written, partial::partial_path, tokenize, write};
use euhedral_inference_settings::{ByteSize, ModelSettings, Settings};

const WAIT: Duration = Duration::from_secs(20 * 60);
const CONTEXT: usize = 32;
const TEXT: &str = "The quick brown fox jumps over the lazy dog while the river runs quietly past the old mill, \
    and the farmers count the sacks of grain stacked beside the barn before the first frost arrives in the valley. \
    In the evening the lamps are lit, the kettle sings on the stove, and the children read aloud from a worn book \
    about ships, harbours and the long winds of the northern sea. Nobody hurries; the clock on the wall ticks, the \
    dog sleeps by the door, and the night settles slowly over the fields, the hills and the quiet road to town.";

fn property(key: &str) -> String {
    test_support::test_setting(key).unwrap_or_else(|| panic!("set the property {key}: a device test needs it"))
}

fn run(runtime: &Arc<EngineRuntime>, out: &Path, tokens: &[Vec<u32>], options: Options) -> Written {
    let forcer: Arc<dyn Forcer> = Arc::clone(runtime) as Arc<dyn Forcer>;
    let reference = Reference {
        context: CONTEXT,
        vocabulary: runtime.config().vocab_size,
        tokens: tokens.to_vec(),
    };
    write(runtime.io(), runtime.host(), forcer, out, reference, options)
        .wait(WAIT)
        .expect("the run ended")
        .expect("the run succeeded")
}

/// Two chunks of a text are forced through the model: the file holds the tokens, records that decode to log
/// probabilities in which the model's own best guess is likely, and a run stopped after one chunk and resumed writes the
/// bytes of a run that was not stopped.
#[test]
#[ignore = "needs a GPU, the engine library and the RCO artifact"]
fn two_chunks_are_written_and_a_stopped_run_resumes_to_the_same_bytes() {
    let library = PathBuf::from(property("euhedral.cuda.library"));
    let artifact = PathBuf::from(property("euhedral.qwen.rco-artifact"));
    let tokens = tokenize(&artifact, TEXT, CONTEXT, 2, false).unwrap();
    assert_eq!(tokens.len(), 2);

    let mut settings = Settings::with_model(ModelSettings {
        artifact: artifact.clone(),
        name: "reference".to_owned(),
    });
    settings.context.max_tokens = 512;
    settings.prefix_cache.size = ByteSize::mib(0);
    settings.batching.forced_rows = 8;
    settings.batching.max_sequences = 4;
    let (owner, ready) = EngineRuntime::start(&settings, &library);
    let runtime = ready
        .wait(WAIT)
        .expect("the engine reports")
        .expect("the engine starts");

    let directory = tempfile::tempdir().unwrap();
    let whole = directory.path().join("whole.kld");
    let stopped = directory.path().join("stopped.kld");
    assert!(run(&runtime, &whole, &tokens, Options::default()).finished);
    let first = run(
        &runtime,
        &stopped,
        &tokens,
        Options {
            stop_after: Some(1),
            ..Options::default()
        },
    );
    assert_eq!((first.completed, first.finished), (1, false));
    assert!(partial_path(&stopped).exists());
    assert!(run(&runtime, &stopped, &tokens, Options::default()).finished);

    let (whole_bytes, stopped_bytes) = (fs::read(&whole).unwrap(), fs::read(&stopped).unwrap());
    assert_eq!(
        whole_bytes, stopped_bytes,
        "a resumed run differs from an unstopped one"
    );
    let parsed = Reference::parse(&whole_bytes).unwrap();
    assert_eq!(parsed.tokens, tokens);
    assert_eq!(parsed.vocabulary, runtime.config().vocab_size);
    let expected = parsed.row_offset(parsed.chunks(), 0) as usize;
    assert_eq!(
        whole_bytes.len(),
        expected,
        "the file is the header, the tokens and the records only"
    );
    assert!(!partial_path(&stopped).exists());
    // The records are not empty rows: every one has a positive scale.
    for chunk in 0..parsed.chunks() {
        for step in 0..parsed.records() {
            let at = parsed.row_offset(chunk, step) as usize;
            let scale = f32::from_le_bytes(whole_bytes[at..at + 4].try_into().unwrap());
            assert!(scale > 0.0, "record {step} of chunk {chunk} has no spread");
        }
    }
    owner.shutdown(runtime).expect("the engine shuts down");
}
