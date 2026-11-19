//! `package-tokenizer` on a real lattice, without a card: the files of a small byte-level tokenizer are read with
//! `AsyncIo`, checked in a host-lake frame, and written after the content of a tiny artifact.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting call: the test thread waits on the hand-off for the run it started, and writes its files"
)]
#![allow(
    clippy::disallowed_types,
    reason = "a channel carries the run's result to the waiting test thread"
)]

use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc;
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_graph::Lake;
use euhedral_inference_edrl::tiny::TinyArtifact;
use euhedral_inference_edrl::{FOOTER_BYTES, TokenizerFooter, TokenizerSection};
use euhedral_inference_engine::lattice_config;
use euhedral_inference_quantization::package_error::PackageError;
use euhedral_inference_quantization::package_tokenizer::{PackageRequest, Packaged, package_tokenizer};
use euhedral_inference_quantization::tokenizer_files::TokenizerFiles;
use euhedral_inference_settings::{ModelSettings, Settings};
use euhedral_io::AsyncIo;
use serde_json::{Value, json};
use tempfile::TempDir;

const TEMPLATE: &str = include_str!("../../api-protocol/tests/data/qwen-chat-template.jinja");

/// The expression the Qwen checkpoints split with.
const SPLIT: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

/// The byte-level spelling of every byte.
fn byte_chars() -> Vec<char> {
    let mut next = 256;
    (0..=255u32)
        .map(|byte| {
            if matches!(byte, 33..=126 | 161..=172 | 174..=255) {
                char::from_u32(byte).unwrap()
            } else {
                next += 1;
                char::from_u32(next - 1).unwrap()
            }
        })
        .collect()
}

fn tokenizer_json(kind: &str) -> String {
    let mut vocab = serde_json::Map::new();
    for (byte, character) in byte_chars().iter().enumerate() {
        vocab.insert(character.to_string(), json!(byte));
    }
    vocab.insert("he".into(), json!(256));
    let added = |id: u32, content: &str| {
        json!({"id": id, "content": content, "single_word": false, "lstrip": false, "rstrip": false,
               "normalized": false, "special": true})
    };
    json!({
        "version": "1.0",
        "added_tokens": [added(257, "<|im_start|>"), added(258, "<|im_end|>"), added(259, "<|endoftext|>")],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": [
            {"type": "Split", "pattern": {"Regex": SPLIT}, "behavior": "Isolated", "invert": false},
            {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        ]},
        "post_processor": null,
        "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
        "model": {
            "type": kind, "dropout": null, "unk_token": null, "continuing_subword_prefix": null,
            "end_of_word_suffix": null, "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
            "vocab": vocab, "merges": ["h e"],
        },
    })
    .to_string()
}

/// What a checkpoint directory holds.
struct Checkpoint {
    tokenizer: String,
    config: Value,
    template_file: Option<String>,
    generation: Value,
}

impl Checkpoint {
    fn standard() -> Checkpoint {
        Checkpoint {
            tokenizer: tokenizer_json("BPE"),
            config: json!({"eos_token": "<|im_end|>", "bos_token": null, "add_bos_token": false}),
            template_file: Some(TEMPLATE.to_owned()),
            generation: json!({"bos_token_id": 259, "eos_token_id": [258, 259], "pad_token_id": 259, "top_k": 20}),
        }
    }

    fn write(&self, directory: &Path) {
        fs::create_dir_all(directory).unwrap();
        fs::write(directory.join("tokenizer.json"), &self.tokenizer).unwrap();
        fs::write(directory.join("tokenizer_config.json"), self.config.to_string()).unwrap();
        fs::write(directory.join("generation_config.json"), self.generation.to_string()).unwrap();
        if let Some(template) = &self.template_file {
            fs::write(directory.join("chat_template.jinja"), template).unwrap();
        }
    }
}

struct Run {
    lattice: Arc<ControlPlaneLattice>,
    io: Arc<AsyncIo>,
    host: Arc<Lake>,
}

impl Run {
    fn start() -> Run {
        let settings = Settings::with_model(ModelSettings::default());
        let lattice = ControlPlaneLattice::new(lattice_config(&settings)).expect("the lattice starts");
        let io = AsyncIo::open().expect("the asynchronous I/O opens");
        lattice
            .add_upstream(Arc::clone(&io) as Arc<dyn LatticeSource>)
            .expect("the lattice takes the I/O");
        let host = Lake::new(
            Arc::clone(&lattice),
            NonZeroUsize::new(2).unwrap(),
            Duration::from_secs(1),
        );
        Run { lattice, io, host }
    }

    fn package(&self, request: PackageRequest) -> Result<Packaged, PackageError> {
        let (sender, receiver) = mpsc::channel();
        package_tokenizer(
            Arc::clone(&self.io),
            Arc::clone(&self.host),
            request,
            Box::new(move |result| {
                sender.send(result).unwrap();
            }),
        )?;
        receiver.recv_timeout(Duration::from_secs(60)).expect("the run reports")
    }
}

impl Drop for Run {
    fn drop(&mut self) {
        self.host.complete_gracefully();
        self.io.complete();
        self.lattice.close();
    }
}

struct Fixture {
    directory: TempDir,
    content: Vec<u8>,
}

impl Fixture {
    fn new(checkpoint: &Checkpoint) -> Fixture {
        let directory = tempfile::tempdir().unwrap();
        checkpoint.write(&directory.path().join("checkpoint"));
        // The tiny artifact carries a tokenizer section of its own; the content is what precedes it.
        let mut content = TinyArtifact::new(3, true).bytes().unwrap();
        let footer = content.split_off(content.len() - FOOTER_BYTES);
        let footer = TokenizerFooter::parse(footer.as_slice().try_into().unwrap())
            .unwrap()
            .expect("a footer");
        content.truncate(content.len() - footer.body_length() as usize);
        fs::write(directory.path().join("artifact.edrl"), &content).unwrap();
        Fixture { directory, content }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.directory.path().join(name)
    }

    fn artifact(&self) -> PathBuf {
        self.path("artifact.edrl")
    }

    fn in_place(&self) -> PackageRequest {
        PackageRequest::in_place(self.artifact(), self.path("checkpoint"))
    }

    fn to_output(&self) -> PackageRequest {
        PackageRequest::to_output(self.artifact(), self.path("checkpoint"), self.path("out.edrl"))
    }
}

/// The section at the end of `bytes`.
fn section_of(bytes: &[u8]) -> TokenizerSection {
    let (rest, footer) = bytes.split_at(bytes.len() - FOOTER_BYTES);
    let footer = TokenizerFooter::parse(footer.try_into().unwrap())
        .unwrap()
        .expect("a footer");
    let body = rest[rest.len() - footer.body_length() as usize..].to_vec();
    TokenizerSection::parse(body, &footer).unwrap()
}

#[test]
fn an_artifact_gains_the_tokenizer_in_place_and_a_second_run_changes_nothing() {
    let run = Run::start();
    let fixture = Fixture::new(&Checkpoint::standard());
    // A stale tail longer than the section the run writes: it is cut away.
    let mut stale = fixture.content.clone();
    stale.extend(vec![0xAB; 2_000_000]);
    fs::write(fixture.artifact(), &stale).unwrap();

    let packaged = run.package(fixture.in_place()).unwrap();
    let bytes = fs::read(fixture.artifact()).unwrap();
    assert_eq!(packaged.target, fixture.artifact());
    assert_eq!(packaged.content_bytes, fixture.content.len() as u64);
    assert_eq!(bytes.len() as u64, packaged.content_bytes + packaged.section_bytes);
    assert_eq!(&bytes[..fixture.content.len()], fixture.content.as_slice());
    let section = section_of(&bytes);
    assert_eq!(
        section.text("tokenizer.json"),
        Some(Checkpoint::standard().tokenizer.as_str())
    );
    assert_eq!(section.text("chat_template.jinja"), Some(TEMPLATE));
    assert_eq!(
        section.text("special_tokens.json"),
        Some(r#"{"bos_token_id":259,"eos_token_id":[258,259],"pad_token_id":259}"#)
    );

    run.package(fixture.in_place()).unwrap();
    assert_eq!(fs::read(fixture.artifact()).unwrap(), bytes);
}

#[test]
fn an_output_gets_the_content_and_the_tokenizer_and_the_source_is_left_alone() {
    let run = Run::start();
    let fixture = Fixture::new(&Checkpoint::standard());
    let in_place_twin = fixture.path("twin.edrl");
    fs::copy(fixture.artifact(), &in_place_twin).unwrap();

    // One byte at a time would be slow; a chunk smaller than the content makes the copy take several reads and writes.
    let mut request = fixture.to_output();
    request.copy_chunk_bytes = 7001;
    let packaged = run.package(request).unwrap();
    assert_eq!(packaged.target, fixture.path("out.edrl"));
    assert_eq!(
        fs::read(fixture.artifact()).unwrap(),
        fixture.content,
        "the source is left alone"
    );
    assert!(!fixture.path("out.edrl.part").exists());

    run.package(PackageRequest::in_place(
        in_place_twin.clone(),
        fixture.path("checkpoint"),
    ))
    .unwrap();
    assert_eq!(
        fs::read(fixture.path("out.edrl")).unwrap(),
        fs::read(in_place_twin).unwrap(),
        "an output holds what packaging in place gives"
    );
}

#[test]
fn a_checkpoint_without_a_template_file_gives_the_template_of_its_config() {
    let run = Run::start();
    let mut checkpoint = Checkpoint::standard();
    checkpoint.template_file = None;
    checkpoint.config = json!({"eos_token": "<|im_end|>", "chat_template": TEMPLATE});
    let fixture = Fixture::new(&checkpoint);
    run.package(fixture.in_place()).unwrap();
    let section = section_of(&fs::read(fixture.artifact()).unwrap());
    assert_eq!(section.text("chat_template.jinja"), Some(TEMPLATE));
}

#[test]
fn a_section_that_was_cut_short_or_belongs_to_another_checkpoint_is_replaced() {
    let run = Run::start();
    let fixture = Fixture::new(&Checkpoint::standard());
    run.package(fixture.in_place()).unwrap();
    let complete = fs::read(fixture.artifact()).unwrap();

    // An interrupted run: the body without the footer.
    fs::write(fixture.artifact(), &complete[..complete.len() - FOOTER_BYTES - 10]).unwrap();
    run.package(fixture.in_place()).unwrap();
    assert_eq!(fs::read(fixture.artifact()).unwrap(), complete);

    // A section of another checkpoint.
    let mut other = Checkpoint::standard();
    other.generation = json!({"eos_token_id": 258});
    other.write(&fixture.path("other"));
    run.package(PackageRequest::in_place(fixture.artifact(), fixture.path("other")))
        .unwrap();
    let section = section_of(&fs::read(fixture.artifact()).unwrap());
    assert_eq!(section.text("special_tokens.json"), Some(r#"{"eos_token_id":258}"#));
    run.package(fixture.in_place()).unwrap();
    assert_eq!(fs::read(fixture.artifact()).unwrap(), complete);
}

/// What each refused run must leave: the artifact as it was, and no output.
fn assert_refused(run: &Run, fixture: &Fixture) {
    for request in [fixture.in_place(), fixture.to_output()] {
        assert!(run.package(request).is_err());
        assert_eq!(fs::read(fixture.artifact()).unwrap(), fixture.content);
        assert!(!fixture.path("out.edrl").exists());
        assert!(!fixture.path("out.edrl.part").exists());
    }
}

#[test]
fn files_that_do_not_pass_the_checks_leave_the_artifact_untouched_and_make_no_output() {
    let run = Run::start();

    let mut changed_template = Checkpoint::standard();
    changed_template.template_file = Some("{{ messages }}".to_owned());
    let fixture = Fixture::new(&changed_template);
    assert_refused(&run, &fixture);
    assert!(matches!(
        run.package(fixture.in_place()),
        Err(PackageError::Template(_))
    ));

    let mut not_bpe = Checkpoint::standard();
    not_bpe.tokenizer = tokenizer_json("WordPiece");
    let fixture = Fixture::new(&not_bpe);
    assert_refused(&run, &fixture);
    assert!(matches!(
        run.package(fixture.in_place()),
        Err(PackageError::Tokenizer(_))
    ));

    let fixture = Fixture::new(&Checkpoint::standard());
    fs::remove_file(fixture.path("checkpoint").join("tokenizer.json")).unwrap();
    assert_refused(&run, &fixture);
    assert!(matches!(
        run.package(fixture.in_place()),
        Err(PackageError::Io {
            what: "tokenizer.json",
            ..
        })
    ));

    let mut no_template = Checkpoint::standard();
    no_template.template_file = None;
    let fixture = Fixture::new(&no_template);
    assert_refused(&run, &fixture);
    assert!(matches!(
        run.package(fixture.in_place()),
        Err(PackageError::NoChatTemplate)
    ));
}

#[test]
fn a_file_that_is_not_an_artifact_is_refused_before_anything_is_written() {
    let run = Run::start();
    let fixture = Fixture::new(&Checkpoint::standard());
    for damaged in [
        vec![0u8; 10],
        vec![0u8; 4096],
        {
            // A version the reader does not handle.
            let mut bytes = fixture.content.clone();
            bytes[4..8].copy_from_slice(&3i32.to_be_bytes());
            bytes
        },
        fixture.content[..fixture.content.len() - 5].to_vec(),
    ] {
        fs::write(fixture.artifact(), &damaged).unwrap();
        assert!(run.package(fixture.in_place()).is_err());
        assert!(run.package(fixture.to_output()).is_err());
        assert_eq!(fs::read(fixture.artifact()).unwrap(), damaged);
        assert!(!fixture.path("out.edrl").exists());
        assert!(!fixture.path("out.edrl.part").exists());
    }
}

#[test]
fn the_special_tokens_keep_the_ids_the_generation_config_sets_in_a_fixed_order() {
    let files = |generation: &str| {
        TokenizerFiles::from_checkpoint(
            Vec::new(),
            b"{}".to_vec(),
            Some(b"template".to_vec()),
            generation.as_bytes(),
        )
        .unwrap()
        .texts()
        .special_tokens
        .to_owned()
    };
    assert_eq!(
        files(r#"{"pad_token_id": 3, "top_p": 0.8, "eos_token_id": [1, 2], "bos_token_id": null}"#),
        r#"{"bos_token_id":null,"eos_token_id":[1,2],"pad_token_id":3}"#
    );
    assert_eq!(files(r#"{"top_k": 20}"#), "{}");
    assert_eq!(files("[]"), "{}");
}
