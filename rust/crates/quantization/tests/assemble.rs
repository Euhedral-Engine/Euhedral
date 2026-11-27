//! Opening the sources of a mixed artifact, selecting the objects an allocation names, and writing the artifact.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it stops on a failed unwrap and prepares and reads its files directly"
)]

mod checkpoint_support;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use checkpoint_support::World;
use euhedral_inference_edrl::tiny::{DRAFTER_PROJECTION, PROJECTIONS, TOKENIZER, TinyArtifact};
use euhedral_inference_edrl::{MAX_DRAFT_DEPTH, TokenizerTexts};
use euhedral_inference_engine::{ArtifactGuard, Handoff};
use euhedral_inference_quantization::assemble::{
    ArtifactFile, AssembleError, Done, Request, SourceSet, View, run, view, write,
};
use euhedral_inference_quantization::mixed::{Allocation, QuantOption, Sources};
use tempfile::TempDir;

const BASE: QuantOption = QuantOption::Trellis3_0;
const OTHER: QuantOption = QuantOption::Nvfp4;

/// Starts a call and waits, outside the lattice, for its result.
fn call<T: Send + 'static>(start: impl FnOnce(Done<T>) -> Result<(), AssembleError>) -> Result<T, AssembleError> {
    let handoff = Handoff::new();
    let sender = Arc::clone(&handoff);
    start(Arc::new(move |result| sender.send(result)))?;
    handoff.wait(Duration::from_secs(20)).expect("the callback never ran")
}

struct Fixture {
    world: World,
    dir: TempDir,
}

impl Fixture {
    fn new() -> Fixture {
        Fixture {
            world: World::start(),
            dir: tempfile::tempdir().unwrap(),
        }
    }

    fn put(&self, name: &str, artifact: &TinyArtifact<'_>) -> PathBuf {
        let path = self.dir.path().join(name);
        fs::write(&path, artifact.bytes().unwrap()).unwrap();
        path
    }

    /// The base (with companions, seed 1) and a second source (without, seed 2).
    fn two(&self) -> BTreeMap<QuantOption, PathBuf> {
        BTreeMap::from([
            (BASE, self.put("base.edrl", &TinyArtifact::new(1, true))),
            (OTHER, self.put("other.edrl", &TinyArtifact::new(2, false))),
        ])
    }

    fn open(&self, artifacts: &BTreeMap<QuantOption, PathBuf>) -> Result<SourceSet, AssembleError> {
        call(|done| SourceSet::open(&self.world.io, artifacts, BASE, 100, done))
    }

    fn write(&self, view: &View, out: &Path, chunk: usize) -> Result<(), AssembleError> {
        call(|done| write(&self.world.io, &self.world.host, view, out, chunk, done))
    }

    fn table(&self, path: &Path) -> ArtifactFile {
        call(|done| ArtifactFile::open(&self.world.io, path, true, done)).unwrap()
    }
}

fn allocation(choices: &[(&str, QuantOption)]) -> Allocation {
    Allocation::new(
        choices
            .iter()
            .map(|(name, option)| ((*name).to_owned(), *option))
            .collect(),
    )
}

/// The payload of `name` in the file at `path`.
fn payload(file: &ArtifactFile, name: &str) -> Vec<u8> {
    let entry = file.table().get(name).unwrap_or_else(|| panic!("no {name}"));
    let bytes = fs::read(file.path()).unwrap();
    bytes[entry.offset() as usize..(entry.offset() + entry.bytes()) as usize].to_vec()
}

#[test]
fn sources_list_the_layers_projections_and_the_drafters_apart() {
    let fixture = Fixture::new();
    let sources = fixture.open(&fixture.two()).unwrap();
    assert_eq!(sources.projections(), PROJECTIONS);
    assert_eq!(sources.drafter_projections(), [DRAFTER_PROJECTION]);
    assert_eq!(sources.options(), [OTHER, BASE]);
    fixture.world.finish();
}

#[test]
fn a_projection_with_companions_counts_all_its_bytes() {
    let fixture = Fixture::new();
    let sources = fixture.open(&fixture.two()).unwrap();
    let name = PROJECTIONS[0];
    let alone = sources.bytes(OTHER, name).unwrap();
    let with = sources.bytes(BASE, name).unwrap();
    // The companions: two fp16 vectors of 64 and one int32.
    assert_eq!(with - alone, 64 * 2 + 64 * 2 + 4 * 4);
    assert_eq!(sources.object_with_companions(BASE, name).unwrap().len(), 4);
    assert_eq!(sources.object_with_companions(OTHER, name).unwrap().len(), 1);
    fixture.world.finish();
}

#[test]
fn a_shared_object_that_differs_is_refused_by_name() {
    let fixture = Fixture::new();
    let mut artifacts = fixture.two();
    let other = TinyArtifact {
        shared_seed: 7,
        ..TinyArtifact::new(2, false)
    };
    artifacts.insert(OTHER, fixture.put("other.edrl", &other));
    let error = fixture.open(&artifacts).unwrap_err().to_string();
    assert!(
        error.contains("text/draft_head_token_ids differs between nvfp4 and trellis-3.0"),
        "{error}"
    );
    assert!(!PROJECTIONS.iter().any(|name| error.contains(name)), "{error}");
    fixture.world.finish();
}

#[test]
fn a_source_without_a_projection_is_refused_by_name() {
    let fixture = Fixture::new();
    let mut artifacts = fixture.two();
    let other = TinyArtifact {
        omitted: PROJECTIONS[2],
        ..TinyArtifact::new(2, false)
    };
    artifacts.insert(OTHER, fixture.put("other.edrl", &other));
    let error = fixture.open(&artifacts).unwrap_err().to_string();
    assert!(error.contains(PROJECTIONS[2]), "{error}");
    fixture.world.finish();
}

#[test]
fn a_source_for_the_base_is_required() {
    let fixture = Fixture::new();
    let mut artifacts = fixture.two();
    artifacts.remove(&BASE);
    assert!(fixture.open(&artifacts).is_err());
    fixture.world.finish();
}

#[test]
fn the_view_takes_chosen_objects_from_their_source_and_drops_the_replaced_companions() {
    let fixture = Fixture::new();
    let sources = fixture.open(&fixture.two()).unwrap();
    let chosen = PROJECTIONS[1];
    let composite = view(&sources, &allocation(&[(chosen, OTHER)]), None).unwrap();
    let names: Vec<&str> = composite.pieces.iter().map(|piece| piece.entry.name()).collect();
    assert!(!names.contains(&format!("{chosen}.suh").as_str()));
    assert!(names.contains(&format!("{}.suh", PROJECTIONS[0]).as_str()));
    let of = |name: &str| {
        composite
            .pieces
            .iter()
            .find(|piece| piece.entry.name() == name)
            .unwrap()
            .option
    };
    assert_eq!(of(chosen), OTHER);
    assert_eq!(of(PROJECTIONS[0]), BASE);
    // The base's order: the chosen object keeps its place.
    let at = |name: &str| names.iter().position(|each| *each == name).unwrap();
    assert!(at(PROJECTIONS[0]) < at(chosen) && at(chosen) < at(PROJECTIONS[2]));
    fixture.world.finish();
}

#[test]
fn a_view_taking_everything_from_one_source_holds_that_sources_objects_in_its_order() {
    let fixture = Fixture::new();
    let artifacts = fixture.two();
    let sources = fixture.open(&artifacts).unwrap();
    let every: Vec<(&str, QuantOption)> = PROJECTIONS.iter().map(|name| (*name, BASE)).collect();
    let pure = view(&sources, &allocation(&every), None).unwrap();
    let base = fixture.table(&artifacts[&BASE]);
    let held: Vec<(&str, u64, u64)> = pure
        .pieces
        .iter()
        .map(|piece| (piece.entry.name(), piece.entry.offset(), piece.entry.bytes()))
        .collect();
    let source: Vec<(&str, u64, u64)> = base
        .table()
        .entries()
        .iter()
        .map(|entry| (entry.name(), entry.offset(), entry.bytes()))
        .collect();
    assert_eq!(held, source);
    assert!(pure.pieces.iter().all(|piece| piece.option == BASE));
    fixture.world.finish();
}

#[test]
fn an_object_chosen_from_a_source_brings_its_companions() {
    let fixture = Fixture::new();
    let artifacts = BTreeMap::from([
        (BASE, fixture.put("base.edrl", &TinyArtifact::new(1, false))),
        (OTHER, fixture.put("other.edrl", &TinyArtifact::new(2, true))),
    ]);
    let sources = fixture.open(&artifacts).unwrap();
    let chosen = PROJECTIONS[0];
    let composite = view(&sources, &allocation(&[(chosen, OTHER)]), None).unwrap();
    let names: Vec<&str> = composite.pieces.iter().map(|piece| piece.entry.name()).collect();
    let at = names.iter().position(|name| *name == chosen).unwrap();
    // The object, then its companions, all from the source that was chosen.
    assert_eq!(
        names[at + 1..at + 4],
        [".suh", ".svh", ".segments"].map(|suffix| format!("{chosen}{suffix}"))
    );
    assert!(composite.pieces[at..at + 4].iter().all(|piece| piece.option == OTHER));
    assert_eq!(names.len(), 12 + 3);
    fixture.world.finish();
}

#[test]
fn a_projection_of_another_shape_is_refused_by_name() {
    let fixture = Fixture::new();
    let mut artifacts = fixture.two();
    let other = TinyArtifact {
        widened: PROJECTIONS[1],
        ..TinyArtifact::new(2, false)
    };
    artifacts.insert(OTHER, fixture.put("other.edrl", &other));
    let error = fixture.open(&artifacts).unwrap_err().to_string();
    assert!(error.contains(PROJECTIONS[1]) && error.contains("shape"), "{error}");
    fixture.world.finish();
}

#[test]
fn a_source_that_is_the_temporary_name_of_the_output_is_refused_and_left_alone() {
    let fixture = Fixture::new();
    let mut artifacts = fixture.two();
    let part = fixture.put("mixed.edrl.part", &TinyArtifact::new(1, true));
    artifacts.insert(BASE, part.clone());
    let before = fs::read(&part).unwrap();
    let sources = fixture.open(&artifacts).unwrap();
    let composite = view(&sources, &allocation(&[]), None).unwrap();
    let out = fixture.dir.path().join("mixed.edrl");
    let error = fixture.write(&composite, &out, 1 << 20).unwrap_err().to_string();
    assert!(error.contains("a source"), "{error}");
    assert_eq!(fs::read(&part).unwrap(), before);
    assert!(!out.exists());
    fixture.world.finish();
}

#[test]
fn a_source_that_supplies_nothing_and_is_the_temporary_name_is_refused_and_left_alone() {
    let fixture = Fixture::new();
    let mut artifacts = fixture.two();
    let part = fixture.put("mixed.edrl.part", &TinyArtifact::new(2, false));
    artifacts.insert(OTHER, part.clone());
    let before = fs::read(&part).unwrap();
    let sources = fixture.open(&artifacts).unwrap();
    let composite = view(&sources, &allocation(&[]), None).unwrap();
    let out = fixture.dir.path().join("mixed.edrl");
    let error = fixture.write(&composite, &out, 1 << 20).unwrap_err().to_string();
    assert!(error.contains("a source"), "{error}");
    assert_eq!(fs::read(&part).unwrap(), before);
    assert!(!out.exists());
    fixture.world.finish();
}

#[test]
fn an_output_that_is_a_source_is_refused_and_the_source_and_the_temporary_file_are_left_clean() {
    let fixture = Fixture::new();
    let artifacts = fixture.two();
    let base = artifacts[&BASE].clone();
    let before = fs::read(&base).unwrap();
    let sources = fixture.open(&artifacts).unwrap();
    let composite = view(&sources, &allocation(&[]), None).unwrap();
    let error = fixture.write(&composite, &base, 1 << 20).unwrap_err().to_string();
    assert!(error.contains("is a source"), "{error}");
    assert_eq!(fs::read(&base).unwrap(), before);
    assert!(!fixture.dir.path().join("base.edrl.part").exists());
    fixture.world.finish();
}

#[test]
fn an_output_that_a_running_engine_holds_is_refused_by_path_and_not_touched() {
    let fixture = Fixture::new();
    let sources = fixture.open(&fixture.two()).unwrap();
    let composite = view(&sources, &allocation(&[]), None).unwrap();
    let out = fixture.dir.path().join("served.edrl");
    fs::write(&out, b"served").unwrap();
    let guard = ArtifactGuard::hold(&out).unwrap();
    let error = fixture.write(&composite, &out, 1 << 20).unwrap_err().to_string();
    assert!(error.contains(&out.display().to_string()), "{error}");
    assert_eq!(fs::read(&out).unwrap(), b"served");
    assert!(!fixture.dir.path().join("served.edrl.part").exists());
    drop(guard);
    fixture.write(&composite, &out, 1 << 20).unwrap();
    fixture.world.finish();
}

#[test]
fn a_view_without_a_tokenizer_cannot_be_written() {
    let fixture = Fixture::new();
    let sources = fixture.open(&fixture.two()).unwrap();
    let mut composite = view(&sources, &allocation(&[]), None).unwrap();
    composite.tokenizer = None;
    let out = fixture.dir.path().join("out.edrl");
    let error = fixture.write(&composite, &out, 1 << 20).unwrap_err().to_string();
    assert!(error.contains("tokenizer"), "{error}");
    assert!(!out.exists() && !fixture.dir.path().join("out.edrl.part").exists());
    fixture.world.finish();
}

#[test]
fn a_choice_that_cannot_be_made_is_refused_naming_it() {
    let fixture = Fixture::new();
    let sources = fixture.open(&fixture.two()).unwrap();
    let refused = |choices: &[(&str, QuantOption)]| view(&sources, &allocation(choices), None).unwrap_err().to_string();
    assert!(refused(&[("text/layers/9/mlp/down", OTHER)]).contains("unknown object text/layers/9/mlp/down"));
    assert!(refused(&[("text/final_norm", OTHER)]).contains("text/final_norm is not a projection"));
    assert!(refused(&[(DRAFTER_PROJECTION, OTHER)]).contains("MTP layer"));
    assert!(refused(&[(PROJECTIONS[0], QuantOption::Trellis4_0)]).contains("no source for trellis-4.0"));
    fixture.world.finish();
}

#[test]
fn the_written_artifact_holds_each_payload_of_its_source_whatever_the_chunk() {
    let fixture = Fixture::new();
    let artifacts = fixture.two();
    let sources = fixture.open(&artifacts).unwrap();
    let chosen = [PROJECTIONS[1], PROJECTIONS[2]];
    let composite = view(
        &sources,
        &allocation(&[(chosen[0], OTHER), (chosen[1], OTHER)]),
        Some(2),
    )
    .unwrap();
    let small = fixture.dir.path().join("small.edrl");
    let large = fixture.dir.path().join("large.edrl");
    fixture.write(&composite, &small, 100).unwrap();
    fixture.write(&composite, &large, 1 << 20).unwrap();
    assert_eq!(fs::read(&small).unwrap(), fs::read(&large).unwrap());
    let written = fixture.table(&small);
    assert_eq!(written.metadata().draft_depth, Some(2));
    assert_eq!(written.table().entries().len(), composite.pieces.len());
    let base = fixture.table(&artifacts[&BASE]);
    let other = fixture.table(&artifacts[&OTHER]);
    for piece in &composite.pieces {
        let name = piece.entry.name();
        let source = if piece.option == OTHER { &other } else { &base };
        assert_eq!(payload(&written, name), payload(source, name), "{name}");
        assert_eq!(written.table().get(name).unwrap().offset() % 4096, 0, "{name}");
    }
    let tokenizer = written.tokenizer().unwrap();
    assert_eq!(tokenizer.text("chat_template.jinja"), Some(TOKENIZER.chat_template));
    let TokenizerTexts { special_tokens, .. } = TOKENIZER;
    assert_eq!(tokenizer.text("special_tokens.json"), Some(special_tokens));
    assert!(!small.with_extension("edrl.part").exists());
    fixture.world.finish();
}

#[test]
fn the_draft_depth_of_the_base_stays_unless_one_is_given() {
    let fixture = Fixture::new();
    let sources = fixture.open(&fixture.two()).unwrap();
    let none = view(&sources, &allocation(&[]), None).unwrap();
    assert_eq!(none.metadata.draft_depth, None);
    let some = view(&sources, &allocation(&[]), Some(3)).unwrap();
    assert_eq!(some.metadata.draft_depth, Some(3));
    fixture.world.finish();
}

#[test]
fn an_assembled_artifact_assembles_to_itself() {
    let fixture = Fixture::new();
    let artifacts = fixture.two();
    let sources = fixture.open(&artifacts).unwrap();
    let first = fixture.dir.path().join("first.edrl");
    fixture
        .write(
            &view(&sources, &allocation(&[(PROJECTIONS[0], OTHER)]), None).unwrap(),
            &first,
            1 << 20,
        )
        .unwrap();
    let again = BTreeMap::from([(BASE, first.clone())]);
    let sources = fixture.open(&again).unwrap();
    let second = fixture.dir.path().join("second.edrl");
    fixture
        .write(&view(&sources, &allocation(&[]), None).unwrap(), &second, 777)
        .unwrap();
    assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
    fixture.world.finish();
}

#[test]
fn a_failed_copy_leaves_no_file() {
    let fixture = Fixture::new();
    let artifacts = fixture.two();
    let sources = fixture.open(&artifacts).unwrap();
    let composite = view(&sources, &allocation(&[]), None).unwrap();
    // The source ends before its last payload does.
    let source = &artifacts[&BASE];
    let length = fs::metadata(source).unwrap().len();
    fs::OpenOptions::new()
        .write(true)
        .open(source)
        .unwrap()
        .set_len(length / 2)
        .unwrap();
    let out = fixture.dir.path().join("out.edrl");
    let error = fixture.write(&composite, &out, 1 << 20).unwrap_err().to_string();
    assert!(error.contains("copying"), "{error}");
    assert!(!out.exists());
    assert!(!fixture.dir.path().join("out.edrl.part").exists());
    fixture.world.finish();
}

#[test]
fn a_zero_copy_buffer_is_refused_before_anything_starts() {
    let fixture = Fixture::new();
    let sources = fixture.open(&fixture.two()).unwrap();
    let composite = view(&sources, &allocation(&[]), None).unwrap();
    let out = fixture.dir.path().join("out.edrl");
    assert!(matches!(
        fixture.write(&composite, &out, 0),
        Err(AssembleError::Invalid(_))
    ));
    assert!(!out.exists());
    fixture.world.finish();
}

fn words(text: &str) -> Vec<String> {
    text.split_whitespace().map(str::to_owned).collect()
}

#[test]
fn the_request_takes_its_flags_and_refuses_the_rest() {
    let good = "--allocation a.json --source nvfp4=n.edrl --source trellis-3.0=t.edrl --base trellis-3.0 --out o.edrl";
    let request = Request::parse(&words(good)).unwrap();
    assert_eq!(request.sources.len(), 2);
    assert_eq!((request.draft_depth, request.copy_buffer_mib), (None, None));
    let request = Request::parse(&words(&format!("{good} --draft-depth 2 --copy-buffer-mib 8"))).unwrap();
    assert_eq!((request.draft_depth, request.copy_buffer_mib), (Some(2), Some(8)));
    for bad in [
        format!("{good} --draft-depth 0"),
        format!("{good} --draft-depth {}", MAX_DRAFT_DEPTH + 1),
        format!("{good} --copy-buffer-mib 0"),
        format!("{good} --source nvfp4=again.edrl"),
        format!("{good} --frobnicate"),
        "--allocation a.json --source nvfp4=n.edrl --base trellis-3.0 --out o.edrl".to_owned(),
        "--allocation a.json --base nvfp4 --source nvfp4=n.edrl".to_owned(),
    ] {
        assert!(Request::parse(&words(&bad)).is_err(), "{bad}");
    }
}

#[test]
fn the_command_reads_the_allocation_and_writes_the_artifact() {
    let fixture = Fixture::new();
    let artifacts = fixture.two();
    let allocation = fixture.dir.path().join("allocation.json");
    fs::write(
        &allocation,
        format!("{{\"objects\":{{\"{}\":\"nvfp4\"}}}}", PROJECTIONS[3]),
    )
    .unwrap();
    let out = fixture.dir.path().join("mixed.edrl");
    let request = Request {
        allocation,
        sources: artifacts.clone(),
        base: BASE,
        out: out.clone(),
        draft_depth: Some(2),
        copy_buffer_mib: Some(1),
    };
    let assembled = call(|done| run(&fixture.world.io, &fixture.world.host, request, 100, done)).unwrap();
    let written = fixture.table(&out);
    let total: u64 = written.table().entries().iter().map(|entry| entry.bytes()).sum();
    assert_eq!(assembled.resident_bytes, total);
    assert_eq!(written.metadata().draft_depth, Some(2));
    let other = fixture.table(&artifacts[&OTHER]);
    assert_eq!(payload(&written, PROJECTIONS[3]), payload(&other, PROJECTIONS[3]));
    fixture.world.finish();
}

#[test]
fn a_stale_temporary_file_is_replaced() {
    let fixture = Fixture::new();
    let sources = fixture.open(&fixture.two()).unwrap();
    let composite = view(&sources, &allocation(&[]), None).unwrap();
    let out = fixture.dir.path().join("out.edrl");
    let reference = fixture.dir.path().join("reference.edrl");
    fixture.write(&composite, &reference, 1 << 20).unwrap();
    let stale = vec![0xAB; fs::read(&reference).unwrap().len() * 2];
    fs::write(fixture.dir.path().join("out.edrl.part"), stale).unwrap();
    fixture.write(&composite, &out, 1 << 20).unwrap();
    assert_eq!(fs::read(&out).unwrap(), fs::read(&reference).unwrap());
    fixture.world.finish();
}

#[cfg(unix)]
#[test]
fn an_output_name_that_cannot_be_opened_is_refused_before_any_file_is_made() {
    use std::os::unix::fs::symlink;
    let fixture = Fixture::new();
    let sources = fixture.open(&fixture.two()).unwrap();
    let composite = view(&sources, &allocation(&[]), None).unwrap();
    let out = fixture.dir.path().join("loop.edrl");
    symlink(&out, &out).unwrap();
    let error = fixture.write(&composite, &out, 1 << 20).unwrap_err().to_string();
    assert!(error.contains("cannot lock") && error.contains("loop.edrl"), "{error}");
    assert!(!fixture.dir.path().join("loop.edrl.part").exists());
    assert_eq!(fs::read_link(&out).unwrap(), out);
    fixture.world.finish();
}

#[test]
fn a_copy_buffer_that_does_not_fit_is_refused() {
    let fixture = Fixture::new();
    let allocation = fixture.dir.path().join("allocation.json");
    fs::write(&allocation, "{\"objects\":{}}").unwrap();
    let request = Request {
        allocation,
        sources: fixture.two(),
        base: BASE,
        out: fixture.dir.path().join("mixed.edrl"),
        draft_depth: None,
        copy_buffer_mib: Some(u64::MAX),
    };
    let error = call(|done| run(&fixture.world.io, &fixture.world.host, request, 100, done)).unwrap_err();
    assert!(error.to_string().contains("copy buffer"), "{error}");
    fixture.world.finish();
}
