//! The `quantization assemble` command, run on tiny synthetic source artifacts.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]

use std::fs;
use std::path::Path;

use euhedral_inference_edrl::tiny::{PROJECTIONS, TinyArtifact};
use euhedral_inference_edrl::{HEADER_BYTES, Header, TensorTable};
use euhedral_inference_quantize::{EXIT_OK, EXIT_RUN_FAILED, EXIT_USAGE, run};
use tempfile::tempdir;

fn assemble(args: &[&str]) -> euhedral_inference_quantize::Outcome {
    let mut line = vec!["assemble".to_owned()];
    line.extend(args.iter().map(|arg| (*arg).to_owned()));
    run(&line)
}

#[test]
fn a_bad_flag_is_a_usage_error_in_the_message_shape_of_the_original() {
    let missing = assemble(&["--allocation", "a.json", "--source", "nvfp4=n.edrl", "--out", "o.edrl"]);
    assert_eq!(
        (missing.code, missing.message.as_str()),
        (EXIT_USAGE, "--base is required\n")
    );
    let unknown = assemble(&["--frob", "1"]);
    assert_eq!(unknown.code, EXIT_USAGE);
    assert!(unknown.message.contains("unknown flag --frob"), "{}", unknown.message);
    let no_base_source = assemble(&[
        "--allocation",
        "a",
        "--source",
        "nvfp4=n",
        "--base",
        "trellis-3.0",
        "--out",
        "o",
    ]);
    assert_eq!(no_base_source.code, EXIT_USAGE);
    assert!(
        no_base_source.message.contains("no --source for the base"),
        "{}",
        no_base_source.message
    );
}

#[test]
fn a_missing_source_file_fails_the_run_and_writes_nothing() {
    let dir = tempdir().unwrap();
    let allocation = dir.path().join("allocation.json");
    fs::write(&allocation, "{\"objects\":{}}").unwrap();
    let out = dir.path().join("mixed.edrl");
    let outcome = assemble(&[
        "--allocation",
        allocation.to_str().unwrap(),
        "--source",
        &format!("nvfp4={}", dir.path().join("absent.edrl").display()),
        "--base",
        "nvfp4",
        "--out",
        out.to_str().unwrap(),
    ]);
    assert_eq!(outcome.code, EXIT_RUN_FAILED, "{}", outcome.message);
    assert!(outcome.message.starts_with("assemble failed: "), "{}", outcome.message);
    assert!(!out.exists());
}

fn payload(path: &Path, name: &str) -> Vec<u8> {
    let bytes = fs::read(path).unwrap();
    let header = Header::parse(bytes[..HEADER_BYTES].try_into().unwrap()).unwrap();
    let range = header.table_range();
    let table = TensorTable::parse(
        &bytes[range.start as usize..range.end as usize],
        &header,
        bytes.len() as u64,
    )
    .unwrap();
    let entry = table.entries().iter().find(|entry| entry.name() == name).unwrap();
    bytes[entry.offset() as usize..(entry.offset() + entry.bytes()) as usize].to_vec()
}

#[test]
fn the_command_writes_each_projection_from_the_source_the_allocation_names_and_the_rest_from_the_base() {
    let dir = tempdir().unwrap();
    let base = dir.path().join("base.edrl");
    let other = dir.path().join("other.edrl");
    fs::write(&base, TinyArtifact::new(1, true).bytes().unwrap()).unwrap();
    fs::write(&other, TinyArtifact::new(2, false).bytes().unwrap()).unwrap();
    let allocation = dir.path().join("allocation.json");
    fs::write(
        &allocation,
        format!("{{\"objects\":{{\"{}\":\"nvfp4\"}}}}", PROJECTIONS[3]),
    )
    .unwrap();
    let out = dir.path().join("mixed.edrl");
    let outcome = assemble(&[
        "--allocation",
        allocation.to_str().unwrap(),
        "--source",
        &format!("trellis-3.0={}", base.display()),
        "--source",
        &format!("nvfp4={}", other.display()),
        "--base",
        "trellis-3.0",
        "--out",
        out.to_str().unwrap(),
        "--copy-buffer-mib",
        "1",
    ]);
    assert_eq!((outcome.code, outcome.message.as_str()), (EXIT_OK, ""));
    assert_eq!(payload(&out, PROJECTIONS[3]), payload(&other, PROJECTIONS[3]));
    assert_eq!(payload(&out, PROJECTIONS[0]), payload(&base, PROJECTIONS[0]));
    assert!(!dir.path().join("mixed.edrl.part").exists());
}
