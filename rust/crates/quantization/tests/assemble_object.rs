//! Reading one object of a source into host memory: its payload, and the checked companions of a trellis object.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it stops on a failed unwrap and prepares its files directly"
)]

mod checkpoint_support;

use std::fs;
use std::sync::Arc;
use std::time::Duration;

use checkpoint_support::World;
use euhedral_inference_edrl::tiny::{TINY_ALIGNMENT, metadata};
use euhedral_inference_edrl::{
    ArtifactPlan, HEADER_BYTES, ObjectSpec, TensorDataType, WeightFormat, WeightLayout, encode_metadata,
};
use euhedral_inference_engine::Handoff;
use euhedral_inference_quantization::assemble::{
    ArtifactFile, AssembleError, COMPANION_SUFFIXES, Done, ObjectBytes, read_object,
};

const TRELLIS: &str = "text/layers/0/mlp/down";
const PLAIN: &str = "text/layers/0/norm";

fn spec(name: &str, shape: &[u64], format: WeightFormat, layout: WeightLayout, bytes: u64) -> ObjectSpec {
    let dtype = match format {
        WeightFormat::I32 => TensorDataType::Int32,
        WeightFormat::Fp16 => TensorDataType::Fp16,
        _ => TensorDataType::Bf16,
    };
    ObjectSpec {
        name: name.to_owned(),
        shape: shape.to_vec(),
        dtype,
        format,
        layout,
        bytes,
    }
}

fn ends(values: &[i32]) -> Vec<u8> {
    values.iter().flat_map(|end| end.to_le_bytes()).collect()
}

/// An artifact file of a trellis object of 256 rows (tiles `[8, 16, 16]`) with companions whose `suh` has
/// `suh_rows` rows and whose segments end at `segment_ends`, filled with 1, 2, 3 and the segments' own bytes, and of a
/// plain object filled with 9. The file holds no tokenizer.
fn file(suh_rows: u64, segment_ends: &[i32]) -> Vec<u8> {
    let contiguous = WeightLayout::ContiguousLeV1;
    let segments = ends(segment_ends);
    let objects = [
        (
            spec(
                TRELLIS,
                &[8, 16, 16],
                WeightFormat::Trellis,
                WeightLayout::TrellisTile16V1,
                8 * 16 * 16 * 2,
            ),
            vec![1; 8 * 16 * 16 * 2],
        ),
        (
            spec(
                &format!("{TRELLIS}{}", COMPANION_SUFFIXES[0]),
                &[suh_rows],
                WeightFormat::Fp16,
                contiguous,
                suh_rows * 2,
            ),
            vec![2; suh_rows as usize * 2],
        ),
        (
            spec(
                &format!("{TRELLIS}{}", COMPANION_SUFFIXES[1]),
                &[256],
                WeightFormat::Fp16,
                contiguous,
                512,
            ),
            vec![3; 512],
        ),
        (
            spec(
                &format!("{TRELLIS}{}", COMPANION_SUFFIXES[2]),
                &[segment_ends.len() as u64],
                WeightFormat::I32,
                contiguous,
                segments.len() as u64,
            ),
            segments,
        ),
        (spec(PLAIN, &[16], WeightFormat::Fp16, contiguous, 32), vec![9; 32]),
    ];
    let specs: Vec<ObjectSpec> = objects.iter().map(|(spec, _)| spec.clone()).collect();
    let metadata = encode_metadata(&metadata()).unwrap();
    let plan = ArtifactPlan::new(metadata.len() as u64, &specs, TINY_ALIGNMENT).unwrap();
    let table = plan.table().unwrap();
    let mut file = vec![0; plan.content_end() as usize];
    file[..HEADER_BYTES].copy_from_slice(&plan.header().encode());
    let at = plan.header().metadata_offset() as usize;
    file[at..at + metadata.len()].copy_from_slice(&metadata);
    let at = plan.header().tensor_table_offset() as usize;
    file[at..at + table.len()].copy_from_slice(&table);
    for (entry, (_, payload)) in plan.entries().iter().zip(&objects) {
        let at = entry.offset() as usize;
        file[at..at + payload.len()].copy_from_slice(payload);
    }
    file
}

/// Opens the file and reads `name`, waiting outside the lattice for the result.
fn read(bytes: &[u8], name: &str) -> Result<ObjectBytes, AssembleError> {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("source.edrl");
    fs::write(&path, bytes).unwrap();
    let opened = Handoff::new();
    let sender = Arc::clone(&opened);
    ArtifactFile::open(&world.io, &path, false, Arc::new(move |artifact| sender.send(artifact))).unwrap();
    let artifact = opened
        .wait(Duration::from_secs(20))
        .expect("the artifact never opened")
        .unwrap();
    let handoff = Handoff::new();
    let sender = Arc::clone(&handoff);
    let done: Done<ObjectBytes> = Arc::new(move |result| sender.send(result));
    let started = read_object(&world.io, &artifact, name, done);
    let result = match started {
        Ok(()) => handoff
            .wait(Duration::from_secs(20))
            .expect("the object was never read"),
        Err(error) => Err(error),
    };
    world.finish();
    result
}

#[test]
fn a_trellis_object_comes_back_with_its_companions_in_suffix_order_and_its_segment_ends() {
    let object = read(&file(2, &[128, 256]), TRELLIS).unwrap();
    assert_eq!(object.bytes, vec![1; 8 * 16 * 16 * 2]);
    let trellis = object.trellis.unwrap();
    assert_eq!(trellis.suh, vec![2; 4]);
    assert_eq!(trellis.svh, vec![3; 512]);
    assert_eq!(trellis.segments, ends(&[128, 256]));
    assert_eq!(&*trellis.segment_ends, [128, 256]);
    assert_eq!(trellis.suh_rows, 2);
}

#[test]
fn an_object_that_is_not_trellis_comes_back_alone() {
    let object = read(&file(2, &[128, 256]), PLAIN).unwrap();
    assert_eq!(object.bytes, vec![9; 32]);
    assert!(object.trellis.is_none());
}

#[test]
fn segments_that_do_not_match_the_rows_of_suh_are_refused_by_name() {
    let error = read(&file(3, &[128, 256]), TRELLIS).unwrap_err().to_string();
    assert!(error.contains(&format!("{TRELLIS}.segments")), "{error}");
    assert!(error.contains("segment count 2 differs from suh rows 3"), "{error}");
}

#[test]
fn segments_that_do_not_end_at_the_rows_of_the_object_are_refused() {
    let error = read(&file(2, &[128, 384]), TRELLIS).unwrap_err().to_string();
    assert!(error.contains("last segment end 384 is not n=256"), "{error}");
}

#[test]
fn an_object_the_source_lacks_is_refused_before_anything_is_read() {
    let error = read(&file(2, &[128, 256]), "text/layers/9/none")
        .unwrap_err()
        .to_string();
    assert!(error.contains("has no object text/layers/9/none"), "{error}");
}
