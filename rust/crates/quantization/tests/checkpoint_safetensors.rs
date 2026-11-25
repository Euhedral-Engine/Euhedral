//! The checkpoint reader on files made by hand: single files and shards, ranges, many reads in flight, and refusals.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it writes its fixture files and stops on a failed unwrap"
)]

mod checkpoint_support;

use std::fs;
use std::path::Path;
use std::sync::Arc;

use checkpoint_support::{Spec, drive, safetensors, write};
use euhedral_inference_quantization::checkpoint::CheckpointError;
use euhedral_inference_quantization::checkpoint::safetensors::{ReadRange, Safetensors};
use euhedral_io::AsyncIo;

fn open(io: &Arc<AsyncIo>, directory: &Path) -> Result<Arc<Safetensors>, CheckpointError> {
    drive(io, |done| Safetensors::open(io, directory, done)).map(Arc::new)
}

fn read(
    io: &Arc<AsyncIo>,
    store: &Arc<Safetensors>,
    name: &str,
    skip: u64,
    len: usize,
) -> Result<Vec<u8>, CheckpointError> {
    drive(io, |done| store.read(name, skip, vec![0u8; len], done))
}

fn single() -> Vec<u8> {
    safetensors(&[
        Spec {
            name: "a",
            dtype: "BF16",
            shape: vec![3, 2],
            bytes: (1u16..=6).flat_map(u16::to_le_bytes).collect(),
        },
        Spec {
            name: "b",
            dtype: "F32",
            shape: vec![2],
            bytes: [1.5f32, -2.0].iter().flat_map(|value| value.to_le_bytes()).collect(),
        },
        Spec {
            name: "scalar",
            dtype: "I32",
            shape: vec![],
            bytes: 7i32.to_le_bytes().to_vec(),
        },
    ])
}

#[test]
fn a_single_file_checkpoint_reads_whole_tensors_and_row_ranges() {
    let directory = tempfile::tempdir().unwrap();
    write(&directory.path().join("model.safetensors"), &single());
    let io = AsyncIo::open().unwrap();
    let store = open(&io, directory.path()).unwrap();

    let a = store.tensor("a").unwrap();
    assert_eq!(
        (a.dtype.as_str(), a.shape.as_slice(), a.bytes, a.elements()),
        ("BF16", &[3, 2][..], 12, Some(6))
    );
    assert_eq!(store.tensor("scalar").unwrap().elements(), Some(1));
    assert!(store.contains("b"));
    assert!(!store.contains("c"));
    let mut names: Vec<_> = store.names().collect();
    names.sort_unstable();
    assert_eq!(names, ["a", "b", "scalar"]);

    // Rows 1..2 of `a`: skip one row of two BF16 values.
    let rows = read(&io, &store, "a", 4, 8).unwrap();
    assert_eq!(
        rows,
        [3u16, 4, 5, 6].iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<_>>()
    );

    let b = drive(&io, |done| store.read_tensor("b", vec![0u8; 8], done)).unwrap();
    assert_eq!(f32::from_le_bytes(b[4..8].try_into().unwrap()), -2.0);
    let scalar = read(&io, &store, "scalar", 0, 4).unwrap();
    assert_eq!(i32::from_le_bytes(scalar.try_into().unwrap()), 7);
}

#[test]
fn a_sharded_checkpoint_follows_its_index_and_finds_tensors_in_every_shard() {
    let directory = tempfile::tempdir().unwrap();
    let one = safetensors(&[Spec {
        name: "x",
        dtype: "U8",
        shape: vec![3],
        bytes: vec![1, 2, 3],
    }]);
    let two = safetensors(&[Spec {
        name: "y",
        dtype: "U8",
        shape: vec![2],
        bytes: vec![8, 9],
    }]);
    write(&directory.path().join("model-00002.safetensors"), &two);
    write(&directory.path().join("model-00001.safetensors"), &one);
    write(
        &directory.path().join("model.safetensors.index.json"),
        br#"{"metadata":{},"weight_map":{"x":"model-00001.safetensors","y":"model-00002.safetensors"}}"#,
    );
    let io = AsyncIo::open().unwrap();
    let store = open(&io, directory.path()).unwrap();
    assert_eq!(read(&io, &store, "x", 1, 2).unwrap(), [2, 3]);
    assert_eq!(read(&io, &store, "y", 0, 2).unwrap(), [8, 9]);
    assert_eq!(store.shard_of(store.tensor("y").unwrap()), "model-00002.safetensors");
}

#[test]
fn many_ranges_in_flight_come_back_in_order() {
    let directory = tempfile::tempdir().unwrap();
    let bytes: Vec<u8> = (0..4096u32).map(|i| (i * 7 % 251) as u8).collect();
    write(
        &directory.path().join("model.safetensors"),
        &safetensors(&[Spec {
            name: "t",
            dtype: "U8",
            shape: vec![4096],
            bytes: bytes.clone(),
        }]),
    );
    let io = AsyncIo::open().unwrap();
    let store = open(&io, directory.path()).unwrap();
    // 64 ranges of 16 bytes, requested back to front.
    let ranges = |_| {
        (0..64u64)
            .rev()
            .map(|piece| ReadRange {
                tensor: "t".to_owned(),
                skip: piece * 64,
                memory: vec![0u8; 16],
            })
            .collect::<Vec<_>>()
    };
    let memories = drive(&io, |done| store.read_ranges(ranges(()), done)).unwrap();
    assert_eq!(memories.len(), 64);
    for (index, memory) in memories.iter().enumerate() {
        let piece = 63 - index;
        assert_eq!(memory[..], bytes[piece * 64..piece * 64 + 16], "range {index}");
    }
}

#[test]
fn a_request_outside_its_tensor_or_for_nothing_is_refused_before_anything_is_read() {
    let directory = tempfile::tempdir().unwrap();
    write(&directory.path().join("model.safetensors"), &single());
    let io = AsyncIo::open().unwrap();
    let store = open(&io, directory.path()).unwrap();
    assert!(matches!(read(&io, &store, "nope", 0, 4), Err(CheckpointError::MissingTensor(name)) if name == "nope"));
    assert!(matches!(read(&io, &store, "a", 8, 8), Err(CheckpointError::Invalid(_))));
    assert!(matches!(
        read(&io, &store, "a", u64::MAX, 8),
        Err(CheckpointError::Invalid(_))
    ));
    assert!(matches!(read(&io, &store, "a", 0, 0), Err(CheckpointError::Invalid(_))));
    let wrong = drive(&io, |done| store.read_tensor("a", vec![0u8; 8], done));
    assert!(matches!(wrong, Err(CheckpointError::Invalid(message)) if message.contains("holds 12 bytes, not 8")));
    // One bad range among good ones: nothing is submitted.
    let ranges = vec![
        ReadRange {
            tensor: "a".to_owned(),
            skip: 0,
            memory: vec![0u8; 4],
        },
        ReadRange {
            tensor: "a".to_owned(),
            skip: 10,
            memory: vec![0u8; 4],
        },
    ];
    assert!(matches!(
        drive(&io, |done| store.read_ranges(ranges, done)),
        Err(CheckpointError::Invalid(_))
    ));
    let none = Vec::<ReadRange<Vec<u8>>>::new();
    assert!(drive(&io, |done| store.read_ranges(none, done)).is_err());
}

#[test]
fn a_shard_with_a_bad_header_is_refused() {
    let io = AsyncIo::open().unwrap();
    let cases: [(&str, Vec<u8>); 4] = [
        ("zero header size", 0u64.to_le_bytes().to_vec()),
        ("negative header size", (-1i64).to_le_bytes().to_vec()),
        ("header beyond a gibibyte", ((1u64 << 30) + 1).to_le_bytes().to_vec()),
        ("truncated header", {
            let mut file = single();
            file.truncate(20);
            file
        }),
    ];
    for (what, file) in cases {
        let directory = tempfile::tempdir().unwrap();
        write(&directory.path().join("model.safetensors"), &file);
        assert!(open(&io, directory.path()).is_err(), "{what}");
    }
    let directory = tempfile::tempdir().unwrap();
    let broken = br#"{"t":{"dtype":"U8","shape":[2],"data_offsets":[5,1]}}"#;
    let mut file = (broken.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(broken);
    write(&directory.path().join("model.safetensors"), &file);
    assert!(
        matches!(open(&io, directory.path()), Err(CheckpointError::Format(_))),
        "an end before its start"
    );
}

#[test]
fn a_missing_checkpoint_and_a_missing_shard_are_io_errors() {
    let io = AsyncIo::open().unwrap();
    let directory = tempfile::tempdir().unwrap();
    assert!(matches!(open(&io, directory.path()), Err(CheckpointError::Io { .. })));
    write(
        &directory.path().join("model.safetensors.index.json"),
        br#"{"weight_map":{"x":"gone.safetensors"}}"#,
    );
    assert!(matches!(open(&io, directory.path()), Err(CheckpointError::Io { .. })));
    fs::write(directory.path().join("model.safetensors.index.json"), b"{}").unwrap();
    assert!(matches!(open(&io, directory.path()), Err(CheckpointError::Format(_))));
}

#[test]
fn a_file_that_shrinks_after_it_was_opened_fails_the_read() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let file = single();
    write(&path, &file);
    let io = AsyncIo::open().unwrap();
    let store = open(&io, directory.path()).unwrap();
    write(&path, &file[..file.len() - 6]);
    assert!(read(&io, &store, "scalar", 0, 4).is_err());
    assert_eq!(read(&io, &store, "a", 0, 12).unwrap().len(), 12);
}

/// A shard whose header is `json` followed by `data` bytes.
fn shard(json: &str, data: usize) -> Vec<u8> {
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(json.as_bytes());
    file.resize(file.len() + data, 0);
    file
}

#[test]
fn what_the_header_claims_beyond_the_file_is_refused() {
    let io = AsyncIo::open().unwrap();
    let past = r#"{"t":{"dtype":"U8","shape":[4],"data_offsets":[0,4]}}"#;
    let overflow = r#"{"t":{"dtype":"U8","shape":[5],"data_offsets":[18446744073709551610,18446744073709551615]}}"#;
    let mut too_large = shard(past, 4);
    too_large[..8].copy_from_slice(&1000u64.to_le_bytes());
    let cases = [
        ("a header larger than the file", too_large, "does not fit"),
        ("a tensor past the end", shard(past, 2), "past the end"),
        ("offsets that overflow", shard(overflow, 5), "overflow"),
    ];
    for (what, file, message) in cases {
        let directory = tempfile::tempdir().unwrap();
        write(&directory.path().join("model.safetensors"), &file);
        match open(&io, directory.path()) {
            Err(CheckpointError::Format(found)) => assert!(found.contains(message), "{what}: {found}"),
            other => panic!("{what}: not refused: {other:?}"),
        }
    }
    // The same header with its data present opens.
    let directory = tempfile::tempdir().unwrap();
    write(&directory.path().join("model.safetensors"), &shard(past, 4));
    assert!(open(&io, directory.path()).is_ok());
}

#[test]
fn an_element_count_that_does_not_fit_is_none() {
    let io = AsyncIo::open().unwrap();
    let json = r#"{"t":{"dtype":"U8","shape":[4294967296,4294967296],"data_offsets":[0,0]},"s":{"dtype":"U8","shape":[3,5],"data_offsets":[0,0]}}"#;
    let directory = tempfile::tempdir().unwrap();
    write(&directory.path().join("model.safetensors"), &shard(json, 0));
    let store = open(&io, directory.path()).unwrap();
    assert_eq!(store.tensor("t").unwrap().elements(), None);
    assert_eq!(store.tensor("s").unwrap().elements(), Some(15));
}
