//! The safetensors writer: what it writes reads back, the file appears only whole, and a failure leaves nothing.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it reads and writes its fixture files and stops on a failed unwrap"
)]

mod checkpoint_support;

use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use checkpoint_support::{Spec, World, safetensors, write};
use euhedral_inference_quantization::checkpoint::CheckpointError;
use euhedral_inference_quantization::checkpoint::safetensors::Safetensors;
use euhedral_inference_quantization::checkpoint::writer::{SafetensorsWriter, Source, bytes_source, write_file};
use serde_json::Value;

fn finish(world: &World, writer: SafetensorsWriter) -> Result<(), CheckpointError> {
    world.call(|done| writer.finish(&world.io, &world.host, done))
}

/// Like `finish`, and also the number of times the callback ran, once the lattice has drained.
fn finish_counted(world: World, writer: SafetensorsWriter) -> (Result<(), CheckpointError>, u64) {
    let calls = Arc::new(AtomicU64::new(0));
    let counted = Arc::clone(&calls);
    let result = world.call(|done| {
        writer.finish(
            &world.io,
            &world.host,
            Arc::new(move |result| {
                counted.fetch_add(1, Ordering::SeqCst);
                done(result);
            }),
        )
    });
    world.finish();
    (result, calls.load(Ordering::SeqCst))
}

fn header_of(file: &[u8]) -> Value {
    let length = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    assert_eq!(length % 8, 0, "the header is padded to a multiple of 8");
    serde_json::from_slice(&file[8..8 + length]).unwrap()
}

fn open(world: &World, directory: &Path) -> Arc<Safetensors> {
    Arc::new(
        world
            .call(|done| Safetensors::open(&world.io, directory, done))
            .unwrap(),
    )
}

fn read_back(world: &World, directory: &Path, name: &str, bytes: usize) -> Vec<u8> {
    let store = open(world, directory);
    world.call(|done| store.read(name, 0, vec![0u8; bytes], done)).unwrap()
}

/// A source that makes element `i` of a 16-bit tensor `i * 31` and counts the chunks it was asked for.
fn counting(asked: &Arc<AtomicU64>, largest: usize) -> Source {
    let asked = Arc::clone(asked);
    Arc::new(move |offset, mut chunk, done| {
        assert!(chunk.len() <= largest, "a chunk is at most the buffer");
        asked.fetch_add(1, Ordering::Relaxed);
        for (index, pair) in chunk.as_chunks_mut::<2>().0.iter_mut().enumerate() {
            let element = offset / 2 + index as u64;
            *pair = ((element * 31) as u16).to_le_bytes();
        }
        done(Ok(chunk));
        Ok(())
    })
}

#[test]
fn what_is_written_reads_back_and_no_part_file_is_left() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let words: Vec<u8> = [1i16, -2, 3, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut writer = SafetensorsWriter::new(&path);
    writer.add("x.trellis", "I16", &[1, 1, 4], words.clone()).unwrap();
    writer
        .add("x.mul1", "I32", &[], 0x83DC_D12Du32.to_le_bytes().to_vec())
        .unwrap();
    assert_eq!(writer.data_bytes(), 12);
    finish(&world, writer).unwrap();
    assert!(!directory.path().join("model.safetensors.part").exists());
    let file = fs::read(&path).unwrap();
    let header = header_of(&file);
    assert_eq!(header["__metadata__"]["format"], "pt");
    assert_eq!(header["x.trellis"]["dtype"], "I16");
    assert_eq!(header["x.mul1"]["shape"], serde_json::json!([]));
    assert_eq!(header["x.mul1"]["data_offsets"], serde_json::json!([8, 12]));
    assert_eq!(read_back(&world, directory.path(), "x.trellis", 8), words);
    assert_eq!(
        read_back(&world, directory.path(), "x.mul1", 4),
        0x83DC_D12Du32.to_le_bytes()
    );
    world.finish();
}

#[test]
fn tensors_larger_than_the_buffer_are_pulled_in_chunks_at_their_offsets() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let elements = 100_000u64;
    let asked = Arc::new(AtomicU64::new(0));
    let mut writer = SafetensorsWriter::with_buffer(&path, 4096).unwrap();
    writer
        .add_source("big", "F16", &[elements], elements * 2, counting(&asked, 4096))
        .unwrap();
    writer.add("tail", "F32", &[1], 7u32.to_le_bytes().to_vec()).unwrap();
    finish(&world, writer).unwrap();

    assert_eq!(asked.load(Ordering::Relaxed), (elements * 2).div_ceil(4096));
    let big = read_back(&world, directory.path(), "big", (elements * 2) as usize);
    for element in 0..elements {
        let at = element as usize * 2;
        assert_eq!(
            u16::from_le_bytes([big[at], big[at + 1]]),
            (element * 31) as u16,
            "element {element}"
        );
    }
    assert_eq!(read_back(&world, directory.path(), "tail", 4), 7u32.to_le_bytes());
    world.finish();
}

#[test]
fn a_tensor_is_copied_from_a_checkpoint_chunk_by_chunk() {
    let world = World::start();
    let from = tempfile::tempdir().unwrap();
    let bytes: Vec<u8> = (0..10_000u32).map(|i| (i * 7 % 251) as u8).collect();
    let tensors = [Spec {
        name: "big",
        dtype: "U8",
        shape: vec![10_000],
        bytes: bytes.clone(),
    }];
    write(&from.path().join("model.safetensors"), &safetensors(&tensors));
    let store = open(&world, from.path());

    let to = tempfile::tempdir().unwrap();
    let path = to.path().join("model.safetensors");
    let mut writer = SafetensorsWriter::with_buffer(&path, 1024).unwrap();
    let tensor = store.tensor("big").unwrap();
    writer
        .add_source(
            "copy",
            &tensor.dtype,
            &tensor.shape,
            tensor.bytes,
            store.source("big").unwrap(),
        )
        .unwrap();
    assert!(store.source("nope").is_err());
    finish(&world, writer).unwrap();
    assert_eq!(read_back(&world, to.path(), "copy", 10_000), bytes);
    world.finish();
}

#[test]
fn inconsistent_tensors_are_refused_when_added() {
    let directory = tempfile::tempdir().unwrap();
    let mut writer = SafetensorsWriter::new(&directory.path().join("model.safetensors"));
    writer.add("a", "F16", &[2], vec![0; 4]).unwrap();
    let message = |result: Result<(), CheckpointError>| match result {
        Err(CheckpointError::Invalid(message)) => message,
        other => panic!("not refused: {other:?}"),
    };
    assert!(message(writer.add("a", "F16", &[2], vec![0; 4])).contains("duplicate"));
    assert!(message(writer.add("b", "F16", &[3], vec![0; 4])).contains("implies 6"));
    assert!(message(writer.add("c", "F8", &[2], vec![0; 4])).contains("dtype"));
    assert!(message(writer.add("d", "I64", &[u64::MAX, 2], vec![])).contains("too large"));
    assert!(message(writer.add("__metadata__", "U8", &[1], vec![0])).contains("__metadata__"));
    for size in [0, 12, 1001] {
        assert!(
            SafetensorsWriter::with_buffer(&directory.path().join("x"), size).is_err(),
            "{size}"
        );
    }
    // Refused tensors leave no trace: only `a` is in the header.
    let header = String::from_utf8(writer.header()).unwrap();
    assert_eq!(header.matches("data_offsets").count(), 1);
    assert!(header.contains("\"a\"") && !header.contains("\"b\"") && !header.contains("\"d\""));
}

#[test]
fn the_header_is_padded_on_its_encoded_length_and_names_are_escaped() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let names = [
        "t\u{e9}nsor\u{e9}",
        "quote\"back\\slash",
        "ctl\u{1}\n\t\u{1f}",
        "\u{4e2d}\u{6587}",
    ];
    let mut writer = SafetensorsWriter::new(&path);
    for name in names {
        writer.add(name, "F16", &[1], vec![0; 2]).unwrap();
    }
    let json = writer.header();
    assert_eq!(json.len() % 8, 0);
    // Short escapes where JSON has them, and upper-case hexadecimal for the other control characters.
    let text = String::from_utf8(json).unwrap();
    assert!(text.contains("ctl\\u0001\\n\\t\\u001F"), "{text}");
    finish(&world, writer).unwrap();
    let header = header_of(&fs::read(&path).unwrap());
    for name in names {
        assert_eq!(header[name]["dtype"], "F16", "{name:?}");
    }
    world.finish();
}

#[test]
fn a_failing_source_leaves_neither_a_part_file_nor_a_final_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let failures: [(&str, Source, &str); 4] = [
        (
            "reports a failure",
            Arc::new(|_, _, done| {
                done(Err(CheckpointError::Invalid("boom".to_owned())));
                Ok(())
            }),
            "cannot read tensor bad: boom",
        ),
        (
            "refuses to start",
            Arc::new(|_, _, _| Err(CheckpointError::Invalid("boom".to_owned()))),
            "cannot read tensor bad: boom",
        ),
        (
            "gives a short chunk",
            Arc::new(|_, chunk, done| {
                done(Ok(chunk[..1].to_vec()));
                Ok(())
            }),
            "the source of tensor bad gave 1 bytes for a chunk of 2",
        ),
        (
            "gives an oversized chunk",
            Arc::new(|_, mut chunk, done| {
                chunk.push(0);
                done(Ok(chunk));
                Ok(())
            }),
            "the source of tensor bad gave 3 bytes for a chunk of 2",
        ),
    ];
    for (what, source, message) in failures {
        let world = World::start();
        let mut writer = SafetensorsWriter::new(&path);
        writer.add("ok", "F16", &[1], vec![0; 2]).unwrap();
        writer.add_source("bad", "F16", &[1], 2, source).unwrap();
        let (result, calls) = finish_counted(world, writer);
        assert_eq!(result.unwrap_err().to_string(), message, "{what}");
        assert_eq!(calls, 1, "{what}: the callback runs once");
        assert!(!directory.path().join("model.safetensors.part").exists(), "{what}");
        assert!(!path.exists(), "{what}");
    }
}

#[test]
fn a_failure_after_the_data_was_written_removes_the_part_file_and_reports_once() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    // A target that is a directory with something in it: the data is written, synced and closed, and the rename fails.
    let path = directory.path().join("model.safetensors");
    fs::create_dir(&path).unwrap();
    fs::write(path.join("keep"), b"kept").unwrap();
    let mut writer = SafetensorsWriter::new(&path);
    writer.add("a", "U8", &[1], vec![1]).unwrap();
    let (result, calls) = finish_counted(world, writer);
    let error = result.unwrap_err();
    assert!(
        matches!(&error, CheckpointError::Io { context, .. } if context.contains("cannot rename")),
        "{error}"
    );
    assert_eq!(calls, 1);
    assert!(!directory.path().join("model.safetensors.part").exists());
    assert_eq!(fs::read(path.join("keep")).unwrap(), b"kept");
}

#[test]
fn an_existing_file_is_replaced_only_when_the_new_one_is_complete() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    fs::write(&path, b"old contents").unwrap();
    let mut failing = SafetensorsWriter::new(&path);
    let boom: Source = Arc::new(|_, _, done| {
        done(Err(CheckpointError::Invalid("boom".to_owned())));
        Ok(())
    });
    failing.add_source("bad", "U8", &[1], 1, boom).unwrap();
    assert!(finish(&world, failing).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"old contents");

    let mut writer = SafetensorsWriter::new(&path);
    writer.add("new", "U8", &[1], vec![9]).unwrap();
    finish(&world, writer).unwrap();
    assert_eq!(read_back(&world, directory.path(), "new", 1), [9]);
    world.finish();
}

#[test]
fn a_file_with_no_tensors_and_a_missing_directory() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let empty = directory.path().join("empty.safetensors");
    finish(&world, SafetensorsWriter::new(&empty)).unwrap();
    let file = fs::read(&empty).unwrap();
    assert_eq!(header_of(&file)["__metadata__"]["format"], "pt");
    assert_eq!(
        file.len(),
        8 + u64::from_le_bytes(file[..8].try_into().unwrap()) as usize
    );

    let nowhere = directory.path().join("missing").join("model.safetensors");
    let mut writer = SafetensorsWriter::new(&nowhere);
    writer.add("a", "U8", &[1], vec![1]).unwrap();
    assert!(matches!(finish(&world, writer), Err(CheckpointError::Io { .. })));
    world.finish();
}

#[test]
fn a_closed_host_lake_refuses_before_anything_is_written() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    world.host.complete_gracefully();
    let mut writer = SafetensorsWriter::new(&path);
    writer.add("a", "U8", &[1], vec![1]).unwrap();
    let refused = writer.finish(&world.io, &world.host, Arc::new(|_| panic!("nothing was started")));
    assert!(matches!(refused, Err(CheckpointError::LakeClosed)));
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 0);
    world.finish();
}

// Mutants caught: a file written in place (a stop leaves a part of it); bytes cut at a chunk boundary of the buffer; an
// error that names no file or says "safetensors" of a file that is not one.
#[test]
fn write_file_writes_whole_and_names_the_file_when_it_fails() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let data: Vec<u8> = (0..3 * 64 + 5).map(|at| at as u8).collect();
    let path = directory.path().join("notes.txt");
    let size = data.len() as u64;
    world
        .call(|done| {
            write_file(
                &path,
                size,
                bytes_source(data.clone()),
                64,
                &world.io,
                &world.host,
                done,
            )
        })
        .unwrap();
    assert_eq!(fs::read(&path).unwrap(), data);
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1, "no part is left");

    let nowhere = directory.path().join("missing").join("notes.txt");
    let failed = world.call(|done| {
        write_file(
            &nowhere,
            size,
            bytes_source(data.clone()),
            64,
            &world.io,
            &world.host,
            done,
        )
    });
    let message = failed.unwrap_err().to_string();
    assert!(message.contains("notes.txt"), "{message}");
    assert!(!message.contains("safetensors"), "{message}");

    // A source that fails halfway: the file named is the one being written, and nothing is left.
    let target = directory.path().join("half.txt");
    let source: Source = Arc::new(|offset, chunk, done| {
        if offset == 0 {
            done(Ok(chunk));
        } else {
            done(Err(CheckpointError::Invalid("cut".to_owned())));
        }
        Ok(())
    });
    let failed = world.call(|done| write_file(&target, 128, source, 64, &world.io, &world.host, done));
    assert!(failed.unwrap_err().to_string().contains("cut"));
    assert!(!target.exists());
    assert_eq!(fs::read_dir(directory.path()).unwrap().count(), 1, "no part is left");
    world.finish();
}
