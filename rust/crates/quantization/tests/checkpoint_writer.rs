//! The safetensors writer: what it writes reads back, the file appears only whole, and a failure leaves nothing.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it reads and writes its fixture files and stops on a failed unwrap"
)]

mod support;

use std::fs;
use std::io::Error as IoError;
use std::path::Path;
use std::sync::Arc;

use euhedral_inference_quantization::checkpoint::CheckpointError;
use euhedral_inference_quantization::checkpoint::safetensors::Safetensors;
use euhedral_inference_quantization::checkpoint::writer::SafetensorsWriter;
use euhedral_io::AsyncIo;
use serde_json::Value;
use support::drive;

fn finish(io: &Arc<AsyncIo>, writer: SafetensorsWriter) -> Result<(), CheckpointError> {
    drive(io, |done| writer.finish(io, done))
}

fn header_of(file: &[u8]) -> Value {
    let length = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
    assert_eq!(length % 8, 0, "the header is padded to a multiple of 8");
    serde_json::from_slice(&file[8..8 + length]).unwrap()
}

fn read_back(io: &Arc<AsyncIo>, directory: &Path, name: &str, bytes: usize) -> Vec<u8> {
    let store = Arc::new(drive(io, |done| Safetensors::open(io, directory, done)).unwrap());
    drive(io, |done| store.read(io, name, 0, vec![0u8; bytes], done)).unwrap()
}

#[test]
fn what_is_written_reads_back_and_no_part_file_is_left() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let io = AsyncIo::open().unwrap();
    let words: Vec<u8> = [1i16, -2, 3, 4].iter().flat_map(|v| v.to_le_bytes()).collect();
    let mut writer = SafetensorsWriter::new(&path);
    writer.add("x.trellis", "I16", &[1, 1, 4], words.clone()).unwrap();
    writer.add("x.mul1", "I32", &[], 0x83DC_D12Du32.to_le_bytes().to_vec()).unwrap();
    assert_eq!(writer.data_bytes(), 12);
    finish(&io, writer).unwrap();

    assert!(!directory.path().join("model.safetensors.part").exists());
    let file = fs::read(&path).unwrap();
    let header = header_of(&file);
    assert_eq!(header["__metadata__"]["format"], "pt");
    assert_eq!(header["x.trellis"]["dtype"], "I16");
    assert_eq!(header["x.mul1"]["shape"], serde_json::json!([]));
    assert_eq!(header["x.mul1"]["data_offsets"], serde_json::json!([8, 12]));
    assert_eq!(read_back(&io, directory.path(), "x.trellis", 8), words);
    assert_eq!(read_back(&io, directory.path(), "x.mul1", 4), 0x83DC_D12Du32.to_le_bytes());
}

#[test]
fn tensors_larger_than_the_buffer_are_pulled_in_chunks_at_their_offsets() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let io = AsyncIo::open().unwrap();
    let elements = 100_000u64;
    let mut writer = SafetensorsWriter::with_buffer(&path, 4096).unwrap();
    let asked = Arc::new(std::sync::atomic::AtomicU64::new(0));
    let seen = Arc::clone(&asked);
    writer
        .add_source(
            "big",
            "F16",
            &[elements],
            elements * 2,
            Box::new(move |offset, destination| {
                assert!(destination.len() <= 4096, "a chunk is at most the buffer");
                seen.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                for (index, pair) in destination.chunks_exact_mut(2).enumerate() {
                    let element = offset / 2 + index as u64;
                    pair.copy_from_slice(&((element * 31) as u16).to_le_bytes());
                }
                Ok(())
            }),
        )
        .unwrap();
    writer.add("tail", "F32", &[1], 7u32.to_le_bytes().to_vec()).unwrap();
    finish(&io, writer).unwrap();

    assert_eq!(asked.load(std::sync::atomic::Ordering::Relaxed), (elements * 2).div_ceil(4096));
    let big = read_back(&io, directory.path(), "big", (elements * 2) as usize);
    for element in [0u64, 1, 2047, 2048, 50_000, elements - 1] {
        let at = element as usize * 2;
        assert_eq!(u16::from_le_bytes([big[at], big[at + 1]]), (element * 31) as u16, "element {element}");
    }
    assert_eq!(read_back(&io, directory.path(), "tail", 4), 7u32.to_le_bytes());
}

#[test]
fn inconsistent_tensors_are_refused_when_added() {
    let directory = tempfile::tempdir().unwrap();
    let mut writer = SafetensorsWriter::new(&directory.path().join("model.safetensors"));
    writer.add("a", "F16", &[2], vec![0; 4]).unwrap();
    assert!(matches!(writer.add("a", "F16", &[2], vec![0; 4]), Err(CheckpointError::Invalid(m)) if m.contains("duplicate")));
    assert!(matches!(writer.add("b", "F16", &[3], vec![0; 4]), Err(CheckpointError::Invalid(m)) if m.contains("implies 6")));
    assert!(matches!(writer.add("c", "F8", &[2], vec![0; 4]), Err(CheckpointError::Invalid(m)) if m.contains("dtype")));
    assert!(matches!(writer.add("d", "I64", &[u64::MAX, 2], vec![]), Err(CheckpointError::Invalid(_))));
    assert!(SafetensorsWriter::with_buffer(&directory.path().join("x"), 0).is_err());
    // Refused tensors leave no trace: only `a` is in the header.
    let header = String::from_utf8(writer.header()).unwrap();
    assert!(header.contains("\"a\"") && !header.contains("\"b\"") && !header.contains("\"d\""));
}

#[test]
fn the_header_is_padded_on_its_encoded_length_and_names_are_escaped() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let io = AsyncIo::open().unwrap();
    let names = ["t\u{e9}nsor\u{e9}", "quote\"back\\slash", "ctl\u{1}\n\t\u{1f}", "\u{4e2d}\u{6587}"];
    let mut writer = SafetensorsWriter::new(&path);
    for name in names {
        writer.add(name, "F16", &[1], vec![0; 2]).unwrap();
    }
    let json = writer.header();
    assert_eq!(json.len() % 8, 0);
    assert!(json.ends_with(b"}") || json.ends_with(b" "));
    // Control characters are written as Jackson writes them: short escapes, and upper-case hex otherwise.
    let text = String::from_utf8(json).unwrap();
    assert!(text.contains("ctl\\u0001\\n\\t\\u001F"), "{text}");
    finish(&io, writer).unwrap();
    let header = header_of(&fs::read(&path).unwrap());
    for name in names {
        assert_eq!(header[name]["dtype"], "F16", "{name:?}");
    }
}

#[test]
fn a_failing_source_leaves_neither_a_part_file_nor_a_final_file() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    let io = AsyncIo::open().unwrap();
    let mut writer = SafetensorsWriter::new(&path);
    writer.add("ok", "F16", &[1], vec![0; 2]).unwrap();
    writer
        .add_source("bad", "F16", &[1], 2, Box::new(|_, _| Err(IoError::other("boom"))))
        .unwrap();
    let error = finish(&io, writer).unwrap_err();
    assert!(error.to_string().contains("bad"), "{error}");
    assert!(!directory.path().join("model.safetensors.part").exists());
    assert!(!path.exists());
}

#[test]
fn an_existing_file_is_replaced_only_when_the_new_one_is_complete() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("model.safetensors");
    fs::write(&path, b"old contents").unwrap();
    let io = AsyncIo::open().unwrap();
    let mut failing = SafetensorsWriter::new(&path);
    failing.add_source("bad", "U8", &[1], 1, Box::new(|_, _| Err(IoError::other("boom")))).unwrap();
    assert!(finish(&io, failing).is_err());
    assert_eq!(fs::read(&path).unwrap(), b"old contents");

    let mut writer = SafetensorsWriter::new(&path);
    writer.add("new", "U8", &[1], vec![9]).unwrap();
    finish(&io, writer).unwrap();
    assert_eq!(read_back(&io, directory.path(), "new", 1), [9]);
}

#[test]
fn a_file_with_no_tensors_and_a_missing_directory() {
    let directory = tempfile::tempdir().unwrap();
    let io = AsyncIo::open().unwrap();
    let empty = directory.path().join("empty.safetensors");
    finish(&io, SafetensorsWriter::new(&empty)).unwrap();
    let file = fs::read(&empty).unwrap();
    assert_eq!(header_of(&file)["__metadata__"]["format"], "pt");
    assert_eq!(file.len(), 8 + u64::from_le_bytes(file[..8].try_into().unwrap()) as usize);

    let nowhere = directory.path().join("missing").join("model.safetensors");
    let mut writer = SafetensorsWriter::new(&nowhere);
    writer.add("a", "U8", &[1], vec![1]).unwrap();
    assert!(matches!(finish(&io, writer), Err(CheckpointError::Io { .. })));
}
