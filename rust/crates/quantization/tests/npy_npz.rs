//! The array formats against the files NumPy and the Java writers made.

#![allow(
    clippy::disallowed_methods,
    clippy::unwrap_used,
    reason = "a test reads its fixture files and stops on a bad one"
)]

use std::fs;
use std::path::{Path, PathBuf};

use euhedral_inference_quantization::npy::{Dtype, Npy, encode};
use euhedral_inference_quantization::npz::{Npz, NpzWriter};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../quantization/src/test/resources/gptq-fixture")
}

fn data(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(name)
}

/// Writes the array a file holds back out: the bytes must be the file's own.
#[test]
fn fixture_arrays_are_written_back_byte_for_byte() {
    let mut seen = Vec::new();
    for entry in fs::read_dir(fixtures()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_none_or(|extension| extension != "npy") {
            continue;
        }
        let bytes = fs::read(&path).unwrap();
        let array = Npy::parse(&bytes).unwrap();
        let written = encode(array.dtype, &array.shape, array.data).unwrap();
        assert_eq!(written, bytes, "{}", path.display());
        if !seen.contains(&array.dtype) {
            seen.push(array.dtype);
        }
    }
    for dtype in [Dtype::I1, Dtype::U1, Dtype::I2, Dtype::F2, Dtype::F4] {
        assert!(seen.contains(&dtype), "no fixture of {dtype:?}");
    }
}

/// An archive Java wrote: its members, read and written again by this writer, are the same arrays in the same order
/// with the same checksums. The container bytes differ, because Java stamps each member with the time it was written.
#[test]
fn java_archives_are_rewritten_with_the_same_members() {
    for name in ["nvfp4.npz", "q3.npz"] {
        let bytes = fs::read(data(name)).unwrap();
        let java = Npz::parse(&bytes).unwrap();
        let mut writer = NpzWriter::new(Vec::new());
        for entry in java.entries() {
            let member = entry.name.strip_suffix(".npy").unwrap();
            let array = Npy::parse(entry.bytes).unwrap();
            if matches!(array.dtype, Dtype::Unicode(_)) {
                writer.add_text(member, &array.text().unwrap()).unwrap();
            } else {
                writer.add_array(member, array.dtype, &array.shape, array.data).unwrap();
            }
        }
        let rewritten = writer.finish().unwrap();
        let ours = Npz::parse(&rewritten).unwrap();
        assert_eq!(ours.entries(), java.entries(), "{name}");
        assert_eq!(
            rewritten.len(),
            bytes.len(),
            "{name}: the container has the same layout"
        );
    }
}

#[test]
fn the_java_state_reads_back_as_its_fields() {
    let bytes = fs::read(data("nvfp4.npz")).unwrap();
    let archive = Npz::parse(&bytes).unwrap();
    assert_eq!(archive.array("format_name").unwrap().text().unwrap(), "NVFP4");
    assert_eq!(archive.array("layout").unwrap().text().unwrap(), "row-split");
    assert_eq!(archive.array("shape").unwrap().to_vec::<i64>().unwrap(), [3, 32]);
    assert_eq!(
        archive.array("global_scale").unwrap().to_vec::<f32>().unwrap(),
        [0.15625]
    );
    assert_eq!(archive.array("table").unwrap().data.len(), 16);
    let codes = archive.array("codes").unwrap();
    assert_eq!(codes.shape, [3, 32]);
    assert_eq!(codes.data[1], 10);
}
