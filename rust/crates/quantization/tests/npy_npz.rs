//! The array writer against the fixture files NumPy made.

#![allow(
    clippy::disallowed_methods,
    clippy::unwrap_used,
    reason = "a test reads its fixture files and stops on a bad one"
)]

use std::fs;
use std::path::{Path, PathBuf};

use euhedral_inference_quantization::npy::{Dtype, Npy, encode};

fn fixtures() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../quantization/src/test/resources/gptq-fixture")
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
