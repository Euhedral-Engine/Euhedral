//! What the checkpoint tests share: a hand-driven `AsyncIo` and a safetensors file made without the writer under test.

#![allow(
    dead_code,
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "a test: it waits for its own hand-off and stops on a failed unwrap"
)]

use std::fs;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::generics::LatticeSource;
use euhedral_inference_quantization::checkpoint::{CheckpointError, Done};
use euhedral_io::AsyncIo;
use serde_json::{Map, Value, json};

const LONGEST: Duration = Duration::from_secs(20);

/// A tensor of a test file: name, dtype, shape and bytes.
pub struct Spec {
    pub name: &'static str,
    pub dtype: &'static str,
    pub shape: Vec<u64>,
    pub bytes: Vec<u8>,
}

/// A safetensors file's bytes.
pub fn safetensors(tensors: &[Spec]) -> Vec<u8> {
    let mut header = Map::new();
    header.insert("__metadata__".to_owned(), json!({"format": "pt"}));
    let mut at = 0;
    for tensor in tensors {
        let end = at + tensor.bytes.len();
        header.insert(
            tensor.name.to_owned(),
            json!({"dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": [at, end]}),
        );
        at = end;
    }
    let json = serde_json::to_vec(&Value::Object(header)).unwrap();
    let mut file = (json.len() as u64).to_le_bytes().to_vec();
    file.extend_from_slice(&json);
    for tensor in tensors {
        file.extend_from_slice(&tensor.bytes);
    }
    file
}

/// BF16 values from `f32`s that BF16 holds exactly.
pub fn bf16(values: &[f32]) -> Vec<u8> {
    values
        .iter()
        .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
        .collect()
}

pub fn write(path: &Path, bytes: &[u8]) {
    fs::write(path, bytes).unwrap();
}

/// Runs `start` and drives `io` by hand until its callback has the result: reaping, then running each continuation's
/// frame where this thread stands, as a lattice would on a worker.
pub fn drive<T: Send + 'static>(
    io: &Arc<AsyncIo>,
    start: impl FnOnce(Done<T>) -> Result<(), CheckpointError>,
) -> Result<T, CheckpointError> {
    let slot: Arc<Mutex<Option<Result<T, CheckpointError>>>> = Arc::new(Mutex::new(None));
    let done: Done<T> = {
        let slot = Arc::clone(&slot);
        Arc::new(move |result| {
            let mut slot = slot.lock().unwrap();
            assert!(slot.is_none(), "a result was delivered twice");
            *slot = Some(result);
        })
    };
    start(done)?;
    let begun = Instant::now();
    loop {
        if let Some(result) = slot.lock().unwrap().take() {
            return result;
        }
        assert!(begun.elapsed() < LONGEST, "the operation never ended");
        io.reap();
        let mut frames = Vec::new();
        io.pull(&mut |frame| frames.push(frame), &mut |_| false, u64::MAX);
        for mut frame in frames {
            let _ = frame.execute();
            frame.do_finally();
        }
    }
}
