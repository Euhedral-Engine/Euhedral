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
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::LatticeSource;
use euhedral_graph::Lake;
use euhedral_inference_engine::{Handoff, lattice_config};
use euhedral_inference_quantization::checkpoint::{CheckpointError, Done};
use euhedral_inference_settings::{ModelSettings, Settings};
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

/// Starts the call `start` on an `AsyncIo` driven by hand, once for each of its steps, and in each run drops that
/// step's frame unrun, as an engine that is closing does. Every run must end its call exactly once, with
/// `CheckpointError::LakeClosed`; the first run that drops nothing (the call has fewer steps) must end once with
/// success. Returns the number of steps the call has.
pub fn every_step_dropped_unrun<T: Send + 'static>(
    start: impl Fn(&Arc<AsyncIo>, Done<T>) -> Result<(), CheckpointError>,
) -> usize {
    let closed = CheckpointError::LakeClosed.to_string();
    for dropped in 0.. {
        let io = AsyncIo::open().unwrap();
        let ended: Arc<Handoff<Result<(), String>>> = Handoff::new();
        let calls = Arc::new(AtomicUsize::new(0));
        let (sink, counted) = (Arc::clone(&ended), Arc::clone(&calls));
        let done: Done<T> = Arc::new(move |result| {
            counted.fetch_add(1, Ordering::SeqCst);
            sink.send(result.map(|_| ()).map_err(|error| error.to_string()));
        });
        start(&io, done).unwrap();
        let begun = Instant::now();
        let mut seen = 0;
        let result = loop {
            if let Some(result) = ended.take() {
                break result;
            }
            assert!(
                begun.elapsed() < LONGEST,
                "the call never ended after step {dropped} was dropped unrun"
            );
            io.reap();
            let mut frames = Vec::new();
            io.pull(&mut |frame| frames.push(frame), &mut |_| false, u64::MAX);
            for mut frame in frames {
                seen += 1;
                if seen - 1 == dropped {
                    drop(frame);
                    continue;
                }
                let _ = frame.execute();
                frame.do_finally();
            }
        };
        assert_eq!(
            calls.load(Ordering::SeqCst),
            1,
            "step {dropped} dropped unrun: the call ends once"
        );
        if seen <= dropped {
            assert_eq!(result, Ok(()), "a call with {seen} steps ends with its result");
            io.complete();
            return dropped;
        }
        assert_eq!(result, Err(closed.clone()), "step {dropped} dropped unrun");
        io.complete();
    }
    unreachable!("the loop ends when a call has fewer steps than the one dropped")
}

/// A lattice that drives an `AsyncIo`, and a lake on it: what the calls that run durability steps need.
pub struct World {
    lattice: Arc<ControlPlaneLattice>,
    pub io: Arc<AsyncIo>,
    pub host: Arc<Lake>,
}

impl World {
    pub fn start() -> World {
        World::with_sinks(2)
    }

    /// A world whose lake has `sinks` ingest sinks: with one, the frames offered to it are taken in the order they
    /// were offered.
    pub fn with_sinks(sinks: usize) -> World {
        let settings = Settings::with_model(ModelSettings::default());
        let lattice = ControlPlaneLattice::new(lattice_config(&settings)).unwrap();
        let io = AsyncIo::open().unwrap();
        lattice.add_upstream(Arc::clone(&io) as Arc<dyn LatticeSource>).unwrap();
        let host = Lake::new(Arc::clone(&lattice), NonZeroUsize::new(sinks).unwrap(), LONGEST);
        World { lattice, io, host }
    }

    /// A world whose `AsyncIo` no lattice drives: a call that reads or writes through it goes no further than the
    /// steps [`World::drive_until`] runs, so a test holds the call in flight at a step and looks at what it holds.
    pub fn driven() -> World {
        let settings = Settings::with_model(ModelSettings::default());
        let lattice = ControlPlaneLattice::new(lattice_config(&settings)).unwrap();
        let io = AsyncIo::open().unwrap();
        let host = Lake::new(Arc::clone(&lattice), NonZeroUsize::new(2).unwrap(), LONGEST);
        World { lattice, io, host }
    }

    /// Runs the steps of the hand-driven `io`, where this thread stands, until `ended` has its value.
    pub fn drive_until<T>(&self, ended: &Arc<Handoff<T>>) -> T {
        let begun = Instant::now();
        loop {
            if let Some(value) = ended.take() {
                return value;
            }
            assert!(begun.elapsed() < LONGEST, "the call never ended");
            self.io.reap();
            let mut frames = Vec::new();
            self.io.pull(&mut |frame| frames.push(frame), &mut |_| false, u64::MAX);
            for mut frame in frames {
                let _ = frame.execute();
                frame.do_finally();
            }
        }
    }

    /// Starts a call and waits, outside the lattice, for the value its callback is given.
    pub fn call<T: Send + 'static>(
        &self,
        start: impl FnOnce(Done<T>) -> Result<(), CheckpointError>,
    ) -> Result<T, CheckpointError> {
        let handoff = Handoff::new();
        let sender = Arc::clone(&handoff);
        start(Arc::new(move |result| sender.send(result)))?;
        handoff.wait(LONGEST).expect("the callback never ran")
    }

    pub fn finish(self) {
        self.host.complete_gracefully();
        self.io.complete();
        self.lattice.close();
    }
}
