//! A call that returns `Err` started nothing: its continuation is never called, on an `AsyncIo` that has already
//! completed and so refuses every submission.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::disallowed_methods,
    reason = "a test: it writes its fixture files and stops on a failed unwrap"
)]

mod checkpoint_support;

use std::collections::BTreeMap;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use checkpoint_support::{Spec, World, drive, safetensors, write};
use euhedral_core::generics::LatticeSource;
use euhedral_inference_quantization::checkpoint::safetensors::{ReadRange, Safetensors};
use euhedral_inference_quantization::checkpoint::writer::{SafetensorsWriter, bytes_source, write_file};
use euhedral_inference_quantization::checkpoint::{CheckpointError, Done, overlay};
use euhedral_io::AsyncIo;

/// A continuation that counts its calls.
fn counting() -> (Arc<AtomicUsize>, Done<()>) {
    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    (
        calls,
        Arc::new(move |_| {
            counted.fetch_add(1, Ordering::SeqCst);
        }),
    )
}

fn closed_io() -> Arc<AsyncIo> {
    let io = AsyncIo::open().unwrap();
    io.complete();
    io
}

// Mutant caught: a submission that fails and leaves its continuation armed, so the drop calls it with `LakeClosed`.
#[test]
fn a_writer_that_the_engine_refuses_returns_err_and_never_calls_its_continuation() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let io = closed_io();
    let (calls, then) = counting();
    let mut writer = SafetensorsWriter::new(&directory.path().join("model.safetensors"));
    writer.add("a", "U8", &[1], vec![1]).unwrap();
    assert!(writer.finish(&io, &world.host, then).is_err());
    let (calls_file, then) = counting();
    let refused = write_file(
        &directory.path().join("file"),
        1,
        bytes_source(vec![1]),
        8,
        &io,
        &world.host,
        then,
    );
    assert!(refused.is_err());
    assert_eq!(
        (calls.load(Ordering::SeqCst), calls_file.load(Ordering::SeqCst)),
        (0, 0)
    );
    world.finish();
}

// Mutants caught: a refused open or index read that calls its continuation; a refused ranged read whose gather is
// dropped armed.
#[test]
fn a_reader_that_the_engine_refuses_returns_err_and_never_calls_its_continuation() {
    let directory = tempfile::tempdir().unwrap();
    write(
        &directory.path().join("model.safetensors"),
        &safetensors(&[Spec {
            name: "a",
            dtype: "U8",
            shape: vec![2],
            bytes: vec![1, 2],
        }]),
    );
    let io = AsyncIo::open().unwrap();
    let store = Arc::new(drive(&io, |done| Safetensors::open(&io, directory.path(), done)).unwrap());
    io.complete();

    let calls = Arc::new(AtomicUsize::new(0));
    let counted = Arc::clone(&calls);
    let read = store.read_ranges(
        vec![ReadRange {
            tensor: "a".to_owned(),
            skip: 0,
            memory: vec![0u8; 2],
        }],
        Arc::new(move |_| {
            counted.fetch_add(1, Ordering::SeqCst);
        }),
    );
    assert!(read.is_err());

    let counted = Arc::clone(&calls);
    let opened = Safetensors::open(
        &io,
        directory.path(),
        Arc::new(move |_| {
            counted.fetch_add(1, Ordering::SeqCst);
        }),
    );
    assert!(opened.is_err());
    assert_eq!(calls.load(Ordering::SeqCst), 0);
}

// Mutant caught: an overlay whose refused first read leaves the overlay armed in its cell.
#[test]
fn an_overlay_that_the_engine_refuses_returns_err_and_never_calls_its_continuation() {
    let world = World::start();
    let directory = tempfile::tempdir().unwrap();
    let (calls, then) = counting();
    let refused = overlay::write(
        &closed_io(),
        &world.host,
        directory.path(),
        &directory.path().join("target"),
        BTreeMap::new(),
        then,
    );
    assert!(matches!(refused, Err(CheckpointError::Io { .. })), "{refused:?}");
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    world.finish();
}
