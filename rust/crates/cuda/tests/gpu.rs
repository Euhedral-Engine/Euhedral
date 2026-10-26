//! Tests that run on a GPU. They are `#[ignore]`d and run locally in a memory-capped scope:
//! `systemd-run --user --scope -p MemoryMax=8G cargo test -p euhedral-inference-cuda -- --ignored`.
//!
//! The library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY` in the environment, or
//! `gradle.properties`); a test skips with a message when it is absent.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: a test synchronizes a lane and copies synchronously to check results"
)]

use std::iter;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use euhedral_graph::{DeviceQueue, LaneId, LanePool, Marker};
use euhedral_inference_cuda::{
    CudaError, CudaLane, CudaLibrary, DeviceBuffer, DevicePtr, GraphRecorder, HostWeightsArena, PinnedBuffer,
    Submission, SubmitMode, ops,
};

fn library() -> Option<Arc<CudaLibrary>> {
    let path = test_support::require_property("euhedral.cuda.library")?;
    Some(CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads"))
}

fn lane(library: &Arc<CudaLibrary>, id: u16, ring: usize) -> Arc<CudaLane> {
    CudaLane::new(library, LaneId(id), ring).expect("a lane")
}

fn submit(lane: &CudaLane) -> Submission<'_> {
    Submission::begin(lane, SubmitMode::default()).expect("a submission")
}

/// A pinned flag word the device waits on, set to 0.
fn flag(library: &Arc<CudaLibrary>) -> (PinnedBuffer, DevicePtr) {
    let buffer = PinnedBuffer::alloc(library, 64).expect("pinned flag");
    buffer.word(0).store(0, Ordering::Release);
    let device = buffer.device_address().expect("the flag's device address");
    (buffer, device)
}

fn download(buffer: &DeviceBuffer) -> Vec<u8> {
    let mut host = vec![0u8; buffer.bytes()];
    buffer.copy_to_host(0, &mut host).expect("download");
    host
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn alloc_copy_roundtrip() {
    let Some(library) = library() else { return };
    let device = library.device();
    assert!(device.sm_count > 0, "{device}");
    assert!(device.compute_capability.0 >= 5, "{device}");
    assert!(library.free_bytes().expect("free bytes") <= device.total_bytes);

    let pattern: Vec<u8> = (0..1 << 20).map(|index| (index * 31 % 251) as u8).collect();
    let buffer = DeviceBuffer::alloc(&library, pattern.len()).expect("device allocation");
    buffer.copy_from_host(0, &pattern).expect("upload");
    assert_eq!(download(&buffer), pattern);

    let second = DeviceBuffer::alloc(&library, 4096).expect("second allocation");
    assert!(second.generation() > buffer.generation(), "serials only grow");
    assert!(
        buffer.copy_from_host(buffer.bytes() - 1, &[1, 2]).is_err(),
        "a copy past the end is refused"
    );

    let mut pinned = PinnedBuffer::alloc(&library, 4096).expect("pinned allocation");
    pinned.as_mut_slice().fill(7);
    second.copy_from_host(0, pinned.as_slice()).expect("upload from pinned");
    assert!(download(&second).iter().all(|&byte| byte == 7));

    let mut arena = HostWeightsArena::alloc(&library, 3 << 20).expect("host weights arena");
    arena.as_mut_slice()[..4].copy_from_slice(&[1, 2, 3, 4]);
    assert_eq!(arena.as_ptr() as usize % (2 << 20), 0, "2 MiB aligned");
    assert_ne!(arena.device_address().expect("mapped").0, 0);
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn lane_marker_reached_after_signal() {
    let Some(library) = library() else { return };
    let lane = lane(&library, 0, 4);
    let pool = LanePool::new(vec![Arc::clone(&lane) as Arc<dyn DeviceQueue>]);
    let (gate, gate_device) = flag(&library);

    // The lane holds at the gate, so the marker cannot have been reached yet.
    let marker = {
        let submission = submit(&lane);
        ops::wait_host_flag(&submission, gate_device, 1).expect("wait");
        submission.finish_signalled().expect("finish")
    };
    assert_eq!(marker.value, 1);
    assert!(!pool.reached(marker), "the lane is held at the gate");
    assert_eq!(lane.completed(), 0);

    gate.word(0).store(1, Ordering::Release);
    lane.synchronize().expect("synchronize");
    assert!(pool.reached(marker));
    assert_eq!(lane.completed(), marker.value);

    // An explicit signal moves the word to any later value.
    {
        let submission = submit(&lane);
        submission.signal(0x1234).expect("signal");
    }
    lane.synchronize().expect("synchronize");
    assert_eq!(lane.completed(), 0x1234);
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn cross_lane_wait_orders_kernels() {
    let Some(library) = library() else { return };
    let (first, second) = (lane(&library, 0, 4), lane(&library, 1, 4));
    let bytes = 8 << 20;
    let source = DeviceBuffer::alloc(&library, bytes).expect("source");
    let destination = DeviceBuffer::alloc(&library, bytes).expect("destination");
    source.copy_from_host(0, &vec![0xFF; bytes]).expect("fill the source");
    destination
        .copy_from_host(0, &vec![0xAA; bytes])
        .expect("fill the destination");
    let (gate, gate_device) = flag(&library);

    // The first lane zeroes the source only once the gate opens; the second lane copies the source after the zeroing.
    // Were the second lane not ordered behind it, it would copy the 0xFF fill immediately.
    let marker = {
        let submission = submit(&first);
        ops::wait_host_flag(&submission, gate_device, 1).expect("wait");
        ops::zero_device_memory(&submission, source.ptr(), bytes as u64).expect("zero");
        submission.finish().expect("finish")
    };
    {
        let submission = submit(&second);
        submission.wait(marker).expect("wait for the marker");
        ops::copy_device_to_device(&submission, destination.ptr(), source.ptr(), bytes as u64).expect("copy");
    }
    gate.word(0).store(1, Ordering::Release);
    first.synchronize().expect("first lane");
    second.synchronize().expect("second lane");
    assert!(
        download(&destination).iter().all(|&byte| byte == 0),
        "the copy ran after the zeroing"
    );
}

/// BF16 1.0, 2.0, 3.0 and 4.0.
const ONE: u16 = 0x3F80;
const TWO: u16 = 0x4000;
const THREE: u16 = 0x4040;
const FOUR: u16 = 0x4080;

fn fill(buffer: &DeviceBuffer, value: u16) {
    let bytes: Vec<u8> = iter::repeat_n(value.to_le_bytes(), buffer.bytes() / 2)
        .flatten()
        .collect();
    buffer.copy_from_host(0, &bytes).expect("fill");
}

fn all_equal(buffer: &DeviceBuffer, value: u16) -> bool {
    let bytes = download(buffer);
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .all(|pair| u16::from_le_bytes(*pair) == value)
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn graph_capture_replays_same_result() {
    let Some(library) = library() else { return };
    let lane = lane(&library, 0, 4);
    let width = 256u32;
    let bytes = width as usize * 2;
    let (left, right) = (
        DeviceBuffer::alloc(&library, bytes).unwrap(),
        DeviceBuffer::alloc(&library, bytes).unwrap(),
    );
    let (sum, copy) = (
        DeviceBuffer::alloc(&library, bytes).unwrap(),
        DeviceBuffer::alloc(&library, bytes).unwrap(),
    );
    fill(&left, ONE);
    fill(&right, TWO);

    let add = |submission: &Submission<'_>| {
        ops::residual_add_bf16(submission, left.ptr(), right.ptr(), sum.ptr(), 1, width).expect("add");
        ops::copy_device_to_device(submission, copy.ptr(), sum.ptr(), bytes as u64).expect("copy");
    };

    // The first launch of a kernel loads its module, which a capture cannot do.
    {
        let submission = submit(&lane);
        add(&submission);
    }
    lane.synchronize().expect("warm-up");
    assert!(all_equal(&sum, THREE));

    let recorder = GraphRecorder::begin(&lane).expect("capture begins");
    {
        let submission = submit(&lane);
        add(&submission);
        let other = Marker {
            lane: LaneId(1),
            value: 1,
        };
        assert!(
            matches!(submission.wait(other), Err(CudaError::Capturing(_))),
            "no cross-lane waits in a capture"
        );
    }
    assert!(
        matches!(submit(&lane).finish(), Err(CudaError::Capturing(_))),
        "no markers in a capture"
    );
    assert!(
        GraphRecorder::begin(&lane).is_err(),
        "a lane captures one graph at a time"
    );
    let graph = recorder.end().expect("capture ends");
    lane.synchronize().expect("nothing ran during the capture");

    for replay in 0..2 {
        fill(&sum, 0);
        fill(&copy, 0);
        {
            let submission = submit(&lane);
            graph.launch(&submission).expect("replay");
        }
        lane.synchronize().expect("replay finishes");
        assert!(all_equal(&sum, THREE), "replay {replay} recomputes the sum");
        assert!(all_equal(&copy, THREE), "replay {replay} repeats the copy");
    }

    // A replay reads the buffers' current contents.
    fill(&left, TWO);
    {
        let submission = submit(&lane);
        graph.launch(&submission).expect("replay");
    }
    lane.synchronize().expect("replay finishes");
    assert!(all_equal(&sum, FOUR));
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn submission_clears_selection_on_drop() {
    let Some(library) = library() else { return };
    let (first, second) = (lane(&library, 0, 2), lane(&library, 1, 2));

    let held = Submission::begin(
        &first,
        SubmitMode {
            pdl: true,
            row_exact: true,
        },
    )
    .expect("first submission");
    assert!(
        Submission::begin(&second, SubmitMode::default()).is_err(),
        "a thread holds one selected stream at a time"
    );
    // The refused attempt must not have cleared the live selection: work still queues on the first lane.
    let buffer = DeviceBuffer::alloc(&library, 4096).expect("buffer");
    ops::zero_device_memory(&held, buffer.ptr(), 4096).expect("the first submission still works");
    drop(held);

    // Dropped: the stream is clear, so another lane can select.
    let other = Submission::begin(&second, SubmitMode::default()).expect("the selection was cleared on drop");
    ops::zero_device_memory(&other, buffer.ptr(), 4096).expect("work on the second lane");
    drop(other);
    first.synchronize().expect("first lane");
    second.synchronize().expect("second lane");
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn stream_ordered_buffers_follow_their_lane() {
    let Some(library) = library() else { return };
    let lane = lane(&library, 0, 2);
    let submission = submit(&lane);
    let buffer = DeviceBuffer::alloc_async(&submission, 1 << 16).expect("stream-ordered allocation");
    ops::zero_device_memory(&submission, buffer.ptr(), 1 << 16).expect("zero");
    buffer.free_async(&submission).expect("stream-ordered free");
    drop(submission);
    lane.synchronize().expect("lane");
    assert!(
        CudaLane::new(&library, LaneId(0), 2).is_err(),
        "a lane id exists once per library"
    );
    assert!(
        CudaLane::new(&library, LaneId(64), 2).is_err(),
        "a pool holds at most 64 lanes"
    );
}

#[test]
#[ignore = "needs a GPU and the native library"]
#[cfg(feature = "test-oracles")]
fn exact_numerics_switch_returns_the_previous_selection() {
    let Some(library) = library() else { return };
    assert!(!library.exact_numerics_for_tests(true), "relaxed is the default");
    assert!(library.exact_numerics_for_tests(false));
}
