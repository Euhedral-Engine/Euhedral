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
    PinnedPool, RetireList, Submission, SubmitMode, ops,
};

/// The loaded library. The tests run in parallel on one device: none calls CUDA's allocator, frees or synchronizes
/// while a lane of its own is parked on a host flag, because a synchronizing call waits for every lane on the device.
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
    // SAFETY: the tests touch the word only through its atomic and keep the buffer alive beside the address.
    let device = unsafe { buffer.device_address() }.expect("the flag's device address");
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
    // SAFETY: no kernel reads the arena.
    assert_ne!(unsafe { arena.device_address() }.expect("mapped").get(), 0);
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
        let submission = recorder
            .submission(SubmitMode::default())
            .expect("a capture submission");
        add(&submission);
        let other = Marker {
            lane: LaneId(1),
            value: 1,
        };
        assert!(
            matches!(submission.wait(other), Err(CudaError::Capturing(_))),
            "no cross-lane waits in a capture"
        );
        assert!(
            matches!(submission.signal(1), Err(CudaError::Capturing(_))),
            "no completion signals in a capture"
        );
        assert!(
            matches!(copy.copy_from_host(0, &[0]), Err(CudaError::Capturing(_))),
            "no blocking copies in a capture"
        );
        let mut one = [0u8];
        assert!(
            matches!(
                ops::copy_device_to_host(&submission, &mut one, copy.ptr()),
                Err(CudaError::Capturing(_))
            ),
            "no blocking copies in a capture"
        );
    }
    assert!(
        matches!(lane.synchronize(), Err(CudaError::Capturing(_))),
        "no synchronization in a capture"
    );
    assert!(
        matches!(
            Submission::begin(&lane, SubmitMode::default()),
            Err(CudaError::LaneBusy(_))
        ),
        "the capture holds the lane"
    );
    assert!(
        matches!(GraphRecorder::begin(&lane), Err(CudaError::LaneBusy(_))),
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
    buffer.free_async(&submission, &[]).expect("stream-ordered free");
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

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_second_submission_on_a_live_lane_is_busy() {
    let Some(library) = library() else { return };
    let lane = lane(&library, 0, 2);
    let held = submit(&lane);
    assert!(matches!(Submission::begin(&lane, SubmitMode::default()), Err(CudaError::LaneBusy(id)) if id == LaneId(0)));
    assert!(matches!(GraphRecorder::begin(&lane), Err(CudaError::LaneBusy(_))));
    drop(held);
    // Given back on drop, and after a refused attempt too.
    drop(submit(&lane));
    drop(GraphRecorder::begin(&lane).expect("free again"));
    drop(submit(&lane));
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn waits_on_stale_and_future_markers_are_refused() {
    let Some(library) = library() else { return };
    let (first, second) = (lane(&library, 0, 2), lane(&library, 1, 2));
    let mut markers = Vec::new();
    for _ in 0..3 {
        markers.push(submit(&first).finish().expect("finish"));
    }
    let waiter = submit(&second);
    assert!(
        matches!(waiter.wait(markers[0]), Err(CudaError::StaleMarker { .. })),
        "marker 1 shares marker 3's event"
    );
    waiter.wait(markers[1]).expect("a marker inside the ring");
    waiter.wait(markers[2]).expect("the newest marker");
    let future = Marker {
        lane: LaneId(0),
        value: 4,
    };
    assert!(matches!(waiter.wait(future), Err(CudaError::FutureMarker { .. })));
    drop(waiter);
    first.synchronize().expect("first");
    second.synchronize().expect("second");
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn pinned_pool_reuses_released_buffers_even_past_its_limit() {
    let Some(library) = library() else { return };
    let pool = PinnedPool::new(&library, 1 << 20);
    let first = pool.acquire(10_000).expect("a pooled buffer");
    assert_eq!(first.len(), 10_000);
    assert_eq!(first.buffer().len(), 16384, "the size class");
    let address = first.buffer().as_ptr();
    drop(first);
    assert_eq!(pool.held_bytes(), 16384);
    let second = pool.acquire(12_000).expect("the same class");
    assert_eq!(second.buffer().as_ptr(), address, "a released buffer is reused");
    assert_eq!(pool.held_bytes(), 0);
    drop(second);

    // Past the limit: the buffers stay in their classes and are reused; only trim frees them.
    let tight_over = PinnedPool::new(&library, 8192);
    let (a, b) = (
        tight_over.acquire(5000).expect("a"),
        tight_over.acquire(5000).expect("b"),
    );
    let addresses_over = [a.buffer().as_ptr(), b.buffer().as_ptr()];
    drop((a, b));
    assert_eq!(tight_over.over_limit_bytes(), 8192);
    let (c, d) = (
        tight_over.acquire(5000).expect("c"),
        tight_over.acquire(5000).expect("d"),
    );
    let mut reused = [c.buffer().as_ptr(), d.buffer().as_ptr()];
    reused.sort();
    let mut expected = addresses_over;
    expected.sort();
    assert_eq!(
        reused, expected,
        "buffers released past the limit are reused by the next acquires"
    );
    drop((c, d));
    assert_eq!(tight_over.trim(), 1, "trim frees down to the limit");
    assert_eq!(tight_over.held_bytes(), 8192);
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn retired_buffers_are_released_after_their_marker_without_blocking() {
    let Some(library) = library() else { return };
    let (lane, reaper) = (lane(&library, 0, 2), lane(&library, 1, 2));
    let lanes = LanePool::new(vec![
        Arc::clone(&lane) as Arc<dyn DeviceQueue>,
        Arc::clone(&reaper) as _,
    ]);
    let pool = PinnedPool::new(&library, 1 << 20);
    let list = RetireList::new();
    let (gate, gate_device) = flag(&library);

    // Everything is allocated before the lane parks on its gate: nothing below calls into CUDA's allocator.
    let pooled = pool.acquire(4096).expect("pooled buffer");
    let device = {
        let submission = submit(&lane);
        DeviceBuffer::alloc_async(&submission, 1 << 16).expect("stream-ordered buffer")
    };
    let plain = DeviceBuffer::alloc(&library, 4096).expect("plain buffer");
    let marker = {
        let submission = submit(&lane);
        ops::wait_host_flag(&submission, gate_device, 1).expect("wait");
        ops::zero_device_memory(&submission, device.ptr(), 1 << 16).expect("zero");
        submission.finish_signalled().expect("finish")
    };

    let (error, plain) = list.retire(plain, marker).unwrap_err();
    assert!(error.to_string().contains("alloc_async"), "{error}");
    list.retire(pooled, marker)
        .map_err(|(error, _)| error)
        .expect("a pooled buffer retires");
    list.retire(device, marker)
        .map_err(|(error, _)| error)
        .expect("a stream-ordered buffer retires");
    assert_eq!(list.pending(), 2);

    let released = |list: &RetireList| {
        let submission = submit(&reaper);
        list.reap(|m| lanes.reached(m), &submission).expect("reap")
    };
    assert_eq!(released(&list), 0, "the lane is parked on its gate");
    gate.word(0).store(1, Ordering::Release);
    lane.synchronize().expect("lane");
    assert_eq!(released(&list), 2);
    reaper.synchronize().expect("the stream-ordered free retires");
    assert_eq!(pool.held_bytes(), 4096, "the pooled buffer went back to its pool");
    drop(plain);
}
