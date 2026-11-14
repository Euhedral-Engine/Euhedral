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
    reason = "no-wait, waiting calls: a test copies synchronously to check results"
)]

use std::path::PathBuf;
use std::sync::Arc;

use euhedral_inference_cuda::{CudaLibrary, DeviceBuffer, HostWeightsArena, PinnedBuffer, PinnedPool};

/// The loaded library. The tests run in parallel on one device.
fn library() -> Option<Arc<CudaLibrary>> {
    let path = test_support::require_property("euhedral.cuda.library")?;
    Some(CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads"))
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
#[cfg(feature = "test-oracles")]
fn exact_numerics_switch_returns_the_previous_selection() {
    let Some(library) = library() else { return };
    assert!(!library.exact_numerics_for_tests(true), "relaxed is the default");
    assert!(library.exact_numerics_for_tests(false));
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
