use std::io::Write;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use euhedral_core::generics::LatticeSource as _;

use super::*;
use crate::frames::{Kind, step};

/// A file of the bytes 0..200 and its handle.
fn file() -> Arc<OwnedFile> {
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(&(0..200).collect::<Vec<u8>>()).unwrap();
    Arc::new(OwnedFile::from(file))
}

fn io() -> Arc<AsyncIo> {
    AsyncIo::open().expect("the I/O engine opens")
}

/// Starts the read of `len` bytes at `offset` of `shard` and runs the engine as a lattice worker's polls do until it
/// ends: the bytes, or why not.
fn read(io: &AsyncIo, shard: &FileShard, offset: u64, len: usize) -> Result<Vec<u8>, String> {
    let ended: Arc<OnceLock<Result<Vec<u8>, String>>> = Arc::new(OnceLock::new());
    let told = Arc::clone(&ended);
    let then: RowRead = Box::new(move |result| {
        let _ = told.set(result);
        step(Kind::NgramRead, 0, || {})
    });
    shard.start(offset, len, then)?;
    let deadline = Instant::now() + Duration::from_secs(30);
    while ended.get().is_none() {
        assert!(
            Instant::now() < deadline,
            "the read of {len} bytes at {offset} never ended"
        );
        io.reap();
        io.pull(
            &mut |mut frame| frame.execute().expect("the frame runs"),
            &mut |_| false,
            u64::MAX,
        );
    }
    ended.get().cloned().unwrap()
}

#[test]
fn a_shard_reads_its_own_range_of_the_file_through_the_io_engine() {
    let io = io();
    let shard = FileShard::over(Arc::clone(&io), file(), 50, 100).unwrap();
    assert_eq!(shard.size(), 100);
    assert_eq!(read(&io, &shard, 0, 4).unwrap(), [50, 51, 52, 53]);
    assert_eq!(read(&io, &shard, 96, 4).unwrap(), [146, 147, 148, 149]);
}

#[test]
fn a_read_past_the_shard_or_of_no_bytes_is_refused_though_the_file_goes_on() {
    let io = io();
    let shard = FileShard::over(Arc::clone(&io), file(), 50, 100).unwrap();
    let past = shard.start(97, 4, Box::new(|_| step(Kind::NgramRead, 0, || {})));
    assert!(past.is_err());
    assert!(
        shard
            .start(u64::MAX, 4, Box::new(|_| step(Kind::NgramRead, 0, || {})))
            .is_err()
    );
    assert!(
        shard
            .start(0, 0, Box::new(|_| step(Kind::NgramRead, 0, || {})))
            .is_err()
    );
}

#[test]
fn a_range_that_overflows_is_refused_when_the_shard_is_made() {
    assert!(FileShard::over(io(), file(), u64::MAX, 2).is_err());
}

#[test]
fn a_read_that_the_file_cannot_fill_fails_naming_the_bytes_it_got() {
    let io = io();
    // The shard claims 100 bytes from 150, where the file holds 50.
    let shard = FileShard::over(Arc::clone(&io), file(), 150, 100).unwrap();
    let error = read(&io, &shard, 0, 100).unwrap_err();
    assert!(error.contains("100 bytes at 150"), "{error}");
}
