use std::io::Write;

use super::*;

/// A file of the bytes 0..200 and its handle.
fn file() -> OwnedFile {
    let mut file = tempfile::tempfile().unwrap();
    file.write_all(&(0..200).collect::<Vec<u8>>()).unwrap();
    OwnedFile::from(file)
}

#[test]
fn a_shard_reads_its_own_range_of_the_file_by_position() {
    let shard = FileShard::over(&file(), 50, 100).unwrap();
    assert_eq!(shard.size(), 100);
    let mut out = [0u8; 4];
    shard.read_at(0, &mut out).unwrap();
    assert_eq!(out, [50, 51, 52, 53]);
    shard.read_at(96, &mut out).unwrap();
    assert_eq!(out, [146, 147, 148, 149]);
}

#[test]
fn a_read_past_the_shard_is_refused_though_the_file_goes_on() {
    let shard = FileShard::over(&file(), 50, 100).unwrap();
    let mut out = [0u8; 4];
    assert!(shard.read_at(97, &mut out).is_err());
    assert!(shard.read_at(u64::MAX, &mut out).is_err());
}

#[test]
fn a_range_that_overflows_is_refused_when_the_shard_is_made() {
    assert!(FileShard::over(&file(), u64::MAX, 2).is_err());
}

#[test]
fn a_shard_outlives_the_handle_it_was_made_from() {
    let handle = file();
    let shard = FileShard::over(&handle, 0, 200).unwrap();
    drop(handle);
    let mut out = [0u8; 2];
    shard.read_at(10, &mut out).unwrap();
    assert_eq!(out, [10, 11]);
}
