//! A shard of the n-gram table read in place from the artifact file: the file is not mapped (no dependency maps files
//! on every platform the engine builds for); each row is one read through RustEE's asynchronous I/O, which a worker
//! of the lattice never waits for (the I/O engine of each platform makes the read, and its end is a frame).

use std::fmt;
use std::sync::Arc;

use euhedral_io::{AsyncIo, IoDoneOwned, OwnedFile};

use super::{RowRead, ShardBytes};
use crate::runtime::raw_file;

/// The `len` bytes at `base` of a file, read by position. It holds the file open, and keeps no cursor, so reads from
/// many threads need no lock.
pub struct FileShard {
    io: Arc<AsyncIo>,
    file: Arc<OwnedFile>,
    base: u64,
    len: u64,
}

impl fmt::Debug for FileShard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileShard")
            .field("base", &self.base)
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl FileShard {
    /// The shard of `len` bytes at `base` of `file`, read through `io`. Refuses a range that overflows 64 bits.
    pub fn over(io: Arc<AsyncIo>, file: Arc<OwnedFile>, base: u64, len: u64) -> Result<FileShard, String> {
        if base.checked_add(len).is_none() {
            return Err(format!("{len} bytes at {base} overflow 64 bits"));
        }
        Ok(FileShard { io, file, base, len })
    }

    /// Where in the file `len` bytes at `offset` of the shard are. Refuses an empty range and one past the shard.
    fn locate(&self, offset: u64, len: usize) -> Result<u64, String> {
        let end = offset.checked_add(len as u64).filter(|end| *end <= self.len);
        if len == 0 || end.is_none() {
            return Err(format!("{len} bytes at {offset} of a shard of {}", self.len));
        }
        // The sum is at most `base + len`, which `over` proved fits.
        Ok(self.base + offset)
    }
}

impl ShardBytes for FileShard {
    fn size(&self) -> u64 {
        self.len
    }

    fn start(&self, offset: u64, len: usize, then: RowRead) -> Result<(), String> {
        let at = self.locate(offset, len)?;
        let ended: euhedral_io::Continuation<IoDoneOwned<Vec<u8>>> =
            Box::new(move |done| then(read_result(done, len, at)));
        self.io
            .read_into(raw_file(&self.file), at, vec![0u8; len], ended)
            .map_err(|error| format!("starting the read of {len} bytes at {at} of the file: {error}"))
    }
}

/// The bytes a read of `len` bytes at `at` of the file brought, or what went wrong.
fn read_result(done: IoDoneOwned<Vec<u8>>, len: usize, at: u64) -> Result<Vec<u8>, String> {
    let IoDoneOwned { buf, result, .. } = done;
    match (buf, result) {
        (Some(bytes), Ok(read)) if read == len => Ok(bytes),
        (Some(_), Ok(read)) => Err(format!("{read} of {len} bytes at {at} of the file")),
        (_, Err(error)) => Err(format!("reading {len} bytes at {at} of the file: {error}")),
        (None, Ok(_)) => Err(format!(
            "reading {len} bytes at {at} of the file: the ring kept the memory"
        )),
    }
}

#[cfg(test)]
mod tests;
