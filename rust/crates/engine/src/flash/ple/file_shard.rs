//! A shard of the n-gram table read in place from the artifact file: the file is not mapped (no dependency maps files
//! on every platform the engine builds for); each row is a positional read of the file's own pages.

use std::fs::File;
use std::io;

use euhedral_io::OwnedFile;

use super::ShardBytes;

/// The `len` bytes at `base` of a file, read by position. It owns a duplicate of the handle, so the file it was made
/// from may close, and it keeps no cursor, so reads from many threads need no lock.
#[derive(Debug)]
pub struct FileShard {
    file: File,
    base: u64,
    len: u64,
}

impl FileShard {
    /// The shard of `len` bytes at `base` of `file`. Refuses a range that overflows 64 bits and a handle that cannot
    /// be duplicated.
    pub fn over(file: &OwnedFile, base: u64, len: u64) -> io::Result<FileShard> {
        if base.checked_add(len).is_none() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("{len} bytes at {base} overflow 64 bits"),
            ));
        }
        Ok(FileShard {
            file: File::from(file.try_clone()?),
            base,
            len,
        })
    }
}

impl ShardBytes for FileShard {
    fn size(&self) -> u64 {
        self.len
    }

    fn read_at(&self, offset: u64, out: &mut [u8]) -> Result<(), String> {
        let end = offset.checked_add(out.len() as u64).filter(|end| *end <= self.len);
        if end.is_none() {
            return Err(format!("{} bytes at {offset} of a shard of {}", out.len(), self.len));
        }
        // The sum is at most `base + len`, which `over` proved fits.
        read_exact_at(&self.file, out, self.base + offset).map_err(|error| {
            format!(
                "reading {} bytes at {} of the file: {error}",
                out.len(),
                self.base + offset
            )
        })
    }
}

#[cfg(unix)]
fn read_exact_at(file: &File, out: &mut [u8], at: u64) -> io::Result<()> {
    use std::os::unix::fs::FileExt;

    file.read_exact_at(out, at)
}

#[cfg(windows)]
fn read_exact_at(file: &File, mut out: &mut [u8], mut at: u64) -> io::Result<()> {
    use std::os::windows::fs::FileExt;

    while !out.is_empty() {
        match file.seek_read(out, at)? {
            0 => return Err(io::ErrorKind::UnexpectedEof.into()),
            read => {
                out = &mut out[read..];
                at += read as u64;
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests;
