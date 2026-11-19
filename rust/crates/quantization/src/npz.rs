//! The `.npz` archive: a zip of `.npy` members, every member stored (not compressed), in the order they are added.
//!
//! The writer needs no seeking: a member's size and checksum are known when it is added, so each local header is
//! complete, and the central directory follows the last member. The modification time of every member is the zip
//! epoch, so the same members give the same bytes. Sizes and offsets are 32 bits: an archive of 4 GiB or more is an
//! error, not a zip64 file.

use std::io::{self, Write};

use crc32fast::Hasher;
use thiserror::Error;

use crate::npy::{self, Dtype, Npy, NpyError};

const LOCAL_HEADER: u32 = 0x0403_4b50;
const CENTRAL_HEADER: u32 = 0x0201_4b50;
const END_OF_DIRECTORY: u32 = 0x0605_4b50;
/// Names are written as UTF-8.
const UTF8_NAMES: u16 = 0x0800;
/// 1980-01-01, the zip epoch.
const DOS_DATE: u16 = 0x21;
const DOS_TIME: u16 = 0;
const MEMBER_LIMIT: u64 = u32::MAX as u64;

/// Why an archive could not be written or read.
#[derive(Debug, Error)]
pub enum NpzError {
    #[error("write failed: {0}")]
    Io(#[from] io::Error),
    #[error("the archive is beyond what a 32-bit zip holds")]
    TooLarge,
    #[error("not a readable archive: {0}")]
    Malformed(String),
    #[error("the member '{0}' is not in the archive")]
    Missing(String),
    #[error("the member '{0}' fails its checksum")]
    Checksum(String),
    #[error(transparent)]
    Npy(#[from] NpyError),
}

#[derive(Debug)]
struct Member {
    name: String,
    crc: u32,
    size: u32,
    offset: u32,
}

/// Writes an archive to `sink`, one member at a time.
#[derive(Debug)]
pub struct NpzWriter<W: Write> {
    sink: W,
    members: Vec<Member>,
    written: u64,
}

impl<W: Write> NpzWriter<W> {
    pub fn new(sink: W) -> NpzWriter<W> {
        NpzWriter {
            sink,
            members: Vec::new(),
            written: 0,
        }
    }

    /// Adds the member `<name>.npy` holding an array of `dtype` and `shape` whose bytes are `data`.
    pub fn add_array(&mut self, name: &str, dtype: Dtype, shape: &[u64], data: &[u8]) -> Result<(), NpzError> {
        let header = npy::header(dtype, shape);
        let expected = shape.iter().try_fold(dtype.item_size() as u64, |total, &dimension| {
            total.checked_mul(dimension)
        });
        if expected != Some(data.len() as u64) {
            return Err(NpzError::Npy(NpyError::DataLength {
                expected: expected.unwrap_or(u64::MAX),
                found: data.len() as u64,
            }));
        }
        self.add_member(name, &header, data)
    }

    /// Adds the member `<name>.npy` holding a 0-d unicode array of `value`.
    pub fn add_text(&mut self, name: &str, value: &str) -> Result<(), NpzError> {
        let bytes = npy::encode_text(value);
        let header_length = 10 + usize::from(u16::from_le_bytes([bytes[8], bytes[9]]));
        self.add_member(name, &bytes[..header_length], &bytes[header_length..])
    }

    fn add_member(&mut self, name: &str, header: &[u8], data: &[u8]) -> Result<(), NpzError> {
        let name = format!("{name}.npy");
        let size = u64::try_from(header.len() + data.len()).map_err(|_| NpzError::TooLarge)?;
        let offset = self.written;
        let end = offset + 30 + name.len() as u64 + size;
        if size >= MEMBER_LIMIT
            || end >= MEMBER_LIMIT
            || name.len() > usize::from(u16::MAX)
            || self.members.len() >= 0xfffe
        {
            return Err(NpzError::TooLarge);
        }
        let mut hasher = Hasher::new();
        hasher.update(header);
        hasher.update(data);
        let crc = hasher.finalize();
        let mut local = Vec::with_capacity(30 + name.len());
        local.extend_from_slice(&LOCAL_HEADER.to_le_bytes());
        local.extend_from_slice(&10u16.to_le_bytes());
        local.extend_from_slice(&UTF8_NAMES.to_le_bytes());
        local.extend_from_slice(&0u16.to_le_bytes());
        local.extend_from_slice(&DOS_TIME.to_le_bytes());
        local.extend_from_slice(&DOS_DATE.to_le_bytes());
        local.extend_from_slice(&crc.to_le_bytes());
        local.extend_from_slice(&(size as u32).to_le_bytes());
        local.extend_from_slice(&(size as u32).to_le_bytes());
        local.extend_from_slice(&(name.len() as u16).to_le_bytes());
        local.extend_from_slice(&0u16.to_le_bytes());
        local.extend_from_slice(name.as_bytes());
        self.sink.write_all(&local)?;
        self.sink.write_all(header)?;
        self.sink.write_all(data)?;
        self.written = end;
        self.members.push(Member {
            name,
            crc,
            size: size as u32,
            offset: offset as u32,
        });
        Ok(())
    }

    /// Writes the central directory and returns the sink.
    pub fn finish(mut self) -> Result<W, NpzError> {
        let mut directory = Vec::new();
        for member in &self.members {
            directory.extend_from_slice(&CENTRAL_HEADER.to_le_bytes());
            directory.extend_from_slice(&20u16.to_le_bytes());
            directory.extend_from_slice(&10u16.to_le_bytes());
            directory.extend_from_slice(&UTF8_NAMES.to_le_bytes());
            directory.extend_from_slice(&0u16.to_le_bytes());
            directory.extend_from_slice(&DOS_TIME.to_le_bytes());
            directory.extend_from_slice(&DOS_DATE.to_le_bytes());
            directory.extend_from_slice(&member.crc.to_le_bytes());
            directory.extend_from_slice(&member.size.to_le_bytes());
            directory.extend_from_slice(&member.size.to_le_bytes());
            directory.extend_from_slice(&(member.name.len() as u16).to_le_bytes());
            // Extra field, comment, disk number, internal and external attributes: none.
            directory.extend_from_slice(&[0; 2 + 2 + 2 + 2 + 4]);
            directory.extend_from_slice(&member.offset.to_le_bytes());
            directory.extend_from_slice(member.name.as_bytes());
        }
        if self.written + directory.len() as u64 + 22 >= MEMBER_LIMIT {
            return Err(NpzError::TooLarge);
        }
        let count = self.members.len() as u16;
        let central_size = directory.len() as u32;
        directory.extend_from_slice(&END_OF_DIRECTORY.to_le_bytes());
        directory.extend_from_slice(&[0; 4]);
        directory.extend_from_slice(&count.to_le_bytes());
        directory.extend_from_slice(&count.to_le_bytes());
        directory.extend_from_slice(&central_size.to_le_bytes());
        directory.extend_from_slice(&(self.written as u32).to_le_bytes());
        directory.extend_from_slice(&0u16.to_le_bytes());
        self.sink.write_all(&directory)?;
        self.sink.flush()?;
        Ok(self.sink)
    }
}

/// One member of an archive read back.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Entry<'a> {
    /// The member's name, with its `.npy`.
    pub name: String,
    pub crc: u32,
    /// The member's bytes: a whole `.npy` file.
    pub bytes: &'a [u8],
}

/// An archive of stored members, read from the caller's bytes.
#[derive(Debug)]
pub struct Npz<'a> {
    entries: Vec<Entry<'a>>,
}

impl<'a> Npz<'a> {
    /// Reads the central directory and checks every member's checksum. A member that is compressed, an archive with
    /// a comment or a zip64 archive is an error.
    pub fn parse(bytes: &'a [u8]) -> Result<Npz<'a>, NpzError> {
        let end = bytes
            .len()
            .checked_sub(22)
            .filter(|&at| u32_at(bytes, at) == Some(END_OF_DIRECTORY))
            .ok_or_else(|| NpzError::Malformed("no end of directory".into()))?;
        let count = usize::from(u16_at(bytes, end + 10).ok_or_else(cut)?);
        let mut at = u32_at(bytes, end + 16).ok_or_else(cut)? as usize;
        let mut entries = Vec::with_capacity(count);
        for _ in 0..count {
            if u32_at(bytes, at) != Some(CENTRAL_HEADER) {
                return Err(NpzError::Malformed("a bad directory entry".into()));
            }
            if u16_at(bytes, at + 10).ok_or_else(cut)? != 0 {
                return Err(NpzError::Malformed("a compressed member".into()));
            }
            let crc = u32_at(bytes, at + 16).ok_or_else(cut)?;
            let size = u32_at(bytes, at + 24).ok_or_else(cut)? as usize;
            let name_length = usize::from(u16_at(bytes, at + 28).ok_or_else(cut)?);
            let extra = usize::from(u16_at(bytes, at + 30).ok_or_else(cut)?);
            let comment = usize::from(u16_at(bytes, at + 32).ok_or_else(cut)?);
            let local = u32_at(bytes, at + 42).ok_or_else(cut)? as usize;
            let name = bytes.get(at + 46..at + 46 + name_length).ok_or_else(cut)?;
            let name = String::from_utf8(name.to_vec()).map_err(|_| NpzError::Malformed("a member name".into()))?;
            at += 46 + name_length + extra + comment;
            if u32_at(bytes, local) != Some(LOCAL_HEADER) {
                return Err(NpzError::Malformed(format!("a bad local header for '{name}'")));
            }
            let local_name = usize::from(u16_at(bytes, local + 26).ok_or_else(cut)?);
            let local_extra = usize::from(u16_at(bytes, local + 28).ok_or_else(cut)?);
            let start = local + 30 + local_name + local_extra;
            let data = bytes.get(start..start + size).ok_or_else(cut)?;
            if crc32fast::hash(data) != crc {
                return Err(NpzError::Checksum(name));
            }
            entries.push(Entry { name, crc, bytes: data });
        }
        Ok(Npz { entries })
    }

    /// The members, in the order the archive lists them.
    pub fn entries(&self) -> &[Entry<'a>] {
        &self.entries
    }

    /// The array `<name>.npy`.
    pub fn array(&self, name: &str) -> Result<Npy<'a>, NpzError> {
        let wanted = format!("{name}.npy");
        let entry = self
            .entries
            .iter()
            .find(|entry| entry.name == wanted)
            .ok_or(NpzError::Missing(wanted))?;
        Ok(Npy::parse(entry.bytes)?)
    }
}

fn cut() -> NpzError {
    NpzError::Malformed("the archive is cut".into())
}

fn u16_at(bytes: &[u8], at: usize) -> Option<u16> {
    bytes
        .get(at..at + 2)
        .map(|field| u16::from_le_bytes([field[0], field[1]]))
}

fn u32_at(bytes: &[u8], at: usize) -> Option<u32> {
    bytes
        .get(at..at + 4)
        .map(|field| u32::from_le_bytes([field[0], field[1], field[2], field[3]]))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::npy::{Half, encode_elements};

    fn sample() -> Vec<u8> {
        let mut writer = NpzWriter::new(Vec::new());
        writer.add_text("format_name", "NVFP4").unwrap();
        writer
            .add_array(
                "shape",
                Dtype::I8,
                &[2],
                &[4, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 0, 0, 0, 0],
            )
            .unwrap();
        writer
            .add_array("codes", Dtype::U1, &[2, 3], &[1, 2, 3, 4, 5, 6])
            .unwrap();
        writer.finish().unwrap()
    }

    #[test]
    fn members_come_back_in_order_with_their_bytes() {
        let bytes = sample();
        let archive = Npz::parse(&bytes).unwrap();
        let names: Vec<&str> = archive.entries().iter().map(|entry| entry.name.as_str()).collect();
        assert_eq!(names, ["format_name.npy", "shape.npy", "codes.npy"]);
        assert_eq!(archive.array("format_name").unwrap().text().unwrap(), "NVFP4");
        assert_eq!(archive.array("shape").unwrap().to_vec::<i64>().unwrap(), [4, 8]);
        let codes = archive.array("codes").unwrap();
        assert_eq!(codes.shape, [2, 3]);
        assert_eq!(codes.data, [1, 2, 3, 4, 5, 6]);
        assert_eq!(
            archive.entries()[2].bytes,
            encode_elements(&[2, 3], &[1u8, 2, 3, 4, 5, 6]).unwrap()
        );
        assert!(matches!(archive.array("table"), Err(NpzError::Missing(_))));
    }

    #[test]
    fn the_same_members_give_the_same_bytes() {
        assert_eq!(sample(), sample());
    }

    #[test]
    fn an_empty_archive_is_an_end_of_directory() {
        let bytes = NpzWriter::new(Vec::new()).finish().unwrap();
        assert_eq!(bytes.len(), 22);
        assert!(Npz::parse(&bytes).unwrap().entries().is_empty());
    }

    #[test]
    fn a_damaged_member_fails_its_checksum() {
        let mut bytes = sample();
        let at = bytes.windows(4).position(|window| window == [1, 2, 3, 4]).unwrap();
        bytes[at] = 9;
        assert!(matches!(Npz::parse(&bytes), Err(NpzError::Checksum(_))));
    }

    #[test]
    fn data_that_does_not_fit_the_shape_is_refused() {
        let mut writer = NpzWriter::new(Vec::new());
        assert!(writer.add_array("x", Dtype::F2, &[3], &[0; 4]).is_err());
        assert!(writer.add_array("x", Dtype::F2, &[2], &[0; 4]).is_ok());
        let bytes = writer.finish().unwrap();
        let half = Npz::parse(&bytes)
            .unwrap()
            .array("x")
            .unwrap()
            .to_vec::<Half>()
            .unwrap();
        assert_eq!(half.len(), 2);
    }

    #[test]
    fn truncated_archives_are_errors() {
        let bytes = sample();
        for cut in [0, 10, bytes.len() - 1, bytes.len() - 30] {
            assert!(Npz::parse(&bytes[..cut]).is_err());
        }
    }
}
