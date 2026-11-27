use std::ops::Range;

use crate::error::EdrlError;
use crate::header::{MAGIC, MAX_METADATA_BYTES, PlannedRead, ReadPlan, Region, footer_range};
use crate::reader::{Reader, non_negative};
use crate::tokenizer_section::{TokenizerFooter, content_end};

/// The size of the fixed header of a Flash-Next artifact.
pub const FLASH_HEADER_BYTES: usize = 64;

/// Payload data starts on this boundary.
pub const DATA_ALIGNMENT: u64 = 4096;
/// Every object starts on this boundary.
pub const OBJECT_ALIGNMENT: u64 = 256;
/// The container version of Flash-Next artifacts.
pub const FLASH_VERSION: u32 = 3;
/// The only architecture of version 3: `qwen4_exp`.
pub const ARCHITECTURE_QWEN4_EXP: u32 = 1;

/// The largest the tables may be.
const MAX_TABLES_BYTES: u64 = 1 << 30;

/// The fixed 64-byte header of a version 3 artifact, big-endian:
/// `int32 magic, int32 version (3), int32 architecture (1), int32 reserved (0), int64 metadataOffset (64),
/// metadataSize, tablesOffset, tablesSize, dataOffset, fileSize`.
///
/// The metadata follows the header and the tables follow the metadata, so those two offsets are derived. `fileSize`
/// is where the content ends, before any tokenizer section.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FlashHeader {
    metadata_size: u64,
    tables_size: u64,
    data_offset: u64,
    file_size: u64,
}

impl FlashHeader {
    /// A header for metadata and tables of the given sizes, with the payloads starting at `data_offset` and the
    /// content ending at `file_size`.
    pub fn new(
        metadata_size: u64,
        tables_size: u64,
        data_offset: u64,
        file_size: u64,
    ) -> Result<FlashHeader, EdrlError> {
        if metadata_size > MAX_METADATA_BYTES {
            return Err(EdrlError::invalid("metadata size is out of range"));
        }
        if tables_size > MAX_TABLES_BYTES {
            return Err(EdrlError::invalid("table size is out of range"));
        }
        let tables_end = FLASH_HEADER_BYTES as u64 + metadata_size + tables_size;
        if tables_end > file_size {
            return Err(EdrlError::invalid("tables extend beyond the file"));
        }
        if data_offset < tables_end || data_offset > file_size || !data_offset.is_multiple_of(DATA_ALIGNMENT) {
            return Err(EdrlError::invalid(format!(
                "data section offset is invalid: {data_offset}"
            )));
        }
        if i64::try_from(file_size).is_err() {
            return Err(EdrlError::invalid("file size is out of range"));
        }
        Ok(FlashHeader {
            metadata_size,
            tables_size,
            data_offset,
            file_size,
        })
    }

    /// The 64 bytes of the header: [`FlashHeader::parse`] reads them back.
    pub fn encode(&self) -> [u8; FLASH_HEADER_BYTES] {
        let mut out = [0; FLASH_HEADER_BYTES];
        out[0..4].copy_from_slice(&MAGIC.to_be_bytes());
        out[4..8].copy_from_slice(&FLASH_VERSION.to_be_bytes());
        out[8..12].copy_from_slice(&ARCHITECTURE_QWEN4_EXP.to_be_bytes());
        // Bytes 12..16 are the reserved field, zero.
        out[16..24].copy_from_slice(&(FLASH_HEADER_BYTES as u64).to_be_bytes());
        out[24..32].copy_from_slice(&self.metadata_size.to_be_bytes());
        out[32..40].copy_from_slice(&self.tables_offset().to_be_bytes());
        out[40..48].copy_from_slice(&self.tables_size.to_be_bytes());
        out[48..56].copy_from_slice(&self.data_offset.to_be_bytes());
        out[56..64].copy_from_slice(&self.file_size.to_be_bytes());
        out
    }

    /// Parses and validates everything the header alone decides. Whether the content ends where it says is
    /// [`FlashHeader::check_content_end`].
    pub fn parse(bytes: &[u8; FLASH_HEADER_BYTES]) -> Result<FlashHeader, EdrlError> {
        let mut reader = Reader::new(bytes);
        let magic = reader.u32("header magic")?;
        if magic != MAGIC {
            return Err(EdrlError::BadMagic { found: magic });
        }
        let version = reader.u32("header version")?;
        if version != FLASH_VERSION {
            return Err(EdrlError::UnsupportedVersion { found: version });
        }
        let architecture = reader.u32("header architecture")?;
        if architecture != ARCHITECTURE_QWEN4_EXP {
            return Err(EdrlError::UnsupportedArchitecture { found: architecture });
        }
        if reader.i32("header reserved field")? != 0 {
            return Err(EdrlError::invalid("header reserved field must be zero"));
        }
        let metadata_offset = non_negative(reader.i64("metadata offset")?, "metadata")?;
        let metadata_size = non_negative(reader.i64("metadata size")?, "metadata")?;
        let tables_offset = non_negative(reader.i64("tables offset")?, "tables")?;
        let tables_size = non_negative(reader.i64("tables size")?, "tables")?;
        let data_offset = non_negative(reader.i64("data offset")?, "data")?;
        let file_size = non_negative(reader.i64("file size")?, "file size")?;
        if metadata_offset != FLASH_HEADER_BYTES as u64 {
            return Err(EdrlError::invalid("metadata must immediately follow the header"));
        }
        if metadata_size > MAX_METADATA_BYTES {
            return Err(EdrlError::invalid("metadata size is out of range"));
        }
        if tables_offset != metadata_offset + metadata_size {
            return Err(EdrlError::invalid("tables must immediately follow the metadata"));
        }
        FlashHeader::new(metadata_size, tables_size, data_offset, file_size)
    }

    /// Checks that the content ends where the header says: `content_end` is the start of the tokenizer section, or
    /// the end of the file (see [`content_end`](crate::content_end)).
    pub fn check_content_end(&self, content_end: u64) -> Result<(), EdrlError> {
        if self.file_size < content_end {
            return Err(EdrlError::invalid(format!(
                "header declares {} bytes but the file's content holds {content_end}: an interrupted package-tokenizer \
                 leaves bytes after the content without a tokenizer footer; running package-tokenizer on the artifact \
                 again repairs it",
                self.file_size
            )));
        }
        if self.file_size != content_end {
            return Err(EdrlError::invalid(format!(
                "header declares {} bytes but the file's content holds {content_end}",
                self.file_size
            )));
        }
        Ok(())
    }

    /// The size of the metadata region.
    pub fn metadata_size(&self) -> u64 {
        self.metadata_size
    }

    /// The size of the tables region.
    pub fn tables_size(&self) -> u64 {
        self.tables_size
    }

    /// Where the tables start: right after the metadata.
    pub fn tables_offset(&self) -> u64 {
        FLASH_HEADER_BYTES as u64 + self.metadata_size
    }

    /// The metadata bytes: they follow the header.
    pub fn metadata_range(&self) -> Range<u64> {
        FLASH_HEADER_BYTES as u64..self.tables_offset()
    }

    /// The tables bytes: they follow the metadata.
    pub fn tables_range(&self) -> Range<u64> {
        self.tables_offset()..self.tables_offset() + self.tables_size
    }

    /// Where the payloads begin: no object starts before it.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Where the content ends, before any tokenizer section.
    pub fn file_size(&self) -> u64 {
        self.file_size
    }
}

impl ReadPlan {
    /// The header and the footer of a Flash-Next artifact of `file_len` bytes.
    pub fn probe_flash(file_len: u64) -> Result<ReadPlan, EdrlError> {
        if file_len < FLASH_HEADER_BYTES as u64 {
            return Err(EdrlError::invalid("artifact is truncated before the header"));
        }
        Ok(ReadPlan {
            reads: vec![
                PlannedRead {
                    region: Region::Header,
                    range: 0..FLASH_HEADER_BYTES as u64,
                },
                PlannedRead {
                    region: Region::TokenizerFooter,
                    range: footer_range(file_len),
                },
            ],
        })
    }

    /// The reads that remain for a Flash-Next artifact once [`ReadPlan::probe`]'s header and footer are fetched and
    /// parsed: the metadata and the tables, then the tokenizer body when `footer` is present. The header must
    /// declare the content end the file's length and footer give.
    pub fn for_flash(
        header: &FlashHeader,
        file_len: u64,
        footer: Option<&TokenizerFooter>,
    ) -> Result<ReadPlan, EdrlError> {
        let content_end = content_end(file_len, footer)?;
        header.check_content_end(content_end)?;
        let mut reads = vec![
            PlannedRead {
                region: Region::Metadata,
                range: header.metadata_range(),
            },
            PlannedRead {
                region: Region::Table,
                range: header.tables_range(),
            },
        ];
        if let Some(footer) = footer {
            reads.push(PlannedRead {
                region: Region::TokenizerBody,
                range: footer.body_range(file_len)?,
            });
        }
        Ok(ReadPlan { reads })
    }
}
