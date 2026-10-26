use std::ops::Range;

use crate::error::EdrlError;
use crate::reader::{Reader, checked_end, non_negative};
use crate::tokenizer_section::{FOOTER_BYTES, TokenizerFooter, content_end};

/// The size of the fixed header.
pub const HEADER_BYTES: usize = 48;

const MAGIC: u32 = 0x5157_454E;
const COMPACT_VERSION: u32 = 2;
/// The most tensors, layers and so on the format counts.
pub(crate) const MAX_COUNT: u32 = 1_000_000;
pub(crate) const MAX_METADATA_BYTES: u64 = 8 << 20;

/// The fixed 48-byte header of an artifact, big-endian:
/// `int32 magic, int32 version, int64 metadataOffset, int64 metadataSize, int64 tensorTableOffset,
/// int32 tensorCount, int32 reserved (zero), int64 tensorDataOffset`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Header {
    pub version: u32,
    pub metadata_offset: u64,
    pub metadata_size: u64,
    pub tensor_table_offset: u64,
    pub tensor_count: u32,
    pub tensor_data_offset: u64,
}

impl Header {
    /// Parses and validates everything the header alone decides. Whether its regions fit the file is
    /// [`Header::check_extent`].
    pub fn parse(bytes: &[u8; HEADER_BYTES]) -> Result<Header, EdrlError> {
        let mut reader = Reader::new(bytes);
        let magic = reader.u32("header magic")?;
        let version = reader.u32("header version")?;
        let metadata_offset = non_negative(reader.i64("metadata offset")?, "metadata")?;
        let metadata_size = non_negative(reader.i64("metadata size")?, "metadata")?;
        let tensor_table_offset = non_negative(reader.i64("tensor table offset")?, "tensor table")?;
        let tensor_count = reader.i32("tensor count")?;
        let reserved = reader.i32("header reserved field")?;
        let tensor_data_offset = non_negative(reader.i64("tensor data offset")?, "tensor data")?;

        if reserved != 0 {
            return Err(EdrlError::invalid("header reserved field must be zero"));
        }
        if magic != MAGIC {
            return Err(EdrlError::BadMagic { found: magic });
        }
        if version != COMPACT_VERSION {
            return Err(EdrlError::UnsupportedVersion { found: version });
        }
        let tensor_count = u32::try_from(tensor_count)
            .ok()
            .filter(|count| *count <= MAX_COUNT)
            .ok_or_else(|| EdrlError::invalid("tensor count is outside the supported range"))?;
        if metadata_offset != HEADER_BYTES as u64 {
            return Err(EdrlError::invalid("metadata must immediately follow the header"));
        }
        let metadata_end = checked_end(metadata_offset, metadata_size, "metadata")?;
        if tensor_table_offset != metadata_end {
            return Err(EdrlError::invalid("tensor table must immediately follow metadata"));
        }
        if tensor_data_offset < tensor_table_offset {
            return Err(EdrlError::invalid("tensor data offset is outside the file"));
        }
        if metadata_size > MAX_METADATA_BYTES {
            return Err(EdrlError::invalid("metadata is too large"));
        }
        Ok(Header {
            version,
            metadata_offset,
            metadata_size,
            tensor_table_offset,
            tensor_count,
            tensor_data_offset,
        })
    }

    /// The metadata bytes: they follow the header.
    pub fn metadata_range(&self) -> Range<u64> {
        self.metadata_offset..self.metadata_offset + self.metadata_size
    }

    /// The tensor table bytes: they follow the metadata and end where the payloads begin.
    pub fn table_range(&self) -> Range<u64> {
        self.tensor_table_offset..self.tensor_data_offset
    }

    /// Where the tensor data section begins: no payload starts before it.
    pub fn data_offset(&self) -> u64 {
        self.tensor_data_offset
    }

    /// Checks that the metadata, the table and the data section start inside the artifact's content, which ends at
    /// `content_end` (the start of the tokenizer section, or the end of the file; see
    /// [`content_end`](crate::content_end)).
    pub fn check_extent(&self, content_end: u64) -> Result<(), EdrlError> {
        if self.metadata_offset + self.metadata_size > content_end {
            return Err(EdrlError::invalid("metadata extends beyond the file"));
        }
        if self.tensor_table_offset > content_end {
            return Err(EdrlError::invalid("tensor table starts beyond the file"));
        }
        if self.tensor_data_offset > content_end {
            return Err(EdrlError::invalid("tensor data offset is outside the file"));
        }
        Ok(())
    }
}

/// What a planned read holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Region {
    Header,
    Metadata,
    Table,
    TokenizerBody,
    TokenizerFooter,
}

/// One byte range to fetch, and what it is.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PlannedRead {
    pub region: Region,
    pub range: Range<u64>,
}

/// The byte ranges a loader fetches to parse an artifact, in file order. Tensor payloads are not in it: a loader
/// reads each where its [`TensorEntry`](crate::TensorEntry) says.
///
/// Planning takes two steps, because the later ranges depend on bytes of the earlier ones:
/// [`ReadPlan::probe`] names the header and the footer, which the file's length alone locates;
/// [`ReadPlan::new`] adds the metadata, the table and the tokenizer body once those are parsed.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ReadPlan {
    reads: Vec<PlannedRead>,
}

impl ReadPlan {
    /// The header and the footer of a file of `file_len` bytes.
    pub fn probe(file_len: u64) -> Result<ReadPlan, EdrlError> {
        if file_len < HEADER_BYTES as u64 {
            return Err(EdrlError::invalid("artifact is truncated before the header"));
        }
        Ok(ReadPlan {
            reads: vec![
                PlannedRead {
                    region: Region::Header,
                    range: 0..HEADER_BYTES as u64,
                },
                PlannedRead {
                    region: Region::TokenizerFooter,
                    range: footer_range(file_len),
                },
            ],
        })
    }

    /// The header, metadata and table, the tokenizer body when `footer` is present, and the footer, for an
    /// artifact of `file_len` bytes. `header` must fit the content: see [`Header::check_extent`].
    pub fn new(header: &Header, file_len: u64, footer: Option<&TokenizerFooter>) -> Result<ReadPlan, EdrlError> {
        let content_end = content_end(file_len, footer)?;
        header.check_extent(content_end)?;
        let mut reads = vec![
            PlannedRead {
                region: Region::Header,
                range: 0..HEADER_BYTES as u64,
            },
            PlannedRead {
                region: Region::Metadata,
                range: header.metadata_range(),
            },
            PlannedRead {
                region: Region::Table,
                range: header.table_range(),
            },
        ];
        if let Some(footer) = footer {
            reads.push(PlannedRead {
                region: Region::TokenizerBody,
                range: footer.body_range(file_len)?,
            });
            reads.push(PlannedRead {
                region: Region::TokenizerFooter,
                range: footer_range(file_len),
            });
        }
        Ok(ReadPlan { reads })
    }

    /// The reads, in file order.
    pub fn reads(&self) -> &[PlannedRead] {
        &self.reads
    }

    /// The range of `region`, if it is planned.
    pub fn range(&self, region: Region) -> Option<Range<u64>> {
        self.reads
            .iter()
            .find(|read| read.region == region)
            .map(|read| read.range.clone())
    }
}

fn footer_range(file_len: u64) -> Range<u64> {
    file_len - FOOTER_BYTES as u64..file_len
}
