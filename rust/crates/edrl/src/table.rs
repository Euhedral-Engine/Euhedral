use std::collections::HashMap;
use std::ops::Range;
use std::str;

use crate::error::EdrlError;
use crate::format::{TensorDataType, WeightFormat, WeightLayout, accepts_byte_size};
use crate::header::{Header, MAX_COUNT};
use crate::reader::{Reader, checked_end, non_negative};

const MAX_RANK: i32 = 64;
const MAX_NAME_BYTES: usize = 1 << 20;
/// The smallest entry: a one-byte name, rank 0, and the ordinals, offset and size.
const MIN_ENTRY_BYTES: usize = 4 + 1 + 4 + 12 + 16;

/// One tensor descriptor: where its payload is and how to read it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorEntry {
    name: String,
    shape: Vec<u64>,
    dtype: TensorDataType,
    format: WeightFormat,
    layout: WeightLayout,
    offset: u64,
    bytes: u64,
}

impl TensorEntry {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape
    }

    pub fn dtype(&self) -> TensorDataType {
        self.dtype
    }

    pub fn format(&self) -> WeightFormat {
        self.format
    }

    pub fn layout(&self) -> WeightLayout {
        self.layout
    }

    /// Where the payload starts in the file.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    /// The payload's size.
    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The payload's byte range.
    pub fn range(&self) -> Range<u64> {
        // `TensorTable::parse` checked that the end does not overflow.
        self.offset..self.offset + self.bytes
    }
}

/// The tensor table, in file order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorTable {
    entries: Vec<TensorEntry>,
    index: HashMap<String, usize>,
}

impl TensorTable {
    /// Parses and validates the table region `bytes` (exactly [`Header::table_range`]).
    ///
    /// `content_end` is where the artifact's content ends: the file's length when it has no tokenizer section, else
    /// the start of that section (see [`content_end`](crate::content_end)). Every payload must lie between the data
    /// offset and `content_end`, have a byte size its shape, format and layout allow, and share no byte with another.
    pub fn parse(bytes: &[u8], header: &Header, content_end: u64) -> Result<TensorTable, EdrlError> {
        header.check_extent(content_end)?;
        let table_range = header.table_range();
        if bytes.len() as u64 != table_range.end - table_range.start {
            return Err(EdrlError::invalid("tensor table has trailing or missing bytes"));
        }
        let mut reader = Reader::new(bytes);
        // The capacity is bounded by what the bytes can hold, so a hostile count allocates nothing large.
        let capacity = (header.tensor_count().min(MAX_COUNT) as usize).min(bytes.len() / MIN_ENTRY_BYTES);
        let mut entries = Vec::with_capacity(capacity);
        for _ in 0..header.tensor_count() {
            entries.push(entry(&mut reader, header, content_end)?);
        }
        if reader.remaining() != 0 {
            return Err(EdrlError::invalid("tensor table has trailing or missing bytes"));
        }
        validate_non_overlapping(&entries)?;
        let mut index = HashMap::with_capacity(entries.len());
        for (position, entry) in entries.iter().enumerate() {
            if index.insert(entry.name.clone(), position).is_some() {
                return Err(EdrlError::DuplicateName {
                    name: entry.name.clone(),
                });
            }
        }
        Ok(TensorTable { entries, index })
    }

    /// The entries, in file order.
    pub fn entries(&self) -> &[TensorEntry] {
        &self.entries
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entry named `name`; names are unique.
    pub fn get(&self, name: &str) -> Option<&TensorEntry> {
        self.index.get(name).map(|position| &self.entries[*position])
    }
}

fn entry(reader: &mut Reader<'_>, header: &Header, content_end: u64) -> Result<TensorEntry, EdrlError> {
    let name_length = reader.i32("tensor name length")?;
    let name_length = usize::try_from(name_length)
        .ok()
        .filter(|length| (1..=MAX_NAME_BYTES).contains(length))
        .ok_or_else(|| {
            EdrlError::invalid(format!(
                "tensor name length is outside the supported range: {name_length}"
            ))
        })?;
    let name = str::from_utf8(reader.take(name_length, "tensor name")?)
        .map_err(|_| EdrlError::invalid("tensor name is not valid UTF-8"))?
        .to_owned();
    let rank = reader.i32("tensor rank")?;
    if !(0..=MAX_RANK).contains(&rank) {
        return Err(EdrlError::invalid(format!(
            "tensor rank is outside the supported range: {rank}"
        )));
    }
    let mut shape = Vec::with_capacity(rank as usize);
    for _ in 0..rank {
        let dimension = reader.i64("tensor shape")?;
        let dimension =
            u64::try_from(dimension).map_err(|_| EdrlError::invalid("tensor shape contains a negative dimension"))?;
        shape.push(dimension);
    }
    let dtype = TensorDataType::from_ordinal(reader.i32("tensor data type")?)?;
    let format = WeightFormat::from_ordinal(reader.i32("tensor weight format")?)?;
    let layout = WeightLayout::from_ordinal(reader.i32("tensor weight layout")?)?;
    let offset = non_negative(reader.i64("tensor data offset")?, "tensor data")?;
    let bytes = non_negative(reader.i64("tensor byte size")?, "tensor data")?;
    if offset < header.data_offset() {
        return Err(EdrlError::invalid("tensor data offset precedes tensor data section"));
    }
    if checked_end(offset, bytes, "tensor data")? > content_end {
        return Err(EdrlError::invalid(format!(
            "tensor data extends beyond the file: {name}"
        )));
    }
    match accepts_byte_size(&shape, dtype, format, layout, bytes) {
        Ok(true) => {}
        Ok(false) => return Err(EdrlError::SizeMismatch { name, bytes }),
        Err(reason) => return Err(EdrlError::UnsupportedTensor { name, reason }),
    }
    Ok(TensorEntry {
        name,
        shape,
        dtype,
        format,
        layout,
        offset,
        bytes,
    })
}

/// Rejects two non-empty payloads that share a byte.
fn validate_non_overlapping(entries: &[TensorEntry]) -> Result<(), EdrlError> {
    let mut by_offset: Vec<&TensorEntry> = entries.iter().filter(|entry| entry.bytes > 0).collect();
    by_offset.sort_by_key(|entry| entry.offset);
    for pair in by_offset.windows(2) {
        if pair[1].offset < pair[0].offset + pair[0].bytes {
            return Err(EdrlError::Overlap {
                first: pair[0].name.clone(),
                second: pair[1].name.clone(),
            });
        }
    }
    Ok(())
}
