//! The draft head's token ids: the tensor `text/draft_head_token_ids` (little-endian `int32`) of an artifact, each id
//! in `[0, vocabulary)`. The caller reads the byte ranges; this parses and checks them.

use std::ops::Range;

use euhedral_inference_edrl::{HEADER_BYTES, Header, TensorDataType, TensorTable};

const NAME: &str = "text/draft_head_token_ids";

/// The ids of an artifact of `length` bytes, whose byte ranges `read` returns.
pub fn read_draft_ids(
    length: u64,
    mut read: impl FnMut(Range<u64>) -> Result<Vec<u8>, String>,
    vocabulary: u64,
) -> Result<Vec<i32>, String> {
    let header = read(0..HEADER_BYTES as u64)?;
    let header = <&[u8; HEADER_BYTES]>::try_from(header.as_slice()).map_err(|_| "no header".to_owned())?;
    let header = Header::parse(header).map_err(|error| error.to_string())?;
    header.check_extent(length).map_err(|error| error.to_string())?;
    let table = read(header.table_range())?;
    let table = TensorTable::parse(&table, &header, length).map_err(|error| error.to_string())?;
    let entry = table.get(NAME).ok_or_else(|| format!("the artifact has no {NAME}"))?;
    if entry.dtype() != TensorDataType::Int32 {
        return Err(format!("{NAME} is not int32"));
    }
    let bytes = read(entry.range())?;
    let ids: Vec<i32> = bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| i32::from_le_bytes(*word))
        .collect();
    check_draft_ids(&ids, vocabulary)?;
    Ok(ids)
}

/// Whether `ids` can be the draft head's: at least one, each a row of an output head of `rows` rows. The refusal names
/// the id that is wrong.
pub fn check_draft_ids(ids: &[i32], rows: u64) -> Result<(), String> {
    if ids.is_empty() {
        return Err(format!("{NAME} is empty"));
    }
    if let Some(id) = ids.iter().find(|id| u64::try_from(**id).map_or(true, |id| id >= rows)) {
        return Err(format!("{NAME} holds {id}, outside the {rows} rows of the output head"));
    }
    Ok(())
}
