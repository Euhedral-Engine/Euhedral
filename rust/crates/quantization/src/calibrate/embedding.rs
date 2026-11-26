//! The first layer's input: each calibration token's row of the embedding table, copied into the hidden states.

use super::CalibrateError;

/// The dtype the embedding table is stored in, and so the states' element type.
const DTYPE: &str = "BF16";
/// Bytes of one BF16 element.
const ELEMENT_BYTES: u64 = 2;

/// The rows of an embedding table stored as `dtype` with `shape`, which must be BF16 `[vocabulary, hidden]`.
pub fn vocabulary(dtype: &str, shape: &[u64], hidden: u64) -> Result<u64, CalibrateError> {
    if dtype != DTYPE {
        return Err(CalibrateError::Invalid(format!(
            "the embedding table is {dtype}, not {DTYPE}"
        )));
    }
    match shape {
        [rows, columns] if *columns == hidden && *rows > 0 => Ok(*rows),
        _ => Err(CalibrateError::Invalid(format!(
            "the embedding table has shape {shape:?}, not [vocabulary, {hidden}]"
        ))),
    }
}

/// Bytes of one row of the table: the hidden size in BF16.
pub fn row_bytes(hidden: u64) -> u64 {
    hidden * ELEMENT_BYTES
}

/// Checks every token is a row of a table of `vocabulary` rows, before the table is read.
pub fn check_tokens(tokens: &[i32], vocabulary: u64) -> Result<(), CalibrateError> {
    for (position, &token) in tokens.iter().enumerate() {
        if u64::try_from(token).ok().filter(|row| *row < vocabulary).is_none() {
            return Err(CalibrateError::Invalid(format!(
                "token {token} at {position} is outside the {vocabulary} embeddings"
            )));
        }
    }
    Ok(())
}

/// Copies each token's row of `table` (rows of `row` bytes) to its position in `states`.
pub fn copy_rows(tokens: &[i32], table: &[u8], row: usize, states: &mut [u8]) -> Result<(), CalibrateError> {
    if states.len() != tokens.len() * row {
        return Err(CalibrateError::Invalid(format!(
            "{} tokens of {row} bytes do not fill {} bytes of states",
            tokens.len(),
            states.len()
        )));
    }
    let vocabulary = (table.len() / row.max(1)) as u64;
    check_tokens(tokens, vocabulary)?;
    for (position, &token) in tokens.iter().enumerate() {
        let from = token as usize * row;
        states[position * row..(position + 1) * row].copy_from_slice(&table[from..from + row]);
    }
    Ok(())
}
