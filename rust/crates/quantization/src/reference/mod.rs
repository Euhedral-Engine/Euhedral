//! The reference writer: the logits of a model over a text, written as a `.kld` file ([`crate::kld`]) that the scorer
//! and the allocation search read.
//!
//! The text is tokenized without a start token and cut into consecutive chunks of `context` tokens. Each chunk is
//! forced through the model: its first `context / 2 + 1` tokens are the prefix, and the logits row that predicts each
//! of the next `context - 1 - context / 2` tokens becomes one record (see [`encode_record`]).
//!
//! The file is written in place as `OUT.partial`, of its final size, with an 8-byte count of the chunks that are
//! complete after the last record ([`partial`]). A run that stops carries on after the last complete chunk when it
//! starts again on the same tokens; finishing cuts the count off and renames the file to `OUT`. [`run`] drives the
//! chunks on a [`Forcer`] (the engine) and writes the rows from frames of the `host` lake.

pub mod partial;
pub mod run;

use std::io;
use std::path::Path;

use euhedral_inference_tokenizer::{Tokenizer, read_artifact_tokenizer};
use thiserror::Error;

pub use partial::{Mark, Partial, RowFile};
pub use run::{DEFAULT_BATCH, DEFAULT_ROWS, Forcer, Options, Written, write};

use crate::kld::FLOOR;
use crate::resume::ResumeError;
use crate::tokenizer_files::read_checkpoint_tokenizer;

/// The smallest context a reference can have: a prefix, and at least one forced token.
pub const MIN_CONTEXT: usize = 4;

/// What stops a reference from being written.
#[derive(Debug, Error)]
pub enum ReferenceError {
    /// The request cannot be run: nothing was written.
    #[error("{0}")]
    Refused(String),
    /// A partial file belongs to other tokens, another context or another vocabulary.
    #[error("{0} was started for other tokens, another context or vocabulary; remove it to start again")]
    Mismatch(String),
    /// The engine did not force a chunk.
    #[error("the model run failed: {0}")]
    Engine(String),
    #[error("{0}")]
    Io(String),
    #[error(transparent)]
    File(#[from] ResumeError),
}

impl ReferenceError {
    fn io(context: &str, error: &io::Error) -> ReferenceError {
        ReferenceError::Io(format!("{context}: {error}"))
    }
}

/// The bytes of one record: the log-softmax of a row of BF16 logits as llama.cpp stores it. The row is clamped below
/// at `low = max(min, max - 16)`; with `lse = ln(sum exp(logit - max))` the record holds `scale = (max - low) / 65535`
/// and `minimum = low - max - lse` as f32, then a code per vocabulary entry, `rint((max(logit, low) - low) / scale)`,
/// padded to an even count. The log-probability of entry `i` is `scale * code[i] + minimum`.
pub fn encode_record(logits: &[u8]) -> Vec<u8> {
    let values: Vec<f32> = logits
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| f32::from_bits(u32::from(u16::from_le_bytes([pair[0], pair[1]])) << 16))
        .collect();
    let (mut max, mut min) = (values[0], values[0]);
    for &value in &values {
        max = max.max(value);
        min = min.min(value);
    }
    let low = min.max(max + FLOOR as f32);
    let mut sum = 0.0f64;
    for &value in &values {
        sum += f64::from(value - max).exp();
    }
    let log_sum_exp = sum.ln() as f32;
    let scale = (max - low) / 65535.0;
    let minimum = low - max - log_sum_exp;
    let inverse = if scale > 0.0 { 1.0 / scale } else { 0.0 };
    let mut record = Vec::with_capacity(8 + 2 * (2 * values.len().div_ceil(2)));
    record.extend_from_slice(&scale.to_le_bytes());
    record.extend_from_slice(&minimum.to_le_bytes());
    for &value in &values {
        let code = (inverse * (value.max(low) - low)).round_ties_even() as i32;
        record.extend_from_slice(&(code.clamp(0, 65535) as u16).to_le_bytes());
    }
    // The count is padded to an even number of codes.
    record.resize(8 + 2 * (2 * values.len().div_ceil(2)), 0);
    record
}

/// `tokens` as consecutive chunks of `context` tokens, `limit` of them when `limit` is positive, else as many whole
/// chunks as there are.
pub fn split_chunks(tokens: &[u32], context: usize, limit: usize) -> Result<Vec<Vec<u32>>, ReferenceError> {
    if context < MIN_CONTEXT {
        return Err(ReferenceError::Refused(format!(
            "the context is at least {MIN_CONTEXT} tokens"
        )));
    }
    let whole = tokens.len() / context;
    if limit > whole {
        return Err(ReferenceError::Refused(format!(
            "{limit} chunks need {} tokens, the text has {}",
            limit.saturating_mul(context),
            tokens.len()
        )));
    }
    let count = if limit > 0 { limit } else { whole };
    if count < 1 {
        return Err(ReferenceError::Refused(format!(
            "the text has {} tokens, fewer than one chunk of {context}",
            tokens.len()
        )));
    }
    Ok((0..count)
        .map(|chunk| tokens[chunk * context..(chunk + 1) * context].to_vec())
        .collect())
}

/// Fails on the first token `is_control` names, unless the text may hold them: llama.cpp reads the spelling of a
/// control token as plain text, so a text that holds one is refused rather than tokenized another way.
pub fn refuse_control(
    tokens: &[u32],
    is_control: impl Fn(u32) -> bool,
    spelling: impl Fn(u32) -> String,
) -> Result<(), ReferenceError> {
    match tokens.iter().find(|token| is_control(**token)) {
        Some(token) => Err(ReferenceError::Refused(format!(
            "the text holds the control token {}; pass --parse-special to make it one token",
            spelling(*token)
        ))),
        None => Ok(()),
    }
}

/// The chunks of `text` as the tokenizer at `tokenizer` makes them: a checkpoint directory (its tokenizer files, as
/// `package-tokenizer` reads them) or an artifact (its tokenizer section). Reads synchronously: call it on a thread
/// that may wait, before the engine starts.
pub fn tokenize(
    tokenizer: &Path,
    text: &str,
    context: usize,
    limit: usize,
    parse_special: bool,
) -> Result<Vec<Vec<u32>>, ReferenceError> {
    let tokenizer = read_tokenizer(tokenizer).map_err(ReferenceError::Refused)?;
    chunk_text(&tokenizer, text, context, limit, parse_special)
}

/// The tokenizer at `path`: a checkpoint directory (its tokenizer files, as `package-tokenizer` reads them) or an
/// artifact (its tokenizer section). Reads synchronously: call it on a thread that may wait, before the engine starts.
pub fn read_tokenizer(path: &Path) -> Result<Tokenizer, String> {
    if path.is_dir() {
        read_checkpoint_tokenizer(path).map_err(|error| error.to_string())
    } else {
        read_artifact_tokenizer(path).map_err(|error| error.to_string())
    }
}

/// The chunks of `text` as `tokenizer` makes them.
pub fn chunk_text(
    tokenizer: &Tokenizer,
    text: &str,
    context: usize,
    limit: usize,
    parse_special: bool,
) -> Result<Vec<Vec<u32>>, ReferenceError> {
    let tokens = tokenizer
        .encode(text)
        .map_err(|error| ReferenceError::Refused(error.to_string()))?;
    if !parse_special {
        refuse_control(
            &tokens,
            |token| tokenizer.is_control_token(token),
            |token| tokenizer.control_token_text(token).unwrap_or_default().to_owned(),
        )?;
    }
    split_chunks(&tokens, context, limit)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kld::Row;

    fn bf16(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|value| ((value.to_bits() >> 16) as u16).to_le_bytes())
            .collect()
    }

    fn log_softmax(values: &[f32]) -> Vec<f64> {
        let max = values.iter().copied().fold(f32::MIN, f32::max);
        let sum: f64 = values.iter().map(|value| f64::from(value - max).exp()).sum();
        values.iter().map(|value| f64::from(value - max) - sum.ln()).collect()
    }

    #[test]
    fn a_record_decodes_to_the_log_softmax_within_its_resolution() {
        let values: Vec<f32> = (0..37).map(|i| ((i * 7) % 13) as f32 - 6.0).collect();
        let record = encode_record(&bf16(&values));
        assert_eq!(record.len(), 8 + 2 * 38);
        let row = Row::new(&record, values.len());
        let scale = f64::from(f32::from_le_bytes([record[0], record[1], record[2], record[3]]));
        for (token, expected) in log_softmax(&values).iter().enumerate() {
            assert!(
                (row.log_p(token) - expected).abs() <= scale / 2.0 + 1e-6,
                "token {token}: {} against {expected}",
                row.log_p(token)
            );
        }
        // The largest logit is the largest code.
        let top = (0..values.len())
            .max_by(|a, b| row.log_p(*a).total_cmp(&row.log_p(*b)))
            .unwrap();
        assert_eq!(values[top], 6.0);
    }

    #[test]
    fn entries_more_than_sixteen_below_the_largest_are_clamped_to_the_minimum() {
        let record = encode_record(&bf16(&[10.0, 9.0, -30.0, -100.0]));
        let row = Row::new(&record, 4);
        let minimum = f64::from(f32::from_le_bytes([record[4], record[5], record[6], record[7]]));
        assert_eq!(row.log_p(2), minimum);
        assert_eq!(row.log_p(3), minimum);
        assert!(row.log_p(1) > minimum);
        // The window spans exactly 16 nats when something is clamped.
        let scale = f64::from(f32::from_le_bytes([record[0], record[1], record[2], record[3]]));
        assert!((scale * 65535.0 - 16.0).abs() < 1e-3);
    }

    #[test]
    fn a_flat_row_has_no_scale_and_every_entry_is_uniform() {
        let record = encode_record(&bf16(&[2.0; 5]));
        let row = Row::new(&record, 5);
        assert_eq!(&record[..4], &0.0f32.to_le_bytes());
        for token in 0..5 {
            assert!((row.log_p(token) - (-(5.0f64).ln())).abs() < 1e-6);
        }
        // Odd vocabulary: the last padding code is zero.
        assert_eq!(record.len(), 8 + 2 * 6);
        assert_eq!(&record[record.len() - 2..], &[0, 0]);
    }

    #[test]
    fn chunks_are_consecutive_and_the_limit_is_checked() {
        let tokens: Vec<u32> = (0..25).collect();
        let all = split_chunks(&tokens, 10, 0).unwrap();
        assert_eq!(all, vec![(0..10).collect::<Vec<u32>>(), (10..20).collect()]);
        assert_eq!(split_chunks(&tokens, 10, 1).unwrap().len(), 1);
        assert!(matches!(
            split_chunks(&tokens, 10, 3),
            Err(ReferenceError::Refused(message)) if message.contains("30 tokens")
        ));
        assert!(matches!(split_chunks(&tokens, 30, 0), Err(ReferenceError::Refused(_))));
        assert!(matches!(split_chunks(&tokens, 3, 0), Err(ReferenceError::Refused(_))));
    }

    #[test]
    fn a_control_token_is_refused_unless_the_text_may_hold_it() {
        let tokens = [1, 2, 99, 3];
        let refused = refuse_control(&tokens, |token| token == 99, |token| format!("<{token}>"));
        assert!(matches!(refused, Err(ReferenceError::Refused(message)) if message.contains("<99>")));
        assert!(refuse_control(&[1, 2, 3], |token| token == 99, |_| String::new()).is_ok());
    }
}
