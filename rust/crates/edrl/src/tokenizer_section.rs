use std::ops::Range;

use crate::error::EdrlError;
use crate::reader::Reader;

/// The size of the footer.
pub const FOOTER_BYTES: usize = 24;

/// The entries of the body, in their order.
pub const TOKENIZER_ENTRIES: [&str; 4] = [
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "special_tokens.json",
];

const VERSION: u32 = 1;
const MAGIC: &[u8; 8] = b"EDRLTOKN";
const MAX_BODY_BYTES: u64 = 1 << 30;

/// The footer at the very end of an artifact that carries a tokenizer section, big-endian:
/// `int64 bodyLength, int32 CRC-32 of the body, int32 version (1), 8 bytes "EDRLTOKN"`. The body precedes it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerFooter {
    body_length: u64,
    crc: u32,
}

impl TokenizerFooter {
    /// The footer in the last 24 bytes of a file: `None` when they do not end in the magic (the artifact carries no
    /// section).
    pub fn parse(bytes: &[u8; FOOTER_BYTES]) -> Result<Option<TokenizerFooter>, EdrlError> {
        let mut reader = Reader::new(bytes);
        let body_length = reader.i64("tokenizer footer")?;
        let crc = reader.u32("tokenizer footer")?;
        let version = reader.u32("tokenizer footer")?;
        if reader.take(8, "tokenizer footer")? != MAGIC {
            return Ok(None);
        }
        if version != VERSION {
            return Err(EdrlError::TokenizerVersion { found: version });
        }
        let body_length = u64::try_from(body_length)
            .ok()
            .filter(|length| *length <= MAX_BODY_BYTES)
            .ok_or_else(|| {
                EdrlError::invalid(format!("the tokenizer section's length is out of range: {body_length}"))
            })?;
        Ok(Some(TokenizerFooter { body_length, crc }))
    }

    /// The size of the section's body.
    pub fn body_length(&self) -> u64 {
        self.body_length
    }

    /// The CRC-32 the footer records for the body.
    pub fn crc(&self) -> u32 {
        self.crc
    }

    /// Where the body is in a file of `file_len` bytes.
    pub fn body_range(&self, file_len: u64) -> Result<Range<u64>, EdrlError> {
        let start = content_end(file_len, Some(self))?;
        Ok(start..start + self.body_length)
    }
}

/// Where the artifact's content ends: the start of its tokenizer section, or `file_len` when it has none. Payloads
/// and tables must end here.
///
/// A footer whose body does not fit before it is an error (the Java reader refuses it the same way), which is why
/// this returns a `Result`.
pub fn content_end(file_len: u64, footer: Option<&TokenizerFooter>) -> Result<u64, EdrlError> {
    let Some(footer) = footer else { return Ok(file_len) };
    file_len
        .checked_sub(FOOTER_BYTES as u64)
        .and_then(|before_footer| before_footer.checked_sub(footer.body_length))
        .ok_or_else(|| {
            EdrlError::invalid(format!(
                "the tokenizer section's length is out of range: {}",
                footer.body_length
            ))
        })
}

/// The tokenizer files an artifact carries, checked against the footer's CRC-32.
///
/// Body layout, big-endian: `int32 entryCount (4)`, then per entry `int32 nameLength, UTF-8 name, int32 dataLength,
/// UTF-8 data`, in the order of [`TOKENIZER_ENTRIES`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerSection {
    body: Vec<u8>,
    data: [Range<usize>; 4],
}

impl TokenizerSection {
    /// Decodes `body` (the bytes before the footer), which must be exactly the footer's body length and match its
    /// CRC-32.
    pub fn parse(body: &[u8], footer: &TokenizerFooter) -> Result<TokenizerSection, EdrlError> {
        if body.len() as u64 != footer.body_length {
            return Err(EdrlError::Truncated {
                what: "tokenizer section",
            });
        }
        if crc32fast::hash(body) != footer.crc {
            return Err(EdrlError::TokenizerCrc);
        }
        let mut reader = Reader::new(body);
        let count = section_int(&mut reader)?;
        if usize::try_from(count).ok() != Some(TOKENIZER_ENTRIES.len()) {
            return Err(EdrlError::invalid(format!(
                "the tokenizer section holds {count} entries, not {}",
                TOKENIZER_ENTRIES.len()
            )));
        }
        let mut data = [0..0, 0..0, 0..0, 0..0];
        for (index, expected) in TOKENIZER_ENTRIES.iter().enumerate() {
            let name = &body[text(&mut reader, body)?];
            if name != expected.as_bytes() {
                return Err(EdrlError::invalid(format!(
                    "tokenizer section entry {index} is {}, not {expected}",
                    String::from_utf8_lossy(name)
                )));
            }
            data[index] = text(&mut reader, body)?;
        }
        if reader.remaining() != 0 {
            return Err(EdrlError::invalid("the tokenizer section has trailing bytes"));
        }
        Ok(TokenizerSection {
            body: body.to_vec(),
            data,
        })
    }

    /// The UTF-8 bytes of the entry `name` (one of [`TOKENIZER_ENTRIES`]).
    pub fn file(&self, name: &str) -> Option<&[u8]> {
        let index = TOKENIZER_ENTRIES.iter().position(|entry| *entry == name)?;
        Some(&self.body[self.data[index].clone()])
    }

    /// The text of the entry `name`.
    pub fn text(&self, name: &str) -> Option<&str> {
        // Validated as UTF-8 by `parse`.
        self.file(name).and_then(|bytes| std::str::from_utf8(bytes).ok())
    }
}

fn section_int(reader: &mut Reader<'_>) -> Result<i32, EdrlError> {
    reader
        .i32("tokenizer section")
        .map_err(|_| EdrlError::invalid("the tokenizer section is truncated"))
}

/// The range in `body` of the next length-prefixed UTF-8 text.
fn text(reader: &mut Reader<'_>, body: &[u8]) -> Result<Range<usize>, EdrlError> {
    let truncated = || EdrlError::invalid("the tokenizer section is truncated");
    let length = usize::try_from(section_int(reader)?).map_err(|_| truncated())?;
    let bytes = reader.take(length, "tokenizer section").map_err(|_| truncated())?;
    if std::str::from_utf8(bytes).is_err() {
        return Err(EdrlError::invalid("the tokenizer section holds text that is not UTF-8"));
    }
    let end = body.len() - reader.remaining();
    Ok(end - length..end)
}
