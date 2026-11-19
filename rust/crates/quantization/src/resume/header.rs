//! The header of a resume file: a map of text keys to text values, kept in the format of a Java `Properties` file so
//! that files written before the port still resume. The text is UTF-8, `\u` escapes are read, and the rest of the
//! 4096 bytes are NULs.
//!
//! Reading follows `Properties.load`: comment lines (`#`, `!`), blank lines, a trailing backslash joining the next line,
//! a key ended by `=`, `:` or white space, and the escapes `\t \n \r \f \uXXXX`. Writing is `Properties.store` without the
//! date comment, so that the same map always gives the same bytes: keys in order, the characters Java escapes escaped, and
//! every NUL and other control character written as `\uXXXX` (a NUL in the text would end the header early).

use std::collections::BTreeMap;
use std::collections::btree_map::Iter;

use super::{HEADER, ResumeError};

/// The header's bytes.
const SIZE: usize = HEADER as usize;

/// The properties of a resume file.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Header {
    entries: BTreeMap<String, String>,
}

impl Header {
    /// An empty header: what a new file, or a file marked invalid, has.
    pub fn new() -> Header {
        Header::default()
    }

    /// The value of `key`.
    pub fn get(&self, key: &str) -> Option<&str> {
        self.entries.get(key).map(String::as_str)
    }

    /// Sets `key` to `value`; returns the value it replaced.
    pub fn set(&mut self, key: impl Into<String>, value: impl Into<String>) -> Option<String> {
        self.entries.insert(key.into(), value.into())
    }

    /// Removes `key`; returns its value.
    pub fn remove(&mut self, key: &str) -> Option<String> {
        self.entries.remove(key)
    }

    /// The entries, in key order.
    pub fn iter(&self) -> Iter<'_, String, String> {
        self.entries.iter()
    }

    /// Whether there are no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// How many bytes the header's text takes. It must not be more than [`HEADER`](super::HEADER).
    pub fn text_len(&self) -> usize {
        self.text().len()
    }

    /// Checks that the header fits its 4096 bytes. A run checks the header it will write last before it invalidates the
    /// one on disk, so that a header too large for the file is found while the old state is still whole.
    pub fn check(&self) -> Result<(), ResumeError> {
        let bytes = self.text_len();
        if bytes > SIZE {
            return Err(ResumeError::HeaderTooLarge { bytes });
        }
        Ok(())
    }

    /// The 4096 bytes of the header: its text, then NULs.
    pub fn encode(&self) -> Result<Vec<u8>, ResumeError> {
        self.check()?;
        let mut bytes = self.text();
        bytes.resize(SIZE, 0);
        Ok(bytes)
    }

    /// The header the bytes hold: the text up to the first NUL.
    pub fn decode(bytes: &[u8]) -> Result<Header, ResumeError> {
        let end = bytes.iter().position(|&byte| byte == 0).unwrap_or(bytes.len());
        parse(&String::from_utf8_lossy(&bytes[..end]))
    }

    fn text(&self) -> Vec<u8> {
        let mut text = String::new();
        for (key, value) in &self.entries {
            escape(&mut text, key, true);
            text.push('=');
            escape(&mut text, value, false);
            text.push('\n');
        }
        text.into_bytes()
    }
}

impl<'a> IntoIterator for &'a Header {
    type Item = (&'a String, &'a String);
    type IntoIter = Iter<'a, String, String>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

fn escape(out: &mut String, text: &str, key: bool) {
    for (index, character) in text.chars().enumerate() {
        match character {
            ' ' if key || index == 0 => out.push_str("\\ "),
            '\\' => out.push_str("\\\\"),
            '\t' => out.push_str("\\t"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\u{c}' => out.push_str("\\f"),
            '=' | ':' | '#' | '!' => {
                out.push('\\');
                out.push(character);
            }
            control if control < ' ' || control == '\u{7f}' => {
                out.push_str(&format!("\\u{:04X}", control as u32));
            }
            other => out.push(other),
        }
    }
}

fn is_blank(character: char) -> bool {
    matches!(character, ' ' | '\t' | '\u{c}')
}

/// The logical lines of `text`: comments and blank lines gone, continuations joined.
fn logical_lines(text: &str) -> Vec<String> {
    let mut lines = Vec::new();
    let mut pending: Option<String> = None;
    let mut natural = Vec::new();
    let mut rest = text;
    while !rest.is_empty() {
        let end = rest.find(['\n', '\r']).unwrap_or(rest.len());
        natural.push(&rest[..end]);
        let mut after = &rest[end..];
        if let Some(stripped) = after.strip_prefix("\r\n") {
            after = stripped;
        } else if !after.is_empty() {
            after = &after[1..];
        }
        rest = after;
    }
    for line in natural {
        let trimmed = line.trim_start_matches(is_blank);
        let mut line = match pending.take() {
            Some(joined) => joined + trimmed,
            None => {
                if trimmed.is_empty() || trimmed.starts_with(['#', '!']) {
                    continue;
                }
                trimmed.to_string()
            }
        };
        let backslashes = line.chars().rev().take_while(|&character| character == '\\').count();
        if backslashes % 2 == 1 {
            line.pop();
            pending = Some(line);
        } else {
            lines.push(line);
        }
    }
    if let Some(last) = pending {
        lines.push(last);
    }
    lines
}

fn parse(text: &str) -> Result<Header, ResumeError> {
    let mut header = Header::new();
    for line in logical_lines(text) {
        let characters: Vec<char> = line.chars().collect();
        let mut key_end = characters.len();
        let mut has_separator = false;
        let mut value_start = characters.len();
        let mut escaped = false;
        for (index, &character) in characters.iter().enumerate() {
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '=' || character == ':' {
                (key_end, value_start, has_separator) = (index, index + 1, true);
                break;
            } else if is_blank(character) {
                (key_end, value_start) = (index, index + 1);
                break;
            }
        }
        while value_start < characters.len() {
            let character = characters[value_start];
            if !is_blank(character) {
                if !has_separator && (character == '=' || character == ':') {
                    has_separator = true;
                } else {
                    break;
                }
            }
            value_start += 1;
        }
        let key = unescape(&characters[..key_end])?;
        let value = unescape(&characters[value_start.min(characters.len())..])?;
        header.set(key, value);
    }
    Ok(header)
}

fn unescape(characters: &[char]) -> Result<String, ResumeError> {
    let mut units: Vec<u16> = Vec::with_capacity(characters.len());
    let mut index = 0;
    let push = |character: char, units: &mut Vec<u16>| {
        let mut buffer = [0u16; 2];
        units.extend_from_slice(character.encode_utf16(&mut buffer));
    };
    while index < characters.len() {
        let character = characters[index];
        index += 1;
        if character != '\\' {
            push(character, &mut units);
            continue;
        }
        let Some(&escape) = characters.get(index) else {
            break;
        };
        index += 1;
        match escape {
            'u' => {
                let digits = characters.get(index..index + 4).ok_or(ResumeError::HeaderText {
                    why: "a \\u escape with fewer than four digits".to_string(),
                })?;
                let mut unit = 0u16;
                for digit in digits {
                    let value = digit.to_digit(16).ok_or(ResumeError::HeaderText {
                        why: "a \\u escape with a character that is not a hexadecimal digit".to_string(),
                    })?;
                    unit = unit << 4 | value as u16;
                }
                units.push(unit);
                index += 4;
            }
            't' => units.push(9),
            'n' => units.push(10),
            'r' => units.push(13),
            'f' => units.push(12),
            other => push(other, &mut units),
        }
    }
    Ok(String::from_utf16_lossy(&units))
}
