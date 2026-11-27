use std::collections::{BTreeMap, HashSet};
use std::str;

use crate::error::EdrlError;
use crate::header::MAX_METADATA_BYTES;

const MAX_ENTRIES: u32 = 4096;
const MAX_STRING_BYTES: usize = 1 << 20;
const MAX_ARRAY: usize = 1 << 20;

const TYPE_INT: u8 = 0;
const TYPE_REAL: u8 = 1;
const TYPE_BOOL: u8 = 2;
const TYPE_TEXT: u8 = 3;
const TYPE_INTS: u8 = 4;
const TYPE_TEXTS: u8 = 5;

/// One typed metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum MetadataValue {
    Int(i64),
    /// A finite IEEE 754 binary64.
    Real(f64),
    Bool(bool),
    Text(String),
    Ints(Vec<i64>),
    Texts(Vec<String>),
}

impl MetadataValue {
    fn type_name(&self) -> &'static str {
        match self {
            MetadataValue::Int(_) => "int64",
            MetadataValue::Real(_) => "binary64",
            MetadataValue::Bool(_) => "boolean",
            MetadataValue::Text(_) => "string",
            MetadataValue::Ints(_) => "int64 array",
            MetadataValue::Texts(_) => "string array",
        }
    }
}

/// The typed key/value metadata of a version 3 artifact, sorted by key (byte order, which is code point order).
///
/// Encoding, big-endian: `int32 count`, then per entry in key order a length-prefixed UTF-8 key, a type byte and the
/// value: `0` int64, `1` binary64, `2` boolean byte (0 or 1), `3` string (`int32` length, UTF-8), `4` int64 array
/// (`int32` count, values), `5` string array (`int32` count, strings).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct FlashMetadata {
    entries: BTreeMap<String, MetadataValue>,
}

impl FlashMetadata {
    pub fn new() -> FlashMetadata {
        FlashMetadata::default()
    }

    /// Sets `key`, replacing a value it already holds.
    pub fn insert(&mut self, key: impl Into<String>, value: MetadataValue) {
        self.entries.insert(key.into(), value);
    }

    pub fn get(&self, key: &str) -> Option<&MetadataValue> {
        self.entries.get(key)
    }

    /// Removes `key` and returns what it held.
    pub fn remove(&mut self, key: &str) -> Option<MetadataValue> {
        self.entries.remove(key)
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The entries in key order.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &MetadataValue)> {
        self.entries.iter().map(|(key, value)| (key.as_str(), value))
    }

    /// Decodes the metadata region. Keys must be strictly increasing, strings valid UTF-8, doubles finite, booleans
    /// 0 or 1, and no byte may trail the last entry.
    pub fn decode(bytes: &[u8]) -> Result<FlashMetadata, EdrlError> {
        let mut cursor = Cursor { bytes, position: 0 };
        let count = cursor.i32("entry count")?;
        let count = u32::try_from(count)
            .ok()
            .filter(|count| *count <= MAX_ENTRIES)
            .ok_or_else(|| {
                EdrlError::invalid(format!(
                    "metadata entry count is out of range: {count} (0 to {MAX_ENTRIES})"
                ))
            })?;
        let mut entries = BTreeMap::new();
        let mut previous: Option<String> = None;
        for _ in 0..count {
            let key = cursor.string("key")?;
            if previous
                .as_deref()
                .is_some_and(|previous| key.as_bytes() <= previous.as_bytes())
            {
                return Err(EdrlError::invalid(format!(
                    "metadata keys are not in strictly increasing order at '{key}'"
                )));
            }
            let value = cursor.value(&key)?;
            previous = Some(key.clone());
            entries.insert(key, value);
        }
        if cursor.remaining() != 0 {
            return Err(EdrlError::invalid(format!(
                "metadata contains {} trailing bytes after its {count} entries",
                cursor.remaining()
            )));
        }
        Ok(FlashMetadata { entries })
    }

    /// The metadata region for these entries, which [`FlashMetadata::decode`] reads back. Refuses what the decoder
    /// would: too many entries, a long string or array, a non-finite double, a region over the size limit.
    pub fn encode(&self) -> Result<Vec<u8>, EdrlError> {
        let count = u32::try_from(self.entries.len())
            .ok()
            .filter(|count| *count <= MAX_ENTRIES)
            .ok_or_else(|| {
                EdrlError::invalid(format!(
                    "metadata has {} entries, over the limit {MAX_ENTRIES}",
                    self.entries.len()
                ))
            })?;
        let mut out = Vec::new();
        out.extend_from_slice(&count.to_be_bytes());
        for (key, value) in &self.entries {
            put_string(&mut out, key, key)?;
            match value {
                MetadataValue::Int(number) => {
                    out.push(TYPE_INT);
                    out.extend_from_slice(&number.to_be_bytes());
                }
                MetadataValue::Real(number) => {
                    if !number.is_finite() {
                        return Err(EdrlError::invalid(format!("metadata {key} is not finite")));
                    }
                    out.push(TYPE_REAL);
                    out.extend_from_slice(&number.to_be_bytes());
                }
                MetadataValue::Bool(flag) => {
                    out.push(TYPE_BOOL);
                    out.push(u8::from(*flag));
                }
                MetadataValue::Text(text) => {
                    out.push(TYPE_TEXT);
                    put_string(&mut out, key, text)?;
                }
                MetadataValue::Ints(numbers) => {
                    out.push(TYPE_INTS);
                    put_count(&mut out, key, numbers.len())?;
                    for number in numbers {
                        out.extend_from_slice(&number.to_be_bytes());
                    }
                }
                MetadataValue::Texts(texts) => {
                    out.push(TYPE_TEXTS);
                    put_count(&mut out, key, texts.len())?;
                    for text in texts {
                        put_string(&mut out, key, text)?;
                    }
                }
            }
        }
        if out.len() as u64 > MAX_METADATA_BYTES {
            return Err(EdrlError::invalid(format!(
                "metadata of {} bytes is over the limit {MAX_METADATA_BYTES}",
                out.len()
            )));
        }
        Ok(out)
    }
}

fn put_count(out: &mut Vec<u8>, key: &str, count: usize) -> Result<(), EdrlError> {
    if count > MAX_ARRAY {
        return Err(EdrlError::invalid(format!(
            "metadata '{key}' array has {count} elements, over the limit {MAX_ARRAY}"
        )));
    }
    out.extend_from_slice(&(count as u32).to_be_bytes());
    Ok(())
}

fn put_string(out: &mut Vec<u8>, key: &str, text: &str) -> Result<(), EdrlError> {
    if text.len() > MAX_STRING_BYTES {
        return Err(EdrlError::invalid(format!(
            "metadata '{key}' string has {} bytes, over the limit {MAX_STRING_BYTES}",
            text.len()
        )));
    }
    out.extend_from_slice(&(text.len() as u32).to_be_bytes());
    out.extend_from_slice(text.as_bytes());
    Ok(())
}

/// A big-endian cursor whose errors name the metadata key being read.
struct Cursor<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl Cursor<'_> {
    fn remaining(&self) -> usize {
        self.bytes.len() - self.position
    }

    fn take(&mut self, length: usize, field: &str) -> Result<&[u8], EdrlError> {
        if length > self.remaining() {
            return Err(EdrlError::invalid(format!(
                "metadata is truncated while reading '{field}'"
            )));
        }
        let taken = &self.bytes[self.position..self.position + length];
        self.position += length;
        Ok(taken)
    }

    fn array<const N: usize>(&mut self, field: &str) -> Result<[u8; N], EdrlError> {
        let mut out = [0; N];
        out.copy_from_slice(self.take(N, field)?);
        Ok(out)
    }

    fn i32(&mut self, field: &str) -> Result<i32, EdrlError> {
        Ok(i32::from_be_bytes(self.array(field)?))
    }

    fn i64(&mut self, field: &str) -> Result<i64, EdrlError> {
        Ok(i64::from_be_bytes(self.array(field)?))
    }

    fn length(&mut self, field: &str, max: usize, what: &str) -> Result<usize, EdrlError> {
        let length = self.i32(field)?;
        usize::try_from(length)
            .ok()
            .filter(|length| *length <= max)
            .ok_or_else(|| {
                EdrlError::invalid(format!(
                    "metadata '{field}' {what} is out of range: {length} (0 to {max})"
                ))
            })
    }

    fn string(&mut self, field: &str) -> Result<String, EdrlError> {
        let length = self.length(field, MAX_STRING_BYTES, "string length")?;
        let bytes = self.take(length, field)?;
        str::from_utf8(bytes)
            .map(str::to_owned)
            .map_err(|_| EdrlError::invalid(format!("metadata '{field}' is not valid UTF-8")))
    }

    fn value(&mut self, key: &str) -> Result<MetadataValue, EdrlError> {
        let [kind] = self.array::<1>(key)?;
        Ok(match kind {
            TYPE_INT => MetadataValue::Int(self.i64(key)?),
            TYPE_REAL => {
                let number = f64::from_be_bytes(self.array(key)?);
                if !number.is_finite() {
                    return Err(EdrlError::invalid(format!("metadata {key} is not finite")));
                }
                MetadataValue::Real(number)
            }
            TYPE_BOOL => match self.array::<1>(key)? {
                [0] => MetadataValue::Bool(false),
                [1] => MetadataValue::Bool(true),
                [other] => {
                    return Err(EdrlError::invalid(format!(
                        "metadata {key} boolean byte is {other}, expected 0 or 1"
                    )));
                }
            },
            TYPE_TEXT => MetadataValue::Text(self.string(key)?),
            TYPE_INTS => {
                let length = self.length(key, MAX_ARRAY, "length")?;
                // The capacity is bounded by what the bytes can hold, so a hostile length allocates nothing large.
                let mut numbers = Vec::with_capacity(length.min(self.remaining() / 8));
                for _ in 0..length {
                    numbers.push(self.i64(key)?);
                }
                MetadataValue::Ints(numbers)
            }
            TYPE_TEXTS => {
                let length = self.length(key, MAX_ARRAY, "length")?;
                let mut texts = Vec::with_capacity(length.min(self.remaining() / 4));
                for _ in 0..length {
                    texts.push(self.string(key)?);
                }
                MetadataValue::Texts(texts)
            }
            unknown => return Err(EdrlError::invalid(format!("metadata {key} has unknown type {unknown}"))),
        })
    }
}

/// Typed access to decoded metadata that tracks which keys were read, so [`MetadataReader::finish`] can refuse a key
/// the reader does not know.
#[derive(Debug)]
pub struct MetadataReader<'a> {
    metadata: &'a FlashMetadata,
    used: HashSet<&'a str>,
}

impl<'a> MetadataReader<'a> {
    pub fn new(metadata: &'a FlashMetadata) -> MetadataReader<'a> {
        MetadataReader {
            metadata,
            used: HashSet::new(),
        }
    }

    fn take(&mut self, key: &str) -> Result<&'a MetadataValue, EdrlError> {
        let Some((stored, value)) = self.metadata.entries.get_key_value(key) else {
            return Err(EdrlError::invalid(format!("metadata is missing '{key}'")));
        };
        self.used.insert(stored.as_str());
        Ok(value)
    }

    fn wrong_type(key: &str, value: &MetadataValue, expected: &str) -> EdrlError {
        EdrlError::invalid(format!(
            "metadata '{key}' has type {}, expected {expected}",
            value.type_name()
        ))
    }

    pub fn integer(&mut self, key: &str) -> Result<i64, EdrlError> {
        match self.take(key)? {
            MetadataValue::Int(number) => Ok(*number),
            other => Err(Self::wrong_type(key, other, "int64")),
        }
    }

    /// A non-negative value that fits an `i32`.
    pub fn count(&mut self, key: &str) -> Result<u32, EdrlError> {
        let value = self.integer(key)?;
        i32::try_from(value)
            .ok()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| EdrlError::invalid(format!("metadata '{key}' is out of range: {value}")))
    }

    /// A signed value that fits an `i32`.
    pub fn signed(&mut self, key: &str) -> Result<i32, EdrlError> {
        let value = self.integer(key)?;
        i32::try_from(value).map_err(|_| EdrlError::invalid(format!("metadata '{key}' is out of range: {value}")))
    }

    pub fn real(&mut self, key: &str) -> Result<f64, EdrlError> {
        match self.take(key)? {
            MetadataValue::Real(number) => Ok(*number),
            other => Err(Self::wrong_type(key, other, "binary64")),
        }
    }

    pub fn boolean(&mut self, key: &str) -> Result<bool, EdrlError> {
        match self.take(key)? {
            MetadataValue::Bool(flag) => Ok(*flag),
            other => Err(Self::wrong_type(key, other, "boolean")),
        }
    }

    pub fn string(&mut self, key: &str) -> Result<String, EdrlError> {
        match self.take(key)? {
            MetadataValue::Text(text) => Ok(text.clone()),
            other => Err(Self::wrong_type(key, other, "string")),
        }
    }

    pub fn longs(&mut self, key: &str) -> Result<Vec<i64>, EdrlError> {
        match self.take(key)? {
            MetadataValue::Ints(numbers) => Ok(numbers.clone()),
            other => Err(Self::wrong_type(key, other, "int64 array")),
        }
    }

    /// An int64 array whose elements each fit an `i32`.
    pub fn ints(&mut self, key: &str) -> Result<Vec<i32>, EdrlError> {
        self.longs(key)?
            .into_iter()
            .map(|value| {
                i32::try_from(value)
                    .map_err(|_| EdrlError::invalid(format!("metadata '{key}' has an out-of-range element {value}")))
            })
            .collect()
    }

    pub fn strings(&mut self, key: &str) -> Result<Vec<String>, EdrlError> {
        match self.take(key)? {
            MetadataValue::Texts(texts) => Ok(texts.clone()),
            other => Err(Self::wrong_type(key, other, "string array")),
        }
    }

    /// Refuses the metadata when it holds a key that was never read.
    pub fn finish(self) -> Result<(), EdrlError> {
        let unknown: Vec<&str> = self
            .metadata
            .entries
            .keys()
            .map(String::as_str)
            .filter(|key| !self.used.contains(key))
            .collect();
        if unknown.is_empty() {
            Ok(())
        } else {
            Err(EdrlError::invalid(format!("metadata has unknown keys: {unknown:?}")))
        }
    }
}
