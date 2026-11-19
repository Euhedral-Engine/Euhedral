//! Encoders for the regions an artifact is made of. Each is a pure function from a description to bytes, held to
//! the rules the parsers enforce, so what it returns reads back as what went in. Nothing is written: the caller
//! puts the bytes where [`ArtifactPlan`] says (through whatever I/O it owns).

use std::collections::HashSet;

use crate::error::EdrlError;
use crate::format::{TensorDataType, WeightFormat, WeightLayout, accepts_byte_size};
use crate::header::{HEADER_BYTES, Header, MAX_COUNT, MAX_METADATA_BYTES};
use crate::metadata::{MAX_DRAFT_DEPTH, ModelMetadata};
use crate::table::{MAX_NAME_BYTES, MAX_RANK, TensorEntry, validate_non_overlapping};
use crate::tokenizer_section::{FOOTER_BYTES, TOKENIZER_ENTRIES};

/// The alignment of each payload in a written artifact: a block multiple, so a payload can be read past the page
/// cache.
pub const PAYLOAD_ALIGNMENT: u64 = 4096;

const MAX_STRING_BYTES: usize = 1 << 20;
const TOKENIZER_VERSION: u32 = 1;
const TOKENIZER_MAGIC: &[u8; 8] = b"EDRLTOKN";

/// One object to place: everything the table stores about it except where it goes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ObjectSpec {
    pub name: String,
    pub shape: Vec<u64>,
    pub dtype: TensorDataType,
    pub format: WeightFormat,
    pub layout: WeightLayout,
    /// The payload's size.
    pub bytes: u64,
}

/// Where everything goes in an artifact that holds `specs` and metadata of a given size.
///
/// The header, the metadata and the table are contiguous from byte 0. The data offset is where the table ends. Each
/// payload starts at the next multiple of the alignment at or after the previous payload's end (the first, after the
/// data offset), in the order of `specs`. The table's size does not depend on the offsets, so the plan is one pass.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArtifactPlan {
    header: Header,
    entries: Vec<TensorEntry>,
    content_end: u64,
}

impl ArtifactPlan {
    /// Places `specs` after metadata of `metadata_size` bytes, each payload on a multiple of `alignment` (at least
    /// 1; [`PAYLOAD_ALIGNMENT`] for an artifact the engine reads).
    pub fn new(metadata_size: u64, specs: &[ObjectSpec], alignment: u64) -> Result<ArtifactPlan, EdrlError> {
        if alignment == 0 {
            return Err(EdrlError::invalid("the payload alignment must be at least 1"));
        }
        let count = u32::try_from(specs.len())
            .ok()
            .filter(|count| *count <= MAX_COUNT)
            .ok_or_else(|| EdrlError::invalid("tensor count is outside the supported range"))?;
        let mut table_bytes: u64 = 0;
        for spec in specs {
            table_bytes += entry_bytes(&spec.name, spec.shape.len());
        }
        let data_offset = checked_add(checked_add(HEADER_BYTES as u64, metadata_size)?, table_bytes)?;
        let header = Header::new(metadata_size, count, data_offset)?;
        let mut entries = Vec::with_capacity(specs.len());
        let mut content_end = data_offset;
        let mut next = align_up(data_offset, alignment)?;
        for spec in specs {
            entries.push(TensorEntry::new(
                spec.name.clone(),
                spec.shape.clone(),
                spec.dtype,
                spec.format,
                spec.layout,
                next,
                spec.bytes,
            ));
            let end = checked_add(next, spec.bytes)?;
            content_end = content_end.max(end);
            next = align_up(end, alignment)?;
        }
        encode_table(&entries)?;
        Ok(ArtifactPlan {
            header,
            entries,
            content_end,
        })
    }

    pub fn header(&self) -> &Header {
        &self.header
    }

    /// The table's entries, with their offsets, in the order of the specs.
    pub fn entries(&self) -> &[TensorEntry] {
        &self.entries
    }

    /// The table's bytes. They are exactly [`Header::table_range`] long.
    pub fn table(&self) -> Result<Vec<u8>, EdrlError> {
        encode_table(&self.entries)
    }

    /// Where the artifact's content ends and its tokenizer section begins: the end of the last payload, or the data
    /// offset when none is longer.
    pub fn content_end(&self) -> u64 {
        self.content_end
    }
}

/// The table for `entries`, in their order: per entry `int32 nameLength, UTF-8 name, int32 rank, int64 dimension
/// per axis, int32 data type, int32 format, int32 layout, int64 offset, int64 size`, big-endian.
///
/// Refused as [`TensorTable::parse`](crate::TensorTable::parse) refuses: a name that is empty, too long or repeated,
/// a rank above 64, a size the shape, format and layout do not allow, and payloads that overlap. An offset against
/// the header (the data offset) is not checked here: the plan places the payloads.
pub fn encode_table(entries: &[TensorEntry]) -> Result<Vec<u8>, EdrlError> {
    if entries.len() > MAX_COUNT as usize {
        return Err(EdrlError::invalid("tensor count is outside the supported range"));
    }
    let mut names = HashSet::with_capacity(entries.len());
    let mut out = Vec::new();
    for entry in entries {
        let name = entry.name();
        if !(1..=MAX_NAME_BYTES).contains(&name.len()) {
            return Err(EdrlError::invalid(format!(
                "tensor name length is outside the supported range: {}",
                name.len()
            )));
        }
        if !names.insert(name) {
            return Err(EdrlError::DuplicateName { name: name.to_owned() });
        }
        if i32::try_from(entry.shape().len()).is_ok_and(|rank| rank > MAX_RANK) {
            return Err(EdrlError::invalid("tensor rank is outside the supported range"));
        }
        match accepts_byte_size(
            entry.shape(),
            entry.dtype(),
            entry.format(),
            entry.layout(),
            entry.bytes(),
        ) {
            Ok(true) => {}
            Ok(false) => {
                return Err(EdrlError::SizeMismatch {
                    name: name.to_owned(),
                    bytes: entry.bytes(),
                });
            }
            Err(reason) => {
                return Err(EdrlError::UnsupportedTensor {
                    name: name.to_owned(),
                    reason,
                });
            }
        }
        put_i32(&mut out, length(name.len())?);
        out.extend_from_slice(name.as_bytes());
        put_i32(&mut out, length(entry.shape().len())?);
        for dimension in entry.shape() {
            put_i64(&mut out, signed(*dimension)?);
        }
        put_i32(&mut out, entry.dtype().ordinal());
        put_i32(&mut out, entry.format().ordinal());
        put_i32(&mut out, entry.layout().ordinal());
        put_i64(&mut out, signed(entry.offset())?);
        put_i64(&mut out, signed(entry.bytes())?);
    }
    validate_non_overlapping(entries)?;
    Ok(out)
}

/// The metadata region for `metadata`, big-endian, in the order [`Metadata::parse`](crate::Metadata::parse) reads
/// it: the draft depth, when there is one, ends it.
pub fn encode_metadata(metadata: &ModelMetadata) -> Result<Vec<u8>, EdrlError> {
    let counts = [
        ("vocab size", metadata.vocab_size),
        ("hidden size", metadata.hidden_size),
        ("hidden layer count", metadata.num_hidden_layers),
        ("attention head count", metadata.num_attention_heads),
        ("key/value head count", metadata.num_key_value_heads),
        ("attention head dimension", metadata.attention_head_dim),
        ("intermediate size", metadata.intermediate_size),
        ("linear key head count", metadata.linear_num_key_heads),
        ("linear value head count", metadata.linear_num_value_heads),
        ("linear key head dimension", metadata.linear_key_head_dim),
        ("linear value head dimension", metadata.linear_value_head_dim),
        ("linear convolution kernel dimension", metadata.linear_conv_kernel_dim),
        ("maximum position embeddings", metadata.max_position_embeddings),
        ("expert count", metadata.num_experts),
        ("experts per token", metadata.num_experts_per_token),
        ("MoE intermediate size", metadata.moe_intermediate_size),
        (
            "shared expert intermediate size",
            metadata.shared_expert_intermediate_size,
        ),
        ("MTP layer count", metadata.mtp_layer_count),
    ];
    for (what, value) in counts {
        if value > MAX_COUNT {
            return Err(EdrlError::invalid(format!(
                "{what} is outside the supported range: {value}"
            )));
        }
    }
    let floats = [
        metadata.rms_norm_epsilon,
        metadata.rope_theta,
        metadata.partial_rotary_factor,
    ];
    if floats.iter().any(|value| !value.is_finite()) {
        return Err(EdrlError::invalid("config floating-point values must be finite"));
    }
    if metadata.layer_types.len() != metadata.num_hidden_layers as usize {
        return Err(EdrlError::invalid("layer type count does not match hidden layer count"));
    }
    if metadata.hidden_activation.len() > MAX_STRING_BYTES {
        return Err(EdrlError::invalid("hidden activation is too long"));
    }
    if let Some(depth) = metadata.draft_depth
        && !(1..=MAX_DRAFT_DEPTH).contains(&depth)
    {
        return Err(EdrlError::invalid("MTP draft depth is outside the supported range"));
    }

    let mut out = Vec::new();
    let m = metadata;
    for value in [
        m.vocab_size,
        m.hidden_size,
        m.num_hidden_layers,
        m.num_attention_heads,
        m.num_key_value_heads,
        m.attention_head_dim,
        m.intermediate_size,
        m.linear_num_key_heads,
        m.linear_num_value_heads,
        m.linear_key_head_dim,
        m.linear_value_head_dim,
        m.linear_conv_kernel_dim,
    ] {
        put_count(&mut out, value)?;
    }
    for value in floats {
        out.extend_from_slice(&value.to_be_bytes());
    }
    put_count(&mut out, m.max_position_embeddings)?;
    put_i32(&mut out, length(m.hidden_activation.len())?);
    out.extend_from_slice(m.hidden_activation.as_bytes());
    put_i32(&mut out, length(m.layer_types.len())?);
    for layer in &m.layer_types {
        put_i32(&mut out, *layer as i32);
    }
    for value in [
        m.num_experts,
        m.num_experts_per_token,
        m.moe_intermediate_size,
        m.shared_expert_intermediate_size,
    ] {
        put_count(&mut out, value)?;
    }
    out.push(u8::from(m.tie_word_embeddings));
    out.push(u8::from(m.attention_output_gate));
    put_count(&mut out, m.mtp_layer_count)?;
    if let Some(depth) = m.draft_depth {
        put_i32(&mut out, i32::from(depth));
    }
    if out.len() as u64 > MAX_METADATA_BYTES {
        return Err(EdrlError::invalid("metadata is too large"));
    }
    Ok(out)
}

/// The four texts of a tokenizer section, in the order the section stores them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TokenizerTexts<'a> {
    pub tokenizer: &'a str,
    pub tokenizer_config: &'a str,
    pub chat_template: &'a str,
    pub special_tokens: &'a str,
}

/// The tokenizer section holding `texts`: the body, then the [`FOOTER_BYTES`]-byte footer (body length, CRC-32 of the
/// body, version, magic). A writer puts the body first and the footer last, so an interrupted write leaves no
/// section: the body is `bytes[..len - FOOTER_BYTES]`.
pub fn encode_tokenizer_section(texts: &TokenizerTexts<'_>) -> Result<Vec<u8>, EdrlError> {
    let data = [
        texts.tokenizer,
        texts.tokenizer_config,
        texts.chat_template,
        texts.special_tokens,
    ];
    let mut out = Vec::new();
    put_i32(&mut out, length(TOKENIZER_ENTRIES.len())?);
    for (name, text) in TOKENIZER_ENTRIES.iter().zip(data) {
        for part in [name.as_bytes(), text.as_bytes()] {
            put_i32(&mut out, length(part.len())?);
            out.extend_from_slice(part);
        }
    }
    let body_length = out.len() as u64;
    if body_length > 1 << 30 {
        return Err(EdrlError::invalid("the tokenizer section is too large"));
    }
    let crc = crc32fast::hash(&out);
    put_i64(&mut out, signed(body_length)?);
    out.extend_from_slice(&crc.to_be_bytes());
    out.extend_from_slice(&TOKENIZER_VERSION.to_be_bytes());
    out.extend_from_slice(TOKENIZER_MAGIC);
    debug_assert_eq!(out.len() as u64, body_length + FOOTER_BYTES as u64);
    Ok(out)
}

/// The bytes one table entry takes: its size depends on the name and the rank, not on the offsets.
fn entry_bytes(name: &str, rank: usize) -> u64 {
    // name length, name, rank, three ordinals, offset and size: 4 + name + 4 + 8 per axis + 12 + 16.
    (4 + name.len() + 4 + 8 * rank + 12 + 16) as u64
}

fn align_up(value: u64, alignment: u64) -> Result<u64, EdrlError> {
    checked_add(value, alignment - 1).map(|sum| sum / alignment * alignment)
}

fn checked_add(left: u64, right: u64) -> Result<u64, EdrlError> {
    left.checked_add(right)
        .ok_or_else(|| EdrlError::invalid("artifact layout overflows"))
}

fn length(value: usize) -> Result<i32, EdrlError> {
    i32::try_from(value).map_err(|_| EdrlError::invalid("a length does not fit the format"))
}

fn signed(value: u64) -> Result<i64, EdrlError> {
    i64::try_from(value).map_err(|_| EdrlError::invalid("a value does not fit the format"))
}

fn put_i32(out: &mut Vec<u8>, value: i32) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_i64(out: &mut Vec<u8>, value: i64) {
    out.extend_from_slice(&value.to_be_bytes());
}

fn put_count(out: &mut Vec<u8>, value: u32) -> Result<(), EdrlError> {
    put_i32(
        out,
        i32::try_from(value).map_err(|_| EdrlError::invalid("a count does not fit the format"))?,
    );
    Ok(())
}
