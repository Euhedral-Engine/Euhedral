//! Parser for `.edrl` model artifacts.
//!
//! An artifact is a 48-byte header, the model metadata, a tensor table, 4096-byte-aligned tensor payloads and an
//! optional tokenizer section at the very end of the file. This crate parses each region from bytes the caller
//! supplies; it performs no I/O. A loader asks [`ReadPlan`] which byte ranges to fetch (with whatever asynchronous
//! I/O it owns), hands them to the `parse` functions, and then reads each tensor payload at the offset its
//! [`TensorEntry`] names.
//!
//! The reference is the Java reader (`ArtifactReader`, `ArtifactCodec`, `TokenizerSection`): every check it makes is
//! made here, on the same bytes. All integers are big-endian, except the payloads: the trellis `.segments`
//! companion holds little-endian `int32` values (see [`TrellisCompanions`]).

mod error;
mod format;
mod header;
mod metadata;
mod reader;
mod table;
mod tokenizer_section;

pub use error::EdrlError;
pub use format::{TensorDataType, TrellisCompanions, WeightFormat, WeightLayout};
pub use header::{HEADER_BYTES, Header, PlannedRead, ReadPlan, Region};
pub use metadata::{LayerType, MAX_DRAFT_DEPTH, Metadata, ModelMetadata};
pub use table::{TensorEntry, TensorTable};
pub use tokenizer_section::{FOOTER_BYTES, TOKENIZER_ENTRIES, TokenizerFooter, TokenizerSection, content_end};
