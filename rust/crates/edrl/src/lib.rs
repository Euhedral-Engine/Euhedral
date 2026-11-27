//! Parser for `.edrl` model artifacts.
//!
//! An artifact is a 48-byte header, the model metadata, a tensor table, 4096-byte-aligned tensor payloads and an
//! optional tokenizer section at the very end of the file. This crate parses each region from bytes the caller
//! supplies; it performs no I/O. A loader asks [`ReadPlan`] which byte ranges to fetch (with whatever asynchronous
//! I/O it owns), hands them to the `parse` functions, and then reads each tensor payload at the offset its
//! [`TensorEntry`] names.
//!
//! The same crate encodes those regions: [`encode_metadata`], [`encode_table`], [`Header::encode`] and
//! [`encode_tokenizer_section`] produce bytes the parsers read back, and [`ArtifactPlan`] places the payloads. They
//! write nothing: the caller puts the bytes where the plan says.
//!
//! A Flash-Next artifact (container version 3, 64-byte header) is read by [`FlashArtifact`]: [`ModelFamily::detect`]
//! tells which reader a file needs, and [`ExpectedInventory`] proves its tables hold exactly the objects its
//! configuration implies.
//!
//! The reference is the Java reader (`ArtifactReader`, `ArtifactCodec`, `TokenizerSection`): every check it makes is
//! made here, on the same bytes. All integers are big-endian, except the payloads: the trellis `.segments`
//! companion holds little-endian `int32` values (see [`TrellisCompanions`]).

mod error;
mod family;
mod flash;
mod format;
mod header;
mod metadata;
mod reader;
mod table;
#[cfg(feature = "test-util")]
pub mod tiny;
#[cfg(feature = "test-util")]
pub mod tiny_flash;
mod tokenizer_section;
mod writer;

pub use error::EdrlError;
pub use family::{DETECT_BYTES, ModelFamily};
pub use flash::artifact::FlashArtifact;
pub use flash::config::{
    ARCHITECTURE, AttentionConfig, FlashConfig, FlashLayerType, GdnConfig, HyperConnectionConfig, MoeConfig, MtpConfig,
    NgramConfig, PleConfig, QsaConfig, TextConfig, VisionConfig,
};
pub use flash::header::{
    ARCHITECTURE_QWEN4_EXP, DATA_ALIGNMENT, FLASH_HEADER_BYTES, FLASH_VERSION, FlashHeader, OBJECT_ALIGNMENT,
};
pub use flash::inventory::{
    ExpectedBank, ExpectedInventory, ExpectedTensor, InventoryProblem, InventoryReport, shard_name,
};
pub use flash::metadata::{FlashMetadata, MetadataReader, MetadataValue};
pub use flash::ngram::NgramLayout;
pub use flash::tables::{
    ComponentGroup, ExpertBank, ExpertProjection, ExpertRecord, FlashTables, FlashTensor, encode_flash_tables,
};
pub use format::{TensorDataType, TrellisCompanions, WeightFormat, WeightLayout};
pub use header::{HEADER_BYTES, Header, PlannedRead, ReadPlan, Region};
pub use metadata::{LayerType, MAX_DRAFT_DEPTH, Metadata, ModelMetadata};
pub use table::{TensorEntry, TensorTable};
pub use tokenizer_section::{FOOTER_BYTES, TOKENIZER_ENTRIES, TokenizerFooter, TokenizerSection, content_end};
pub use writer::{
    ArtifactPlan, ObjectSpec, PAYLOAD_ALIGNMENT, TokenizerTexts, encode_metadata, encode_table,
    encode_tokenizer_section,
};
