use std::ops::Range;

use crate::error::EdrlError;
use crate::flash::config::FlashConfig;
use crate::flash::header::FLASH_HEADER_BYTES;
use crate::flash::header::FlashHeader;
use crate::flash::inventory::{ExpectedInventory, InventoryReport};
use crate::flash::metadata::FlashMetadata;
use crate::flash::tables::FlashTables;
use crate::header::{ReadPlan, Region};
use crate::tokenizer_section::TokenizerSection;
use crate::tokenizer_section::{FOOTER_BYTES, TokenizerFooter};

/// A checked Flash-Next artifact: its header, configuration, tables and tokenizer, with the tables proven to hold
/// exactly the inventory the configuration implies. No tensor bytes are read.
///
/// Call sequence: `ReadPlan::probe_flash` names the header and footer; parse them with [`FlashHeader::parse`] and
/// `TokenizerFooter::parse`; `ReadPlan::for_flash` names the metadata, tables and tokenizer body; hand those bytes
/// to [`FlashArtifact::parse`].
#[derive(Debug, Clone, PartialEq)]
pub struct FlashArtifact {
    header: FlashHeader,
    config: FlashConfig,
    tables: FlashTables,
    tokenizer: Option<TokenizerSection>,
    report: InventoryReport,
}

impl FlashArtifact {
    /// Opens an artifact of `file_len` bytes, fetching each region it needs through `read` (which returns exactly the
    /// bytes of the range it is given): the header and footer, then the metadata, tables and tokenizer body.
    /// Tensor payloads are never read.
    pub fn open(
        file_len: u64,
        mut read: impl FnMut(Range<u64>) -> Result<Vec<u8>, EdrlError>,
    ) -> Result<FlashArtifact, EdrlError> {
        let mut fetch = |region: &'static str, range: Range<u64>| {
            let bytes = read(range.clone())?;
            if bytes.len() as u64 != range.end - range.start {
                return Err(EdrlError::Truncated { what: region });
            }
            Ok(bytes)
        };
        let probe = ReadPlan::probe_flash(file_len)?;
        let [head, tail] = probe.reads() else {
            return Err(EdrlError::invalid("the probe names two reads"));
        };
        let header = FlashHeader::parse(&array::<FLASH_HEADER_BYTES>(fetch("header", head.range.clone())?)?)?;
        let footer = TokenizerFooter::parse(&array::<FOOTER_BYTES>(fetch("tokenizer footer", tail.range.clone())?)?)?;
        let plan = ReadPlan::for_flash(&header, file_len, footer.as_ref())?;
        let (mut metadata, mut tables, mut body) = (None, None, None);
        for planned in plan.reads() {
            match planned.region {
                Region::Metadata => metadata = Some(fetch("metadata", planned.range.clone())?),
                Region::Table => tables = Some(fetch("tables", planned.range.clone())?),
                Region::TokenizerBody => body = Some(fetch("tokenizer section", planned.range.clone())?),
                Region::Header | Region::TokenizerFooter => {}
            }
        }
        let tokenizer = match (footer, body) {
            (Some(footer), Some(body)) => Some(TokenizerSection::parse(body, &footer)?),
            _ => None,
        };
        let (Some(metadata), Some(tables)) = (metadata, tables) else {
            return Err(EdrlError::invalid("the read plan lacks the metadata or the tables"));
        };
        FlashArtifact::parse(header, &metadata, &tables, tokenizer)
    }

    /// Decodes `metadata` and `tables` (exactly the header's ranges), reads the configuration, and validates the
    /// tables against it. Refuses the first malformed region, or every inventory problem at once.
    pub fn parse(
        header: FlashHeader,
        metadata: &[u8],
        tables: &[u8],
        tokenizer: Option<TokenizerSection>,
    ) -> Result<FlashArtifact, EdrlError> {
        if metadata.len() as u64 != header.metadata_size() {
            return Err(EdrlError::Truncated { what: "metadata" });
        }
        let config = FlashConfig::from_metadata(&FlashMetadata::decode(metadata)?)?;
        let tables = FlashTables::parse(tables, &header)?;
        let report = ExpectedInventory::from_config(&config)?.validate(&tables)?;
        Ok(FlashArtifact {
            header,
            config,
            tables,
            tokenizer,
            report,
        })
    }

    pub fn header(&self) -> &FlashHeader {
        &self.header
    }

    pub fn config(&self) -> &FlashConfig {
        &self.config
    }

    pub fn tables(&self) -> &FlashTables {
        &self.tables
    }

    /// The packaged tokenizer, when the file carries one.
    pub fn tokenizer(&self) -> Option<&TokenizerSection> {
        self.tokenizer.as_ref()
    }

    /// The object counts the validation found.
    pub fn report(&self) -> InventoryReport {
        self.report
    }
}

fn array<const N: usize>(bytes: Vec<u8>) -> Result<[u8; N], EdrlError> {
    <[u8; N]>::try_from(bytes).map_err(|_| EdrlError::Truncated {
        what: "fixed-size region",
    })
}
