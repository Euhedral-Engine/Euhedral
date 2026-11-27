use crate::error::EdrlError;
use crate::header::MAGIC;
use crate::reader::Reader;

/// How many leading bytes [`ModelFamily::detect`] reads: magic, version and architecture.
pub const DETECT_BYTES: usize = 12;

const DENSE_VERSION: u32 = 2;
const FLASH_NEXT_VERSION: u32 = 3;
const FLASH_NEXT_ARCHITECTURE: u32 = 1;

/// The model family an artifact holds, which decides the reader a caller dispatches to.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelFamily {
    /// Dense Qwen (container version 2): read with [`Header`](crate::Header) and [`TensorTable`](crate::TensorTable).
    DenseV2,
    /// Flash-Next (`qwen4_exp`, container version 3): read with [`FlashHeader`](crate::FlashHeader) and
    /// [`FlashArtifact`](crate::FlashArtifact).
    FlashNextV3,
}

impl ModelFamily {
    /// The family of an artifact from its first [`DETECT_BYTES`] bytes (`ModelArchitecture.detect`). A version 3 file
    /// names its architecture; version 2 has none.
    pub fn detect(head: &[u8; DETECT_BYTES]) -> Result<ModelFamily, EdrlError> {
        let mut reader = Reader::new(head);
        let magic = reader.u32("artifact magic")?;
        if magic != MAGIC {
            return Err(EdrlError::BadMagic { found: magic });
        }
        let version = reader.u32("artifact version")?;
        let architecture = reader.u32("artifact architecture")?;
        match version {
            DENSE_VERSION => Ok(ModelFamily::DenseV2),
            FLASH_NEXT_VERSION if architecture == FLASH_NEXT_ARCHITECTURE => Ok(ModelFamily::FlashNextV3),
            FLASH_NEXT_VERSION => Err(EdrlError::UnsupportedArchitecture { found: architecture }),
            _ => Err(EdrlError::UnsupportedVersion { found: version }),
        }
    }
}
