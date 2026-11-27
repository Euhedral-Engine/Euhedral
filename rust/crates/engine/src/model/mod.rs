//! The model: its dimensions, and its weights on the device.

mod config;
pub(crate) mod flash_config;
mod inventory;
mod weights;

pub use config::{ArtifactProfile, LayerKind, ModelConfig, Quantization, Speculation};
pub use flash_config::{FlashModelConfig, KV_ROW_VALUES};
pub use inventory::{TRELLIS_COMPANIONS, expected_names, trellis_logical, validate_inventory};
pub use weights::{
    LayerIndex, MTP_ATTENTION_PACK, MTP_PACK_HALVES, MtpRole, NormKind, Residence, Role, TrellisParts, WeightRef,
    Weights, companion_name, is_companion, object_key,
};
