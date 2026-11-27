//! The model: its dimensions, and its weights on the device.

mod config;
pub mod dflash2;
mod inventory;
mod weights;

pub use config::{ArtifactProfile, LayerKind, ModelConfig, Quantization, Speculation};
pub use inventory::{
    TRELLIS_COMPANIONS, drafter_names, expected_names, trellis_logical, validate_base_inventory, validate_inventory,
};
pub use weights::{
    DrafterRole, LayerIndex, MTP_ATTENTION_PACK, MTP_PACK_HALVES, MtpRole, NormKind, Residence, Role, TrellisParts,
    WeightRef, Weights, companion_name, is_companion, object_key,
};
