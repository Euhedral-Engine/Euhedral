//! What a Flash-Next start decides before it touches the device: the artifact's model family, how many sequences it
//! holds and the host budget its plan is made against. Each is a pure function over what the startup read, so a
//! refusal is a returned value that names the setting and its value.

use euhedral_inference_edrl::{DETECT_BYTES, EdrlError, ModelFamily};
use euhedral_inference_settings::{BatchingSettings, MemorySettings, Settings};

use crate::error::EngineError;
use crate::residency::budget::HostBudget;

/// The sequences a Flash-Next engine holds: one, its state being a single sequence's. `batching.max-sequences` 0
/// (automatic) gives 1; 1 is accepted; a larger value is refused with the value (never clamped).
pub fn one_sequence(batching: &BatchingSettings) -> Result<usize, EngineError> {
    match batching.max_sequences {
        0 | 1 => Ok(1),
        stated => Err(EngineError::Runtime(format!(
            "batching.max-sequences is {stated}, but a Flash-Next engine serves one sequence at a time; set it to 1 \
             (or leave it automatic)"
        ))),
    }
}

/// The compute streams a Flash-Next engine opens: `memory.weight-streaming.gpu-streams`, at least 1. Zero is refused,
/// not raised.
pub fn compute_streams(memory: &MemorySettings) -> Result<usize, EngineError> {
    match memory.weight_streaming.gpu_streams {
        0 => Err(EngineError::Runtime(
            "memory.weight-streaming.gpu-streams is 0: a step needs at least 1 stream".to_string(),
        )),
        streams => Ok(streams as usize),
    }
}

/// Refuses the settings a Flash-Next engine cannot serve, before anything is opened or allocated: the sequences
/// ([`one_sequence`]) and the streams ([`compute_streams`], returned).
pub fn check_settings(settings: &Settings) -> Result<usize, EngineError> {
    one_sequence(&settings.batching)?;
    compute_streams(&settings.memory)
}

/// The family of the artifact whose first bytes are `head`, or the refusal that names what the file is: a magic, a
/// container version or an architecture this engine does not serve.
pub fn family_of(head: &[u8; DETECT_BYTES]) -> Result<ModelFamily, EngineError> {
    ModelFamily::detect(head).map_err(|error| {
        EngineError::Runtime(match error {
            EdrlError::BadMagic { found } => format!("the artifact is not an EDRL file: its magic is {found:#010x}"),
            EdrlError::UnsupportedVersion { found } => {
                format!(
                    "the artifact's container version {found} is not one this engine serves (2: dense, 3: Flash-Next)"
                )
            }
            EdrlError::UnsupportedArchitecture { found } => {
                format!("the artifact's model family (architecture {found}) is not served: only qwen4_exp (1) is")
            }
            other => other.to_string(),
        })
    })
}

/// The host budget a plan is made against: `memory.host-limit` as stated, else what the machine can spare
/// (`available`, `None` when it does not report it) less `memory.host-margin`. The margin covers the runtime's own
/// use, so no reserve is taken.
pub fn host_budget(memory: &MemorySettings, available: Option<u64>) -> HostBudget {
    let stated = memory.host_limit.value().map(|size| size.bytes());
    HostBudget::of(stated, available, memory.host_margin, 0)
}

#[cfg(test)]
mod tests {
    use euhedral_inference_settings::{Auto, ByteSize, ModelSettings};

    use crate::residency::budget::BudgetSource;

    use super::*;

    /// The container magic every artifact starts with.
    const MAGIC: u32 = 0x5157_454E;

    fn head(version: u32, architecture: u32) -> [u8; DETECT_BYTES] {
        let mut bytes = [0u8; DETECT_BYTES];
        bytes[..4].copy_from_slice(&MAGIC.to_be_bytes());
        bytes[4..8].copy_from_slice(&version.to_be_bytes());
        bytes[8..].copy_from_slice(&architecture.to_be_bytes());
        bytes
    }

    #[test]
    fn a_flash_next_head_selects_its_family_and_an_unknown_architecture_is_refused_by_value() {
        assert_eq!(family_of(&head(3, 1)).expect("flash"), ModelFamily::FlashNextV3);
        let refused = family_of(&head(3, 7)).expect_err("unknown architecture").to_string();
        assert!(refused.contains("architecture 7"), "{refused}");
    }

    #[test]
    fn a_foreign_version_or_magic_is_refused_naming_it() {
        let version = family_of(&head(9, 0)).expect_err("version").to_string();
        assert!(version.contains("version 9"), "{version}");
        let mut foreign = head(3, 1);
        foreign[0] ^= 0xff;
        assert!(
            family_of(&foreign)
                .expect_err("magic")
                .to_string()
                .contains("not an EDRL file")
        );
    }

    #[test]
    fn automatic_sequences_are_one_and_more_than_one_is_refused_not_clamped() {
        let mut settings = Settings::with_model(ModelSettings::default());
        for stated in [0, 1] {
            settings.batching.max_sequences = stated;
            assert_eq!(one_sequence(&settings.batching).expect("one"), 1);
        }
        settings.batching.max_sequences = 4;
        let refused = one_sequence(&settings.batching).expect_err("refused").to_string();
        assert!(refused.contains("batching.max-sequences is 4"), "{refused}");
    }

    #[test]
    fn settings_a_flash_engine_cannot_serve_are_refused_naming_the_setting_and_the_streams_are_returned() {
        let mut settings = Settings::with_model(ModelSettings::default());
        settings.memory.weight_streaming.gpu_streams = 3;
        assert_eq!(check_settings(&settings).expect("served"), 3);
        settings.memory.weight_streaming.gpu_streams = 0;
        let streams = check_settings(&settings).expect_err("no streams").to_string();
        assert!(streams.contains("gpu-streams is 0"), "{streams}");
        settings.memory.weight_streaming.gpu_streams = 3;
        settings.batching.max_sequences = 2;
        let sequences = check_settings(&settings).expect_err("two sequences").to_string();
        assert!(sequences.contains("max-sequences is 2"), "{sequences}");
    }

    #[test]
    fn a_stated_host_limit_is_the_budget_whatever_the_machine_spares() {
        let mut settings = Settings::with_model(ModelSettings::default());
        settings.memory.host_limit = Auto::Value(ByteSize::gib(24));
        let budget = host_budget(&settings.memory, Some(8 << 30));
        assert_eq!(budget.source, BudgetSource::Explicit);
        assert_eq!(budget.resident_bytes, 24 << 30);
        settings.memory.host_limit = Auto::Auto;
        assert_eq!(host_budget(&settings.memory, None).source, BudgetSource::Unknown);
    }
}
