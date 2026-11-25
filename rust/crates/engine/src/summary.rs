//! The startup summary: what an operator reads in the log once the engine is ready. The lines are in MiB, as the
//! operations guide is.

use euhedral_inference_settings::Settings;

use crate::residency::planner::{ResidencyPlan, Tier};

const MIB: u64 = 1 << 20;

/// The numbers of the summary, in bytes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Facts {
    /// Tokens of the longest context of one sequence.
    pub context_tokens: u64,
    /// The sequences the engine holds.
    pub sequences: u64,
    /// The device bytes the plan reserves.
    pub device_bytes: u64,
    /// The device bytes the plan was made against.
    pub available_bytes: u64,
    /// The pinned host bytes that hold weights.
    pub host_weight_bytes: u64,
    /// The bytes of the weights the plan leaves in the artifact file, which are read through the disk window.
    pub disk_weight_bytes: u64,
    /// The prefix cache's bytes and the tokens between its checkpoints; `None` when it is off.
    pub prefix_cache: Option<(u64, u64)>,
}

impl Facts {
    /// The facts of a plan; `prefix_cache_bytes` is what the opened cache holds (`None` when it is off).
    pub fn of(plan: &ResidencyPlan, settings: &Settings, prefix_cache_bytes: Option<u64>) -> Facts {
        Facts {
            context_tokens: plan.max_context_tokens as u64,
            sequences: plan.sequences as u64,
            device_bytes: plan.device_bytes,
            available_bytes: plan.budget_bytes,
            host_weight_bytes: plan.offload.host_arena_bytes,
            disk_weight_bytes: plan
                .device
                .iter()
                .filter(|placement| placement.tier == Tier::Disk)
                .map(|placement| placement.bytes)
                .sum(),
            prefix_cache: prefix_cache_bytes.map(|bytes| (bytes, u64::from(settings.prefix_cache.checkpoint_tokens))),
        }
    }
}

/// The lines of the summary: the context's fit, then the prefix cache. Sizes are whole MiB, rounded down.
pub fn lines(facts: &Facts) -> Vec<String> {
    let sequences = if facts.sequences == 1 { "sequence" } else { "sequences" };
    let mut fit = format!(
        "Context {} tokens x {} {sequences}: {} MiB of the {} MiB available on the GPU, {} MiB of weights in host memory",
        facts.context_tokens,
        facts.sequences,
        facts.device_bytes / MIB,
        facts.available_bytes / MIB,
        facts.host_weight_bytes / MIB,
    );
    if facts.disk_weight_bytes > 0 {
        fit.push_str(&format!(
            ", {} MiB of weights read from the artifact file",
            facts.disk_weight_bytes / MIB
        ));
    }
    let cache = match facts.prefix_cache {
        Some((bytes, stride)) => format!(
            "Prefix cache: {} MiB pinned, a checkpoint every {stride} tokens",
            bytes / MIB
        ),
        None => "Prefix cache: off".to_owned(),
    };
    vec![fit, cache]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn facts() -> Facts {
        Facts {
            context_tokens: 32768,
            sequences: 2,
            device_bytes: 13854 * MIB,
            available_bytes: 15100 * MIB,
            host_weight_bytes: 0,
            disk_weight_bytes: 0,
            prefix_cache: Some((4096 * MIB, 2048)),
        }
    }

    #[test]
    fn sizes_round_down_and_a_single_sequence_is_singular() {
        let odd = Facts {
            sequences: 1,
            device_bytes: 13854 * MIB + MIB - 1,
            available_bytes: 15100 * MIB + MIB - 1,
            ..facts()
        };
        let line = &lines(&odd)[0];
        assert_eq!(
            line,
            "Context 32768 tokens x 1 sequence: 13854 MiB of the 15100 MiB available on the GPU, 0 MiB of weights in host memory"
        );
        assert!(!lines(&facts())[0].contains("weights read"));
    }

    #[test]
    fn the_fit_with_host_weights_and_disk_reads() {
        let held = Facts {
            host_weight_bytes: 3485 * MIB,
            disk_weight_bytes: 96 * MIB,
            ..facts()
        };
        let line = &lines(&held)[0];
        assert!(line.contains(", 3485 MiB of weights in host memory, 96 MiB of weights read from the artifact file"));
    }

    #[test]
    fn the_cache_on_and_off() {
        assert_eq!(
            lines(&facts())[1],
            "Prefix cache: 4096 MiB pinned, a checkpoint every 2048 tokens"
        );
        let off = Facts {
            prefix_cache: None,
            ..facts()
        };
        assert_eq!(lines(&off)[1], "Prefix cache: off");
    }
}
