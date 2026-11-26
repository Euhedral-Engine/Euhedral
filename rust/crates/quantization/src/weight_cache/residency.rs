//! What an engine needs on the device beyond its objects, from the engine's residency planner.

use euhedral_inference_edrl::TensorTable;
use euhedral_inference_engine::error::PlanError;
use euhedral_inference_engine::model::ModelConfig;
use euhedral_inference_engine::residency::planner::{PlanInputs, plan_closest};
use euhedral_inference_settings::Settings;

/// The device figures of one engine: whether it keeps every object on the device, and what it needs besides them.
///
/// The planner owns the per-sequence figures (the cache of every sequence and the state of those beyond the first), so
/// nothing here restates them: `settings` carry the context (`context.max-tokens` and `context.kv-cache-on-gpu-tokens`
/// are the tokens of one sequence) and `sequences` how many run at once.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Residency {
    /// Every object stays on the device and nothing is read from the file at run time.
    pub fits_resident: bool,
    /// The device bytes beyond the objects when all of them are resident: the cache of `sequences` sequences, their
    /// state, the workspace and the reserve. A [`Need`](super::Need) takes the objects from the cache's keys.
    pub other_bytes: u64,
}

impl Residency {
    /// The figures of `sequences` sequences of `table`'s model on a device with `device_free` bytes free.
    pub fn of(
        config: &ModelConfig,
        table: &TensorTable,
        settings: &Settings,
        device_free: u64,
        sequences: usize,
    ) -> Result<Residency, PlanError> {
        let plan_at = |free: u64| {
            plan_closest(
                config,
                table,
                settings,
                PlanInputs {
                    device_free: free,
                    multiprocessors: None,
                    exact_numerics: false,
                    host_pinned_bytes: Some(u64::MAX),
                    sequences,
                },
            )
        };
        let (here, refusal) = plan_at(device_free)?;
        let fits_resident = refusal.is_none() && here.offload.is_empty();
        // The figure of an engine that has the room: a plan with the device's own bytes may have offloaded objects,
        // whose bytes are then not among `weights_bytes`.
        let (roomy, _) = plan_at(1 << 50)?;
        Ok(Residency {
            fits_resident,
            other_bytes: roomy.device_bytes.saturating_sub(roomy.weights_bytes),
        })
    }
}
