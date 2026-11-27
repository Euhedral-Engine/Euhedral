//! A started Flash-Next engine as the server holds it: the model on the device, the one sequence over it, and the
//! owner that stops them in the right order.
//!
//! Role: [`FlashRuntime`] is what the dense [`EngineRuntime`](crate::runtime::EngineRuntime) is to the server, for the
//! Flash-Next family: it starts requests, answers what the server reports about the engine, and shuts down. Call
//! sequence: [`start`] (the model's start), [`FlashRuntime::over`] once the model is ready, [`FlashRuntime::generate`]
//! for each request, [`FlashRuntime::shutdown`] last, on the thread that started the engine.

use std::fmt::{self, Debug, Formatter};
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_inference_cuda::{CudaLibrary, PinnedPool};
use euhedral_inference_edrl::ModelFamily;

use super::engine::FlashEngine;
use super::start::family_name;
use super::startup::FlashModel;
use crate::error::EngineError;
use crate::generation::{AdmissionError, CancelHandle, GenerationRequest};
use crate::runtime::RuntimeOwner;

const MIB: u64 = 1 << 20;

/// The model, its sequence and the owner of its lattice.
pub struct FlashRuntime {
    model: FlashModel,
    engine: FlashEngine,
    owner: RuntimeOwner,
}

impl Debug for FlashRuntime {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("FlashRuntime").finish_non_exhaustive()
    }
}

impl FlashRuntime {
    /// The runtime over a ready `model`: allocates the sequence and logs the startup summary. When the sequence cannot
    /// be allocated the model is shut down here (its memory freed) and the refusal returned.
    pub fn over(model: FlashModel, owner: RuntimeOwner) -> Result<FlashRuntime, EngineError> {
        match FlashEngine::new(&model) {
            Ok(engine) => {
                for line in summary(&model) {
                    log::info!("{line}");
                }
                Ok(FlashRuntime { model, engine, owner })
            }
            Err(error) => {
                model.shutdown(owner, true);
                Err(EngineError::Runtime(format!("the sequence's device memory: {error}")))
            }
        }
    }

    /// Starts `request` on the one sequence; see [`FlashEngine::generate`].
    pub fn generate(&self, request: GenerationRequest, cancel: CancelHandle) -> Result<(), AdmissionError> {
        self.engine.generate(request, cancel)
    }

    /// The longest prompt plus completion one request may use.
    pub fn context_tokens(&self) -> usize {
        let positions = usize::try_from(self.model.dimensions.max_position_embeddings).unwrap_or(usize::MAX);
        self.model.plan.max_context_tokens.min(positions)
    }

    /// The model's vocabulary size.
    pub fn vocabulary_size(&self) -> usize {
        usize::try_from(self.model.dimensions.vocab_size).unwrap_or(usize::MAX)
    }

    /// Device bytes the engine allocated and has not freed.
    pub fn allocated_device_bytes(&self) -> u64 {
        self.model.core().teardown.device_bytes.load(Ordering::Acquire)
    }

    /// Weight bytes held in pinned host memory.
    pub fn host_backed_weight_bytes(&self) -> u64 {
        self.model.plan.host.pinned()
    }

    /// The engine's lattice: the server's frames run here.
    pub fn lattice(&self) -> &Arc<ControlPlaneLattice> {
        &self.model.core().lattice
    }

    /// The CUDA library the engine runs on.
    pub fn library(&self) -> &Arc<CudaLibrary> {
        &self.model.core().library
    }

    /// The pinned staging pool.
    pub fn pinned_pool(&self) -> &Arc<PinnedPool> {
        &self.model.core().pool
    }

    /// Stops the engine: cancels the running request and waits at most `window` for it to end (and no longer than the
    /// lattice's close timeout), closes the sequence, then closes the lattice and frees the model's memory. A request
    /// that did not end leaves the sequence and the model allocated until the process exits (the bytes are logged) and
    /// is the refusal returned. Call it on the thread that started the engine, with no other handle alive.
    pub fn shutdown(self: Arc<Self>, window: Duration) -> Result<(), EngineError> {
        let bound = window.min(self.model.core().teardown.cancel_bound);
        self.engine.stop(bound);
        let Ok(runtime) = Arc::try_unwrap(self) else {
            return Err(EngineError::Runtime(
                "another handle to the runtime is alive: its device memory stays allocated until the process exits"
                    .to_string(),
            ));
        };
        let FlashRuntime { model, engine, owner } = runtime;
        let ended = engine.close();
        model.shutdown(owner, ended);
        if ended {
            return Ok(());
        }
        Err(EngineError::Runtime(
            "a generation did not end after its cancel: the device memory stays allocated until the process exits"
                .to_string(),
        ))
    }
}

/// The startup summary: the family, the context, and where the plan puts the model.
fn summary(model: &FlashModel) -> Vec<String> {
    let device = &model.plan.device;
    let host = &model.plan.host;
    vec![
        format!("Model family: {}", family_name(ModelFamily::FlashNextV3)),
        format!(
            "Context {} tokens x 1 sequence: {} MiB free on the GPU at the plan, {} MiB of experts cached on the \
             device in {} slots, {} MiB of pinned host memory",
            model.plan.max_context_tokens,
            device.free / MIB,
            device.expert_cache / MIB,
            model.plan.expert_cache.slot_count,
            host.pinned() / MIB,
        ),
    ]
}
