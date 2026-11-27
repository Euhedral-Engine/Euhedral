//! A started Flash-Next engine as the server holds it: the model on the device, the one sequence over it, and the
//! owner that stops them in the right order.
//!
//! Role: [`FlashRuntime`] is what the dense [`EngineRuntime`](crate::runtime::EngineRuntime) is to the server, for the
//! Flash-Next family: it starts requests, answers what the server reports about the engine, and shuts down. Call
//! sequence: [`start`] (the model's start), [`FlashRuntime::over`] once the model is ready, [`FlashRuntime::generate`]
//! for each request, [`FlashRuntime::shutdown`] last, on the thread that started the engine.

use std::fmt::{self, Debug, Formatter};
use std::mem;
use std::sync::Arc;
use std::sync::atomic::Ordering;
use std::time::{Duration, Instant};

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

/// The model and its sequence. The owner of the lattice is not here: it stays with whoever shuts the engine down, so
/// that no last handle ever closes the lattice from a worker.
pub struct FlashRuntime {
    model: FlashModel,
    engine: FlashEngine,
}

impl Debug for FlashRuntime {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("FlashRuntime").finish_non_exhaustive()
    }
}

impl FlashRuntime {
    /// The runtime over a ready `model`, and the `owner` back for its shutdown: allocates the sequence and logs the
    /// startup summary. When the sequence cannot be allocated the model is shut down here (its memory freed) and the
    /// refusal returned.
    pub fn over(model: FlashModel, owner: RuntimeOwner) -> Result<(FlashRuntime, RuntimeOwner), EngineError> {
        match FlashEngine::new(&model) {
            Ok(engine) => {
                for line in summary(&model) {
                    log::info!("{line}");
                }
                Ok((FlashRuntime { model, engine }, owner))
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

    /// Stops the engine within `window`, as the dense engine's `shutdown_within` does: cancels the running request,
    /// closes the sequence, closes the lattice through `owner` and frees the model's memory. When the window is spent
    /// before the sequence closes or before the lattice has closed, a request did not end, or another handle to the runtime is alive, nothing is
    /// freed (the device memory stays allocated until the process exits, and the bytes are logged) and the refusal is
    /// returned. A free that has begun is not interrupted. Call it on the thread that started the engine.
    pub fn shutdown(runtime: Arc<FlashRuntime>, owner: RuntimeOwner, window: Duration) -> Result<(), EngineError> {
        let begun = Instant::now();
        runtime
            .engine
            .stop(window.min(runtime.model.core().teardown.cancel_bound));
        let bytes = runtime.allocated_device_bytes();
        let Ok((runtime, owner)) = sole(runtime, owner, drop) else {
            log::error!("{bytes} bytes of device memory stay allocated until the process exits: a handle is alive");
            return Err(EngineError::Runtime(
                "another handle to the runtime is alive: its device memory stays allocated until the process exits"
                    .to_string(),
            ));
        };
        let FlashRuntime { model, engine } = runtime;
        // The engine's close frees the sequence and may wait: it does not begin once the window is spent.
        close_within(begun, window, engine, FlashEngine::close, mem::forget, |ended, left| {
            model.shutdown_within(owner, ended, left)
        })
    }
}

/// Closes the `engine` unless the `window` that began at `begun` is spent, then hands the model's shutdown whether the
/// engine ended and what is left of the window. A spent window forgets the engine (its memory stays allocated) and
/// tells the model it did not end, so the model is kept too.
fn close_within<E>(
    begun: Instant,
    window: Duration,
    engine: E,
    close: impl FnOnce(E) -> bool,
    forget: impl FnOnce(E),
    model: impl FnOnce(bool, Duration) -> Result<(), EngineError>,
) -> Result<(), EngineError> {
    if begun.elapsed() >= window {
        forget(engine);
        return model(false, Duration::ZERO);
    }
    let ended = close(engine);
    model(ended, window.saturating_sub(begun.elapsed()))
}

/// The runtime and the owner back when no other handle to the runtime is alive. Otherwise the owner is closed through
/// `close` (on this thread: it closes the lattice) and the handle is forgotten, not dropped: the last holder that
/// is a frame on the lattice must never free the weights or close the lattice, so the runtime is left allocated.
fn sole<R, O>(runtime: Arc<R>, owner: O, close: impl FnOnce(O)) -> Result<(R, O), ()> {
    match Arc::try_unwrap(runtime) {
        Ok(runtime) => Ok((runtime, owner)),
        Err(shared) => {
            close(owner);
            mem::forget(shared);
            Err(())
        }
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

#[cfg(test)]
mod tests {
    use std::cell::RefCell;
    use std::sync::atomic::AtomicU32;

    use super::*;

    /// A runtime stand-in that counts its drops.
    struct Counted(Arc<AtomicU32>);

    impl Drop for Counted {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::AcqRel);
        }
    }

    #[test]
    fn a_spent_window_forgets_the_engine_without_closing_it_and_keeps_the_model() {
        let order = RefCell::new(Vec::new());
        let told = RefCell::new(None);
        let result = close_within(
            Instant::now(),
            Duration::ZERO,
            (),
            |()| {
                order.borrow_mut().push("close");
                true
            },
            |()| order.borrow_mut().push("forget"),
            |ended, left| {
                told.replace(Some((ended, left)));
                Ok(())
            },
        );
        assert!(result.is_ok());
        assert_eq!(
            *order.borrow(),
            ["forget"],
            "the engine was closed after the window was spent"
        );
        assert_eq!(told.take(), Some((false, Duration::ZERO)));
    }

    #[test]
    fn an_open_window_closes_the_engine_and_passes_on_what_is_left_of_it() {
        let told = RefCell::new(None);
        let window = Duration::from_secs(3600);
        let result = close_within(
            Instant::now(),
            window,
            (),
            |()| true,
            |()| panic!("the engine was forgotten inside an open window"),
            |ended, left| {
                told.replace(Some((ended, left)));
                Ok(())
            },
        );
        assert!(result.is_ok());
        let (ended, left) = told.take().expect("the model was told");
        assert!(ended);
        assert!(left <= window && left > Duration::ZERO, "{left:?}");
    }

    #[test]
    fn an_engine_that_did_not_end_tells_the_model_so() {
        let told = RefCell::new(None);
        let _ = close_within(
            Instant::now(),
            Duration::MAX,
            (),
            |()| false,
            |()| {},
            |ended, _| {
                told.replace(Some(ended));
                Ok(())
            },
        );
        assert_eq!(told.take(), Some(false));
    }

    #[test]
    fn a_runtime_nobody_else_holds_is_handed_back_with_its_owner_and_nothing_is_closed() {
        let drops = Arc::new(AtomicU32::new(0));
        let mut closed = 0;
        let runtime = Arc::new(Counted(Arc::clone(&drops)));
        let (back, owner) = sole(runtime, 7, |_| closed += 1).expect("the only handle");
        assert_eq!((owner, closed, drops.load(Ordering::Acquire)), (7, 0, 0));
        drop(back);
        assert_eq!(drops.load(Ordering::Acquire), 1);
    }

    #[test]
    fn a_runtime_another_handle_holds_closes_the_owner_once_and_is_never_dropped() {
        let drops = Arc::new(AtomicU32::new(0));
        let mut closed = 0;
        let runtime = Arc::new(Counted(Arc::clone(&drops)));
        let other = Arc::clone(&runtime);
        assert!(sole(runtime, 7, |owner| closed += owner).is_err());
        assert_eq!(closed, 7, "the owner closes the lattice");
        // The other handle was the last holder: its drop must not free the runtime.
        drop(other);
        assert_eq!(
            drops.load(Ordering::Acquire),
            0,
            "the runtime was dropped by its last holder"
        );
    }
}
