//! The resolved trellis kernels and the enqueue of planned steps.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use euhedral_gpu::{Gpu, KernelOptions, LaneEnqueue, Module as GpuModule};

use super::rate::{DEFAULT_SHARED_BYTES, Rate};
use super::{Step, TrellisError};
use crate::kernels::Compiled;

/// The kernels of the compiled trellis module, with the Viterbi kernels opted in to the shared memory they use.
pub struct TrellisKernels<M: GpuModule> {
    kernels: HashMap<&'static str, <M::Gpu as Gpu>::Kernel>,
}

impl<M: GpuModule> Debug for TrellisKernels<M> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("TrellisKernels")
            .field("kernels", &self.kernels.len())
            .finish()
    }
}

impl<M: GpuModule> TrellisKernels<M> {
    /// The kernels of `compiled`, which must be the trellis module. A Viterbi kernel whose block needs more dynamic
    /// shared memory than the default is looked up again with the limit it needs, once, here.
    pub fn new(compiled: &Compiled<M>) -> Result<Self, TrellisError> {
        let mut kernels: HashMap<_, _> = compiled
            .kernels()
            .map(|(name, kernel)| (name, kernel.clone()))
            .collect();
        for lower in 2..=6u32 {
            let name = format!("viterbi_k{lower}");
            let Ok(rate) = Rate::new(f64::from(lower)) else {
                continue;
            };
            let shared = rate.viterbi_shared_bytes();
            if shared <= DEFAULT_SHARED_BYTES {
                continue;
            }
            let Some(listed) = kernels.keys().copied().find(|listed| *listed == name) else {
                return Err(TrellisError::NoKernel(name));
            };
            let opted = compiled.module().kernel_with(
                listed,
                KernelOptions {
                    max_dynamic_shared_bytes: shared,
                },
            )?;
            kernels.insert(listed, opted);
        }
        Ok(Self { kernels })
    }

    /// Enqueues `steps` in order on `lane`.
    ///
    /// # Safety
    /// Every address in the steps is valid for the access its kernel makes until the work retires, and no other work
    /// uses those ranges meanwhile.
    pub unsafe fn enqueue<E>(&self, lane: &mut E, steps: &[Step]) -> Result<(), TrellisError>
    where
        E: LaneEnqueue<Kernel = <M::Gpu as Gpu>::Kernel>,
    {
        for step in steps {
            let kernel = self
                .kernels
                .get(step.kernel)
                .ok_or_else(|| TrellisError::NoKernel(step.kernel.to_owned()))?;
            // SAFETY: the caller vouches for the addresses; the arguments were built with the kernel's parameter
            // list by the planner that made the step.
            unsafe { lane.launch(kernel, step.launch, &step.args) }?;
        }
        Ok(())
    }
}
