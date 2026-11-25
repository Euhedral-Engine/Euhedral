//! The resolved trellis kernels and the enqueue of planned steps.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use euhedral_gpu::{Gpu, KernelOptions, LaneEnqueue, Module as GpuModule};

use super::plan::rate_kernel;
use super::rate::Rate;
use super::{Step, TrellisError};
use crate::kernels::Compiled;

/// The rates the kernels support.
const RATES: [f64; 7] = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0];

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
    /// The kernels of `compiled`, which must be the trellis module. Every Viterbi kernel is looked up again with the
    /// dynamic shared memory its block needs, once, here: the large ones need the opt-in, the others are unharmed.
    pub fn new(compiled: &Compiled<M>) -> Result<Self, TrellisError> {
        let mut kernels: HashMap<_, _> = compiled
            .kernels()
            .map(|(name, kernel)| (name, kernel.clone()))
            .collect();
        for bits in RATES {
            let rate = Rate::new(bits)?;
            let name = rate_kernel("viterbi", rate)?;
            let opted = compiled.module().kernel_with(
                name,
                KernelOptions {
                    max_dynamic_shared_bytes: rate.viterbi_shared_bytes(),
                },
            )?;
            kernels.insert(name, opted);
        }
        Ok(Self { kernels })
    }

    /// Enqueues `steps` in order on `lane`. Every kernel is looked up before the first launch, so a step the table
    /// lacks leaves nothing on the lane.
    ///
    /// On an `Err` from the lane itself (the driver refusing a launch) the steps before it are already on the lane:
    /// their addresses must stay valid until the lane retires them.
    ///
    /// # Safety
    /// Every address in the steps is valid for the access its kernel makes until the work retires, and no other work
    /// uses those ranges meanwhile.
    pub unsafe fn enqueue<E>(&self, lane: &mut E, steps: &[Step]) -> Result<(), TrellisError>
    where
        E: LaneEnqueue<Kernel = <M::Gpu as Gpu>::Kernel>,
    {
        let resolved = steps
            .iter()
            .map(|step| {
                self.kernels
                    .get(step.kernel)
                    .ok_or_else(|| TrellisError::NoKernel(step.kernel.to_owned()))
            })
            .collect::<Result<Vec<_>, _>>()?;
        for (step, kernel) in steps.iter().zip(resolved) {
            // SAFETY: the caller vouches for the addresses; only the planners build a step, with the arguments in
            // the kernel's parameter order.
            unsafe { lane.launch(kernel, step.launch, &step.args) }?;
        }
        Ok(())
    }
}
