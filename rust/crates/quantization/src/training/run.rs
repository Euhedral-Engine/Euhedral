//! The resolved training kernels and the enqueue of planned steps.

use std::collections::HashMap;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use euhedral_gpu::{Gpu, LaneEnqueue, Module as GpuModule};

use super::{Step, TrainError};
use crate::kernels::{Compiled, Module, enqueue_steps};

/// The kernels of the compiled training module.
pub struct TrainKernels<M: GpuModule> {
    kernels: HashMap<&'static str, <M::Gpu as Gpu>::Kernel>,
}

impl<M: GpuModule> Debug for TrainKernels<M> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("TrainKernels")
            .field("kernels", &self.kernels.len())
            .finish()
    }
}

impl<M: GpuModule> TrainKernels<M> {
    /// The kernels of `compiled`, which must be the training module.
    pub fn new(compiled: &Compiled<M>) -> Self {
        Self {
            kernels: compiled
                .kernels()
                .map(|(name, kernel)| (name, kernel.clone()))
                .collect(),
        }
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
    pub unsafe fn enqueue<E>(&self, lane: &mut E, steps: &[Step]) -> Result<(), TrainError>
    where
        E: LaneEnqueue<Kernel = <M::Gpu as Gpu>::Kernel>,
    {
        // SAFETY: the caller vouches for the addresses.
        unsafe { enqueue_steps(Module::Train, &self.kernels, lane, steps) }?;
        Ok(())
    }
}
