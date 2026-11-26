//! The resolved kernels of the drafter and the enqueue of its work.

use std::collections::HashMap;
use std::ffi::c_void;
use std::fmt::{Debug, Formatter, Result as FmtResult};

use euhedral_gpu::{Gpu, LaneStream, Module as GpuModule};
use euhedral_inference_cuda::math::LaneHandles;

use super::{DrafterError, Work};
use crate::kernels::{Compiled, Module, Step, enqueue_steps};
use crate::training::{TrainError, TrainKernels};

/// The kernels of the compiled training module and of the compiled drafter module.
pub struct DrafterKernels<M: GpuModule> {
    train: TrainKernels<M>,
    drafter: HashMap<&'static str, <M::Gpu as Gpu>::Kernel>,
}

impl<M: GpuModule> Debug for DrafterKernels<M> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("DrafterKernels")
            .field("drafter", &self.drafter.len())
            .finish_non_exhaustive()
    }
}

impl<M: GpuModule> DrafterKernels<M> {
    /// The kernels of `train`, which must be the training module, and of `drafter`, which must be the drafter module.
    pub fn new(train: &Compiled<M>, drafter: &Compiled<M>) -> Self {
        Self {
            train: TrainKernels::new(train),
            drafter: drafter.kernels().map(|(name, kernel)| (name, kernel.clone())).collect(),
        }
    }

    /// Enqueues `work` in order on `lane`, whose cuBLAS `handles` run the products on the stream the lane lends. A
    /// piece whose kernel is not resolved leaves nothing of that piece on the lane.
    ///
    /// On an `Err` from the lane itself the work before it is already on the lane: its addresses must stay valid
    /// until the lane retires it.
    ///
    /// # Safety
    /// Every address in the work is valid for the access its kernel or product makes until the lane retires it, no
    /// other work uses those ranges meanwhile, and `handles` belong to the lane.
    pub unsafe fn enqueue<E>(&self, lane: &mut E, handles: &mut LaneHandles, work: &[Work]) -> Result<(), DrafterError>
    where
        E: LaneStream<Kernel = <M::Gpu as Gpu>::Kernel, Raw = *mut c_void>,
    {
        for piece in work {
            match piece {
                // SAFETY: the caller vouches for the addresses.
                Work::Steps(steps) => unsafe { self.steps(lane, steps) }?,
                Work::Product(gemm) => {
                    let stream = lane
                        .raw_stream()
                        .ok_or_else(|| TrainError::Gpu("the lane lends no stream".to_owned()))?;
                    // SAFETY: the caller vouches for the addresses and the handles; the stream is the lane's own.
                    unsafe { gemm.enqueue(handles, &stream) }?;
                }
            }
        }
        Ok(())
    }

    /// Steps of one module: the training kernels or the drafter's.
    /// # Safety
    /// As [`enqueue`](Self::enqueue).
    unsafe fn steps<E>(&self, lane: &mut E, steps: &[Step]) -> Result<(), TrainError>
    where
        E: LaneStream<Kernel = <M::Gpu as Gpu>::Kernel, Raw = *mut c_void>,
    {
        let Some(first) = steps.first() else {
            return Ok(());
        };
        match first.module() {
            // SAFETY: the caller's contract.
            Module::Train => unsafe { self.train.enqueue(lane, steps) },
            // SAFETY: the caller's contract. A step of any other module is refused as one of the wrong module.
            _ => Ok(unsafe { enqueue_steps(Module::Drafter, &self.drafter, lane, steps) }?),
        }
    }
}
