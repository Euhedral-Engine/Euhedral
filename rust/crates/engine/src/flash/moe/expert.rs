//! The expert stage of one expert of a block: the expert's kernels, once the expert is held, and the closing of its
//! lease behind a marker of the lane that read it.
//!
//! The `index`-th expert stage runs the `index`-th expert of the block's plan, behind the fetch stage of the same
//! index. It writes the lease's device address into the slot of the block's description, queues the copy of the slot and
//! the two expert kernels after the copy that fills the expert's slot in the cache, and fences the lease behind its own
//! marker, so the cache does not refill the slot before the kernels have read it. A stage past the block's experts, or
//! of a stopped quantum, completes without work and gives back a lease it was handed.
//!
//! A lease is dropped only once nothing queued reads its slot: after the fence, or, when the stage failed after queueing
//! part of its work, at the quantum's retirement ([`Stage::retired`], [`Stage::abandoned`]), by which the device has
//! stopped.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use euhedral_graph::{Marker, Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::KernelOps;

use super::block::BlockPlan;
use super::layer::MoeLayer;
use super::source::HeldExpert;
use super::stages::block_of;
use crate::flash::sequence::{Op, queue_all};
use crate::stages::{Launch, Quantum, fatal, run_stage};

/// Runs `submit` over the expert `lease` of slot `index` and settles the lease by how it ended: fenced behind the
/// marker of the submission and dropped when it ended with one; kept in its slot when `submit` asked for a retry
/// (nothing was queued, and the same stage runs again); kept in its slot, to be dropped at retirement, when it failed.
pub(super) fn settle(
    plan: &BlockPlan,
    index: u32,
    mut lease: Box<dyn HeldExpert>,
    submit: impl FnOnce(&dyn HeldExpert) -> Result<Submitted, StageError>,
) -> Result<Submitted, StageError> {
    match submit(&*lease) {
        Ok(submitted) => {
            if let Some(marker) = submitted.marker {
                lease.fence_after(marker);
            }
            Ok(submitted)
        }
        Err(error) => {
            plan.hold(index, lease);
            Err(error)
        }
    }
}

/// The queued work of one expert: the copy of its slot address and its kernels.
struct ExpertRun<'a> {
    layer: &'a MoeLayer,
    index: u32,
    ready: Option<Marker>,
    first_item: u32,
    item_count: u32,
    pairs: u32,
}

impl Launch for ExpertRun<'_> {
    fn waits(&self, _quantum: &Quantum) -> Vec<Marker> {
        self.ready.into_iter().collect()
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let layout = self.layer.shape().layout();
        let [gate_up, down] = self
            .layer
            .expert_launches(self.first_item, self.item_count, self.pairs)
            .map_err(fatal)?;
        let ops = [
            Op::Upload {
                destination: self.layer.slot_address(self.index),
                source: block_of(quantum)?.staging().descriptor(),
                offset: layout.slot_offset(self.index),
                bytes: 8,
            },
            Op::Launch(gate_up),
            Op::Launch(down),
        ];
        queue_all(&ops, kernels).map_err(fatal)
    }
}

/// The expert stage of the `index`-th expert of a layer's block.
#[derive(Debug)]
pub struct MoeExpert {
    layer: Arc<MoeLayer>,
    index: u32,
}

impl MoeExpert {
    pub fn new(layer: Arc<MoeLayer>, index: u32) -> Self {
        Self { layer, index }
    }
}

impl Stage<Quantum> for MoeExpert {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let block = block_of(quantum)?;
        let plan = block.plan();
        let Some((first_item, item_count)) = plan.items(self.index).filter(|_| !quantum.stop_requested()) else {
            drop(plan.take(self.index));
            return Ok(Submitted::done());
        };
        let lease = plan.take(self.index).ok_or_else(|| {
            fatal(format!(
                "layer {}: expert slot {} is named by the block but not held",
                self.layer.index(),
                self.index
            ))
        })?;
        // The slot's words are the kernels' only view of the expert's address, so they are written before the copy that
        // carries them; writing them again on a retry changes nothing.
        let descriptor = block.staging().descriptor();
        for (word, value) in self
            .layer
            .shape()
            .layout()
            .slot_words(self.index, lease.device_address())
        {
            descriptor.word(word).store(value, Ordering::Relaxed);
        }
        settle(plan, self.index, lease, |lease| {
            let run = ExpertRun {
                layer: &self.layer,
                index: self.index,
                ready: lease.ready(),
                first_item,
                item_count,
                pairs: plan.pairs(),
            };
            run_stage(&run, quantum, cx)
        })
    }

    fn abandoned(&self, quantum: &Quantum, _home: &dyn euhedral_graph::DeviceQueue) {
        self.release(quantum);
    }

    fn retired(&self, quantum: &Quantum, _committed: bool) {
        self.release(quantum);
    }
}

impl MoeExpert {
    /// Gives back the lease this stage still holds, at a time when no queued work of the quantum runs.
    fn release(&self, quantum: &Quantum) {
        if let Some(block) = quantum.moe_block() {
            drop(block.plan().take(self.index));
        }
    }
}

#[cfg(test)]
mod tests;
