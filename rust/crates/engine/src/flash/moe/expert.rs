//! The expert stage of one expert of a block: the expert's kernels, once the expert is held, and the closing of its
//! lease behind a marker of the lane that read it.
//!
//! The `index`-th expert stage runs the `index`-th expert of the block's plan, behind the fetch stage of the same
//! index. It writes the lease's device address into the slot of the block's description, queues the copy of the slot and
//! the two expert kernels after the copy that fills the expert's slot in the cache, and fences the lease behind its own
//! marker, so the cache does not refill the slot before the kernels have read it. A stage past the block's experts, or
//! of a stopped quantum, completes without work and gives back a lease it was handed.
//!
//! A lease is dropped only once nothing queued reads its slot: after the fence, or, when the stage failed after it may
//! have queued part of its work, at the quantum's retirement ([`Stage::retired`]), by which the device has stopped.
//! The quantum's abandonment ([`Stage::abandoned`]) comes before that and drops only a lease no queued work reads: one
//! that was never run, or whose submission asked to retry.

use std::sync::Arc;

use euhedral_graph::{DeviceQueue, Stage, StageCx, StageError, Submitted};

use super::block::BlockPlan;
use super::host::{MoeHost, MoeWork};
use super::layer::MoeLayer;
use super::source::HeldExpert;
use crate::stages::fatal;

/// Runs `submit` over the expert `lease` of slot `index` and settles the lease by how it ended: fenced behind the
/// marker of the submission and dropped when it ended with one; kept in its slot when `submit` asked for a retry
/// (nothing was queued, and the same stage runs again); kept apart, to be dropped at retirement, when it failed.
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
        Err(StageError::Retry) => {
            plan.hold(index, lease);
            Err(StageError::Retry)
        }
        Err(error) => {
            plan.hold_queued(index, lease);
            Err(error)
        }
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

impl<Q: MoeHost> Stage<Q> for MoeExpert {
    fn run(&self, quantum: &Q, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let plan = quantum.block_plan()?;
        let Some((first_item, item_count)) = plan.items(self.index).filter(|_| !quantum.stopped()) else {
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
        quantum.check_staging(self.layer.shape().layout())?;
        // The slot's words are the kernels' only view of the expert's address, so they are written before the copy that
        // carries them; writing them again on a retry changes nothing.
        for (word, value) in self
            .layer
            .shape()
            .layout()
            .slot_words(self.index, lease.device_address())
        {
            quantum.put_word(word, value);
        }
        settle(plan, self.index, lease, |lease| {
            let work = MoeWork::Expert {
                index: self.index,
                first_item,
                item_count,
                pairs: plan.pairs(),
            };
            quantum.queue(cx, &self.layer, &lease.ready().into_iter().collect::<Vec<_>>(), work)
        })
    }

    fn abandoned(&self, quantum: &Q, _home: &dyn DeviceQueue) {
        if let Ok(plan) = quantum.block_plan() {
            plan.release_idle(self.index);
        }
    }

    fn retired(&self, quantum: &Q, _committed: bool) {
        if let Ok(plan) = quantum.block_plan() {
            plan.release_queued(self.index);
        }
    }
}

#[cfg(test)]
mod tests;
