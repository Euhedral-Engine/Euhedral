//! What the MoE stages that keep state across a block read and write of the quantum they run for, and the device work
//! they queue through it.
//!
//! The stages are written against [`MoeHost`], so their contracts (which lease goes where, what a refusal leaves
//! behind, what an idle stage does) are tested with a fake host that needs no device. [`Quantum`] is the host of the
//! real engine: it keeps the block's hand-over ([`MoeBlock`]) and queues the work as launches of a stage.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use euhedral_graph::{Marker, StageCx, StageError, Submitted};
use euhedral_inference_cuda::KernelOps;

use super::block::{BlockPlan, MoeBlock};
use super::layer::MoeLayer;
use super::routing::{DescriptorLayout, Routing};
use crate::flash::sequence::{Op, queue_all};
use crate::stages::{Launch, Quantum, fatal, run_stage};

/// The device work of a MoE stage that depends on the block's plan.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MoeWork {
    /// The copy of the `index`-th slot address to the device, then the gate-up and the down kernels over `item_count`
    /// work items from `first_item`, for a block of `pairs` pairs.
    Expert {
        index: u32,
        first_item: u32,
        item_count: u32,
        pairs: u32,
    },
    /// The copy of the whole description to the device.
    Description,
}

/// What the plan, fetch and expert stages read of the quantum they run for.
pub trait MoeHost: Send + Sync {
    /// The experts the quantum's current block names and the leases of those that arrived.
    fn block_plan(&self) -> Result<&Arc<BlockPlan>, StageError>;

    /// Whether the quantum was stopped, so that its experts are not needed.
    fn stopped(&self) -> bool;

    /// The flag that ends a request for an expert when the quantum is cancelled, when the quantum has one of its own.
    fn cancel_flag(&self) -> Option<Arc<AtomicBool>>;

    /// The rows of the quantum.
    fn rows(&self) -> usize;

    /// Whether the quantum's staging memory is the size `layout` needs; the stages that write it refuse otherwise.
    fn check_staging(&self, layout: &DescriptorLayout) -> Result<(), StageError>;

    /// Takes the grouping scratch for one block's plan; [`MoeHost::give_back`] returns it.
    fn take_routing(&self) -> Result<Box<Routing>, StageError>;

    fn give_back(&self, routing: Box<Routing>);

    /// The router's `i`-th expert id and its routing weight (BF16 bits), as the device copied them.
    fn router_pick(&self, i: usize) -> (i32, u16);

    /// Stores one word of the block's description, in the memory the device copies it from.
    fn put_word(&self, word: usize, value: u32);

    /// Queues `work` of `layer` behind the markers `waits`.
    fn queue(
        &self,
        cx: &StageCx<'_>,
        layer: &MoeLayer,
        waits: &[Marker],
        work: MoeWork,
    ) -> Result<Submitted, StageError>;
}

/// The block hand-over of `quantum`, or the failure of a quantum built without one.
pub(super) fn block_of(quantum: &Quantum) -> Result<&Arc<MoeBlock>, StageError> {
    quantum
        .moe_block()
        .ok_or_else(|| fatal("the quantum carries no mixture-of-experts block"))
}

/// `MoeWork` as the launch of a stage.
struct Queued<'a> {
    layer: &'a MoeLayer,
    waits: &'a [Marker],
    work: MoeWork,
}

impl Launch for Queued<'_> {
    fn waits(&self, _quantum: &Quantum) -> Vec<Marker> {
        self.waits.to_vec()
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let layout = self.layer.shape().layout();
        let descriptor = block_of(quantum)?.staging().descriptor();
        match self.work {
            MoeWork::Expert {
                index,
                first_item,
                item_count,
                pairs,
            } => {
                let [gate_up, down] = self
                    .layer
                    .expert_launches(first_item, item_count, pairs)
                    .map_err(fatal)?;
                let ops = [
                    Op::Upload {
                        destination: self.layer.slot_address(index),
                        source: descriptor,
                        offset: layout.slot_offset(index),
                        bytes: 8,
                    },
                    Op::Launch(gate_up),
                    Op::Launch(down),
                ];
                queue_all(&ops, kernels)
            }
            MoeWork::Description => Op::Upload {
                destination: self.layer.descriptor(),
                source: descriptor,
                offset: 0,
                bytes: layout.bytes(),
            }
            .queue(kernels),
        }
        .map_err(fatal)
    }
}

impl MoeHost for Quantum {
    fn block_plan(&self) -> Result<&Arc<BlockPlan>, StageError> {
        block_of(self).map(|block| block.plan())
    }

    fn stopped(&self) -> bool {
        self.stop_requested()
    }

    fn cancel_flag(&self) -> Option<Arc<AtomicBool>> {
        match self.segments() {
            [segment] => segment.cancel.clone(),
            _ => None,
        }
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn check_staging(&self, layout: &DescriptorLayout) -> Result<(), StageError> {
        let staged = block_of(self)?.staging().layout();
        if staged != layout {
            return Err(fatal(format!(
                "the quantum's staging is sized for {staged:?}, not for the layer's {layout:?}"
            )));
        }
        Ok(())
    }

    fn take_routing(&self) -> Result<Box<Routing>, StageError> {
        block_of(self)?.staging().take_routing()
    }

    fn give_back(&self, routing: Box<Routing>) {
        if let Some(block) = self.moe_block() {
            block.staging().give_back(routing);
        }
    }

    fn router_pick(&self, i: usize) -> (i32, u16) {
        self.moe_block()
            .map_or((0, 0), |block| (block.staging().id(i), block.staging().weight(i)))
    }

    fn put_word(&self, word: usize, value: u32) {
        if let Some(block) = self.moe_block() {
            block.staging().descriptor().word(word).store(value, Ordering::Relaxed);
        }
    }

    fn queue(
        &self,
        cx: &StageCx<'_>,
        layer: &MoeLayer,
        waits: &[Marker],
        work: MoeWork,
    ) -> Result<Submitted, StageError> {
        run_stage(&Queued { layer, waits, work }, self, cx)
    }
}
