//! The device stages of a layer's MoE block that run once per block: the router, the shared expert, the plan and the
//! finish. The per-expert stages are in [`super::fetch`] and [`super::expert`].
//!
//! Route copies the router's choice to pinned memory; the plan stage, behind a device-completion edge from it, reads
//! that choice on the host, groups the pairs by expert and copies the description to the device. Neither the plan nor
//! anything after it waits in a frame: the edge fires when the copies are done.

use std::sync::Arc;
use std::sync::atomic::Ordering;

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::KernelOps;

use super::block::MoeBlock;
use super::layer::MoeLayer;
use crate::flash::sequence::{Op, queue_all};
use crate::stages::{Launch, Quantum, fatal, run_stage};

/// The block hand-over of `quantum`, or the failure of a quantum built without one.
pub(super) fn block_of(quantum: &Quantum) -> Result<&Arc<MoeBlock>, StageError> {
    quantum
        .moe_block()
        .ok_or_else(|| fatal("the quantum carries no mixture-of-experts block"))
}

/// The rows of `quantum`, within what `layer` routes.
fn rows_of(quantum: &Quantum, layer: &MoeLayer) -> Result<u32, StageError> {
    let rows = quantum.rows_u32()?;
    if rows > layer.max_rows() {
        return Err(fatal(format!(
            "a quantum of {rows} rows exceeds the {} rows layer {} routes",
            layer.max_rows(),
            layer.index()
        )));
    }
    Ok(rows)
}

/// The router: the experts of each row and their weights, copied to pinned memory.
#[derive(Debug)]
pub struct MoeRoute {
    layer: Arc<MoeLayer>,
}

impl MoeRoute {
    pub fn new(layer: Arc<MoeLayer>) -> Self {
        Self { layer }
    }
}

impl Launch for MoeRoute {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let rows = rows_of(quantum, &self.layer)?;
        let staging = block_of(quantum)?.staging();
        let (ids, weights) = self.layer.router_outputs();
        let top_k = self.layer.shape().layout().top_k() as usize;
        let pairs = rows as usize * top_k;
        let [logits, router] = self
            .layer
            .route_launches(rows, kernels.exact_numerics())
            .map_err(fatal)?;
        let ops = [
            Op::Launch(logits),
            Op::Launch(router),
            Op::Readback {
                destination: staging.ids(),
                offset: 0,
                source: ids,
                bytes: 4 * pairs,
            },
            Op::Readback {
                destination: staging.weights(),
                offset: 0,
                source: weights,
                bytes: 2 * pairs,
            },
        ];
        queue_all(&ops, kernels).map_err(fatal)
    }
}

/// The shared expert, beside the routed ones.
#[derive(Debug)]
pub struct MoeShared {
    layer: Arc<MoeLayer>,
}

impl MoeShared {
    pub fn new(layer: Arc<MoeLayer>) -> Self {
        Self { layer }
    }
}

impl Launch for MoeShared {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let rows = rows_of(quantum, &self.layer)?;
        let ops = self.layer.shared_ops(rows, kernels.exact_numerics()).map_err(fatal)?;
        queue_all(&ops, kernels).map_err(fatal)
    }
}

/// The plan: groups the block's (row, expert) pairs by expert on the host and copies the description to the device.
#[derive(Debug)]
pub struct MoePlan {
    layer: Arc<MoeLayer>,
}

impl MoePlan {
    pub fn new(layer: Arc<MoeLayer>) -> Self {
        Self { layer }
    }
}

/// The copy of the whole description to the device, as the launches of the plan stage.
struct DescriptorUpload<'a> {
    layer: &'a MoeLayer,
}

impl Launch for DescriptorUpload<'_> {
    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let staging = block_of(quantum)?.staging();
        Op::Upload {
            destination: self.layer.descriptor(),
            source: staging.descriptor(),
            offset: 0,
            bytes: self.layer.shape().layout().bytes(),
        }
        .queue(kernels)
        .map_err(fatal)
    }
}

impl Stage<Quantum> for MoePlan {
    /// Plans the block and publishes it, then queues the copy. The plan is made again when the copy is retried
    /// (nothing was queued), which changes nothing: no expert is fetched before this stage submits.
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let block = block_of(quantum)?;
        let (plan, staging) = (block.plan(), block.staging());
        if quantum.rows == 0 {
            plan.publish_none();
            return Ok(Submitted::done());
        }
        let rows = rows_of(quantum, &self.layer)?;
        let mut routing = staging.take_routing()?;
        let planned = routing.plan(rows, |i| staging.id(i), |i| staging.weight(i));
        if let Err(error) = planned {
            staging.give_back(routing);
            return Err(fatal(format!("layer {}: {error}", self.layer.index())));
        }
        let descriptor = staging.descriptor();
        routing.fill(&mut |word, value| descriptor.word(word).store(value, Ordering::Relaxed));
        plan.publish(&routing);
        staging.give_back(routing);
        run_stage(&DescriptorUpload { layer: &self.layer }, quantum, cx)
    }
}

/// The ordered sum of the experts' outputs, combined with the shared expert's gated output into the block's output.
#[derive(Debug)]
pub struct MoeFinish {
    layer: Arc<MoeLayer>,
}

impl MoeFinish {
    pub fn new(layer: Arc<MoeLayer>) -> Self {
        Self { layer }
    }
}

impl Launch for MoeFinish {
    fn idle(&self, quantum: &Quantum) -> bool {
        quantum.rows == 0
    }

    fn launch(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), StageError> {
        let rows = rows_of(quantum, &self.layer)?;
        let pairs = block_of(quantum)?.plan().pairs();
        let ops = self.layer.finish_ops(rows, pairs).map_err(fatal)?;
        queue_all(&ops, kernels).map_err(fatal)
    }
}

eager!(MoeRoute, MoeShared, MoeFinish);
