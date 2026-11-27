//! The device stages of a layer's MoE block that run once per block: the router, the shared expert, the plan and the
//! finish. The per-expert stages are in [`super::fetch`] and [`super::expert`].
//!
//! Route copies the router's choice to pinned memory; the plan stage, behind a device-completion edge from it, reads
//! that choice on the host, groups the pairs by expert and copies the description to the device. Neither the plan nor
//! anything after it waits in a frame: the edge fires when the copies are done.

use std::sync::Arc;

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::KernelOps;

use super::host::{MoeHost, MoeWork, block_of};
use super::layer::MoeLayer;
use crate::flash::sequence::{Op, queue_all};
use crate::stages::{Launch, Quantum, fatal};

/// The rows of `quantum`, within what `layer` routes.
fn rows_of(quantum: &Quantum, layer: &MoeLayer) -> Result<u32, StageError> {
    check_rows(quantum.rows_u32()?, layer)
}

/// `rows`, when `layer` routes that many.
pub(super) fn check_rows(rows: u32, layer: &MoeLayer) -> Result<u32, StageError> {
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
        quantum.check_staging(self.layer.shape().layout())?;
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

impl<Q: MoeHost> Stage<Q> for MoePlan {
    /// Plans the block and publishes it, then queues the copy. The plan is made again when the copy is retried
    /// (nothing was queued), which changes nothing: no expert is fetched before this stage submits.
    fn run(&self, quantum: &Q, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let plan = quantum.block_plan()?;
        if quantum.rows() == 0 {
            plan.publish_none();
            return Ok(Submitted::done());
        }
        let rows = u32::try_from(quantum.rows())
            .map_err(|_| fatal(format!("{} rows exceed a launch", quantum.rows())))
            .and_then(|rows| check_rows(rows, &self.layer))?;
        quantum.check_staging(self.layer.shape().layout())?;
        let mut routing = quantum.take_routing()?;
        let planned = routing.plan(rows, |i| quantum.router_pick(i).0, |i| quantum.router_pick(i).1);
        if let Err(error) = planned {
            quantum.give_back(routing);
            return Err(fatal(format!("layer {}: {error}", self.layer.index())));
        }
        routing.fill(&mut |word, value| quantum.put_word(word, value));
        let published = plan.publish(&routing);
        quantum.give_back(routing);
        published.map_err(|error| fatal(format!("layer {}: {error}", self.layer.index())))?;
        quantum.queue(cx, &self.layer, &[], MoeWork::Description)
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

#[cfg(test)]
mod tests;
