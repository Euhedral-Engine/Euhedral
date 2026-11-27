//! The fetch stage of one expert of a block: a host stage that holds its place in the graph until the expert is in the
//! device cache.
//!
//! One fetch stage exists for each expert the block can name (the layer's fixed width). The `index`-th takes the
//! `index`-th expert of the block's plan; a block with fewer experts, or a quantum that was stopped, completes it in
//! place and asks for nothing. Otherwise the stage takes its gate, asks the source for the expert and returns: the
//! arrival (on whichever thread the cache finishes the copy) puts the lease into the plan and releases the gate, and the
//! graph then runs the stage, which has nothing left to do. A failed arrival fails the gate, and the quantum, with the
//! layer and expert named.

use std::fmt::Display;
use std::sync::Arc;

use euhedral_graph::{Gate, GateRef, Stage, StageCx, StageError, Submitted};

use super::host::MoeHost;
use super::layer::MoeLayer;
use super::source::Arrived;
use crate::handoff::Handoff;
use crate::residency::experts::ExpertId;
use crate::stages::fatal;

/// The fetch of the `index`-th expert of a layer's block.
#[derive(Debug)]
pub struct MoeFetch {
    layer: Arc<MoeLayer>,
    index: u32,
}

impl MoeFetch {
    pub fn new(layer: Arc<MoeLayer>, index: u32) -> Self {
        Self { layer, index }
    }

    fn named(&self, expert: u32, why: impl Display) -> StageError {
        fatal(format!("layer {} expert {expert}: {why}", self.layer.index()))
    }
}

impl<Q: MoeHost> Stage<Q> for MoeFetch {
    fn run(&self, quantum: &Q, _cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        quantum.block_plan()?;
        Ok(Submitted::done())
    }

    fn pass_gate(&self, quantum: &Q, gate: GateRef<'_>) -> bool {
        let plan = match quantum.block_plan() {
            Ok(plan) => plan,
            Err(error) => {
                gate.hold().fail(error);
                return false;
            }
        };
        let Some(expert) = plan.expert(self.index).filter(|_| !quantum.stopped()) else {
            return true;
        };
        let gate = gate.hold();
        // The gate is shared with the arrival, which spends it, and with this call, which spends it when the source
        // refuses the request and so never calls the arrival.
        let held: Arc<Handoff<Gate>> = Handoff::new();
        held.send(gate);
        let arrival = {
            let (held, plan) = (Arc::clone(&held), Arc::clone(plan));
            let (layer, index) = (self.layer.index(), self.index);
            Box::new(move |arrived| {
                let Some(gate) = held.take() else { return };
                match arrived {
                    Arrived::Held(lease) => {
                        plan.hold(index, lease);
                        gate.release();
                    }
                    Arrived::Failed(why) => gate.fail(fatal(format!("layer {layer} expert {expert}: {why}"))),
                    Arrived::Cancelled => gate.release(),
                }
            })
        };
        let cancel = quantum.cancel_flag().unwrap_or_else(|| Arc::clone(plan.spare_cancel()));
        let id = ExpertId {
            bank: self.layer.bank(),
            expert,
        };
        if let Err(why) = self.layer.source().fetch(id, plan.rows() > 1, cancel, arrival)
            && let Some(gate) = held.take()
        {
            gate.fail(self.named(expert, why));
        }
        false
    }
}

#[cfg(test)]
mod tests;
