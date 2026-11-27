//! The Flash-Next model step as one plan for a row bucket (`ExecutionPlan.rowBuckets` and `Shape`): the dense stages,
//! the layer's block, its mixture of experts and the end injection for every layer in Java's order, then the final
//! mix and the head. A step runs as a quantum on the graph of its bucket: [`Buckets::bucket_for`] chooses the bucket
//! of a row count, [`FlashStep`] holds one RustEE graph per bucket, and [`FlashStep::start`] starts a quantum on the
//! graph of its bucket behind the quantum before it on the workspace.
//!
//! One quantum at a time per workspace: the layer scratch, the streams and the logits are shared by every quantum,
//! so a new quantum's first stage (the embedding, node 0, which writes the streams) waits for the sink (the head,
//! the last node, whose marker covers the quantum's device work) of the quantum before it ([`dependencies`]). The
//! host roots of the n-gram embedding touch only the quantum's own state and need no such edge.
//!
//! Departures from Java, with the reason: the step is not captured into a CUDA graph. It has host stages (the
//! expert fetches, the plan that reads the route's copy), a device-completion edge, kernel arguments the host
//! decides from the route (an expert's pair range, the finish's pair count) and the position as a kernel argument in
//! the sparse attention, so no one body holds for every quantum of a key; Java's step does not capture either
//! (`FLASH_NEXT_EXECUTION.md`, "Dynamic boundaries"). Java's decode-only prefetch and group stages and its scratch
//! chain over the layer stages of 9 or more rows have no node here.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;

use euhedral_graph::{
    EdgeKind, ExternalEdge, GraphError, GraphRuntime, LaneId, QuantumOutcome, Stage, StageGraph, StageId,
};
use euhedral_inference_edrl::FlashConfig;
use smallvec::SmallVec;
use thiserror::Error;

use super::FlashError;
use super::plan::{DensePlan, NodeKind};
use crate::quantum::{Dependency, Predecessor, Started, external_edges, no_slot_readers};
use crate::stages::Quantum;

const OP: &str = "step";

/// The rows up to which the middle bucket serves (`ExecutionPlan.rowBuckets`).
const MIDDLE_BUCKET: u32 = 16;

/// The row buckets of a model, and the width of the MoE block of each.
///
/// A bucket is the row count a plan is built for: 1 for a decode row, `min(16, max_rows)` up to 16 rows, and `max_rows`
/// above. Its MoE block names at most `min(experts, bucket * experts_per_token)` experts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Buckets {
    max_rows: u32,
    experts: u32,
    experts_per_token: u32,
}

impl Buckets {
    /// The buckets of a step of at most `max_rows` rows (the residency plan's prefill chunk) over a model of `experts`
    /// experts, `experts_per_token` of them chosen for each row.
    pub fn new(max_rows: u32, experts: u32, experts_per_token: u32) -> Result<Self, FlashError> {
        for (name, value) in [
            ("rows", max_rows),
            ("experts", experts),
            ("experts per token", experts_per_token),
        ] {
            if value == 0 {
                return Err(FlashError::refused(OP, format!("a step of 0 {name}")));
            }
        }
        if experts_per_token > experts {
            return Err(FlashError::refused(
                OP,
                format!("{experts_per_token} experts per token of {experts} experts"),
            ));
        }
        Ok(Self {
            max_rows,
            experts,
            experts_per_token,
        })
    }

    /// The bucket of a step of `rows` rows. A step of no rows, or of more than the largest bucket, is refused with
    /// the count and the limit.
    pub fn bucket_for(&self, rows: u32) -> Result<u32, FlashError> {
        if rows == 0 {
            return Err(FlashError::refused(OP, "a step of 0 rows".to_owned()));
        }
        if rows > self.max_rows {
            return Err(FlashError::refused(
                OP,
                format!("a step of {rows} rows exceeds the largest bucket of {}", self.max_rows),
            ));
        }
        Ok(match rows {
            1 => 1,
            2..=MIDDLE_BUCKET => MIDDLE_BUCKET.min(self.max_rows),
            _ => self.max_rows,
        })
    }

    /// Where the bucket of a step of `rows` rows is among the [`sizes`](Buckets::sizes): the place of its graph, which
    /// [`FlashStep`] builds in that order. Refuses what [`bucket_for`](Buckets::bucket_for) refuses, and a count that
    /// does not fit 32 bits.
    pub fn index_for(&self, rows: usize) -> Result<usize, FlashError> {
        let rows = u32::try_from(rows).map_err(|_| FlashError::refused(OP, format!("a step of {rows} rows")))?;
        let bucket = self.bucket_for(rows)?;
        self.sizes()
            .iter()
            .position(|size| *size == bucket)
            .ok_or_else(|| FlashError::refused(OP, format!("no graph for the bucket of {bucket} rows")))
    }

    /// The distinct buckets, ascending.
    pub fn sizes(&self) -> SmallVec<[u32; 3]> {
        let mut sizes: SmallVec<[u32; 3]> = [1, MIDDLE_BUCKET.min(self.max_rows), self.max_rows]
            .into_iter()
            .collect();
        sizes.dedup();
        sizes
    }

    /// The experts the MoE block of a `bucket` names at most (`MoeLayer`'s expert width).
    pub fn expert_width(&self, bucket: u32) -> u32 {
        self.experts.min(bucket.saturating_mul(self.experts_per_token))
    }

    /// The plan of the step of `bucket` rows. A row count that is not one of the [`sizes`](Buckets::sizes) is refused.
    pub fn plan(&self, config: &FlashConfig, bucket: u32) -> Result<DensePlan, FlashError> {
        if !self.sizes().contains(&bucket) {
            return Err(FlashError::refused(
                OP,
                format!("{bucket} rows is not one of the buckets {:?}", self.sizes()),
            ));
        }
        DensePlan::with_moe(config, self.expert_width(bucket))
    }
}

/// The edges a quantum starts with behind the quantum before it on the workspace (`after` is that quantum, or empty
/// when it retired): the embedding waits for its sink, and nothing else of the quantum waits for it.
pub fn dependencies(after: &[Predecessor<'_>]) -> SmallVec<[Dependency; 16]> {
    external_edges(&[], after)
}

/// The stage whose marker covers a quantum's device work: the last node of `plan`. A plan holds at most
/// [`MAX_NODES`](super::plan::MAX_NODES) nodes ([`DensePlan`] refuses more), so the index fits a stage id.
pub fn sink_of(plan: &DensePlan) -> Result<StageId, FlashError> {
    let last = plan.nodes().len() - 1;
    u16::try_from(last)
        .map(StageId)
        .map_err(|_| FlashError::refused(OP, format!("a plan of {} nodes", last + 1)))
}

/// What builds the RustEE stage of a node.
pub trait NodeStages {
    /// The stage of the node `kind` in the plan of `bucket` rows.
    fn stage(&self, bucket: u32, kind: NodeKind) -> Result<Arc<dyn Stage<Quantum>>, FlashError>;
}

/// The stages of `plan`, one for each node in node order: the order of the graph's stage ids.
pub fn stages_of(
    plan: &DensePlan,
    bucket: u32,
    source: &dyn NodeStages,
) -> Result<Vec<Arc<dyn Stage<Quantum>>>, FlashError> {
    plan.nodes()
        .iter()
        .map(|node| source.stage(bucket, node.kind))
        .collect()
}

/// What stops a step from being built or started.
#[derive(Debug, Error)]
pub enum StepError {
    #[error(transparent)]
    Flash(#[from] FlashError),
    #[error(transparent)]
    Graph(#[from] GraphError),
}

/// The graph of one bucket and the stage that ends it.
struct BucketGraph {
    bucket: u32,
    graph: Arc<StageGraph<Quantum>>,
    sink: StageId,
}

/// The step's graphs, one for each bucket, on one home lane.
#[derive(Debug)]
pub struct FlashStep {
    buckets: Buckets,
    graphs: Vec<BucketGraph>,
}

impl fmt::Debug for BucketGraph {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("BucketGraph")
            .field("bucket", &self.bucket)
            .field("sink", &self.sink)
            .finish_non_exhaustive()
    }
}

impl FlashStep {
    /// The graph of every bucket of `buckets` for `config`, its device stages on lane `home`, run on `runtime`.
    /// `retire` receives each quantum, with its outcome, once its device work has completed.
    pub fn new(
        config: &FlashConfig,
        buckets: Buckets,
        runtime: &GraphRuntime,
        home: LaneId,
        source: &dyn NodeStages,
        retire: impl Fn(Quantum, QuantumOutcome) + Send + Sync + 'static,
    ) -> Result<Self, StepError> {
        let retire: Arc<dyn Fn(Quantum, QuantumOutcome) + Send + Sync> = Arc::new(retire);
        let mut graphs = Vec::new();
        for bucket in buckets.sizes() {
            let plan = buckets.plan(config, bucket)?;
            let spec = plan.spec(home)?;
            let stages = stages_of(&plan, bucket, source)?;
            let again = Arc::clone(&retire);
            let graph = StageGraph::new(spec, stages, runtime.clone(), move |quantum, outcome| {
                again(quantum, outcome)
            })?;
            graphs.push(BucketGraph {
                bucket,
                graph,
                sink: sink_of(&plan)?,
            });
        }
        Ok(Self { buckets, graphs })
    }

    /// Starts `quantum`, whose input record the owner has prepared, on the graph of its bucket, behind `after` (the
    /// quantum before it on the workspace, until it retires).
    pub fn start(&self, quantum: Quantum, kill: Arc<AtomicBool>, after: &[&Started]) -> Result<Started, StepError> {
        let index = self.buckets.index_for(quantum.rows())?;
        let entry = self
            .graphs
            .get(index)
            .ok_or_else(|| FlashError::refused(OP, format!("no graph for the bucket of {} rows", quantum.rows())))?;
        let serial = quantum.id();
        let predecessors: SmallVec<[Predecessor<'_>; 4]> = after
            .iter()
            .map(|previous| Predecessor {
                root: previous.root,
                sink: previous.sink,
                slot_readers: &previous.slot_readers,
            })
            .collect();
        let external = dependencies(&predecessors).into_iter().map(|edge| ExternalEdge {
            from: after[edge.predecessor].reference.clone(),
            from_stage: edge.from_stage,
            to_stage: edge.to_stage,
            kind: EdgeKind::Submission,
        });
        let reference = entry.graph.start(quantum, kill, external);
        Ok(Started {
            serial,
            reference,
            root: StageId(0),
            sink: entry.sink,
            slot_readers: no_slot_readers(),
        })
    }
}

#[cfg(test)]
mod tests;
