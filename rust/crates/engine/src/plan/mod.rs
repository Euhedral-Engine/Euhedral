//! The execution plan: the stage graph of each view of the model, the lanes they run on, and the graphs a quantum is
//! captured into (`ExecutionPlan`, `Shape`).
//!
//! [`Plan::build`] makes one list of stages per view family from the model's weights: the base model's (decode,
//! verification and prefill share the fused topology) and the MTP layer's (draft and catch-up). A list is the
//! stages in dependency order, with the dependencies and the ordering edges the shared storage and scratch add
//! ([`hazards`]). From a list the plan makes the [`StageGraphSpec`] of a view, with each stage on the lane the path
//! policy gives it ([`placement`]), and the stages that run it. The number of lanes the graphs need is the plan's
//! [`fan_out_width`](Plan::fan_out_width).

mod buffers;
pub mod builder;
mod hazards;
mod placement;
pub mod views;

use std::fmt;
use std::sync::Arc;

use euhedral_graph::{EdgeKind, GraphError, LaneId, StageGraphSpec, StageSpec};
use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_edrl::{TensorDataType, WeightFormat, WeightLayout};
use thiserror::Error;

use crate::model::Speculation;
use crate::model::{LayerIndex, LayerKind, ModelConfig, Role, WeightRef};
use crate::residency::planner::ResidencyPlan;
use crate::stages::Placed;
use buffers::Access;
pub use builder::{Entry, Formats, StageKind, WeightLookup};
use builder::{Node, base_nodes, mtp_nodes};
pub(crate) use views::Variant;
pub use views::{DECODE_MAX_ROWS, MAX_CAPTURED_ROWS, View};

/// A weight lookup that has every weight: the same described tensor for any layer and role.
struct Described(WeightRef);

impl WeightLookup for Described {
    fn weight(&self, _layer: LayerIndex, _role: Role) -> Option<&WeightRef> {
        Some(&self.0)
    }
}

/// Why a plan could not be built.
#[derive(Debug, Error)]
pub enum PlanBuildError {
    #[error("the weights have no {role:?} for layer {layer:?}")]
    MissingWeight { layer: LayerIndex, role: Role },
    #[error("{name} is stored as {format:?} in {layout:?}, which no linear format of the plan runs")]
    Format {
        name: String,
        format: WeightFormat,
        layout: WeightLayout,
    },
    #[error("unsupported geometry: {0}")]
    Geometry(String),
    #[error(transparent)]
    Graph(#[from] GraphError),
}

/// One view's stage list and graph.
struct VariantPlan {
    nodes: Arc<Vec<Node>>,
    /// The edges of each stage: its dependencies, then the ordering edges the storages add.
    edges: Vec<Vec<usize>>,
    /// The scratch region each stage takes.
    regions: Vec<Option<usize>>,
    entries: Vec<Entry>,
    /// The predecessor whose lane each stage continues (RustEE's placement).
    path: Vec<Option<usize>>,
    spec: StageGraphSpec,
    width: usize,
    /// Whether each stage ends with a signalled marker: the ends of the chains of lane continuations.
    signalled: Vec<bool>,
}

/// The execution plan of a loaded model.
pub struct Plan {
    config: Arc<ModelConfig>,
    variants: Vec<Option<VariantPlan>>,
    lanes: usize,
    draft_depth: usize,
}

impl fmt::Debug for Plan {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Plan")
            .field("lanes", &self.lanes)
            .field("draft_depth", &self.draft_depth)
            .finish_non_exhaustive()
    }
}

impl Plan {
    /// The plan of the model `config` describes, over its `weights`, run through `formats`. The MTP views are built
    /// when the artifact drafts with MTP (`residency.profile`).
    pub fn build<W: WeightLookup + ?Sized>(
        config: &Arc<ModelConfig>,
        weights: &W,
        residency: &ResidencyPlan,
        formats: &Formats,
    ) -> Result<Plan, PlanBuildError> {
        if config.layer_kinds.len() != config.num_hidden_layers {
            return Err(PlanBuildError::Geometry(format!(
                "{} layer kinds for {} layers",
                config.layer_kinds.len(),
                config.num_hidden_layers
            )));
        }
        let drafts = residency.profile.speculation == Speculation::Mtp && residency.draft_depth > 0;
        let base = Arc::new(base_nodes(config, weights)?);
        let mtp = if drafts {
            Some(Arc::new(mtp_nodes(config, weights)?))
        } else {
            None
        };
        let mut variants: Vec<Option<VariantPlan>> = Vec::with_capacity(Variant::ALL.len());
        for variant in Variant::ALL {
            let nodes = if variant.mtp() {
                mtp.clone()
            } else {
                Some(Arc::clone(&base))
            };
            variants.push(match nodes {
                Some(nodes) => Some(VariantPlan::make(variant, nodes, formats)?),
                None => None,
            });
        }
        let lanes = variants
            .iter()
            .flatten()
            .map(|plan| plan.width)
            .max()
            .unwrap_or(1)
            .max(1);
        for plan in variants.iter_mut().flatten() {
            plan.place(lanes)?;
        }
        Ok(Plan {
            config: Arc::clone(config),
            variants,
            lanes,
            draft_depth: if drafts { residency.draft_depth } else { 0 },
        })
    }

    /// The lanes the graphs of any model of this family need, from the topology of its views alone: the width of the
    /// widest view of a model with one layer of each kind and an MTP layer. The width comes from the fan-out inside a
    /// layer, so more layers do not change it; [`Plan::build`] measures the real one, and a runtime that made fewer
    /// lanes refuses the plan. The lanes of a runtime are made before the model's metadata is read, which is why the
    /// width is asked for here and not from a plan.
    pub fn lane_width() -> usize {
        let weight = WeightRef {
            name: String::new(),
            shape: vec![1, 1],
            dtype: TensorDataType::Bf16,
            format: WeightFormat::Nvfp4,
            layout: WeightLayout::ContiguousLeV1,
            ptr: DevicePtr::NULL,
            bytes: 0,
            trellis: None,
        };
        let config = ModelConfig {
            vocab_size: 1,
            hidden_size: 1,
            num_hidden_layers: 2,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            attention_head_dim: 1,
            intermediate_size: 1,
            linear_num_key_heads: 1,
            linear_num_value_heads: 1,
            linear_key_head_dim: 1,
            linear_value_head_dim: 1,
            linear_conv_kernel_dim: 1,
            rms_norm_epsilon: 1.0,
            rope_theta: 1.0,
            partial_rotary_factor: 1.0,
            max_position_embeddings: 1,
            layer_kinds: vec![LayerKind::GatedDeltaNet, LayerKind::FullAttention],
            num_experts: 0,
            tie_word_embeddings: false,
            attention_output_gate: true,
            mtp_layer_count: 1,
            draft_depth: None,
        };
        let stub = Described(weight);
        let lists = [base_nodes(&config, &stub), mtp_nodes(&config, &stub)];
        let mut width = 1;
        for nodes in lists.iter().flatten() {
            for variant in Variant::ALL {
                let regions: Vec<Option<usize>> = nodes.iter().map(|node| node.make.scratch_region(variant)).collect();
                let region_count = regions.iter().flatten().max().map_or(1, |region| region + 1);
                let access: Vec<Access> = nodes
                    .iter()
                    .zip(&regions)
                    .map(|(node, region)| Access {
                        scratch: *region,
                        ..node.make.access()
                    })
                    .collect();
                let deps: Vec<Vec<usize>> = nodes.iter().map(|node| node.deps.clone()).collect();
                let edges = hazards::with_hazards(&access, &deps, region_count);
                width = width.max(placement::fan_out_width(&edges));
            }
        }
        width
    }

    fn variant(&self, variant: Variant) -> Option<&VariantPlan> {
        self.variants[variant.index()].as_ref()
    }

    /// The model's dimensions.
    pub fn config(&self) -> &Arc<ModelConfig> {
        &self.config
    }

    /// Whether the plan has the MTP layer's views.
    pub fn drafts(&self) -> bool {
        self.draft_depth > 0
    }

    /// The lanes the plan's graphs need: the width of the widest of them, the most stages that can run at once
    /// (`LanePool` size).
    pub fn fan_out_width(&self) -> usize {
        self.lanes
    }

    /// The width of the graph of `view`: the most of its stages that no edge orders.
    pub fn width(&self, view: View) -> Option<usize> {
        self.variant(view.variant()).map(|plan| plan.width)
    }

    /// The stage graph of `view`, to be run by a `StageGraph` over a lane pool of at least
    /// [`Plan::fan_out_width`] lanes. Every edge is a submission edge: a stage orders itself behind its predecessors'
    /// markers on the device and the host never waits. `None` for a view the plan does not have (the MTP views of a
    /// model that does not draft).
    pub fn spec(&self, view: View) -> Option<StageGraphSpec> {
        self.variant(view.variant()).map(|plan| plan.spec.clone())
    }

    /// The stages of `view`, in the order of its graph.
    pub fn stages(&self, view: View) -> Option<&[Entry]> {
        self.variant(view.variant()).map(|plan| plan.entries.as_slice())
    }

    /// The kinds of the stages of `view`, in order.
    pub fn kinds(&self, view: View) -> Option<Vec<StageKind>> {
        self.stages(view)
            .map(|stages| stages.iter().map(|entry| entry.kind).collect())
    }

    /// The edges of each stage of `view`: its dependencies and the ordering edges its storages add (every one a
    /// stage before it).
    pub fn edges(&self, view: View) -> Option<&[Vec<usize>]> {
        self.variant(view.variant()).map(|plan| plan.edges.as_slice())
    }

    /// The dependencies alone of each stage of `view` (`Instruction.dependencies`).
    pub fn dependencies(&self, view: View) -> Option<Vec<Vec<usize>>> {
        self.variant(view.variant())
            .map(|plan| plan.nodes.iter().map(|node| node.deps.clone()).collect())
    }

    /// The scratch region each stage of `view` takes, if it takes one.
    pub fn scratch_regions(&self, view: View) -> Option<&[Option<usize>]> {
        self.variant(view.variant()).map(|plan| plan.regions.as_slice())
    }

    /// The predecessor whose lane each stage of `view` continues under the path policy: its critical predecessor, the one
    /// for which the stage is the critical successor (the successor with the greatest height, the first listed on
    /// ties; of several predecessors that name a join critical, the last visited). A stage with none, a root or a
    /// branch of a fan-out, takes a lane in turn. RustEE's `StageGraph` places the stages by this rule.
    pub fn path_predecessors(&self, view: View) -> Option<&[Option<usize>]> {
        self.variant(view.variant()).map(|plan| plan.path.as_slice())
    }

    /// Whether each stage of `view` ends with a signalled marker: the stages no other stage continues the lane of. A
    /// graph retires when the latest marker of each lane it used is reached, and a lane's completion word reaches only
    /// signalled markers; the last marker of a lane belongs to the end of a chain of continuations (two chains that
    /// meet on one lane both end in a signalling stage), so these stages signal and the others do not.
    pub fn signalled(&self, view: View) -> Option<&[bool]> {
        self.variant(view.variant()).map(|plan| plan.signalled.as_slice())
    }

    /// The stages with no predecessor in the graph of `view`: where a quantum enters, and what the previous quantum
    /// of the chain must be ordered before.
    pub fn roots(&self, view: View) -> Option<Vec<usize>> {
        self.edges(view).map(|edges| {
            edges
                .iter()
                .enumerate()
                .filter(|(_, predecessors)| predecessors.is_empty())
                .map(|(stage, _)| stage)
                .collect()
        })
    }

    /// The stage of `view` that every other stage precedes: the selection, whose marker covers the quantum's device
    /// work. The next quantum of a chain orders its roots behind it.
    pub fn sink(&self, view: View) -> Option<usize> {
        self.variant(view.variant()).map(|plan| plan.entries.len() - 1)
    }

    /// The rows a verification covers at most: one more than the drafts, as a view captured into graphs.
    pub fn verified_rows(&self) -> usize {
        1 + self.draft_depth
    }
}

impl VariantPlan {
    /// The list of `variant` over `nodes`: edges, regions and stages. The lanes and the graph spec follow
    /// ([`VariantPlan::place`]) once the plan knows its lane count.
    fn make(variant: Variant, nodes: Arc<Vec<Node>>, formats: &Formats) -> Result<VariantPlan, PlanBuildError> {
        let regions: Vec<Option<usize>> = nodes.iter().map(|node| node.make.scratch_region(variant)).collect();
        let region_count = regions.iter().flatten().max().map_or(1, |region| region + 1);
        let access: Vec<Access> = nodes
            .iter()
            .zip(&regions)
            .map(|(node, region)| Access {
                scratch: *region,
                ..node.make.access()
            })
            .collect();
        let deps: Vec<Vec<usize>> = nodes.iter().map(|node| node.deps.clone()).collect();
        let edges = hazards::with_hazards(&access, &deps, region_count);
        let entries = nodes
            .iter()
            .map(|node| node.make.stage(variant, formats))
            .collect::<Result<Vec<_>, _>>()?;
        let width = placement::fan_out_width(&edges);
        Ok(VariantPlan {
            nodes,
            edges,
            regions,
            entries,
            path: Vec::new(),
            spec: StageGraphSpec::new(),
            width,
            signalled: Vec::new(),
        })
    }

    /// Makes the graph over `lanes` lanes, and finds where its chains of lane continuations end.
    fn place(&mut self, lanes: usize) -> Result<(), PlanBuildError> {
        self.path = placement::path_predecessors(&self.edges);
        self.signalled = placement::chain_ends(&self.edges);
        for (entry, signal) in self.entries.iter_mut().zip(&self.signalled) {
            entry.stage = Arc::new(Placed::new(Arc::clone(&entry.launch), *signal));
        }
        // A root enters on the home lane, as Java's graph does, so a quantum's entry (the wait for its input record and
        // the embedding) has a fixed lane. That does not keep consecutive quanta on one lane: the chain of lane
        // continuations breaks where a join continues another branch (the last visited of the predecessors that name
        // it critical), so the lane a quantum ends on varies. Every other stage may run on any of the plan's lanes, and
        // RustEE's graph puts it on the lane of its path predecessor or, as a branch, on the next lane in turn.
        let compute: Vec<LaneId> = (0..lanes).map(|lane| LaneId(lane as u16)).collect();
        let mut spec = StageGraphSpec::new();
        let ids: Vec<_> = self
            .entries
            .iter()
            .zip(&self.edges)
            .map(|(entry, producers)| {
                spec.stage(StageSpec {
                    name: entry.kind.name(),
                    ordered: None,
                    lanes: if producers.is_empty() {
                        vec![LaneId(0)]
                    } else {
                        compute.clone()
                    },
                })
            })
            .collect();
        for (stage, producers) in self.edges.iter().enumerate() {
            for &producer in producers {
                spec.edge(ids[producer], ids[stage], EdgeKind::Submission);
            }
        }
        spec.validate()?;
        self.spec = spec;
        Ok(())
    }
}
