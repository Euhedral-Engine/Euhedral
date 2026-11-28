//! The execution plan: the stage graph of each view of the model, the lanes they run on, and the graphs a quantum is
//! captured into.
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
mod staging;
pub mod views;

use std::fmt;
use std::sync::Arc;

use euhedral_graph::{EdgeKind, GraphError, LaneId, Stage, StageGraphSpec, StageId, StageSpec};
use euhedral_inference_cuda::DevicePtr;
use euhedral_inference_edrl::{TensorDataType, WeightFormat, WeightLayout};
use thiserror::Error;

use crate::model::Speculation;
use crate::model::dflash2::Dflash2Config;
use crate::model::{LayerIndex, LayerKind, ModelConfig, Residence, Role, WeightRef};
use crate::quantum::home::lane_for;
use crate::residency::disk::DiskUse;
use crate::residency::planner::ResidencyPlan;
use crate::stages::transfer::DiskTier;
use crate::stages::{Chain, Placed, Quantum};
use buffers::Access;
pub use builder::{Entry, Formats, StageKind, WeightLookup};
use builder::{Make, Node, base_nodes, block_nodes, context_nodes, mtp_nodes};
pub use staging::StagingRing;
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
    #[error("{name} is held in host memory and the staging ring has no slot")]
    NoSlots { name: String },
    #[error("{name} is held in host memory and the plan was given no staging ring")]
    NoStagingRing { name: String },
    #[error("the plan has {count} stages in a graph, more than a graph holds")]
    TooManyUnits { count: usize },
    #[error("{name} is {bytes} bytes and a staging slot holds {slot_bytes}")]
    WeightTooLarge { name: String, bytes: u64, slot_bytes: u64 },
    #[error("{name} is held in host memory but no stage of the plan can read it from a staging slot")]
    NotStageable { name: String },
    #[error("{name} is held on disk and the plan was given no disk window")]
    NoDiskWindow { name: String },
    #[error("the disk window has {window} slots, which is not a whole multiple of the {staging} staging slots")]
    WindowSlots { window: usize, staging: usize },
    #[error("{name} is read through {bytes} bytes of the disk window and a window slot holds {slot_bytes}")]
    ReadTooLarge { name: String, bytes: u64, slot_bytes: u64 },
    #[error("the plan stages weights and was given no transfer lane")]
    NoTransferLane,
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
    /// The stages that run in one frame, each a list of consecutive stages ([`placement::fused_units`]).
    units: Vec<Vec<usize>>,
    /// The edges between the units: for each, the units before it.
    unit_edges: Vec<Vec<usize>>,
    /// What a graph runs for each unit.
    unit_stages: Vec<Arc<dyn Stage<Quantum>>>,
    width: usize,
    /// The lane offset of each stage from its quantum's home lane (0 for a transfer, which has a lane of its own).
    offsets: Vec<u16>,
    /// Whether each stage is a weight transfer.
    transfers: Vec<bool>,
    /// The lane every transfer runs on.
    transfer_lane: Option<LaneId>,
    /// The staging slots the view reads (0 when it has no transfer).
    slots: usize,
    /// The stages that are not transfers, and the edges between them as the view has them with no transfer (`None`
    /// when the view has no transfer: the stages and edges are then the plan's own). Lanes and width come from these.
    compute: Option<Compute>,
    /// For each staging slot, the unit that reads it last.
    slot_readers: Vec<Option<StageId>>,
    /// For each staging slot that is read, its first transfer's unit.
    first_transfers: Vec<(usize, StageId)>,
    /// The disk reads of the view, in the order its copies consume them.
    disk_uses: Arc<[DiskUse]>,
}

/// The stages of a view that are not transfers, and the edges between them.
struct Compute {
    /// The index in the view of each compute stage.
    nodes: Vec<usize>,
    edges: Vec<Vec<usize>>,
}

/// The execution plan of a loaded model.
pub struct Plan {
    config: Arc<ModelConfig>,
    variants: Vec<Option<VariantPlan>>,
    lanes: usize,
    draft_depth: usize,
    /// Whether the drafts are the DFlash2 drafter's (its context and block views, and the taps of the base views), not
    /// the MTP layer's.
    dflash2: bool,
    /// The disk tier the plan's views read through, when any of them reads a weight from disk.
    disk: Option<Arc<DiskTier>>,
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
    ///
    /// With a `staging` ring, each use of a weight whose [`Residence`] is host memory reads a slot of the ring, and a
    /// transfer of its own, on `transfer_lane`, fills the slot first; the weight's offset in the host arena is the one
    /// its `Residence::Host` names, and the arena is the lookup's. The MTP views stage nothing. A weight on disk, or in
    /// host memory with no ring, is refused; with none held off the device the plan is the one of resident weights.
    pub fn build<W: WeightLookup + ?Sized>(
        config: &Arc<ModelConfig>,
        weights: &W,
        residency: &ResidencyPlan,
        formats: &Formats,
        staging: Option<&StagingRing>,
        transfer_lane: Option<LaneId>,
    ) -> Result<Plan, PlanBuildError> {
        Self::build_tiered(config, weights, residency, formats, staging, transfer_lane, None)
    }

    /// [`Plan::build`] over weights that may also be on disk: with a `disk` tier, a weight whose [`Residence`] is disk is
    /// read through the tier's window into a slot of the ring, and a transfer of its own copies it from there. The use
    /// `u` of a ring slot `s` reads the window slot `s + slots * (k mod (window / slots))`, `k` counting the disk uses
    /// of `s` before it in the view, so the read of the next use of a slot runs while the copy of the last one does.
    pub fn build_tiered<W: WeightLookup + ?Sized>(
        config: &Arc<ModelConfig>,
        weights: &W,
        residency: &ResidencyPlan,
        formats: &Formats,
        staging: Option<&StagingRing>,
        transfer_lane: Option<LaneId>,
        disk: Option<&Arc<DiskTier>>,
    ) -> Result<Plan, PlanBuildError> {
        if config.layer_kinds.len() != config.num_hidden_layers {
            return Err(PlanBuildError::Geometry(format!(
                "{} layer kinds for {} layers",
                config.layer_kinds.len(),
                config.num_hidden_layers
            )));
        }
        let drafts = residency.profile.speculation == Speculation::Mtp && residency.draft_depth > 0;
        let dflash2 = residency.profile.speculation == Speculation::Dflash2 && residency.draft_depth > 0;
        let mut base = base_nodes(config, weights, dflash2)?;
        let mut slots = 0;
        if staging.is_none()
            && let Some(name) = staging::first_held(&base)
        {
            return Err(PlanBuildError::NoStagingRing { name });
        }
        if let Some(ring) = staging
            && let Some(staged) = staging::staged(&base, weights.host_arena().as_ref(), ring, disk)?
        {
            if transfer_lane.is_none() {
                return Err(PlanBuildError::NoTransferLane);
            }
            base = staged;
            slots = ring.slots;
        }
        let base = Arc::new(base);
        let mtp = if drafts {
            Some(Arc::new(mtp_nodes(config, weights)?))
        } else {
            None
        };
        // The drafter's views stage nothing: its weights, and the head the block scores with, are held on the device.
        let drafter = if dflash2 {
            let context = context_nodes(config, weights)?;
            let block = block_nodes(config, weights)?;
            if let Some(name) = staging::first_held(&block) {
                return Err(PlanBuildError::NoStagingRing { name });
            }
            Some((Arc::new(context), Arc::new(block)))
        } else {
            None
        };
        let mut variants: Vec<Option<VariantPlan>> = Vec::with_capacity(Variant::ALL.len());
        for variant in Variant::ALL {
            let nodes = if variant.mtp() {
                mtp.clone()
            } else if variant.dflash() {
                drafter.as_ref().map(|(context, block)| match variant {
                    Variant::Context => Arc::clone(context),
                    _ => Arc::clone(block),
                })
            } else {
                Some(Arc::clone(&base))
            };
            variants.push(match nodes {
                Some(nodes) => Some(VariantPlan::make(
                    variant,
                    nodes,
                    formats,
                    if variant.mtp() || variant.dflash() { 0 } else { slots },
                    transfer_lane,
                )?),
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
            plan.place()?;
        }
        let variants_read_disk = variants.iter().flatten().any(|plan| !plan.disk_uses.is_empty());
        Ok(Plan {
            config: Arc::clone(config),
            variants,
            lanes,
            draft_depth: if drafts || dflash2 { residency.draft_depth } else { 0 },
            dflash2,
            disk: disk.filter(|_| variants_read_disk).cloned(),
        })
    }

    /// The node lists (base views, then MTP views, then with `dflash2` the tapped base views and the drafter's) of a
    /// model of the family whose layers are `kinds`, over described weights: the plan's topology without a model.
    fn probe_nodes(kinds: Vec<LayerKind>, dflash2: bool) -> Vec<Vec<Node>> {
        let weight = WeightRef {
            name: String::new(),
            shape: vec![1, 1],
            dtype: TensorDataType::Bf16,
            format: WeightFormat::Nvfp4,
            layout: WeightLayout::ContiguousLeV1,
            ptr: DevicePtr::NULL,
            bytes: 0,
            residence: Residence::Device,
            trellis: None,
        };
        let config = ModelConfig {
            vocab_size: 1,
            hidden_size: 1,
            num_hidden_layers: kinds.len(),
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
            layer_kinds: kinds,
            num_experts: 0,
            tie_word_embeddings: false,
            attention_output_gate: true,
            mtp_layer_count: 1,
            draft_depth: None,
            dflash2: None,
        };
        let drafter = Dflash2Config {
            layers: 1,
            hidden: 1,
            intermediate: 1,
            heads: 1,
            kv_heads: 1,
            head_dim: 1,
            block_size: 1,
            mask_token: 0,
            conv_kernel: 1,
            conv_group: 1,
            selector_top_k: 1,
            selector_rank: 1,
            sliding_window: 1,
            vocab: 1,
            eps: 1.0,
            theta: 1.0,
            taps: (0..config.layer_kinds.len()).collect(),
        };
        let drafting = ModelConfig {
            dflash2: Some(drafter),
            ..config.clone()
        };
        let stub = Described(weight);
        let mut views = vec![base_nodes(&config, &stub, false), mtp_nodes(&config, &stub)];
        if dflash2 {
            views.push(base_nodes(&drafting, &stub, true));
            views.push(context_nodes(&drafting, &stub));
            views.push(block_nodes(&drafting, &stub));
        }
        views.into_iter().flatten().collect()
    }

    /// The lanes the graphs of a model of this family need, from the topology of its views alone: the width of the
    /// widest view of a model with both kinds of layer and an MTP layer, and with `dflash2` (the artifact holds the
    /// DFlash2 drafter) each layer tapped beside the next layer's projections and the drafter's views. The width
    /// comes from the fan-out inside a layer, so more layers do not change it; [`Plan::build`] measures the real one,
    /// and a runtime that made fewer lanes refuses the plan. The lanes are made once the artifact is parsed and
    /// before the plan is built, which is why the width is asked for here and not from a plan.
    pub fn lane_width(dflash2: bool) -> usize {
        let mut width = 1;
        for nodes in Self::probe_nodes(
            vec![
                LayerKind::GatedDeltaNet,
                LayerKind::GatedDeltaNet,
                LayerKind::FullAttention,
                LayerKind::GatedDeltaNet,
            ],
            dflash2,
        ) {
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
                let edges = hazards::with_hazards(&access, &deps, region_count, 0);
                width = width.max(placement::fan_out_width(&edges));
            }
        }
        width
    }

    /// The most stages any view of the model the engine runs can have, from the topology alone: the stages of
    /// [`ModelConfig::SUPPORTED_LAYERS`] layers of the kind with more of them. A quantum records a marker per stage it
    /// runs, so this bounds the markers one lane records for a quantum, and sizes the lanes' event rings before the
    /// model is read. [`Plan::build`] has the real count.
    pub fn stage_bound() -> usize {
        [LayerKind::GatedDeltaNet, LayerKind::FullAttention]
            .into_iter()
            .flat_map(|kind| Self::probe_nodes(vec![kind; ModelConfig::SUPPORTED_LAYERS], true))
            .map(|nodes| nodes.len())
            .max()
            .unwrap_or(0)
    }

    fn variant(&self, variant: Variant) -> Option<&VariantPlan> {
        self.variants[variant.index()].as_ref()
    }

    /// The model's dimensions.
    pub fn config(&self) -> &Arc<ModelConfig> {
        &self.config
    }

    /// Whether the plan has a drafter's views: the MTP layer's, or the DFlash2 drafter's.
    pub fn drafts(&self) -> bool {
        self.draft_depth > 0
    }

    /// Whether the drafter is the DFlash2 drafter, whose rows the base views tap.
    pub fn dflash2(&self) -> bool {
        self.dflash2
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

    /// The graph of `view` for a quantum homed on lane `home` of `stage_lanes` stage lanes: its fused stages
    /// ([`Plan::unit_stages`]), each on `(home + offset) mod stage_lanes`. `stage_lanes` must be at least the plan's [`Plan::lane_width`].
    pub fn spec_homed(
        &self,
        view: View,
        home: usize,
        stage_lanes: usize,
    ) -> Option<Result<StageGraphSpec, GraphError>> {
        self.variant(view.variant())
            .map(|plan| plan.spec_homed(home, stage_lanes))
    }

    /// The stages of `view`, in the order of its graph.
    pub fn stages(&self, view: View) -> Option<&[Entry]> {
        self.variant(view.variant()).map(|plan| plan.entries.as_slice())
    }

    /// What a graph runs for `view`: its stages with each run of single links on one lane fused into one stage, in
    /// the order of the graph [`Plan::spec_homed`] gives.
    pub fn unit_stages(&self, view: View) -> Option<&[Arc<dyn Stage<Quantum>>]> {
        self.variant(view.variant()).map(|plan| plan.unit_stages.as_slice())
    }

    /// The stages of `view` that run as one stage of the graph, each a list of stages (indices into
    /// [`Plan::stages`]) that run one after the other, in the order of the graph. A unit's stages are consecutive in a
    /// view with no transfer; with transfers, the transfer listed among them is a unit of its own and is skipped.
    pub fn units(&self, view: View) -> Option<&[Vec<usize>]> {
        self.variant(view.variant()).map(|plan| plan.units.as_slice())
    }

    /// The units that must complete before each unit of `view` ([`Plan::units`]).
    pub fn unit_edges(&self, view: View) -> Option<&[Vec<usize>]> {
        self.variant(view.variant()).map(|plan| plan.unit_edges.as_slice())
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
    /// branch of a fan-out, starts a chain on a lane of its own ([`placement::chain_lanes`]); the rest follow this rule.
    pub fn path_predecessors(&self, view: View) -> Option<&[Option<usize>]> {
        self.variant(view.variant()).map(|plan| plan.path.as_slice())
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

    /// Whether quanta of `view` are captured into graphs: [`View::capturable`], and the view has no weight transfer. A
    /// captured body runs on one lane, so a staged view's copies would run in line with its kernels and overlap
    /// nothing.
    pub fn capturable(&self, view: View) -> bool {
        view.capturable() && self.variant(view.variant()).is_none_or(|plan| plan.compute.is_none())
    }

    /// The unit of `view` that reads each staging slot last, for the transfers of the quantum that follows to order
    /// themselves behind; empty for a view that stages nothing. The slot a view never reads is `None`. The stage that
    /// reads a slot last is the last of its unit, so the unit completes when the slot is free and the quantum's
    /// remaining stages are not waited for.
    pub fn slot_readers(&self, view: View) -> &[Option<StageId>] {
        self.variant(view.variant())
            .map_or(&[], |plan| plan.slot_readers.as_slice())
    }

    /// The first transfer of each staging slot of `view`: the slot and the transfer's unit. These are roots of the
    /// view's graph, together with its first stage. Empty for a view that stages nothing.
    pub fn first_transfers(&self, view: View) -> &[(usize, StageId)] {
        self.variant(view.variant())
            .map_or(&[], |plan| plan.first_transfers.as_slice())
    }

    /// The disk reads of `view`, in the order its copies consume them: what a quantum of the view registers with the
    /// disk reader as it starts. Empty for a view that reads nothing from disk.
    pub fn disk_uses(&self, view: View) -> &[DiskUse] {
        self.variant(view.variant()).map_or(&[], |plan| &plan.disk_uses)
    }

    /// The disk tier the plan's views read through; `None` when none of them reads from disk.
    pub fn disk_tier(&self) -> Option<&Arc<DiskTier>> {
        self.disk.as_ref()
    }

    /// The rows a verification covers at most: one more than the drafts, as a view captured into graphs.
    pub fn verified_rows(&self) -> usize {
        1 + self.draft_depth
    }
}

impl VariantPlan {
    /// The list of `variant` over `nodes`: edges, regions and stages. The lanes and the graph spec follow
    /// ([`VariantPlan::place`]) once the plan knows its lane count. `slots` is the number of staging slots the list
    /// reads (0 for a list with no transfer).
    fn make(
        variant: Variant,
        nodes: Arc<Vec<Node>>,
        formats: &Formats,
        slots: usize,
        transfer_lane: Option<LaneId>,
    ) -> Result<VariantPlan, PlanBuildError> {
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
        let edges = hazards::with_hazards(&access, &deps, region_count, slots);
        let entries = nodes
            .iter()
            .map(|node| node.make.stage(variant, formats))
            .collect::<Result<Vec<_>, _>>()?;
        let transfers: Vec<bool> = nodes
            .iter()
            .map(|node| node.make.kind() == StageKind::WeightTransfer)
            .collect();
        let disk_uses: Arc<[DiskUse]> = nodes
            .iter()
            .filter_map(|node| match &node.make {
                Make::DiskTransfer { use_, .. } => Some(use_.clone()),
                _ => None,
            })
            .collect();
        // Lanes and width come from the stages that are not transfers, ordered as they are with none: a transfer
        // runs on its own lane and must not change where the rest run.
        let compute = transfers.iter().any(|&transfer| transfer).then(|| {
            let kept: Vec<usize> = (0..nodes.len()).filter(|&node| !transfers[node]).collect();
            let mut position = vec![usize::MAX; nodes.len()];
            for (index, &node) in kept.iter().enumerate() {
                position[node] = index;
            }
            let access: Vec<Access> = kept
                .iter()
                .map(|&node| Access {
                    slot_read: None,
                    ..access[node].clone()
                })
                .collect();
            let deps: Vec<Vec<usize>> = kept
                .iter()
                .map(|&node| {
                    deps[node]
                        .iter()
                        .filter(|&&dependency| !transfers[dependency])
                        .map(|&dependency| position[dependency])
                        .collect()
                })
                .collect();
            Compute {
                edges: hazards::with_hazards(&access, &deps, region_count, 0),
                nodes: kept,
            }
        });
        let width = placement::fan_out_width(compute.as_ref().map_or(&edges, |compute| &compute.edges));
        Ok(VariantPlan {
            nodes,
            edges,
            regions,
            entries,
            path: Vec::new(),
            spec: StageGraphSpec::new(),
            units: Vec::new(),
            unit_edges: Vec::new(),
            unit_stages: Vec::new(),
            width,
            offsets: Vec::new(),
            transfers,
            transfer_lane,
            slots,
            compute,
            slot_readers: Vec::new(),
            first_transfers: Vec::new(),
            disk_uses,
        })
    }

    /// Makes the graph over as many lanes as the view is wide, and its stages.
    fn place(&mut self) -> Result<(), PlanBuildError> {
        let count = self.entries.len();
        let compute_edges = self.compute.as_ref().map_or(&self.edges, |compute| &compute.edges);
        let path = placement::path_predecessors(compute_edges);
        for entry in self.entries.iter_mut().filter(|entry| !entry.own_stage) {
            entry.stage = Arc::new(Placed::new(Arc::clone(&entry.launch)));
        }
        // Each chain of lane continuations has one lane offset, fixed here, so the one lane a stage is allowed is the
        // lane RustEE's graph puts it on: it never picks a branch's lane at random, and the branches of a fan-out do
        // not share one. A root is at offset 0, the quantum's home lane.
        let chain = placement::chain_lanes(compute_edges, self.width);
        match &self.compute {
            None => {
                self.path = path;
                self.offsets = chain;
            }
            Some(compute) => {
                self.path = vec![None; count];
                self.offsets = vec![0; count];
                for (index, &node) in compute.nodes.iter().enumerate() {
                    self.path[node] = path[index].map(|predecessor| compute.nodes[predecessor]);
                    self.offsets[node] = chain[index];
                }
            }
        }
        // A transfer's lane is not a compute lane: no stage joins it in a unit.
        let lane_keys: Vec<u16> = (0..count)
            .map(|node| {
                if self.transfers[node] {
                    u16::MAX
                } else {
                    self.offsets[node]
                }
            })
            .collect();
        // The stage that reads a slot last ends its unit: the transfers that refill the slot in the next quantum then
        // wait for it, not for the stages that follow.
        let mut ends = vec![false; count];
        let mut last_reader: Vec<Option<usize>> = vec![None; self.slots];
        for (index, node) in self.nodes.iter().enumerate() {
            if let Make::Staged { slot, .. } = &node.make {
                last_reader[*slot] = Some(index);
            }
        }
        for node in last_reader.into_iter().flatten() {
            ends[node] = true;
        }
        self.units = placement::fused_units(&self.edges, &lane_keys, &ends);
        debug_assert!(
            self.units
                .iter()
                .all(|unit| unit.len() == 1 || unit.iter().all(|&node| !self.entries[node].own_stage)),
            "a stage with a gate of its own runs alone in its unit"
        );
        let mut unit_of = vec![0usize; count];
        for (unit, nodes) in self.units.iter().enumerate() {
            for &node in nodes {
                unit_of[node] = unit;
            }
        }
        self.unit_edges = self
            .units
            .iter()
            .enumerate()
            .map(|(unit, nodes)| {
                let mut before: Vec<usize> = Vec::new();
                for &node in &self.edges[nodes[0]] {
                    let other = unit_of[node];
                    if other != unit && !before.contains(&other) {
                        before.push(other);
                    }
                }
                before
            })
            .collect();
        self.unit_stages = self
            .units
            .iter()
            .map(|nodes| match nodes.as_slice() {
                [only] => Arc::clone(&self.entries[*only].stage),
                _ => {
                    let launches = nodes
                        .iter()
                        .map(|&node| Arc::clone(&self.entries[node].launch))
                        .collect();
                    Arc::new(Placed::new(Arc::new(Chain::new(launches)))) as Arc<dyn Stage<Quantum>>
                }
            })
            .collect();
        self.place_slots(&unit_of)?;
        let spec = self.node_spec_homed(0, self.width.max(1))?;
        self.spec = spec;
        Ok(())
    }

    /// The unit that reads each staging slot last, and the unit of each slot's first transfer.
    fn place_slots(&mut self, unit_of: &[usize]) -> Result<(), PlanBuildError> {
        let count = self.units.len();
        if u16::try_from(count).is_err() {
            return Err(PlanBuildError::TooManyUnits { count });
        }
        let unit = |node: usize| StageId(u16::try_from(unit_of[node]).unwrap_or(u16::MAX));
        let mut readers: Vec<Option<StageId>> = vec![None; self.slots];
        let mut first: Vec<(usize, StageId)> = Vec::new();
        for (index, node) in self.nodes.iter().enumerate() {
            match &node.make {
                Make::Staged { slot, .. } => {
                    readers[*slot] = Some(unit(index));
                }
                Make::Transfer { slot, .. } | Make::DiskTransfer { slot, .. }
                    if !first.iter().any(|(seen, _)| seen == slot) =>
                {
                    first.push((*slot, unit(index)));
                }
                _ => {}
            }
        }
        self.slot_readers = readers;
        self.first_transfers = first;
        Ok(())
    }

    /// The graph with every stage alone, each on lane `(home + offset) mod stage_lanes`.
    fn node_spec_homed(&self, home: usize, stage_lanes: usize) -> Result<StageGraphSpec, GraphError> {
        let names: Vec<_> = self.entries.iter().map(|entry| entry.kind.name()).collect();
        self.graph(home, stage_lanes, &names, self.entries.len(), &self.edges, |node| node)
    }

    /// The graph a quantum runs: the units, each on the lane `(home + offset) mod stage_lanes` of its stages.
    fn spec_homed(&self, home: usize, stage_lanes: usize) -> Result<StageGraphSpec, GraphError> {
        let names: Vec<_> = self
            .units
            .iter()
            .map(|nodes| self.entries[nodes[0]].kind.name())
            .collect();
        self.graph(home, stage_lanes, &names, self.units.len(), &self.unit_edges, |unit| {
            self.units[unit][0]
        })
    }

    /// A graph of `count` stages named `names` over `edges` (the stages before each), the lane of each that of
    /// the node `node_of` gives: the transfer lane for a transfer. Every edge is a submission edge.
    fn graph(
        &self,
        home: usize,
        stage_lanes: usize,
        names: &[&'static str],
        count: usize,
        edges: &[Vec<usize>],
        node_of: impl Fn(usize) -> usize,
    ) -> Result<StageGraphSpec, GraphError> {
        let mut spec = StageGraphSpec::new();
        let ids: Vec<_> = (0..count)
            .map(|stage| {
                let node = node_of(stage);
                let lane = match self.transfer_lane {
                    Some(lane) if self.transfers[node] => lane,
                    _ => LaneId(lane_for(home, usize::from(self.offsets[node]), stage_lanes) as u16),
                };
                spec.stage(StageSpec {
                    name: names[stage],
                    ordered: None,
                    lanes: vec![lane],
                })
            })
            .collect();
        for (stage, producers) in edges.iter().enumerate() {
            for &producer in producers {
                spec.edge(ids[producer], ids[stage], EdgeKind::Submission);
            }
        }
        spec.validate()?;
        Ok(spec)
    }
}
