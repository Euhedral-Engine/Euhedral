//! The nodes and edges of the dense part of a Flash-Next quantum (`ExecutionPlan`'s stage list for the non-diagnostic
//! shape): the embedding, the n-gram embedding's host roots and its device stage at the layer that holds it, and for
//! each layer the block, the layer's mixture of experts and the end injection, then the final mix and the head.
//!
//! The nodes are what a session builds a stage for; [`DensePlan::spec`] is the RustEE graph over them. A plan made
//! with [`DensePlan::with_moe`] puts a layer's MoE nodes between its block and its end injection (the shape of
//! `docs/flash-next/FLASH_NEXT_LATTICE.md`): the route and the shared expert follow the block; the plan follows the
//! route by a device-completion edge, because it reads what the route copied to the host; a fetch and an expert node
//! for each expert the block can name follow the plan, an expert after its fetch; and the finish follows the shared
//! expert and every expert, and the end injection follows the finish. A plan made with [`DensePlan::from_config`] has
//! none of them: the end injection follows the block.

use euhedral_graph::{EdgeKind, GraphError, LaneId, StageGraphSpec, StageId, StageSpec};
use euhedral_inference_edrl::FlashConfig;

use super::FlashError;
use super::ple::PARTS;

/// The most nodes a plan may have: the stages a RustEE graph holds, whose ids are 16 bits. This limit is RustEE's, not
/// the engine's, and goes once RustEE lifts it.
pub const MAX_NODES: usize = 1 << 16;

/// What a node of the dense part does.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NodeKind {
    /// The embedding lookup and the stream repeat.
    Embed,
    /// The n-gram row ids of the quantum's tokens (host).
    PleIds,
    /// One part of the gather of the n-gram records (host).
    PleGather(usize),
    /// The device half of the n-gram embedding.
    Ple,
    /// The mix, block, injection and mix of layer `n`.
    Block(usize),
    /// The injection of layer `n`'s MoE output into the streams.
    End(usize),
    /// The final mixer over the last row.
    FinalMix,
    /// The logits of the last row and the copy of what the host reads of them.
    Head,
    /// The router of layer `n`'s MoE block, and the copy of its choice to the host.
    MoeRoute(usize),
    /// The shared expert of layer `n`.
    MoeShared(usize),
    /// The grouping of layer `n`'s pairs by expert and the copy of the description to the device (host work, then a copy).
    MoePlan(usize),
    /// The `i`-th expert layer `n`'s block names, held in the device cache (host).
    MoeFetch(usize, u32),
    /// The kernels of the `i`-th expert of layer `n`'s block.
    MoeExpert(usize, u32),
    /// The sum of layer `n`'s experts, with the shared expert's gated output.
    MoeFinish(usize),
}

impl NodeKind {
    /// Whether the node runs on the host, with no lane.
    pub fn host(self) -> bool {
        matches!(self, NodeKind::PleIds | NodeKind::PleGather(_) | NodeKind::MoeFetch(..))
    }

    /// The node's name in the graph.
    pub fn name(self) -> &'static str {
        match self {
            NodeKind::Embed => "flash embed",
            NodeKind::PleIds => "flash ple ids",
            NodeKind::PleGather(_) => "flash ple gather",
            NodeKind::Ple => "flash ple",
            NodeKind::Block(_) => "flash block",
            NodeKind::End(_) => "flash end inject",
            NodeKind::FinalMix => "flash final mix",
            NodeKind::Head => "flash head",
            NodeKind::MoeRoute(_) => "flash moe route",
            NodeKind::MoeShared(_) => "flash moe shared",
            NodeKind::MoePlan(_) => "flash moe plan",
            NodeKind::MoeFetch(..) => "flash moe fetch",
            NodeKind::MoeExpert(..) => "flash moe expert",
            NodeKind::MoeFinish(_) => "flash moe finish",
        }
    }
}

/// A node and the nodes before it, by index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub kind: NodeKind,
    /// The nodes whose submission the node waits for.
    pub after: Vec<usize>,
    /// The nodes whose device work must have completed (not only been submitted) before the node runs.
    pub completed: Vec<usize>,
}

impl Node {
    fn new(kind: NodeKind, after: Vec<usize>) -> Self {
        Self {
            kind,
            after,
            completed: vec![],
        }
    }
}

/// The nodes of a model's dense part, in an order where every node follows the nodes before it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DensePlan {
    nodes: Vec<Node>,
}

impl DensePlan {
    /// The plan of `config`'s layers. The embedding of n-grams runs at the one layer `config.ple.layers` names; a configuration that names more is refused with the layers.
    pub fn from_config(config: &FlashConfig) -> Result<Self, FlashError> {
        Self::build(config, None)
    }

    /// The plan of `config`'s layers with the MoE nodes of each layer, whose blocks name at most `width` experts.
    pub fn with_moe(config: &FlashConfig, width: u32) -> Result<Self, FlashError> {
        if width == 0 {
            return Err(FlashError::refused("dense plan", "a MoE block of 0 experts".to_owned()));
        }
        Self::build(config, Some(width))
    }

    fn build(config: &FlashConfig, moe_width: Option<u32>) -> Result<Self, FlashError> {
        let layers = config.text.num_layers as usize;
        if layers == 0 {
            return Err(FlashError::refused("dense plan", "a model of 0 layers".to_owned()));
        }
        let ple_layer = match config.ple.layers.as_slice() {
            [] => None,
            [layer] => Some(
                usize::try_from(*layer)
                    .ok()
                    .filter(|layer| *layer < layers)
                    .ok_or_else(|| {
                        FlashError::refused("dense plan", format!("n-gram layer {layer} of {layers} layers"))
                    })?,
            ),
            several => {
                return Err(FlashError::refused(
                    "dense plan",
                    format!("the n-gram embedding runs at one layer, the model names {several:?}"),
                ));
            }
        };
        // The embedding, the final mix and the head; per layer a block and an end injection, and the MoE nodes.
        let moe = moe_width.map_or(0, |width| 4 + 2 * u64::from(width));
        let ple = if ple_layer.is_some() { PARTS as u64 + 2 } else { 0 };
        let count = (layers as u64).saturating_mul(2 + moe).saturating_add(3 + ple);
        if count > MAX_NODES as u64 {
            return Err(FlashError::refused(
                "dense plan",
                format!("a plan of {count} nodes exceeds the {MAX_NODES} stages a graph holds (RustEE's limit)"),
            ));
        }
        let mut nodes = vec![Node::new(NodeKind::Embed, vec![])];
        let mut last = 0;
        for layer in 0..layers {
            if ple_layer == Some(layer) {
                let ids = nodes.len();
                nodes.push(Node::new(NodeKind::PleIds, vec![]));
                let gathers: Vec<usize> = (0..PARTS)
                    .map(|part| {
                        nodes.push(Node::new(NodeKind::PleGather(part), vec![ids]));
                        nodes.len() - 1
                    })
                    .collect();
                let mut after = gathers;
                after.push(last);
                nodes.push(Node::new(NodeKind::Ple, after));
                last = nodes.len() - 1;
            }
            nodes.push(Node::new(NodeKind::Block(layer), vec![last]));
            last = nodes.len() - 1;
            if let Some(width) = moe_width {
                last = Self::push_moe(&mut nodes, layer, last, width);
            }
            nodes.push(Node::new(NodeKind::End(layer), vec![last]));
            last = nodes.len() - 1;
        }
        nodes.push(Node::new(NodeKind::FinalMix, vec![last]));
        nodes.push(Node::new(NodeKind::Head, vec![nodes.len() - 1]));
        Ok(Self { nodes })
    }

    /// Adds layer `layer`'s MoE nodes after node `block`, and returns the index of the finish.
    fn push_moe(nodes: &mut Vec<Node>, layer: usize, block: usize, width: u32) -> usize {
        let mut add = |node: Node| {
            nodes.push(node);
            nodes.len() - 1
        };
        let route = add(Node::new(NodeKind::MoeRoute(layer), vec![block]));
        let shared = add(Node::new(NodeKind::MoeShared(layer), vec![block]));
        let plan = add(Node {
            kind: NodeKind::MoePlan(layer),
            after: vec![],
            completed: vec![route],
        });
        let mut finish_after = vec![shared];
        for index in 0..width {
            let fetch = add(Node::new(NodeKind::MoeFetch(layer, index), vec![plan]));
            // The expert's lane may differ from the plan's, so it waits for the plan's copy of the description itself.
            finish_after.push(add(Node::new(NodeKind::MoeExpert(layer, index), vec![fetch, plan])));
        }
        add(Node::new(NodeKind::MoeFinish(layer), finish_after))
    }

    /// The nodes, each after the nodes it lists.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// The index of the node of `kind`.
    pub fn index_of(&self, kind: NodeKind) -> Option<usize> {
        self.nodes.iter().position(|node| node.kind == kind)
    }

    /// The graph of the nodes: the device nodes on lane `home`, the host nodes on none; an edge is a submission edge
    /// unless the node lists its predecessor as completed.
    pub fn spec(&self, home: LaneId) -> Result<StageGraphSpec, GraphError> {
        self.spec_on(home, &[])
    }

    /// As [`DensePlan::spec`], with the expert nodes free to run on any of `expert_lanes` (when there are some), so that
    /// experts of a block run beside one another; the nodes ordered after them wait for their markers.
    pub fn spec_on(&self, home: LaneId, expert_lanes: &[LaneId]) -> Result<StageGraphSpec, GraphError> {
        let mut spec = StageGraphSpec::new();
        let ids: Vec<StageId> = self
            .nodes
            .iter()
            .map(|node| {
                spec.stage(StageSpec {
                    name: node.kind.name(),
                    ordered: None,
                    lanes: match node.kind {
                        kind if kind.host() => vec![],
                        NodeKind::MoeExpert(..) if !expert_lanes.is_empty() => expert_lanes.to_vec(),
                        _ => vec![home],
                    },
                })
            })
            .collect();
        for (index, node) in self.nodes.iter().enumerate() {
            for &before in &node.after {
                spec.edge(ids[before], ids[index], EdgeKind::Submission);
            }
            for &before in &node.completed {
                spec.edge(ids[before], ids[index], EdgeKind::Completion);
            }
        }
        spec.validate()?;
        Ok(spec)
    }
}

#[cfg(test)]
mod tests;
