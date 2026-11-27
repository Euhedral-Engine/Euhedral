//! The nodes and edges of the dense part of a Flash-Next quantum (`ExecutionPlan`'s stage list for the non-diagnostic
//! shape): the embedding, the n-gram embedding's host roots and its device stage at the layer that holds it, and for
//! each layer the block and the end injection, then the final mix.
//!
//! The nodes are what a session builds a stage for; [`DensePlan::spec`] is the RustEE graph over them. The MoE
//! layer's nodes go between a layer's block and its end injection: the end injection waits for the block now, and
//! waits for the last MoE node once there is one.

use euhedral_graph::{EdgeKind, GraphError, LaneId, StageGraphSpec, StageId, StageSpec};
use euhedral_inference_edrl::FlashConfig;

use super::FlashError;
use super::ple::PARTS;

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
}

impl NodeKind {
    /// Whether the node runs on the host, with no lane.
    pub fn host(self) -> bool {
        matches!(self, NodeKind::PleIds | NodeKind::PleGather(_))
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
        }
    }
}

/// A node and the nodes before it, by index.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Node {
    pub kind: NodeKind,
    pub after: Vec<usize>,
}

/// The nodes of a model's dense part, in an order where every node follows the nodes before it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DensePlan {
    nodes: Vec<Node>,
}

impl DensePlan {
    /// The plan of `config`'s layers. The embedding of n-grams runs at the one layer `config.ple.layers` names, as it
    /// does in Java; a configuration that names more is refused with the layers.
    pub fn from_config(config: &FlashConfig) -> Result<Self, FlashError> {
        let layers = config.text.num_layers as usize;
        if layers == 0 {
            return Err(FlashError::refused("dense plan", "a model of 0 layers".to_owned()));
        }
        let ple_layer = match config.ple.layers.as_slice() {
            [] => None,
            [layer] => Some(usize::try_from(*layer).ok().filter(|layer| *layer < layers).ok_or_else(|| {
                FlashError::refused("dense plan", format!("n-gram layer {layer} of {layers} layers"))
            })?),
            several => {
                return Err(FlashError::refused(
                    "dense plan",
                    format!("the n-gram embedding runs at one layer, the model names {several:?}"),
                ));
            }
        };
        let mut nodes = vec![Node {
            kind: NodeKind::Embed,
            after: vec![],
        }];
        let mut last = 0;
        for layer in 0..layers {
            if ple_layer == Some(layer) {
                let ids = nodes.len();
                nodes.push(Node {
                    kind: NodeKind::PleIds,
                    after: vec![],
                });
                let gathers: Vec<usize> = (0..PARTS)
                    .map(|part| {
                        nodes.push(Node {
                            kind: NodeKind::PleGather(part),
                            after: vec![ids],
                        });
                        nodes.len() - 1
                    })
                    .collect();
                let mut after = gathers;
                after.push(last);
                nodes.push(Node {
                    kind: NodeKind::Ple,
                    after,
                });
                last = nodes.len() - 1;
            }
            nodes.push(Node {
                kind: NodeKind::Block(layer),
                after: vec![last],
            });
            nodes.push(Node {
                kind: NodeKind::End(layer),
                after: vec![nodes.len() - 1],
            });
            last = nodes.len() - 1;
        }
        nodes.push(Node {
            kind: NodeKind::FinalMix,
            after: vec![last],
        });
        Ok(Self { nodes })
    }

    /// The nodes, each after the nodes it lists.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    /// The index of the node of `kind`.
    pub fn index_of(&self, kind: NodeKind) -> Option<usize> {
        self.nodes.iter().position(|node| node.kind == kind)
    }

    /// The graph of the nodes: the device nodes on lane `home`, the host nodes on none, every edge a submission edge.
    pub fn spec(&self, home: LaneId) -> Result<StageGraphSpec, GraphError> {
        let mut spec = StageGraphSpec::new();
        let ids: Vec<StageId> = self
            .nodes
            .iter()
            .map(|node| {
                spec.stage(StageSpec {
                    name: node.kind.name(),
                    ordered: None,
                    lanes: if node.kind.host() { vec![] } else { vec![home] },
                })
            })
            .collect();
        for (index, node) in self.nodes.iter().enumerate() {
            for &before in &node.after {
                spec.edge(ids[before], ids[index], EdgeKind::Submission);
            }
        }
        spec.validate()?;
        Ok(spec)
    }
}

#[cfg(test)]
mod tests;
