//! The stages of a Flash-Next model, resolved by name: where each tensor of the artifact goes into the stages of the
//! step (`Weights.java` and `Session.java`'s construction of the stages).
//!
//! Building them is two steps because the regions the stages share are sized by what the stages need:
//! 1. [`Blueprint::new`] reads the weights by name and builds everything that does not lie in the regions (the bodies
//!    of the layers over the sequence's state, the shapes, the MoE weights of each bucket), and reports the scratch it
//!    needs ([`Blueprint::scratch`]) and the bytes of the sequence's state ([`sequence_spec`]);
//! 2. [`Blueprint::stages`] takes the regions carved over an allocation of those sizes, builds the stages and the MoE
//!    layers of every bucket, and is the [`NodeStages`] the step builds its graphs from.
//!
//! Every refusal (a missing weight, a layer the kernels are not written for, regions that are too small) is returned by
//! these constructors, before a graph exists.

use std::sync::Arc;

use euhedral_graph::Stage;
use euhedral_inference_cuda::{DevicePtr, PinnedPool};
use euhedral_inference_edrl::{ExpertBank, FlashConfig, FlashLayerType};

use super::block::{BlockBody, BlockStage, EndInject, FinalMix};
use super::buffers::{FlashRegions, ScratchSizes, StreamShape};
use super::embed::EmbedStreams;
use super::gdn::{GdnBody, GdnShape, GdnWeights};
use super::head::HeadStage;
use super::hyper::{HyperShape, HyperWeights};
use super::linear::{Nvfp4Device, Nvfp4Weight};
use super::load::{FixedWeights, layer_prefix};
use super::moe::expert::MoeExpert;
use super::moe::fetch::MoeFetch;
use super::moe::layer::{MoeLayer, MoeShape, MoeWeights};
use super::moe::ops::ExpertGeometry;
use super::moe::routing::DescriptorLayout;
use super::moe::source::ExpertSource;
use super::moe::stages::{MoeFinish, MoePlan, MoeRoute, MoeShared};
use super::plan::NodeKind;
use super::ple::{NgramIds, NgramRows, PARTS, PleApply, PleGatherStage, PleIdsStage, PleShape, PleWeights};
use super::qsa::{QsaBody, QsaShape, QsaWeights};
use super::seq_state::{LayerSizes, LayerState, SequenceLayout, SequenceSpec};
use super::startup::PaddedDown;
use super::step::{Buckets, NodeStages};
use super::{FlashError, Weight};
use crate::stages::Quantum;

const OP: &str = "flash stages";

/// The scores scratch of the sparse attention: a tile of rows times the blocks they see (`QsaState`'s default).
const QSA_SCORE_BYTES: u64 = 16 << 20;

/// What a sequence of up to `max_tokens` positions keeps on the device, from the layers of `config`: the convolution
/// history and the recurrent matrices of each GDN layer, the indexer buffers of each sparse-attention layer and the
/// n-gram embedding's convolution history.
pub fn sequence_spec(config: &FlashConfig, max_tokens: u32) -> Result<SequenceSpec, FlashError> {
    let gdn = gdn_shape(config)?;
    let (history_bytes, recurrent_bytes) = gdn.state_bytes();
    let layers = config
        .text
        .layer_types
        .iter()
        .map(|kind| match kind {
            FlashLayerType::GatedDeltaNet => LayerSizes::GatedDeltaNet {
                history_bytes,
                recurrent_bytes,
            },
            FlashLayerType::SparseAttention => LayerSizes::Attention,
        })
        .collect();
    let rows = u64::from(
        config
            .ple
            .conv_kernel_size
            .saturating_sub(1)
            .saturating_mul(config.ngram.size),
    );
    let width = u64::from(config.hyper_connection.count) * u64::from(config.text.hidden_size);
    Ok(SequenceSpec {
        layers,
        max_tokens,
        ple_history_bytes: rows * width * 2,
    })
}

/// The ids of the n-gram embedding, from the configuration (`NgramIds` in Java's session).
pub fn ngram_ids(config: &FlashConfig) -> Result<NgramIds, FlashError> {
    let ngram = &config.ngram;
    NgramIds::new(
        ngram.size,
        ngram.heads_per_ngram,
        &ngram.layer_multipliers,
        &ngram.heads_vocab_sizes,
        &ngram.heads_offsets,
        config.text.eos_token_id,
    )
}

fn epsilon(config: &FlashConfig) -> f32 {
    // Java's `(float) config.rmsNormEps`.
    config.text.rms_norm_epsilon as f32
}

fn gdn_shape(config: &FlashConfig) -> Result<GdnShape, FlashError> {
    GdnShape::new(
        config.text.hidden_size,
        &config.gdn,
        &config.attention.output_gate,
        epsilon(config),
    )
}

fn hyper_shape(config: &FlashConfig) -> Result<HyperShape, FlashError> {
    HyperShape::new(
        config.hyper_connection.count,
        config.text.hidden_size,
        config.hyper_connection.lowrank,
        epsilon(config),
    )
}

/// The sites of a layer's hyper-connections, by name.
fn site(weights: &FixedWeights, prefix: &str, inject: bool) -> Result<HyperWeights, FlashError> {
    Ok(HyperWeights {
        norm: weights.weight(&format!("{prefix}/hc_norm"))?,
        down: weights.weight(&format!("{prefix}/input_mix_down"))?,
        up: weights.weight(&format!("{prefix}/input_mix_up"))?,
        inject: if inject {
            Some(weights.weight(&format!("{prefix}/block_inject"))?)
        } else {
            None
        },
    })
}

/// What a layer's block is built from: its body (over the layer's state), its two sites and, for a layer that reads
/// the cache, its place among the cache layers.
#[derive(Debug)]
struct LayerBlock {
    attention_site: HyperWeights,
    moe_site: HyperWeights,
    body: Arc<dyn BlockBody>,
    kv_layer: Option<usize>,
}

/// What a layer's MoE block is built from, for each bucket.
#[derive(Debug)]
struct LayerMoe {
    bank: u32,
    weights: MoeWeights,
    shapes: Vec<(u32, MoeShape)>,
}

/// Everything the stages are built from that does not lie in the regions.
#[derive(Debug)]
pub struct Blueprint {
    config: FlashConfig,
    hyper: HyperShape,
    streams: StreamShape,
    device: Nvfp4Device,
    buckets: Buckets,
    capacity: u32,
    blocks: Vec<LayerBlock>,
    moe: Vec<LayerMoe>,
    ple: PleBlueprint,
    embedding: Weight,
    head: Weight,
    final_site: HyperWeights,
    scratch: ScratchSizes,
}

#[derive(Debug)]
struct PleBlueprint {
    ids: Arc<NgramIds>,
    table: Arc<NgramRows>,
    shape: PleShape,
    weights: PleWeights,
    history: DevicePtr,
}

/// What [`Blueprint::new`] reads: the model's weights and tables, and the sequence's state the bodies work in.
#[derive(Debug)]
pub struct Ingredients<'a> {
    pub config: &'a FlashConfig,
    pub weights: &'a FixedWeights,
    /// The widened shared down projection of a layer, if it has one.
    pub padded_down: &'a PaddedDown,
    pub tables: &'a [(u32, Arc<NgramRows>)],
    pub banks: &'a [ExpertBank],
    pub layout: &'a SequenceLayout,
    pub buckets: Buckets,
    pub device: Nvfp4Device,
    pub max_tokens: u32,
}

impl Blueprint {
    /// Resolves every weight the stages need by name and builds the bodies, shapes and MoE weights, refusing the first
    /// thing that is missing or that the kernels are not written for.
    pub fn new(inputs: &Ingredients<'_>) -> Result<Self, FlashError> {
        let config = inputs.config;
        let hyper = hyper_shape(config)?;
        let streams = StreamShape::new(config.text.hidden_size, config.hyper_connection.count)?;
        let capacity = inputs.buckets.sizes().last().copied().unwrap_or(0);
        let layers = config.text.layer_types.len();
        if layers != config.text.num_layers as usize {
            return Err(FlashError::refused(
                OP,
                format!("{layers} layer types for {} layers", config.text.num_layers),
            ));
        }
        let mut blocks = Vec::with_capacity(layers);
        let mut moe = Vec::with_capacity(layers);
        let mut kv_layers = 0usize;
        for layer in 0..layers {
            let ordinal = u32::try_from(layer).map_err(|_| FlashError::refused(OP, format!("layer {layer}")))?;
            let kind = config.text.layer_types[layer];
            let state = inputs
                .layout
                .layer(layer)
                .ok_or_else(|| FlashError::refused(OP, format!("the sequence state has no layer {layer}")))?;
            let (body, kv_layer) = match (kind, state) {
                (FlashLayerType::GatedDeltaNet, LayerState::Gdn(state)) => (
                    Arc::new(GdnBody::new(
                        gdn_shape(config)?,
                        gdn_weights(inputs.weights, ordinal)?,
                        *state,
                        inputs.device,
                    )) as Arc<dyn BlockBody>,
                    None,
                ),
                (FlashLayerType::SparseAttention, LayerState::Attention(state)) => {
                    let shape = QsaShape::new(
                        config.text.hidden_size,
                        &config.attention,
                        &config.qsa,
                        config.text.rms_norm_epsilon,
                        inputs.max_tokens,
                        QSA_SCORE_BYTES,
                    )?;
                    let body = QsaBody::new(shape, qsa_weights(inputs.weights, ordinal)?, state, inputs.device)?;
                    let place = kv_layers;
                    kv_layers += 1;
                    (Arc::new(body) as Arc<dyn BlockBody>, Some(place))
                }
                _ => {
                    return Err(FlashError::refused(
                        OP,
                        format!("layer {layer} is {kind:?} and its state in the sequence is another kind"),
                    ));
                }
            };
            let prefix = layer_prefix(ordinal);
            blocks.push(LayerBlock {
                attention_site: site(inputs.weights, &format!("{prefix}attn_hc"), true)?,
                moe_site: site(inputs.weights, &format!("{prefix}mlp_hc"), true)?,
                body,
                kv_layer,
            });
            moe.push(layer_moe(inputs, ordinal)?);
        }
        let ple = ple_blueprint(inputs, &hyper)?;
        let embedding = inputs.weights.weight("text/token_embedding")?;
        let head = inputs.weights.weight("text/output_head")?;
        let final_site = site(inputs.weights, "text/hyper_connection_mixer", false)?;
        let mut blueprint = Self {
            config: config.clone(),
            hyper,
            streams,
            device: inputs.device,
            buckets: inputs.buckets,
            capacity,
            blocks,
            moe,
            ple,
            embedding,
            head,
            final_site,
            scratch: ScratchSizes::default(),
        };
        blueprint.scratch = blueprint.measure();
        Ok(blueprint)
    }

    /// The scratch the stages take from the regions: the largest the mixes, a block, the n-gram embedding or a MoE block
    /// of any bucket takes at the capacity rows (the layer scratch and the expansion are shared by the one stage in
    /// flight).
    fn measure(&self) -> ScratchSizes {
        let capacity = self.capacity;
        let mut sizes = ScratchSizes {
            hyper: self.hyper.scratch_bytes(capacity),
            layer: self.ple.shape.scratch_bytes(capacity),
            expansion: self.ple.shape.expansion_bytes(&self.device, capacity),
        };
        for block in &self.blocks {
            sizes.layer = sizes.layer.max(block.body.scratch_bytes(capacity));
            sizes.expansion = sizes.expansion.max(block.body.expansion_bytes(capacity));
        }
        for layer in &self.moe {
            for (_, shape) in &layer.shapes {
                let needs = MoeLayer::scratch_sizes(shape, layer.weights.shared_down_padded.is_some(), &self.device);
                sizes.layer = sizes.layer.max(needs.layer);
                sizes.expansion = sizes.expansion.max(needs.expansion);
            }
        }
        sizes
    }

    /// The scratch the regions must hold.
    pub fn scratch(&self) -> ScratchSizes {
        self.scratch
    }

    /// The streams' shape.
    pub fn streams(&self) -> StreamShape {
        self.streams
    }

    /// The rows the regions are carved for: the largest bucket.
    pub fn capacity_rows(&self) -> u32 {
        self.capacity
    }

    /// The bytes of the regions the stages need.
    pub fn region_bytes(&self) -> u64 {
        FlashRegions::bytes(self.capacity, self.streams, self.scratch)
    }

    /// The layout of the MoE block of a quantum of `bucket` rows: what the quantum's staging is made for.
    pub fn moe_layout(&self, bucket: u32) -> Result<DescriptorLayout, FlashError> {
        let experts = self.config.moe.num_experts;
        let top_k = self.config.moe.experts_per_token;
        DescriptorLayout::new(experts, top_k, bucket)
    }

    /// The stages over `regions` (carved with [`Blueprint::scratch`] and [`Blueprint::capacity_rows`]): the logits of the
    /// head go to `logits`, the n-gram records are gathered into `pool`, and the experts come from `source`.
    pub fn stages(
        self,
        regions: FlashRegions,
        logits: DevicePtr,
        pool: &Arc<PinnedPool>,
        source: &Arc<dyn ExpertSource>,
    ) -> Result<FlashStages, FlashError> {
        let config = &self.config;
        let hidden = config.text.hidden_size;
        let vocabulary = config.text.vocab_size;
        let embed = Arc::new(EmbedStreams::new(self.embedding, vocabulary, self.streams, regions)?);
        let ple_ids = Arc::new(PleIdsStage::new(Arc::clone(&self.ple.ids)));
        let gathers = (0..PARTS)
            .map(|part| PleGatherStage::new(part, Arc::clone(&self.ple.table), Arc::clone(pool)).map(Arc::new))
            .collect::<Result<Vec<_>, _>>()?;
        let ple = Arc::new(PleApply::new(
            self.ple.shape,
            self.ple.weights,
            self.ple.history,
            regions,
            self.device,
        )?);
        let mut blocks = Vec::with_capacity(self.blocks.len());
        let mut ends = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let mut stage = BlockStage::new(
                self.hyper,
                block.attention_site,
                block.moe_site,
                Arc::clone(&block.body),
                regions,
            )?;
            if let Some(place) = block.kv_layer {
                stage = stage.with_kv_layer(place);
            }
            blocks.push(Arc::new(stage));
            ends.push(Arc::new(EndInject::new(self.hyper, regions)?));
        }
        let final_mix = Arc::new(FinalMix::new(self.hyper, self.final_site, regions)?);
        let head = Arc::new(HeadStage::new(self.head, logits, &regions, hidden, vocabulary)?);
        let mut layers = Vec::with_capacity(self.buckets.sizes().len());
        for bucket in self.buckets.sizes() {
            let mut of_bucket = Vec::with_capacity(self.moe.len());
            for (index, layer) in self.moe.iter().enumerate() {
                let shape = layer
                    .shapes
                    .iter()
                    .find(|(rows, _)| *rows == bucket)
                    .map(|(_, shape)| *shape)
                    .ok_or_else(|| {
                        FlashError::refused(OP, format!("layer {index} has no MoE shape for {bucket} rows"))
                    })?;
                let index = u32::try_from(index).map_err(|_| FlashError::refused(OP, format!("layer {index}")))?;
                of_bucket.push(Arc::new(MoeLayer::new(
                    index,
                    layer.bank,
                    shape,
                    layer.weights,
                    regions,
                    self.device,
                    Arc::clone(source),
                )?));
            }
            layers.push((bucket, of_bucket));
        }
        Ok(FlashStages {
            embed,
            ple_ids,
            gathers,
            ple,
            blocks,
            ends,
            final_mix,
            head,
            layers,
        })
    }
}

/// The stages of every node of the step, shared by the graphs of all buckets where a stage does not depend on the
/// bucket.
#[derive(Debug)]
pub struct FlashStages {
    embed: Arc<EmbedStreams>,
    ple_ids: Arc<PleIdsStage>,
    gathers: Vec<Arc<PleGatherStage>>,
    ple: Arc<PleApply>,
    blocks: Vec<Arc<BlockStage>>,
    ends: Vec<Arc<EndInject>>,
    final_mix: Arc<FinalMix>,
    head: Arc<HeadStage>,
    layers: Vec<(u32, Vec<Arc<MoeLayer>>)>,
}

impl FlashStages {
    /// The head stage, whose row a quantum reads back.
    pub fn head(&self) -> &Arc<HeadStage> {
        &self.head
    }

    fn moe(&self, bucket: u32, layer: usize) -> Result<&Arc<MoeLayer>, FlashError> {
        self.layers
            .iter()
            .find(|(rows, _)| *rows == bucket)
            .and_then(|(_, layers)| layers.get(layer))
            .ok_or_else(|| FlashError::refused(OP, format!("no MoE block for layer {layer} at {bucket} rows")))
    }
}

impl NodeStages for FlashStages {
    fn stage(&self, bucket: u32, kind: NodeKind) -> Result<Arc<dyn Stage<Quantum>>, FlashError> {
        let missing = |what: &str, index: usize| FlashError::refused(OP, format!("no {what} {index}"));
        Ok(match kind {
            NodeKind::Embed => Arc::clone(&self.embed) as Arc<dyn Stage<Quantum>>,
            NodeKind::PleIds => Arc::clone(&self.ple_ids) as Arc<dyn Stage<Quantum>>,
            NodeKind::PleGather(part) => {
                Arc::clone(self.gathers.get(part).ok_or_else(|| missing("gather part", part))?)
                    as Arc<dyn Stage<Quantum>>
            }
            NodeKind::Ple => Arc::clone(&self.ple) as Arc<dyn Stage<Quantum>>,
            NodeKind::Block(layer) => {
                Arc::clone(self.blocks.get(layer).ok_or_else(|| missing("block", layer))?) as Arc<dyn Stage<Quantum>>
            }
            NodeKind::End(layer) => {
                Arc::clone(self.ends.get(layer).ok_or_else(|| missing("end", layer))?) as Arc<dyn Stage<Quantum>>
            }
            NodeKind::FinalMix => Arc::clone(&self.final_mix) as Arc<dyn Stage<Quantum>>,
            NodeKind::Head => Arc::clone(&self.head) as Arc<dyn Stage<Quantum>>,
            NodeKind::MoeRoute(layer) => Arc::new(MoeRoute::new(Arc::clone(self.moe(bucket, layer)?))),
            NodeKind::MoeShared(layer) => Arc::new(MoeShared::new(Arc::clone(self.moe(bucket, layer)?))),
            NodeKind::MoePlan(layer) => Arc::new(MoePlan::new(Arc::clone(self.moe(bucket, layer)?))),
            NodeKind::MoeFetch(layer, index) => Arc::new(MoeFetch::new(Arc::clone(self.moe(bucket, layer)?), index)),
            NodeKind::MoeExpert(layer, index) => Arc::new(MoeExpert::new(Arc::clone(self.moe(bucket, layer)?), index)),
            NodeKind::MoeFinish(layer) => Arc::new(MoeFinish::new(Arc::clone(self.moe(bucket, layer)?))),
        })
    }
}

fn gdn_weights(weights: &FixedWeights, layer: u32) -> Result<GdnWeights, FlashError> {
    let prefix = format!("{}gdn/", layer_prefix(layer));
    let named = |name: &str| weights.weight(&format!("{prefix}{name}"));
    let linear = |name: &str| weights.nvfp4(&format!("{prefix}{name}"));
    Ok(GdnWeights {
        qkv: linear("in_proj_qkv")?,
        z: linear("in_proj_z")?,
        out: linear("out_proj")?,
        a: named("in_proj_a")?,
        b: named("in_proj_b")?,
        conv: named("conv1d")?,
        a_log: named("a_log")?,
        dt_bias: named("dt_bias")?,
        norm: named("norm")?,
    })
}

fn qsa_weights(weights: &FixedWeights, layer: u32) -> Result<QsaWeights, FlashError> {
    let prefix = format!("{}attention/", layer_prefix(layer));
    let named = |name: &str| weights.weight(&format!("{prefix}{name}"));
    let linear = |name: &str| weights.nvfp4(&format!("{prefix}{name}"));
    Ok(QsaWeights {
        query: linear("q_proj")?,
        key: linear("k_proj")?,
        value: linear("v_proj")?,
        output: linear("o_proj")?,
        query_norm: named("q_norm")?,
        key_norm: named("k_norm")?,
        index: named("indexer/index_qk_proj")?,
        index_query_norm: named("indexer/q_layernorm")?,
        index_key_norm: named("indexer/k_layernorm")?,
    })
}

/// The MoE weights of a layer, its bank's place in the expert cache, and its shape for every bucket.
fn layer_moe(inputs: &Ingredients<'_>, layer: u32) -> Result<LayerMoe, FlashError> {
    let prefix = format!("{}moe/", layer_prefix(layer));
    let name = format!("{prefix}experts");
    let (bank, found) = inputs
        .banks
        .iter()
        .enumerate()
        .find(|(_, bank)| bank.name() == name)
        .ok_or_else(|| FlashError::refused(OP, format!("the artifact has no expert bank {name}")))?;
    let projection = |wanted: &str| {
        found
            .projections()
            .iter()
            .find(|projection| projection.name == wanted)
            .ok_or_else(|| FlashError::refused(OP, format!("the bank {name} has no projection {wanted}")))
    };
    let (gate_up, down) = (projection("gate_up")?, projection("down")?);
    let extent = |index: usize| {
        down.shape
            .get(index)
            .copied()
            .and_then(|value| u32::try_from(value).ok())
            .ok_or_else(|| {
                FlashError::refused(
                    OP,
                    format!("the bank {name}'s down projection has shape {:?}", down.shape),
                )
            })
    };
    // The down projection is `[hidden, inter]`.
    let geometry = ExpertGeometry::new(extent(0)?, extent(1)?, gate_up.record_offset, down.record_offset)?;
    let shared = |what: &str| inputs.weights.nvfp4(&format!("{prefix}shared_expert/{what}"));
    let padded = inputs.padded_down.weight(layer).map(|weight| {
        let sd4 = inputs
            .weights
            .nvfp4(&format!("{prefix}shared_expert/down_proj"))
            .is_ok_and(|down| down.sd4);
        Nvfp4Weight { weight, sd4 }
    });
    let weights = MoeWeights {
        router: inputs.weights.weight(&format!("{prefix}router"))?,
        shared_gate: shared("gate_proj")?,
        shared_up: shared("up_proj")?,
        shared_down: shared("down_proj")?,
        shared_down_padded: padded,
        shared_expert_gate: inputs.weights.weight(&format!("{prefix}shared_expert_gate"))?,
    };
    let moe = &inputs.config.moe;
    let mut shapes = Vec::new();
    for bucket in inputs.buckets.sizes() {
        let layout = DescriptorLayout::new(moe.num_experts, moe.experts_per_token, bucket)?;
        shapes.push((
            bucket,
            MoeShape::new(layout, geometry, moe.shared_expert_intermediate_size)?,
        ));
    }
    let bank = u32::try_from(bank).map_err(|_| FlashError::refused(OP, format!("bank {bank}")))?;
    Ok(LayerMoe { bank, weights, shapes })
}

/// The n-gram embedding: its ids, its table, its shape and its weights, at the one layer that holds it.
fn ple_blueprint(inputs: &Ingredients<'_>, hyper: &HyperShape) -> Result<PleBlueprint, FlashError> {
    let config = inputs.config;
    let layer = match config.ple.layers.as_slice() {
        [layer] => u32::try_from(*layer)
            .map_err(|_| FlashError::refused(OP, format!("the n-gram embedding's layer is {layer}")))?,
        layers => {
            return Err(FlashError::refused(
                OP,
                format!("the n-gram embedding is at one layer, the artifact names {layers:?}"),
            ));
        }
    };
    let table = inputs
        .tables
        .iter()
        .find(|(held, _)| *held == layer)
        .map(|(_, table)| Arc::clone(table))
        .ok_or_else(|| FlashError::refused(OP, format!("no n-gram table for layer {layer}")))?;
    let ids = Arc::new(ngram_ids(config)?);
    let shape = PleShape::new(
        config.text.hidden_size,
        hyper.streams(),
        config.ple.embed_dim,
        &ids,
        &table,
        config.ple.conv_kernel_size,
        config.ngram.size,
        epsilon(config),
    )?;
    let prefix = format!("{}ple/", layer_prefix(layer));
    let named = |name: &str| inputs.weights.weight(&format!("{prefix}{name}"));
    let linear = |name: &str| inputs.weights.nvfp4(&format!("{prefix}{name}"));
    let weights = PleWeights {
        key_proj: linear("key_proj")?,
        value_proj: linear("value_proj")?,
        norm_key: named("norm_key")?,
        norm_query: named("norm_query")?,
        norm_conv: named("norm_conv")?,
        convolution: named("conv1d")?,
    };
    Ok(PleBlueprint {
        ids,
        table,
        shape,
        weights,
        history: inputs.layout.ple_history(),
    })
}

#[cfg(test)]
mod tests;
