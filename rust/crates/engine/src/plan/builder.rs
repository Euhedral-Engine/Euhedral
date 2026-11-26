//! The stage lists of the plan's views, in the order and with the dependencies of Java's views
//! (`ExecutionPlan.fullModel` rewritten by `prefillView` for the base model's views, `mtpDraft` for the MTP layer's),
//! and the stages that run them.
//!
//! The base views are the fused form every executed view uses: the rounded residual add and the next RMSNorm in one
//! launch, the joint GDN A/B projection with its control in one launch, and the GDN control ahead of the query/key
//! and value/z projections. The MTP layer's views are the unfused form `mtpDraft` builds. The pilot's text linears
//! are trellis, so no view fuses a gate/up projection with its SwiGLU (a format that offered it would).

use std::fmt;
use std::iter::once;
use std::sync::Arc;

use euhedral_graph::Stage;
use euhedral_inference_cuda::{DevicePtr, HostWeightsArena};
use euhedral_inference_edrl::{WeightFormat, WeightLayout};

use super::PlanBuildError;
use super::buffers::Access;
use super::views::Variant;
use crate::formats::{LinearFormat, TokenEmbedding};
use crate::model::{LayerIndex, LayerKind, ModelConfig, MtpRole, NormKind, Role, WeightRef, Weights};
use crate::stages::attention::{AttentionCausal, KvAppend, KvSource, QkNormRope};
use crate::stages::elementwise::{ResidualAdd, Swiglu};
use crate::stages::embed::Embed;
use crate::stages::gdn::{GdnConvolution, GdnGatedRmsNorm, GdnProjectControl, GdnRecurrence};
use crate::stages::linear::{Linear, LinearRows};
use crate::stages::mtp::MtpStem;
use crate::stages::norm::{NormRows, ResidualRmsNorm, RmsNormUnitOffset};
use crate::stages::sampling::{Argmax, CopyRows, SelectedRows, SelectedToken};
use crate::stages::transfer::WeightTransfer;
use crate::stages::{Launch, Quantum, StageLaunch};
use crate::workspace::BufferSlot::{
    self, AttentionContext, AttentionQkNormalized, FfnDelta, FinalHiddenState, FinalNormalized, GateUp, GdnAlpha,
    GdnBeta, GdnConvolved, GdnNormalized, GdnRecurrent, HiddenState, InputNormalized, Logits, MixerDelta, MixerHidden,
    MtpNormed, MtpPacked, PostMixerNormalized, QkProjected, Swiglu as SwigluSlot, ValueZProjected,
};

/// The weights of a model, found by layer and role: what the plan reads, whether they sit on a device or are only
/// described.
pub trait WeightLookup {
    fn weight(&self, layer: LayerIndex, role: Role) -> Option<&WeightRef>;

    /// The host arena the weights held in host memory are staged from, if the lookup has one. A lookup of described
    /// weights has none: a plan over it builds, and a transfer of it fails when it runs.
    fn host_arena(&self) -> Option<Arc<HostWeightsArena>> {
        None
    }
}

impl WeightLookup for Weights {
    fn weight(&self, layer: LayerIndex, role: Role) -> Option<&WeightRef> {
        self.get(layer, role)
    }

    fn host_arena(&self) -> Option<Arc<HostWeightsArena>> {
        Weights::host_arena(self).cloned()
    }
}

/// The formats the plan's linears run through, by the format their weights are stored in.
#[derive(Clone)]
pub struct Formats {
    trellis: Arc<dyn LinearFormat>,
    nvfp4: Arc<dyn LinearFormat>,
    grouped: Option<Arc<dyn LinearFormat>>,
}

impl Formats {
    /// The trellis format for trellis weights and the NVFP4 format for NVFP4 weights.
    pub fn new(trellis: Arc<dyn LinearFormat>, nvfp4: Arc<dyn LinearFormat>) -> Self {
        Self {
            trellis,
            nvfp4,
            grouped: None,
        }
    }

    /// The same formats, with `grouped` for the Q3, Q4 and Q5 weights (and the P2E2 compressed Q3).
    pub fn with_grouped(mut self, grouped: Arc<dyn LinearFormat>) -> Self {
        self.grouped = Some(grouped);
        self
    }

    fn of(&self, weight: &WeightRef) -> Result<Arc<dyn LinearFormat>, PlanBuildError> {
        match weight.format {
            WeightFormat::Trellis => Ok(Arc::clone(&self.trellis)),
            WeightFormat::Nvfp4 => Ok(Arc::clone(&self.nvfp4)),
            // Only a Q3 tensor is stored compressed: a Q4 or Q5 one in the P2E2 layout is refused, here, by name.
            WeightFormat::Q3G64Fp16 | WeightFormat::Q4G64Fp16 | WeightFormat::Q5G64Fp16
                if weight.layout == WeightLayout::RowSplitK128V1
                    || (weight.format == WeightFormat::Q3G64Fp16 && weight.layout == WeightLayout::RowSplitP2e2V1) =>
            {
                self.grouped
                    .as_ref()
                    .map(Arc::clone)
                    .ok_or_else(|| PlanBuildError::Format {
                        name: weight.name.clone(),
                        format: weight.format,
                        layout: weight.layout,
                    })
            }
            format => Err(PlanBuildError::Format {
                name: weight.name.clone(),
                format,
                layout: weight.layout,
            }),
        }
    }
}

impl fmt::Debug for Formats {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Formats").finish_non_exhaustive()
    }
}

/// What a stage is, for the plan's lists and the names of its graphs' stages.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum StageKind {
    Embed,
    RmsNormUnitOffset,
    ResidualRmsNorm,
    ResidualAdd,
    GdnProjectControl,
    Linear,
    /// The gate/up projection and the SwiGLU that follows it, which a format may run as one launch.
    GateUpSwiglu,
    GdnConvolution,
    GdnRecurrence,
    GdnGatedRmsNorm,
    QkNormRope,
    KvAppend,
    AttentionCausal,
    Swiglu,
    MtpStem,
    Argmax,
    CopyRows,
    SelectedToken,
    /// The copy of a host-held weight into a staging slot.
    WeightTransfer,
}

impl StageKind {
    /// The name of the stage in a graph.
    pub fn name(self) -> &'static str {
        match self {
            StageKind::Embed => "embed",
            StageKind::RmsNormUnitOffset => "rms-norm",
            StageKind::ResidualRmsNorm => "residual-rms-norm",
            StageKind::ResidualAdd => "residual-add",
            StageKind::GdnProjectControl => "gdn-project-control",
            StageKind::Linear => "linear",
            StageKind::GateUpSwiglu => "gate-up-swiglu",
            StageKind::GdnConvolution => "gdn-convolution",
            StageKind::GdnRecurrence => "gdn-recurrence",
            StageKind::GdnGatedRmsNorm => "gdn-gated-rms-norm",
            StageKind::QkNormRope => "qk-norm-rope",
            StageKind::KvAppend => "kv-append",
            StageKind::AttentionCausal => "attention",
            StageKind::Swiglu => "swiglu",
            StageKind::MtpStem => "mtp-stem",
            StageKind::Argmax => "argmax",
            StageKind::CopyRows => "copy-rows",
            StageKind::SelectedToken => "selected-token",
            StageKind::WeightTransfer => "weight-transfer",
        }
    }
}

/// How to make one stage: its kind with the weights and slots that do not change from quantum to quantum.
#[derive(Clone, Debug)]
pub(crate) enum Make {
    Embed {
        table: WeightRef,
    },
    RmsNorm {
        input: BufferSlot,
        output: BufferSlot,
        weight: WeightRef,
        rows: NormRows,
    },
    ResidualRmsNorm {
        residual: BufferSlot,
        delta: BufferSlot,
        weight: WeightRef,
        hidden: BufferSlot,
        normalized: BufferSlot,
    },
    ResidualAdd {
        residual: BufferSlot,
        delta: BufferSlot,
        output: BufferSlot,
    },
    /// The A and B projections, the decay base and the time-step bias.
    GdnProjectControl {
        weights: Box<[WeightRef; 4]>,
    },
    Linear {
        weight: WeightRef,
        input: BufferSlot,
        output: BufferSlot,
        rows: LinearRows,
        /// The slot the SwiGLU of `output` goes to, when the linear is a gate/up projection that carries it: its
        /// format may run both as one launch (`ExecutionPlan`'s fused gate/up node).
        swiglu: Option<BufferSlot>,
    },
    GdnConvolution {
        layer: usize,
        weights: WeightRef,
    },
    GdnRecurrence {
        layer: usize,
    },
    GdnGatedRmsNorm {
        weight: WeightRef,
    },
    QkNormRope {
        query_norm: WeightRef,
        key_norm: WeightRef,
    },
    KvAppend {
        source: KvSource,
        layer: usize,
    },
    AttentionCausal {
        source: KvSource,
        layer: usize,
    },
    Swiglu,
    MtpStem {
        embedding_norm: WeightRef,
        hidden_norm: WeightRef,
    },
    Argmax {
        vocabulary: usize,
    },
    CopyRows,
    SelectedToken,
    /// The stage `inner`, which reads the staging slot `slot` in place of its weight.
    Staged {
        slot: usize,
        inner: Box<Make>,
    },
    /// The copy of `bytes` bytes at `offset` of the host arena into the staging slot `slot`, at `destination`.
    Transfer {
        name: String,
        source: Option<Arc<HostWeightsArena>>,
        offset: u64,
        bytes: u64,
        slot: usize,
        destination: DevicePtr,
    },
}

impl Make {
    pub(crate) fn kind(&self) -> StageKind {
        match self {
            Make::Embed { .. } => StageKind::Embed,
            Make::RmsNorm { .. } => StageKind::RmsNormUnitOffset,
            Make::ResidualRmsNorm { .. } => StageKind::ResidualRmsNorm,
            Make::ResidualAdd { .. } => StageKind::ResidualAdd,
            Make::GdnProjectControl { .. } => StageKind::GdnProjectControl,
            Make::Linear { swiglu: Some(_), .. } => StageKind::GateUpSwiglu,
            Make::Linear { .. } => StageKind::Linear,
            Make::GdnConvolution { .. } => StageKind::GdnConvolution,
            Make::GdnRecurrence { .. } => StageKind::GdnRecurrence,
            Make::GdnGatedRmsNorm { .. } => StageKind::GdnGatedRmsNorm,
            Make::QkNormRope { .. } => StageKind::QkNormRope,
            Make::KvAppend { .. } => StageKind::KvAppend,
            Make::AttentionCausal { .. } => StageKind::AttentionCausal,
            Make::Swiglu => StageKind::Swiglu,
            Make::MtpStem { .. } => StageKind::MtpStem,
            Make::Argmax { .. } => StageKind::Argmax,
            Make::CopyRows => StageKind::CopyRows,
            Make::SelectedToken => StageKind::SelectedToken,
            Make::Staged { inner, .. } => inner.kind(),
            Make::Transfer { .. } => StageKind::WeightTransfer,
        }
    }

    /// The weights the stage reads through a pointer of their own: those a transfer may stage.
    pub(crate) fn weights(&self) -> Vec<&WeightRef> {
        match self {
            Make::Embed { table } => vec![table],
            Make::RmsNorm { weight, .. }
            | Make::ResidualRmsNorm { weight, .. }
            | Make::Linear { weight, .. }
            | Make::GdnGatedRmsNorm { weight }
            | Make::GdnConvolution { weights: weight, .. } => vec![weight],
            Make::GdnProjectControl { weights } => weights.iter().collect(),
            Make::QkNormRope { query_norm, key_norm } => vec![query_norm, key_norm],
            Make::MtpStem {
                embedding_norm,
                hidden_norm,
            } => vec![embedding_norm, hidden_norm],
            Make::Staged { inner, .. } => inner.weights(),
            Make::ResidualAdd { .. }
            | Make::GdnRecurrence { .. }
            | Make::KvAppend { .. }
            | Make::AttentionCausal { .. }
            | Make::Swiglu
            | Make::Argmax { .. }
            | Make::CopyRows
            | Make::SelectedToken
            | Make::Transfer { .. } => Vec::new(),
        }
    }

    /// The workspace slots the stage reads and writes (`Instruction.inputBuffers` and `outputBuffers`).
    pub(crate) fn access(&self) -> Access {
        let (reads, writes): (Vec<BufferSlot>, Vec<BufferSlot>) = match self {
            Make::Embed { .. } => (vec![], vec![HiddenState]),
            Make::RmsNorm { input, output, .. } => (vec![*input], vec![*output]),
            Make::ResidualRmsNorm {
                residual,
                delta,
                hidden,
                normalized,
                ..
            } => (vec![*residual, *delta], vec![*hidden, *normalized]),
            Make::ResidualAdd {
                residual,
                delta,
                output,
            } => (vec![*residual, *delta], vec![*output]),
            Make::GdnProjectControl { .. } => (vec![InputNormalized], vec![GdnAlpha, GdnBeta]),
            Make::Linear {
                input, output, swiglu, ..
            } => (vec![*input], once(*output).chain(*swiglu).collect()),
            Make::GdnConvolution { .. } => (vec![QkProjected, ValueZProjected], vec![GdnConvolved]),
            Make::GdnRecurrence { .. } => (vec![GdnConvolved, GdnAlpha, GdnBeta], vec![GdnRecurrent]),
            Make::GdnGatedRmsNorm { .. } => (vec![GdnRecurrent, ValueZProjected], vec![GdnNormalized]),
            Make::QkNormRope { .. } => (vec![QkProjected], vec![AttentionQkNormalized]),
            Make::KvAppend { .. } => (vec![AttentionQkNormalized, ValueZProjected], vec![]),
            Make::AttentionCausal { .. } => (vec![AttentionQkNormalized, ValueZProjected], vec![AttentionContext]),
            Make::Swiglu => (vec![GateUp], vec![SwigluSlot]),
            Make::MtpStem { .. } => (vec![HiddenState], vec![MtpNormed, MtpPacked]),
            Make::Argmax { .. } => (vec![Logits], vec![]),
            Make::CopyRows => (vec![Logits], vec![]),
            Make::SelectedToken => (vec![], vec![]),
            Make::Staged { slot, inner } => {
                return Access {
                    slot_read: Some(*slot),
                    ..inner.access()
                };
            }
            Make::Transfer { slot, .. } => {
                return Access {
                    slot_write: Some(*slot),
                    ..Access::default()
                };
            }
        };
        Access {
            reads,
            writes,
            ..Access::default()
        }
    }

    /// The scratch region the stage takes in `variant`, if it declares a scratch use (`Shape.scratchUse` and
    /// `Shape.scratchRegion`): a trellis linear always, and a P2E2 linear (which expands its tensor wherever its fused
    /// decode does not run, as it does under exact numerics), any other quantized linear in the multi-row views; the
    /// value/z projection of a decode-type view has a region of its own, which it takes beside the query/key
    /// projection.
    pub(crate) fn scratch_region(&self, variant: Variant) -> Option<usize> {
        let Make::Linear { weight, output, .. } = self else {
            return match self {
                Make::Staged { inner, .. } => inner.scratch_region(variant),
                _ => None,
            };
        };
        if weight.format == WeightFormat::Trellis || weight.layout == WeightLayout::RowSplitP2e2V1 {
            Some(usize::from(variant.decode_type() && *output == ValueZProjected))
        } else {
            variant.multi_row().then_some(0)
        }
    }

    /// The stage, for `variant`.
    pub(crate) fn stage(&self, variant: Variant, formats: &Formats) -> Result<Entry, PlanBuildError> {
        let region = self.scratch_region(variant);
        let ptr = |weight: &WeightRef| weight.ptr;
        Ok(match self.clone() {
            Make::Embed { table } => {
                // A table the lookup cannot read is refused here, not at its first launch.
                TokenEmbedding::check(&table).map_err(PlanBuildError::Geometry)?;
                entry(self.kind(), Embed::new(table, HiddenState))
            }
            Make::RmsNorm {
                input,
                output,
                weight,
                rows,
            } => entry(self.kind(), RmsNormUnitOffset::new(input, output, ptr(&weight), rows)),
            Make::ResidualRmsNorm {
                residual,
                delta,
                weight,
                hidden,
                normalized,
            } => entry(
                self.kind(),
                ResidualRmsNorm::new(residual, delta, ptr(&weight), hidden, normalized),
            ),
            Make::ResidualAdd {
                residual,
                delta,
                output,
            } => entry(self.kind(), ResidualAdd::new(residual, delta, output)),
            Make::GdnProjectControl { weights } => entry(
                self.kind(),
                GdnProjectControl::new(
                    InputNormalized,
                    ptr(&weights[0]),
                    ptr(&weights[1]),
                    ptr(&weights[2]),
                    ptr(&weights[3]),
                    GdnAlpha,
                    GdnBeta,
                ),
            ),
            Make::Linear {
                weight,
                input,
                output,
                rows,
                swiglu,
            } => {
                let format = formats.of(&weight)?;
                let mut linear = Linear::new(format, weight, input, output, rows, region);
                if let Some(slot) = swiglu {
                    linear = linear.with_swiglu(slot);
                }
                entry(self.kind(), linear)
            }
            Make::GdnConvolution { layer, weights } => entry(
                self.kind(),
                GdnConvolution::new(layer, QkProjected, ValueZProjected, ptr(&weights), GdnConvolved),
            ),
            Make::GdnRecurrence { layer } => entry(
                self.kind(),
                GdnRecurrence::new(layer, GdnConvolved, GdnAlpha, GdnBeta, GdnRecurrent),
            ),
            Make::GdnGatedRmsNorm { weight } => entry(
                self.kind(),
                GdnGatedRmsNorm::new(GdnRecurrent, ValueZProjected, ptr(&weight), GdnNormalized),
            ),
            Make::QkNormRope { query_norm, key_norm } => entry(
                self.kind(),
                QkNormRope::new(QkProjected, AttentionQkNormalized, ptr(&query_norm), ptr(&key_norm)),
            ),
            Make::KvAppend { source, layer } => entry(
                self.kind(),
                KvAppend::new(source, layer, AttentionQkNormalized, ValueZProjected),
            ),
            Make::AttentionCausal { source, layer } => entry(
                self.kind(),
                AttentionCausal::new(source, layer, AttentionQkNormalized, ValueZProjected, AttentionContext),
            ),
            Make::Swiglu => entry(self.kind(), Swiglu::new(GateUp, SwigluSlot)),
            Make::MtpStem {
                embedding_norm,
                hidden_norm,
            } => entry(
                self.kind(),
                MtpStem::new(
                    HiddenState,
                    ptr(&embedding_norm),
                    ptr(&hidden_norm),
                    MtpNormed,
                    MtpPacked,
                ),
            ),
            Make::Argmax { vocabulary } => {
                let rows = if variant.selects_each() {
                    SelectedRows::Each
                } else {
                    SelectedRows::Last
                };
                entry(self.kind(), Argmax::new(Logits, vocabulary, rows))
            }
            Make::CopyRows => {
                let rows = if variant.selects_each() {
                    SelectedRows::Each
                } else {
                    SelectedRows::Last
                };
                entry(self.kind(), CopyRows::new(Logits, rows))
            }
            Make::SelectedToken => entry(self.kind(), SelectedToken),
            Make::Staged { inner, .. } => inner.stage(variant, formats)?,
            Make::Transfer {
                name,
                source,
                offset,
                bytes,
                destination,
                ..
            } => entry(
                self.kind(),
                WeightTransfer::new(name, source, offset, bytes, destination),
            ),
        })
    }
}

/// A stage of a view: the same object as the stage a graph runs and as the launches a capture records.
#[derive(Clone)]
pub struct Entry {
    pub kind: StageKind,
    pub stage: Arc<dyn Stage<Quantum>>,
    pub launch: Arc<dyn StageLaunch>,
}

impl fmt::Debug for Entry {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Entry")
            .field("kind", &self.kind)
            .finish_non_exhaustive()
    }
}

fn entry<S>(kind: StageKind, stage: S) -> Entry
where
    S: Launch + Stage<Quantum> + 'static,
{
    let stage = Arc::new(stage);
    Entry {
        kind,
        stage: Arc::clone(&stage) as Arc<dyn Stage<Quantum>>,
        launch: stage as Arc<dyn StageLaunch>,
    }
}

/// One stage of a view's list: how to make it, and the stages it depends on (earlier in the list).
#[derive(Clone, Debug)]
pub(crate) struct Node {
    pub make: Make,
    pub deps: Vec<usize>,
}

/// The nodes of a list as it is built.
struct Nodes<'w, W: WeightLookup + ?Sized> {
    weights: &'w W,
    list: Vec<Node>,
}

impl<W: WeightLookup + ?Sized> Nodes<'_, W> {
    fn weight(&self, layer: LayerIndex, role: Role) -> Result<WeightRef, PlanBuildError> {
        self.weights
            .weight(layer, role)
            .cloned()
            .ok_or(PlanBuildError::MissingWeight { layer, role })
    }

    fn push(&mut self, make: Make, deps: &[usize]) -> usize {
        self.list.push(Node {
            make,
            deps: deps.to_vec(),
        });
        self.list.len() - 1
    }

    fn linear(
        &mut self,
        layer: LayerIndex,
        role: Role,
        input: BufferSlot,
        output: BufferSlot,
        deps: &[usize],
    ) -> Result<usize, PlanBuildError> {
        let weight = self.weight(layer, role)?;
        Ok(self.push(
            Make::Linear {
                weight,
                input,
                output,
                rows: LinearRows::All,
                swiglu: None,
            },
            deps,
        ))
    }

    /// The gate/up projection of `layer` with the SwiGLU after it: one node whose stage runs the pair as one launch
    /// where the weight's format has one (`ExecutionPlan`'s fusion of a Q3 gate/up and its SwiGLU), and the two
    /// launches where it has none. Any other weight keeps its linear and its SwiGLU as two nodes.
    fn gate_up_swiglu(&mut self, layer: LayerIndex, deps: &[usize]) -> Result<usize, PlanBuildError> {
        let weight = self.weight(layer, Role::GateUp)?;
        let fused = weight.format == WeightFormat::Q3G64Fp16;
        let gate_up = self.push(
            Make::Linear {
                weight,
                input: PostMixerNormalized,
                output: GateUp,
                rows: LinearRows::All,
                swiglu: fused.then_some(SwigluSlot),
            },
            deps,
        );
        Ok(if fused {
            gate_up
        } else {
            self.push(Make::Swiglu, &[gate_up])
        })
    }

    fn norm_weight(&self, layer: LayerIndex, kind: NormKind) -> Result<WeightRef, PlanBuildError> {
        self.weight(layer, Role::Norm(kind))
    }
}

/// Where the logits come from and go: the final norm, the head, and the selection after it (and, for the base model, the
/// copy of a sampled quantum's row to the host).
fn head<W: WeightLookup + ?Sized>(
    nodes: &mut Nodes<'_, W>,
    after: usize,
    final_norm: WeightRef,
    head: WeightRef,
    vocabulary: usize,
    copies_rows: bool,
) -> usize {
    let norm = nodes.push(
        Make::RmsNorm {
            input: FinalHiddenState,
            output: FinalNormalized,
            weight: final_norm,
            rows: NormRows::Final,
        },
        &[after],
    );
    let logits = nodes.push(
        Make::Linear {
            weight: head,
            input: FinalNormalized,
            output: Logits,
            rows: LinearRows::Logits,
            swiglu: None,
        },
        &[norm],
    );
    let selected = nodes.push(Make::Argmax { vocabulary }, &[logits]);
    if copies_rows {
        // After the argmax, in its fused chain: a quantum of a sampling request copies its row here and its argmax is
        // idle, any other quantum's copy is.
        nodes.push(Make::CopyRows, &[selected])
    } else {
        selected
    }
}

/// The base model's views: embedding, every layer in the fused form, the final norm, the head and the selection.
pub(crate) fn base_nodes<W: WeightLookup + ?Sized>(
    config: &ModelConfig,
    weights: &W,
) -> Result<Vec<Node>, PlanBuildError> {
    let global = LayerIndex::GLOBAL;
    let layers = config.layer_kinds.len();
    let mut nodes = Nodes {
        weights,
        list: Vec::with_capacity(layers * 14 + 8),
    };
    let table = nodes.weight(global, Role::Embedding)?;
    let embedding = nodes.push(Make::Embed { table }, &[]);
    let first_norm = nodes.norm_weight(LayerIndex(0), NormKind::Input)?;
    // The node that produces the layer's normalized input, and the one that produces the layer's input rows.
    let mut normalized = nodes.push(
        Make::RmsNorm {
            input: HiddenState,
            output: InputNormalized,
            weight: first_norm,
            rows: NormRows::All,
        },
        &[embedding],
    );
    let mut previous = embedding;
    let mut attention_index = 0;
    for (index, kind) in config.layer_kinds.iter().enumerate() {
        let layer =
            LayerIndex(u16::try_from(index).map_err(|_| PlanBuildError::Geometry("more than 65535 layers".into()))?);
        let layer_input = if index == 0 { HiddenState } else { FinalHiddenState };
        let mixer_out = match kind {
            LayerKind::GatedDeltaNet => {
                let a = nodes.weight(layer, Role::GdnA)?;
                let b = nodes.weight(layer, Role::GdnB)?;
                let a_log = nodes.weight(layer, Role::ALog)?;
                let dt_bias = nodes.weight(layer, Role::DtBias)?;
                let weights = Box::new([a, b, a_log, dt_bias]);
                let control = nodes.push(Make::GdnProjectControl { weights }, &[normalized]);
                let qk = nodes.linear(layer, Role::Qk, InputNormalized, QkProjected, &[normalized])?;
                let value_z = nodes.linear(layer, Role::ValueZ, InputNormalized, ValueZProjected, &[normalized])?;
                let convolution_weights = nodes.weight(layer, Role::Convolution)?;
                let convolution = nodes.push(
                    Make::GdnConvolution {
                        layer: index,
                        weights: convolution_weights,
                    },
                    &[qk, value_z],
                );
                let recurrence = nodes.push(Make::GdnRecurrence { layer: index }, &[control, convolution]);
                let norm = nodes.norm_weight(layer, NormKind::Gdn)?;
                let gated = nodes.push(Make::GdnGatedRmsNorm { weight: norm }, &[recurrence, value_z]);
                nodes.linear(layer, Role::Out, GdnNormalized, MixerDelta, &[gated])?
            }
            LayerKind::FullAttention => {
                let qk = nodes.linear(layer, Role::QueryKey, InputNormalized, QkProjected, &[normalized])?;
                let gate_value =
                    nodes.linear(layer, Role::GateValue, InputNormalized, ValueZProjected, &[normalized])?;
                let query_norm = nodes.norm_weight(layer, NormKind::Query)?;
                let key_norm = nodes.norm_weight(layer, NormKind::Key)?;
                let rope = nodes.push(Make::QkNormRope { query_norm, key_norm }, &[qk]);
                let source = KvSource::Base;
                let append = nodes.push(
                    Make::KvAppend {
                        source,
                        layer: attention_index,
                    },
                    &[rope, gate_value],
                );
                let attend = nodes.push(
                    Make::AttentionCausal {
                        source,
                        layer: attention_index,
                    },
                    &[rope, gate_value, append],
                );
                attention_index += 1;
                nodes.linear(layer, Role::Out, AttentionContext, MixerDelta, &[attend])?
            }
        };
        // The mixer's residual add and the post-attention norm, in one launch.
        let post_norm = nodes.norm_weight(layer, NormKind::PostAttention)?;
        let mixer = nodes.push(
            Make::ResidualRmsNorm {
                residual: layer_input,
                delta: MixerDelta,
                weight: post_norm,
                hidden: MixerHidden,
                normalized: PostMixerNormalized,
            },
            &[previous, mixer_out],
        );
        let swiglu = nodes.gate_up_swiglu(layer, &[mixer])?;
        let down = nodes.linear(layer, Role::Down, SwigluSlot, FfnDelta, &[swiglu])?;
        if index + 1 < layers {
            // The layer's residual add and the next layer's input norm, in one launch.
            let next_norm = nodes.norm_weight(LayerIndex(layer.0 + 1), NormKind::Input)?;
            let fused = nodes.push(
                Make::ResidualRmsNorm {
                    residual: MixerHidden,
                    delta: FfnDelta,
                    weight: next_norm,
                    hidden: FinalHiddenState,
                    normalized: InputNormalized,
                },
                &[mixer, down],
            );
            normalized = fused;
            previous = fused;
        } else {
            previous = nodes.push(
                Make::ResidualAdd {
                    residual: MixerHidden,
                    delta: FfnDelta,
                    output: FinalHiddenState,
                },
                &[mixer, down],
            );
        }
    }
    let final_norm = nodes.norm_weight(global, NormKind::Final)?;
    let lm_head = nodes.weight(global, Role::LmHead)?;
    head(&mut nodes, previous, final_norm, lm_head, config.vocab_size, true);
    Ok(nodes.list)
}

/// The MTP layer's views (`mtpDraft`): the embedding of each row's token, the stem with the seed hidden rows, the
/// input projection, the MTP layer as one full-attention layer on its own cache (unfused), the MTP final norm, the
/// draft head and the selection, which passes the drafted token on.
pub(crate) fn mtp_nodes<W: WeightLookup + ?Sized>(
    config: &ModelConfig,
    weights: &W,
) -> Result<Vec<Node>, PlanBuildError> {
    let global = LayerIndex::GLOBAL;
    let mtp = LayerIndex(
        u16::try_from(config.layer_kinds.len())
            .map_err(|_| PlanBuildError::Geometry("more than 65535 layers".into()))?,
    );
    let mut nodes = Nodes {
        weights,
        list: Vec::with_capacity(24),
    };
    let table = nodes.weight(global, Role::Embedding)?;
    let embedding = nodes.push(Make::Embed { table }, &[]);
    let embedding_norm = nodes.weight(global, Role::Mtp(MtpRole::EmbeddingNorm))?;
    let hidden_norm = nodes.weight(global, Role::Mtp(MtpRole::HiddenNorm))?;
    let stem = nodes.push(
        Make::MtpStem {
            embedding_norm,
            hidden_norm,
        },
        &[embedding],
    );
    let projection = nodes.linear(
        global,
        Role::Mtp(MtpRole::InputProjection),
        MtpPacked,
        HiddenState,
        &[stem],
    )?;
    let input_norm = nodes.norm_weight(mtp, NormKind::Input)?;
    let normalized = nodes.push(
        Make::RmsNorm {
            input: HiddenState,
            output: InputNormalized,
            weight: input_norm,
            rows: NormRows::All,
        },
        &[projection],
    );
    let qk = nodes.linear(mtp, Role::QueryKey, InputNormalized, QkProjected, &[normalized])?;
    let gate_value = nodes.linear(mtp, Role::GateValue, InputNormalized, ValueZProjected, &[normalized])?;
    let query_norm = nodes.norm_weight(mtp, NormKind::Query)?;
    let key_norm = nodes.norm_weight(mtp, NormKind::Key)?;
    let rope = nodes.push(Make::QkNormRope { query_norm, key_norm }, &[qk]);
    let append = nodes.push(
        Make::KvAppend {
            source: KvSource::Mtp,
            layer: 0,
        },
        &[rope, gate_value],
    );
    let attend = nodes.push(
        Make::AttentionCausal {
            source: KvSource::Mtp,
            layer: 0,
        },
        &[rope, gate_value, append],
    );
    let out = nodes.linear(mtp, Role::Out, AttentionContext, MixerDelta, &[attend])?;
    let mixer = nodes.push(
        Make::ResidualAdd {
            residual: HiddenState,
            delta: MixerDelta,
            output: MixerHidden,
        },
        &[projection, out],
    );
    let post_norm = nodes.norm_weight(mtp, NormKind::PostAttention)?;
    let post = nodes.push(
        Make::RmsNorm {
            input: MixerHidden,
            output: PostMixerNormalized,
            weight: post_norm,
            rows: NormRows::All,
        },
        &[mixer],
    );
    let gate_up = nodes.linear(mtp, Role::GateUp, PostMixerNormalized, GateUp, &[post])?;
    let swiglu = nodes.push(Make::Swiglu, &[gate_up]);
    let down = nodes.linear(mtp, Role::Down, SwigluSlot, FfnDelta, &[swiglu])?;
    let finish = nodes.push(
        Make::ResidualAdd {
            residual: MixerHidden,
            delta: FfnDelta,
            output: FinalHiddenState,
        },
        &[mixer, down],
    );
    let final_norm = nodes.weight(global, Role::Mtp(MtpRole::FinalNorm))?;
    let draft_head = nodes.weight(global, Role::DraftHead)?;
    let vocabulary = usize::try_from(draft_head.shape.first().copied().unwrap_or(0))
        .map_err(|_| PlanBuildError::Geometry("the draft head has too many rows".into()))?;
    let selected = head(&mut nodes, finish, final_norm, draft_head, vocabulary, false);
    nodes.push(Make::SelectedToken, &[selected]);
    Ok(nodes.list)
}
