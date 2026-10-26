//! Kernel module warm-up: one tiny launch per module, so that the first launch of a module (an NVRTC compile of
//! its source and a module load, which can take a second) happens at startup and not inside a request.
//!
//! The launches are real ones over the engine's own buffers and weights, one row each, with zeroed inputs. They run
//! in one frame of the stages lake, on the first transfer lane, which is idle by then. The compile happens inside
//! the launch call, in the C library, and has no asynchronous form: it blocks the worker that launches, once, at
//! startup. Nothing waits for the launches to run: the startup continues when the lane reaches the last launch's
//! marker (`Completions::after_marker`). A launch that fails does not stop the startup: the
//! module is listed in the report as not warmed, and compiles on its first real use.
//!
//! The modules of the elementwise, norm, sampling, embedding and NVFP4 sources are warmed here. The trellis,
//! attention and Gated DeltaNet launches take shapes, tables and state that only a stage of the decode graph has; the
//! stages that own those launches warm them.

use euhedral_graph::Marker;
use euhedral_inference_cuda::{CudaError, CudaLane, Submission, SubmitMode, ops};
use euhedral_inference_edrl::{WeightFormat, WeightLayout};

use crate::model::{LayerIndex, ModelConfig, NormKind, Role, WeightRef, Weights};
use crate::runtime::Core;
use crate::workspace::{BufferSlot, Workspace};

/// Which modules were warmed.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Warmed {
    /// Modules whose launch ran.
    pub modules: Vec<&'static str>,
    /// Modules whose launch was refused, with the reason; they compile on first use.
    pub failed: Vec<String>,
}

/// The launches queued so far on the warm-up lane.
struct Launcher<'a> {
    lane: &'a CudaLane,
    warmed: Warmed,
    /// The marker of the last launch queued: the lane reaches it when every launch has run.
    last: Option<Marker>,
}

impl Launcher<'_> {
    fn launch(&mut self, module: &'static str, work: impl FnOnce(&Submission<'_>) -> Result<(), CudaError>) {
        let result = Submission::begin(self.lane, SubmitMode::default()).and_then(|submission| {
            work(&submission)?;
            submission.finish_signalled()
        });
        match result {
            Ok(marker) => {
                self.warmed.modules.push(module);
                self.last = Some(marker);
            }
            Err(error) => self.warmed.failed.push(format!("{module}: {error}")),
        }
    }
}

/// Warms the modules the engine's loaded model can launch with only its own buffers.
pub(crate) fn run(
    core: &Core,
    config: &ModelConfig,
    weights: &Weights,
    workspace: &Workspace,
) -> (Warmed, Option<Marker>) {
    let lane = &core.transfer[0];
    let mut launcher = Launcher {
        lane,
        warmed: Warmed::default(),
        last: None,
    };
    let hidden = config.hidden_size as u32;
    let intermediate = config.intermediate_size as u32;
    let slot = |slot: BufferSlot| workspace.buffer(slot).ptr();

    // Defined inputs: the slots the launches read are zeros.
    for input in [
        BufferSlot::HiddenState,
        BufferSlot::MixerDelta,
        BufferSlot::GateUp,
        BufferSlot::Swiglu,
        BufferSlot::InputNormalized,
        BufferSlot::Logits,
        BufferSlot::TokenIds,
    ] {
        let buffer = workspace.buffer(input);
        launcher.launch("zero", |submission| {
            ops::zero_device_memory(submission, buffer.ptr(), buffer.bytes() as u64)
        });
    }

    if let Some(norm) = weights.get(LayerIndex::GLOBAL, Role::Norm(NormKind::Final)) {
        let epsilon = config.rms_norm_epsilon as f32;
        launcher.launch("norm", |submission| {
            ops::rms_norm_unit_offset_bf16(
                submission,
                slot(BufferSlot::HiddenState),
                norm.ptr,
                slot(BufferSlot::InputNormalized),
                1,
                hidden,
                epsilon,
            )
        });
    }
    launcher.launch("elementwise", |submission| {
        ops::residual_add_bf16(
            submission,
            slot(BufferSlot::HiddenState),
            slot(BufferSlot::MixerDelta),
            slot(BufferSlot::HiddenState),
            1,
            hidden,
        )?;
        ops::swiglu_bf16(
            submission,
            slot(BufferSlot::GateUp),
            slot(BufferSlot::Swiglu),
            1,
            intermediate,
        )
    });
    launcher.launch("sampling", |submission| {
        ops::argmax_bf16(
            submission,
            slot(BufferSlot::Logits),
            config.vocab_size as u32,
            slot(BufferSlot::FinalNormalized),
        )
    });
    if let Some(embedding) = weights
        .get(LayerIndex::GLOBAL, Role::Embedding)
        .filter(|weight| weight.format == WeightFormat::Q3G64Fp16 && weight.layout == WeightLayout::RowSplitK128V1)
    {
        launcher.launch("embedding", |submission| {
            ops::embed_q3(
                submission,
                slot(BufferSlot::TokenIds),
                embedding.ptr,
                slot(BufferSlot::HiddenState),
                1,
                config.vocab_size as u32,
                hidden,
                embedding.bytes,
            )
        });
    }
    let head = weights
        .get(LayerIndex::GLOBAL, Role::DraftHead)
        .filter(|weight| weight.format == WeightFormat::Nvfp4);
    if let Some(head) = head
        && let Some((out, input)) = matrix(head)
    {
        launcher.launch("nvfp4", |submission| {
            ops::linear_nvfp4_bf16(
                submission,
                slot(BufferSlot::InputNormalized),
                head.ptr,
                slot(BufferSlot::Logits),
                1,
                input,
                out,
                head.bytes,
            )
        });
    }
    // The native route: a skinny launch (2 to 64 rows) and a tile launch (more), on an NVFP4 down projection that the
    // workspace's scratch was sized for.
    let native = weights
        .get(LayerIndex(config.num_hidden_layers as u16), Role::Down)
        .filter(|weight| weight.format == WeightFormat::Nvfp4);
    if let (Some(down), true, Some(scratch)) = (native, core.library.nvfp4_native_available(), workspace.scratch(0))
        && let Some((out, input)) = matrix(down)
    {
        {
            for (module, rows) in [("nvfp4_native", 64u32), ("nvfp4_native_tile", 128)] {
                if u64::from(rows) > workspace.rows().rows {
                    continue;
                }
                let needed = core.library.nvfp4_native_scratch_bytes(rows, input, out);
                if needed > scratch.bytes() as u64 {
                    continue;
                }
                launcher.launch(module, |submission| {
                    ops::linear_nvfp4_native_bf16(
                        submission,
                        slot(BufferSlot::Swiglu),
                        down.ptr,
                        slot(BufferSlot::FfnDelta),
                        scratch.ptr(),
                        rows,
                        input,
                        out,
                        down.bytes,
                        scratch.bytes() as u64,
                    )
                });
            }
        }
    }
    (launcher.warmed, launcher.last)
}

/// `(rows, columns)` of a two-dimensional tensor, as the launches take them.
fn matrix(weight: &WeightRef) -> Option<(u32, u32)> {
    match weight.shape.as_slice() {
        [out, input] => Some((u32::try_from(*out).ok()?, u32::try_from(*input).ok()?)),
        _ => None,
    }
}
