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
use euhedral_inference_cuda::{CudaError, KernelOps};
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

/// The launches made so far in the warm-up's one submission.
struct Launcher<'a> {
    kernels: &'a mut dyn KernelOps,
    warmed: Warmed,
}

impl Launcher<'_> {
    fn launch(&mut self, module: &'static str, work: impl FnOnce(&mut dyn KernelOps) -> Result<(), CudaError>) {
        match work(&mut *self.kernels) {
            Ok(()) => self.warmed.modules.push(module),
            Err(error) => self.warmed.failed.push(format!("{module}: {error}")),
        }
    }
}

/// Warms the modules the engine's loaded model can launch with only its own buffers: every launch is made in one
/// submission on the first transfer lane, so the lane's ring holds one marker for all of them. Returns the marker the
/// lane reaches when they have all run (none if the lane refused the submission, which then warmed nothing).
pub(crate) fn run(
    core: &Core,
    config: &ModelConfig,
    weights: &Weights,
    workspace: &Workspace,
) -> (Warmed, Option<Marker>) {
    let mut warmed = Warmed::default();
    let submitted = core.lane_table.submit_ops(core.transfer_lane(0), false, |lane| {
        warmed = warm(core, config, weights, workspace, lane);
        Ok::<(), CudaError>(())
    });
    match submitted {
        Ok(((), marker)) => (warmed, Some(marker)),
        Err(error) => (
            Warmed {
                modules: Vec::new(),
                failed: vec![format!("warm-up: {error}")],
            },
            None,
        ),
    }
}

fn warm(
    core: &Core,
    config: &ModelConfig,
    weights: &Weights,
    workspace: &Workspace,
    kernels: &mut dyn KernelOps,
) -> Warmed {
    let mut launcher = Launcher {
        kernels,
        warmed: Warmed::default(),
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
        launcher.launch("zero", |kernels| {
            kernels.zero_device_memory(buffer.ptr(), buffer.bytes() as u64)
        });
    }

    if let Some(norm) = weights.get(LayerIndex::GLOBAL, Role::Norm(NormKind::Final)) {
        let epsilon = config.rms_norm_epsilon as f32;
        launcher.launch("norm", |kernels| {
            kernels.rms_norm_unit_offset_bf16(
                slot(BufferSlot::HiddenState),
                norm.ptr,
                slot(BufferSlot::InputNormalized),
                1,
                hidden,
                epsilon,
            )
        });
    }
    launcher.launch("elementwise", |kernels| {
        kernels.residual_add_bf16(
            slot(BufferSlot::HiddenState),
            slot(BufferSlot::MixerDelta),
            slot(BufferSlot::HiddenState),
            1,
            hidden,
        )?;
        kernels.swiglu_bf16(slot(BufferSlot::GateUp), slot(BufferSlot::Swiglu), 1, intermediate)
    });
    launcher.launch("sampling", |kernels| {
        kernels.argmax_bf16(
            slot(BufferSlot::Logits),
            config.vocab_size as u32,
            slot(BufferSlot::FinalNormalized),
        )
    });
    if let Some(embedding) = weights
        .get(LayerIndex::GLOBAL, Role::Embedding)
        .filter(|weight| weight.format == WeightFormat::Q3G64Fp16 && weight.layout == WeightLayout::RowSplitK128V1)
    {
        launcher.launch("embedding", |kernels| {
            kernels.embed_q3(
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
        launcher.launch("nvfp4", |kernels| {
            kernels.linear_nvfp4_bf16(
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
                launcher.launch(module, |kernels| {
                    kernels.linear_nvfp4_native_bf16(
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
    launcher.warmed
}

/// `(rows, columns)` of a two-dimensional tensor, as the launches take them.
fn matrix(weight: &WeightRef) -> Option<(u32, u32)> {
    match weight.shape.as_slice() {
        [out, input] => Some((u32::try_from(*out).ok()?, u32::try_from(*input).ok()?)),
        _ => None,
    }
}
