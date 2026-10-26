//! The Gated DeltaNet stages: the joint A/B projection with its control, the convolution over the sequence's
//! convolution state, the recurrence over its recurrent state, and the gated RMSNorm (`Stages.GdnProjectControl`,
//! `GdnConvolution`, `GdnRecurrence`, `GdnGatedNorm`).
//!
//! A verification checkpoints each layer's state and the rows it ran, so that the rows the verifier accepted can be
//! rebuilt: the state has run past them (ReplaySSM, `docs/speculation/MTP_CONTRACT.md`). The sequence's owner
//! leaves the number of accepted rows in the layer's `pending_replay_rows` once acceptance is known; the next
//! quantum's convolution stage of the layer restores the checkpoint and replays those rows, row-exact, before it
//! runs its own rows.

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{CudaError, DevicePtr, Submission, ops};

use super::{Quantum, QuantumKind, fatal, launch_u32, submit};
use crate::state::gdn::{GdnSpeculative, GdnState};
use crate::workspace::BufferSlot;

/// The GDN state of `layer`.
fn layer_state(quantum: &Quantum, layer: usize) -> Result<&GdnState, StageError> {
    quantum
        .seq
        .state()
        .gdn
        .layer(layer)
        .ok_or_else(|| fatal(format!("layer {layer} has no GDN state")))
}

/// The buffers a verification checkpoints into, sized for `rows`.
fn speculative(state: &GdnState, rows: usize) -> Result<&GdnSpeculative, StageError> {
    let buffers = state
        .speculative()
        .ok_or_else(|| fatal("the sequence's GDN state has no speculative buffers"))?;
    if rows as u64 > buffers.capacity {
        return Err(fatal(format!(
            "{rows} rows exceed the {} the speculative buffers hold",
            buffers.capacity
        )));
    }
    Ok(buffers)
}

/// The scale of the recurrence's output: one over the square root of the key head dimension.
fn output_scale(key_head_dim: usize) -> f32 {
    (1.0 / (key_head_dim as f64).sqrt()) as f32
}

/// The joint A/B projection and control of a layer: `alpha` and `beta` in FP32 from the normalized input
/// (`gdn_project_control_fp32`).
#[derive(Clone, Debug)]
pub struct GdnProjectControl {
    input: BufferSlot,
    a_weight: DevicePtr,
    b_weight: DevicePtr,
    a_log: DevicePtr,
    dt_bias: DevicePtr,
    alpha: BufferSlot,
    beta: BufferSlot,
}

impl GdnProjectControl {
    pub fn new(
        input: BufferSlot,
        a_weight: DevicePtr,
        b_weight: DevicePtr,
        a_log: DevicePtr,
        dt_bias: DevicePtr,
        alpha: BufferSlot,
        beta: BufferSlot,
    ) -> Self {
        Self {
            input,
            a_weight,
            b_weight,
            a_log,
            dt_bias,
            alpha,
            beta,
        }
    }
}

impl Stage<Quantum> for GdnProjectControl {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let rows = quantum.rows_u32()?;
        let width = launch_u32(quantum.config.hidden_size, "the hidden size")?;
        let heads = launch_u32(quantum.config.linear_num_value_heads, "the value heads")?;
        submit(quantum, cx, false, |submission| {
            ops::gdn_project_control_fp32(
                submission,
                quantum.slot(self.input),
                self.a_weight,
                self.b_weight,
                self.a_log,
                self.dt_bias,
                quantum.slot(self.alpha),
                quantum.slot(self.beta),
                rows,
                width,
                heads,
            )
            .map_err(fatal)
        })
    }
}

/// The convolution of layer `layer` over the sequence's convolution state. A verification checkpoints the state and
/// its input rows first; a verification before it that accepted only part of its rows leaves a replay, which this
/// stage runs before its own rows.
#[derive(Clone, Debug)]
pub struct GdnConvolution {
    layer: usize,
    query_key: BufferSlot,
    value_z: BufferSlot,
    weights: DevicePtr,
    output: BufferSlot,
}

impl GdnConvolution {
    /// The convolution of model layer `layer`, with the kernel at `weights`.
    pub fn new(
        layer: usize,
        query_key: BufferSlot,
        value_z: BufferSlot,
        weights: DevicePtr,
        output: BufferSlot,
    ) -> Self {
        Self {
            layer,
            query_key,
            value_z,
            weights,
            output,
        }
    }

    /// Restores the layer's state from the checkpoint of the last verification and runs the accepted rows through
    /// the same convolution and recurrence kernels, row-exact, so that the state is bit for bit what one-row decode
    /// of those rows leaves.
    fn replay(&self, quantum: &Quantum, submission: &Submission<'_>, state: &GdnState) -> Result<(), StageError> {
        let config = &quantum.config;
        let rows = state.pending_replay_rows() as usize;
        let buffers = speculative(state, rows)?;
        let (query_key_width, value_width, channels) = widths(quantum)?;
        let kernel = launch_u32(config.linear_conv_kernel_dim, "the convolution kernel")?;
        let rows = rows as u32;
        let key_heads = launch_u32(config.linear_num_key_heads, "the key heads")?;
        let value_heads = launch_u32(config.linear_num_value_heads, "the value heads")?;
        let key_head_dim = launch_u32(config.linear_key_head_dim, "the key head dimension")?;
        let value_head_dim = launch_u32(config.linear_value_head_dim, "the value head dimension")?;
        ops::select_row_exact(submission, true);
        let replayed = (|| {
            ops::copy_device_to_device(
                submission,
                state.convolution().ptr(),
                buffers.convolution_checkpoint.ptr(),
                state.convolution().bytes() as u64,
            )?;
            ops::gdn_convolution_bf16(
                submission,
                buffers.query_key_rows.ptr(),
                buffers.value_z_rows.ptr(),
                self.weights,
                state.convolution().ptr(),
                buffers.replay_convolved.ptr(),
                rows,
                query_key_width,
                value_width,
                channels,
                kernel,
            )?;
            ops::copy_device_to_device(
                submission,
                state.recurrent().ptr(),
                buffers.recurrent_checkpoint.ptr(),
                state.recurrent().bytes() as u64,
            )?;
            ops::gdn_recurrence_bf16(
                submission,
                buffers.replay_convolved.ptr(),
                buffers.alpha_rows.ptr(),
                buffers.beta_rows.ptr(),
                state.recurrent().ptr(),
                buffers.replay_output.ptr(),
                rows,
                key_heads,
                value_heads,
                key_head_dim,
                value_head_dim,
                output_scale(config.linear_key_head_dim),
            )?;
            Ok::<(), CudaError>(())
        })();
        // The rest of the stage runs as the quantum does.
        ops::select_row_exact(submission, quantum.mode.row_exact);
        replayed.map_err(fatal)?;
        state.set_pending_replay_rows(0).map_err(fatal)
    }
}

/// The widths of the GDN query/key rows, the value rows and the convolved channels, as launches take them.
fn widths(quantum: &Quantum) -> Result<(u32, u32, u32), StageError> {
    let config = &quantum.config;
    Ok((
        launch_u32(config.gdn_query_key_width(), "the query/key width")?,
        launch_u32(config.gdn_value_width(), "the value width")?,
        launch_u32(config.gdn_convolution_channels(), "the convolution channels")?,
    ))
}

impl Stage<Quantum> for GdnConvolution {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let config = &quantum.config;
        let state = layer_state(quantum, self.layer)?;
        let rows = quantum.rows_u32()?;
        let (query_key_width, value_width, channels) = widths(quantum)?;
        let kernel = launch_u32(config.linear_conv_kernel_dim, "the convolution kernel")?;
        let query_key = quantum.slot(self.query_key);
        let value_z = quantum.slot(self.value_z);
        submit(quantum, cx, false, |submission| {
            // A verification before this quantum accepted only part of its rows: rebuild this layer's state first.
            if state.pending_replay_rows() > 0 {
                self.replay(quantum, submission, state)?;
            }
            if quantum.kind == QuantumKind::Verify {
                let buffers = speculative(state, quantum.rows)?;
                let rows = quantum.rows as u64;
                ops::copy_device_to_device(
                    submission,
                    buffers.convolution_checkpoint.ptr(),
                    state.convolution().ptr(),
                    state.convolution().bytes() as u64,
                )
                .map_err(fatal)?;
                ops::copy_device_to_device(
                    submission,
                    buffers.query_key_rows.ptr(),
                    query_key,
                    rows * u64::from(query_key_width) * 2,
                )
                .map_err(fatal)?;
                ops::copy_device_to_device(
                    submission,
                    buffers.value_z_rows.ptr(),
                    value_z,
                    rows * 2 * u64::from(value_width) * 2,
                )
                .map_err(fatal)?;
            }
            ops::gdn_convolution_bf16(
                submission,
                query_key,
                value_z,
                self.weights,
                state.convolution().ptr(),
                quantum.slot(self.output),
                rows,
                query_key_width,
                value_width,
                channels,
                kernel,
            )
            .map_err(fatal)
        })
    }
}

/// The recurrence of layer `layer` over the sequence's recurrent state; a verification checkpoints it, and the
/// control rows it runs, first.
#[derive(Clone, Debug)]
pub struct GdnRecurrence {
    layer: usize,
    convolved: BufferSlot,
    alpha: BufferSlot,
    beta: BufferSlot,
    output: BufferSlot,
}

impl GdnRecurrence {
    pub fn new(layer: usize, convolved: BufferSlot, alpha: BufferSlot, beta: BufferSlot, output: BufferSlot) -> Self {
        Self {
            layer,
            convolved,
            alpha,
            beta,
            output,
        }
    }
}

impl Stage<Quantum> for GdnRecurrence {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let config = &quantum.config;
        let state = layer_state(quantum, self.layer)?;
        let rows = quantum.rows_u32()?;
        let key_heads = launch_u32(config.linear_num_key_heads, "the key heads")?;
        let value_heads = launch_u32(config.linear_num_value_heads, "the value heads")?;
        let key_head_dim = launch_u32(config.linear_key_head_dim, "the key head dimension")?;
        let value_head_dim = launch_u32(config.linear_value_head_dim, "the value head dimension")?;
        let alpha = quantum.slot(self.alpha);
        let beta = quantum.slot(self.beta);
        submit(quantum, cx, false, |submission| {
            if quantum.kind == QuantumKind::Verify {
                let buffers = speculative(state, quantum.rows)?;
                let control_bytes = quantum.rows as u64 * u64::from(value_heads) * 4;
                ops::copy_device_to_device(
                    submission,
                    buffers.recurrent_checkpoint.ptr(),
                    state.recurrent().ptr(),
                    state.recurrent().bytes() as u64,
                )
                .map_err(fatal)?;
                ops::copy_device_to_device(submission, buffers.alpha_rows.ptr(), alpha, control_bytes)
                    .map_err(fatal)?;
                ops::copy_device_to_device(submission, buffers.beta_rows.ptr(), beta, control_bytes).map_err(fatal)?;
            }
            ops::gdn_recurrence_bf16(
                submission,
                quantum.slot(self.convolved),
                alpha,
                beta,
                state.recurrent().ptr(),
                quantum.slot(self.output),
                rows,
                key_heads,
                value_heads,
                key_head_dim,
                value_head_dim,
                output_scale(config.linear_key_head_dim),
            )
            .map_err(fatal)
        })
    }
}

/// The gated RMSNorm of the recurrence's output by the z rows.
#[derive(Clone, Debug)]
pub struct GdnGatedRmsNorm {
    recurrent: BufferSlot,
    value_z: BufferSlot,
    weight: DevicePtr,
    output: BufferSlot,
}

impl GdnGatedRmsNorm {
    pub fn new(recurrent: BufferSlot, value_z: BufferSlot, weight: DevicePtr, output: BufferSlot) -> Self {
        Self {
            recurrent,
            value_z,
            weight,
            output,
        }
    }
}

impl Stage<Quantum> for GdnGatedRmsNorm {
    fn run(&self, quantum: &Quantum, cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let config = &quantum.config;
        let rows = quantum.rows_u32()?;
        let heads = launch_u32(config.linear_num_value_heads, "the value heads")?;
        let head_dim = launch_u32(config.linear_value_head_dim, "the value head dimension")?;
        submit(quantum, cx, false, |submission| {
            ops::gdn_gated_rms_norm_bf16(
                submission,
                quantum.slot(self.recurrent),
                quantum.slot(self.value_z),
                self.weight,
                quantum.slot(self.output),
                rows,
                heads,
                head_dim,
                quantum.epsilon(),
            )
            .map_err(fatal)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_recurrence_scale_is_the_float_of_the_double_root() {
        assert_eq!(output_scale(128), (1.0f64 / 128f64.sqrt()) as f32);
        assert_eq!(output_scale(128), 0.088_388_346);
    }
}
