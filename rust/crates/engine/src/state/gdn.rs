//! The Gated DeltaNet state of a sequence: per GDN layer the convolution and recurrent buffers, and the buffers a
//! speculative verification checkpoints and replays from (`GdnState`, `GdnStates`; docs/speculation/MTP_CONTRACT.md,
//! ReplaySSM).

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use euhedral_inference_cuda::{CudaError, CudaLibrary, DeviceBuffer, Submission, ops};

use crate::model::{LayerKind, ModelConfig};
use crate::residency::footprint::MINIMUM_VERIFY_ROWS;

/// The state before the latest verification, its recorded inputs, and the replay scratch, for up to `capacity`
/// verified rows.
#[derive(Debug)]
pub struct GdnSpeculative {
    pub capacity: u64,
    pub recurrent_checkpoint: DeviceBuffer,
    pub convolution_checkpoint: DeviceBuffer,
    /// BF16 query and key rows of the verified rows.
    pub query_key_rows: DeviceBuffer,
    /// BF16 value and z rows.
    pub value_z_rows: DeviceBuffer,
    /// FP32 alpha rows, one value per value head.
    pub alpha_rows: DeviceBuffer,
    /// FP32 beta rows.
    pub beta_rows: DeviceBuffer,
    /// BF16 convolved rows the replay recomputes.
    pub replay_convolved: DeviceBuffer,
    /// BF16 recurrent outputs the replay recomputes.
    pub replay_output: DeviceBuffer,
}

/// One GDN layer's persistent state.
#[derive(Debug)]
pub struct GdnState {
    convolution: DeviceBuffer,
    recurrent: DeviceBuffer,
    speculative: Option<GdnSpeculative>,
    /// Verified rows the next quantum must replay from the checkpoint before touching this state.
    /// An atomic: the stages of a quantum read and clear it through a shared reference, and the sequence's owner
    /// sets it between quanta.
    pending_replay_rows: AtomicU32,
}

impl GdnState {
    /// The convolution state: the last `kernel - 1` rows of the convolved channels, BF16.
    pub fn convolution(&self) -> &DeviceBuffer {
        &self.convolution
    }

    /// The recurrent state: `value heads x key dim x value dim` FP32.
    pub fn recurrent(&self) -> &DeviceBuffer {
        &self.recurrent
    }

    /// The speculative buffers, when the sequence verifies.
    pub fn speculative(&self) -> Option<&GdnSpeculative> {
        self.speculative.as_ref()
    }

    /// Verified rows the next quantum must replay.
    pub fn pending_replay_rows(&self) -> u32 {
        self.pending_replay_rows.load(Ordering::Acquire)
    }

    /// After a verification committed `rows` of its rows: replay them from the checkpoint before the state is next
    /// used, or nothing when every row was committed (0).
    pub fn set_pending_replay_rows(&self, rows: u32) -> Result<(), &'static str> {
        if rows > 0 && self.speculative.is_none() {
            return Err("a state with no speculative buffers has nothing to replay from");
        }
        self.pending_replay_rows.store(rows, Ordering::Release);
        Ok(())
    }

    /// The device bytes this layer holds.
    pub fn device_bytes(&self) -> u64 {
        let speculative = self.speculative.as_ref().map_or(0, |buffers| {
            [
                &buffers.recurrent_checkpoint,
                &buffers.convolution_checkpoint,
                &buffers.query_key_rows,
                &buffers.value_z_rows,
                &buffers.alpha_rows,
                &buffers.beta_rows,
                &buffers.replay_convolved,
                &buffers.replay_output,
            ]
            .iter()
            .map(|buffer| buffer.bytes() as u64)
            .sum()
        });
        (self.convolution.bytes() + self.recurrent.bytes()) as u64 + speculative
    }
}

/// The GDN state of every GDN layer of one sequence.
#[derive(Debug)]
pub struct GdnStates {
    layers: Vec<Option<GdnState>>,
}

impl GdnStates {
    /// Allocates and zeroes the state of every GDN layer, with speculative buffers for `verified_rows` rows (at
    /// least [`MINIMUM_VERIFY_ROWS`], as `GdnState.speculative` allocates); 0 allocates none. The allocations wait
    /// (startup only); the zeroing is queued on `submission`'s lane, and the caller waits for its marker.
    pub fn allocate(
        library: &Arc<CudaLibrary>,
        config: &ModelConfig,
        verified_rows: u64,
        submission: &Submission<'_>,
    ) -> Result<GdnStates, CudaError> {
        let channels = config.gdn_convolution_channels() as u64;
        let heads = config.linear_num_value_heads as u64;
        let convolution_bytes = channels * (config.linear_conv_kernel_dim as u64 - 1) * 2;
        let recurrent_bytes = heads * config.linear_key_head_dim as u64 * config.linear_value_head_dim as u64 * 4;
        let (query_key, value) = (config.gdn_query_key_width() as u64, config.gdn_value_width() as u64);
        let capacity = verified_rows.max(MINIMUM_VERIFY_ROWS);
        let alloc = |bytes: u64| DeviceBuffer::alloc(library, bytes as usize);
        let mut layers = Vec::with_capacity(config.layer_kinds.len());
        for kind in &config.layer_kinds {
            if *kind != LayerKind::GatedDeltaNet {
                layers.push(None);
                continue;
            }
            let convolution = alloc(convolution_bytes)?;
            let recurrent = alloc(recurrent_bytes)?;
            ops::zero_device_memory(submission, convolution.ptr(), convolution_bytes)?;
            ops::zero_device_memory(submission, recurrent.ptr(), recurrent_bytes)?;
            let speculative = if verified_rows == 0 {
                None
            } else {
                Some(GdnSpeculative {
                    capacity,
                    recurrent_checkpoint: alloc(recurrent_bytes)?,
                    convolution_checkpoint: alloc(convolution_bytes)?,
                    query_key_rows: alloc(capacity * query_key * 2)?,
                    value_z_rows: alloc(capacity * 2 * value * 2)?,
                    alpha_rows: alloc(capacity * heads * 4)?,
                    beta_rows: alloc(capacity * heads * 4)?,
                    replay_convolved: alloc(capacity * channels * 2)?,
                    replay_output: alloc(capacity * value * 2)?,
                })
            };
            layers.push(Some(GdnState {
                convolution,
                recurrent,
                speculative,
                pending_replay_rows: AtomicU32::new(0),
            }));
        }
        Ok(GdnStates { layers })
    }

    /// The state of layer `layer`, if it is a GDN layer.
    pub fn layer(&self, layer: usize) -> Option<&GdnState> {
        self.layers.get(layer)?.as_ref()
    }

    /// The state of layer `layer`, mutably, if it is a GDN layer.
    pub fn layer_mut(&mut self, layer: usize) -> Option<&mut GdnState> {
        self.layers.get_mut(layer)?.as_mut()
    }

    /// The device bytes every layer holds.
    pub fn device_bytes(&self) -> u64 {
        self.layers.iter().flatten().map(GdnState::device_bytes).sum()
    }
}
