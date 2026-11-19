//! A layer's weights: read from the checkpoint into host memory ([`stage`]), then copied to the device on a lane
//! ([`LayerWeights::upload`]).

use std::mem::{replace, take};
use std::sync::Arc;

use euhedral_inference_cuda::{DeviceBuffer, DevicePtr, LaneOps, PinnedPool, PooledPinned};

use super::ForwardError;
use crate::checkpoint::objects::{MTP_OBJECTS, Qwen38Objects};
use crate::checkpoint::safetensors::ReadRange;
use crate::checkpoint::{CheckpointError, Done};

const MTP_PREFIX: &str = "mtp/layer/";

/// What to read: a decoder layer or the MTP block.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Target {
    /// Decoder layer `n`.
    Layer(usize),
    /// The MTP block: a full-attention layer with one fused query/key/gate/value object.
    Mtp,
}

/// One tensor of a layer: a role the kernels know, or a quantized object by name.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Part {
    InputNorm,
    PostNorm,
    QueryNorm,
    KeyNorm,
    /// GDN's `a` projection.
    AProjection,
    /// GDN's `b` projection.
    BProjection,
    /// GDN's gated norm.
    GdnNorm,
    /// GDN's `A_log`, widened to FP32.
    ALog,
    /// GDN's `dt_bias`, widened to FP32.
    DtBias,
    /// GDN's convolution, as `[taps, channels]`.
    Convolution,
    /// A quantized object (`text/layers/N/mlp/down`, ...): BF16 `[rows, columns]`.
    Object(String),
}

#[derive(Debug)]
struct Entry {
    part: Part,
    bytes: Vec<u8>,
    shape: Option<(u64, u64)>,
}

/// A layer read into host memory, not yet on the device.
#[derive(Debug)]
pub struct StagedLayer {
    prefix: String,
    attention: bool,
    entries: Vec<Entry>,
}

impl StagedLayer {
    /// The object names' common prefix (`text/layers/N/`, or `mtp/layer/`).
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Whether the layer is a full-attention layer.
    pub fn attention(&self) -> bool {
        self.attention
    }

    /// The bytes of `part` as they are uploaded.
    pub fn bytes(&self, part: &Part) -> Option<&[u8]> {
        self.entries
            .iter()
            .find(|entry| &entry.part == part)
            .map(|entry| entry.bytes.as_slice())
    }

    /// Every part with its bytes, in upload order.
    pub fn parts(&self) -> impl Iterator<Item = (&Part, &[u8])> {
        self.entries.iter().map(|entry| (&entry.part, entry.bytes.as_slice()))
    }
}

/// How the bytes of a part are made from what was read.
#[derive(Clone, Copy)]
enum Post {
    Plain,
    Floats,
    Convolution { channels: usize, taps: usize },
}

struct Plan {
    part: Part,
    first: usize,
    count: usize,
    post: Post,
    shape: Option<(u64, u64)>,
}

/// A BF16 vector as FP32 little-endian bytes.
pub fn widen_bf16(bytes: &[u8]) -> Vec<u8> {
    let (pairs, _) = bytes.as_chunks::<2>();
    pairs
        .iter()
        .flat_map(|pair| (u32::from(u16::from_le_bytes(*pair)) << 16).to_le_bytes())
        .collect()
}

/// A convolution `[channels, 1, taps]` of 16-bit values as `[taps, channels]`.
pub fn transpose_convolution(bytes: &[u8], channels: usize, taps: usize) -> Result<Vec<u8>, ForwardError> {
    if channels.checked_mul(taps).and_then(|count| count.checked_mul(2)) != Some(bytes.len()) {
        return Err(ForwardError::Invalid(format!(
            "{} bytes are not {channels} channels of {taps} taps",
            bytes.len()
        )));
    }
    let mut out = vec![0u8; bytes.len()];
    for channel in 0..channels {
        for tap in 0..taps {
            let from = 2 * (channel * taps + tap);
            let to = 2 * (tap * channels + channel);
            out[to..to + 2].copy_from_slice(&bytes[from..from + 2]);
        }
    }
    Ok(out)
}

/// Reads `target` from `objects` into host memory with one set of reads; `then` gets the layer from the frame that
/// ends the last read. `Err` started nothing.
pub fn stage(objects: &Qwen38Objects, target: Target, then: Done<StagedLayer>) -> Result<(), CheckpointError> {
    let store = objects.store();
    let mut ranges: Vec<ReadRange<Vec<u8>>> = Vec::new();
    let mut plans: Vec<Plan> = Vec::new();
    let (prefix, attention, names, tensors, gdn_prefix) = match target {
        Target::Layer(index) => {
            let attention = *objects
                .config
                .full_attention
                .get(index)
                .ok_or_else(|| CheckpointError::Invalid(format!("the model has no layer {index}")))?;
            (
                format!("text/layers/{index}/"),
                attention,
                objects.layer_objects(index)?,
                format!("model.language_model.layers.{index}."),
                (!attention).then_some("linear_attn."),
            )
        }
        Target::Mtp => (
            MTP_PREFIX.to_owned(),
            true,
            MTP_OBJECTS
                .iter()
                .filter(|name| name.starts_with(MTP_PREFIX))
                .map(|name| (*name).to_owned())
                .collect(),
            "mtp.layers.0.".to_owned(),
            None,
        ),
    };
    for name in names {
        let source = objects.object(&name)?;
        let too_large = || CheckpointError::Invalid(format!("{name}: the object is too large"));
        let row_bytes = source.columns.checked_mul(2).ok_or_else(too_large)?;
        let first = ranges.len();
        for piece in &source.pieces {
            let bytes = piece.rows.checked_mul(row_bytes).ok_or_else(too_large)?;
            ranges.push(ReadRange {
                tensor: piece.tensor.clone(),
                skip: piece.first_row.checked_mul(row_bytes).ok_or_else(too_large)?,
                memory: vec![0u8; usize::try_from(bytes).map_err(|_| too_large())?],
            });
        }
        plans.push(Plan {
            part: Part::Object(name),
            first,
            count: source.pieces.len(),
            post: Post::Plain,
            shape: Some((source.rows, source.columns)),
        });
    }
    let mut tensor = |part: Part, name: String, post: Post| -> Result<(), CheckpointError> {
        let bytes = usize::try_from(store.tensor(&name)?.bytes)
            .map_err(|_| CheckpointError::Invalid(format!("{name}: the tensor is too large")))?;
        plans.push(Plan {
            part,
            first: ranges.len(),
            count: 1,
            post,
            shape: None,
        });
        ranges.push(ReadRange {
            tensor: name,
            skip: 0,
            memory: vec![0u8; bytes],
        });
        Ok(())
    };
    tensor(Part::InputNorm, format!("{tensors}input_layernorm.weight"), Post::Plain)?;
    tensor(
        Part::PostNorm,
        format!("{tensors}post_attention_layernorm.weight"),
        Post::Plain,
    )?;
    match gdn_prefix {
        None => {
            tensor(
                Part::QueryNorm,
                format!("{tensors}self_attn.q_norm.weight"),
                Post::Plain,
            )?;
            tensor(Part::KeyNorm, format!("{tensors}self_attn.k_norm.weight"), Post::Plain)?;
        }
        Some(gdn) => {
            tensor(
                Part::AProjection,
                format!("{tensors}{gdn}in_proj_a.weight"),
                Post::Plain,
            )?;
            tensor(
                Part::BProjection,
                format!("{tensors}{gdn}in_proj_b.weight"),
                Post::Plain,
            )?;
            tensor(Part::GdnNorm, format!("{tensors}{gdn}norm.weight"), Post::Plain)?;
            tensor(Part::ALog, format!("{tensors}{gdn}A_log"), Post::Floats)?;
            tensor(Part::DtBias, format!("{tensors}{gdn}dt_bias"), Post::Floats)?;
            let name = format!("{tensors}{gdn}conv1d.weight");
            let shape = store.tensor(&name)?.shape.clone();
            let [channels, _, taps] = shape[..] else {
                return Err(CheckpointError::Invalid(format!(
                    "{name}: a convolution has three dimensions"
                )));
            };
            let (channels, taps) = (
                usize::try_from(channels)
                    .map_err(|_| CheckpointError::Invalid(format!("{name}: too many channels")))?,
                usize::try_from(taps).map_err(|_| CheckpointError::Invalid(format!("{name}: too many taps")))?,
            );
            tensor(Part::Convolution, name, Post::Convolution { channels, taps })?;
        }
    }
    store.read_ranges(
        ranges,
        Arc::new(move |result| {
            let assembled = result.and_then(|mut memories| {
                let mut entries = Vec::with_capacity(plans.len());
                for plan in &plans {
                    let pieces = &mut memories[plan.first..plan.first + plan.count];
                    let joined = if let [only] = pieces {
                        take(only)
                    } else {
                        pieces.iter().flat_map(|piece| piece.iter().copied()).collect()
                    };
                    let bytes = match plan.post {
                        Post::Plain => joined,
                        Post::Floats => widen_bf16(&joined),
                        Post::Convolution { channels, taps } => transpose_convolution(&joined, channels, taps)
                            .map_err(|error| CheckpointError::Invalid(error.to_string()))?,
                    };
                    entries.push(Entry {
                        part: plan.part.clone(),
                        bytes,
                        shape: plan.shape,
                    });
                }
                Ok(StagedLayer {
                    prefix: prefix.clone(),
                    attention,
                    entries,
                })
            });
            then(assembled);
        }),
    )
}

/// A layer's weights on the device.
#[derive(Debug)]
pub struct LayerWeights {
    prefix: String,
    attention: bool,
    buffers: Vec<(Part, DeviceBuffer)>,
    shapes: Vec<(String, (u64, u64))>,
}

impl LayerWeights {
    /// Allocates the layer's buffers in stream order on `lane` and queues their uploads, `chunk_bytes` at a time
    /// through pinned memory from `pool`. The pinned chunks go to `retained`: keep them until the lane has retired
    /// the work, then drop them. On `Err` the buffers already made are freed in stream order.
    pub fn upload(
        lane: &mut dyn LaneOps,
        pool: &Arc<PinnedPool>,
        staged: &StagedLayer,
        chunk_bytes: usize,
        retained: &mut Vec<PooledPinned>,
    ) -> Result<LayerWeights, ForwardError> {
        if chunk_bytes == 0 {
            return Err(ForwardError::Invalid("an upload chunk of 0 bytes".to_owned()));
        }
        let mut weights = LayerWeights {
            prefix: staged.prefix.clone(),
            attention: staged.attention,
            buffers: Vec::with_capacity(staged.entries.len()),
            shapes: Vec::new(),
        };
        let outcome = (|| {
            for entry in &staged.entries {
                let buffer = lane.alloc_async(entry.bytes.len().max(1))?;
                let destination = buffer.ptr();
                weights.buffers.push((entry.part.clone(), buffer));
                if let (Part::Object(name), Some(shape)) = (&entry.part, entry.shape) {
                    weights.shapes.push((name.clone(), shape));
                }
                let mut offset = 0;
                for chunk in entry.bytes.chunks(chunk_bytes) {
                    let mut pinned = pool.acquire(chunk.len())?;
                    pinned.buffer_mut().as_mut_slice()[..chunk.len()].copy_from_slice(chunk);
                    // SAFETY: the pinned chunk is kept in `retained`, unmodified, until the caller has seen the lane
                    // retire this copy.
                    unsafe {
                        lane.copy_upload_to_device(destination.offset(offset as u64), pinned.buffer(), 0, chunk.len())
                    }?;
                    retained.push(pinned);
                    offset += chunk.len();
                }
            }
            Ok::<(), ForwardError>(())
        })();
        match outcome {
            Ok(()) => Ok(weights),
            Err(error) => {
                // The upload failed already; a failure to free adds nothing the caller can act on.
                let _ = weights.release(lane);
                Err(error)
            }
        }
    }

    /// The object names' common prefix.
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Whether the layer is a full-attention layer.
    pub fn attention(&self) -> bool {
        self.attention
    }

    /// The device address of `part`.
    pub fn ptr(&self, part: &Part) -> Option<DevicePtr> {
        self.buffers
            .iter()
            .find(|(held, _)| held == part)
            .map(|(_, buffer)| buffer.ptr())
    }

    /// The device address of the quantized object `name`.
    pub fn object(&self, name: &str) -> Option<DevicePtr> {
        self.buffers
            .iter()
            .find(|(held, _)| matches!(held, Part::Object(object) if object == name))
            .map(|(_, buffer)| buffer.ptr())
    }

    /// The quantized objects with their `[rows, columns]`, in upload order.
    pub fn shapes(&self) -> &[(String, (u64, u64))] {
        &self.shapes
    }

    /// Replaces object `name` with `decoded` (BF16, the same size), which this layer then owns; the old buffer is
    /// freed in stream order on `lane`.
    pub fn replace(&mut self, lane: &mut dyn LaneOps, name: &str, decoded: DeviceBuffer) -> Result<(), ForwardError> {
        let slot = self
            .buffers
            .iter_mut()
            .find(|(held, _)| matches!(held, Part::Object(object) if object == name))
            .ok_or_else(|| ForwardError::Invalid(format!("the layer has no object {name}")))?;
        if slot.1.bytes() != decoded.bytes() {
            return Err(ForwardError::Invalid(format!(
                "{name} holds {} bytes, its replacement {}",
                slot.1.bytes(),
                decoded.bytes()
            )));
        }
        let old = replace(&mut slot.1, decoded);
        lane.free_async(old)?;
        Ok(())
    }

    /// Frees every buffer in stream order on `lane`.
    pub fn release(mut self, lane: &mut dyn LaneOps) -> Result<(), ForwardError> {
        for (_, buffer) in self.buffers.drain(..) {
            lane.free_async(buffer)?;
        }
        Ok(())
    }
}
