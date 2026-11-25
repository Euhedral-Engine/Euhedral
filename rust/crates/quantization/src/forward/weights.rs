//! A layer's weights: read from the checkpoint into host memory ([`stage`]), then copied to the device on a lane
//! ([`LayerWeights::upload`]).

use std::mem::replace;
use std::sync::Arc;

use euhedral_inference_cuda::{CudaError, DeviceBuffer, DevicePtr, LaneOps, PooledPinned};
use euhedral_io::IoMemory;

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

/// Host memory the checkpoint reads into: the first [`len`](IoMemory::len) bytes are the region.
pub trait HostMemory: IoMemory {
    /// The region.
    fn bytes(&self) -> &[u8];

    /// The region, writable.
    fn bytes_mut(&mut self) -> &mut [u8];

    /// Makes the region `len` bytes, at most what the memory was made with.
    fn set_len(&mut self, len: usize) -> Result<(), ForwardError>;
}

impl HostMemory for Vec<u8> {
    fn bytes(&self) -> &[u8] {
        self
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        self
    }

    fn set_len(&mut self, len: usize) -> Result<(), ForwardError> {
        self.resize(len, 0);
        Ok(())
    }
}

impl HostMemory for PooledPinned {
    fn bytes(&self) -> &[u8] {
        &self.buffer().as_slice()[..self.len()]
    }

    fn bytes_mut(&mut self) -> &mut [u8] {
        let len = self.len();
        &mut self.buffer_mut().as_mut_slice()[..len]
    }

    fn set_len(&mut self, len: usize) -> Result<(), ForwardError> {
        Ok(PooledPinned::set_len(self, len)?)
    }
}

#[derive(Debug)]
struct Entry<M> {
    part: Part,
    pieces: Vec<M>,
    bytes: usize,
    shape: Option<(u64, u64)>,
}

/// A layer read into host memory, not yet on the device. With pinned memory ([`StagedLayer<PooledPinned>`]) the
/// upload copies from the memory the checkpoint was read into.
#[derive(Debug)]
pub struct StagedLayer<M = PooledPinned> {
    prefix: String,
    attention: bool,
    entries: Vec<Entry<M>>,
}

impl<M: HostMemory> StagedLayer<M> {
    /// The object names' common prefix (`text/layers/N/`, or `mtp/layer/`).
    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    /// Whether the layer is a full-attention layer.
    pub fn attention(&self) -> bool {
        self.attention
    }

    /// A copy of the bytes of `part` as they are uploaded.
    pub fn bytes(&self, part: &Part) -> Option<Vec<u8>> {
        self.entries.iter().find(|entry| &entry.part == part).map(|entry| {
            entry
                .pieces
                .iter()
                .flat_map(|piece| piece.bytes().iter().copied())
                .collect()
        })
    }

    /// Every part with its size in bytes, in upload order.
    pub fn parts(&self) -> impl Iterator<Item = (&Part, usize)> {
        self.entries.iter().map(|entry| (&entry.part, entry.bytes))
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

/// Widens the BF16 vector `memory` holds to FP32 in the same memory, which was made for twice the bytes.
fn widen_in_place<M: HostMemory>(memory: &mut M) -> Result<(), ForwardError> {
    let raw = memory.len();
    if !raw.is_multiple_of(2) {
        return Err(ForwardError::Invalid(format!("{raw} bytes are not BF16 values")));
    }
    memory.set_len(raw * 2)?;
    let bytes = memory.bytes_mut();
    // From the back: each value's four bytes end at or after where its two began, so a value not yet widened is
    // never overwritten.
    for index in (0..raw / 2).rev() {
        let value = u32::from(u16::from_le_bytes([bytes[2 * index], bytes[2 * index + 1]])) << 16;
        bytes[4 * index..4 * index + 4].copy_from_slice(&value.to_le_bytes());
    }
    Ok(())
}

/// Transposes the convolution `memory` holds in the same memory.
fn transpose_in_place<M: HostMemory>(memory: &mut M, channels: usize, taps: usize) -> Result<(), ForwardError> {
    let transposed = transpose_convolution(memory.bytes(), channels, taps)?;
    memory.bytes_mut().copy_from_slice(&transposed);
    Ok(())
}

/// Reads `target` from `objects` into host memory with one set of reads, the memory from `alloc` (pinned memory from
/// a pool, made here and not later, when the lanes run); `then` gets the layer from the frame that ends the last
/// read. `Err` started nothing.
///
/// A quantized object that spans several pieces of the checkpoint keeps them apart, each read into memory of its own.
pub fn stage<M: HostMemory>(
    objects: &Qwen38Objects,
    target: Target,
    mut alloc: impl FnMut(usize) -> Result<M, CudaError>,
    then: Done<StagedLayer<M>>,
) -> Result<(), CheckpointError> {
    let store = objects.store();
    let mut memory = |bytes: usize| {
        alloc(bytes)
            .map_err(|error| CheckpointError::Invalid(format!("no memory for {bytes} bytes to read into: {error}")))
    };
    let mut ranges: Vec<ReadRange<M>> = Vec::new();
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
        for piece in &source.pieces {
            let bytes = piece.rows.checked_mul(row_bytes).ok_or_else(too_large)?;
            ranges.push(ReadRange {
                tensor: piece.tensor.clone(),
                skip: piece.first_row.checked_mul(row_bytes).ok_or_else(too_large)?,
                memory: memory(usize::try_from(bytes).map_err(|_| too_large())?)?,
            });
        }
        plans.push(Plan {
            part: Part::Object(name),
            count: source.pieces.len(),
            post: Post::Plain,
            shape: Some((source.rows, source.columns)),
        });
    }
    let mut tensor = |part: Part, name: String, post: Post| -> Result<(), CheckpointError> {
        let bytes = usize::try_from(store.tensor(&name)?.bytes)
            .map_err(|_| CheckpointError::Invalid(format!("{name}: the tensor is too large")))?;
        // A widened vector is read into the front of memory made for the result.
        let mut destination = memory(if matches!(post, Post::Floats) { bytes * 2 } else { bytes })?;
        destination
            .set_len(bytes)
            .map_err(|error| CheckpointError::Invalid(error.to_string()))?;
        plans.push(Plan {
            part,
            count: 1,
            post,
            shape: None,
        });
        ranges.push(ReadRange {
            tensor: name,
            skip: 0,
            memory: destination,
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
            let assembled = result.and_then(|memories| {
                let mut memories = memories.into_iter();
                let mut entries = Vec::with_capacity(plans.len());
                for plan in &plans {
                    let mut pieces: Vec<M> = memories.by_ref().take(plan.count).collect();
                    let invalid = |error: ForwardError| CheckpointError::Invalid(error.to_string());
                    match (plan.post, pieces.as_mut_slice()) {
                        (Post::Plain, _) => {}
                        (Post::Floats, [only]) => widen_in_place(only).map_err(invalid)?,
                        (Post::Convolution { channels, taps }, [only]) => {
                            transpose_in_place(only, channels, taps).map_err(invalid)?;
                        }
                        _ => return Err(CheckpointError::Invalid("a converted tensor is one piece".to_owned())),
                    }
                    entries.push(Entry {
                        part: plan.part.clone(),
                        bytes: pieces.iter().map(|piece| piece.len()).sum(),
                        pieces,
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
    /// Allocates the layer's buffers in stream order on `lane` and queues their uploads from `staged`'s pinned memory,
    /// `chunk_bytes` at a time, each piece of an object at its place in the object. Nothing is copied on the host and
    /// no pinned memory is made here.
    ///
    /// The allocations and the uploads are ordered only on `lane`: work on another lane must wait for this lane's
    /// work to retire first. On `Err` the buffers already made are freed in stream order.
    ///
    /// # Safety
    ///
    /// The queued copies read `staged`'s pinned buffers asynchronously. `staged` must stay alive, and its buffers
    /// must not return to the pool or be written, until `lane` has retired every copy this call queued. That holds
    /// on `Err` too: copies queued before the failure are still in flight.
    pub unsafe fn upload(
        lane: &mut dyn LaneOps,
        staged: &StagedLayer,
        chunk_bytes: usize,
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
                let buffer = lane.alloc_async(entry.bytes.max(1))?;
                let destination = buffer.ptr();
                weights.buffers.push((entry.part.clone(), buffer));
                if let (Part::Object(name), Some(shape)) = (&entry.part, entry.shape) {
                    weights.shapes.push((name.clone(), shape));
                }
                let mut offset = 0;
                for piece in &entry.pieces {
                    let mut start = 0;
                    while start < piece.len() {
                        let bytes = chunk_bytes.min(piece.len() - start);
                        // SAFETY: the piece is in `staged`, unmodified, and the caller of `upload` keeps `staged`
                        // until the lane has retired this copy.
                        unsafe {
                            lane.copy_upload_to_device(destination.offset(offset as u64), piece.buffer(), start, bytes)
                        }?;
                        start += bytes;
                        offset += bytes;
                    }
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

    /// Replaces object `name` with `decoded` (BF16, the same size, allocated in stream order), which this layer then
    /// owns; the old buffer is freed in stream order on `lane`, which orders it only against that lane's work.
    ///
    /// A replacement that is refused is freed in stream order too when it can be.
    pub fn replace(&mut self, lane: &mut dyn LaneOps, name: &str, decoded: DeviceBuffer) -> Result<(), ForwardError> {
        let refusal = match self
            .buffers
            .iter_mut()
            .find(|(held, _)| matches!(held, Part::Object(object) if object == name))
        {
            None => ForwardError::Invalid(format!("the layer has no object {name}")),
            Some(_) if !decoded.is_stream_ordered() => ForwardError::Invalid(format!(
                "the replacement of {name} was not allocated in stream order, so the layer could not free it"
            )),
            Some(slot) if slot.1.bytes() != decoded.bytes() => ForwardError::Invalid(format!(
                "{name} holds {} bytes, its replacement {}",
                slot.1.bytes(),
                decoded.bytes()
            )),
            Some(slot) => {
                let old = replace(&mut slot.1, decoded);
                lane.free_async(old)?;
                return Ok(());
            }
        };
        if decoded.is_stream_ordered() {
            lane.free_async(decoded)?;
        }
        Err(refusal)
    }

    /// Frees every buffer in stream order on `lane`, all of them even when one refuses; the first refusal is the
    /// error. The frees are ordered only against `lane`'s work.
    pub fn release(mut self, lane: &mut dyn LaneOps) -> Result<(), ForwardError> {
        let mut first = None;
        for (_, buffer) in self.buffers.drain(..) {
            if let Err(error) = lane.free_async(buffer) {
                first.get_or_insert(error);
            }
        }
        first.map_or(Ok(()), |error| Err(error.into()))
    }
}
