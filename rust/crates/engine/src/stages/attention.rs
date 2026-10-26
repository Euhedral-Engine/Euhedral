//! The attention stages: the Q/K RMSNorm with RoPE, the append of the quantum's keys and values to the sequence's
//! cache, and causal attention over it (`Stages.QkNormRope`, `KvAppend`, `CausalAttention`).
//!
//! The launch decisions (the per-head kernel below 2048 keys, the three-warp kernel from there, the row twins for
//! two to eight rows, the fused verifier) are made by the native library from the arguments given here; they are
//! the same in both engines.

use euhedral_graph::{Stage, StageCx, StageError, Submitted};
use euhedral_inference_cuda::{CudaError, DevicePtr, Submission, ops};
use euhedral_inference_settings::KvFormat;

use super::{Launch, Quantum, QuantumKind, eager, fatal, launch_u32};
use crate::residency::footprint::DRAFT_TWIN_ROWS;
use crate::state::KvView;
use crate::workspace::BufferSlot;

/// Which of a sequence's caches a layer reads and writes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvSource {
    /// The cache of the base model's full-attention layers.
    Base,
    /// The MTP layer's own cache.
    Mtp,
}

/// The signature of the append of one cache format.
type Append = fn(
    &Submission<'_>,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u64,
    DevicePtr,
) -> Result<(), CudaError>;

/// The signature of the attention over one cache format.
type Attend = fn(
    &Submission<'_>,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    DevicePtr,
    u32,
    u32,
    u32,
    u32,
    u32,
    u64,
    DevicePtr,
    DevicePtr,
) -> Result<(), CudaError>;

fn cache(quantum: &Quantum, source: KvSource) -> Result<&KvView, StageError> {
    quantum
        .seq
        .kv(source)
        .ok_or_else(|| fatal(format!("the quantum has no view of the {source:?} cache")))
}

/// The rotary dimension: the head dimension times the partial rotary factor, rounded (`Math.round`).
fn rotary_dim(head_dim: usize, partial_rotary_factor: f64) -> usize {
    (head_dim as f64 * partial_rotary_factor + 0.5).floor() as usize
}

/// The Q/K RMSNorm and RoPE of the projected query and key rows, at the quantum's positions.
#[derive(Clone, Debug)]
pub struct QkNormRope {
    input: BufferSlot,
    output: BufferSlot,
    query_norm: DevicePtr,
    key_norm: DevicePtr,
}

impl QkNormRope {
    pub fn new(input: BufferSlot, output: BufferSlot, query_norm: DevicePtr, key_norm: DevicePtr) -> Self {
        Self {
            input,
            output,
            query_norm,
            key_norm,
        }
    }
}

impl Launch for QkNormRope {
    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError> {
        let config = &quantum.config;
        let rotary = launch_u32(
            rotary_dim(config.attention_head_dim, config.partial_rotary_factor),
            "the rotary dimension",
        )?;
        ops::attention_qk_norm_rope_bf16(
            submission,
            quantum.slot(self.input),
            self.query_norm,
            self.key_norm,
            quantum.slot(self.output),
            quantum.rows_u32()?,
            launch_u32(config.num_attention_heads, "the query heads")?,
            launch_u32(config.num_key_value_heads, "the key value heads")?,
            launch_u32(config.attention_head_dim, "the head dimension")?,
            rotary,
            quantum.input.start_position(),
            quantum.position(),
            quantum.epsilon(),
            config.rope_theta,
        )
        .map_err(fatal)
    }
}

/// Appends the quantum's keys and values to layer `layer` of a cache. The rows are queued at once and become
/// visible to other quanta only when the quantum's owner commits them, once its device work retired. The pages must
/// be reserved before the quantum is admitted.
///
/// With host pages, the rows this quantum reads come into the device region first, and the pages it writes go
/// back after the append, on the same lane; the layer's attention reads them through the page table.
#[derive(Clone, Debug)]
pub struct KvAppend {
    source: KvSource,
    layer: usize,
    query_key: BufferSlot,
    gate_value: BufferSlot,
}

impl KvAppend {
    /// The append of the layer at position `layer` of the cache `source`: its keys follow the query rows of
    /// `query_key`, its values follow the gate rows of `gate_value`.
    pub fn new(source: KvSource, layer: usize, query_key: BufferSlot, gate_value: BufferSlot) -> Self {
        Self {
            source,
            layer,
            query_key,
            gate_value,
        }
    }
}

impl Launch for KvAppend {
    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError> {
        let config = &quantum.config;
        let cache = cache(quantum, self.source)?;
        let start = quantum.input.start_position();
        let rows = quantum.rows as u64;
        if !cache.reserves(start, rows) {
            return Err(fatal(format!(
                "an append of {rows} rows at {start} is not reserved: the cache held {} rows at admission",
                cache.reserved()
            )));
        }
        let (Some(keys), Some(values)) = (cache.key_table(self.layer), cache.value_table(self.layer)) else {
            return Err(fatal(format!("layer {} of the cache has no page table", self.layer)));
        };
        cache.stage_in(submission, self.layer, start, rows).map_err(fatal)?;
        let append: Append = match cache.format() {
            KvFormat::Nvfp4 => ops::attention_kv_append_nvfp4,
            KvFormat::Bf16 => ops::attention_kv_append_bf16,
        };
        append(
            submission,
            quantum.slot(self.query_key),
            quantum.slot(self.gate_value),
            keys,
            values,
            quantum.rows_u32()?,
            launch_u32(config.attention_query_width(), "the query width")?,
            launch_u32(config.kv_width(), "the key width")?,
            start,
            quantum.position(),
        )
        .map_err(fatal)?;
        cache.stage_out(submission, self.layer, start, rows).map_err(fatal)
    }
}

/// Whether a quantum's attention runs on the decode scratch: one-row decode, a verification (row twins), and a
/// small draft quantum, whose catch-up rows run on the twins as well (far more parallel over a long cache than a
/// 32-row prefill tile, and drafts only propose tokens).
fn takes_decode_scratch(kind: QuantumKind, rows: usize) -> bool {
    rows == 1 || kind == QuantumKind::Verify || (kind == QuantumKind::Draft && rows as u64 <= DRAFT_TWIN_ROWS)
}

/// Causal attention over layer `layer` of a cache, including the rows this quantum appended, with the output gate
/// applied.
#[derive(Clone, Debug)]
pub struct AttentionCausal {
    source: KvSource,
    layer: usize,
    query_key: BufferSlot,
    gate_value: BufferSlot,
    output: BufferSlot,
}

impl AttentionCausal {
    pub fn new(
        source: KvSource,
        layer: usize,
        query_key: BufferSlot,
        gate_value: BufferSlot,
        output: BufferSlot,
    ) -> Self {
        Self {
            source,
            layer,
            query_key,
            gate_value,
            output,
        }
    }
}

impl Launch for AttentionCausal {
    fn launch(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError> {
        let config = &quantum.config;
        let cache = cache(quantum, self.source)?;
        let start = quantum.input.start_position();
        let (Some(keys), Some(values)) = (cache.key_table(self.layer), cache.value_table(self.layer)) else {
            return Err(fatal(format!("layer {} of the cache has no page table", self.layer)));
        };
        // The append stage queued these rows earlier on the quantum's lane: readable here, not yet committed.
        let cache_length = u32::try_from(start + quantum.rows as u64)
            .map_err(|_| fatal(format!("{} keys exceed a launch", start + quantum.rows as u64)))?;
        let scratch = if takes_decode_scratch(quantum.kind, quantum.rows) {
            let scratch = &quantum.seq.device().scratch;
            if quantum.rows as u64 > scratch.decode_rows() {
                return Err(fatal(format!(
                    "{} attention rows exceed the {} the decode scratch holds",
                    quantum.rows,
                    scratch.decode_rows()
                )));
            }
            scratch.decode().ptr()
        } else {
            DevicePtr::NULL
        };
        let attend: Attend = match cache.format() {
            KvFormat::Nvfp4 => ops::attention_causal_nvfp4,
            KvFormat::Bf16 => ops::attention_causal_bf16,
        };
        attend(
            submission,
            quantum.slot(self.query_key),
            quantum.slot(self.gate_value),
            keys,
            values,
            quantum.slot(self.output),
            quantum.rows_u32()?,
            launch_u32(config.num_attention_heads, "the query heads")?,
            launch_u32(config.num_key_value_heads, "the key value heads")?,
            launch_u32(config.attention_head_dim, "the head dimension")?,
            cache_length,
            start,
            quantum.position(),
            scratch,
        )
        .map_err(fatal)
    }
}

eager!(QkNormRope, KvAppend, AttentionCausal);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_rotary_dimension_rounds_like_java() {
        assert_eq!(rotary_dim(256, 0.25), 64);
        assert_eq!(rotary_dim(256, 0.5), 128);
        assert_eq!(rotary_dim(10, 0.25), 3);
        assert_eq!(rotary_dim(6, 0.25), 2);
    }

    #[test]
    fn decode_scratch_serves_decode_verification_and_small_drafts() {
        assert!(takes_decode_scratch(QuantumKind::Decode, 1));
        assert!(takes_decode_scratch(QuantumKind::Prefill, 1));
        assert!(takes_decode_scratch(QuantumKind::Verify, 5));
        assert!(takes_decode_scratch(QuantumKind::Draft, 8));
        assert!(!takes_decode_scratch(QuantumKind::Draft, 9));
        assert!(!takes_decode_scratch(QuantumKind::Prefill, 8));
        assert!(!takes_decode_scratch(QuantumKind::Decode, 2));
    }
}
