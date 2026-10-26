//! The layer operation stages on a GPU. The tests are `#[ignore]`d and run locally in a memory-capped scope:
//! `systemd-run --user --scope -p MemoryMax=16G cargo test -p euhedral-inference-engine --test stages_gpu -- --ignored`.
//!
//! The library comes from the `euhedral.cuda.library` property (`EUHEDRAL_TEST_CUDA_LIBRARY` in the environment, or
//! `gradle.properties`); a test skips with a message when it is absent. Each test builds its own library, lane and
//! workspace, at the model's real dimensions with two layers (one GDN, one full attention): the stages are run
//! directly, one at a time, with a lane the test synchronizes to read the results.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: a test synchronizes a lane and copies synchronously to check results"
)]
#![allow(
    clippy::print_stdout,
    reason = "a test reports the error it measured against its reference"
)]

use std::path::PathBuf;
use std::sync::Arc;

use euhedral_graph::{DeviceQueue, LaneId, LanePool, Marker, Stage, StageCx, StageId};
use euhedral_inference_cuda::{
    CudaError, CudaLane, CudaLibrary, DeviceBuffer, DevicePtr, PinnedPool, Submission, SubmitMode, ops,
};
use euhedral_inference_edrl::{LayerType, ModelMetadata, TensorDataType, WeightFormat, WeightLayout};
use euhedral_inference_engine::formats::{LinearFormat, LinearMode, ScratchRegion};
use euhedral_inference_engine::model::{ModelConfig, WeightRef};
use euhedral_inference_engine::stages::attention::{AttentionCausal, KvAppend, KvSource, QkNormRope};
use euhedral_inference_engine::stages::elementwise::{ResidualAdd, Swiglu};
use euhedral_inference_engine::stages::gdn::{GdnConvolution, GdnGatedRmsNorm, GdnProjectControl, GdnRecurrence};
use euhedral_inference_engine::stages::linear::{Linear, LinearRows};
use euhedral_inference_engine::stages::mtp::MtpStem;
use euhedral_inference_engine::stages::norm::{NormRows, ResidualRmsNorm, RmsNormUnitOffset};
use euhedral_inference_engine::stages::sampling::{Argmax, SelectedRows, SelectedToken, Selection};
use euhedral_inference_engine::stages::{
    DraftSeed, DraftTokenWrite, InputRecordRef, LaneTable, Quantum, QuantumKind, SequenceRefs,
};
use euhedral_inference_engine::state::SequenceState;
use euhedral_inference_engine::state::attention::AttentionScratch;
use euhedral_inference_engine::state::gdn::GdnStates;
use euhedral_inference_engine::state::kv::{KvCache, KvRetained};
use euhedral_inference_engine::workspace::{BufferSlot, Workspace, WorkspaceRows};
use euhedral_inference_settings::KvFormat;

// The rig.

fn library() -> Option<Arc<CudaLibrary>> {
    let path = test_support::require_property("euhedral.cuda.library")?;
    Some(CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads"))
}

/// The model's dimensions with `kinds` as its layers.
fn config(kinds: Vec<LayerType>) -> Arc<ModelConfig> {
    Arc::new(ModelConfig::from(&ModelMetadata {
        vocab_size: 248_320,
        hidden_size: 5120,
        num_hidden_layers: kinds.len() as u32,
        num_attention_heads: 24,
        num_key_value_heads: 4,
        attention_head_dim: 256,
        intermediate_size: 17_408,
        linear_num_key_heads: 16,
        linear_num_value_heads: 48,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        rms_norm_epsilon: 1e-6,
        rope_theta: 1e7,
        partial_rotary_factor: 0.25,
        max_position_embeddings: 262_144,
        hidden_activation: "silu".to_string(),
        layer_types: kinds,
        num_experts: 0,
        num_experts_per_token: 0,
        moe_intermediate_size: 0,
        shared_expert_intermediate_size: 0,
        tie_word_embeddings: false,
        attention_output_gate: true,
        mtp_layer_count: 1,
        draft_depth: Some(4),
    }))
}

struct Rig {
    library: Arc<CudaLibrary>,
    lane: Arc<CudaLane>,
    lanes: LaneTable,
    pool: LanePool,
    pinned: Arc<PinnedPool>,
}

fn rig() -> Option<Rig> {
    let library = library()?;
    let lane = CudaLane::new(&library, LaneId(0), 256).unwrap();
    let pool = LanePool::new(vec![Arc::clone(&lane) as Arc<dyn DeviceQueue>]);
    Some(Rig {
        lanes: LaneTable::new(vec![Arc::clone(&lane)]).unwrap(),
        pinned: PinnedPool::new(&library, 64 << 20),
        library,
        lane,
        pool,
    })
}

impl Rig {
    fn workspace(&self, config: &ModelConfig, rows: u64, logits_rows: u64, scratch: &[u64]) -> Arc<Workspace> {
        Arc::new(Workspace::allocate(&self.library, config, WorkspaceRows { rows, logits_rows }, scratch).unwrap())
    }

    /// Runs `stage` on the rig's lane for `quantum` and returns its marker, if it queued work.
    fn run(&self, stage: &dyn Stage<Quantum>, quantum: &Quantum) -> Option<Marker> {
        let cx = StageCx {
            stage: StageId(0),
            lane: Some(LaneId(0)),
            waits: &[],
            lanes: &self.pool,
        };
        stage.run(quantum, &cx).unwrap().marker
    }

    fn sync(&self) {
        self.lane.synchronize().unwrap();
    }

    /// Queues `work` in a submission of the rig's lane and waits for it.
    fn submit(&self, work: impl FnOnce(&Submission<'_>) -> Result<(), CudaError>) {
        let submission = Submission::begin(&self.lane, SubmitMode::default()).unwrap();
        work(&submission).unwrap();
        submission.finish().unwrap();
        self.sync();
    }

    fn quantum(
        &self,
        kind: QuantumKind,
        rows: usize,
        start: u64,
        workspace: &Arc<Workspace>,
        sequence: &Arc<SequenceState>,
        config: &Arc<ModelConfig>,
    ) -> Quantum {
        let mode = SubmitMode {
            pdl: false,
            row_exact: kind == QuantumKind::Verify,
        };
        // The admission uploads the input record: the start position, after the token ids.
        let offset = InputRecordRef::position_offset(rows) as usize;
        workspace
            .buffer(BufferSlot::TokenIds)
            .copy_from_host(offset, &start.to_ne_bytes())
            .unwrap();
        Quantum::new(
            kind,
            rows,
            InputRecordRef::new(start),
            Arc::clone(workspace),
            SequenceRefs::new(Arc::clone(sequence)),
            mode,
            Arc::clone(config),
            self.lanes.clone(),
        )
        .unwrap()
    }

    /// A sequence with a GDN state (speculative buffers for 8 rows), a cache of `kv_layers` layers in `format` for
    /// `kv_tokens` positions, and draft buffers.
    fn sequence(
        &self,
        config: &ModelConfig,
        kv_layers: usize,
        format: KvFormat,
        kv_tokens: u64,
    ) -> (SequenceState, KvRetained) {
        let submission = Submission::begin(&self.lane, SubmitMode::default()).unwrap();
        let gdn = GdnStates::allocate(&self.library, config, 8, &submission).unwrap();
        let mut kv = KvCache::new(&self.library, kv_layers, 1024, format, kv_tokens, 0).unwrap();
        let mut retained = KvRetained::default();
        if kv_layers > 0 && kv_tokens > 0 {
            kv.reserve(&submission, &self.pinned, 0, kv_tokens, &mut retained)
                .unwrap();
        }
        submission.finish_signalled().unwrap();
        self.sync();
        let scratch = AttentionScratch::allocate(&self.library, config, 8, true, 16).unwrap();
        (
            SequenceState {
                kv,
                mtp_kv: None,
                gdn,
                scratch,
            },
            retained,
        )
    }
}

// Host data.

struct Rng(u64);

impl Rng {
    /// A value in `[-1, 1)`.
    fn next(&mut self) -> f32 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }

    fn values(&mut self, count: usize, scale: f32) -> Vec<f32> {
        (0..count).map(|_| self.next() * scale).collect()
    }
}

fn to_bf16(value: f32) -> u16 {
    let bits = value.to_bits();
    if value.is_nan() {
        return 0x7fc0;
    }
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn from_bf16(bits: u16) -> f32 {
    f32::from_bits(u32::from(bits) << 16)
}

fn bf16_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| to_bf16(*value).to_ne_bytes()).collect()
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_ne_bytes()).collect()
}

fn bf16_values(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(2)
        .map(|pair| from_bf16(u16::from_ne_bytes([pair[0], pair[1]])))
        .collect()
}

/// Writes `bytes` at `offset` of the workspace slot `slot`.
fn put(workspace: &Workspace, slot: BufferSlot, offset: usize, bytes: &[u8]) {
    workspace.buffer(slot).copy_from_host(offset, bytes).unwrap();
}

/// Reads `bytes` bytes at `offset` of the workspace slot `slot`.
fn get(workspace: &Workspace, slot: BufferSlot, offset: usize, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    workspace.buffer(slot).copy_to_host(offset, &mut out).unwrap();
    out
}

fn upload(library: &Arc<CudaLibrary>, bytes: &[u8]) -> DeviceBuffer {
    let buffer = DeviceBuffer::alloc(library, bytes.len()).unwrap();
    buffer.copy_from_host(0, bytes).unwrap();
    buffer
}

fn download(buffer: &DeviceBuffer) -> Vec<u8> {
    let mut out = vec![0u8; buffer.bytes()];
    buffer.copy_to_host(0, &mut out).unwrap();
    out
}

/// Whether `actual` is within `relative` of `expected` (and `absolute` near zero), element by element.
fn assert_close(what: &str, actual: &[f32], expected: &[f32], relative: f32, absolute: f32) {
    assert_eq!(actual.len(), expected.len(), "{what}: length");
    let mut worst = 0.0f32;
    for (index, (a, e)) in actual.iter().zip(expected).enumerate() {
        let bound = relative * e.abs() + absolute;
        let error = (a - e).abs();
        worst = worst.max(error / bound);
        assert!(error <= bound, "{what}[{index}]: {a} against {e} (bound {bound})");
    }
    println!("{what}: worst error {worst:.3} of its bound");
}

// Norms and elementwise.

#[test]
#[ignore = "needs a GPU and the native library"]
fn norms_and_elementwise_match_the_host_reference() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 0, KvFormat::Nvfp4, 0);
    let sequence = Arc::new(sequence);
    let workspace = rig.workspace(&config, 4, 4, &[]);
    let hidden = 5120usize;
    let rows = 4usize;
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let residual = rng.values(rows * hidden, 2.0);
    let delta = rng.values(rows * hidden, 1.0);
    let weight = rng.values(hidden, 0.5);
    let weight_buffer = upload(&rig.library, &bf16_bytes(&weight));
    let weight_bf16: Vec<f32> = bf16_values(&bf16_bytes(&weight));
    let residual_bf16 = bf16_values(&bf16_bytes(&residual));
    let delta_bf16 = bf16_values(&bf16_bytes(&delta));
    put(&workspace, BufferSlot::HiddenState, 0, &bf16_bytes(&residual));
    put(&workspace, BufferSlot::MixerDelta, 0, &bf16_bytes(&delta));

    // The rounded add.
    let quantum = rig.quantum(QuantumKind::Prefill, rows, 0, &workspace, &sequence, &config);
    let add = ResidualAdd::new(BufferSlot::HiddenState, BufferSlot::MixerDelta, BufferSlot::MixerHidden);
    assert!(rig.run(&add, &quantum).is_some());
    rig.sync();
    let sum: Vec<f32> = residual_bf16
        .iter()
        .zip(&delta_bf16)
        .map(|(a, b)| from_bf16(to_bf16(a + b)))
        .collect();
    let got = bf16_values(&get(&workspace, BufferSlot::MixerHidden, 0, rows * hidden * 2));
    assert_eq!(got, sum, "the add is the BF16 rounding of the FP32 sum");

    // The (1 + w) norm of every row, and the add fused with it.
    let norm = |row: &[f32]| -> Vec<f32> {
        let mean = row.iter().map(|x| x * x).sum::<f32>() / row.len() as f32;
        let scale = 1.0 / (mean + 1e-6).sqrt();
        row.iter()
            .zip(&weight_bf16)
            .map(|(x, w)| x * scale * (1.0 + w))
            .collect()
    };
    let expected: Vec<f32> = sum.chunks(hidden).flat_map(norm).collect();
    let fused = ResidualRmsNorm::new(
        BufferSlot::HiddenState,
        BufferSlot::MixerDelta,
        weight_buffer.ptr(),
        BufferSlot::FinalHiddenState,
        BufferSlot::InputNormalized,
    );
    assert!(rig.run(&fused, &quantum).is_some());
    rig.sync();
    let hidden_out = bf16_values(&get(&workspace, BufferSlot::FinalHiddenState, 0, rows * hidden * 2));
    assert_eq!(hidden_out, sum, "the fused add rounds as the add does");
    let normalized = bf16_values(&get(&workspace, BufferSlot::InputNormalized, 0, rows * hidden * 2));
    assert_close("the fused norm", &normalized, &expected, 0.02, 1e-3);

    let plain = RmsNormUnitOffset::new(
        BufferSlot::MixerHidden,
        BufferSlot::PostMixerNormalized,
        weight_buffer.ptr(),
        NormRows::All,
    );
    assert!(rig.run(&plain, &quantum).is_some());
    rig.sync();
    let got = bf16_values(&get(&workspace, BufferSlot::PostMixerNormalized, 0, rows * hidden * 2));
    assert_close("the norm", &got, &expected, 0.02, 1e-3);

    // The final norm covers the logits rows, the last ones of the input: here the last row.
    let last = rig
        .quantum(QuantumKind::Prefill, rows, 0, &workspace, &sequence, &config)
        .with_logits_rows(1)
        .unwrap();
    let final_norm = RmsNormUnitOffset::new(
        BufferSlot::MixerHidden,
        BufferSlot::FinalNormalized,
        weight_buffer.ptr(),
        NormRows::Final,
    );
    assert!(rig.run(&final_norm, &last).is_some());
    rig.sync();
    let got = bf16_values(&get(&workspace, BufferSlot::FinalNormalized, 0, hidden * 2));
    assert_close(
        "the final norm of the last row",
        &got,
        &expected[(rows - 1) * hidden..],
        0.02,
        1e-3,
    );

    // A prefill chunk that is not the last scores nothing: no work, so nothing for a successor to wait on.
    let none = rig
        .quantum(QuantumKind::Prefill, rows, 0, &workspace, &sequence, &config)
        .with_logits_rows(0)
        .unwrap();
    assert!(rig.run(&final_norm, &none).is_none());

    // SwiGLU: silu(gate) * up, the gate half first.
    let intermediate = 17_408usize;
    let gate_up = rng.values(rows * 2 * intermediate, 3.0);
    put(&workspace, BufferSlot::GateUp, 0, &bf16_bytes(&gate_up));
    let swiglu = Swiglu::new(BufferSlot::GateUp, BufferSlot::Swiglu);
    assert!(rig.run(&swiglu, &quantum).is_some());
    rig.sync();
    let gate_up = bf16_values(&bf16_bytes(&gate_up));
    let expected: Vec<f32> = (0..rows * intermediate)
        .map(|index| {
            let (row, column) = (index / intermediate, index % intermediate);
            let gate = gate_up[row * 2 * intermediate + column];
            let up = gate_up[row * 2 * intermediate + intermediate + column];
            gate / (1.0 + (-gate).exp()) * up
        })
        .collect();
    let got = bf16_values(&get(&workspace, BufferSlot::Swiglu, 0, rows * intermediate * 2));
    assert_close("swiglu", &got, &expected, 0.02, 1e-3);
}

// The linear stage.

/// A format that queues one device copy of its input rows (four BF16 values each) to its output, and remembers the
/// shape of the call.
struct StubFormat {
    calls: AtomicUsize,
    rows: AtomicUsize,
    flags: AtomicUsize,
    scratch_bytes: AtomicUsize,
    scratch_needed: usize,
}

use std::sync::atomic::{AtomicUsize, Ordering};

impl LinearFormat for StubFormat {
    fn submit(
        &self,
        submission: &Submission<'_>,
        weight: &WeightRef,
        x: DevicePtr,
        y: DevicePtr,
        rows: usize,
        scratch: ScratchRegion,
        mode: LinearMode,
    ) -> Result<(), CudaError> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        self.rows.store(rows, Ordering::SeqCst);
        self.flags.store(
            usize::from(mode.row_exact) | usize::from(mode.prefill) << 1,
            Ordering::SeqCst,
        );
        self.scratch_bytes.store(scratch.bytes, Ordering::SeqCst);
        assert_eq!(weight.name, "stub");
        // Four BF16 values per row.
        ops::copy_device_to_device(submission, y, x, (rows * 8) as u64)
    }

    fn scratch_bytes(&self, _rows: usize) -> usize {
        self.scratch_needed
    }
}

fn stub_weight() -> WeightRef {
    WeightRef {
        name: "stub".to_string(),
        shape: vec![1, 1],
        dtype: TensorDataType::Bf16,
        format: WeightFormat::Bf16,
        layout: WeightLayout::ContiguousLeV1,
        ptr: DevicePtr::NULL,
        bytes: 2,
        trellis: None,
    }
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn the_linear_stage_hands_its_format_the_geometry_the_mode_and_the_scratch() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 0, KvFormat::Nvfp4, 0);
    let sequence = Arc::new(sequence);
    let workspace = rig.workspace(&config, 8, 8, &[1 << 20]);
    let rows = 5usize;
    let input = Rng(7).values(rows * 4, 1.0);
    put(&workspace, BufferSlot::FinalNormalized, 0, &bf16_bytes(&input));

    let format = Arc::new(StubFormat {
        calls: AtomicUsize::new(0),
        rows: AtomicUsize::new(0),
        flags: AtomicUsize::new(0),
        scratch_bytes: AtomicUsize::new(0),
        scratch_needed: 4096,
    });
    let head = Linear::new(
        Arc::clone(&format) as Arc<dyn LinearFormat>,
        stub_weight(),
        BufferSlot::FinalNormalized,
        BufferSlot::Logits,
        LinearRows::Logits,
        Some(0),
    );
    // A verification: every row is a logits row, and the launch is row-exact.
    let verify = rig.quantum(QuantumKind::Verify, rows, 0, &workspace, &sequence, &config);
    assert!(rig.run(&head, &verify).is_some());
    rig.sync();
    assert_eq!(format.calls.load(Ordering::SeqCst), 1);
    assert_eq!(format.rows.load(Ordering::SeqCst), rows);
    assert_eq!(format.flags.load(Ordering::SeqCst), 0b01, "row-exact, not prefill");
    assert_eq!(format.scratch_bytes.load(Ordering::SeqCst), 1 << 20);
    assert_eq!(
        bf16_values(&get(&workspace, BufferSlot::Logits, 0, rows * 4 * 2)),
        bf16_values(&bf16_bytes(&input)),
        "the format wrote the logits slot from the input slot"
    );

    // A prefill chunk that scores one row.
    let prefill = rig
        .quantum(QuantumKind::Prefill, rows, 0, &workspace, &sequence, &config)
        .with_logits_rows(1)
        .unwrap();
    assert!(rig.run(&head, &prefill).is_some());
    rig.sync();
    assert_eq!(format.rows.load(Ordering::SeqCst), 1);
    assert_eq!(format.flags.load(Ordering::SeqCst), 0b10, "prefill, not row-exact");

    // A chunk that scores none: the format is not called.
    let none = rig
        .quantum(QuantumKind::Prefill, rows, 0, &workspace, &sequence, &config)
        .with_logits_rows(0)
        .unwrap();
    assert!(rig.run(&head, &none).is_none());
    assert_eq!(format.calls.load(Ordering::SeqCst), 2);

    // The projection of all rows, without the scratch region the route does not name, refuses a route that needs it.
    let all = Linear::new(
        Arc::clone(&format) as Arc<dyn LinearFormat>,
        stub_weight(),
        BufferSlot::FinalNormalized,
        BufferSlot::GateUp,
        LinearRows::All,
        None,
    );
    let cx = StageCx {
        stage: StageId(0),
        lane: Some(LaneId(0)),
        waits: &[],
        lanes: &rig.pool,
    };
    let refused = all.run(&verify, &cx).unwrap_err();
    assert!(format!("{refused:?}").contains("scratch"), "{refused:?}");
    assert_eq!(format.calls.load(Ordering::SeqCst), 2, "the format was not called");
}

// The GDN stages.

const GDN_QUERY_KEY: usize = 4096;
const GDN_VALUE_Z: usize = 12_288;
const GDN_VALUE: usize = 6144;
const GDN_HEADS: usize = 48;
const GDN_CHANNELS: usize = 10_240;

/// One row of the GDN stages' inputs: the projected query/key and value/z rows and the control.
struct GdnRow {
    query_key: Vec<f32>,
    value_z: Vec<f32>,
    alpha: Vec<f32>,
    beta: Vec<f32>,
}

fn gdn_row(rng: &mut Rng) -> GdnRow {
    GdnRow {
        query_key: rng.values(GDN_QUERY_KEY, 1.0),
        value_z: rng.values(GDN_VALUE_Z, 1.0),
        alpha: (0..GDN_HEADS).map(|_| 0.8 + 0.19 * (rng.next() * 0.5 + 0.5)).collect(),
        beta: (0..GDN_HEADS).map(|_| 0.1 + 0.8 * (rng.next() * 0.5 + 0.5)).collect(),
    }
}

struct GdnRig<'r> {
    rig: &'r Rig,
    config: Arc<ModelConfig>,
    workspace: Arc<Workspace>,
    convolution: GdnConvolution,
    recurrence: GdnRecurrence,
    // Declared last: the stages hold its address.
    _weights: DeviceBuffer,
}

impl<'r> GdnRig<'r> {
    fn new(rig: &'r Rig, config: &Arc<ModelConfig>, rng: &mut Rng) -> Self {
        let weights = upload(&rig.library, &bf16_bytes(&rng.values(GDN_CHANNELS * 4, 0.5)));
        GdnRig {
            rig,
            config: Arc::clone(config),
            workspace: rig.workspace(config, 8, 8, &[]),
            convolution: GdnConvolution::new(
                0,
                BufferSlot::QkProjected,
                BufferSlot::ValueZProjected,
                weights.ptr(),
                BufferSlot::GdnConvolved,
            ),
            recurrence: GdnRecurrence::new(
                0,
                BufferSlot::GdnConvolved,
                BufferSlot::GdnAlpha,
                BufferSlot::GdnBeta,
                BufferSlot::GdnRecurrent,
            ),
            _weights: weights,
        }
    }

    fn sequence(&self) -> Arc<SequenceState> {
        Arc::new(self.rig.sequence(&self.config, 0, KvFormat::Nvfp4, 0).0)
    }

    /// Runs the convolution and the recurrence over `rows` as one quantum of `kind`; returns the recurrence output.
    fn step(&self, sequence: &Arc<SequenceState>, kind: QuantumKind, rows: &[GdnRow]) -> Vec<f32> {
        for (index, row) in rows.iter().enumerate() {
            put(
                &self.workspace,
                BufferSlot::QkProjected,
                index * GDN_QUERY_KEY * 2,
                &bf16_bytes(&row.query_key),
            );
            put(
                &self.workspace,
                BufferSlot::ValueZProjected,
                index * GDN_VALUE_Z * 2,
                &bf16_bytes(&row.value_z),
            );
            put(
                &self.workspace,
                BufferSlot::GdnAlpha,
                index * GDN_HEADS * 4,
                &f32_bytes(&row.alpha),
            );
            put(
                &self.workspace,
                BufferSlot::GdnBeta,
                index * GDN_HEADS * 4,
                &f32_bytes(&row.beta),
            );
        }
        let quantum = self
            .rig
            .quantum(kind, rows.len(), 0, &self.workspace, sequence, &self.config);
        assert!(self.rig.run(&self.convolution, &quantum).is_some());
        assert!(self.rig.run(&self.recurrence, &quantum).is_some());
        self.rig.sync();
        bf16_values(&get(
            &self.workspace,
            BufferSlot::GdnRecurrent,
            0,
            rows.len() * GDN_VALUE * 2,
        ))
    }

    /// The layer's convolution and recurrent state.
    fn state(&self, sequence: &Arc<SequenceState>) -> (Vec<u8>, Vec<u8>) {
        let layer = sequence.gdn.layer(0).unwrap();
        (download(layer.convolution()), download(layer.recurrent()))
    }
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn gdn_replay_restores_committed_rows() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let mut rng = Rng(0x2545_f491_4f6c_dd1d);
    let gdn = GdnRig::new(&rig, &config, &mut rng);
    let rows: Vec<GdnRow> = (0..8).map(|_| gdn_row(&mut rng)).collect();
    // Two rows of history, five rows of a verification of which the first three are accepted, and a row after it.
    let (history, verified, next) = (&rows[0..2], &rows[2..7], &rows[7..8]);

    // One-row decode of the history, the accepted rows and the next row: the state the replay must reproduce.
    let decoded = gdn.sequence();
    for row in history {
        gdn.step(&decoded, QuantumKind::Decode, std::slice::from_ref(row));
    }
    let mut decoded_outputs = Vec::new();
    for row in verified {
        decoded_outputs.push(gdn.step(&decoded, QuantumKind::Decode, std::slice::from_ref(row)));
    }
    // The state after the accepted three: the verification's row-exact rows are what decode would have computed.
    let verified_whole = gdn.sequence();
    for row in history {
        gdn.step(&verified_whole, QuantumKind::Decode, std::slice::from_ref(row));
    }
    let verified_outputs = gdn.step(&verified_whole, QuantumKind::Verify, verified);
    assert_eq!(
        gdn.state(&verified_whole),
        gdn.state(&decoded),
        "a row-exact verification leaves the state one-row decode of its rows leaves"
    );
    for (index, output) in decoded_outputs.iter().enumerate() {
        assert_eq!(
            &verified_outputs[index * GDN_VALUE..(index + 1) * GDN_VALUE],
            &output[..],
            "verified row {index}"
        );
    }

    // The accepted prefix: decode of the history and the first three rows, then the next row.
    let accepted = gdn.sequence();
    for row in history.iter().chain(&verified[..3]) {
        gdn.step(&accepted, QuantumKind::Decode, std::slice::from_ref(row));
    }
    let accepted_next = gdn.step(&accepted, QuantumKind::Decode, next);

    // The verification ran past the accepted rows; the next quantum replays them before its own.
    let replayed = gdn.sequence();
    for row in history {
        gdn.step(&replayed, QuantumKind::Decode, std::slice::from_ref(row));
    }
    gdn.step(&replayed, QuantumKind::Verify, verified);
    let layer = replayed.gdn.layer(0).unwrap();
    assert_eq!(layer.pending_replay_rows(), 0);
    layer.set_pending_replay_rows(3).unwrap();
    let replayed_next = gdn.step(&replayed, QuantumKind::Decode, next);
    assert_eq!(
        layer.pending_replay_rows(),
        0,
        "the replay clears the layer's pending rows"
    );
    assert_eq!(replayed_next, accepted_next, "the next row after a replay");
    assert_eq!(
        gdn.state(&replayed),
        gdn.state(&accepted),
        "the replayed state is the accepted rows' decode state, then the next row"
    );

    // Without the replay the state has run past the accepted rows.
    let unreplayed = gdn.sequence();
    for row in history {
        gdn.step(&unreplayed, QuantumKind::Decode, std::slice::from_ref(row));
    }
    gdn.step(&unreplayed, QuantumKind::Verify, verified);
    let unreplayed_next = gdn.step(&unreplayed, QuantumKind::Decode, next);
    assert_ne!(unreplayed_next, accepted_next);
    assert_ne!(gdn.state(&unreplayed).1, gdn.state(&accepted).1);

    // A replay of every row of the verification is its own checkpoint run again: all five rows, then the next.
    let full = gdn.sequence();
    for row in history {
        gdn.step(&full, QuantumKind::Decode, std::slice::from_ref(row));
    }
    gdn.step(&full, QuantumKind::Verify, verified);
    full.gdn.layer(0).unwrap().set_pending_replay_rows(5).unwrap();
    let full_next = gdn.step(&full, QuantumKind::Decode, next);
    let all_decoded = gdn.sequence();
    for row in history.iter().chain(verified) {
        gdn.step(&all_decoded, QuantumKind::Decode, std::slice::from_ref(row));
    }
    assert_eq!(full_next, gdn.step(&all_decoded, QuantumKind::Decode, next));
    assert_eq!(gdn.state(&full), gdn.state(&all_decoded));
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn gdn_project_control_and_gated_norm_match_the_host_reference() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 0, KvFormat::Nvfp4, 0);
    let sequence = Arc::new(sequence);
    let workspace = rig.workspace(&config, 4, 4, &[]);
    let (rows, hidden) = (3usize, 5120usize);
    let mut rng = Rng(0x1234_5678_9abc_def1);
    let input = bf16_values(&bf16_bytes(&rng.values(rows * hidden, 1.0)));
    let a_weight = bf16_values(&bf16_bytes(&rng.values(GDN_HEADS * hidden, 0.05)));
    let b_weight = bf16_values(&bf16_bytes(&rng.values(GDN_HEADS * hidden, 0.05)));
    let a_log = rng.values(GDN_HEADS, 1.0);
    let dt_bias = rng.values(GDN_HEADS, 1.0);
    put(&workspace, BufferSlot::InputNormalized, 0, &bf16_bytes(&input));
    let a_buffer = upload(&rig.library, &bf16_bytes(&a_weight));
    let b_buffer = upload(&rig.library, &bf16_bytes(&b_weight));
    let log_buffer = upload(&rig.library, &f32_bytes(&a_log));
    let bias_buffer = upload(&rig.library, &f32_bytes(&dt_bias));
    let stage = GdnProjectControl::new(
        BufferSlot::InputNormalized,
        a_buffer.ptr(),
        b_buffer.ptr(),
        log_buffer.ptr(),
        bias_buffer.ptr(),
        BufferSlot::GdnAlpha,
        BufferSlot::GdnBeta,
    );
    let quantum = rig.quantum(QuantumKind::Prefill, rows, 0, &workspace, &sequence, &config);
    assert!(rig.run(&stage, &quantum).is_some());
    rig.sync();
    let read = |slot| -> Vec<f32> {
        get(&workspace, slot, 0, rows * GDN_HEADS * 4)
            .chunks_exact(4)
            .map(|word| f32::from_ne_bytes([word[0], word[1], word[2], word[3]]))
            .collect()
    };
    let (alpha, beta) = (read(BufferSlot::GdnAlpha), read(BufferSlot::GdnBeta));
    let dot = |x: &[f32], w: &[f32]| x.iter().zip(w).map(|(x, w)| f64::from(*x) * f64::from(*w)).sum::<f64>();
    let (mut expected_alpha, mut expected_beta) = (Vec::new(), Vec::new());
    for row in 0..rows {
        for head in 0..GDN_HEADS {
            let x = &input[row * hidden..(row + 1) * hidden];
            let a = dot(x, &a_weight[head * hidden..(head + 1) * hidden]) + f64::from(dt_bias[head]);
            let softplus = a.max(0.0) + (-a.abs()).exp().ln_1p();
            expected_alpha.push((-f64::from(a_log[head]).exp() * softplus).exp() as f32);
            let b = dot(x, &b_weight[head * hidden..(head + 1) * hidden]);
            expected_beta.push((1.0 / (1.0 + (-b).exp())) as f32);
        }
    }
    assert_close("alpha", &alpha, &expected_alpha, 1e-4, 1e-6);
    assert_close("beta", &beta, &expected_beta, 1e-4, 1e-6);

    // The gated norm: x * rsqrt(mean(x^2) + eps) * w * silu(z) per head.
    let recurrent = bf16_values(&bf16_bytes(&rng.values(rows * GDN_VALUE, 2.0)));
    let value_z = bf16_values(&bf16_bytes(&rng.values(rows * GDN_VALUE_Z, 2.0)));
    let weight = bf16_values(&bf16_bytes(&rng.values(128, 1.0)));
    put(&workspace, BufferSlot::GdnRecurrent, 0, &bf16_bytes(&recurrent));
    put(&workspace, BufferSlot::ValueZProjected, 0, &bf16_bytes(&value_z));
    let weight_buffer = upload(&rig.library, &bf16_bytes(&weight));
    let norm = GdnGatedRmsNorm::new(
        BufferSlot::GdnRecurrent,
        BufferSlot::ValueZProjected,
        weight_buffer.ptr(),
        BufferSlot::GdnNormalized,
    );
    assert!(rig.run(&norm, &quantum).is_some());
    rig.sync();
    let got = bf16_values(&get(&workspace, BufferSlot::GdnNormalized, 0, rows * GDN_VALUE * 2));
    let mut expected = Vec::new();
    for row in 0..rows {
        for head in 0..GDN_HEADS {
            let x = &recurrent[row * GDN_VALUE + head * 128..][..128];
            let z = &value_z[row * GDN_VALUE_Z + GDN_VALUE + head * 128..][..128];
            let inverse = 1.0 / (x.iter().map(|x| x * x).sum::<f32>() / 128.0 + 1e-6).sqrt();
            expected.extend((0..128).map(|i| x[i] * inverse * weight[i] * (z[i] / (1.0 + (-z[i]).exp()))));
        }
    }
    assert_close("the gated norm", &got, &expected, 0.02, 1e-3);
}
