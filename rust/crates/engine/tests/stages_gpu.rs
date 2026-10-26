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

#[path = "common/linear_gpu.rs"]
mod linear_gpu;

use std::cell::RefCell;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::slice;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::time::Duration;

use euhedral_core::control_plane::ControlPlaneLattice;
use euhedral_core::generics::{FrameSink, LatticeSource};
use euhedral_graph::{
    Completions, DEFAULT_RETRY_BACKSTOP, DeviceQueue, EdgeKind, GraphRuntime, Join, Lake, LaneId, LanePool, Latch,
    Marker, QuantumOutcome, RetryCounts, Stage, StageCx, StageError, StageGraph, StageGraphSpec, StageId, StageSpec,
    Submitted,
};
use euhedral_inference_cuda::{
    CudaError, CudaLane, CudaLibrary, DeviceBuffer, DevicePtr, GraphRecorder, PinnedPool, Submission, SubmitMode, ops,
};
use euhedral_inference_edrl::{LayerType, ModelMetadata, TensorDataType, WeightFormat, WeightLayout};
use euhedral_inference_engine::formats::{LinearFormat, LinearMode, Q3Embedding, ScratchRegion};
use euhedral_inference_engine::lattice_config;
use euhedral_inference_engine::model::{ModelConfig, WeightRef};
use euhedral_inference_engine::stages::attention::{AttentionCausal, KvAppend, KvSource, QkNormRope};
use euhedral_inference_engine::stages::elementwise::{ResidualAdd, Swiglu};
use euhedral_inference_engine::stages::embed::Embed;
use euhedral_inference_engine::stages::gdn::{GdnConvolution, GdnGatedRmsNorm, GdnProjectControl, GdnRecurrence};
use euhedral_inference_engine::stages::linear::{Linear, LinearRows};
use euhedral_inference_engine::stages::mtp::MtpStem;
use euhedral_inference_engine::stages::norm::{NormRows, ResidualRmsNorm, RmsNormUnitOffset};
use euhedral_inference_engine::stages::sampling::{Argmax, SelectedRows, SelectedToken, Selection};
use euhedral_inference_engine::stages::{
    DraftSeed, DraftTokenWrite, InputRecordRef, LaneTable, Launch, Quantum, QuantumError, QuantumKind, SequenceRefs,
};
use euhedral_inference_engine::state::attention::AttentionScratch;
use euhedral_inference_engine::state::gdn::GdnStates;
use euhedral_inference_engine::state::kv::{KvCache, KvRetained};
use euhedral_inference_engine::state::{SequenceDevice, SequenceState};
use euhedral_inference_engine::workspace::{BufferSlot, Workspace, WorkspaceRows};
use euhedral_inference_settings::{KvFormat, ModelSettings, Settings};

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

thread_local! {
    /// The lane of the rig made on this thread: the host copies of the helpers run on it.
    static LANE: RefCell<Option<Arc<CudaLane>>> = const { RefCell::new(None) };
}

/// Runs `copy` (a blocking host copy) on the rig's lane. Outside a submission such a copy runs on the legacy stream,
/// which does not order against the lane's work: a kernel could start before the copy's DMA completed, or a read
/// could see a kernel's results half written. In a submission the copy is queued on the lane and waits for it.
fn on_lane<T>(copy: impl FnOnce() -> T) -> T {
    let lane = LANE
        .with(|lane| lane.borrow().clone())
        .expect("a rig was made on this thread");
    let submission = Submission::begin(&lane, SubmitMode::default()).unwrap();
    let out = copy();
    drop(submission);
    out
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
    LANE.with(|slot| *slot.borrow_mut() = Some(Arc::clone(&lane)));
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
        sequence: &SequenceState,
        config: &Arc<ModelConfig>,
    ) -> Quantum {
        self.quantum_replaying(kind, rows, start, workspace, sequence, config, Vec::new())
    }

    /// As `quantum`, with the rows each GDN layer replays first (what the owner sets at admission).
    #[allow(clippy::too_many_arguments, reason = "a test helper over the fields of a quantum")]
    fn quantum_replaying(
        &self,
        kind: QuantumKind,
        rows: usize,
        start: u64,
        workspace: &Arc<Workspace>,
        sequence: &SequenceState,
        config: &Arc<ModelConfig>,
        replay: Vec<u32>,
    ) -> Quantum {
        let mode = SubmitMode {
            pdl: false,
            row_exact: kind == QuantumKind::Verify,
        };
        // The admission uploads the input record: the start position, after the token ids.
        let offset = InputRecordRef::position_offset(rows) as usize;
        put(workspace, BufferSlot::TokenIds, offset, &start.to_ne_bytes());
        Quantum::new(
            kind,
            rows,
            InputRecordRef::new(start),
            Arc::clone(workspace),
            SequenceRefs::new(Arc::clone(&sequence.device))
                .with_kv(sequence.kv.view())
                .with_replay_rows(replay),
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
                device: Arc::new(SequenceDevice { gdn, scratch }),
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
        .as_chunks::<2>()
        .0
        .iter()
        .map(|pair| from_bf16(u16::from_ne_bytes(*pair)))
        .collect()
}

/// Writes `bytes` at `offset` of the workspace slot `slot`.
fn put(workspace: &Workspace, slot: BufferSlot, offset: usize, bytes: &[u8]) {
    on_lane(|| workspace.buffer(slot).copy_from_host(offset, bytes)).unwrap();
}

/// Reads `bytes` bytes at `offset` of the workspace slot `slot`.
fn get(workspace: &Workspace, slot: BufferSlot, offset: usize, bytes: usize) -> Vec<u8> {
    let mut out = vec![0u8; bytes];
    on_lane(|| workspace.buffer(slot).copy_to_host(offset, &mut out)).unwrap();
    out
}

fn upload(library: &Arc<CudaLibrary>, bytes: &[u8]) -> DeviceBuffer {
    let buffer = DeviceBuffer::alloc(library, bytes.len()).unwrap();
    on_lane(|| buffer.copy_from_host(0, bytes)).unwrap();
    buffer
}

fn download(buffer: &DeviceBuffer) -> Vec<u8> {
    let mut out = vec![0u8; buffer.bytes()];
    on_lane(|| buffer.copy_to_host(0, &mut out)).unwrap();
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

    fn scratch_bytes(&self, _weight: &WeightRef, _rows: usize, _mode: LinearMode) -> usize {
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
    fn step(&self, sequence: &SequenceState, kind: QuantumKind, rows: &[GdnRow]) -> Vec<f32> {
        self.step_replaying(sequence, kind, rows, 0)
    }

    /// As `step`, replaying `replay` rows of the layer's checkpoint first.
    fn step_replaying(&self, sequence: &SequenceState, kind: QuantumKind, rows: &[GdnRow], replay: u32) -> Vec<f32> {
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
        let quantum = self.rig.quantum_replaying(
            kind,
            rows.len(),
            0,
            &self.workspace,
            sequence,
            &self.config,
            vec![replay],
        );
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
    fn state(&self, sequence: &SequenceState) -> (Vec<u8>, Vec<u8>) {
        let layer = sequence.device.gdn.layer(0).unwrap();
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
        gdn.step(&decoded, QuantumKind::Decode, slice::from_ref(row));
    }
    let mut decoded_outputs = Vec::new();
    for row in verified {
        decoded_outputs.push(gdn.step(&decoded, QuantumKind::Decode, slice::from_ref(row)));
    }
    // The state after the accepted three: the verification's row-exact rows are what decode would have computed.
    let verified_whole = gdn.sequence();
    for row in history {
        gdn.step(&verified_whole, QuantumKind::Decode, slice::from_ref(row));
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
        gdn.step(&accepted, QuantumKind::Decode, slice::from_ref(row));
    }
    let accepted_next = gdn.step(&accepted, QuantumKind::Decode, next);

    // The verification ran past the accepted rows; the next quantum replays them before its own.
    let replayed = gdn.sequence();
    for row in history {
        gdn.step(&replayed, QuantumKind::Decode, slice::from_ref(row));
    }
    gdn.step(&replayed, QuantumKind::Verify, verified);
    // The owner admits the next quantum with the three accepted rows to replay.
    let replayed_next = gdn.step_replaying(&replayed, QuantumKind::Decode, next, 3);
    assert_eq!(replayed_next, accepted_next, "the next row after a replay");
    assert_eq!(
        gdn.state(&replayed),
        gdn.state(&accepted),
        "the replayed state is the accepted rows' decode state, then the next row"
    );

    // Without the replay the state has run past the accepted rows.
    let unreplayed = gdn.sequence();
    for row in history {
        gdn.step(&unreplayed, QuantumKind::Decode, slice::from_ref(row));
    }
    gdn.step(&unreplayed, QuantumKind::Verify, verified);
    let unreplayed_next = gdn.step(&unreplayed, QuantumKind::Decode, next);
    assert_ne!(unreplayed_next, accepted_next);
    assert_ne!(gdn.state(&unreplayed).1, gdn.state(&accepted).1);

    // A replay of every row of the verification is its own checkpoint run again: all five rows, then the next.
    let full = gdn.sequence();
    for row in history {
        gdn.step(&full, QuantumKind::Decode, slice::from_ref(row));
    }
    gdn.step(&full, QuantumKind::Verify, verified);
    let full_next = gdn.step_replaying(&full, QuantumKind::Decode, next, 5);
    let all_decoded = gdn.sequence();
    for row in history.iter().chain(verified) {
        gdn.step(&all_decoded, QuantumKind::Decode, slice::from_ref(row));
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
            .as_chunks::<4>()
            .0
            .iter()
            .map(|word| f32::from_ne_bytes(*word))
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

// Attention.

/// Width of a projected query and key row, and of a gate and value row: 24 query heads and 4 key heads of 256.
const ATTENTION_ROW: usize = 7168;
const ATTENTION_QUERY: usize = 6144;

/// The attention layer's stages over the cache's layer 0, from the workspace slots a layer of the plan uses.
fn attention_append() -> KvAppend {
    KvAppend::new(
        KvSource::Base,
        0,
        BufferSlot::AttentionQkNormalized,
        BufferSlot::ValueZProjected,
    )
}

fn attention_causal() -> AttentionCausal {
    AttentionCausal::new(
        KvSource::Base,
        0,
        BufferSlot::AttentionQkNormalized,
        BufferSlot::ValueZProjected,
        BufferSlot::AttentionContext,
    )
}

/// Appends `rows` rows of random keys and values at `start` (a prefill chunk), in chunks the workspace holds, and
/// returns the rows' query/key and gate/value data.
fn fill_cache(
    rig: &Rig,
    config: &Arc<ModelConfig>,
    workspace: &Arc<Workspace>,
    sequence: &SequenceState,
    rng: &mut Rng,
    start: u64,
    rows: u64,
) -> (Vec<f32>, Vec<f32>) {
    let chunk = workspace.rows().rows;
    let (mut query_key, mut gate_value) = (Vec::new(), Vec::new());
    let mut done = 0;
    while done < rows {
        let n = chunk.min(rows - done);
        let qk = bf16_values(&bf16_bytes(&rng.values(n as usize * ATTENTION_ROW, 1.0)));
        let gv = bf16_values(&bf16_bytes(&rng.values(n as usize * ATTENTION_ROW, 1.0)));
        put(workspace, BufferSlot::AttentionQkNormalized, 0, &bf16_bytes(&qk));
        put(workspace, BufferSlot::ValueZProjected, 0, &bf16_bytes(&gv));
        let quantum = rig.quantum(
            QuantumKind::Prefill,
            n as usize,
            start + done,
            workspace,
            sequence,
            config,
        );
        assert!(rig.run(&attention_append(), &quantum).is_some());
        rig.sync();
        query_key.extend(qk);
        gate_value.extend(gv);
        done += n;
    }
    (query_key, gate_value)
}

/// Checks that the multi-row attention of `m` rows at `start`, on the verification's row-exact route and on the
/// draft route (the same twins without the row-exact mode), equals `m` one-row attentions bit for bit.
fn check_row_twins(format: KvFormat, start: u64, m: usize) {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 1, format, start + m as u64);
    let sequence = Arc::new(sequence);
    let workspace = rig.workspace(&config, 512, 8, &[]);
    let mut rng = Rng(0xdead_beef_cafe_f00d ^ start);
    fill_cache(&rig, &config, &workspace, &sequence, &mut rng, 0, start);

    // The rows of the step: appended once, then attended to by the twins and by the singles.
    let qk = bf16_values(&bf16_bytes(&rng.values(m * ATTENTION_ROW, 1.0)));
    let gv = bf16_values(&bf16_bytes(&rng.values(m * ATTENTION_ROW, 1.0)));
    put(&workspace, BufferSlot::AttentionQkNormalized, 0, &bf16_bytes(&qk));
    put(&workspace, BufferSlot::ValueZProjected, 0, &bf16_bytes(&gv));
    let verify = rig.quantum(QuantumKind::Verify, m, start, &workspace, &sequence, &config);
    assert!(rig.run(&attention_append(), &verify).is_some());
    assert!(rig.run(&attention_causal(), &verify).is_some());
    rig.sync();
    let bytes = m * ATTENTION_QUERY * 2;
    let twins = get(&workspace, BufferSlot::AttentionContext, 0, bytes);
    assert!(
        bf16_values(&twins).iter().any(|value| *value != 0.0),
        "the attention wrote nothing"
    );

    let draft = rig.quantum(QuantumKind::Draft, m, start, &workspace, &sequence, &config);
    assert!(rig.run(&attention_causal(), &draft).is_some());
    rig.sync();
    let draft_twins = get(&workspace, BufferSlot::AttentionContext, 0, bytes);
    assert!(
        draft_twins == twins,
        "the draft route's twins differ from the verification's"
    );

    let single = AttentionCausal::new(
        KvSource::Base,
        0,
        BufferSlot::QkProjected,
        BufferSlot::GateUp,
        BufferSlot::AttentionContext,
    );
    for row in 0..m {
        put(
            &workspace,
            BufferSlot::QkProjected,
            0,
            &bf16_bytes(&qk[row * ATTENTION_ROW..(row + 1) * ATTENTION_ROW]),
        );
        put(
            &workspace,
            BufferSlot::GateUp,
            0,
            &bf16_bytes(&gv[row * ATTENTION_ROW..(row + 1) * ATTENTION_ROW]),
        );
        let decode = rig.quantum(
            QuantumKind::Decode,
            1,
            start + row as u64,
            &workspace,
            &sequence,
            &config,
        );
        assert!(rig.run(&single, &decode).is_some());
        rig.sync();
        let one = get(&workspace, BufferSlot::AttentionContext, 0, ATTENTION_QUERY * 2);
        assert!(
            twins[row * ATTENTION_QUERY * 2..(row + 1) * ATTENTION_QUERY * 2] == one[..],
            "{format:?} {m} rows at {start}: row {row} differs from its one-row attention"
        );
    }
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn attention_row_twins_equal_single_rows() {
    for format in [KvFormat::Nvfp4, KvFormat::Bf16] {
        // Below 2048 keys, the per-head kernel; above, the three-warp one; and a step that crosses 2048.
        check_row_twins(format, 1000, 5);
        check_row_twins(format, 2100, 8);
        check_row_twins(format, 2044, 8);
        check_row_twins(format, 300, 2);
    }
}

/// The reference of one decode row's attention over BF16 keys and values: per query head the softmax of
/// `q . k / 16` over the keys, the weighted values, and the output gate `1 / (1 + exp(-gate))`.
fn reference_decode(query: &[f32], gate: &[f32], keys: &[Vec<f32>], values: &[Vec<f32>]) -> Vec<f32> {
    let group = 6; // 24 query heads over 4 key heads
    let mut out = Vec::new();
    for head in 0..24 {
        let kv_head = head / group;
        let q = &query[head * 256..(head + 1) * 256];
        let scores: Vec<f64> = keys
            .iter()
            .map(|key| {
                q.iter()
                    .zip(&key[kv_head * 256..(kv_head + 1) * 256])
                    .map(|(q, k)| f64::from(*q) * f64::from(*k))
                    .sum::<f64>()
                    / 16.0
            })
            .collect();
        let maximum = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let weights: Vec<f64> = scores.iter().map(|score| (score - maximum).exp()).collect();
        let total: f64 = weights.iter().sum();
        for d in 0..256 {
            let context: f64 = weights
                .iter()
                .zip(values)
                .map(|(w, value)| w * f64::from(value[kv_head * 256 + d]))
                .sum::<f64>()
                / total;
            let g = f64::from(gate[head * 256 + d]);
            out.push((context / (1.0 + (-g).exp())) as f32);
        }
    }
    out
}

/// The `rows`-row cache `format` after a prefill of `n` rows and one decode row at `n`: returns the decode row's
/// attention output, the cache (to read its pages) and the data.
struct ReadBack {
    sequence: Arc<SequenceState>,
    output: Vec<f32>,
    keys: Vec<Vec<f32>>,
    values: Vec<Vec<f32>>,
    gate: Vec<f32>,
    query: Vec<f32>,
}

fn append_then_decode(rig: &Rig, format: KvFormat, n: u64) -> ReadBack {
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 1, format, n + 1);
    let sequence = Arc::new(sequence);
    let workspace = rig.workspace(&config, 512, 8, &[]);
    let mut rng = Rng(0x0123_4567_89ab_cdef);
    let (qk, gv) = fill_cache(rig, &config, &workspace, &sequence, &mut rng, 0, n);

    // One decode row after the prompt: appended, then attended.
    let step_qk = bf16_values(&bf16_bytes(&rng.values(ATTENTION_ROW, 1.0)));
    let step_gv = bf16_values(&bf16_bytes(&rng.values(ATTENTION_ROW, 1.0)));
    put(&workspace, BufferSlot::AttentionQkNormalized, 0, &bf16_bytes(&step_qk));
    put(&workspace, BufferSlot::ValueZProjected, 0, &bf16_bytes(&step_gv));
    let decode = rig.quantum(QuantumKind::Decode, 1, n, &workspace, &sequence, &config);
    assert!(rig.run(&attention_append(), &decode).is_some());
    assert!(rig.run(&attention_causal(), &decode).is_some());
    rig.sync();
    let output = bf16_values(&get(&workspace, BufferSlot::AttentionContext, 0, ATTENTION_QUERY * 2));

    // The keys and values of every position, 1024 wide each: the key part follows the query part of a row.
    let rows = |data: &[f32], extra: &[f32]| -> Vec<Vec<f32>> {
        data.chunks(ATTENTION_ROW)
            .chain(extra.chunks(ATTENTION_ROW))
            .map(|row| row[ATTENTION_QUERY..].to_vec())
            .collect()
    };
    ReadBack {
        keys: rows(&qk, &step_qk),
        values: rows(&gv, &step_gv),
        gate: step_gv[..ATTENTION_QUERY].to_vec(),
        query: step_qk[..ATTENTION_QUERY].to_vec(),
        output,
        sequence,
    }
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn kv_append_then_decode_reads_back() {
    let Some(rig) = rig() else { return };
    let n = 300u64;

    // BF16 pages hold the rows exactly as appended: 512 bytes per head row, token-major.
    let bf16 = append_then_decode(&rig, KvFormat::Bf16, n);
    let kv = &bf16.sequence.kv;
    let plane = kv.plane_page_bytes() as usize;
    assert_eq!(plane, 256 * 4 * 512);
    let pages: Vec<DevicePtr> = kv.pages(0).collect();
    assert_eq!(pages.len(), 2, "301 rows take two pages");
    for (page, address) in pages.iter().enumerate() {
        let mut bytes = vec![0u8; 2 * plane];
        rig.submit(|submission| ops::copy_device_to_host(submission, &mut bytes, *address));
        let tokens = if page == 0 { 256 } else { 45 };
        for token in 0..tokens {
            let position = page * 256 + token;
            for head in 0..4 {
                let at = (token * 4 + head) * 512;
                let key = &bf16.keys[position][head * 256..(head + 1) * 256];
                let value = &bf16.values[position][head * 256..(head + 1) * 256];
                assert!(bytes[at..at + 512] == bf16_bytes(key)[..], "key {position} head {head}");
                assert!(
                    bytes[plane + at..plane + at + 512] == bf16_bytes(value)[..],
                    "value {position} head {head}"
                );
            }
        }
    }
    // The decode row reads them back: its attention is the reference over the appended rows.
    let expected = reference_decode(&bf16.query, &bf16.gate, &bf16.keys, &bf16.values);
    assert_close("decode attention over BF16 pages", &bf16.output, &expected, 0.02, 2e-3);

    // The NVFP4 cache quantizes the same rows; the decode row reads them back within the quantization's error.
    let nvfp4 = append_then_decode(&rig, KvFormat::Nvfp4, n);
    let error = nvfp4
        .output
        .iter()
        .zip(&expected)
        .map(|(a, e)| f64::from(a - e).powi(2))
        .sum::<f64>()
        .sqrt();
    let scale = expected.iter().map(|e| f64::from(*e).powi(2)).sum::<f64>().sqrt();
    println!("decode attention over NVFP4 pages: relative error {:.4}", error / scale);
    assert!(
        error / scale < 0.15,
        "NVFP4 decode attention is {:.3} off its reference",
        error / scale
    );
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn qk_norm_rope_rows_equal_single_rows() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 0, KvFormat::Nvfp4, 0);
    let sequence = Arc::new(sequence);
    let workspace = rig.workspace(&config, 8, 8, &[]);
    let mut rng = Rng(0x0f0f_0f0f_1234_9876);
    let (m, start) = (5usize, 40_000u64);
    let query_norm = upload(&rig.library, &bf16_bytes(&rng.values(256, 0.5)));
    let key_norm = upload(&rig.library, &bf16_bytes(&rng.values(256, 0.5)));
    let rows = bf16_values(&bf16_bytes(&rng.values(m * ATTENTION_ROW, 2.0)));
    let stage = |input, output| QkNormRope::new(input, output, query_norm.ptr(), key_norm.ptr());
    put(&workspace, BufferSlot::QkProjected, 0, &bf16_bytes(&rows));
    let verify = rig.quantum(QuantumKind::Verify, m, start, &workspace, &sequence, &config);
    assert!(
        rig.run(
            &stage(BufferSlot::QkProjected, BufferSlot::AttentionQkNormalized),
            &verify
        )
        .is_some()
    );
    rig.sync();
    let together = get(&workspace, BufferSlot::AttentionQkNormalized, 0, m * ATTENTION_ROW * 2);
    assert!(
        bf16_values(&together) != rows,
        "the norm and the rotation change the rows"
    );
    for row in 0..m {
        put(
            &workspace,
            BufferSlot::GateUp,
            0,
            &bf16_bytes(&rows[row * ATTENTION_ROW..(row + 1) * ATTENTION_ROW]),
        );
        let decode = rig.quantum(
            QuantumKind::Decode,
            1,
            start + row as u64,
            &workspace,
            &sequence,
            &config,
        );
        assert!(
            rig.run(&stage(BufferSlot::GateUp, BufferSlot::AttentionContext), &decode)
                .is_some()
        );
        rig.sync();
        let one = get(&workspace, BufferSlot::AttentionContext, 0, ATTENTION_ROW * 2);
        assert!(
            together[row * ATTENTION_ROW * 2..(row + 1) * ATTENTION_ROW * 2] == one[..],
            "row {row} of the verification differs from its one-row launch"
        );
    }
}

// Sampling.

const VOCABULARY: usize = 248_320;

/// The token the host's argmax chooses over a BF16 row: the lowest id among the maxima, never NaN or negative
/// infinity; `None` when no entry is selectable.
fn host_argmax(row: &[f32]) -> Option<u32> {
    let mut best: Option<(usize, f32)> = None;
    for (id, value) in row.iter().enumerate() {
        if value.is_nan() || *value == f32::NEG_INFINITY {
            continue;
        }
        if best.is_none_or(|(_, maximum)| *value > maximum) {
            best = Some((id, *value));
        }
    }
    best.map(|(id, _)| id as u32)
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn argmax_lowest_id_tie_break() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 0, KvFormat::Nvfp4, 0);
    let sequence = Arc::new(sequence);
    let workspace = rig.workspace(&config, 8, 8, &[]);
    let selection = Arc::new(Selection::new(&rig.library, 8).unwrap());
    let mut rng = Rng(0xabcd_ef01_2345_6789);

    // Five logits rows: a unique maximum, three equal maxima, nothing selectable, NaNs around one value, and a
    // maximum at the last id.
    let mut rows: Vec<Vec<f32>> = (0..5).map(|_| rng.values(VOCABULARY, 1.0)).collect();
    rows[0][12_345] = 5.0;
    for id in [200_000, 5000, 77] {
        rows[1][id] = 6.0;
    }
    rows[2] = vec![f32::NEG_INFINITY; VOCABULARY];
    rows[3] = vec![f32::NAN; VOCABULARY];
    rows[3][9] = 1.0;
    rows[4][VOCABULARY - 1] = 7.0;
    for (index, row) in rows.iter().enumerate() {
        put(&workspace, BufferSlot::Logits, index * VOCABULARY * 2, &bf16_bytes(row));
    }
    let expected: Vec<Option<u32>> = rows
        .iter()
        .map(|row| host_argmax(&bf16_values(&bf16_bytes(row))))
        .collect();
    assert_eq!(
        expected,
        [Some(12_345), Some(77), None, Some(9), Some(VOCABULARY as u32 - 1)]
    );

    // A verification selects every row, and the stage's marker is signalled for the continuation that follows it.
    let every = Argmax::new(BufferSlot::Logits, VOCABULARY, SelectedRows::Each);
    let verify = rig
        .quantum(QuantumKind::Verify, 5, 0, &workspace, &sequence, &config)
        .with_selection(Arc::clone(&selection));
    let marker = rig.run(&every, &verify).expect("a marker for the completion edge");
    rig.sync();
    assert!(
        rig.lane.completed() >= marker.value,
        "the lane's completion word reaches the argmax stage's marker"
    );
    let selected: Vec<Option<u32>> = (0..5).map(|row| selection.token(row)).collect();
    assert_eq!(selected, expected);
    assert_eq!(selection.key(2), Some(0));
    assert_eq!(selection.key(1).unwrap() & 0xFFFF_FFFF, 0xFFFF_FFFF - 77);

    // A prompt chunk or a decode row selects the last logits row only.
    let last = Argmax::new(BufferSlot::Logits, VOCABULARY, SelectedRows::Last);
    let prefill = rig
        .quantum(QuantumKind::Prefill, 5, 0, &workspace, &sequence, &config)
        .with_logits_rows(5)
        .unwrap()
        .with_selection(Arc::clone(&selection));
    assert!(rig.run(&last, &prefill).is_some());
    rig.sync();
    assert_eq!(selection.token(0), Some(VOCABULARY as u32 - 1));

    // A chunk that is not the last scores nothing: no work and no marker.
    let silent = rig
        .quantum(QuantumKind::Prefill, 5, 0, &workspace, &sequence, &config)
        .with_logits_rows(0)
        .unwrap()
        .with_selection(Arc::clone(&selection));
    assert!(rig.run(&last, &silent).is_none());

    // The drafted token passes to the step's tokens: the row itself, or through the draft head's ids.
    let draft = sequence.device.scratch.draft().unwrap();
    let ids = upload(&rig.library, &vec![0u8; VOCABULARY * 4]);
    on_lane(|| ids.copy_from_host(12_345 * 4, &31_337i32.to_ne_bytes())).unwrap();
    let pass = SelectedToken;
    for (slot, ids_ptr, expected) in [(2usize, DevicePtr::NULL, 12_345i32), (3, ids.ptr(), 31_337)] {
        let drafting = rig
            .quantum(QuantumKind::Draft, 1, 0, &workspace, &sequence, &config)
            .with_selection(Arc::clone(&selection))
            .with_draft_token_write(DraftTokenWrite { slot, ids: ids_ptr });
        assert!(rig.run(&last, &drafting).is_some());
        assert!(rig.run(&pass, &drafting).is_some());
        rig.sync();
        let tokens = download(&draft.tokens);
        let at = slot * 4;
        assert_eq!(
            i32::from_ne_bytes(tokens[at..at + 4].try_into().unwrap()),
            expected,
            "slot {slot}"
        );
    }
    // Nothing to pass without a draft token write.
    let plain = rig
        .quantum(QuantumKind::Decode, 1, 0, &workspace, &sequence, &config)
        .with_selection(Arc::clone(&selection));
    assert!(rig.run(&pass, &plain).is_none());
}

// The MTP stem.

#[test]
#[ignore = "needs a GPU and the native library"]
fn mtp_stem_concatenates_e_h() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 0, KvFormat::Nvfp4, 0);
    let sequence = Arc::new(sequence);
    let workspace = rig.workspace(&config, 8, 8, &[]);
    let (rows, hidden) = (4usize, 5120usize);
    let mut rng = Rng(0x7777_1111_2222_3333);
    let embedding = bf16_values(&bf16_bytes(&rng.values(rows * hidden, 3.0)));
    let seed = bf16_values(&bf16_bytes(&rng.values(rows * hidden, 0.5)));
    let embedding_norm = bf16_values(&bf16_bytes(&rng.values(hidden, 0.5)));
    let hidden_norm = bf16_values(&bf16_bytes(&rng.values(hidden, 0.5)));
    put(&workspace, BufferSlot::HiddenState, 0, &bf16_bytes(&embedding));
    let embedding_norm_buffer = upload(&rig.library, &bf16_bytes(&embedding_norm));
    let hidden_norm_buffer = upload(&rig.library, &bf16_bytes(&hidden_norm));
    let seed_buffer = upload(&rig.library, &bf16_bytes(&seed));
    let stem = MtpStem::new(
        BufferSlot::HiddenState,
        embedding_norm_buffer.ptr(),
        hidden_norm_buffer.ptr(),
        BufferSlot::MtpNormed,
        BufferSlot::MtpPacked,
    );

    // The (1 + w) norm of one row, on the host.
    let norm = |row: &[f32], weight: &[f32]| -> Vec<f32> {
        let scale = 1.0 / (row.iter().map(|x| x * x).sum::<f32>() / hidden as f32 + 1e-6).sqrt();
        row.iter().zip(weight).map(|(x, w)| x * scale * (1.0 + w)).collect()
    };
    let expected: Vec<f32> = (0..rows)
        .flat_map(|row| {
            let mut packed = norm(&embedding[row * hidden..(row + 1) * hidden], &embedding_norm);
            packed.extend(norm(&seed[row * hidden..(row + 1) * hidden], &hidden_norm));
            packed
        })
        .collect();

    // Seeded from rows at an address (a recursive draft's hidden).
    let draft = rig
        .quantum(QuantumKind::Draft, rows, 0, &workspace, &sequence, &config)
        .with_draft_seed(DraftSeed::Address(seed_buffer.ptr()));
    assert!(rig.run(&stem, &draft).is_some());
    rig.sync();
    let packed = get(&workspace, BufferSlot::MtpPacked, 0, rows * 2 * hidden * 2);
    assert_close("the packed rows", &bf16_values(&packed), &expected, 0.02, 1e-3);

    // The halves are the single norm launches, bit for bit: [norm(e); norm(h)] per row.
    let single = |input: &DeviceBuffer, weight: &DeviceBuffer| -> Vec<u8> {
        let output = DeviceBuffer::alloc(&rig.library, rows * hidden * 2).unwrap();
        rig.submit(|submission| {
            ops::rms_norm_unit_offset_bf16(
                submission,
                input.ptr(),
                weight.ptr(),
                output.ptr(),
                rows as u32,
                hidden as u32,
                1e-6,
            )
        });
        download(&output)
    };
    let embedding_buffer = upload(&rig.library, &bf16_bytes(&embedding));
    let (e_half, h_half) = (
        single(&embedding_buffer, &embedding_norm_buffer),
        single(&seed_buffer, &hidden_norm_buffer),
    );
    for row in 0..rows {
        let at = row * 2 * hidden * 2;
        assert!(
            packed[at..at + hidden * 2] == e_half[row * hidden * 2..(row + 1) * hidden * 2],
            "e of row {row}"
        );
        assert!(
            packed[at + hidden * 2..at + 2 * hidden * 2] == h_half[row * hidden * 2..(row + 1) * hidden * 2],
            "h of row {row}"
        );
    }

    // Seeded from the sequence's draft seed rows, from row 2 (a catch-up after the quantum that wrote them).
    let seed_rows = sequence.device.scratch.draft().unwrap().seed_rows.ptr();
    rig.submit(|submission| {
        ops::copy_device_to_device(
            submission,
            seed_rows.offset((2 * hidden * 2) as u64),
            seed_buffer.ptr(),
            (rows * hidden * 2) as u64,
        )
    });
    let catch_up = rig
        .quantum(QuantumKind::Draft, rows, 0, &workspace, &sequence, &config)
        .with_draft_seed(DraftSeed::SeedRow(2));
    assert!(rig.run(&stem, &catch_up).is_some());
    rig.sync();
    assert!(get(&workspace, BufferSlot::MtpPacked, 0, rows * 2 * hidden * 2) == packed);

    // A stem with no rows to seed it, or seeded past the rows the sequence holds, is refused.
    let unseeded = rig.quantum(QuantumKind::Draft, rows, 0, &workspace, &sequence, &config);
    let cx = StageCx {
        stage: StageId(0),
        lane: Some(LaneId(0)),
        waits: &[],
        lanes: &rig.pool,
    };
    assert!(stem.run(&unseeded, &cx).is_err());
    let past = rig
        .quantum(QuantumKind::Draft, rows, 0, &workspace, &sequence, &config)
        .with_draft_seed(DraftSeed::SeedRow(14));
    assert!(stem.run(&past, &cx).is_err());
}

// The stages in a graph.

/// A host stage behind the argmax by a completion edge: it reads the selection the argmax wrote.
struct ReadSelection {
    seen: Arc<Join<u64>>,
}

impl Stage<Quantum> for ReadSelection {
    fn run(&self, quantum: &Quantum, _cx: &StageCx<'_>) -> Result<Submitted, StageError> {
        let selection = quantum.selection().expect("the quantum selects");
        self.seen.arrive(0, selection.key(0).unwrap());
        Ok(Submitted::done())
    }
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_completion_edge_after_the_argmax_finds_its_keys_on_the_host() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 0, KvFormat::Nvfp4, 0);
    let sequence = Arc::new(sequence);
    let workspace = rig.workspace(&config, 8, 8, &[]);
    let selection = Arc::new(Selection::new(&rig.library, 8).unwrap());
    let mut row = Rng(99).values(VOCABULARY, 1.0);
    row[4242] = 9.0;
    put(&workspace, BufferSlot::Logits, 0, &bf16_bytes(&row));
    let quantum = rig
        .quantum(QuantumKind::Decode, 1, 0, &workspace, &sequence, &config)
        .with_selection(Arc::clone(&selection));

    // The argmax runs on a lane of the graph, and the host stage after it by a completion edge.
    let settings = Settings::with_model(ModelSettings::default());
    let lattice = ControlPlaneLattice::new(lattice_config(&settings)).unwrap();
    let lake = Lake::new(
        Arc::clone(&lattice),
        NonZeroUsize::new(2).unwrap(),
        Duration::from_secs(1),
    );
    let lanes = Arc::new(LanePool::new(vec![Arc::clone(&rig.lane) as Arc<dyn DeviceQueue>]));
    let completions = Completions::new(Arc::clone(&lanes));
    lattice
        .add_upstream(Arc::clone(&completions) as Arc<dyn LatticeSource>)
        .unwrap();
    let (done, finished) = Latch::new();
    // Part 0 is the key the host stage read, part 1 the retirement of the quantum.
    let join = Join::new(2, move |parts: Vec<Option<u64>>| done.set(parts));
    let retired = Arc::clone(&join);
    let mut spec = StageGraphSpec::new();
    let argmax = spec.stage(StageSpec {
        name: "argmax",
        ordered: None,
        lanes: vec![LaneId(0)],
    });
    let read = spec.stage(StageSpec {
        name: "read",
        ordered: None,
        lanes: vec![],
    });
    spec.edge(argmax, read, EdgeKind::Completion);
    let graph = StageGraph::new(
        spec,
        vec![
            Arc::new(Argmax::new(BufferSlot::Logits, VOCABULARY, SelectedRows::Last)),
            Arc::new(ReadSelection { seen: join }),
        ],
        GraphRuntime {
            stages: Arc::clone(&lake) as Arc<dyn FrameSink>,
            completions,
            lanes,
            retries: Arc::new(RetryCounts::default()),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        },
        move |_quantum, outcome| retired.arrive(1, u64::from(outcome == QuantumOutcome::Completed)),
    )
    .unwrap();
    let unit = lake.admit().unwrap();
    let _ = graph.start(quantum, Arc::new(AtomicBool::new(false)), Vec::new());
    let parts = finished.wait();
    drop(unit);
    lake.complete_gracefully();
    lattice.close();
    assert_eq!(parts[1], Some(1), "the quantum completed");
    assert_eq!(parts[0].map(|key| key & 0xFFFF_FFFF), Some(0xFFFF_FFFF - 4242));
}

// Independent references for the GDN convolution and recurrence and for the Q/K norm with RoPE, at the model's
// dimensions. The tolerances are those of the Java operation tests (`LayerGpuOperationsIntegrationTest`,
// `AttentionCudaIntegrationTest`): 0.004 on the convolution (its state exactly), 0.008 on the recurrence output and
// 1e-4 on its state, 0.012 on the rotated rows.

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

/// The bf16 values of `values` as the device sees them.
fn rounded(values: &[f32]) -> Vec<f32> {
    bf16_values(&bf16_bytes(values))
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn gdn_convolution_matches_the_host_reference_and_carries_the_tail() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 0, KvFormat::Nvfp4, 0);
    let workspace = rig.workspace(&config, 8, 8, &[]);
    let (rows, kernel) = (3usize, config.linear_conv_kernel_dim);
    let (qk_width, value_width, channels) = (
        config.gdn_query_key_width(),
        config.gdn_value_width(),
        config.gdn_convolution_channels(),
    );
    let mut rng = Rng(0x5eed_0001);
    let query_key = rounded(&rng.values(rows * qk_width, 0.25));
    let value_z = rounded(&rng.values(rows * 2 * value_width, 0.25));
    let weights = rounded(&rng.values(kernel * channels, 0.2));
    let initial = rounded(&rng.values(channels * (kernel - 1), 0.25));
    put(&workspace, BufferSlot::QkProjected, 0, &bf16_bytes(&query_key));
    put(&workspace, BufferSlot::ValueZProjected, 0, &bf16_bytes(&value_z));
    let weight_buffer = upload(&rig.library, &bf16_bytes(&weights));
    let state = sequence.device.gdn.layer(0).unwrap().convolution();
    on_lane(|| state.copy_from_host(0, &bf16_bytes(&initial))).unwrap();
    let stage = GdnConvolution::new(
        0,
        BufferSlot::QkProjected,
        BufferSlot::ValueZProjected,
        weight_buffer.ptr(),
        BufferSlot::GdnConvolved,
    );
    let quantum = rig.quantum(QuantumKind::Prefill, rows, 0, &workspace, &sequence, &config);
    assert!(rig.run(&stage, &quantum).is_some());
    rig.sync();

    let input = |row: usize, channel: usize| -> f32 {
        if channel < qk_width {
            query_key[row * qk_width + channel]
        } else {
            value_z[row * 2 * value_width + channel - qk_width]
        }
    };
    let mut expected = vec![0f32; rows * channels];
    let mut tail = vec![0f32; channels * (kernel - 1)];
    for channel in 0..channels {
        for row in 0..rows {
            let mut sum = 0f64;
            for tap in 0..kernel {
                let source = (row + tap) as isize - (kernel as isize - 1);
                let value = if source < 0 {
                    initial[channel * (kernel - 1) + (source + kernel as isize - 1) as usize]
                } else {
                    input(source as usize, channel)
                };
                sum += f64::from(value) * f64::from(weights[tap * channels + channel]);
            }
            expected[row * channels + channel] = silu(sum) as f32;
        }
        for history in 0..kernel - 1 {
            let source = (rows + history) as isize - (kernel as isize - 1);
            tail[channel * (kernel - 1) + history] = if source < 0 {
                initial[channel * (kernel - 1) + (source + kernel as isize - 1) as usize]
            } else {
                input(source as usize, channel)
            };
        }
    }
    let got = bf16_values(&get(&workspace, BufferSlot::GdnConvolved, 0, rows * channels * 2));
    assert_close("the convolution", &got, &expected, 0.0, 0.004);
    assert_eq!(
        bf16_values(&download(state)),
        tail,
        "the state carries the last rows of the input"
    );
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn gdn_recurrence_matches_the_host_reference() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 0, KvFormat::Nvfp4, 0);
    let workspace = rig.workspace(&config, 8, 8, &[]);
    let rows = 2usize;
    let (key_heads, value_heads, dim) = (
        config.linear_num_key_heads,
        config.linear_num_value_heads,
        config.linear_key_head_dim,
    );
    let (qk_width, value_width) = (config.gdn_query_key_width(), config.gdn_value_width());
    let channels = config.gdn_convolution_channels();
    let mut rng = Rng(0x5eed_0002);
    let convolved = rounded(&rng.values(rows * channels, 1.0));
    let alpha: Vec<f32> = (0..rows * value_heads)
        .map(|_| 0.7 + 0.29 * (rng.next() * 0.5 + 0.5))
        .collect();
    let beta: Vec<f32> = (0..rows * value_heads)
        .map(|_| 0.2 + 0.6 * (rng.next() * 0.5 + 0.5))
        .collect();
    let initial = rng.values(value_heads * dim * dim, 0.05);
    put(&workspace, BufferSlot::GdnConvolved, 0, &bf16_bytes(&convolved));
    put(&workspace, BufferSlot::GdnAlpha, 0, &f32_bytes(&alpha));
    put(&workspace, BufferSlot::GdnBeta, 0, &f32_bytes(&beta));
    let state = sequence.device.gdn.layer(0).unwrap().recurrent();
    on_lane(|| state.copy_from_host(0, &f32_bytes(&initial))).unwrap();
    let stage = GdnRecurrence::new(
        0,
        BufferSlot::GdnConvolved,
        BufferSlot::GdnAlpha,
        BufferSlot::GdnBeta,
        BufferSlot::GdnRecurrent,
    );
    let quantum = rig.quantum(QuantumKind::Prefill, rows, 0, &workspace, &sequence, &config);
    assert!(rig.run(&stage, &quantum).is_some());
    rig.sync();

    let scale = 1.0 / (dim as f64).sqrt();
    let normalized = |at: usize| -> Vec<f64> {
        let row = &convolved[at..at + dim];
        let inverse = 1.0 / (row.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() + 1e-6).sqrt();
        row.iter().map(|x| f64::from(*x) * inverse).collect()
    };
    let mut state_ref: Vec<f64> = initial.iter().map(|x| f64::from(*x)).collect();
    let mut expected = vec![0f32; rows * value_width];
    for row in 0..rows {
        for head in 0..value_heads {
            let key_head = head / (value_heads / key_heads);
            let query = normalized(row * channels + key_head * dim);
            let key = normalized(row * channels + key_heads * dim + key_head * dim);
            let a = f64::from(alpha[row * value_heads + head]);
            let b = f64::from(beta[row * value_heads + head]);
            for dv in 0..dim {
                let at = (head * dim + dv) * dim;
                let dot: f64 = (0..dim).map(|k| state_ref[at + k] * key[k]).sum();
                let value = f64::from(convolved[row * channels + qk_width + head * dim + dv]);
                let delta = b * (value - a * dot);
                let mut out = 0.0;
                for k in 0..dim {
                    state_ref[at + k] = a * state_ref[at + k] + delta * key[k];
                    out += state_ref[at + k] * query[k];
                }
                expected[row * value_width + head * dim + dv] = (out * scale) as f32;
            }
        }
    }
    let got = bf16_values(&get(&workspace, BufferSlot::GdnRecurrent, 0, rows * value_width * 2));
    assert_close("the recurrence output", &got, &expected, 0.0, 0.008);
    let got_state: Vec<f32> = download(state)
        .as_chunks::<4>()
        .0
        .iter()
        .map(|word| f32::from_ne_bytes(*word))
        .collect();
    let want_state: Vec<f32> = state_ref.iter().map(|x| *x as f32).collect();
    assert_close("the recurrent state", &got_state, &want_state, 0.0, 1e-4);
}

#[test]
#[ignore = "needs a GPU and the native library"]
fn qk_norm_rope_matches_the_host_reference() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 0, KvFormat::Nvfp4, 0);
    let workspace = rig.workspace(&config, 8, 8, &[]);
    let (rows, start) = (3usize, 1234u64);
    let (query_heads, key_heads, head_dim) = (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.attention_head_dim,
    );
    // Everything below comes from the config: the partial rotary dimension, the theta and the head counts.
    let rotary = (head_dim as f64 * config.partial_rotary_factor + 0.5).floor() as usize;
    let width = (query_heads + key_heads) * head_dim;
    let mut rng = Rng(0x5eed_0003);
    let input = rounded(&rng.values(rows * width, 0.6));
    let query_norm = rounded(&(0..head_dim).map(|i| 0.02 * (i % 9) as f32 - 0.08).collect::<Vec<_>>());
    let key_norm = rounded(
        &(0..head_dim)
            .map(|i| -0.015 * (i % 9) as f32 + 0.06)
            .collect::<Vec<_>>(),
    );
    put(&workspace, BufferSlot::QkProjected, 0, &bf16_bytes(&input));
    let query_buffer = upload(&rig.library, &bf16_bytes(&query_norm));
    let key_buffer = upload(&rig.library, &bf16_bytes(&key_norm));
    let stage = QkNormRope::new(
        BufferSlot::QkProjected,
        BufferSlot::AttentionQkNormalized,
        query_buffer.ptr(),
        key_buffer.ptr(),
    );
    let quantum = rig.quantum(QuantumKind::Prefill, rows, start, &workspace, &sequence, &config);
    assert!(rig.run(&stage, &quantum).is_some());
    rig.sync();

    let mut expected = vec![0f32; rows * width];
    for row in 0..rows {
        for head in 0..query_heads + key_heads {
            let at = row * width + head * head_dim;
            let weights = if head < query_heads { &query_norm } else { &key_norm };
            let squares: f64 = input[at..at + head_dim].iter().map(|x| f64::from(*x).powi(2)).sum();
            let inverse = 1.0 / (squares / head_dim as f64 + config.rms_norm_epsilon).sqrt();
            let normalized: Vec<f64> = (0..head_dim)
                .map(|d| f64::from(input[at + d]) * inverse * (1.0 + f64::from(weights[d])))
                .collect();
            let half = rotary / 2;
            for d in 0..head_dim {
                expected[at + d] = if d < rotary {
                    let frequency = if d < half { d } else { d - half };
                    let angle =
                        (start + row as u64) as f64 * config.rope_theta.powf(-2.0 * frequency as f64 / rotary as f64);
                    let (cosine, sine) = (angle.cos(), angle.sin());
                    if d < half {
                        normalized[d] * cosine - normalized[d + half] * sine
                    } else {
                        normalized[d] * cosine + normalized[d - half] * sine
                    }
                } else {
                    normalized[d]
                } as f32;
            }
        }
    }
    let got = bf16_values(&get(&workspace, BufferSlot::AttentionQkNormalized, 0, rows * width * 2));
    assert_close("the rotated rows", &got, &expected, 0.0, 0.012);
}

// The quantum's mode, and the embedding stage.

#[test]
#[ignore = "needs a GPU and the native library"]
fn a_quantum_runs_row_exact_if_and_only_if_it_verifies() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 0, KvFormat::Nvfp4, 0);
    let workspace = rig.workspace(&config, 8, 8, &[]);
    let make = |kind, row_exact| {
        Quantum::new(
            kind,
            2,
            InputRecordRef::new(0),
            Arc::clone(&workspace),
            SequenceRefs::new(Arc::clone(&sequence.device)),
            SubmitMode { pdl: false, row_exact },
            Arc::clone(&config),
            rig.lanes.clone(),
        )
    };
    assert!(make(QuantumKind::Verify, true).is_ok());
    assert!(make(QuantumKind::Decode, false).is_ok());
    assert_eq!(
        make(QuantumKind::Verify, false).unwrap_err(),
        QuantumError::RowExact {
            kind: QuantumKind::Verify,
            expected: true,
            given: false
        }
    );
    assert!(make(QuantumKind::Draft, true).is_err());
    assert!(make(QuantumKind::Prefill, true).is_err());
}

#[test]
#[ignore = "needs a GPU, the native library and the RCO artifact"]
fn embed_looks_up_the_ids_of_the_input_record_and_passes_step_tokens() {
    let Some(rig) = rig() else { return };
    let Some(mut artifact) = linear_gpu::Artifact::open() else {
        return;
    };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 0, KvFormat::Nvfp4, 0);
    let workspace = rig.workspace(&config, 8, 8, &[]);
    let (_table, weight) = artifact.plain(&rig.library, "text/token_embedding");
    // The shared helper uploads on the legacy stream; the lane must not start before the upload is done.
    rig.library.synchronize().unwrap();
    let hidden = weight.shape[1] as usize;
    assert_eq!(hidden, config.hidden_size);
    let stage = Embed::new(weight.clone(), BufferSlot::HiddenState);
    let tokens = &sequence.device.scratch.draft().unwrap().tokens;
    let id_bytes = |ids: &[i32]| -> Vec<u8> { ids.iter().flat_map(|id| id.to_ne_bytes()).collect() };

    // The reference: the Q3 lookup of `ids` by itself.
    let reference = |ids: &[i32]| -> Vec<u8> {
        let input = upload(&rig.library, &id_bytes(ids));
        let output = DeviceBuffer::alloc(&rig.library, ids.len() * hidden * 2).unwrap();
        rig.submit(|submission| Q3Embedding.submit(submission, &weight, input.ptr(), output.ptr(), ids.len()));
        download(&output)
    };
    let looked_up = |rows: usize| get(&workspace, BufferSlot::HiddenState, 0, rows * hidden * 2);

    // The ids of the input record.
    let ids = [0, 151_643, 151_643, 42, 7];
    put(&workspace, BufferSlot::TokenIds, 0, &id_bytes(&ids));
    let quantum = rig.quantum(QuantumKind::Prefill, ids.len(), 0, &workspace, &sequence, &config);
    assert!(rig.run(&stage, &quantum).is_some());
    rig.sync();
    assert!(
        looked_up(ids.len()) == reference(&ids),
        "the stage looks up what the format does"
    );
    assert!(
        download(tokens)[..4] != 0i32.to_ne_bytes() || ids[0] == 0,
        "no step token is written unasked"
    );

    // Rows that read their ids from slots 2.. of the step's tokens, and the last of them the step's current token.
    let step = [10, 101, 202, 303, 404, 505, 606];
    on_lane(|| tokens.copy_from_host(0, &id_bytes(&step))).unwrap();
    let rows = 3;
    let quantum = rig
        .quantum(QuantumKind::Draft, rows, 0, &workspace, &sequence, &config)
        .reading_draft_tokens(2)
        .writing_current_token();
    assert!(rig.run(&stage, &quantum).is_some());
    rig.sync();
    assert!(looked_up(rows) == reference(&step[2..5]), "ids from the step's tokens");
    assert_eq!(get(&workspace, BufferSlot::TokenIds, 0, 12), id_bytes(&step[2..5]));
    assert_eq!(
        download(tokens)[..4],
        404i32.to_ne_bytes(),
        "slot 0 holds the last token"
    );
    assert_eq!(download(tokens)[4..8], 101i32.to_ne_bytes(), "the other slots stay");

    // Slots past the step's tokens are refused.
    let past = rig
        .quantum(QuantumKind::Draft, rows, 0, &workspace, &sequence, &config)
        .reading_draft_tokens(16);
    let cx = StageCx {
        stage: StageId(0),
        lane: Some(LaneId(0)),
        waits: &[],
        lanes: &rig.pool,
    };
    assert!(stage.run(&past, &cx).is_err());

    // Its launch records into a graph, and the replay does what the eager launch did.
    put(&workspace, BufferSlot::HiddenState, 0, &vec![0u8; rows * hidden * 2]);
    let recorder = GraphRecorder::begin(&rig.lane).unwrap();
    {
        let submission = recorder.submission(quantum.mode()).unwrap();
        stage.launch(&quantum, &submission).unwrap();
    }
    let graph = recorder.end().unwrap();
    let submission = Submission::begin(&rig.lane, SubmitMode::default()).unwrap();
    graph.launch(&submission).unwrap();
    submission.finish().unwrap();
    rig.sync();
    assert!(
        looked_up(rows) == reference(&step[2..5]),
        "the replay looks up the same rows"
    );
}

// Capture.

#[test]
#[ignore = "needs a GPU and the native library"]
fn every_stage_launches_inside_a_graph_capture_and_the_replay_selects() {
    let Some(rig) = rig() else { return };
    let config = config(vec![LayerType::GatedDeltaNet, LayerType::FullAttention]);
    let (sequence, _retained) = rig.sequence(&config, 1, KvFormat::Nvfp4, 256);
    let workspace = rig.workspace(&config, 8, 8, &[1 << 20]);
    let selection = Arc::new(Selection::new(&rig.library, 8).unwrap());
    let mut rng = Rng(0xcafe_0001);
    let hidden = config.hidden_size;
    let heads = config.linear_num_value_heads;

    // Inputs for every slot the stages read, and the weights they name.
    for (slot, width, scale) in [
        (BufferSlot::HiddenState, hidden, 1.0),
        (BufferSlot::MixerDelta, hidden, 1.0),
        (BufferSlot::InputNormalized, hidden, 1.0),
        (BufferSlot::GateUp, 2 * config.intermediate_size, 1.0),
        (BufferSlot::QkProjected, 7168, 0.5),
        (BufferSlot::ValueZProjected, 12_288, 0.5),
        (BufferSlot::GdnRecurrent, config.gdn_value_width(), 1.0),
        (BufferSlot::AttentionQkNormalized, 7168, 0.5),
    ] {
        put(&workspace, slot, 0, &bf16_bytes(&rng.values(width, scale)));
    }
    put(&workspace, BufferSlot::GdnAlpha, 0, &f32_bytes(&vec![0.9; heads]));
    put(&workspace, BufferSlot::GdnBeta, 0, &f32_bytes(&vec![0.5; heads]));
    let mut logits = rng.values(VOCABULARY, 1.0);
    logits[31_337] = 8.0;
    put(&workspace, BufferSlot::Logits, 0, &bf16_bytes(&logits));
    let norm_weight = upload(&rig.library, &bf16_bytes(&rng.values(hidden, 0.2)));
    let head_weight = upload(&rig.library, &bf16_bytes(&rng.values(256, 0.2)));
    let conv_weight = upload(
        &rig.library,
        &bf16_bytes(&rng.values(config.gdn_convolution_channels() * 4, 0.2)),
    );
    let a_weight = upload(&rig.library, &bf16_bytes(&rng.values(heads * hidden, 0.05)));
    let control = upload(&rig.library, &f32_bytes(&vec![0.1; heads]));
    let seed = upload(&rig.library, &bf16_bytes(&rng.values(hidden, 0.5)));
    let format = Arc::new(StubFormat {
        calls: AtomicUsize::new(0),
        rows: AtomicUsize::new(0),
        flags: AtomicUsize::new(0),
        scratch_bytes: AtomicUsize::new(0),
        scratch_needed: 0,
    });

    // One decode quantum that selects, passes its token on and seeds drafting, with every stage of a layer's pass.
    let quantum = rig
        .quantum(QuantumKind::Decode, 1, 0, &workspace, &sequence, &config)
        .with_selection(Arc::clone(&selection))
        .with_draft_seed(DraftSeed::Address(seed.ptr()))
        .with_draft_token_write(DraftTokenWrite {
            slot: 1,
            ids: DevicePtr::NULL,
        });
    let stages: Vec<(&str, Box<dyn LaunchOnly>)> = vec![
        (
            "residual add",
            boxed(ResidualAdd::new(
                BufferSlot::HiddenState,
                BufferSlot::MixerDelta,
                BufferSlot::MixerHidden,
            )),
        ),
        (
            "fused norm",
            boxed(ResidualRmsNorm::new(
                BufferSlot::HiddenState,
                BufferSlot::MixerDelta,
                norm_weight.ptr(),
                BufferSlot::FinalHiddenState,
                BufferSlot::PostMixerNormalized,
            )),
        ),
        (
            "norm",
            boxed(RmsNormUnitOffset::new(
                BufferSlot::MixerHidden,
                BufferSlot::InputNormalized,
                norm_weight.ptr(),
                NormRows::All,
            )),
        ),
        (
            "final norm",
            boxed(RmsNormUnitOffset::new(
                BufferSlot::MixerHidden,
                BufferSlot::FinalNormalized,
                norm_weight.ptr(),
                NormRows::Final,
            )),
        ),
        ("swiglu", boxed(Swiglu::new(BufferSlot::GateUp, BufferSlot::Swiglu))),
        (
            "linear",
            boxed(Linear::new(
                Arc::clone(&format) as Arc<dyn LinearFormat>,
                stub_weight(),
                BufferSlot::InputNormalized,
                BufferSlot::MixerDelta,
                LinearRows::All,
                None,
            )),
        ),
        (
            "gdn project",
            boxed(GdnProjectControl::new(
                BufferSlot::InputNormalized,
                a_weight.ptr(),
                a_weight.ptr(),
                control.ptr(),
                control.ptr(),
                BufferSlot::AProjected,
                BufferSlot::BProjected,
            )),
        ),
        (
            "gdn convolution",
            boxed(GdnConvolution::new(
                0,
                BufferSlot::QkProjected,
                BufferSlot::ValueZProjected,
                conv_weight.ptr(),
                BufferSlot::GdnConvolved,
            )),
        ),
        (
            "gdn recurrence",
            boxed(GdnRecurrence::new(
                0,
                BufferSlot::GdnConvolved,
                BufferSlot::GdnAlpha,
                BufferSlot::GdnBeta,
                BufferSlot::GdnRecurrent,
            )),
        ),
        (
            "gdn norm",
            boxed(GdnGatedRmsNorm::new(
                BufferSlot::GdnRecurrent,
                BufferSlot::ValueZProjected,
                head_weight.ptr(),
                BufferSlot::GdnNormalized,
            )),
        ),
        (
            "qk norm rope",
            boxed(QkNormRope::new(
                BufferSlot::QkProjected,
                BufferSlot::AttentionQkNormalized,
                head_weight.ptr(),
                head_weight.ptr(),
            )),
        ),
        ("kv append", boxed(attention_append())),
        ("attention", boxed(attention_causal())),
        (
            "mtp stem",
            boxed(MtpStem::new(
                BufferSlot::HiddenState,
                norm_weight.ptr(),
                norm_weight.ptr(),
                BufferSlot::MtpNormed,
                BufferSlot::MtpPacked,
            )),
        ),
        (
            "argmax",
            boxed(Argmax::new(BufferSlot::Logits, VOCABULARY, SelectedRows::Last)),
        ),
        ("selected token", boxed(SelectedToken)),
    ];

    // Eagerly first: it loads every kernel module, which a capture cannot.
    for (name, stage) in &stages {
        let submission = Submission::begin(&rig.lane, quantum.mode()).unwrap();
        stage
            .launch_on(&quantum, &submission)
            .unwrap_or_else(|error| panic!("{name}: {error:?}"));
        submission.finish().unwrap();
        rig.sync();
    }

    // Then every launch in one capture: no begin, wait, marker or signal among them.
    let recorder = GraphRecorder::begin(&rig.lane).unwrap();
    {
        let submission = recorder.submission(quantum.mode()).unwrap();
        for (name, stage) in &stages {
            stage
                .launch_on(&quantum, &submission)
                .unwrap_or_else(|error| panic!("{name} in a capture: {error:?}"));
        }
    }
    let graph = recorder.end().unwrap();

    // The graph's launch is followed by the argmax's signal step: the marker the lane's completion word reaches.
    let submission = Submission::begin(&rig.lane, quantum.mode()).unwrap();
    graph.launch(&submission).unwrap();
    let marker = Argmax::signal(submission).unwrap();
    rig.sync();
    assert!(rig.lane.completed() >= marker.value);
    assert_eq!(selection.token(0), Some(31_337));
    let tokens = download(&sequence.device.scratch.draft().unwrap().tokens);
    assert_eq!(tokens[4..8], 31_337i32.to_ne_bytes(), "the drafted token was passed on");
}

/// A stage as the capture test holds it: only its launches.
trait LaunchOnly {
    fn launch_on(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError>;
}

impl<L: Launch> LaunchOnly for L {
    fn launch_on(&self, quantum: &Quantum, submission: &Submission<'_>) -> Result<(), StageError> {
        self.launch(quantum, submission)
    }
}

fn boxed<L: Launch + 'static>(stage: L) -> Box<dyn LaunchOnly> {
    Box::new(stage)
}
