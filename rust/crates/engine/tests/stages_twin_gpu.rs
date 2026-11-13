//! Temporary: removed together with the native launch path, once the stages run through RustEE.
//!
//! The engine's stages for one decode, one verify, one prefill and one draft quantum, run twice in one CUDA context:
//! on the native library's lane through a `Submission`, and on a RustEE lane through `RusteeOps`. Both sides start
//! from the same inputs in buffers of their own (the weights are shared, read only), and after every stage each
//! buffer a stage can write is compared byte for byte: the workspace's slots and scratch, the GDN states and their
//! speculative buffers, the attention scratch, the draft buffers and the selection words. The KV cache's pages are
//! not compared directly (their size is the cache's own); the attention stage reads every position the append wrote,
//! so its context is.
//!
//! The prefill quantum has the NVFP4 native tile route (non-skinny rows), which needs compute capability 12.x: the
//! test skips that quantum, and says so, elsewhere.

#![allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test file: a failed unwrap is the failure"
)]
#![allow(
    clippy::print_stdout,
    clippy::print_stderr,
    reason = "a test reports what it compared and what it skipped"
)]
#![allow(
    clippy::disallowed_methods,
    reason = "no-wait, waiting calls: a test synchronizes both lanes and copies synchronously to compare results"
)]

use std::mem::replace;
use std::path::PathBuf;
use std::sync::Arc;

use euhedral_gpu::cuda::{CudaGpu, CudaStream};
use euhedral_gpu::{Gpu, GpuError, Stream, StreamOptions};
use euhedral_graph::LaneId;
use euhedral_inference_cuda::launch::Modes;
use euhedral_inference_cuda::launch::load::{KernelTable, load};
use euhedral_inference_cuda::launch::nvfp4::{Nvfp4Layout, nvfp4_byte_size};
use euhedral_inference_cuda::launch::ops::RusteeOps;
use euhedral_inference_cuda::{
    CudaLane, CudaLibrary, DeviceBuffer, DevicePtr, KernelOps, PinnedPool, RetireList, Submission, SubmitMode,
};
use euhedral_inference_edrl::{LayerType, ModelMetadata, TensorDataType, WeightFormat, WeightLayout};
use euhedral_inference_engine::formats::{LinearFormat, LinearMode, Nvfp4Linear};
use euhedral_inference_engine::kernels::kernel_sources;
use euhedral_inference_engine::model::{ModelConfig, WeightRef};
use euhedral_inference_engine::stages::attention::{AttentionCausal, KvAppend, KvSource, QkNormRope};
use euhedral_inference_engine::stages::elementwise::{ResidualAdd, Swiglu};
use euhedral_inference_engine::stages::gdn::{GdnConvolution, GdnGatedRmsNorm, GdnProjectControl, GdnRecurrence};
use euhedral_inference_engine::stages::linear::{Linear, LinearRows};
use euhedral_inference_engine::stages::mtp::MtpStem;
use euhedral_inference_engine::stages::norm::{NormRows, ResidualRmsNorm, RmsNormUnitOffset};
use euhedral_inference_engine::stages::sampling::{Argmax, SelectedRows, SelectedToken, Selection, SelectionPool};
use euhedral_inference_engine::stages::{
    DraftSeed, DraftTokenWrite, InputRecordRef, LaneTable, Launch, Quantum, QuantumKind, SequenceRefs,
};
use euhedral_inference_engine::state::attention::AttentionScratch;
use euhedral_inference_engine::state::gdn::GdnStates;
use euhedral_inference_engine::state::kv::{KvCache, KvRetained};
use euhedral_inference_engine::state::{SequenceDevice, SequenceState};
use euhedral_inference_engine::workspace::{BufferSlot, Workspace, WorkspaceRows};
use euhedral_inference_settings::{KvFormat, ModelSettings, Settings};

const VOCABULARY: usize = 248_320;
const HIDDEN: usize = 5120;

fn config() -> Arc<ModelConfig> {
    Arc::new(ModelConfig::from(&ModelMetadata {
        vocab_size: VOCABULARY as u32,
        hidden_size: HIDDEN as u32,
        num_hidden_layers: 2,
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
        layer_types: vec![LayerType::GatedDeltaNet, LayerType::FullAttention],
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

/// A splitmix64 stream: the same values for the same seed.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
        z ^ (z >> 31)
    }

    /// A value in `[-scale, scale)`.
    fn value(&mut self, scale: f32) -> f32 {
        ((self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * scale
    }
}

fn bf16_bits(value: f32) -> u16 {
    let bits = value.to_bits();
    (bits.wrapping_add(0x7fff + ((bits >> 16) & 1)) >> 16) as u16
}

fn bf16_bytes(rng: &mut Rng, count: usize, scale: f32) -> Vec<u8> {
    (0..count)
        .flat_map(|_| bf16_bits(rng.value(scale)).to_ne_bytes())
        .collect()
}

fn f32_bytes(count: usize, value: f32) -> Vec<u8> {
    (0..count).flat_map(|_| value.to_ne_bytes()).collect()
}

/// The native library's lane and a RustEE lane on the same device and context.
struct Twin {
    library: Arc<CudaLibrary>,
    table: KernelTable,
    lane: Arc<CudaLane>,
    stream: CudaStream,
    lanes: LaneTable,
    retire: Arc<RetireList>,
    pinned: Arc<PinnedPool>,
}

/// Everything one side of a quantum owns.
struct Side {
    quantum: Quantum,
    workspace: Arc<Workspace>,
    sequence: SequenceState,
    selection: Arc<Selection>,
    _retained: KvRetained,
}

impl Twin {
    fn open(library: &Arc<CudaLibrary>) -> Option<Twin> {
        let (major, minor) = library.device().compute_capability;
        let wanted = (i32::try_from(major).ok()?, i32::try_from(minor).ok()?);
        let mut gpu = None;
        for ordinal in 0..16 {
            let Ok(candidate) = CudaGpu::open(ordinal) else { break };
            if candidate.arch() == wanted {
                gpu = Some(candidate);
                break;
            }
        }
        let gpu = gpu.expect("RustEE finds a device of the library's compute capability");
        let settings = Settings::with_model(ModelSettings::default());
        let table = load(&gpu, &kernel_sources(library, &settings).unwrap()).unwrap();
        let lane = CudaLane::new(library, LaneId(0), 256).unwrap();
        let mut streams = gpu
            .open_lanes_at(LaneId(1), &["twin"], &[StreamOptions::default()])
            .unwrap();
        let stream = streams.pop().expect("RustEE opened a lane");
        Some(Twin {
            library: Arc::clone(library),
            table,
            lanes: LaneTable::new(vec![Arc::clone(&lane)]).unwrap(),
            lane,
            stream,
            retire: Arc::new(RetireList::new()),
            pinned: PinnedPool::new(library, 64 << 20),
        })
    }

    /// Runs a blocking host copy on the native lane, in a submission, so it orders against the lane's work.
    fn host<T>(&self, copy: impl FnOnce() -> T) -> T {
        let submission = Submission::begin(&self.lane, SubmitMode::default()).unwrap();
        let out = copy();
        drop(submission);
        self.lane.synchronize().unwrap();
        out
    }

    fn put(&self, buffer: &DeviceBuffer, offset: usize, bytes: &[u8]) {
        self.host(|| buffer.copy_from_host(offset, bytes)).unwrap();
    }

    fn get(&self, buffer: &DeviceBuffer) -> Vec<u8> {
        let mut out = vec![0u8; buffer.bytes()];
        self.host(|| buffer.copy_to_host(0, &mut out)).unwrap();
        out
    }

    fn upload(&self, bytes: &[u8]) -> DeviceBuffer {
        let buffer = DeviceBuffer::alloc(&self.library, bytes.len().max(256)).unwrap();
        self.put(&buffer, 0, bytes);
        buffer
    }

    fn zero(&self, buffer: &DeviceBuffer) {
        self.put(buffer, 0, &vec![0u8; buffer.bytes()]);
    }

    /// One side of a quantum: its workspace, its sequence (GDN state, KV cache, attention scratch, selection words),
    /// all zeroed, and the quantum over them.
    fn side(
        &self,
        config: &Arc<ModelConfig>,
        kind: QuantumKind,
        rows: usize,
        logits_rows: u64,
        scratch: &[u64],
    ) -> Side {
        let workspace = Arc::new(
            Workspace::allocate(
                &self.library,
                config,
                WorkspaceRows {
                    rows: rows as u64,
                    logits_rows,
                },
                scratch,
            )
            .unwrap(),
        );
        for slot in BufferSlot::ALL {
            self.zero(workspace.buffer(slot));
        }
        for region in 0..scratch.len() {
            self.zero(workspace.scratch(region).unwrap());
        }
        let submission = Submission::begin(&self.lane, SubmitMode::default()).unwrap();
        let gdn = GdnStates::allocate(&self.library, config, 8, &mut &submission).unwrap();
        let kv_layers = usize::from(rows <= 8);
        let mut kv = KvCache::new(&self.library, &self.retire, kv_layers, 1024, KvFormat::Nvfp4, 256, 0).unwrap();
        let mut retained = KvRetained::default();
        if kv_layers > 0 {
            kv.reserve(&submission, &self.pinned, 0, 256, &mut retained).unwrap();
        }
        gdn.zero(&mut &submission).unwrap();
        submission.finish_signalled().unwrap();
        self.lane.synchronize().unwrap();
        if let Some(speculative) = gdn.layer(0).and_then(|state| state.speculative()) {
            for buffer in [
                &speculative.recurrent_checkpoint,
                &speculative.convolution_checkpoint,
                &speculative.query_key_rows,
                &speculative.value_z_rows,
                &speculative.alpha_rows,
                &speculative.beta_rows,
                &speculative.replay_convolved,
                &speculative.replay_output,
            ] {
                self.zero(buffer);
            }
        }
        let attention = AttentionScratch::allocate(&self.library, config, 8, true, 16).unwrap();
        self.zero(attention.decode());
        let draft = attention.draft().unwrap();
        for buffer in [&draft.seed_rows, &draft.recursion_hidden, &draft.tokens] {
            self.zero(buffer);
        }
        let sequence = SequenceState::from_parts(
            kv,
            None,
            Arc::new(SequenceDevice {
                gdn,
                scratch: attention,
            }),
            SelectionPool::new(&self.library, 2, 8).unwrap(),
        );
        let selection = Arc::new(Selection::new(&self.library, 8).unwrap());
        let mode = SubmitMode {
            pdl: false,
            row_exact: kind == QuantumKind::Verify,
        };
        // The admission uploads the input record: the start position, after the token ids.
        self.put(
            workspace.buffer(BufferSlot::TokenIds),
            InputRecordRef::position_offset(rows) as usize,
            &0u64.to_ne_bytes(),
        );
        let quantum = Quantum::new(
            kind,
            rows,
            InputRecordRef::new(0),
            Arc::clone(&workspace),
            SequenceRefs::new(Arc::clone(&sequence.device))
                .with_kv(sequence.kv.view())
                .with_replay_rows(Vec::new()),
            mode,
            Arc::clone(config),
            self.lanes.clone(),
        )
        .unwrap()
        .with_selection(Arc::clone(&selection));
        Side {
            quantum,
            workspace,
            sequence,
            selection,
            _retained: retained,
        }
    }

    /// Runs `stage` on the native lane.
    fn run_native(&self, stage: &dyn Stage, side: &Side) -> Result<(), String> {
        let submission = Submission::begin(&self.lane, side.quantum.mode()).map_err(|error| format!("{error:?}"))?;
        stage
            .launch_on(&side.quantum, &mut &submission)
            .map_err(|error| format!("{error:?}"))?;
        submission.finish().map_err(|error| format!("{error:?}"))?;
        self.lane.synchronize().map_err(|error| format!("{error:?}"))
    }

    /// Runs `stage` through `RusteeOps` on the RustEE lane.
    fn run_rustee(&mut self, stage: &dyn Stage, side: &Side) -> Result<(), String> {
        let modes = Modes {
            row_exact: side.quantum.mode().row_exact,
            exact_numerics: self.library.exact_numerics(),
        };
        let table = &self.table;
        let mut failure = None;
        self.stream
            .submit(|queue| {
                let mut ops = RusteeOps::new(queue, table, modes);
                if let Err(error) = stage.launch_on(&side.quantum, &mut ops) {
                    failure = Some(format!("{error:?}"));
                }
                Ok::<(), GpuError>(())
            })
            .map_err(|error| format!("{error:?}"))?;
        self.stream.quiesce();
        failure.map_or(Ok(()), Err)
    }

    /// Every buffer a stage can write on one side, named.
    fn buffers<'a>(&self, side: &'a Side) -> Vec<(String, &'a DeviceBuffer)> {
        let mut out: Vec<(String, &DeviceBuffer)> = BufferSlot::ALL
            .iter()
            .map(|slot| (format!("{slot:?}"), side.workspace.buffer(*slot)))
            .collect();
        for region in 0.. {
            let Some(buffer) = side.workspace.scratch(region) else {
                break;
            };
            out.push((format!("scratch {region}"), buffer));
        }
        let device = &side.sequence.device;
        if let Some(state) = device.gdn.layer(0) {
            out.push(("gdn convolution".to_string(), state.convolution()));
            out.push(("gdn recurrent".to_string(), state.recurrent()));
            if let Some(speculative) = state.speculative() {
                for (name, buffer) in [
                    ("recurrent checkpoint", &speculative.recurrent_checkpoint),
                    ("convolution checkpoint", &speculative.convolution_checkpoint),
                    ("query key rows", &speculative.query_key_rows),
                    ("value z rows", &speculative.value_z_rows),
                    ("alpha rows", &speculative.alpha_rows),
                    ("beta rows", &speculative.beta_rows),
                    ("replay convolved", &speculative.replay_convolved),
                    ("replay output", &speculative.replay_output),
                ] {
                    out.push((format!("gdn {name}"), buffer));
                }
            }
        }
        out.push(("attention decode scratch".to_string(), device.scratch.decode()));
        if let Some(draft) = device.scratch.draft() {
            out.push(("draft seed rows".to_string(), &draft.seed_rows));
            out.push(("draft recursion hidden".to_string(), &draft.recursion_hidden));
            out.push(("draft tokens".to_string(), &draft.tokens));
        }
        out
    }

    /// Compares every buffer of the two sides; the first difference as `(buffer, offset, native, rustee)`, else the
    /// bytes compared.
    fn compare(&self, native: &Side, rustee: &Side) -> Result<usize, (String, usize, u8, u8)> {
        let mut compared = 0;
        for ((name, left), (_, right)) in self.buffers(native).into_iter().zip(self.buffers(rustee)) {
            let (left, right) = (self.get(left), self.get(right));
            if let Some(offset) = left.iter().zip(&right).position(|(a, b)| a != b) {
                return Err((name, offset, left[offset], right[offset]));
            }
            compared += left.len();
        }
        for row in 0..8 {
            let (left, right) = (native.selection.key(row), rustee.selection.key(row));
            if left != right {
                return Err((format!("selection key {row}"), 0, 0, 0));
            }
            compared += 8;
        }
        Ok(compared)
    }
}

/// A stage as the test holds it: only its launches.
trait Stage {
    fn launch_on(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), String>;
}

impl<L: Launch> Stage for L {
    fn launch_on(&self, quantum: &Quantum, kernels: &mut dyn KernelOps) -> Result<(), String> {
        self.launch(quantum, kernels).map_err(|error| format!("{error:?}"))
    }
}

fn boxed<L: Launch + 'static>(stage: L) -> Box<dyn Stage> {
    Box::new(stage)
}

/// The weights the stages read, shared by both sides.
struct Weights {
    norm: DeviceBuffer,
    head: DeviceBuffer,
    convolution: DeviceBuffer,
    a: DeviceBuffer,
    control: DeviceBuffer,
    seed: DeviceBuffer,
    _linear: DeviceBuffer,
    linear_ref: WeightRef,
}

impl Weights {
    fn new(twin: &Twin, config: &ModelConfig) -> Weights {
        let mut rng = Rng(0x5eed_0001);
        let heads = config.linear_num_value_heads;
        let (out, input) = (HIDDEN as u32, HIDDEN as u32);
        let bytes = nvfp4_byte_size(out, input, Nvfp4Layout::Plain) as usize;
        let codes = (out as usize * input as usize) / 2;
        let scales_at = codes.next_multiple_of(256);
        let scales = out as usize * input as usize / 16;
        let mut weight = vec![0u8; bytes];
        for byte in &mut weight[..codes] {
            *byte = rng.next() as u8;
        }
        // E4M3 scales of moderate magnitude, and a global scale of 1.
        for byte in &mut weight[scales_at..scales_at + scales] {
            *byte = 0x30 + (rng.next() % 0x18) as u8;
        }
        weight[bytes - 4..].copy_from_slice(&1.0f32.to_ne_bytes());
        let linear = twin.upload(&weight);
        let linear_ref = WeightRef {
            name: "synthetic".to_string(),
            shape: vec![u64::from(out), u64::from(input)],
            dtype: TensorDataType::Bf16,
            format: WeightFormat::Nvfp4,
            layout: WeightLayout::RowSplitK128V1,
            ptr: linear.ptr(),
            bytes: bytes as u64,
            trellis: None,
        };
        Weights {
            norm: twin.upload(&bf16_bytes(&mut rng, HIDDEN, 0.2)),
            head: twin.upload(&bf16_bytes(&mut rng, 256, 0.2)),
            convolution: twin.upload(&bf16_bytes(&mut rng, config.gdn_convolution_channels() * 4, 0.2)),
            a: twin.upload(&bf16_bytes(&mut rng, heads * HIDDEN, 0.05)),
            control: twin.upload(&f32_bytes(heads, 0.1)),
            seed: twin.upload(&bf16_bytes(&mut rng, HIDDEN, 0.5)),
            _linear: linear,
            linear_ref,
        }
    }
}

/// The inputs of every slot the stages read, for `rows` rows: the same values for the same seed.
fn fill(twin: &Twin, config: &ModelConfig, side: &Side, rows: usize, seed: u64) {
    let mut rng = Rng(seed);
    let heads = config.linear_num_value_heads;
    for (slot, width, scale) in [
        (BufferSlot::HiddenState, HIDDEN, 1.0),
        (BufferSlot::MixerDelta, HIDDEN, 1.0),
        (BufferSlot::InputNormalized, HIDDEN, 1.0),
        (BufferSlot::GateUp, 2 * config.intermediate_size, 1.0),
        (BufferSlot::QkProjected, 7168, 0.5),
        (BufferSlot::ValueZProjected, 12_288, 0.5),
        (BufferSlot::GdnRecurrent, config.gdn_value_width(), 1.0),
        (BufferSlot::AttentionQkNormalized, 7168, 0.5),
    ] {
        twin.put(
            side.workspace.buffer(slot),
            0,
            &bf16_bytes(&mut rng, rows * width, scale),
        );
    }
    twin.put(
        side.workspace.buffer(BufferSlot::GdnAlpha),
        0,
        &f32_bytes(rows * heads, 0.9),
    );
    twin.put(
        side.workspace.buffer(BufferSlot::GdnBeta),
        0,
        &f32_bytes(rows * heads, 0.5),
    );
    let logits_rows = side.workspace.buffer(BufferSlot::Logits).bytes() / (2 * VOCABULARY);
    let mut logits = bf16_bytes(&mut rng, logits_rows * VOCABULARY, 1.0);
    // A clear maximum in the first row, and a tie in the second.
    logits[2 * 31_337..2 * 31_337 + 2].copy_from_slice(&bf16_bits(8.0).to_ne_bytes());
    if logits_rows > 1 {
        for id in [100, 200_000] {
            let at = 2 * (VOCABULARY + id);
            logits[at..at + 2].copy_from_slice(&bf16_bits(9.0).to_ne_bytes());
        }
    }
    twin.put(side.workspace.buffer(BufferSlot::Logits), 0, &logits);
}

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

/// The stages of one layer's pass, with `linear` as the projection.
fn stages(
    weights: &Weights,
    linear: Arc<dyn LinearFormat>,
    scratch_region: Option<usize>,
    rows: SelectedRows,
) -> Vec<(&'static str, Box<dyn Stage>)> {
    vec![
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
                weights.norm.ptr(),
                BufferSlot::FinalHiddenState,
                BufferSlot::PostMixerNormalized,
            )),
        ),
        (
            "norm",
            boxed(RmsNormUnitOffset::new(
                BufferSlot::MixerHidden,
                BufferSlot::InputNormalized,
                weights.norm.ptr(),
                NormRows::All,
            )),
        ),
        (
            "final norm",
            boxed(RmsNormUnitOffset::new(
                BufferSlot::MixerHidden,
                BufferSlot::FinalNormalized,
                weights.norm.ptr(),
                NormRows::Final,
            )),
        ),
        ("swiglu", boxed(Swiglu::new(BufferSlot::GateUp, BufferSlot::Swiglu))),
        (
            "nvfp4 linear",
            boxed(Linear::new(
                linear,
                weights.linear_ref.clone(),
                BufferSlot::InputNormalized,
                BufferSlot::MixerDelta,
                LinearRows::All,
                scratch_region,
            )),
        ),
        (
            "gdn project",
            boxed(GdnProjectControl::new(
                BufferSlot::InputNormalized,
                weights.a.ptr(),
                weights.a.ptr(),
                weights.control.ptr(),
                weights.control.ptr(),
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
                weights.convolution.ptr(),
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
                weights.head.ptr(),
                BufferSlot::GdnNormalized,
            )),
        ),
        (
            "qk norm rope",
            boxed(QkNormRope::new(
                BufferSlot::QkProjected,
                BufferSlot::AttentionQkNormalized,
                weights.head.ptr(),
                weights.head.ptr(),
            )),
        ),
        ("kv append", boxed(attention_append())),
        ("attention", boxed(attention_causal())),
        ("argmax", boxed(Argmax::new(BufferSlot::Logits, VOCABULARY, rows))),
    ]
}

/// What a quantum of the test is.
struct Case {
    name: &'static str,
    kind: QuantumKind,
    rows: usize,
    logits_rows: u64,
    selected: SelectedRows,
    /// Run only the stages named here (all when empty).
    only: &'static [&'static str],
    draft: bool,
    /// The case takes the native tile route: its linear must have quantized into the scratch.
    tile: bool,
}

#[test]
#[ignore = "needs a GPU, the native library and the CUDA compiler"]
fn a_decode_and_a_verify_quantum_through_rustee_ops_equal_the_native_path_bit_for_bit() {
    let Some(path) = test_support::require_property("euhedral.cuda.library") else {
        return;
    };
    let library = CudaLibrary::load(&PathBuf::from(path)).expect("the CUDA library loads");
    let mut twin = Twin::open(&library).expect("a twin");
    let config = config();
    let weights = Weights::new(&twin, &config);
    let format = Arc::new(Nvfp4Linear::new(&library));
    let tile_route = library.nvfp4_native_available();
    println!(
        "native NVFP4 route available: {tile_route} (compute capability {:?})",
        library.device().compute_capability
    );

    let mut cases = vec![
        Case {
            name: "decode",
            kind: QuantumKind::Decode,
            rows: 1,
            logits_rows: 1,
            selected: SelectedRows::Last,
            only: &[],
            draft: false,
            tile: false,
        },
        Case {
            name: "verify",
            kind: QuantumKind::Verify,
            rows: 4,
            logits_rows: 4,
            selected: SelectedRows::Each,
            only: &[],
            draft: false,
            tile: false,
        },
        Case {
            name: "draft",
            kind: QuantumKind::Draft,
            rows: 1,
            logits_rows: 1,
            selected: SelectedRows::Last,
            only: &[],
            draft: true,
            tile: false,
        },
    ];
    if tile_route {
        // Non-skinny rows: the native tile route, with the scratch it takes.
        assert!(
            !library.exact_numerics(),
            "exact numerics make both sides refuse the native route"
        );
        for (name, rows, tile) in [("prefill tile route", 72, true), ("skinny route, 32 rows", 32, false)] {
            cases.push(Case {
                name,
                kind: QuantumKind::Prefill,
                rows,
                logits_rows: 1,
                selected: SelectedRows::Last,
                only: &["residual add", "fused norm", "norm", "swiglu", "nvfp4 linear"],
                draft: false,
                tile,
            });
        }
    } else {
        println!(
            "SKIPPED the NVFP4 native tile-route quantum: the device has no native NVFP4 route (needs compute capability 12.x)"
        );
    }

    let mut failures = Vec::new();
    for case in &cases {
        let scratch: Vec<u64> = if case.rows >= 9 {
            vec![format.scratch_bytes(
                &weights.linear_ref,
                case.rows,
                LinearMode {
                    row_exact: false,
                    prefill: true,
                },
            ) as u64]
        } else {
            Vec::new()
        };
        let region = (!scratch.is_empty()).then_some(0);
        let native = twin.side(&config, case.kind, case.rows, case.logits_rows, &scratch);
        let rustee = twin.side(&config, case.kind, case.rows, case.logits_rows, &scratch);
        for side in [&native, &rustee] {
            fill(&twin, &config, side, case.rows, 0xfeed_0000 + case.rows as u64);
        }
        let (mut native, mut rustee) = (native, rustee);
        if case.draft {
            for side in [&mut native, &mut rustee] {
                let stand_in = placeholder(&twin, &config, side);
                let quantum = replace(&mut side.quantum, stand_in);
                side.quantum = quantum
                    .with_draft_seed(DraftSeed::Address(weights.seed.ptr()))
                    .unwrap()
                    .with_draft_token_write(DraftTokenWrite {
                        slot: 1,
                        ids: DevicePtr::NULL,
                    })
                    .unwrap();
            }
        }
        let mut list = stages(
            &weights,
            Arc::clone(&format) as Arc<dyn LinearFormat>,
            region,
            case.selected,
        );
        if case.draft {
            list.push((
                "mtp stem",
                boxed(MtpStem::new(
                    BufferSlot::HiddenState,
                    weights.norm.ptr(),
                    weights.norm.ptr(),
                    BufferSlot::MtpNormed,
                    BufferSlot::MtpPacked,
                )),
            ));
            list.push(("selected token", boxed(SelectedToken)));
        }
        let mut compared = 0;
        'stages: for (name, stage) in &list {
            if !case.only.is_empty() && !case.only.contains(name) {
                continue;
            }
            let left = twin.run_native(stage.as_ref(), &native);
            let right = twin.run_rustee(stage.as_ref(), &rustee);
            match (left, right) {
                (Ok(()), Ok(())) => {}
                (left, right) => {
                    let message = format!("{}: {name}: native {left:?}, RustEE {right:?}", case.name);
                    println!("FAILED {message}");
                    failures.push(message);
                    break 'stages;
                }
            }
            if case.tile && *name == "nvfp4 linear" {
                let scratch = twin.get(native.workspace.scratch(0).unwrap());
                assert!(
                    scratch.iter().any(|byte| *byte != 0),
                    "the tile route quantizes into the scratch: the native side took another route"
                );
            }
            match twin.compare(&native, &rustee) {
                Ok(bytes) => compared = bytes,
                Err((buffer, offset, left, right)) => {
                    let message = format!(
                        "{}: after {name}: {buffer} differs at byte {offset}: native {left:#04x}, RustEE {right:#04x}",
                        case.name
                    );
                    println!("MISMATCH {message}");
                    failures.push(message);
                    break 'stages;
                }
            }
        }
        if failures.is_empty() {
            println!(
                "{}: every stage identical, {compared} bytes compared after the last",
                case.name
            );
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

/// A quantum to move the real one out of its side while it takes its draft settings.
fn placeholder(twin: &Twin, config: &Arc<ModelConfig>, side: &Side) -> Quantum {
    Quantum::new(
        QuantumKind::Decode,
        1,
        InputRecordRef::new(0),
        Arc::clone(&side.workspace),
        SequenceRefs::new(Arc::clone(&side.sequence.device)),
        SubmitMode::default(),
        Arc::clone(config),
        twin.lanes.clone(),
    )
    .unwrap()
}
