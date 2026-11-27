//! The device side of one Flash-Next sequence (`Session.java`'s allocation and its per-chunk launch): the state, the
//! regions and the logits the stages bake into their addresses, the per-bucket graphs, and the quantum each step of
//! [`FlashOwner`] runs.
//!
//! Role: [`DeviceStepper`] is the [`Stepper`] of the real engine. It owns the sequence's device memory for as long as
//! the engine lives and builds one quantum per step: the owner lane reserves the chunk's cache rows, zeroes the state
//! of a request's first chunk, uploads the input record and frees what the caches let go of, all in one marked
//! submission whose marker orders the graph behind it. The graph's retirement is posted to the owner.
//!
//! The quantum is the dense [`Quantum`] over a degenerate [`ModelConfig`] (no layers: only its vocabulary is read),
//! because the Flash stages take everything else from the carrier they are given: the workspace's input record, the
//! chunk placement, the n-gram rows and the MoE block.
//!
//! Departures from Java: the sequence's state is allocated once and zeroed per request (the stages bake its addresses;
//! Java allocates a `Sequence` per request), and the cache's pages stay allocated between requests and are truncated
//! at the next request's first chunk. The device allocations are plain allocations, not claims on the memory account.

use std::fmt::{self, Debug, Display, Formatter};
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use euhedral_core::generics::FrameSink;
use euhedral_graph::{DEFAULT_RETRY_BACKSTOP, GraphRuntime, LaneId, QuantumOutcome, RetryCounts};
use euhedral_inference_cuda::{DeviceBuffer, SubmitMode};
use euhedral_inference_settings::KvFormat;

use super::FlashError;
use super::assembly::{Blueprint, Ingredients, ngram_ids, sequence_spec};
use super::buffers::FlashRegions;
use super::generate::{FlashOwner, Limits, Message, Retired, Score, Step, Stepper};
use super::linear::Nvfp4Device;
use super::moe::block::{MoeBlock, MoeStaging};
use super::moe::routing::DescriptorLayout;
use super::moe::source::ExpertSource;
use super::ple::PleRows;
use super::seq_state::{SequenceLayout, SequenceState};
use super::startup::FlashModel;
use super::step::{Buckets, FlashStep};
use crate::generation::Address;
use crate::model::ModelConfig;
use crate::quantum::Started;
use crate::runtime::Core;
use crate::stages::sampling::{RowReadback, Selection};
use crate::stages::{InputRecordRef, Outcome, Quantum, SequenceRefs, conclude};
use crate::state::SequenceDevice;
use crate::state::attention::AttentionScratch;
use crate::state::gdn::GdnStates;
use crate::state::kv::{KvCache, KvRetained};
use crate::workspace::{Workspace, WorkspaceRows};

const OP: &str = "flash device";

/// Where the owner of the engine's one sequence is reached from a retiring quantum.
pub(crate) type OwnerAddress = Address<FlashOwner<DeviceStepper>>;

fn refuse<E: Display>(what: &'static str) -> impl FnOnce(E) -> FlashError {
    move |error| FlashError::refused(OP, format!("{what}: {error}"))
}

/// The configuration the dense carrier is built over: the vocabulary is the model's, every other size is the smallest
/// positive one, and there are no layers (the Flash stages run on their own regions).
fn carrier_config(vocabulary: usize) -> ModelConfig {
    ModelConfig {
        vocab_size: vocabulary,
        hidden_size: 128,
        num_hidden_layers: 0,
        num_attention_heads: 1,
        num_key_value_heads: 1,
        attention_head_dim: 256,
        intermediate_size: 128,
        linear_num_key_heads: 1,
        linear_num_value_heads: 1,
        linear_key_head_dim: 128,
        linear_value_head_dim: 128,
        linear_conv_kernel_dim: 4,
        rms_norm_epsilon: 1e-6,
        rope_theta: 1e4,
        partial_rotary_factor: 0.25,
        max_position_embeddings: 1,
        layer_kinds: Vec::new(),
        num_experts: 0,
        tie_word_embeddings: false,
        attention_output_gate: false,
        mtp_layer_count: 0,
        draft_depth: None,
        dflash2: None,
    }
}

/// The quantum in flight, and what it holds until it retires.
struct InFlight {
    /// Keeps the graph's run referenced until the next quantum starts.
    _started: Started,
    /// The pinned staging of the cache reservation the quantum's stages read.
    _retained: KvRetained,
}

/// One sequence's device memory, graphs and quantum builder.
pub struct DeviceStepper {
    core: Arc<Core>,
    step: FlashStep,
    state: SequenceState,
    workspace: Arc<Workspace>,
    config: Arc<ModelConfig>,
    device: Arc<SequenceDevice>,
    selection: Arc<Selection>,
    readback: Arc<RowReadback>,
    buckets: Buckets,
    layouts: Vec<(u32, DescriptorLayout)>,
    fresh_context: Box<[i32]>,
    attention_layers: usize,
    flight: Option<InFlight>,
    // Freed after the graphs and the state that address them.
    _logits: DeviceBuffer,
    _regions: DeviceBuffer,
}

impl Debug for DeviceStepper {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("DeviceStepper").finish_non_exhaustive()
    }
}

impl DeviceStepper {
    /// Allocates the sequence's device memory over `model` and builds the graphs of every bucket. The retirement of each
    /// quantum is posted to `address`. Returns the stepper and the limits its owner admits requests by.
    pub(crate) fn new(model: &FlashModel, address: &Arc<OwnerAddress>) -> Result<(Self, Limits), FlashError> {
        let core = Arc::clone(model.core());
        let library = &core.library;
        let config = model.artifact.config();
        let max_tokens = u32::try_from(model.plan.max_context_tokens)
            .map_err(|_| FlashError::refused(OP, format!("a context of {} tokens", model.plan.max_context_tokens)))?;
        let max_rows = u32::try_from(model.plan.prefill_chunk_tokens)
            .map_err(|_| FlashError::refused(OP, format!("a chunk of {} tokens", model.plan.prefill_chunk_tokens)))?;
        let buckets = Buckets::new(max_rows, config.moe.num_experts, config.moe.experts_per_token)?;
        let source: Arc<dyn ExpertSource> = model
            .shared_cache()
            .ok_or_else(|| FlashError::refused(OP, "the model has no expert cache"))?;

        let spec = sequence_spec(config, max_tokens)?;
        let state_buffer = DeviceBuffer::alloc(library, spec.bytes() as usize).map_err(refuse("the state"))?;
        let layout = SequenceLayout::over(&spec, state_buffer.ptr())?;
        let blueprint = Blueprint::new(&Ingredients {
            config,
            weights: &model.weights,
            padded_down: &model.padded_down,
            tables: &model.tables,
            banks: model.artifact.tables().banks(),
            layout: &layout,
            buckets,
            device: Nvfp4Device::new(library),
            max_tokens,
        })?;
        let regions_buffer =
            DeviceBuffer::alloc(library, blueprint.region_bytes() as usize).map_err(refuse("the regions"))?;
        let regions = FlashRegions::over(
            regions_buffer.ptr(),
            blueprint.capacity_rows(),
            blueprint.streams(),
            blueprint.scratch(),
        )?;
        let vocabulary = config.text.vocab_size as usize;
        let logits = DeviceBuffer::alloc(library, vocabulary * 2).map_err(refuse("the logits"))?;
        let layouts = buckets
            .sizes()
            .into_iter()
            .map(|bucket| blueprint.moe_layout(bucket).map(|layout| (bucket, layout)))
            .collect::<Result<Vec<_>, _>>()?;
        let stages = blueprint.stages(regions, logits.ptr(), &core.pool, &source)?;

        let attention_layers = layout.attention().count();
        let kv_width = u64::from(config.attention.num_kv_heads) * u64::from(config.attention.head_dim);
        let kv = KvCache::new(
            library,
            &core.released,
            attention_layers,
            kv_width,
            KvFormat::Nvfp4,
            u64::from(max_tokens),
            0,
        )
        .map_err(refuse("the cache"))?;
        let fresh_context = ngram_ids(config)?.fresh_context();
        let state = SequenceState::new(state_buffer, layout, kv, fresh_context.clone())?;

        let carrier = Arc::new(carrier_config(vocabulary));
        let workspace = Arc::new(
            Workspace::allocate(
                library,
                &carrier,
                WorkspaceRows {
                    rows: u64::from(max_rows),
                    logits_rows: 1,
                },
                &[],
            )
            .map_err(refuse("the workspace"))?,
        );
        let device = Arc::new(sequence_device(&core, &carrier)?);
        let selection = Arc::new(Selection::new(library, 1).map_err(refuse("the selection"))?);
        let readback = Arc::new(RowReadback::new(library, 1, vocabulary).map_err(refuse("the row readback"))?);

        let runtime = GraphRuntime {
            stages: Arc::clone(&core.stages) as Arc<dyn FrameSink>,
            completions: Arc::clone(core.completions()),
            lanes: Arc::clone(core.lanes()),
            retries: Arc::new(RetryCounts::default()),
            retry_backstop: DEFAULT_RETRY_BACKSTOP,
        };
        let retiring = Arc::clone(address);
        let step = FlashStep::new(
            config,
            buckets,
            &runtime,
            LaneId(0),
            &stages,
            move |quantum, outcome| {
                retiring.post(Message::Retired(retired(&quantum, &outcome)));
            },
        )
        .map_err(refuse("the step"))?;
        let limits = Limits {
            vocabulary,
            capacity: max_tokens,
            chunk: max_rows,
        };
        let stepper = Self {
            core,
            step,
            state,
            workspace,
            config: carrier,
            device,
            selection,
            readback,
            buckets,
            layouts,
            fresh_context,
            attention_layers,
            flight: None,
            _logits: logits,
            _regions: regions_buffer,
        };
        Ok((stepper, limits))
    }

    /// The cache layers of the sequence: the sparse-attention layers.
    pub fn attention_layers(&self) -> usize {
        self.attention_layers
    }

    /// Builds the quantum of `step` and starts it. The owner lane's one submission reserves the cache rows (after
    /// zeroing the state of a request's first chunk), uploads the input record and frees the buffers the caches let go
    /// of.
    fn launch(&mut self, step: &Step) -> Result<(), String> {
        let rows = u32::try_from(step.tokens.len()).map_err(|_| "a chunk of too many rows".to_owned())?;
        let bucket = self.buckets.bucket_for(rows).map_err(|error| error.to_string())?;
        let layout = self
            .layouts
            .iter()
            .find(|(of, _)| *of == bucket)
            .map(|(_, layout)| *layout)
            .ok_or_else(|| format!("no MoE layout for the bucket of {rows} rows"))?;
        let staging = MoeStaging::new(&self.core.pool, layout).map_err(|error| error.to_string())?;
        let block = Arc::new(MoeBlock::new(staging, &layout).map_err(|error| error.to_string())?);
        let mut retained = KvRetained::default();
        let core = &self.core;
        let (state, fresh_context) = (&mut self.state, &self.fresh_context);
        let (workspace, config, device) = (&self.workspace, &self.config, &self.device);
        let (selection, readback) = (&self.selection, &self.readback);
        let (quantum, marker) = core
            .lane_table()
            .submit_ops(core.owner_lane(), false, |lane| {
                if step.fresh {
                    state.reset(lane, fresh_context).map_err(|error| error.to_string())?;
                }
                let ple = Arc::new(PleRows::new(state.tracker().context().into()));
                let placement = state
                    .begin_chunk(rows, lane, &core.pool, &mut retained)
                    .map_err(|error| error.to_string())?;
                let mode = SubmitMode {
                    pdl: false,
                    row_exact: false,
                };
                let mut quantum = Quantum::new(
                    step.kind,
                    rows as usize,
                    InputRecordRef::new(u64::from(placement.start())),
                    Arc::clone(workspace),
                    SequenceRefs::new(Arc::clone(device)),
                    mode,
                    Arc::clone(config),
                    core.lane_table().clone(),
                )
                .map_err(|error| error.to_string())?
                .with_tokens(step.tokens.clone())
                .map_err(|error| error.to_string())?
                .with_ple_rows(ple)
                .with_moe_block(block)
                .with_chunk_placement(Arc::new(placement));
                quantum = match step.score {
                    Score::None => quantum.with_logits_rows(0).map_err(|error| error.to_string())?,
                    Score::Device => quantum.with_selection(Arc::clone(selection)),
                    Score::Host => quantum.with_row_readback(Arc::clone(readback)),
                };
                quantum.prepare(&core.pool, lane).map_err(|error| error.to_string())?;
                let mut freed = Ok(());
                core.released.drain(|buffer| {
                    if let Err(error) = lane.free_async(buffer) {
                        freed = Err(error.to_string());
                    }
                });
                freed.map(|()| quantum)
            })
            .map_err(|error| format!("preparing the quantum: {error}"))?;
        let mut quantum = quantum;
        quantum.prepared_after(marker);
        let started = self
            .step
            .start(quantum, Arc::new(AtomicBool::new(false)), &[])
            .map_err(|error| error.to_string())?;
        self.flight = Some(InFlight {
            _started: started,
            _retained: retained,
        });
        Ok(())
    }
}

/// The GDN and attention state the carrier names: none (the config has no layers), made on the owner lane.
fn sequence_device(core: &Core, config: &ModelConfig) -> Result<SequenceDevice, FlashError> {
    let (gdn, _) = core
        .lane_table()
        .submit_ops(core.owner_lane(), false, |kernels| {
            GdnStates::allocate(&core.library, config, 0, kernels)
        })
        .map_err(refuse("the carrier's GDN state"))?;
    let scratch =
        AttentionScratch::allocate(&core.library, config, 1, false, 0).map_err(refuse("the carrier's scratch"))?;
    Ok(SequenceDevice { gdn, scratch })
}

/// What a quantum's retirement tells the owner: the context after its rows, or why it did not complete.
fn retired(quantum: &Quantum, outcome: &QuantumOutcome) -> Retired {
    let result = match conclude(outcome, false) {
        Outcome::Completed => quantum
            .ple_rows()
            .and_then(|rows| rows.next_context())
            .map(Box::from)
            .ok_or_else(|| "the quantum retired without the context after its rows".to_owned()),
        Outcome::Cancelled => Err("the quantum was cancelled".to_owned()),
        Outcome::Failed(why) => Err(why),
    };
    Retired {
        result,
        at: Instant::now(),
    }
}

impl Stepper for DeviceStepper {
    fn begin(&mut self, step: &Step) -> Result<(), String> {
        self.flight = None;
        self.launch(step)
    }

    fn selected(&self) -> Option<u32> {
        self.selection.token(0)
    }

    fn row(&self) -> Option<&[u8]> {
        self.readback.row(0)
    }

    fn commit(&mut self, next_context: &[i32]) -> Result<(), String> {
        self.flight = None;
        self.state
            .commit(next_context)
            .map(|_| ())
            .map_err(|error| error.to_string())
    }

    fn discard(&mut self) {
        self.flight = None;
        self.state.discard();
    }
}
