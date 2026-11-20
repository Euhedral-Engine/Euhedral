//! The residency plan: which objects go where on the device, and whether the model, its cache and its working memory
//! fit the memory the device has (`ResidencyPlanner`).
//!
//! The device budget is what the device has free, at most `memory.gpu-limit` (`InferenceEngine.deviceBudget`; the
//! dense model does not use `memory.gpu-reserve`). Against it
//! the plan sets the resident weights, the attention cache of a context of `context.max-tokens`, the runtime's own
//! reserve (the first sequence's state, the execution workspaces, the shared scratch and the kernel modules loaded
//! after the model: `ResidencyPlanner.RESERVE_BYTES`) and the prefill scratch of the largest trellis object. When
//! those do not fit, the cache pages past `context.kv-cache-on-gpu-tokens` move to pinned host memory, which the
//! host budget must hold, before anything else is considered. Then weights leave the device
//! ([`offload`](crate::residency::offload)), the output head last: they are held in pinned host memory or read from
//! the artifact file. A plan that does not fit even then is refused with the bytes it needed and the bytes there
//! were, naming the memory that is short.

use euhedral_inference_cuda::launch::nvfp4::{SKINNY_MAX_ROWS, nvfp4_native_scratch_bytes, skinny_target_ctas};
use euhedral_inference_cuda::launch::q3::{P2E2_EXPAND_LIMIT, linear_scratch_bytes, q3_byte_size};
use euhedral_inference_edrl::{TensorDataType, TensorEntry, TensorTable, WeightFormat, WeightLayout};
use euhedral_inference_settings::{Auto, BatchingSettings, KvFormat, Settings};

use euhedral_memory::{
    LocationId, LocationKind, LocationSpec, OwnerSpec, Placed, PlanRequest, Planner, Priority, Scope,
};

use crate::error::{HostFigures, PlanError};
use crate::model::{
    ArtifactProfile, LayerIndex, MTP_ATTENTION_PACK, MTP_PACK_HALVES, ModelConfig, Role, Speculation, object_key,
    trellis_logical, validate_inventory,
};
use crate::residency::accounting::{Consumer, ConsumerBytes, plan_items};
use crate::residency::footprint::{self, PAGE_TOKENS};
use crate::residency::offload::{self, Offload, OffloadInputs, Reach};
use crate::residency::scratch::{
    MAX_SEGMENTS, TRELLIS_BLOCK, TrellisThresholds, trellis_mma_bytes, trellis_plan_bytes,
};
use crate::workspace::{self, WorkspaceRows};

/// What `memory.runtime-reserve: auto` means: Java's `ResidencyPlanner.RESERVE_BYTES`. The reserve is everything
/// outside the weights and the cache that one running sequence needs on the device: its GDN state, the execution
/// workspaces, the shared scratch and the kernel modules the first launches compile.
pub const DEFAULT_RUNTIME_RESERVE_BYTES: u64 = 1280 << 20;

/// The multiprocessor count of the device the headroom was measured on.
pub const HEADROOM_MEASURED_MULTIPROCESSORS: u32 = 70;

/// What `memory.runtime-headroom: auto` means on a device of [`HEADROOM_MEASURED_MULTIPROCESSORS`] multiprocessors,
/// and the least it means on any device: the device bytes kept above the demand that raises the runtime reserve, for
/// what the driver takes after admission. Measured on 70 multiprocessors with drafting on, BF16 KV, 8192 tokens and
/// the reserve at its tightest: 246 MiB lost beyond the engine's allocations after admission (118 MiB by the time the
/// model was ready, and 128 MiB at the first prefill, the code and local memory of the trellis, attention and Gated
/// DeltaNet kernels); this is that figure with about 30% over. The local memory grows with the multiprocessors, so a
/// wider device scales it ([`auto_runtime_headroom`]).
pub const DEFAULT_RUNTIME_HEADROOM_BYTES: u64 = 320 << 20;

/// The headroom `auto` resolves to: [`DEFAULT_RUNTIME_HEADROOM_BYTES`] scaled by the device's multiprocessors over
/// [`HEADROOM_MEASURED_MULTIPROCESSORS`] and rounded up, never less than the default. A count that is unknown (the
/// plan made without a device) gets the default.
pub fn auto_runtime_headroom(multiprocessors: Option<u32>) -> u64 {
    let Some(count) = multiprocessors else {
        return DEFAULT_RUNTIME_HEADROOM_BYTES;
    };
    let scaled = (u128::from(DEFAULT_RUNTIME_HEADROOM_BYTES) * u128::from(count))
        .div_ceil(u128::from(HEADROOM_MEASURED_MULTIPROCESSORS));
    u64::try_from(scaled)
        .unwrap_or(u64::MAX)
        .max(DEFAULT_RUNTIME_HEADROOM_BYTES)
}

/// Device allocations start on this many bytes, like the driver's.
pub const ARENA_ALIGNMENT: u64 = 256;

/// The rows a draft quantum may have beyond a verification's: the most the decode row twins run
/// (`AttentionStates.DRAFT_TWIN_ROWS`). The logits of the workspace hold at least this many rows.
const DRAFT_LOGITS_ROWS: u64 = footprint::DRAFT_TWIN_ROWS;

/// The most drafts one verification checks (`MtpDecoder.MAX_VERIFIED`).
pub const MAX_VERIFIED: u64 = 7;

/// Where an object's bytes come from in the artifact.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlacementSource {
    /// The object's whole payload, as the artifact stores it.
    Object,
    /// Rows `first_row..first_row + rows` of the packed MTP attention projection `object`.
    Rows {
        object: String,
        first_row: u64,
        object_rows: u64,
    },
}

/// Where an object's bytes are held while the model runs.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tier {
    /// In the device arena, at the placement's `offset`.
    Device,
    /// In the pinned host arena, at `offset` (a multiple of [`HOST_ALIGNMENT`](offload::HOST_ALIGNMENT)).
    Host { offset: u64 },
    /// In the artifact file, read through the disk window.
    Disk,
}

/// One object of the model: what it is, where its bytes are in the artifact and where they are held.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Placement {
    /// The artifact's name for the object (the name of the half for a split MTP projection).
    pub name: String,
    pub source: PlacementSource,
    pub shape: Vec<u64>,
    pub dtype: TensorDataType,
    pub format: WeightFormat,
    pub layout: WeightLayout,
    /// The first payload byte of the source object in the artifact.
    pub file_offset: u64,
    /// The bytes the object occupies on the device.
    pub bytes: u64,
    /// Where the object starts in the device weight arena (a multiple of [`ARENA_ALIGNMENT`]); 0 for an object that
    /// is not on the device, whose place is in `tier`.
    pub offset: u64,
    /// Where the object is held.
    pub tier: Tier,
}

impl Placement {
    /// The `(layer, role)` the object is found by, if it is an object and not a companion.
    pub fn key(&self, layers: usize) -> Option<(LayerIndex, Role)> {
        object_key(&self.name, layers)
    }
}

/// The plan: where every object goes, and how much of everything else fits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ResidencyPlan {
    /// Every object the load places, in artifact order (the halves of a split projection in place of the packed
    /// one), with its tier and, on the device, its arena offset.
    pub device: Vec<Placement>,
    /// The bytes of the device weight arena: the placements on the device, each aligned.
    pub arena_bytes: u64,
    /// The payload bytes of the objects on the device.
    pub weights_bytes: u64,
    /// What leaves the device: the host and disk sets and the memory that takes their place. Empty when the plan is
    /// resident.
    pub offload: Offload,
    /// Tokens of the context whose cache pages are on the device.
    pub kv_device_tokens: usize,
    /// Tokens of the context whose cache pages are in pinned host memory (0 when every page is on the device).
    pub kv_host_tokens: usize,
    /// Bytes of the device cache of one sequence: the pages of every attention layer, and the one staging region of
    /// the host pages.
    pub kv_device_bytes: u64,
    /// Pinned host bytes of the cache pages past the device share, of every sequence.
    pub kv_host_bytes: u64,
    /// The sequences the plan holds.
    pub sequences: usize,
    /// The device bytes each sequence beyond the first needs besides its cache ([`footprint::extra_sequence_bytes`]);
    /// 0 for a plan of one sequence.
    pub extra_sequence_bytes: u64,
    /// Bytes of the workspace slots at the prefill chunk's rows.
    pub workspace_bytes: u64,
    /// Bytes of the largest linear scratch (the trellis prefill route of the largest object at the chunk's rows).
    pub scratch_bytes: u64,
    /// Bytes of the scratch of the tensor-core route alone, which a decode-type launch running beside another takes
    /// (`LinearFormat.decodeScratchBytes`).
    pub decode_scratch_bytes: u64,
    /// The device bytes the runtime reserves beyond the weights, the cache and the scratch (`memory.runtime-reserve`):
    /// the execution workspace, the decode scratch, the first sequence's state, the MTP layer's cache at the longest
    /// context and what the kernel modules take. At least what the settings make the first four of these need, and
    /// `memory.runtime-headroom` above them for the kernels, whatever the setting says: the startup's claims take the
    /// first four, so the headroom is left.
    pub runtime_reserve_bytes: u64,
    /// The device bytes the plan reserves in all: the weights, every sequence's cache, the runtime's reserve, the
    /// scratch and the state of the sequences beyond the first.
    pub device_bytes: u64,
    /// The device bytes the plan was made against: free, at most `memory.gpu-limit`.
    pub budget_bytes: u64,
    /// The cache format.
    pub kv_format: KvFormat,
    /// The rows of the workspace.
    pub rows: WorkspaceRows,
    /// What the projections and the drafter make of the artifact.
    pub profile: ArtifactProfile,
    /// Drafts per verification (0 without a drafter).
    pub draft_depth: usize,
    /// The longest context of one sequence.
    pub max_context_tokens: usize,
    /// Pinned host bytes the prefix cache's arena may take: what `prefix-cache.size` asks, when the memory planner
    /// places it beside everything else; 0 when the cache is off or the host has no room for it.
    pub prefix_cache_bytes: u64,
    /// The pinned host bytes the plan was made against (`PlanInputs::host_pinned_bytes`); `None` when the budget was
    /// unknown, which allowed no spill. The account admits the plan against it.
    pub host_budget_bytes: Option<u64>,
    /// The most rows one shared pass holds, which the workspace's logits rows are sized for.
    pub pass_rows: usize,
    /// The most sequences one shared pass holds.
    pub pass_parts: usize,
}

/// The rows a shared pass holds by the settings: `batching.max-tokens-per-pass`, at most a prefill chunk (which is its
/// default).
pub fn pass_rows_limit(batching: &BatchingSettings) -> usize {
    let chunk = batching.prefill_chunk_tokens;
    batching
        .max_tokens_per_pass
        .into_value()
        .map_or(chunk, |rows| rows.min(chunk)) as usize
}

/// What one shared pass of the verifications of `sequences` sequences of `verified` rows each holds at most: its rows
/// and its sequences, as the settings bound them. Never zero.
pub fn pass_bounds(batching: &BatchingSettings, sequences: usize, verified: u64) -> (usize, usize) {
    let parts = batching
        .max_sequences_per_pass
        .into_value()
        .map_or(sequences, |parts| sequences.min(parts as usize))
        .max(1);
    let wanted = (parts as u64).saturating_mul(verified).max(1);
    let rows = pass_rows_limit(batching)
        .min(usize::try_from(wanted).unwrap_or(usize::MAX))
        .max(1);
    (rows, parts)
}

impl ResidencyPlan {
    /// The bytes the plan asks of each location: what `plan_with` had the memory planner admit.
    pub fn location_bytes(&self) -> LocationBytes {
        LocationBytes {
            device: self.device_bytes,
            host: self.kv_host_bytes + self.offload.host_bytes(),
        }
    }

    /// The payload bytes of the objects held outside the device, host and disk both.
    pub fn offloaded_bytes(&self) -> u64 {
        self.device
            .iter()
            .filter(|placement| placement.tier != Tier::Device)
            .map(|placement| placement.bytes)
            .sum()
    }

    /// What the plan asks, by consumer: the list the memory planner admitted and the account's reservation is made of.
    /// Its device bytes sum to `device_bytes`, its host bytes to `kv_host_bytes`, the offload's host bytes and
    /// `prefix_cache_bytes`.
    pub fn consumers(&self) -> Vec<ConsumerBytes> {
        Layout {
            weights: self.arena_bytes,
            runtime_reserve: self.runtime_reserve_bytes,
            scratch: self.scratch_bytes,
            cache_device: self.kv_device_bytes,
            cache_host: self.kv_host_bytes,
            sequences: self.sequences as u64,
            extra_sequence_bytes: self.extra_sequence_bytes,
            staging_ring: self.offload.staging_bytes(),
            host_weights: self.offload.host_arena_bytes,
            disk_window: self.offload.window_bytes(),
        }
        .consumers(self.prefix_cache_bytes)
    }

    /// The placement named `name`.
    pub fn placement(&self, name: &str) -> Option<&Placement> {
        self.device.iter().find(|placement| placement.name == name)
    }

    /// The placements held on the device.
    pub fn on_device(&self) -> impl Iterator<Item = &Placement> {
        self.device.iter().filter(|placement| placement.tier == Tier::Device)
    }
}

/// The bytes a plan asks of the device and of pinned host memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LocationBytes {
    pub device: u64,
    pub host: u64,
}

/// What the plan is made against besides the model: the device and the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanInputs {
    /// Device bytes free now.
    pub device_free: u64,
    /// The device's streaming multiprocessors; sizes the trellis K split's partials when
    /// `kernels.trellis.target-blocks` is `auto`. `None`: the widest split the shapes allow.
    pub multiprocessors: Option<u32>,
    /// Pinned host bytes the cache pages may take. `None`: not known, and no spill is allowed (the prefix cache still
    /// keeps its stated size).
    pub host_pinned_bytes: Option<u64>,
    /// The sequences to size for (at least 1).
    pub sequences: usize,
}

/// Plans residency for `config`'s model in an artifact with `table`, on a device with `device_free` bytes free and
/// no limit on pinned host memory beyond `memory.host-weights-limit` and `memory.host-limit`. With neither set the
/// host budget is taken as known and unlimited (`Some(u64::MAX)`), not unknown: this entry point does not read the
/// machine, so a spill is never refused here for want of a figure.
pub fn plan(
    config: &ModelConfig,
    table: &TensorTable,
    settings: &Settings,
    device_free: u64,
) -> Result<ResidencyPlan, PlanError> {
    let host_pinned_bytes = settings
        .memory
        .host_weights_limit
        .value()
        .or(settings.memory.host_limit.value())
        .map_or(Some(u64::MAX), |size| Some(size.bytes()));
    plan_with(
        config,
        table,
        settings,
        PlanInputs {
            device_free,
            multiprocessors: None,
            host_pinned_bytes,
            sequences: 1,
        },
    )
}

/// Plans residency against `inputs` (see [`plan`]).
pub fn plan_with(
    config: &ModelConfig,
    table: &TensorTable,
    settings: &Settings,
    inputs: PlanInputs,
) -> Result<ResidencyPlan, PlanError> {
    match plan_closest(config, table, settings, inputs)? {
        (plan, None) => Ok(plan),
        (_, Some(refusal)) => Err(refusal),
    }
}

/// Plans residency against `inputs` as [`plan_with`] does, and when the plan is refused for want of memory returns the
/// plan that came closest (everything offloadable offloaded) beside the refusal, so that what the refusal is about can
/// be read: the sets, the cache pages and the bytes. The plan beside a refusal is never to be run.
pub fn plan_closest(
    config: &ModelConfig,
    table: &TensorTable,
    settings: &Settings,
    inputs: PlanInputs,
) -> Result<(ResidencyPlan, Option<PlanError>), PlanError> {
    config.check_supported()?;
    validate_inventory(config, table)?;
    let profile = ArtifactProfile::of(table, config)?;
    let draft_depth = profile.speculative_depth()?;
    if draft_depth > 0 && profile.speculation != Speculation::Mtp {
        return Err(PlanError::Unsupported(
            "the engine drafts with the artifact's MTP layer only".to_string(),
        ));
    }
    let context = settings.context;
    let max_tokens = u64::from(context.max_tokens);
    if max_tokens == 0 {
        return Err(PlanError::Setting("context.max-tokens must be positive".to_string()));
    }
    if max_tokens > config.max_position_embeddings as u64 {
        return Err(PlanError::Setting(format!(
            "a context of {max_tokens} tokens is longer than the {} positions the model supports; set a smaller \
             context.max-tokens",
            config.max_position_embeddings
        )));
    }
    if context.kv_cache_on_gpu_tokens == 0 {
        return Err(PlanError::Setting(
            "context.kv-cache-on-gpu-tokens must be positive".to_string(),
        ));
    }
    if inputs.sequences == 0 {
        return Err(PlanError::Setting("the plan needs at least one sequence".to_string()));
    }
    let sequences = inputs.sequences as u64;
    let chunk = u64::from(settings.batching.prefill_chunk_tokens);
    if chunk == 0 {
        return Err(PlanError::Setting(
            "batching.prefill-chunk-tokens must be positive".to_string(),
        ));
    }
    let kv_format = context
        .kv_cache_format
        .into_value()
        .unwrap_or_else(|| profile.kv_format());

    let (mut device, arena_all, _) = place(table, &profile)?;

    let verified = (1 + (draft_depth as u64).min(MAX_VERIFIED)).max(DRAFT_LOGITS_ROWS);
    // A pass of several sequences' verifications scores every row of every member: its logits rows are the pass's.
    let (pass_rows, pass_parts) = pass_bounds(
        &settings.batching,
        inputs.sequences,
        1 + (draft_depth as u64).min(MAX_VERIFIED),
    );
    let pass_logits = if draft_depth > 0 && pass_parts > 1 {
        pass_rows as u64
    } else {
        0
    };
    let rows = WorkspaceRows {
        rows: chunk,
        logits_rows: verified.max(pass_logits),
    };
    let workspace_bytes = workspace::workspace_bytes_with_records(
        config,
        rows,
        workspace::input_records_for(draft_depth, inputs.sequences),
    );
    let thresholds = trellis_thresholds(settings, inputs.multiprocessors);
    let widest = inputs.multiprocessors.is_none();
    let scratch_bytes = trellis_scratch(table, &profile, chunk, thresholds, widest, Route::Plan)
        .max(nvfp4_native_scratch(
            &device,
            chunk,
            inputs.multiprocessors.map_or(u64::MAX, skinny_target_ctas),
        ))
        .max(grouped_scratch(
            &device,
            config.num_hidden_layers,
            chunk,
            rows.logits_rows,
            inputs.multiprocessors,
        )?);
    let decode_scratch_bytes = trellis_scratch(table, &profile, chunk, thresholds, widest, Route::TensorCore);

    let available = inputs
        .device_free
        .min(settings.memory.gpu_limit.value().map_or(u64::MAX, |size| size.bytes()));
    let reserve_setting = settings
        .memory
        .runtime_reserve
        .value()
        .map_or(DEFAULT_RUNTIME_RESERVE_BYTES, |size| size.bytes());

    let whole_kv = footprint::kv_bytes(config, max_tokens, kv_format);
    let pages = footprint::pages_for(max_tokens);
    let device_pages = pages.min(footprint::pages_for(u64::from(context.kv_cache_on_gpu_tokens)));
    let host_pages = pages - device_pages;
    let page = footprint::page_bytes(config, kv_format);
    let region = host_pages * page;
    let spilled = (region * config.attention_layers() as u64).saturating_mul(sequences);
    let verified_rows = if draft_depth > 0 { 1 + draft_depth as u64 } else { 0 };
    // The runtime reserve pays for what the settings make grow: the workspace's slots (with the rows of a pass), the
    // decode scratch, the first sequence's state and the MTP layer's cache at the longest context. When they are
    // more than the setting, the reserve is what they need: admission counts it and the account refuses less.
    let first_state = footprint::first_state_bytes(config, kv_format, verified_rows, draft_depth > 0, chunk);
    let mtp_cache = if draft_depth > 0 {
        footprint::mtp_cache_bytes(config, kv_format, max_tokens)
    } else {
        0
    };
    // The kernels the driver loads after admission (their code and local memory, at each one's first launch) are paid
    // from what the claims leave. A reserve raised to the claims alone would leave them nothing, so the raised reserve
    // keeps the headroom above them.
    let headroom = settings
        .memory
        .runtime_headroom
        .value()
        .map_or_else(|| auto_runtime_headroom(inputs.multiprocessors), |size| size.bytes());
    let reserve_demand = workspace_bytes
        .saturating_add(decode_scratch_bytes)
        .saturating_add(first_state)
        .saturating_add(mtp_cache)
        .saturating_add(headroom);
    let runtime_reserve = reserve_setting.max(reserve_demand);
    if reserve_demand > reserve_setting {
        log::info!(
            "the runtime reserve is raised from {reserve_setting} to {reserve_demand} bytes: the workspace, the decode \
             scratch, the first sequence's state and the MTP cache need that much at these settings, and {headroom} \
             bytes are kept above them for the kernels loaded after admission"
        );
    }
    let extra = if sequences > 1 {
        footprint::extra_sequence_bytes(config, kv_format, max_tokens, verified_rows, draft_depth > 0, chunk)
    } else {
        0
    };
    let kept = footprint::kv_bytes(config, device_pages * PAGE_TOKENS, kv_format) + region;
    let original_host = inputs.host_pinned_bytes.unwrap_or(u64::MAX);
    let budgets = Budgets {
        device: available,
        host: original_host,
    };
    let whole = Layout {
        weights: arena_all,
        runtime_reserve,
        scratch: scratch_bytes,
        cache_device: whole_kv,
        cache_host: 0,
        sequences,
        extra_sequence_bytes: extra,
        staging_ring: 0,
        host_weights: 0,
        disk_window: 0,
    };
    let split = (host_pages > 0).then_some(Layout {
        cache_device: kept,
        cache_host: spilled,
        ..whole
    });
    let staging_slots = settings
        .memory
        .staging_slots
        .into_value()
        .unwrap_or(DEFAULT_STAGING_SLOTS) as usize;
    let window_slots = settings
        .memory
        .disk_window_slots
        .into_value()
        .map_or(2 * staging_slots, |slots| slots as usize);
    if staging_slots == 0 || !window_slots.is_multiple_of(staging_slots) {
        return Err(PlanError::Setting(format!(
            "memory.disk-window-slots ({window_slots}) must be a whole multiple of memory.staging-slots \
             ({staging_slots}): the window is read one staging ring ahead, whole rings at a time"
        )));
    }
    let head_rows = sequences.saturating_mul(1 + draft_depth as u64);
    let (layout, spills, offload, reach) = match decide(budgets, whole, split) {
        Ok((layout, spills)) => (layout, spills, Offload::default(), None),
        Err(_) => {
            // The pages go to the host first, when it holds them; then weights leave the device.
            let (base, spills) = match split.filter(|split| split.cache_host <= budgets.host) {
                Some(split) => (split, true),
                None => (whole, false),
            };
            let objects = offload::placed_objects(&device);
            let reach = offload::offload(
                &objects,
                &OffloadInputs {
                    device_free: available,
                    fixed_bytes: base.device_bytes() - base.weights,
                    host_budget: original_host.saturating_sub(base.cache_host),
                    staging_slots,
                    window_slots,
                    head_rows,
                },
            );
            let layout = Layout {
                weights: reach.resident_bytes,
                staging_ring: reach.offload.staging_bytes(),
                host_weights: reach.offload.host_arena_bytes,
                disk_window: reach.offload.window_bytes(),
                ..base
            };
            (layout, spills, reach.offload.clone(), Some(reach))
        }
    };
    let refusal = match &reach {
        Some(reach) if !reach.fits() => Some(refusal_of(reach, &layout, inputs, available, max_tokens, original_host)),
        Some(_) if inputs.host_pinned_bytes.is_none() => Some(PlanError::HostBudgetUnknown {
            needed: layout.host_bytes(),
        }),
        None if spills && inputs.host_pinned_bytes.is_none() => Some(PlanError::HostBudgetUnknown { needed: spilled }),
        _ => None,
    };
    let (kv_device_tokens, kv_host_tokens) = if spills {
        (device_pages * PAGE_TOKENS, host_pages * PAGE_TOKENS)
    } else {
        (pages * PAGE_TOKENS, 0)
    };
    let (kv_device_bytes, kv_host_bytes) = (layout.cache_device, layout.cache_host);
    let device_bytes = layout.device_bytes();
    let asked = settings.prefix_cache.size.bytes();
    let prefix_cache_bytes = if asked > 0 && config.layer_kinds.len() > 1 && refusal.is_none() {
        shrink_to_fit(budgets, layout, asked)
    } else {
        0
    };
    let weights_bytes = assign(&mut device, &offload);
    Ok((
        ResidencyPlan {
            device,
            arena_bytes: layout.weights,
            weights_bytes,
            offload,
            kv_device_tokens: kv_device_tokens as usize,
            kv_host_tokens: kv_host_tokens as usize,
            kv_device_bytes,
            kv_host_bytes,
            sequences: inputs.sequences,
            extra_sequence_bytes: extra,
            workspace_bytes,
            scratch_bytes,
            decode_scratch_bytes,
            runtime_reserve_bytes: layout.runtime_reserve,
            device_bytes,
            budget_bytes: available,
            kv_format,
            rows,
            profile,
            draft_depth,
            max_context_tokens: max_tokens as usize,
            prefix_cache_bytes,
            host_budget_bytes: inputs.host_pinned_bytes,
            pass_rows,
            pass_parts,
        },
        refusal,
    ))
}

/// What a plan that does not fit is refused for. The device fits and the host is short (the arena, the window and the
/// cache pages put on the host together are more than the host budget): more host memory is what would help, and the
/// refusal is the host's. When the device does not fit even with everything offloaded, it is the device's, and its
/// message names the host shortfall too when there is one.
fn refusal_of(
    reach: &Reach,
    layout: &Layout,
    inputs: PlanInputs,
    available: u64,
    max_tokens: u64,
    original_host: u64,
) -> PlanError {
    let needed = reach.host_bytes().saturating_add(layout.cache_host);
    if reach.device_fits && needed > original_host {
        return PlanError::HostTooSmall {
            needed,
            cache_pages: layout.cache_host,
            budget: original_host,
        };
    }
    PlanError::DeviceTooSmallWithOffload {
        weights_needed: layout.weights + layout.staging_ring,
        state_needed: layout.device_bytes() - layout.weights - layout.staging_ring,
        available,
        free: inputs.device_free,
        sequences: inputs.sequences,
        context_tokens: max_tokens,
        host: HostFigures {
            weights: reach.host_bytes(),
            cache_pages: layout.cache_host,
            budget: original_host,
        },
    }
}

/// The slots of the staging ring when `memory.staging-slots` is `auto`.
///
/// Not a setting of its own: it is what `auto` means, and a machine that wants another count sets the setting.
pub const DEFAULT_STAGING_SLOTS: u32 = 4;

/// Gives every placement its tier and its place there: device objects take the arena in artifact order, each aligned to
/// [`ARENA_ALIGNMENT`]; host objects take the host arena in artifact order, each aligned to
/// [`HOST_ALIGNMENT`](offload::HOST_ALIGNMENT). Returns the payload bytes of the device objects.
fn assign(placements: &mut [Placement], offload: &Offload) -> u64 {
    let (mut device, mut host, mut payload) = (0u64, 0u64, 0u64);
    for placement in placements {
        if offload.holds_on_host(&placement.name) {
            placement.offset = 0;
            placement.tier = Tier::Host { offset: host };
            host += placement.bytes.div_ceil(offload::HOST_ALIGNMENT) * offload::HOST_ALIGNMENT;
        } else if offload.holds_on_disk(&placement.name) {
            placement.offset = 0;
            placement.tier = Tier::Disk;
        } else {
            placement.offset = device;
            placement.tier = Tier::Device;
            device += align(placement.bytes);
            payload += placement.bytes;
        }
    }
    debug_assert_eq!(host, offload.host_arena_bytes);
    payload
}

/// The largest count of sequences, at most `cap`, whose plan against `inputs` (its `sequences` is not read) fits
/// without offloading more of the weights than a plan of one sequence does: 1 when `cap` is below 1. The fit only
/// worsens with the count, so the search halves the range. The refusal of a plan of
/// one sequence is the answer when even one does not fit, so a device too small for the model says so here.
pub fn sequences_for(
    config: &ModelConfig,
    table: &TensorTable,
    settings: &Settings,
    inputs: PlanInputs,
    cap: usize,
) -> Result<usize, PlanError> {
    let with = |sequences: usize| plan_with(config, table, settings, PlanInputs { sequences, ..inputs });
    // More sequences never offload more than one does: what leaves the device is the cost of every token, and the
    // sequences share it.
    let offloaded = with(1)?.offloaded_bytes();
    let (mut low, mut high) = (1, cap.max(1));
    while low < high {
        let middle = low + (high - low).div_ceil(2);
        match with(middle) {
            Ok(plan) if plan.offloaded_bytes() <= offloaded => low = middle,
            Ok(_)
            | Err(
                PlanError::HostTooSmall { .. }
                | PlanError::DeviceTooSmallWithOffload { .. }
                | PlanError::HostBudgetUnknown { .. },
            ) => high = middle - 1,
            Err(other) => return Err(other),
        }
    }
    Ok(low)
}

/// An unknown or unlimited budget is given to the memory planner as this many bytes, or as the bytes the plan asks of
/// that location if that is more, so the capacity stays below `u64::MAX / 2` for any plan the sums can hold and the
/// answer is the same as with the budget itself.
pub(crate) const CAPACITY_CAP: u64 = u64::MAX / 4;

/// The two memory spaces of a plan: pinned host memory first, so a refusal at the host is reported before one at the
/// device. When both are short the result relies on `Planner::admit` reporting the lowest location index, which
/// RustEE does not promise; the unit test `a_refusal_reports_the_split_layout_only_when_the_host_holds_its_pages`
/// (host and device both short) guards it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Location {
    Host,
    Device,
}

#[derive(Clone, Copy)]
struct Budgets {
    device: u64,
    host: u64,
}

/// What a plan asks of the two locations, by consumer.
#[derive(Clone, Copy)]
struct Layout {
    weights: u64,
    runtime_reserve: u64,
    scratch: u64,
    /// The device cache of one sequence.
    cache_device: u64,
    /// The pinned host pages of every sequence.
    cache_host: u64,
    sequences: u64,
    /// What each sequence beyond the first needs besides its cache.
    extra_sequence_bytes: u64,
    /// The device ring that weights outside the device are copied through.
    staging_ring: u64,
    /// The pinned host arena of the weights held on the host.
    host_weights: u64,
    /// The pinned host window of the weights read from the file.
    disk_window: u64,
}

impl Layout {
    fn device_bytes(&self) -> u64 {
        self.weights
            .saturating_add(self.runtime_reserve)
            .saturating_add(self.scratch)
            .saturating_add(self.cache_device.saturating_mul(self.sequences))
            .saturating_add(self.extra_sequence_bytes.saturating_mul(self.sequences - 1))
            .saturating_add(self.staging_ring)
    }

    /// The pinned host bytes the plan must have: the cache pages, the weights and the window (not the prefix arena,
    /// which the plan may drop).
    fn host_bytes(&self) -> u64 {
        self.cache_host
            .saturating_add(self.host_weights)
            .saturating_add(self.disk_window)
    }

    /// The consumers of the layout in the order the memory planner admits them, with the prefix cache's arena of
    /// `prefix` bytes (the only item that is not required) last when there is one. The one list both the admission
    /// of a plan and the account of its allocations read.
    fn consumers(&self, prefix: u64) -> Vec<ConsumerBytes> {
        let mut consumers = vec![
            ConsumerBytes::device(Consumer::Weights, self.weights),
            ConsumerBytes::device(Consumer::RuntimeReserve, self.runtime_reserve),
            ConsumerBytes::device(Consumer::Scratch, self.scratch),
            ConsumerBytes::device(Consumer::Cache, self.cache_device.saturating_mul(self.sequences)),
            ConsumerBytes::device(
                Consumer::ExtraSequences,
                self.extra_sequence_bytes.saturating_mul(self.sequences - 1),
            ),
            ConsumerBytes::device(Consumer::StagingRing, self.staging_ring),
            ConsumerBytes::host(Consumer::HostPages, self.cache_host),
            ConsumerBytes::host(Consumer::HostWeights, self.host_weights),
            ConsumerBytes::host(Consumer::DiskWindow, self.disk_window),
        ];
        if prefix > 0 {
            consumers.push(ConsumerBytes::optional_host(Consumer::PrefixCache, prefix));
        }
        consumers
    }
}

/// Everything on the device first; when that is refused and there is a `split` (the cache pages past the device
/// minimum on pinned host memory), the split before any weight leaves the device: each attention layer's host pages
/// are copied to one device region of the sequence's before the layer reads them. The error is the layout whose
/// numbers the refusal reports (`Ok` says whether the cache spilled): the split one when the host holds its pages and
/// the device is short, else the whole.
fn decide(budgets: Budgets, whole: Layout, split: Option<Layout>) -> Result<(Layout, bool), Layout> {
    if admit(budgets, whole).is_ok() {
        return Ok((whole, false));
    }
    match split.map(|split| (split, admit(budgets, split))) {
        Some((split, Ok(()))) => Ok((split, true)),
        Some((split, Err(Location::Device))) => Err(split),
        Some((_, Err(Location::Host))) | None => Err(whole),
    }
}

/// Asks the memory planner whether `layout` fits `budgets`: every consumer is required. The planner is the plan's own
/// and holds nothing after this returns. The error is the first location that is short.
fn admit(budgets: Budgets, layout: Layout) -> Result<(), Location> {
    admit_with_cache(budgets, layout, 0).map(|_| ())
}

/// The largest arena of at most `asked` bytes that the host has room for beside the plan's own pages: `asked` when it
/// fits, else what is left (with a warning), else 0 (the engine then runs without the cache).
fn shrink_to_fit(budgets: Budgets, layout: Layout, asked: u64) -> u64 {
    if let Ok(true) = admit_with_cache(budgets, layout, asked) {
        return asked;
    }
    let room = budgets.host.saturating_sub(layout.host_bytes()).min(asked);
    if room > 0
        && let Ok(true) = admit_with_cache(budgets, layout, room)
    {
        log::warn!("the host has room for {room} of the {asked} bytes asked for the prefix cache: it is that size");
        return room;
    }
    log::warn!("the host has no room for the {asked}-byte prefix cache: the engine runs without it");
    0
}

/// As [`admit`], with the prefix cache's arena of `prefix` bytes as the lowest-priority item and the only one that is
/// not required (0: no arena): the planner drops it before it refuses the plan. Says whether the arena stays.
fn admit_with_cache(budgets: Budgets, layout: Layout, prefix: u64) -> Result<bool, Location> {
    let planner = Planner::with_owners(
        [
            LocationSpec {
                kind: LocationKind::HostPinned,
                capacity: budgets
                    .host
                    .min(CAPACITY_CAP.max(layout.host_bytes().saturating_add(prefix))),
                reserve: 0,
            },
            LocationSpec {
                kind: LocationKind::Device { ordinal: 0 },
                capacity: budgets.device.min(CAPACITY_CAP.max(layout.device_bytes())),
                reserve: 0,
            },
        ],
        1,
    );
    let host = LocationId(0);
    let device = LocationId(1);
    let owner = planner.open_owner(OwnerSpec::new("residency", Priority(0), Scope::Plan(0)));
    let items = plan_items(&layout.consumers(prefix), host, device);
    match planner.admit(&PlanRequest { owner, items }) {
        Ok(admission) => Ok(prefix > 0 && admission.placed.last() != Some(&Placed::Dropped)),
        Err(shortfall) if shortfall.location == host => Err(Location::Host),
        Err(_) => Err(Location::Device),
    }
}

/// Which of a trellis object's routes a scratch is for.
#[derive(Clone, Copy)]
enum Route {
    /// Every route the object may take at the rows: the tensor-core route and, above its row limit, the prefill route.
    Plan,
    /// The tensor-core route alone.
    TensorCore,
}

/// The thresholds of the trellis routes (`TrellisSettings.of`): the target blocks are 8 per multiprocessor unless
/// stated.
fn trellis_thresholds(settings: &Settings, multiprocessors: Option<u32>) -> TrellisThresholds {
    let trellis = settings.kernels.trellis;
    let target = match (trellis.target_blocks, multiprocessors) {
        (Auto::Value(blocks), _) => u64::from(blocks),
        (Auto::Auto, Some(multiprocessors)) => 8 * u64::from(multiprocessors),
        (Auto::Auto, None) => u64::MAX / 2,
    };
    TrellisThresholds {
        mma_max_rows: u64::from(trellis.tensor_core_max_tokens),
        target_ctas: target,
        whole_k_rows: u64::from(trellis.whole_k_min_tokens),
    }
}

/// The scratch the largest trellis object needs at `rows` (`ResidencyPlanner.trellisScratchBytes`); 0 for an
/// artifact with no trellis object. With the device unknown (`widest`), the K split is the widest the shape allows,
/// which bounds the partials whatever the device.
fn trellis_scratch(
    table: &TensorTable,
    profile: &ArtifactProfile,
    rows: u64,
    thresholds: TrellisThresholds,
    widest: bool,
    route: Route,
) -> u64 {
    let mut largest = 0;
    for entry in table.entries() {
        if entry.format() != WeightFormat::Trellis || !profile.uploads(entry.name()) {
            continue;
        }
        let Some((n, k)) = trellis_logical(entry.shape()) else {
            continue;
        };
        let thresholds = if widest {
            TrellisThresholds {
                target_ctas: (k / 256).max(1) * (n / TRELLIS_BLOCK).max(1),
                ..thresholds
            }
        } else {
            thresholds
        };
        largest = largest.max(match route {
            Route::Plan => trellis_plan_bytes(thresholds, rows, k, n),
            Route::TensorCore => trellis_mma_bytes(thresholds, rows, k, n, MAX_SEGMENTS),
        });
    }
    largest
}

/// The largest scratch the NVFP4 native route needs for any NVFP4 object of `placements` in a launch of up to `rows`
/// rows: the one the workspace's first scratch region is sized by, when it is larger than the trellis routes'. Beyond
/// 64 rows the tile route's scratch is smaller than the skinny route's at 64, so the answer is the larger of the two
/// (as `Nvfp4Linear::scratch_bytes` has it). `target_ctas` is the skinny kernels' grid on the device, the most any
/// device splits to where it is not known.
fn nvfp4_native_scratch(placements: &[Placement], rows: u64, target_ctas: u64) -> u64 {
    let skinny_rows = rows.min(u64::from(SKINNY_MAX_ROWS));
    placements
        .iter()
        .filter(|placement| placement.format == WeightFormat::Nvfp4)
        .filter_map(|placement| match placement.shape.as_slice() {
            [out, input] => Some(
                nvfp4_native_scratch_bytes(rows, *input, *out, target_ctas).max(nvfp4_native_scratch_bytes(
                    skinny_rows,
                    *input,
                    *out,
                    target_ctas,
                )),
            ),
            _ => None,
        })
        .max()
        .unwrap_or(0)
}

/// The largest scratch any Q3, Q4 or Q5 object of `placements` takes in a launch of up to `rows` rows (the heads, of
/// up to `logits_rows`): the FP8 route's activations, behind the expansion of a P2E2 tensor. The embedding table is
/// looked up, not multiplied. The FP8 route is counted on every device, and where the device is not known
/// (`multiprocessors` is `None`) with the most K splits any device makes.
///
/// A P2E2 tensor that a launch of this many rows must expand, and that is larger than the scratch takes whole
/// ([`P2E2_EXPAND_LIMIT`]), is refused by name: its run in chunks of output rows is not ported, and the launch would
/// fail at its first use.
fn grouped_scratch(
    placements: &[Placement],
    layers: usize,
    rows: u64,
    logits_rows: u64,
    multiprocessors: Option<u32>,
) -> Result<u64, PlanError> {
    let mut largest = 0;
    for placement in placements.iter().filter(|placement| {
        matches!(
            placement.format,
            WeightFormat::Q3G64Fp16 | WeightFormat::Q4G64Fp16 | WeightFormat::Q5G64Fp16
        )
    }) {
        let rows = match placement.key(layers).map(|(_, role)| role) {
            Some(Role::Embedding) => continue,
            Some(Role::LmHead | Role::DraftHead) => logits_rows,
            _ => rows,
        };
        let [out, input] = placement.shape.as_slice() else {
            continue;
        };
        let (Ok(out), Ok(input)) = (u32::try_from(*out), u32::try_from(*input)) else {
            continue;
        };
        let p2e2 = placement.layout == WeightLayout::RowSplitP2e2V1;
        // The production numerics: the exact kernels expand a tensor the fused decode reads in place, which a tensor
        // of this size cannot do whole (and every other P2E2 tensor is sized below at the prefill chunk's rows).
        let needed = linear_scratch_bytes(
            p2e2,
            u32::try_from(rows).unwrap_or(u32::MAX),
            input,
            out,
            true,
            multiprocessors,
            false,
        );
        match needed {
            Some(bytes) => largest = largest.max(bytes),
            None => {
                return Err(PlanError::Unsupported(format!(
                    "{} is a compressed Q3 tensor of {} bytes expanded, more than the {P2E2_EXPAND_LIMIT} bytes a launch \
                     of {rows} rows can expand into the scratch",
                    placement.name,
                    q3_byte_size(out, input),
                )));
            }
        }
    }
    Ok(largest)
}

fn align(value: u64) -> u64 {
    value.div_ceil(ARENA_ALIGNMENT) * ARENA_ALIGNMENT
}

/// Places every object the load uploads: in artifact order, each aligned in one arena, the packed MTP attention
/// projection as its two halves.
fn place(table: &TensorTable, profile: &ArtifactProfile) -> Result<(Vec<Placement>, u64, u64), PlanError> {
    let mut wanted: Vec<&TensorEntry> = table
        .entries()
        .iter()
        .filter(|entry| profile.uploads(entry.name()))
        .collect();
    wanted.sort_by_key(|entry| entry.offset());
    let mut placements = Vec::with_capacity(wanted.len() + 1);
    let (mut next, mut payload) = (0u64, 0u64);
    let mut put = |placement: &mut Placement| {
        placement.offset = next;
        next += align(placement.bytes);
        payload += placement.bytes;
    };
    for entry in wanted {
        if entry.name() == MTP_ATTENTION_PACK {
            for mut half in split_pack(entry)? {
                put(&mut half);
                placements.push(half);
            }
            continue;
        }
        let mut placement = Placement {
            name: entry.name().to_string(),
            source: PlacementSource::Object,
            shape: entry.shape().to_vec(),
            dtype: entry.dtype(),
            format: entry.format(),
            layout: entry.layout(),
            file_offset: entry.offset(),
            bytes: entry.bytes(),
            offset: 0,
            tier: Tier::Device,
        };
        put(&mut placement);
        placements.push(placement);
    }
    Ok((placements, next, payload))
}

/// The two halves of the packed MTP attention projection (the base layers' query and key rows, then their gate and
/// value rows), as the `NVFP4` row-split layouts cut them (`Nvfp4Format.rows`): the same bytes of rows, the same
/// shared global scale.
fn split_pack(entry: &TensorEntry) -> Result<[Placement; 2], PlanError> {
    let unsupported = |why: String| PlanError::Unsupported(format!("{MTP_ATTENTION_PACK}: {why}"));
    let (WeightFormat::Nvfp4, [rows, k]) = (entry.format(), entry.shape()) else {
        return Err(unsupported(format!(
            "the packed projection must be a rank-2 NVFP4 tensor to split into its halves, not {:?} {:?}",
            entry.format(),
            entry.shape()
        )));
    };
    if rows % 2 != 0 {
        return Err(unsupported("the packed projection has an odd row count".to_string()));
    }
    let half = rows / 2;
    let layout = entry.layout();
    let bytes = nvfp4::byte_size(half, *k, layout)
        .ok_or_else(|| unsupported(format!("layout {layout:?} is not an NVFP4 row-split layout")))?;
    let names = ["mtp/layer/attention/query_key", "mtp/layer/attention/gate_value"];
    debug_assert_eq!(MTP_PACK_HALVES.len(), names.len());
    let make = |index: usize| Placement {
        name: names[index].to_string(),
        source: PlacementSource::Rows {
            object: entry.name().to_string(),
            first_row: index as u64 * half,
            object_rows: *rows,
        },
        shape: vec![half, *k],
        dtype: entry.dtype(),
        format: entry.format(),
        layout,
        file_offset: entry.offset(),
        bytes,
        offset: 0,
        tier: Tier::Device,
    };
    Ok([make(0), make(1)])
}

/// The row-split NVFP4 geometry (`Nvfp4Layout`): rows of K values, K padded to a multiple of 128; the code plane,
/// the scale plane at the next 256 bytes, the trailer at the next 256 bytes after it.
pub mod nvfp4 {
    use euhedral_inference_edrl::WeightLayout;

    const PLANE_ALIGNMENT: u64 = 256;
    const K_ALIGNMENT: u64 = 128;
    /// Values per block scale.
    pub const BLOCK: u64 = 16;
    /// Entries of an SD4 tensor's scale table.
    pub const TABLE: u64 = 16;

    fn align(value: u64) -> u64 {
        value.div_ceil(PLANE_ALIGNMENT) * PLANE_ALIGNMENT
    }

    /// `k` padded to a multiple of 128.
    pub fn padded_k(k: u64) -> u64 {
        k.div_ceil(K_ALIGNMENT) * K_ALIGNMENT
    }

    fn sd4(layout: WeightLayout) -> Option<bool> {
        match layout {
            WeightLayout::RowSplitK128V1 => Some(false),
            WeightLayout::RowSplitK128Sd4V1 => Some(true),
            _ => None,
        }
    }

    /// Bytes of the code plane per row.
    pub fn row_code_bytes(k: u64) -> u64 {
        padded_k(k) / 2
    }

    /// Bytes of the scale plane per row: one per block, or (SD4) one per two blocks.
    pub fn row_scale_bytes(k: u64, layout: WeightLayout) -> Option<u64> {
        Some(padded_k(k) / if sd4(layout)? { 2 * BLOCK } else { BLOCK })
    }

    /// Where the scale plane starts.
    pub fn scale_offset(rows: u64, k: u64) -> u64 {
        align(rows * padded_k(k) / 2)
    }

    /// Where the trailer starts: the global scale, or (SD4) the scale table followed by the global scale.
    pub fn trailer_offset(rows: u64, k: u64, layout: WeightLayout) -> Option<u64> {
        Some(align(scale_offset(rows, k) + rows * row_scale_bytes(k, layout)?))
    }

    /// Bytes of the trailer.
    pub fn trailer_bytes(layout: WeightLayout) -> Option<u64> {
        Some(if sd4(layout)? { TABLE } else { 0 } + 4)
    }

    /// Bytes of a tensor of `rows` rows of `k` values.
    pub fn byte_size(rows: u64, k: u64, layout: WeightLayout) -> Option<u64> {
        Some(trailer_offset(rows, k, layout)? + trailer_bytes(layout)?)
    }
}

#[cfg(test)]
mod tests {
    use std::slice;

    use super::*;

    fn nvfp4(shape: &[u64]) -> Placement {
        Placement {
            name: "object".to_string(),
            source: PlacementSource::Object,
            shape: shape.to_vec(),
            dtype: TensorDataType::Bf16,
            format: WeightFormat::Nvfp4,
            layout: WeightLayout::RowInterleavedNvfp4V1,
            file_offset: 0,
            bytes: 0,
            offset: 0,
            tier: Tier::Device,
        }
    }

    fn p2e2(name: &str, shape: &[u64]) -> Placement {
        Placement {
            name: name.to_string(),
            format: WeightFormat::Q3G64Fp16,
            layout: WeightLayout::RowSplitP2e2V1,
            ..nvfp4(shape)
        }
    }

    /// A compressed tensor too large to expand whole is refused by name, with the limit, wherever a launch of its rows
    /// must expand it; at the rows its fused decode reads it in place it is sized, not refused.
    #[test]
    fn a_compressed_tensor_above_the_expansion_limit_is_refused_by_name_where_it_must_be_expanded() {
        let head = p2e2("text/draft_head", &[262_144, 5120]);
        let error = grouped_scratch(slice::from_ref(&head), 64, 8, 9, Some(70)).unwrap_err();
        let message = error.to_string();
        assert!(message.contains("text/draft_head"), "{message}");
        assert!(message.contains(&P2E2_EXPAND_LIMIT.to_string()), "{message}");
        assert!(grouped_scratch(slice::from_ref(&head), 64, 8, 8, Some(70)).is_ok());
    }

    /// An artifact with no trellis objects has the native route's scratch as its first region: the largest of its
    /// NVFP4 objects at the chunk's rows, which only the NVFP4 objects decide.
    #[test]
    fn the_native_scratch_is_the_largest_of_the_nvfp4_objects_and_nothing_else_decides_it() {
        let objects = [nvfp4(&[4096, 1024]), nvfp4(&[1024, 17_408]), nvfp4(&[8, 8, 8])];
        let target = skinny_target_ctas(70);
        let scratch = nvfp4_native_scratch(&objects, 512, target);
        assert_eq!(
            scratch,
            nvfp4_native_scratch_bytes(512, 1024, 4096, target)
                .max(nvfp4_native_scratch_bytes(512, 17_408, 1024, target))
        );
        assert!(scratch > 0);
        assert!(nvfp4_native_scratch(&objects, 2048, target) > scratch);
        // Past 64 rows the skinny route's partials at 64 rows are larger than the tile route's scratch: a chunk of 65
        // rows is sized by them, as is the linear's own answer.
        let wide = [nvfp4(&[5120, 17_408])];
        let skinny = nvfp4_native_scratch_bytes(64, 17_408, 5120, target);
        assert!(skinny > nvfp4_native_scratch_bytes(65, 17_408, 5120, target));
        assert_eq!(nvfp4_native_scratch(&wide, 65, target), skinny);
        assert_eq!(nvfp4_native_scratch(&wide, 64, target), skinny);
        // A device that is not known is sized for the widest split.
        assert!(nvfp4_native_scratch(&objects, 8, u64::MAX) >= nvfp4_native_scratch(&objects, 8, target));
        let mut other = nvfp4(&[4096, 1024]);
        other.format = WeightFormat::Trellis;
        assert_eq!(nvfp4_native_scratch(&[other], 512, 140), 0);
        assert_eq!(nvfp4_native_scratch(&[], 512, 140), 0);
    }

    /// A model of 1550 device bytes with the whole cache on the device, and the same with a part of the cache
    /// (250 bytes of the device's 400) moved to 300 bytes of pinned host pages.
    fn layouts() -> (Layout, Layout) {
        let whole = Layout {
            weights: 1000,
            runtime_reserve: 100,
            scratch: 50,
            cache_device: 400,
            cache_host: 0,
            sequences: 1,
            extra_sequence_bytes: 0,
            staging_ring: 0,
            host_weights: 0,
            disk_window: 0,
        };
        let split = Layout {
            cache_device: 150,
            cache_host: 300,
            ..whole
        };
        (whole, split)
    }

    #[test]
    fn the_whole_layout_is_chosen_whenever_the_device_holds_it() {
        let (whole, split) = layouts();
        assert_eq!(whole.device_bytes(), 1550);
        assert_eq!(split.device_bytes(), 1300);
        for host in [0, 300, u64::MAX] {
            for with_split in [None, Some(split)] {
                let budgets = Budgets { device: 1550, host };
                let Ok((layout, spills)) = decide(budgets, whole, with_split) else {
                    panic!("the device holds the whole layout at host {host}");
                };
                assert!(!spills && layout.cache_host == 0 && layout.cache_device == 400);
            }
        }
    }

    #[test]
    fn the_cache_spills_when_the_device_is_short_and_the_host_holds_the_pages() {
        let (whole, split) = layouts();
        for (device, host) in [(1549, 300), (1300, 300), (1300, u64::MAX), (u64::MAX - 1, u64::MAX)] {
            let Ok((layout, spilled)) = decide(Budgets { device, host }, whole, Some(split)) else {
                panic!("device {device}, host {host} holds the split layout");
            };
            let short = device < 1550;
            assert_eq!(spilled, short, "device {device}, host {host}");
            assert_eq!(layout.cache_host, if short { 300 } else { 0 });
        }
    }

    #[test]
    fn a_refusal_reports_the_split_layout_only_when_the_host_holds_its_pages() {
        let (whole, split) = layouts();
        // The device is short of the split, the host holds its pages: the split's numbers are reported.
        let Err(reported) = decide(
            Budgets {
                device: 1299,
                host: 300,
            },
            whole,
            Some(split),
        ) else {
            panic!("1299 device bytes cannot hold the split layout");
        };
        assert_eq!((reported.cache_device, reported.cache_host), (150, 300));
        // The host is short of the pages by a byte: the whole layout's numbers are reported, whatever the device.
        for device in [0, 1300, 1549] {
            let Err(reported) = decide(Budgets { device, host: 299 }, whole, Some(split)) else {
                panic!("299 host bytes cannot hold the pages");
            };
            assert_eq!(
                (reported.cache_device, reported.cache_host),
                (400, 0),
                "device {device}"
            );
        }
        // With no split the whole layout is refused one byte short of it.
        let Err(reported) = decide(
            Budgets {
                device: 1549,
                host: u64::MAX,
            },
            whole,
            None,
        ) else {
            panic!("1549 device bytes cannot hold 1550");
        };
        assert_eq!(reported.cache_device, 400);
    }

    #[test]
    fn a_need_above_the_cap_is_judged_against_the_budget_itself() {
        let (mut whole, _) = layouts();
        whole.runtime_reserve = u64::MAX / 4 + 1;
        let need = whole.device_bytes();
        for device in [u64::MAX, need] {
            assert!(
                decide(Budgets { device, host: u64::MAX }, whole, None).is_ok(),
                "device {device}"
            );
        }
        assert!(
            decide(
                Budgets {
                    device: need - 1,
                    host: u64::MAX
                },
                whole,
                None
            )
            .is_err()
        );
    }

    #[test]
    fn unknown_budgets_do_not_overflow() {
        let whole = Layout {
            weights: 1 << 40,
            runtime_reserve: 1 << 30,
            scratch: 1 << 30,
            cache_device: 1 << 36,
            cache_host: 0,
            sequences: 1,
            extra_sequence_bytes: 0,
            staging_ring: 0,
            host_weights: 0,
            disk_window: 0,
        };
        let budgets = Budgets {
            device: u64::MAX,
            host: u64::MAX,
        };
        assert!(matches!(decide(budgets, whole, None), Ok((_, false))));
    }
}
