//! The residency plan: which objects go where on the device, and whether the model, its cache and its working memory
//! fit the memory the device has (`ResidencyPlanner`).
//!
//! The device budget is what the device has free, at most `memory.gpu-limit`, less `memory.gpu-reserve`. Against it
//! the plan sets the resident weights, the attention cache of a context of `context.max-tokens`, the runtime's own
//! reserve (the first sequence's state, the execution workspaces, the shared scratch and the kernel modules loaded
//! after the model: `ResidencyPlanner.RESERVE_BYTES`) and the prefill scratch of the largest trellis object. When
//! those do not fit, the cache pages past `context.kv-cache-on-gpu-tokens` move to pinned host memory, which the
//! host budget must hold, before anything else is considered. Weights never move: a plan whose weights do not fit
//! is refused with the bytes it needed and the bytes there were.

use euhedral_inference_edrl::{TensorDataType, TensorEntry, TensorTable, WeightFormat, WeightLayout};
use euhedral_inference_settings::{Auto, KvFormat, Settings};

use crate::error::PlanError;
use crate::model::{
    ArtifactProfile, LayerIndex, MTP_ATTENTION_PACK, MTP_PACK_HALVES, ModelConfig, Role, Speculation, object_key,
    trellis_logical, validate_inventory,
};
use crate::residency::footprint::{self, PAGE_TOKENS};
use crate::residency::scratch::{
    MAX_SEGMENTS, TRELLIS_BLOCK, TrellisThresholds, trellis_mma_bytes, trellis_plan_bytes,
};
use crate::workspace::{self, WorkspaceRows};

/// Everything outside the weights and the cache that one running sequence needs on the device
/// (`ResidencyPlanner.RESERVE_BYTES`): its GDN state, the execution workspaces, the shared scratch and the kernel
/// modules the first launches compile.
pub const RUNTIME_RESERVE_BYTES: u64 = 1280 << 20;

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

/// One object on the device: what it is, where its bytes are in the artifact and where they go in the arena.
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
    /// Where the object starts in the weight arena (a multiple of [`ARENA_ALIGNMENT`]).
    pub offset: u64,
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
    /// Every object the load places on the device, in artifact order (the halves of a split projection in place of
    /// the packed one), with its arena offset.
    pub device: Vec<Placement>,
    /// The bytes of the weight arena: the placements, each aligned.
    pub arena_bytes: u64,
    /// The payload bytes of the objects the load places (`ResidencyPlanner.deviceNeed`).
    pub weights_bytes: u64,
    /// Tokens of the context whose cache pages are on the device.
    pub kv_device_tokens: usize,
    /// Tokens of the context whose cache pages are in pinned host memory (0 when every page is on the device).
    pub kv_host_tokens: usize,
    /// Bytes of the device cache of one sequence: the pages of every attention layer, and the one staging region of
    /// the host pages.
    pub kv_device_bytes: u64,
    /// Pinned host bytes of the cache pages past the device share.
    pub kv_host_bytes: u64,
    /// Bytes of the workspace slots at the prefill chunk's rows.
    pub workspace_bytes: u64,
    /// Bytes of the largest linear scratch (the trellis prefill route of the largest object at the chunk's rows).
    pub scratch_bytes: u64,
    /// Bytes of the scratch of the tensor-core route alone, which a decode-type launch running beside another takes
    /// (`LinearFormat.decodeScratchBytes`).
    pub decode_scratch_bytes: u64,
    /// The device bytes the plan reserves in all: the weights, the cache, the runtime's reserve and the scratch.
    pub device_bytes: u64,
    /// The device bytes the plan was made against: free, at most the limit, less the reserve.
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
}

impl ResidencyPlan {
    /// The placement named `name`.
    pub fn placement(&self, name: &str) -> Option<&Placement> {
        self.device.iter().find(|placement| placement.name == name)
    }
}

/// What the plan is made against besides the model: the device and the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlanInputs {
    /// Device bytes free now.
    pub device_free: u64,
    /// The device's streaming multiprocessors; sizes the trellis K split's partials when
    /// `kernels.trellis.target-blocks` is `auto`. `None`: the widest split the shapes allow.
    pub multiprocessors: Option<u32>,
    /// Pinned host bytes the cache pages may take.
    pub host_pinned_bytes: u64,
}

/// Plans residency for `config`'s model in an artifact with `table`, on a device with `device_free` bytes free and
/// no limit on pinned host memory beyond `memory.host-weights-limit` and `memory.host-limit`.
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
        .map_or(u64::MAX, |size| size.bytes());
    plan_with(
        config,
        table,
        settings,
        PlanInputs {
            device_free,
            multiprocessors: None,
            host_pinned_bytes,
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

    let (device, arena_bytes, weights_bytes) = place(table, &profile)?;

    let verified = (1 + (draft_depth as u64).min(MAX_VERIFIED)).max(DRAFT_LOGITS_ROWS);
    let rows = WorkspaceRows {
        rows: chunk,
        logits_rows: verified,
    };
    let workspace_bytes = workspace::workspace_bytes(config, rows);
    let thresholds = trellis_thresholds(settings, inputs.multiprocessors);
    let widest = inputs.multiprocessors.is_none();
    let scratch_bytes = trellis_scratch(table, &profile, chunk, thresholds, widest, Route::Plan);
    let decode_scratch_bytes = trellis_scratch(table, &profile, chunk, thresholds, widest, Route::TensorCore);

    let budget = inputs
        .device_free
        .min(settings.memory.gpu_limit.value().map_or(u64::MAX, |size| size.bytes()));
    let reserve = settings.memory.gpu_reserve.bytes();
    let available = budget.saturating_sub(reserve);
    if available == 0 {
        return Err(PlanError::NoBudget { free: budget, reserve });
    }

    let whole_kv = footprint::kv_bytes(config, max_tokens, kv_format);
    let fixed = whole_kv + RUNTIME_RESERVE_BYTES + scratch_bytes;
    let mut kv_device_tokens = footprint::pages_for(max_tokens) * PAGE_TOKENS;
    let (mut kv_host_tokens, mut kv_device_bytes, mut kv_host_bytes) = (0, whole_kv, 0);
    let mut device_bytes = arena_bytes + fixed;
    if device_bytes > available {
        // The cache past the device minimum moves to host memory before any weight leaves the device: each attention
        // layer's host pages are copied to one device region of the sequence's before the layer reads them.
        let pages = footprint::pages_for(max_tokens);
        let device_pages = pages.min(footprint::pages_for(u64::from(context.kv_cache_on_gpu_tokens)));
        let host_pages = pages - device_pages;
        let page = footprint::page_bytes(config, kv_format);
        let region = host_pages * page;
        let spilled = region * config.attention_layers() as u64;
        if host_pages > 0 && spilled <= inputs.host_pinned_bytes {
            let kept = footprint::kv_bytes(config, device_pages * PAGE_TOKENS, kv_format) + region;
            device_bytes = arena_bytes + (fixed - whole_kv + kept);
            kv_device_tokens = device_pages * PAGE_TOKENS;
            kv_host_tokens = host_pages * PAGE_TOKENS;
            kv_device_bytes = kept;
            kv_host_bytes = spilled;
        }
    }
    if device_bytes > available {
        return Err(PlanError::WeightsDoNotFit {
            weights_needed: arena_bytes,
            state_needed: device_bytes - arena_bytes,
            available,
            free: inputs.device_free,
        });
    }
    Ok(ResidencyPlan {
        device,
        arena_bytes,
        weights_bytes,
        kv_device_tokens: kv_device_tokens as usize,
        kv_host_tokens: kv_host_tokens as usize,
        kv_device_bytes,
        kv_host_bytes,
        workspace_bytes,
        scratch_bytes,
        decode_scratch_bytes,
        device_bytes,
        budget_bytes: available,
        kv_format,
        rows,
        profile,
        draft_depth,
        max_context_tokens: max_tokens as usize,
    })
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
