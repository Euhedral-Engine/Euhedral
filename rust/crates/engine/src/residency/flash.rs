//! The residency plan of a Flash-Next model: which objects live on the device for good, which are held on the host
//! (read in place, or staged through a device ring), which stay unloaded, and how many device bytes are left for the
//! routed-expert cache (`ResidencyPlanner`, `FLASH_NEXT_RESIDENCY.md`).
//!
//! ```text
//! free device memory
//!   - runtime reserve (CUDA context, kernel modules, graph pools) and memory.gpu-reserve
//!   - workspaces of the execution plan (a decode token, a short chunk, a full chunk)
//!   - sequence state of the maximum context (KV, indexer keys, GDN state)
//!   - fixed objects placed on the device
//!   - the staging ring of fixed objects moved to the host
//!   = the expert cache, in whole slots
//! ```
//!
//! Context has priority: a longer context takes memory from the expert cache first, and only when the cache would fall
//! below its minimum (the selected experts of a token and a second set in flight) does a fixed object move to the
//! host, lowest priority first and the fewest bytes that suffice ([`selection`]). A fixed byte is read on every token
//! while a cached expert is hit only on some, so the plan never moves a fixed object to enlarge the cache.
//!
//! The plan keeps no replacement state and asks nothing of the device: it is arithmetic over the artifact's object
//! sizes, the device's free bytes, the host budget and the settings `memory.gpu-reserve` (kept free beside the
//! runtime's own reserve) and `memory.weight-streaming.*` (`concurrent-reads` sizes the pinned staging of expert
//! records; `gpu-streams` is the device lane count the runtime opens). A budget that cannot hold the minimum cache is
//! refused with the bytes it needed and the bytes there were. The dense model's plan ([`planner`](super::planner))
//! reads none of these settings; the two share the host budget, the consumers the memory planner admits and the
//! staging geometry.

pub mod selection;
pub mod state;

use euhedral_inference_edrl::{ComponentGroup, ExpertBank, FlashTensor};
use euhedral_inference_settings::MemorySettings;

use crate::error::PlanError;
use crate::model::FlashModelConfig;
use crate::residency::accounting::{Consumer, ConsumerBytes};
use crate::residency::budget::{BudgetSource, HostBudget};
use crate::residency::offload::{HOST_ALIGNMENT, SELECTION_STEP, StagingGeometry};
use crate::residency::planner::{ARENA_ALIGNMENT, DEFAULT_STAGING_SLOTS};
use selection::{Movable, Selection, offload_rank, read_in_place, select};
pub use state::{PREFILL_CHUNK_TOKENS, SequenceState};

/// The CUDA context, kernel modules and graph pools that exist beside the model (about 0.9 GiB measured on the dense
/// model's engine). `memory.gpu-reserve` is kept free on top.
pub const KERNEL_RESERVE_BYTES: u64 = 1024 << 20;

/// The largest prefill chunk the plan makes room for.
pub const LARGEST_PREFILL_CHUNK_TOKENS: u64 = 4096;

/// The share of the expert cache's memory that a larger prefill chunk may take: a chunk of many tokens reads most of
/// the experts of every layer once however many tokens it holds, so its cost is amortized over its tokens, and the
/// cache gives up a tenth of its memory at most for that.
const CHUNK_SHARE_DIVISOR: u64 = 10;

/// Where an object lives at runtime and how execution reaches it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StorageClass {
    /// Permanently in device memory.
    DeviceResident,
    /// In pinned host memory that kernels read in place over the bus (a gather: the token embedding).
    HostMapped,
    /// Held on the host; execution stages the object (a fixed projection) or the rows it needs (n-gram tables) to
    /// the device on use.
    HostStaged,
    /// Read from the artifact file's own mapping, outside the pinned tier (the n-gram tables when the host cannot pin
    /// them).
    FileMapped,
    /// Brought into the bounded device cache on demand and evicted when its slot is needed (routed experts).
    DeviceCached,
    /// In the artifact and validated, but not loaded by the selected mode.
    Deferred,
}

/// What a load prepares beyond the text model. The engine derives it from the generation strategy; it is not a user
/// setting.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlashMode {
    pub mtp: bool,
    pub vision: bool,
}

impl FlashMode {
    pub const TEXT: FlashMode = FlashMode {
        mtp: false,
        vision: false,
    };
}

/// How the routed experts are held on the host behind the device cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ExpertStoreMode {
    /// Every record sits in ordinary memory, loaded once at startup.
    RamResident,
    /// A bounded cache of records in ordinary memory, filled from the artifact.
    RamCached,
    /// No RAM tier: every device miss reads its record from the artifact.
    FileBacked,
}

/// How the n-gram tables are held on the host.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NgramMode {
    PinnedArena,
    /// The artifact file mapped read-only; rows are gathered from it.
    MappedFile,
}

/// The size of the routed-expert cache a budget allows: whole slots, each able to hold the largest expert record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertCacheGeometry {
    pub slot_bytes: u64,
    pub slot_count: usize,
    pub total_experts: u64,
    pub minimum_slots: u64,
}

impl ExpertCacheGeometry {
    /// The device bytes the cache holds.
    pub fn bytes(&self) -> u64 {
        self.slot_bytes * self.slot_count as u64
    }

    /// The slots a budget of `budget_bytes` buys, capped at one per expert: more would never be used.
    fn derive(budget_bytes: u64, slot_bytes: u64, total_experts: u64, minimum_slots: u64) -> Self {
        let affordable = budget_bytes / slot_bytes;
        Self {
            slot_bytes,
            slot_count: affordable.min(total_experts) as usize,
            total_experts,
            minimum_slots,
        }
    }
}

/// The pinned host slots every expert record passes through when it is read from the artifact file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ExpertStagingGeometry {
    pub slots: usize,
    /// Each slot holds the largest record, aligned to a block of the file system.
    pub slot_bytes: u64,
}

impl ExpertStagingGeometry {
    pub fn bytes(&self) -> u64 {
        self.slots as u64 * self.slot_bytes
    }
}

/// Device bytes of a plan. `expert_cache` is the whole slots of the cache; `slack` what is left below one slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DeviceBytes {
    pub free: u64,
    pub kv: u64,
    pub indexer: u64,
    pub gdn_state: u64,
    /// The workspaces of the execution plan and the shared expert's padded down projections.
    pub workspace: u64,
    /// The runtime's own reserve and `memory.gpu-reserve`.
    pub runtime_reserve: u64,
    pub fixed_resident: u64,
    pub staging_ring: u64,
    pub expert_cache: u64,
    pub slack: u64,
}

impl DeviceBytes {
    /// What one sequence holds: KV, indexer keys and GDN state.
    pub fn context(&self) -> u64 {
        self.kv + self.indexer + self.gdn_state
    }

    /// Everything the plan places on the device.
    pub fn planned(&self) -> u64 {
        self.context()
            + self.workspace
            + self.runtime_reserve
            + self.fixed_resident
            + self.staging_ring
            + self.expert_cache
    }
}

/// Host bytes by use. Pinned bytes are page-locked; the expert tier is ordinary memory; file bytes are the artifact's
/// own pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct HostBytes {
    pub mapped: u64,
    pub staged: u64,
    pub expert_staging_pinned: u64,
    pub expert_ram: u64,
    pub expert_ram_slots: usize,
    pub expert_file: u64,
    pub ngram_pinned: u64,
    pub ngram_file: u64,
    pub deferred_mtp: u64,
    pub deferred_vision: u64,
}

impl HostBytes {
    pub fn pinned(&self) -> u64 {
        self.mapped + self.staged + self.expert_staging_pinned + self.ngram_pinned
    }

    /// Ordinary memory the plan holds in all: the expert tier and everything pinned.
    pub fn resident(&self) -> u64 {
        self.pinned() + self.expert_ram
    }
}

/// The storage of one fixed object.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ObjectPlacement {
    pub name: String,
    pub group: ComponentGroup,
    pub bytes: u64,
    pub storage: StorageClass,
}

/// The storage of one expert bank.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BankPlacement {
    pub name: String,
    pub experts: usize,
    pub storage: StorageClass,
}

/// Where every part of a Flash-Next model lives for one device, one host and one maximum context.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlashPlan {
    pub max_context_tokens: usize,
    pub mode: FlashMode,
    /// Every fixed object of the artifact, in table order.
    pub objects: Vec<ObjectPlacement>,
    /// Every expert bank, in table order.
    pub banks: Vec<BankPlacement>,
    pub device: DeviceBytes,
    pub host: HostBytes,
    pub expert_cache: ExpertCacheGeometry,
    /// The device ring staged fixed objects pass through; `None` when nothing is staged.
    pub staging: Option<StagingGeometry>,
    pub expert_staging: ExpertStagingGeometry,
    pub expert_store: ExpertStoreMode,
    pub ngram: NgramMode,
    /// The tokens of the full prefill workspace the plan chose.
    pub prefill_chunk_tokens: u64,
    /// The device lanes the runtime opens (`memory.weight-streaming.gpu-streams`).
    pub lanes: u32,
}

impl FlashPlan {
    /// The storage class of the object or bank named `name`.
    pub fn storage_of(&self, name: &str) -> Option<StorageClass> {
        self.objects
            .iter()
            .find(|object| object.name == name)
            .map(|object| object.storage)
            .or_else(|| {
                self.banks
                    .iter()
                    .find(|bank| bank.name == name)
                    .map(|bank| bank.storage)
            })
    }

    /// What the plan asks of the memory planner, by consumer: the same list the dense plan is admitted by.
    pub fn consumers(&self) -> Vec<ConsumerBytes> {
        let device = &self.device;
        let host = &self.host;
        vec![
            ConsumerBytes::device(Consumer::Weights, device.fixed_resident),
            ConsumerBytes::device(Consumer::RuntimeReserve, device.runtime_reserve),
            ConsumerBytes::device(Consumer::Scratch, device.workspace),
            ConsumerBytes::device(Consumer::Cache, device.context()),
            ConsumerBytes::device(Consumer::StagingRing, device.staging_ring),
            ConsumerBytes::device(Consumer::ExpertCache, device.expert_cache),
            ConsumerBytes::host(Consumer::HostWeights, host.mapped + host.staged + host.ngram_pinned),
            ConsumerBytes::host(Consumer::ExpertStaging, host.expert_staging_pinned),
        ]
    }
}

/// What the plan is made against besides the model and the settings: the device, the host and the context.
#[derive(Clone, Copy, Debug)]
pub struct FlashPlanInputs {
    /// Device bytes free now.
    pub device_free: u64,
    pub host: HostBudget,
    pub max_context_tokens: usize,
    pub mode: FlashMode,
}

/// The smallest expert cache that can serve one token: its selected experts plus a second set in flight.
pub fn minimum_slots(config: &FlashModelConfig) -> u64 {
    2 * config.experts_per_token
}

/// Pinned staging slots when expert records pass through the host from the artifact file: one per read the disk has in
/// flight (`concurrent_reads`) but no more than a layer has experts, as each load holds its slot until its copy
/// retires, and at least two tokens' experts.
pub fn file_staging_slots(config: &FlashModelConfig, concurrent_reads: u32) -> u64 {
    u64::from(concurrent_reads)
        .min(config.num_experts)
        .max(minimum_slots(config))
}

/// Plans the residency of an artifact (its `tensors` and `banks`) in `inputs.mode` for a context of
/// `inputs.max_context_tokens` on a device with `inputs.device_free` bytes free, keeping `memory.gpu-reserve` free
/// beside the runtime's own reserve and staging one expert record per read `memory.weight-streaming.concurrent-reads`
/// has in flight.
pub fn plan_flash(
    config: &FlashModelConfig,
    tensors: &[FlashTensor],
    banks: &[ExpertBank],
    memory: &MemorySettings,
    inputs: FlashPlanInputs,
) -> Result<FlashPlan, PlanError> {
    let streaming = memory.weight_streaming;
    if streaming.concurrent_reads == 0 {
        return Err(PlanError::Setting(
            "memory.weight-streaming.concurrent-reads is 0: at least one read must be in flight to stage expert \
             records"
                .to_owned(),
        ));
    }
    if streaming.gpu_streams == 0 {
        return Err(PlanError::Setting(
            "memory.weight-streaming.gpu-streams is 0: the runtime needs at least one device lane".to_owned(),
        ));
    }
    let context = inputs.max_context_tokens;
    if context == 0 {
        return Err(PlanError::Setting(
            "context.max-tokens is 0: a context needs at least 1 token".to_owned(),
        ));
    }
    if context as u64 > config.max_position_embeddings {
        return Err(PlanError::Setting(format!(
            "context.max-tokens of {context} is longer than the {} positions the model supports",
            config.max_position_embeddings
        )));
    }
    config.check_supported()?;
    let free = inputs.device_free;
    let gpu_reserve = memory.gpu_reserve.bytes();
    let runtime_reserve = KERNEL_RESERVE_BYTES.saturating_add(gpu_reserve);
    if runtime_reserve >= free {
        return Err(PlanError::FlashReserveTooLarge {
            gpu_reserve,
            runtime_reserve: KERNEL_RESERVE_BYTES,
            free,
        });
    }
    let mode = inputs.mode;
    let sizes = SequenceState::new(config);
    // `check_supported` proved these sizes for the longest context and chunk, so they exist for this one.
    let sized = |size: Option<u64>| size.ok_or_else(|| PlanError::Unsupported("sizes overflow 64 bits".to_owned()));

    let mut objects: Vec<ObjectPlacement> = tensors
        .iter()
        .map(|tensor| ObjectPlacement {
            name: tensor.name.clone(),
            group: tensor.group,
            bytes: tensor.bytes,
            storage: StorageClass::DeviceResident,
        })
        .collect();
    let (mut fixed, mut embedding) = (Vec::new(), None);
    let (mut ngram_bytes, mut deferred_mtp, mut deferred_vision) = (0u64, 0u64, 0u64);
    for (index, object) in objects.iter_mut().enumerate() {
        match object.group {
            ComponentGroup::Mtp if !mode.mtp => {
                deferred_mtp += object.bytes;
                object.storage = StorageClass::Deferred;
            }
            ComponentGroup::Vision if !mode.vision => {
                deferred_vision += object.bytes;
                object.storage = StorageClass::Deferred;
            }
            ComponentGroup::Ngram => {
                ngram_bytes += object.bytes;
                object.storage = StorageClass::HostStaged;
            }
            ComponentGroup::TokenEmbedding => embedding = Some(index),
            _ => fixed.push(index),
        }
    }

    let mut bank_placements = Vec::with_capacity(banks.len());
    let (mut expert_bytes, mut total_experts, mut slot_bytes) = (0u64, 0u64, 0u64);
    for bank in banks {
        let bytes: u64 = bank.records().iter().map(|record| record.bytes).sum();
        let storage = if bank.group() == ComponentGroup::Mtp && !mode.mtp {
            deferred_mtp += bytes;
            StorageClass::Deferred
        } else {
            expert_bytes += bytes;
            total_experts += bank.expert_count() as u64;
            slot_bytes = slot_bytes.max(bank.max_record_bytes());
            StorageClass::DeviceCached
        };
        bank_placements.push(BankPlacement {
            name: bank.name().to_owned(),
            experts: bank.expert_count(),
            storage,
        });
    }
    let minimum = minimum_slots(config);
    if slot_bytes == 0 || total_experts < minimum {
        return Err(PlanError::Unsupported(format!(
            "the artifact has {total_experts} routed experts of up to {slot_bytes} bytes to cache, fewer than the \
             {minimum} slots of the smallest expert cache"
        )));
    }

    // Device state that does not depend on placement.
    let kv = sized(sizes.kv_bytes(context))?;
    let indexer = sized(sizes.indexer_bytes(context))?;
    let gdn_state = sized(sizes.gdn_state_bytes())?;
    let padding = sized(sizes.shared_down_padding_bytes())?;
    let workspace_of = |chunk: u64| sized(sizes.workspace_bytes(chunk)).map(|bytes| bytes + padding);
    let state = kv + indexer + gdn_state;
    let reserved_with = |workspace: u64| runtime_reserve + workspace + state;

    // The token embedding is a gather: its rows are read in place from pinned host memory.
    let pinnable = inputs.host.pinnable_bytes;
    let embedding_mapped = embedding.is_some_and(|index| objects[index].bytes <= pinnable);
    if let Some(index) = embedding {
        if embedding_mapped {
            objects[index].storage = StorageClass::HostMapped;
        } else {
            fixed.push(index);
        }
    }
    let mapped_embedding = embedding
        .filter(|_| embedding_mapped)
        .map_or(0, |index| objects[index].bytes);

    let fixed_bytes: u64 = fixed.iter().map(|index| objects[*index].bytes).sum();
    let movable: Vec<(usize, Movable<'_>)> = fixed
        .iter()
        .filter_map(|index| {
            let object = &objects[*index];
            let rank = offload_rank(object.group, object.bytes)?;
            (!(mode.mtp && object.group == ComponentGroup::Mtp)).then_some((
                *index,
                Movable {
                    name: &tensors[*index].name,
                    bytes: object.bytes,
                    rank,
                    mapped: read_in_place(object.group),
                },
            ))
        })
        .collect();
    let movable_objects: Vec<Movable<'_>> = movable.iter().map(|(_, object)| *object).collect();
    let movable_bytes: u64 = movable_objects.iter().map(|object| object.bytes).sum();
    let minimum_cache = minimum.saturating_mul(slot_bytes);
    let ring_of = |selection: &Selection| ring_geometry(selection).map_or(0, |ring| ring.bytes());
    let device_need =
        |selection: &Selection| fixed_bytes - selection.staged_bytes - selection.mapped_bytes + ring_of(selection);

    // The fewest bytes moved to the host that leave the minimum cache.
    let base_reserved = reserved_with(workspace_of(PREFILL_CHUNK_TOKENS)?);
    let mut target = 0u64;
    let selection = loop {
        let selection = select(&movable_objects, target);
        let fits = device_need(&selection)
            .saturating_add(base_reserved)
            .saturating_add(minimum_cache)
            <= free;
        if fits {
            break selection;
        }
        if target >= movable_bytes {
            let everything = select(&movable_objects, movable_bytes);
            return Err(PlanError::FlashDeviceTooSmall {
                context_tokens: context as u64,
                free,
                state_bytes: state + workspace_of(PREFILL_CHUNK_TOKENS)?,
                runtime_reserve,
                fixed_stay_bytes: fixed_bytes - movable_bytes,
                fixed_after_move_bytes: device_need(&everything),
                ring_bytes: ring_of(&everything),
                minimum_slots: minimum,
                minimum_cache_bytes: minimum_cache,
                needed_bytes: device_need(&everything)
                    .saturating_add(base_reserved)
                    .saturating_add(minimum_cache),
            });
        }
        target += SELECTION_STEP;
    };
    let staging = ring_geometry(&selection);
    let ring = staging.map_or(0, |ring| ring.bytes());
    let need = device_need(&selection);
    let resident = need - ring;
    for index in &selection.taken {
        let (object_index, object) = &movable[*index];
        objects[*object_index].storage = if object.mapped {
            StorageClass::HostMapped
        } else {
            StorageClass::HostStaged
        };
    }

    // A longer chunk spends the device's memory on workspace instead of expert slots. A prefill chunk reads most of the
    // experts of every layer once, whatever its length, so the longest chunk the cache can afford to give up memory
    // for is the one that serves a prompt best. A decode token needs none of it.
    let budget_at_base = free - need - base_reserved;
    let base = workspace_of(PREFILL_CHUNK_TOKENS)?;
    let mut chunk = PREFILL_CHUNK_TOKENS;
    let mut candidate = chunk * 2;
    while candidate <= LARGEST_PREFILL_CHUNK_TOKENS && candidate / 2 < context as u64 {
        let extra = workspace_of(candidate)?.saturating_sub(base);
        if extra > budget_at_base / CHUNK_SHARE_DIVISOR || budget_at_base.saturating_sub(extra) < minimum_cache {
            break;
        }
        chunk = candidate;
        candidate *= 2;
    }
    let workspace = workspace_of(chunk)?;
    let cache_budget = free - need - reserved_with(workspace);
    let geometry = ExpertCacheGeometry::derive(cache_budget, slot_bytes, total_experts, minimum);
    let slack = cache_budget - geometry.bytes();

    // The host. Pinned memory is a small tier: the gathered embedding, staged fixed objects, and the staging slots every
    // expert record passes through. The routed experts themselves live in ordinary memory, in what the resident budget
    // has left after the pinned part (pinned memory comes out of the same physical memory).
    let expert_staging = ExpertStagingGeometry {
        slots: file_staging_slots(config, streaming.concurrent_reads) as usize,
        slot_bytes: slot_bytes.next_multiple_of(HOST_ALIGNMENT),
    };
    let pinned = mapped_embedding + selection.staged_bytes + selection.mapped_bytes + expert_staging.bytes();
    if pinned > pinnable {
        return Err(match inputs.host.source {
            BudgetSource::Unknown => PlanError::HostBudgetUnknown { needed: pinned },
            _ => PlanError::FlashHostTooSmall {
                needed_bytes: pinned,
                expert_staging_bytes: expert_staging.bytes(),
                pinnable_bytes: pinnable,
            },
        });
    }
    let pinned_left = pinnable - pinned;
    let resident_left = inputs.host.resident_bytes.saturating_sub(pinned);
    let ram_slot_bytes = expert_staging.slot_bytes;
    let mut ram_slots = (resident_left / ram_slot_bytes).min(total_experts);
    let expert_store = if ram_slots >= total_experts {
        ExpertStoreMode::RamResident
    } else if ram_slots > geometry.slot_count as u64 {
        // A tier no larger than the device cache would only repeat it.
        ExpertStoreMode::RamCached
    } else {
        ram_slots = 0;
        ExpertStoreMode::FileBacked
    };
    let expert_ram = ram_slots * ram_slot_bytes;
    // The tables are sparse gathers: they are pinned only when every expert is already resident and memory is left
    // over; otherwise they are read from the mapped file, and what is left serves the experts' page cache.
    let ngram_pinned_fits = expert_store == ExpertStoreMode::RamResident
        && ngram_bytes <= pinned_left
        && ngram_bytes <= resident_left - expert_ram;
    let (ngram, ngram_pinned) = if ngram_pinned_fits {
        (NgramMode::PinnedArena, ngram_bytes)
    } else {
        for object in objects
            .iter_mut()
            .filter(|object| object.group == ComponentGroup::Ngram)
        {
            object.storage = StorageClass::FileMapped;
        }
        (NgramMode::MappedFile, 0)
    };

    Ok(FlashPlan {
        max_context_tokens: context,
        mode,
        objects,
        banks: bank_placements,
        device: DeviceBytes {
            free,
            kv,
            indexer,
            gdn_state,
            workspace,
            runtime_reserve,
            fixed_resident: resident,
            staging_ring: ring,
            expert_cache: geometry.bytes(),
            slack,
        },
        host: HostBytes {
            mapped: mapped_embedding + selection.mapped_bytes,
            staged: selection.staged_bytes,
            expert_staging_pinned: expert_staging.bytes(),
            expert_ram,
            expert_ram_slots: ram_slots as usize,
            expert_file: expert_bytes,
            ngram_pinned,
            ngram_file: ngram_bytes - ngram_pinned,
            deferred_mtp,
            deferred_vision,
        },
        expert_cache: geometry,
        staging,
        expert_staging,
        expert_store,
        ngram,
        prefill_chunk_tokens: chunk,
        lanes: streaming.gpu_streams,
    })
}

/// The ring that stages the objects `selection` moved: [`DEFAULT_STAGING_SLOTS`] slots of the largest one, aligned like
/// a device allocation; `None` when nothing is staged.
fn ring_geometry(selection: &Selection) -> Option<StagingGeometry> {
    (selection.staged_max > 0).then(|| StagingGeometry {
        slots: DEFAULT_STAGING_SLOTS as usize,
        slot_bytes: selection.staged_max.next_multiple_of(ARENA_ALIGNMENT),
    })
}

#[cfg(test)]
mod tests;
