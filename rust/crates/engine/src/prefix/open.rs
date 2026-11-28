//! Opening the prefix cache at startup: the pinned arena, its slots and the cache over them.

use std::sync::Arc;

use euhedral_inference_cuda::CudaLibrary;
use euhedral_inference_settings::{KvFormat, PrefixCacheSettings};
use euhedral_memory::{Claim, LocationKind, LocationSpec, OwnerSpec, Planner, Priority, Scope, SlotArena};

use crate::generation::Spawn;
use crate::model::ModelConfig;
use crate::model::dflash2::Dflash2Config;
use crate::prefix::cache::{CacheParts, PrefixCacheHandle};
use crate::prefix::copier::{FIRST_COPY_SERIAL, PinnedArena};
use crate::prefix::layout::CheckpointLayout;
use crate::prefix::tree::Speculation;

/// What the cache is opened with.
#[derive(Debug)]
pub struct CacheSpec<'a> {
    pub library: &'a Arc<CudaLibrary>,
    pub config: &'a ModelConfig,
    /// The DFlash2 drafter whose rings the sequences keep, when they draft with it: its checkpoints hold them.
    pub drafter: Option<&'a Dflash2Config>,
    /// The sequences draft with the model's MTP layer: its checkpoints hold the layer's cache.
    pub mtp: bool,
    pub format: KvFormat,
    pub settings: PrefixCacheSettings,
    /// The pinned bytes the memory plan granted the arena; 0 for none.
    pub granted: u64,
    /// Tokens a prefill chunk takes.
    pub chunk: u32,
    /// The tokens of a sequence's cache that are on the device: no checkpoint reaches past them.
    pub device_tokens: u32,
}

/// Opens the cache the plan granted memory for. `None` when it is off (no bytes granted, or a model with no checkpoint
/// layout), and, with a warning, when the memory cannot be pinned or the settings do not fit the model: the engine then
/// serves without it. Allocates pinned memory with the waiting allocator: startup only.
///
/// The arena's bytes are accounted to a planner of the cache's own: the engine does not use this, it opens the cache
/// with [`open_prefix_cache_claimed`] and the claim of its own account.
pub fn open_prefix_cache(spec: CacheSpec<'_>, spawn: Spawn) -> Option<PrefixCacheHandle> {
    open_with(spec, spawn, None)
}

/// As [`open_prefix_cache`], the arena's bytes accounted to `granted`, the claim of the engine's account for the
/// `spec.granted` bytes the plan set aside. The claim is held for as long as the cache is, and what the arena does
/// not use of it (the slots are whole) goes back at once. It is dropped when the cache is not opened.
pub fn open_prefix_cache_claimed(spec: CacheSpec<'_>, spawn: Spawn, granted: Claim) -> Option<PrefixCacheHandle> {
    open_with(spec, spawn, Some(granted))
}

fn open_with(spec: CacheSpec<'_>, spawn: Spawn, granted: Option<Claim>) -> Option<PrefixCacheHandle> {
    if spec.granted == 0 || spec.config.layer_kinds.len() <= 1 {
        return None;
    }
    let mut layout = CheckpointLayout::new(spec.config, spec.format)?;
    if let Some(drafter) = spec.drafter {
        layout = match layout.with_drafter(drafter) {
            Ok(layout) => layout,
            Err(error) => {
                log::warn!("the prefix cache is off: {error}");
                return None;
            }
        };
    }
    let slot_bytes = layout.slot_bytes();
    let slots = u32::try_from(spec.granted / slot_bytes).unwrap_or(u32::MAX);
    if slots == 0 {
        log::warn!(
            "the prefix cache's {} bytes do not hold one {slot_bytes}-byte checkpoint slot: the engine runs without it",
            spec.granted
        );
        return None;
    }
    let bytes = u64::from(slots) * slot_bytes;
    let claim = match granted {
        Some(mut claim) => {
            claim.shrink(claim.bytes().saturating_sub(bytes));
            claim
        }
        None => local_claim(bytes)?,
    };
    let host = match PinnedArena::allocate(spec.library, bytes) {
        // The claim stays with the pinned memory, which the copy jobs share: it is released when the last holder is gone.
        Ok(host) => host.with_claim(claim),
        Err(error) => {
            log::warn!("the prefix cache's {bytes} bytes cannot be pinned ({error}): the engine runs without it");
            return None;
        }
    };
    // The slots' lease book accounts for no bytes: the memory is the pinned arena's.
    let arena = SlotArena::new(0, slot_bytes, slots, local_claim(0)?);
    let parts = CacheParts {
        layout,
        arena,
        host: Arc::new(host),
        format: spec.format,
        interval: spec.settings.checkpoint_tokens,
        chunk: spec.chunk,
        drafting: match (spec.drafter, spec.mtp) {
            (Some(_), _) => Some(Speculation::Dflash2),
            (None, true) => Some(Speculation::Mtp),
            (None, false) => None,
        },
        piece_bytes: spec.settings.copy_piece_bytes.bytes(),
        first_serial: FIRST_COPY_SERIAL,
    };
    match PrefixCacheHandle::open(parts, spawn) {
        Ok(cache) => {
            cache.limit_checkpoints(spec.device_tokens);
            Some(cache)
        }
        Err(error) => {
            log::warn!("the prefix cache does not open ({error}): the engine runs without it");
            None
        }
    }
}

/// A claim of `bytes` on a planner of its own: the engine's plan has already placed the bytes.
fn local_claim(bytes: u64) -> Option<Claim> {
    let planner = Planner::new([LocationSpec {
        kind: LocationKind::HostPinned,
        capacity: bytes,
        reserve: 0,
    }]);
    let location = planner.location(LocationKind::HostPinned)?;
    let owner = planner.open_owner(OwnerSpec::new("prefix cache", Priority(0), Scope::Plan(0)));
    planner.claim(owner, location, bytes, "prefix cache").ok()
}
