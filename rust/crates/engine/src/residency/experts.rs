//! The store under the GPU expert cache and the MoE layer: where a routed expert's record is read from when the router
//! picks it, the artifact file or a tier of records in host memory.
//!
//! - [`layout`]: the records of every bank under one flat key, checked against the file's length.
//! - [`arena`]: ordinary host memory in page-aligned slots, with leases (one writer or many readers, refused rather than
//!   waited for) that move with the work that holds them.
//! - [`source`]: a record read from the artifact in page-aligned parts, each straight into its place; the parts' ends
//!   are frames that join into one result.
//! - [`tier`]: the bookkeeping of the host tier (which record a slot holds, what to replace) and its settle records.
//! - [`store`]: one load of a record, by hit, fill or bypass, as the cache owner asks it.
//! - [`preload`]: the startup fill of a tier's slots.
//!
//! The tier's bookkeeping is plain state with one owner (the cache owner of the shard). A load runs elsewhere, on frames,
//! and touches only slot memory; it ends by handing the owner a [`tier::TierSettle`] record to apply.

pub mod arena;
pub mod layout;
pub mod preload;
pub mod source;
pub mod store;
pub mod tier;

#[cfg(test)]
mod tests;

pub use arena::{HostArena, SlotRead, SlotWrite, WritePart};
pub use layout::{ExpertError, ExpertId, ExpertLayout, shard_of};
pub use source::{FilePartPort, FileRecordSource, PartPort, ReadLoad, ReadOutcome, Started};
pub use store::{ExpertStore, LoadEnd, StoreLoad};
pub use tier::{RamTier, RamTierShard, TierDirective, TierSettle, TierStats, VictimSampler};
