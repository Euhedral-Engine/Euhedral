//! The device cache of routed experts under the Flash-Next MoE layer: resident experts are leased at once, misses are
//! loaded from the host expert store, and nothing waits on a lock.
//!
//! - `cache`: the public face, [`ExpertCache`]: it posts requests to the owner and closes.
//! - `source`: the owner as a source of the lattice, and the inbox every record is posted to.
//! - `owner`: the bookkeeping that applies the records: loads, waiting requests, staging and the tier.
//! - `shard`: the bookkeeping of the device slots (S3-FIFO over slots that leases pin).
//! - `lease`: the right to use one slot.
//! - `host`: the interfaces to the lake and the copy lane; `lanes` is the engine's implementation of them.
//! - [`stats`]: the counters the engine's metrics read.

mod cache;
mod host;
mod lanes;
mod lease;
mod owner;
mod shard;
mod source;
pub mod stats;

#[cfg(test)]
mod cache_tests;
#[cfg(test)]
mod shard_tests;

pub use cache::{CacheConfig, ExpertCache};
pub use host::{CopyJob, CopyLane, DeviceSlots, Frames, SubmitFailure};
pub use lease::ExpertLease;
pub use owner::{Listener, Outcome};
pub use stats::{CacheSnapshot, CacheStats, Cause};

/// Why the cache refused, or could not do, what it was asked.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum CacheError {
    #[error("the cache holds {keys} experts: it needs at least one and at most {limit}")]
    Keys { keys: usize, limit: usize },
    #[error("the cache has {slots} device slots: it needs at least one and at most {limit}")]
    Slots { slots: usize, limit: usize },
    #[error("the expert cache is closed")]
    Closed,
    #[error("expert key {key} is outside the {keys} the cache holds")]
    Key { key: usize, keys: usize },
    #[error("slot {slot} has no load that {what}")]
    BadLoad { slot: u32, what: &'static str },
    #[error("the device pool has {free} free slots: the cache asked for {wanted}")]
    PoolTooSmall { wanted: usize, free: usize },
    #[error("the {what} slots are {slot_bytes} bytes: a record needs {record}")]
    SlotTooSmall {
        what: &'static str,
        slot_bytes: u64,
        record: u64,
    },
    #[error(
        "the staging free list has {free_list} slots over an arena of {arena}: it needs at least one and at most the arena's"
    )]
    StagingSlots { free_list: usize, arena: usize },
    #[error("{reads} concurrent reads: the cache needs at least one")]
    Reads { reads: usize },
    #[error("the cache was abandoned: {why}")]
    Abandoned { why: String },
    #[error("the cache's owner panicked and serves no more: {why}")]
    Poisoned { why: String },
    #[error("every one of the {lost} staging slots is lost to a failed copy: no load can start")]
    StagingLost { lost: u32 },
    #[error("loading expert {expert} of bank {bank} failed: {why}")]
    Transfer { bank: String, expert: u32, why: String },
    #[error("{loads} expert transfers are still in flight")]
    LoadsInFlight { loads: usize },
}
