//! The device cache of routed experts under the Flash-Next MoE layer: resident experts are leased at once, misses are
//! loaded from the host expert store, and nothing waits on a lock.
//!
//! - [`stats`]: the counters the engine's metrics read.
//! - `shard`: the bookkeeping of the device slots (S3-FIFO over slots that leases pin).

mod shard;
pub mod stats;

#[cfg(test)]
mod shard_tests;

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
    #[error("{loads} expert transfers are still in flight")]
    LoadsInFlight { loads: usize },
}
