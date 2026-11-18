//! The prefix cache's host-side parts: a radix tree of checkpoints over token ids, and the layout of one checkpoint
//! in the slots of a host arena.
//!
//! A checkpoint is a snapshot of a live sequence at a token position: the GDN state at that position, the KV pages of
//! the span since its parent's position, and optionally the speculative state. It lives in slots leased from one
//! [`SlotArena`]; the tree owns the leases and the layout says what each slot holds and which device
//! buffer each piece of it is copied from or to.

#[cfg(doc)]
use euhedral_memory::SlotArena;

#[allow(dead_code, reason = "the runtime opens the cache once it is wired in")]
mod cache;
mod copier;
#[allow(dead_code, reason = "the runtime opens the cache once it is wired in")]
mod copy;
mod layout;
mod open;
mod tree;

pub use cache::{CacheParts, Hit, OpenError, PrefixCacheHandle, PrefixCacheStats, RestoreError, Submit};
pub use copier::{Copier, PinnedArena, SequenceAddresses};
pub use copy::{CopyJob, Direction, Ended, HostArena, Piece, Resolve};
pub use layout::{Buffer, CheckpointLayout, SlotCopy};
pub(crate) use open::{CacheSpec, open_prefix_cache};
pub use tree::{Match, NodeId, POSITION_GRANULE, PrefixNode, PrefixTree, ReserveError, Speculation};

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "the fake device and the jobs' queue are shared between the test and the closures the cache is given"
)]
mod cache_tests;
#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod layout_tests;
#[cfg(test)]
mod tree_tests;
