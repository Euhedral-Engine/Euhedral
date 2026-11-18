//! The prefix cache's host-side parts: a radix tree of checkpoints over token ids, and the layout of one checkpoint
//! in the slots of a host arena.
//!
//! A checkpoint is a snapshot of a live sequence at a token position: the GDN state at that position, the KV pages of
//! the span since its parent's position, and optionally the speculative state. It lives in slots leased from one
//! [`euhedral_memory::SlotArena`]; the tree owns the leases and the layout says what each slot holds and which device
//! buffer each piece of it is copied from or to.

mod layout;
mod tree;

pub use layout::{Buffer, CheckpointLayout, Copy};
pub use tree::{Match, NodeId, POSITION_GRANULE, PrefixNode, PrefixTree, ReserveError, Speculation};

#[cfg(test)]
mod fixtures;
#[cfg(test)]
mod layout_tests;
#[cfg(test)]
mod tree_tests;
