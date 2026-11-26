//! The weights several scoring engines share on one machine: each distinct object is loaded once and held for as long
//! as any engine using it runs, and an engine starts only while what it adds fits.
//!
//! - [`WeightCache`]: the shared objects. An object is identified by [`Key`] (the file it is read from, its name and
//!   its offset there); [`WeightCache::acquire`] gives every engine that asks for it the same [`Lease`], loading it once
//!   in a frame of its own. A lease is a reference: the engine that drops the last one frees the object, so the
//!   cache never decides when an object ends. What an object is (host bytes, a pinned arena) is the [`Backend`]'s.
//! - [`Frames`] and [`HostFrames`]: where loads run: the `host` lake in production, whatever the test supplies.
//! - [`Admission`] and [`Need`]: whether the next engine starts beside those running: the bytes it would add, with
//!   the objects already in the cache not counted, against the memory that is free; a refusal says by how much.
//! - [`Residency`]: what an engine needs beyond its objects, from the engine's residency planner.
//!
//! Nothing waits: an acquisition ends in a callback, on the thread that completed the load. The table of objects is
//! lifecycle state behind a [`LifecycleMutex`](euhedral_data_structures::sync::LifecycleMutex) that is held only for
//! the table's own reads and writes, never while an object is read from disk, freed, or handed to a callback.

mod admission;
mod cache;
mod frames;
mod residency;

pub use admission::{Admission, Need, Refusal};
pub use cache::{Backend, Key, Lease, LoadError, Outcome, WeightCache};
pub use frames::{Frames, HostFrames};
pub use residency::Residency;

#[cfg(test)]
mod tests;
