//! What the stages of one layer's MoE block hand to one another within a quantum.
//!
//! [`BlockPlan`] is the part the host stages share: the plan stage publishes which experts the block names, the fetch
//! stages put the leases they were given into its slots, and the expert stages take them. Every field has one writer
//! at a time, ordered by the graph's edges (plan before fetch before expert), so they are atomics and hand-off cells
//! and nothing locks. [`MoeStaging`] is the pinned host memory the router's choice arrives in and the description
//! leaves from.
//!
//! The quantum carries one [`MoeBlock`] for all its layers: a layer's plan stage starts only after the previous layer's
//! finish stage has run, and the router that follows it waits for the device (a completion edge), so the previous block's
//! copies from and to the staging memory have retired.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, Ordering};

use euhedral_graph::StageError;
use euhedral_inference_cuda::{CudaError, PinnedBuffer, PinnedPool, PooledPinned};

use super::routing::{DescriptorLayout, Routing};
use super::source::HeldExpert;
use crate::flash::FlashError;
use crate::handoff::Handoff;
use crate::stages::fatal;

/// The experts of one planned block and the leases of those that arrived.
#[derive(Debug)]
pub struct BlockPlan {
    cancel: Arc<AtomicBool>,
    rows: AtomicU32,
    pairs: AtomicU32,
    active: AtomicU32,
    experts: Box<[AtomicU32]>,
    item_starts: Box<[AtomicU32]>,
    held: Box<[Handoff<Box<dyn HeldExpert>>]>,
}

impl BlockPlan {
    /// A plan for blocks of at most `max_active` experts, with no expert named.
    pub fn new(max_active: u32) -> Self {
        let width = max_active as usize;
        Self {
            cancel: Arc::new(AtomicBool::new(false)),
            rows: AtomicU32::new(0),
            pairs: AtomicU32::new(0),
            active: AtomicU32::new(0),
            experts: (0..width).map(|_| AtomicU32::new(0)).collect(),
            item_starts: (0..=width).map(|_| AtomicU32::new(0)).collect(),
            held: (0..width).map(|_| Handoff::empty()).collect(),
        }
    }

    /// The most experts the plan holds.
    pub fn width(&self) -> u32 {
        self.experts.len() as u32
    }

    /// Publishes the grouping of a block, and gives back the leases a block before it left behind. A block of no rows
    /// is published by [`BlockPlan::publish_none`].
    pub fn publish(&self, routing: &Routing) {
        self.release_all();
        let active = routing.active_experts();
        for index in 0..active {
            let at = index as usize;
            self.experts[at].store(routing.active_expert(index).unwrap_or(0), Ordering::Relaxed);
            self.item_starts[at].store(routing.item_start(index).unwrap_or(0), Ordering::Relaxed);
        }
        let last = active.saturating_sub(1);
        let items = routing.item_start(last).unwrap_or(0) + routing.item_count(last).unwrap_or(0);
        self.item_starts[active as usize].store(items, Ordering::Relaxed);
        self.rows.store(routing.rows(), Ordering::Relaxed);
        self.pairs.store(routing.pair_count(), Ordering::Relaxed);
        self.active.store(active, Ordering::Release);
    }

    /// Publishes a block that names no expert (a quantum of no rows).
    pub fn publish_none(&self) {
        self.release_all();
        self.rows.store(0, Ordering::Relaxed);
        self.pairs.store(0, Ordering::Relaxed);
        self.active.store(0, Ordering::Release);
    }

    /// The experts the block names.
    pub fn active(&self) -> u32 {
        self.active.load(Ordering::Acquire)
    }

    pub fn rows(&self) -> u32 {
        self.rows.load(Ordering::Relaxed)
    }

    /// The pairs of the block that were not padding.
    pub fn pairs(&self) -> u32 {
        self.pairs.load(Ordering::Relaxed)
    }

    /// The id of the `index`-th active expert, in ascending order.
    pub fn expert(&self, index: u32) -> Option<u32> {
        (index < self.active()).then(|| self.experts[index as usize].load(Ordering::Relaxed))
    }

    /// The first work item of the `index`-th active expert and how many it has.
    pub fn items(&self, index: u32) -> Option<(u32, u32)> {
        (index < self.active()).then(|| {
            let first = self.item_starts[index as usize].load(Ordering::Relaxed);
            (
                first,
                self.item_starts[index as usize + 1].load(Ordering::Relaxed) - first,
            )
        })
    }

    /// The flag a request is cancelled by when the quantum has no flag of its own for it; it is never set.
    pub fn spare_cancel(&self) -> &Arc<AtomicBool> {
        &self.cancel
    }

    /// Stores the lease of the `index`-th active expert for its expert stage. A lease that cannot be stored (no such
    /// slot) is dropped.
    pub fn hold(&self, index: u32, lease: Box<dyn HeldExpert>) {
        if let Some(slot) = self.held.get(index as usize) {
            slot.send(lease);
        }
    }

    /// Takes the lease of the `index`-th active expert, if it arrived and was not taken.
    pub fn take(&self, index: u32) -> Option<Box<dyn HeldExpert>> {
        self.held.get(index as usize)?.take()
    }

    /// Whether the lease of the `index`-th active expert is held.
    pub fn is_held(&self, index: u32) -> bool {
        self.held.get(index as usize).is_some_and(Handoff::is_set)
    }

    /// Drops every lease still held.
    pub fn release_all(&self) {
        self.held.iter().for_each(|slot| drop(slot.take()));
    }
}

impl Drop for BlockPlan {
    fn drop(&mut self) {
        self.release_all();
    }
}

/// The pinned host memory of a block: the router's ids and weights as the device copies them, and the description the
/// plan stage writes and copies to the device. Words the host reads or writes while a copy may be in flight go through
/// atomics.
#[derive(Debug)]
pub struct MoeStaging {
    ids: PooledPinned,
    weights: PooledPinned,
    descriptor: PooledPinned,
    routing: Handoff<Box<Routing>>,
}

impl MoeStaging {
    /// The staging of blocks of `layout`'s extents, in loans of `pool`.
    pub fn new(pool: &Arc<PinnedPool>, layout: DescriptorLayout) -> Result<Self, CudaError> {
        let pairs = layout.max_pairs() as usize;
        let routing = Handoff::empty();
        routing.send(Box::new(Routing::new(layout)));
        Ok(Self {
            ids: pool.acquire(4 * pairs)?,
            weights: pool.acquire((2 * pairs).next_multiple_of(4))?,
            descriptor: pool.acquire(layout.bytes())?,
            routing,
        })
    }

    /// The pinned memory the router's ids are copied into.
    pub(crate) fn ids(&self) -> &PinnedBuffer {
        self.ids.buffer()
    }

    /// The pinned memory the router's weights are copied into.
    pub(crate) fn weights(&self) -> &PinnedBuffer {
        self.weights.buffer()
    }

    /// The pinned memory of the description the device copies in.
    pub(crate) fn descriptor(&self) -> &PinnedBuffer {
        self.descriptor.buffer()
    }

    /// The router's `i`-th expert id, as the device copied it.
    pub(crate) fn id(&self, i: usize) -> i32 {
        self.ids.buffer().word(i).load(Ordering::Relaxed) as i32
    }

    /// The router's `i`-th routing weight (BF16 bits), as the device copied it.
    pub(crate) fn weight(&self, i: usize) -> u16 {
        let word = self.weights.buffer().word(i / 2).load(Ordering::Relaxed);
        (word >> (16 * (i % 2))) as u16
    }

    /// Takes the grouping scratch for one block's plan; the plan stage gives it back with [`MoeStaging::give_back`].
    pub(crate) fn take_routing(&self) -> Result<Box<Routing>, StageError> {
        self.routing
            .take()
            .ok_or_else(|| fatal("two plan stages of one quantum ran together: the routing scratch is in use"))
    }

    pub(crate) fn give_back(&self, routing: Box<Routing>) {
        self.routing.send(routing);
    }
}

/// What a quantum carries for its layers' MoE blocks.
#[derive(Debug)]
pub struct MoeBlock {
    plan: Arc<BlockPlan>,
    staging: MoeStaging,
}

impl MoeBlock {
    /// The block state of `staging`, whose plan holds up to `layout.max_active()` experts.
    pub fn new(staging: MoeStaging, layout: &DescriptorLayout) -> Result<Self, FlashError> {
        if layout.max_active() == 0 {
            return Err(FlashError::refused("moe block", "a layout of no experts".to_owned()));
        }
        Ok(Self {
            plan: Arc::new(BlockPlan::new(layout.max_active())),
            staging,
        })
    }

    pub fn plan(&self) -> &Arc<BlockPlan> {
        &self.plan
    }

    pub fn staging(&self) -> &MoeStaging {
        &self.staging
    }
}

#[cfg(test)]
mod tests;
