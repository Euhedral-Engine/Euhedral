//! What the cache counts, for the engine's metrics.
//!
//! The counters are monotone and independent: each is bumped by the owner's turn (`Relaxed`, nothing is published
//! with them) and read by any thread for reporting, so a snapshot may mix two moments but never loses a count.

use std::sync::atomic::{AtomicU64, Ordering};

/// Why a request that needed a load could not start it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Cause {
    /// Every device slot is loading or leased.
    Slots,
    /// The expert is loading and its copy is not submitted yet: nothing to lease.
    Copying,
    /// Every staging buffer is in use.
    Staging,
    /// Every read the cache may have in flight is in use.
    Reads,
}

/// The cache's counters.
#[derive(Debug, Default)]
pub struct CacheStats {
    hits: AtomicU64,
    misses: AtomicU64,
    coalesced: AtomicU64,
    evictions: AtomicU64,
    failed_transfers: AtomicU64,
    cancelled_misses: AtomicU64,
    transfers: AtomicU64,
    transferred_bytes: AtomicU64,
    peak_slots_in_use: AtomicU64,
    forced_leases: AtomicU64,
    waits: [AtomicU64; 4],
    asked_again: AtomicU64,
}

/// A look at the counters.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CacheSnapshot {
    /// Requests answered by a resident expert.
    pub hits: u64,
    /// Requests that started a load.
    pub misses: u64,
    /// Requests that joined a load another request started.
    pub coalesced: u64,
    /// Residents evicted for a load.
    pub evictions: u64,
    /// Loads that failed (the read, the submission or the copy).
    pub failed_transfers: u64,
    /// Loads that ended without a lease because the request was cancelled or the cache closed.
    pub cancelled_misses: u64,
    /// Loads whose copy retired.
    pub transfers: u64,
    pub transferred_bytes: u64,
    /// The most device slots loading or leased at once.
    pub peak_slots_in_use: u64,
    /// Leases still open when the cache closed.
    pub forced_leases: u64,
    /// Times a request had to wait, by [`Cause`] in declaration order.
    pub waits: [u64; 4],
    /// Times a waiting request was asked again.
    pub asked_again: u64,
}

impl CacheStats {
    pub fn snapshot(&self) -> CacheSnapshot {
        let load = |counter: &AtomicU64| counter.load(Ordering::Relaxed);
        CacheSnapshot {
            hits: load(&self.hits),
            misses: load(&self.misses),
            coalesced: load(&self.coalesced),
            evictions: load(&self.evictions),
            failed_transfers: load(&self.failed_transfers),
            cancelled_misses: load(&self.cancelled_misses),
            transfers: load(&self.transfers),
            transferred_bytes: load(&self.transferred_bytes),
            peak_slots_in_use: load(&self.peak_slots_in_use),
            forced_leases: load(&self.forced_leases),
            waits: [
                load(&self.waits[0]),
                load(&self.waits[1]),
                load(&self.waits[2]),
                load(&self.waits[3]),
            ],
            asked_again: load(&self.asked_again),
        }
    }

    pub(super) fn hit(&self) {
        self.hits.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn miss(&self) {
        self.misses.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn coalesce(&self) {
        self.coalesced.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn eviction(&self) {
        self.evictions.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn failed_transfer(&self) {
        self.failed_transfers.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn cancelled_miss(&self) {
        self.cancelled_misses.fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn transferred(&self, bytes: u64) {
        self.transfers.fetch_add(1, Ordering::Relaxed);
        self.transferred_bytes.fetch_add(bytes, Ordering::Relaxed);
    }

    pub(super) fn slots_in_use(&self, now: u64) {
        self.peak_slots_in_use.fetch_max(now, Ordering::Relaxed);
    }

    pub(super) fn forced(&self, leases: u64) {
        self.forced_leases.fetch_add(leases, Ordering::Relaxed);
    }

    pub(super) fn waited(&self, cause: Cause) {
        self.waits[cause as usize].fetch_add(1, Ordering::Relaxed);
    }

    pub(super) fn asked_again(&self) {
        self.asked_again.fetch_add(1, Ordering::Relaxed);
    }
}
