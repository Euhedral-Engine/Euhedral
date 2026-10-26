//! The attention cache: pages of key and value rows per full-attention layer, in the sequence's [`KvFormat`]
//! (`AttentionKvState`).
//!
//! A page holds 256 token rows of every KV head, a K plane then a V plane, and never moves. A layer's page table is a
//! device array of raw page addresses (the K planes, then the V planes), grown by doubling; the attention kernels
//! read the pages through it. Reservation runs inside the owning quantum on its lane: it allocates pages in stream
//! order and uploads a rewritten table from pinned staging by a queued copy, so it never waits for the device.
//! The staging and an outgrown table are handed back in a [`KvRetained`] for the caller to retire once the lane
//! has reached the quantum's marker.
//!
//! Rows move through three frontiers. `reserved` rows are backed by pages. `submitted` rows are written by work
//! queued on the owning quantum's lane; later stages of that quantum may read them, because the lane's order puts
//! those reads after the writes. `committed` rows are published once the quantum's device work retired; nothing
//! beyond them is visible to other quanta. One cache has one set of frontiers: its layers append in lockstep.
//!
//! With host pages, only the first `device_pages` pages of each layer are device memory. The pages past them live in
//! one pinned host arena, and the page tables address them in one device region shared by the layers: before a layer
//! reads, [`KvCache::stage_in`] copies its host pages into the region, and afterwards [`KvCache::stage_out`] copies
//! the pages it wrote back, both on the quantum's lane.

use std::sync::Arc;

use euhedral_inference_cuda::{
    AllocationId, CudaError, CudaLibrary, DeviceBuffer, DevicePtr, HostWeightsArena, PinnedPool, PooledPinned,
    Submission, ops,
};
use euhedral_inference_settings::KvFormat;
use thiserror::Error;

use crate::residency::footprint::{PAGE_TOKENS, plane_page_bytes};

/// Rows of the quanta whose host page copies depend only on the reservation: decode, verification and draft quanta,
/// which append at the end of the reserved rows and touch at most its last two pages
/// (`AttentionKvState.STAGED_TAIL_ROWS`).
pub const STAGED_TAIL_ROWS: u64 = 8;

/// Why a cache refused a request.
#[derive(Debug, Error)]
pub enum KvError {
    #[error("a KV append must continue the submitted frontier at {submitted}, not {start}, with at least one row")]
    NotContiguous { start: u64, submitted: u64 },
    #[error("an attention cache of {required} rows is longer than the {planned} rows planned for it")]
    TooLong { required: u64, planned: u64 },
    #[error("{what}: {rows} rows exceed the {limit} rows available")]
    Frontier { what: &'static str, rows: u64, limit: u64 },
    #[error("a KV cache is truncated between quanta, but an append is in flight")]
    AppendInFlight,
    #[error("the KV cache holds {heads} wide rows; it needs complete 256-value heads")]
    Width { heads: u64 },
    #[error(transparent)]
    Cuda(#[from] CudaError),
}

/// What a reservation leaves for the caller: pinned staging that a queued table upload reads, and tables the
/// reservation outgrew, which earlier quanta may still read. Retire both with the marker of the lane work that
/// queued the reservation (`RetireList::retire`), whether the reservation succeeded or failed.
#[derive(Debug, Default)]
pub struct KvRetained {
    pub staging: Vec<PooledPinned>,
    pub outgrown: Vec<DeviceBuffer>,
}

impl KvRetained {
    /// Whether nothing is held.
    pub fn is_empty(&self) -> bool {
        self.staging.is_empty() && self.outgrown.is_empty()
    }
}

/// The pages past the device share: one pinned arena for every layer and the device region they share.
#[derive(Debug)]
struct HostPages {
    /// Declared first: it is freed before the region the tables address.
    arena: HostWeightsArena,
    region: DeviceBuffer,
    device_pages: u64,
    /// Bytes of one layer's host pages.
    layer_bytes: u64,
}

#[derive(Debug, Default)]
struct LayerPages {
    pages: Vec<DeviceBuffer>,
    table: Option<DeviceBuffer>,
}

/// The pages of the attention layers of one sequence.
#[derive(Debug)]
pub struct KvCache {
    format: KvFormat,
    plane_page_bytes: u64,
    /// Pages a layer may hold, device and host.
    max_pages: u64,
    layers: Vec<LayerPages>,
    host: Option<HostPages>,
    table_slots: u64,
    reserved: u64,
    submitted: u64,
    committed: u64,
}

impl KvCache {
    /// A cache of `layers` layers of rows `kv_width` wide, for up to `device_tokens + host_tokens` positions. The
    /// tokens past `device_tokens` live in pinned host memory (`host_tokens` of 0: every page on the device). This
    /// allocates the host arena and the device region with the waiting allocators: startup only. Pages and tables
    /// are allocated on demand, by [`KvCache::reserve`].
    pub fn new(
        library: &Arc<CudaLibrary>,
        layers: usize,
        kv_width: u64,
        format: KvFormat,
        device_tokens: u64,
        host_tokens: u64,
    ) -> Result<KvCache, KvError> {
        if kv_width == 0 || !kv_width.is_multiple_of(256) {
            return Err(KvError::Width { heads: kv_width });
        }
        let plane = plane_page_bytes(kv_width, format);
        let device_pages = device_tokens.div_ceil(PAGE_TOKENS);
        let host_pages = host_tokens.div_ceil(PAGE_TOKENS);
        let host = if host_pages > 0 && layers > 0 {
            let region_bytes = host_pages * 2 * plane;
            let layer_bytes = region_bytes;
            Some(HostPages {
                arena: HostWeightsArena::alloc(library, (layer_bytes * layers as u64) as usize)?,
                region: DeviceBuffer::alloc(library, region_bytes as usize)?,
                device_pages,
                layer_bytes,
            })
        } else {
            None
        };
        Ok(KvCache {
            format,
            plane_page_bytes: plane,
            max_pages: device_pages + if host.is_some() { host_pages } else { 0 },
            layers: (0..layers).map(|_| LayerPages::default()).collect(),
            host,
            table_slots: 0,
            reserved: 0,
            submitted: 0,
            committed: 0,
        })
    }

    /// The format of the rows.
    pub fn format(&self) -> KvFormat {
        self.format
    }

    /// Layers in the cache.
    pub fn layers(&self) -> usize {
        self.layers.len()
    }

    /// Bytes of one plane (K or V) of one page.
    pub fn plane_page_bytes(&self) -> u64 {
        self.plane_page_bytes
    }

    /// The rows the cache can hold at most.
    pub fn planned_rows(&self) -> u64 {
        self.max_pages * PAGE_TOKENS
    }

    /// Rows backed by pages.
    pub fn reserved(&self) -> u64 {
        self.reserved
    }

    /// Rows whose writes are queued on the owning quantum's lane.
    pub fn submitted(&self) -> u64 {
        self.submitted
    }

    /// Rows published to every reader.
    pub fn committed(&self) -> u64 {
        self.committed
    }

    /// Pages each layer holds on the device now.
    pub fn device_pages(&self) -> usize {
        self.layers.first().map_or(0, |layer| layer.pages.len())
    }

    /// The device bytes the cache holds: every page, every table, and the host pages' region.
    pub fn device_bytes(&self) -> u64 {
        let pages: u64 = self.layers.iter().map(|layer| layer.pages.len() as u64).sum();
        let tables: u64 = self
            .layers
            .iter()
            .filter_map(|layer| layer.table.as_ref())
            .map(|table| table.bytes() as u64)
            .sum();
        pages * 2 * self.plane_page_bytes + tables + self.host.as_ref().map_or(0, |host| host.region.bytes() as u64)
    }

    /// Allocates, in `pool`, the pinned staging every reservation of this cache can take, so that
    /// [`KvCache::reserve`] never allocates: for each size of page table the cache can grow to, `copies` sets of one
    /// buffer per layer (a reservation holds one set until the caller retires it behind its marker). The allocation
    /// waits: startup only.
    pub fn prime_staging(&self, pool: &Arc<PinnedPool>, copies: usize) -> Result<(), CudaError> {
        let mut slots = 1u64;
        loop {
            let loans = (0..copies * self.layers.len())
                .map(|_| pool.acquire((slots * 16) as usize))
                .collect::<Result<Vec<_>, _>>()?;
            drop(loans);
            if slots >= self.max_pages {
                return Ok(());
            }
            slots *= 2;
        }
    }

    /// Whether appending `rows` rows at `start` fits the reserved pages, so that reserving neither allocates nor
    /// uploads a page table.
    pub fn reserves(&self, start: u64, rows: u64) -> bool {
        start + rows <= self.reserved
    }

    /// Reserves `rows` contiguous append rows at `start`, without publishing them. The rows continue the submitted
    /// frontier. This queues stream-ordered allocations and a table upload on `submission`'s lane and waits for
    /// nothing; what the upload reads, and the tables it replaced, are added to `retained`.
    pub fn reserve(
        &mut self,
        submission: &Submission<'_>,
        pool: &Arc<PinnedPool>,
        start: u64,
        rows: u64,
        retained: &mut KvRetained,
    ) -> Result<(), KvError> {
        if start != self.submitted || rows == 0 {
            return Err(KvError::NotContiguous {
                start,
                submitted: self.submitted,
            });
        }
        let required = start + rows;
        if required <= self.reserved {
            return Ok(());
        }
        let count = required.div_ceil(PAGE_TOKENS);
        if count > self.max_pages {
            return Err(KvError::TooLong {
                required,
                planned: self.planned_rows(),
            });
        }
        let device = self.host.as_ref().map_or(count, |host| count.min(host.device_pages));
        let mut slots = self.table_slots.max(1);
        while slots < count {
            slots *= 2;
        }
        let table_bytes = slots * 2 * 8;
        let plane = self.plane_page_bytes;
        let region = self.host.as_ref().map(|host| host.region.ptr());
        for layer in &mut self.layers {
            while (layer.pages.len() as u64) < device {
                layer
                    .pages
                    .push(DeviceBuffer::alloc_async(submission, (2 * plane) as usize)?);
            }
            let next = if slots == self.table_slots {
                None
            } else {
                Some(DeviceBuffer::alloc_async(submission, table_bytes as usize)?)
            };
            let mut staging = pool.acquire(table_bytes as usize)?;
            let entries = staging.buffer_mut().as_mut_slice();
            entries[..table_bytes as usize].fill(0);
            for page in 0..count {
                // `count` was checked against the device pages when there is no host region.
                let address = if page < device {
                    layer.pages[page as usize].ptr()
                } else {
                    region.unwrap_or(DevicePtr::NULL).offset((page - device) * 2 * plane)
                };
                write_entry(entries, page, address.get());
                write_entry(entries, slots + page, address.offset(plane).get());
            }
            let target = next
                .as_ref()
                .or(layer.table.as_ref())
                .map(DeviceBuffer::ptr)
                .unwrap_or(DevicePtr::NULL);
            // The staging is retained from here on: the queued copy reads it even if a later call fails.
            retained.staging.push(staging);
            let staged = retained.staging[retained.staging.len() - 1].buffer();
            // SAFETY: the staging stays alive and unmodified in `retained` until the caller retires it behind the
            // lane's marker, which is after the copy has run.
            unsafe { ops::copy_upload_to_device(submission, target, staged, 0, table_bytes as usize) }?;
            if let Some(next) = next
                && let Some(old) = layer.table.replace(next)
            {
                retained.outgrown.push(old);
            }
        }
        self.table_slots = slots;
        self.reserved = count * PAGE_TOKENS;
        Ok(())
    }

    /// Records that the writes for `rows` reserved rows after the submitted frontier were queued on the owning
    /// quantum's lane (a prompt's chunks, one after another).
    pub fn appended(&mut self, rows: u64) -> Result<(), KvError> {
        if rows == 0 || self.submitted + rows > self.reserved {
            return Err(KvError::Frontier {
                what: "a KV append exceeds the reserved rows",
                rows: self.submitted + rows,
                limit: self.reserved,
            });
        }
        self.submitted += rows;
        Ok(())
    }

    /// Publishes the whole submitted frontier. Called only after the quantum's device work retired.
    pub fn commit_submitted(&mut self) {
        self.committed = self.submitted;
    }

    /// Publishes the `rows` rows a quantum appended, every one of them, once its device work retired. Rows a later
    /// quantum appended meanwhile stay submitted.
    pub fn commit(&mut self, rows: u64) -> Result<(), KvError> {
        if self.committed + rows > self.submitted {
            return Err(KvError::Frontier {
                what: "committed rows exceed the submitted frontier",
                rows: self.committed + rows,
                limit: self.submitted,
            });
        }
        self.committed += rows;
        Ok(())
    }

    /// Publishes only the first `rows` submitted rows (a speculative verification's accepted prefix); the rest are
    /// dropped, and the next append overwrites them.
    pub fn commit_prefix(&mut self, rows: u64) -> Result<(), KvError> {
        self.commit(rows)?;
        self.submitted = self.committed;
        Ok(())
    }

    /// Drops a submitted frontier that must not become visible, such as a failed quantum's.
    pub fn discard(&mut self) {
        self.submitted = self.committed;
    }

    /// Drops committed rows from `rows` on (speculative draft rows), between quanta. The rows stay in their pages
    /// until the next append overwrites them; no reader sees past the frontier.
    pub fn truncate(&mut self, rows: u64) -> Result<(), KvError> {
        if self.submitted != self.committed {
            return Err(KvError::AppendInFlight);
        }
        if rows > self.committed {
            return Err(KvError::Frontier {
                what: "a cache can only truncate committed rows",
                rows,
                limit: self.committed,
            });
        }
        self.committed = rows;
        self.submitted = rows;
        Ok(())
    }

    /// The device address of layer `layer`'s K page-address table; the V table follows at
    /// [`KvCache::value_table`].
    pub fn key_table(&self, layer: usize) -> Option<DevicePtr> {
        self.layers.get(layer)?.table.as_ref().map(DeviceBuffer::ptr)
    }

    /// The device address of layer `layer`'s V page-address table.
    pub fn value_table(&self, layer: usize) -> Option<DevicePtr> {
        self.key_table(layer).map(|key| key.offset(self.table_slots * 8))
    }

    /// The allocation behind layer `layer`'s page table, for the capture key of a graph that reads it.
    pub fn table_generation(&self, layer: usize) -> Option<AllocationId> {
        self.layers.get(layer)?.table.as_ref().map(DeviceBuffer::generation)
    }

    /// The device region the host pages are read and written through, when the cache has host pages.
    pub fn host_region(&self) -> Option<&DeviceBuffer> {
        self.host.as_ref().map(|host| &host.region)
    }

    /// The pinned arena that holds the host pages, one run of pages per layer, when the cache has host pages.
    pub fn host_arena_mut(&mut self) -> Option<&mut HostWeightsArena> {
        self.host.as_mut().map(|host| &mut host.arena)
    }

    /// The device pages of layer `layer`, each its K plane then its V plane.
    pub fn pages(&self, layer: usize) -> impl Iterator<Item = DevicePtr> + '_ {
        self.layers
            .get(layer)
            .into_iter()
            .flat_map(|layer| layer.pages.iter().map(DeviceBuffer::ptr))
    }

    /// Host pages the reserved rows reach (0 without host pages).
    pub fn reserved_host_pages(&self) -> u64 {
        self.host.as_ref().map_or(0, |host| {
            (self.reserved / PAGE_TOKENS).saturating_sub(host.device_pages)
        })
    }

    /// Queues on `submission`'s lane the copy of layer `layer`'s host pages that an append of `rows` rows at `start`
    /// and the attention after it read into the device region: every reserved one for a quantum of at most
    /// [`STAGED_TAIL_ROWS`] rows, otherwise those holding the rows before `start + rows`. Nothing without host
    /// pages.
    pub fn stage_in(&self, submission: &Submission<'_>, layer: usize, start: u64, rows: u64) -> Result<(), KvError> {
        let Some(host) = &self.host else { return Ok(()) };
        let reserved = self.reserved_host_pages();
        if reserved == 0 {
            return Ok(());
        }
        let pages = if rows <= STAGED_TAIL_ROWS {
            reserved
        } else {
            reserved.min((start + rows).div_ceil(PAGE_TOKENS).saturating_sub(host.device_pages))
        };
        if pages == 0 {
            return Ok(());
        }
        let offset = layer as u64 * host.layer_bytes;
        // SAFETY: the arena lives as long as the cache; the host pages of a layer are written only by the copy
        // `stage_out` queues on the same lane, which the lane orders before this one.
        unsafe {
            ops::copy_host_weights_to_device(
                submission,
                host.region.ptr(),
                &host.arena,
                offset as usize,
                (pages * 2 * self.plane_page_bytes) as usize,
            )
        }?;
        Ok(())
    }

    /// Queues on `submission`'s lane the copy back to host memory of the host pages that an append of `rows` rows at
    /// `start` wrote: the last two reserved ones for a quantum of at most [`STAGED_TAIL_ROWS`] rows (an append at the
    /// end of the reservation touches no other), otherwise exactly those holding its rows.
    pub fn stage_out(&self, submission: &Submission<'_>, layer: usize, start: u64, rows: u64) -> Result<(), KvError> {
        let Some(host) = &self.host else { return Ok(()) };
        let reserved = self.reserved_host_pages();
        if reserved == 0 {
            return Ok(());
        }
        let (first, end) = if rows <= STAGED_TAIL_ROWS {
            (reserved.saturating_sub(2), reserved)
        } else {
            (
                (start / PAGE_TOKENS).saturating_sub(host.device_pages),
                reserved.min((start + rows).div_ceil(PAGE_TOKENS).saturating_sub(host.device_pages)),
            )
        };
        if end <= first {
            return Ok(());
        }
        let page = 2 * self.plane_page_bytes;
        let offset = layer as u64 * host.layer_bytes + first * page;
        // SAFETY: as `stage_in`: the arena outlives the copy, and no other copy writes these pages meanwhile.
        unsafe {
            ops::copy_device_to_host_weights(
                submission,
                &host.arena,
                offset as usize,
                host.region.ptr().offset(first * page),
                ((end - first) * page) as usize,
            )
        }?;
        Ok(())
    }
}

fn write_entry(entries: &mut [u8], index: u64, address: u64) {
    let at = (index * 8) as usize;
    entries[at..at + 8].copy_from_slice(&address.to_ne_bytes());
}
