//! Released pinned buffers kept for reuse.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::error::CudaError;
use crate::library::CudaLibrary;
use crate::memory::PinnedBuffer;
use crate::stack::Stack;

/// The smallest size class: one page.
const MIN_CLASS_BYTES: usize = 4096;
/// Size classes are powers of two from [`MIN_CLASS_BYTES`]; a `usize` has at most this many.
const CLASSES: usize = usize::BITS as usize;

/// Pinned buffers, kept after release up to a byte limit so the next request reuses one.
///
/// Allocating pinned memory takes about a millisecond and freeing it waits for the device, so staging buffers are
/// recycled. Buffers come in power-of-two size classes. A released buffer is kept while the pool holds at most
/// `limit_bytes`; beyond that it goes to an excess list that [`PinnedPool::trim`] frees, from a thread that may wait.
///
/// `limit_bytes` is the `memory.pinned-buffer-pool` setting; this crate does not read settings, so the caller passes
/// the resolved value.
pub struct PinnedPool {
    library: Arc<CudaLibrary>,
    limit_bytes: u64,
    held_bytes: AtomicU64,
    overflows: AtomicU64,
    classes: Box<[Stack<PinnedBuffer>]>,
    excess: Stack<PinnedBuffer>,
}

impl fmt::Debug for PinnedPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PinnedPool")
            .field("limit_bytes", &self.limit_bytes)
            .field("held_bytes", &self.held_bytes())
            .finish_non_exhaustive()
    }
}

impl PinnedPool {
    pub fn new(library: &Arc<CudaLibrary>, limit_bytes: u64) -> Arc<PinnedPool> {
        Arc::new(PinnedPool {
            library: Arc::clone(library),
            limit_bytes,
            held_bytes: AtomicU64::new(0),
            overflows: AtomicU64::new(0),
            classes: (0..CLASSES).map(|_| Stack::new()).collect(),
            excess: Stack::new(),
        })
    }

    /// A buffer of at least `bytes`: a released one of its size class, or a new allocation.
    pub fn acquire(self: &Arc<Self>, bytes: usize) -> Result<PooledPinned, CudaError> {
        let capacity = bytes
            .max(MIN_CLASS_BYTES)
            .checked_next_power_of_two()
            .ok_or_else(|| CudaError::Argument(format!("{bytes} bytes is too large for a pinned buffer")))?;
        let class = capacity.trailing_zeros() as usize;
        let buffer = match self.classes[class].pop() {
            Some(buffer) => {
                self.held_bytes.fetch_sub(capacity as u64, Ordering::Relaxed);
                buffer
            }
            None => PinnedBuffer::alloc(&self.library, capacity)?,
        };
        Ok(PooledPinned {
            pool: Arc::clone(self),
            buffer: Some(buffer),
            len: bytes,
        })
    }

    /// Bytes of released buffers kept for reuse.
    pub fn held_bytes(&self) -> u64 {
        self.held_bytes.load(Ordering::Relaxed)
    }

    /// Released buffers that found no room, since the pool was made.
    pub fn overflows(&self) -> u64 {
        self.overflows.load(Ordering::Relaxed)
    }

    /// Frees the released buffers that found no room. Freeing pinned memory waits for the device: call this from a
    /// thread that may wait (startup or shutdown), never from a lattice worker.
    pub fn trim(&self) -> usize {
        self.excess.take_all().len()
    }

    fn release(&self, buffer: PinnedBuffer) {
        let capacity = buffer.len() as u64;
        let held = self.held_bytes.fetch_add(capacity, Ordering::Relaxed) + capacity;
        if held > self.limit_bytes {
            self.held_bytes.fetch_sub(capacity, Ordering::Relaxed);
            self.overflows.fetch_add(1, Ordering::Relaxed);
            self.excess.push(buffer);
        } else {
            self.classes[buffer.len().trailing_zeros() as usize].push(buffer);
        }
    }
}

/// A pinned buffer on loan from a [`PinnedPool`]; dropping it returns the buffer to the pool, and does not free it
/// or wait. Retire it with a [`RetireList`](crate::RetireList) while a queued copy still uses it.
pub struct PooledPinned {
    pool: Arc<PinnedPool>,
    buffer: Option<PinnedBuffer>,
    len: usize,
}

impl fmt::Debug for PooledPinned {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PooledPinned")
            .field("len", &self.len)
            .finish_non_exhaustive()
    }
}

impl PooledPinned {
    /// The buffer. Its length is the size class; use [`PooledPinned::len`] for the requested size.
    pub fn buffer(&self) -> &PinnedBuffer {
        self.buffer.as_ref().expect("a loan holds its buffer until dropped")
    }

    pub fn buffer_mut(&mut self) -> &mut PinnedBuffer {
        self.buffer.as_mut().expect("a loan holds its buffer until dropped")
    }

    /// The bytes requested.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

impl Drop for PooledPinned {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer);
        }
    }
}
