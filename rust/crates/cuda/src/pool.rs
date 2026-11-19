//! Released pinned buffers kept for reuse.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};

#[cfg(any(target_os = "linux", target_os = "android", windows))]
use euhedral_io::IoMemory;

use crate::error::CudaError;
use crate::library::CudaLibrary;
use crate::memory::PinnedBuffer;

/// The smallest size class: one page.
const MIN_CLASS_BYTES: usize = 4096;
/// Size classes are powers of two from [`MIN_CLASS_BYTES`]; a `usize` has at most this many.
const CLASSES: usize = usize::BITS as usize;

/// Slots in one chunk of a class's queue. The queue is unbounded and grows by chunks, so this is the size of an
/// allocation step and never a limit: a released buffer is always kept.
const CHUNK: usize = 64;

/// The released buffers of one size class, oldest first: any thread releases and any thread takes, and neither waits.
///
/// A take can come back empty while the class holds buffers: another consumer is polling at that moment, or a release
/// is part-way through publishing its buffer. The caller then allocates a new buffer; nothing is lost, because the
/// buffer the other thread is taking or publishing stays in the class or goes to that thread.
struct Class<T> {
    producer: Producer<T, Multi>,
    consumer: Consumer<T, Multi>,
}

impl<T> Class<T> {
    fn new() -> Class<T> {
        let queue = MpmcQueue::<T>::new(CHUNK);
        Class {
            producer: queue.producer(),
            consumer: queue.consumer(),
        }
    }

    /// Keeps `item` for reuse. The queue is unbounded, so the offer is always taken; if it is ever refused, the item
    /// comes back to the caller.
    fn give(&self, item: T) -> Result<(), T> {
        self.producer.offer(item)
    }

    /// The longest-held item, if any.
    fn take(&self) -> Option<T> {
        self.consumer.poll()
    }
}

/// What a pool reports about its held buffers, read at one moment.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PoolGauges {
    /// Bytes of released buffers waiting for reuse.
    pub held_bytes: u64,
    /// Buffers the pool had no room for: each buffer the pool allocated counts once, when its first release leaves
    /// the pool holding more than its limit. A buffer kept past the limit and reused is not counted again, so the
    /// count does not grow with the releases. The buffers are kept anyway until [`PinnedPool::trim`] frees them.
    pub overflows: u64,
}

/// The held-byte count, the limit and the overflow count of a pool: plain atomics, no lock.
struct Ledger {
    limit_bytes: u64,
    held_bytes: AtomicU64,
    overflows: AtomicU64,
}

impl Ledger {
    fn new(limit_bytes: u64) -> Ledger {
        Ledger {
            limit_bytes,
            held_bytes: AtomicU64::new(0),
            overflows: AtomicU64::new(0),
        }
    }

    /// A buffer of `bytes` was released into the pool. `fresh` is true for the first release of a buffer the pool
    /// allocated; only that release can count an overflow, when the pool then holds more than its limit.
    fn released(&self, bytes: u64, fresh: bool) {
        let held = self.held_bytes.fetch_add(bytes, Ordering::Relaxed) + bytes;
        if fresh && held > self.limit_bytes {
            self.overflows.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// A held buffer of `bytes` left the pool.
    fn taken(&self, bytes: u64) {
        self.held_bytes.fetch_sub(bytes, Ordering::Relaxed);
    }

    fn held_bytes(&self) -> u64 {
        self.held_bytes.load(Ordering::Relaxed)
    }

    fn gauges(&self) -> PoolGauges {
        PoolGauges {
            held_bytes: self.held_bytes(),
            overflows: self.overflows.load(Ordering::Relaxed),
        }
    }
}

/// Pinned buffers, kept after release so the next request reuses one.
///
/// Allocating pinned memory takes about a millisecond and freeing it waits for the device, so staging buffers are
/// recycled. Buffers come in power-of-two size classes. A released buffer always goes back to its class and the next
/// [`PinnedPool::acquire`] of that class reuses it, so a burst of concurrent buffers is not freed and re-allocated.
/// `limit_bytes` is the size the pool is trimmed to: [`PinnedPool::over_limit_bytes`] says how far it is over, and
/// [`PinnedPool::trim`] frees buffers (largest class first) until it is within the limit. Freeing waits for the
/// device, so trim from startup or shutdown code, never from a lattice worker; until then the pool may exceed the
/// limit by the buffers that were in use at once.
///
/// `limit_bytes` is the `memory.pinned-buffer-pool` setting; this crate does not read settings, so the caller passes
/// the resolved value.
pub struct PinnedPool {
    library: Arc<CudaLibrary>,
    ledger: Ledger,
    allocated_bytes: AtomicU64,
    classes: Box<[Class<PinnedBuffer>]>,
}

impl fmt::Debug for PinnedPool {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("PinnedPool")
            .field("limit_bytes", &self.ledger.limit_bytes)
            .field("held_bytes", &self.held_bytes())
            .finish_non_exhaustive()
    }
}

impl PinnedPool {
    pub fn new(library: &Arc<CudaLibrary>, limit_bytes: u64) -> Arc<PinnedPool> {
        Arc::new(PinnedPool {
            library: Arc::clone(library),
            ledger: Ledger::new(limit_bytes),
            allocated_bytes: AtomicU64::new(0),
            classes: (0..CLASSES).map(|_| Class::new()).collect(),
        })
    }

    /// The bytes a buffer asked for as `bytes` really holds: its size class. `None` when no class is that large.
    pub fn class_bytes(bytes: usize) -> Option<usize> {
        bytes.max(MIN_CLASS_BYTES).checked_next_power_of_two()
    }

    /// A buffer of at least `bytes`: a released one of its size class, or a new allocation.
    pub fn acquire(self: &Arc<Self>, bytes: usize) -> Result<PooledPinned, CudaError> {
        let capacity = bytes
            .max(MIN_CLASS_BYTES)
            .checked_next_power_of_two()
            .ok_or_else(|| CudaError::Argument(format!("{bytes} bytes is too large for a pinned buffer")))?;
        let class = capacity.trailing_zeros() as usize;
        let mut fresh = false;
        let buffer = match self.classes[class].take() {
            Some(buffer) => {
                self.ledger.taken(capacity as u64);
                buffer
            }
            None => {
                let buffer = PinnedBuffer::alloc(&self.library, capacity)?;
                self.allocated_bytes.fetch_add(capacity as u64, Ordering::Relaxed);
                fresh = true;
                buffer
            }
        };
        Ok(PooledPinned {
            pool: Arc::clone(self),
            buffer: Some(buffer),
            len: bytes,
            fresh,
        })
    }

    /// Bytes of pinned memory this pool has allocated and not freed: the buffers on loan and the released ones. An
    /// `acquire` that has to allocate raises it, so a stretch of work that must not allocate leaves it where it was.
    pub fn allocated_bytes(&self) -> u64 {
        self.allocated_bytes.load(Ordering::Relaxed)
    }

    /// Bytes of released buffers waiting for reuse.
    pub fn held_bytes(&self) -> u64 {
        self.ledger.held_bytes()
    }

    /// Releases that left the pool over its limit, since it was made (a counter).
    pub fn overflows(&self) -> u64 {
        self.ledger.gauges().overflows
    }

    /// The held bytes and the overflow count.
    pub fn gauges(&self) -> PoolGauges {
        self.ledger.gauges()
    }

    /// How many of the held bytes exceed the limit.
    pub fn over_limit_bytes(&self) -> u64 {
        self.held_bytes().saturating_sub(self.ledger.limit_bytes)
    }

    /// Frees released buffers, largest class first, until the pool holds no more than its limit; returns how many.
    /// Freeing pinned memory waits for the device: call this where waiting is allowed (startup, shutdown). It is for
    /// startup and shutdown: while other threads acquire and release, it may skip a class whose buffers are being
    /// taken or published at that moment, and so stop above the limit.
    pub fn trim(&self) -> usize {
        let mut freed = 0;
        for class in (0..CLASSES).rev() {
            while self.over_limit_bytes() > 0 {
                let Some(buffer) = self.classes[class].take() else {
                    break;
                };
                self.ledger.taken(buffer.len() as u64);
                self.allocated_bytes.fetch_sub(buffer.len() as u64, Ordering::Relaxed);
                drop(buffer);
                freed += 1;
            }
        }
        freed
    }

    fn release(&self, buffer: PinnedBuffer, fresh: bool) {
        self.ledger.released(buffer.len() as u64, fresh);
        let class = &self.classes[buffer.len().trailing_zeros() as usize];
        if let Err(refused) = class.give(buffer) {
            // The unbounded queue never refuses; if it does, the buffer is freed and no longer counted.
            debug_assert!(false, "an unbounded queue refused a buffer");
            self.ledger.taken(refused.len() as u64);
            self.allocated_bytes.fetch_sub(refused.len() as u64, Ordering::Relaxed);
            drop(refused);
        }
    }
}

/// A pinned buffer on loan from a [`PinnedPool`]; dropping it returns the buffer to the pool, and does not free it
/// or wait. Keep it while a queued copy still uses it: the pool may hand the buffer out again as soon as it is
/// dropped.
pub struct PooledPinned {
    pool: Arc<PinnedPool>,
    buffer: Option<PinnedBuffer>,
    len: usize,
    /// The pool allocated this buffer for this loan, so its release may count an overflow.
    fresh: bool,
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
    #[allow(
        clippy::expect_used,
        reason = "invariant: a loan holds its buffer until it is dropped"
    )]
    pub fn buffer(&self) -> &PinnedBuffer {
        self.buffer.as_ref().expect("a loan holds its buffer until dropped")
    }

    #[allow(
        clippy::expect_used,
        reason = "invariant: a loan holds its buffer until it is dropped"
    )]
    pub fn buffer_mut(&mut self) -> &mut PinnedBuffer {
        self.buffer.as_mut().expect("a loan holds its buffer until dropped")
    }

    /// The bytes requested.
    pub fn len(&self) -> usize {
        self.len
    }

    /// The bytes the loan can hold: its size class.
    pub fn capacity(&self) -> usize {
        self.buffer().len()
    }

    /// Sets the bytes the loan stands for, at most its [`capacity`](Self::capacity): a loan reused for a shorter
    /// piece, or a longer one, without a new allocation.
    pub fn set_len(&mut self, len: usize) -> Result<(), CudaError> {
        if len == 0 || len > self.capacity() {
            return Err(CudaError::Argument(format!(
                "a loan of {} bytes cannot stand for {len} bytes",
                self.capacity()
            )));
        }
        self.len = len;
        Ok(())
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// A loan is a region `AsyncIo::read_into` can fill: the first [`PooledPinned::len`] bytes of the pooled buffer. The
/// loan comes back in the continuation, so dropping it there returns the buffer to the pool.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
// SAFETY: the region is the first `len` bytes of the pooled `PinnedBuffer`, which `IoMemory for PinnedBuffer` covers:
// live, valid for reads and writes, and in place when the loan moves (the loan holds the buffer, whose pointer is
// raw). `len` is at most the buffer's size (`acquire` rounds the request up to a size class) and never changes. The
// loan is exclusive, so nothing else refers to the region while the operation owns the loan. A loan the engine
// leaks (a ring that broke) is never dropped, so its buffer never returns to the pool while the kernel may write it.
unsafe impl IoMemory for PooledPinned {
    #[allow(
        clippy::expect_used,
        reason = "invariant: a loan holds its buffer until it is dropped"
    )]
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.buffer
            .as_mut()
            .expect("a loan holds its buffer until dropped")
            .as_mut_ptr()
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl Drop for PooledPinned {
    fn drop(&mut self) {
        if let Some(buffer) = self.buffer.take() {
            self.pool.release(buffer, self.fresh);
        }
    }
}

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    reason = "the test starts threads to release and acquire at once, and joins them"
)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::thread;

    use super::*;

    /// Counts a producer as finished when it returns or unwinds, so a panic fails the test instead of hanging it.
    struct CountOnDrop<'a>(&'a AtomicUsize);

    impl Drop for CountOnDrop<'_> {
        fn drop(&mut self) {
            self.0.fetch_add(1, Ordering::SeqCst);
        }
    }

    #[test]
    fn a_first_burst_over_the_limit_counts_each_buffer_the_pool_had_no_room_for() {
        let ledger = Ledger::new(8192);
        ledger.released(4096, true);
        ledger.released(4096, true);
        assert_eq!(
            ledger.gauges(),
            PoolGauges {
                held_bytes: 8192,
                overflows: 0
            },
            "at the limit is room"
        );
        ledger.released(4096, true);
        assert_eq!(
            ledger.gauges(),
            PoolGauges {
                held_bytes: 12288,
                overflows: 1
            }
        );
    }

    #[test]
    fn cycling_a_buffer_after_the_burst_counts_nothing() {
        let ledger = Ledger::new(4096);
        ledger.released(4096, true);
        ledger.released(4096, true);
        assert_eq!(ledger.gauges().overflows, 1);
        for _ in 0..10 {
            ledger.taken(4096);
            ledger.released(4096, false);
        }
        assert_eq!(ledger.gauges().overflows, 1, "a reused buffer is not counted again");
    }

    #[test]
    fn a_buffer_reused_from_the_pool_and_released_above_the_limit_counts_nothing() {
        let ledger = Ledger::new(4096);
        ledger.released(8192, false);
        assert_eq!(
            ledger.gauges(),
            PoolGauges {
                held_bytes: 8192,
                overflows: 0
            }
        );
    }

    #[test]
    fn a_class_hands_out_in_release_order() {
        let class = Class::<u32>::new();
        assert_eq!(class.take(), None);
        for value in 0..200 {
            class.give(value).unwrap();
        }
        for value in 0..200 {
            assert_eq!(class.take(), Some(value));
        }
        assert_eq!(class.take(), None);
    }

    #[test]
    fn buffers_released_while_others_acquire_are_each_taken_once_and_none_is_lost() {
        const PRODUCERS: usize = 4;
        const CONSUMERS: usize = 4;
        const EACH: usize = 2000;
        let class = Class::<usize>::new();
        let taken_total = AtomicUsize::new(0);
        let producers_done = AtomicUsize::new(0);
        let mut seen = vec![0u8; PRODUCERS * EACH];
        thread::scope(|scope| {
            for producer in 0..PRODUCERS {
                let (class, done) = (&class, &producers_done);
                scope.spawn(move || {
                    let _counted = CountOnDrop(done);
                    for index in 0..EACH {
                        class.give(producer * EACH + index).unwrap();
                        if index % 7 == 0 {
                            thread::yield_now();
                        }
                    }
                });
            }
            let consumers: Vec<_> = (0..CONSUMERS)
                .map(|_| {
                    let (class, done, total) = (&class, &producers_done, &taken_total);
                    scope.spawn(move || {
                        let mut mine = Vec::new();
                        // Stops once every producer is done and a take finds the class empty; the second take
                        // after the last `done` sees every release, which happened before it.
                        loop {
                            let finished = done.load(Ordering::SeqCst) == PRODUCERS;
                            match class.take() {
                                Some(item) => {
                                    mine.push(item);
                                    total.fetch_add(1, Ordering::Relaxed);
                                }
                                None if finished => break,
                                None => thread::yield_now(),
                            }
                        }
                        mine
                    })
                })
                .collect();
            for consumer in consumers {
                for item in consumer.join().unwrap() {
                    seen[item] += 1;
                }
            }
        });
        assert_eq!(taken_total.load(Ordering::Relaxed), PRODUCERS * EACH);
        assert!(seen.iter().all(|&count| count == 1), "a buffer was lost or taken twice");
        assert_eq!(class.take(), None);
    }
}
