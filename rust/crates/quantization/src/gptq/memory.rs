//! The memory of a GPTQ chain: stream-ordered device buffers that are never freed inside a frame, and pooled pinned
//! staging.
//!
//! A device buffer is allocated in stream order on the lane's stream ([`DeviceBuffer::alloc_on_lane`]). Dropping a
//! [`Device`] on any thread, a failed step's partial state or a [`Factor`](crate::gptq::Factor) clone included, gives the
//! buffer to the [`Released`] queue, and the owner of the lane frees what waits there in stream order
//! ([`reclaim`]): a plain `cudaFree` waits for every lane's running kernels. Pinned staging comes from the
//! [`PinnedPool`], which recycles buffers and never frees on a running engine.
//!
//! Both kinds are charged to the memory planner: a step takes [`Claims`] sized by its `*_bytes` function, and each
//! allocation takes its share of them. A device buffer's share is given back when the buffer is freed.

use std::ffi::c_void;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_gpu::LaneStream;
use euhedral_inference_cuda::launch::ops::FreeFailure;
use euhedral_inference_cuda::{CudaError, CudaLibrary, DeviceBuffer, DevicePtr, PinnedPool, PooledPinned};
use euhedral_memory::Claim;

use crate::gptq::GptqError;

/// Slots in one chunk of the queue: an allocation step of an unbounded queue, never a limit.
const CHUNK: usize = 16;

/// The bytes a step charges to the memory planner: take claims of these sizes and pass them with the step.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StepBytes {
    /// Device bytes the step allocates, which stay allocated until the buffers are released.
    pub device: u64,
    /// Page-locked staging bytes the step holds until it ends.
    pub pinned: u64,
}

/// The memory planner's claims for one step, each at least as large as the step's [`StepBytes`]. A step that allocates
/// refuses a missing claim: no allocation is unaccounted.
#[derive(Debug, Default)]
pub struct Claims {
    pub device: Option<Claim>,
    pub pinned: Option<Claim>,
}

struct Held {
    buffer: DeviceBuffer,
    claim: Option<Claim>,
}

/// The device buffers given up and waiting for the owner of a lane to free them in stream order.
pub struct Released {
    producer: Producer<Held, Multi>,
    consumer: Consumer<Held, Multi>,
}

impl Debug for Released {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("Released")
            .field("waiting", &self.consumer.size_long())
            .finish()
    }
}

impl Default for Released {
    fn default() -> Self {
        Self::new()
    }
}

impl Released {
    pub fn new() -> Released {
        let queue = MpmcQueue::<Held>::new(CHUNK);
        Released {
            producer: queue.producer(),
            consumer: queue.consumer(),
        }
    }

    /// Buffers given up and not yet freed.
    pub fn waiting(&self) -> u64 {
        self.consumer.size_long()
    }

    fn give(&self, held: Held) {
        let offered = self.producer.offer(held);
        debug_assert!(offered.is_ok(), "an unbounded queue refused a buffer");
    }
}

/// Frees, in stream order on the stream `queue` lends, every buffer given up, and returns how many. A buffer the stream
/// could not free is parked again, still allocated and still charged, for the next call (a plain free would wait for
/// every lane). The first failure is returned after the rest are freed.
pub fn reclaim<Q: LaneStream<Raw = *mut c_void>>(
    released: &Released,
    library: &Arc<CudaLibrary>,
    queue: &Q,
) -> Result<u64, CudaError> {
    let mut failure = None;
    let mut parked = Vec::new();
    let freed = released.consumer.drain_all(|held| {
        let Held { buffer, claim } = held;
        match buffer.free_on_lane(library, queue) {
            // The free is enqueued: the bytes are the planner's again.
            Ok(()) => drop(claim),
            Err(FreeFailure { buffer, error }) => {
                failure.get_or_insert(error);
                match buffer {
                    Some(buffer) => parked.push(Held { buffer, claim }),
                    None => drop(claim),
                }
            }
        }
    });
    let freed = freed - parked.len() as u64;
    for held in parked {
        released.give(held);
    }
    failure.map_or(Ok(freed), Err)
}

/// A stream-ordered device buffer that is handed to the [`Released`] queue when dropped.
#[derive(Debug)]
pub struct Device {
    held: Option<Held>,
    released: Arc<Released>,
}

impl Debug for Held {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter
            .debug_struct("Held")
            .field("bytes", &self.buffer.bytes())
            .finish()
    }
}

impl Device {
    fn buffer(&self) -> Option<&DeviceBuffer> {
        self.held.as_ref().map(|held| &held.buffer)
    }

    /// Whether this buffer was allocated on the lane whose queue is `released`.
    pub fn belongs_to(&self, released: &Arc<Released>) -> bool {
        Arc::ptr_eq(&self.released, released)
    }

    /// The device address.
    pub fn addr(&self) -> u64 {
        self.buffer().map_or(0, DeviceBuffer::addr)
    }

    /// The device address as a pointer.
    pub fn ptr(&self) -> DevicePtr {
        self.buffer().map_or(DevicePtr::NULL, DeviceBuffer::ptr)
    }

    /// The size in bytes.
    pub fn bytes(&self) -> usize {
        self.buffer().map_or(0, DeviceBuffer::bytes)
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        if let Some(held) = self.held.take() {
            self.released.give(held);
        }
    }
}

/// Pooled page-locked bytes; the pool takes them back when this is dropped.
#[derive(Debug)]
pub struct Host {
    pooled: PooledPinned,
    bytes: usize,
    _claim: Option<Claim>,
}

impl Host {
    /// The size asked for (the pooled buffer may be larger).
    pub fn len(&self) -> usize {
        self.bytes
    }

    pub fn is_empty(&self) -> bool {
        self.bytes == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        &self.pooled.buffer().as_slice()[..self.bytes]
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        let bytes = self.bytes;
        &mut self.pooled.buffer_mut().as_mut_slice()[..bytes]
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.pooled.buffer().as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.pooled.buffer_mut().as_mut_ptr()
    }
}

/// The allocations of one step, made inside its submission on the lane's stream.
pub struct Memory<'a, Q> {
    queue: &'a Q,
    library: &'a Arc<CudaLibrary>,
    released: &'a Arc<Released>,
    pool: &'a Arc<PinnedPool>,
    claims: Claims,
}

impl<Q> Debug for Memory<'_, Q> {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> FmtResult {
        formatter.debug_struct("Memory").finish_non_exhaustive()
    }
}

impl<'a, Q: LaneStream<Raw = *mut c_void>> Memory<'a, Q> {
    pub fn new(
        queue: &'a Q,
        library: &'a Arc<CudaLibrary>,
        released: &'a Arc<Released>,
        pool: &'a Arc<PinnedPool>,
        claims: Claims,
    ) -> Self {
        Self {
            queue,
            library,
            released,
            pool,
            claims,
        }
    }

    /// `bytes` of device memory, in stream order.
    pub fn device(&mut self, bytes: usize) -> Result<Device, GptqError> {
        let bytes = bytes.max(1);
        let claim = charge(&mut self.claims.device, bytes, "device")?;
        let buffer = DeviceBuffer::alloc_on_lane(self.library, self.queue, bytes)?;
        Ok(Device {
            held: Some(Held { buffer, claim }),
            released: Arc::clone(self.released),
        })
    }

    /// `bytes` of pooled page-locked memory.
    pub fn pinned(&mut self, bytes: usize) -> Result<Host, GptqError> {
        let bytes = bytes.max(1);
        // The pool hands out its size class: that is what the buffer takes of the pinned memory. Charge it before the
        // pool can allocate, so a refused step allocates nothing.
        let class = PinnedPool::class_bytes(bytes)
            .ok_or_else(|| GptqError::Request(format!("{bytes} bytes is too large for a pinned buffer")))?;
        let claim = charge(&mut self.claims.pinned, class, "pinned")?;
        let pooled = self.pool.acquire(bytes)?;
        Ok(Host {
            pooled,
            bytes,
            _claim: claim,
        })
    }
}

/// `bytes` out of `claim`, as a claim of their own.
fn charge(claim: &mut Option<Claim>, bytes: usize, kind: &'static str) -> Result<Option<Claim>, GptqError> {
    let Some(claim) = claim else {
        return Err(GptqError::Request(format!(
            "the step was given no {kind} claim for the {bytes} bytes it allocates"
        )));
    };
    let owner = claim.owner();
    claim.take(owner, bytes as u64, "gptq").map(Some).ok_or_else(|| {
        GptqError::Request(format!(
            "the {kind} claim has fewer than the {bytes} bytes an allocation needs"
        ))
    })
}

/// The pinned bytes of staging buffers asked for as `sizes`: each takes its pool size class, which is what a claim
/// must cover.
pub fn pinned_bytes(sizes: impl IntoIterator<Item = u64>) -> u64 {
    sizes
        .into_iter()
        .map(|bytes| {
            let bytes = usize::try_from(bytes.max(1)).unwrap_or(usize::MAX);
            PinnedPool::class_bytes(bytes).map_or(u64::MAX, |class| class as u64)
        })
        .fold(0, u64::saturating_add)
}
