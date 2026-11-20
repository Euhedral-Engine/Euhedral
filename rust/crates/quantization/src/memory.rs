//! The memory of a quantization chain: stream-ordered device buffers that are never freed inside a frame, and pooled pinned
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

use std::array::from_fn;
use std::ffi::c_void;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::sync::Arc;

use euhedral_data_structures::queues::{Consumer, MpmcQueue, Multi, Producer};
use euhedral_gpu::{GpuError, LaneEnqueue, LaneStream};
use euhedral_inference_cuda::launch::ops::FreeFailure;
use euhedral_inference_cuda::{CudaError, CudaLibrary, DeviceBuffer, DevicePtr, PinnedPool, PooledPinned};
use euhedral_memory::Claim;
use thiserror::Error;

/// Why an allocation was refused or failed.
#[derive(Debug, Error)]
pub enum MemoryError {
    /// The allocator or the pool failed.
    #[error(transparent)]
    Cuda(#[from] CudaError),
    /// The claims do not cover the allocation, or the size cannot be allocated.
    #[error("{0}")]
    Refused(String),
}

/// The memory a lane works with: the C library its stream-ordered allocator belongs to, the queue where dropped device
/// buffers wait to be freed in stream order, and the pool the pinned staging comes from (its limit is the pinned
/// buffer pool setting). The queue belongs to the one lane it is made for: a buffer is freed on the stream it was
/// allocated on, so two lanes never share one.
#[derive(Debug)]
pub struct LaneMemory {
    pub(crate) library: Arc<CudaLibrary>,
    pub(crate) released: Arc<Released>,
    pub(crate) pinned: Arc<PinnedPool>,
}

impl LaneMemory {
    /// The memory of a new lane, with a queue of its own.
    pub fn new(library: Arc<CudaLibrary>, pinned: Arc<PinnedPool>) -> LaneMemory {
        LaneMemory {
            library,
            released: Arc::new(Released::new()),
            pinned,
        }
    }

    /// The device buffers given up and not yet freed.
    pub fn waiting(&self) -> u64 {
        self.released.waiting()
    }
}

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

/// A range of device memory a step reads or writes, with its size.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeviceRange {
    addr: u64,
    bytes: u64,
}

impl DeviceRange {
    /// `bytes` of device memory at `addr`.
    ///
    /// # Safety
    /// The range must be live device memory of at least `bytes` bytes, which nothing writes (when the step reads it)
    /// or reads (when the step writes it) and which is not freed until the step's work has run: its `done` call, or,
    /// when the step failed with the lane's `unsettled` count not 0, until that count is back
    /// to 0 or the lane's next step is confirmed.
    pub unsafe fn new(addr: u64, bytes: u64) -> DeviceRange {
        DeviceRange { addr, bytes }
    }

    pub fn addr(&self) -> u64 {
        self.addr
    }

    pub fn bytes(&self) -> u64 {
        self.bytes
    }

    /// The address, when the range holds at least `needed` bytes.
    pub fn holding(&self, what: &str, needed: u64) -> Result<u64, MemoryError> {
        if self.addr == 0 || self.bytes < needed {
            return Err(MemoryError::Refused(format!(
                "the {what} needs {needed} device bytes and the range holds {}",
                self.bytes
            )));
        }
        Ok(self.addr)
    }
}

impl Claims {
    /// Refuses a missing claim, or one smaller than `needed`, before the step allocates anything. A claim that
    /// passes can still be refused part-way when the planner has meanwhile shrunk it (an elastic claim) or its owner
    /// has gone; the step then gives back what it took.
    pub fn cover(&self, needed: StepBytes) -> Result<(), MemoryError> {
        for (kind, claim, bytes) in [
            ("device", &self.device, needed.device),
            ("pinned", &self.pinned, needed.pinned),
        ] {
            match claim {
                None => {
                    return Err(MemoryError::Refused(format!(
                        "the step was given no {kind} claim for the {bytes} bytes it allocates"
                    )));
                }
                Some(claim) if claim.bytes() < bytes => {
                    return Err(MemoryError::Refused(format!(
                        "the {kind} claim has {} bytes, fewer than the {bytes} the step allocates",
                        claim.bytes()
                    )));
                }
                Some(_) => {}
            }
        }
        Ok(())
    }
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
    /// The buffer, for a copy that waits (a test's, or a caller that has confirmed the stream is done with it).
    pub fn buffer(&self) -> Option<&DeviceBuffer> {
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
    pub(crate) fn new(
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
    pub fn device(&mut self, bytes: usize) -> Result<Device, MemoryError> {
        let bytes = bytes.max(1);
        let claim = charge(&mut self.claims.device, bytes, "device")?;
        let buffer = DeviceBuffer::alloc_on_lane(self.library, self.queue, bytes)?;
        Ok(Device {
            held: Some(Held { buffer, claim }),
            released: Arc::clone(self.released),
        })
    }

    /// `bytes` of pooled page-locked memory.
    pub fn pinned(&mut self, bytes: usize) -> Result<Host, MemoryError> {
        let bytes = bytes.max(1);
        // The pool hands out its size class: that is what the buffer takes of the pinned memory. Charge it before the
        // pool can allocate, so a refused step allocates nothing.
        let class = PinnedPool::class_bytes(bytes)
            .ok_or_else(|| MemoryError::Refused(format!("{bytes} bytes is too large for a pinned buffer")))?;
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
fn charge(claim: &mut Option<Claim>, bytes: usize, kind: &'static str) -> Result<Option<Claim>, MemoryError> {
    let Some(claim) = claim else {
        return Err(MemoryError::Refused(format!(
            "the step was given no {kind} claim for the {bytes} bytes it allocates"
        )));
    };
    let owner = claim.owner();
    claim.take(owner, bytes as u64, "quantize").map(Some).ok_or_else(|| {
        MemoryError::Refused(format!(
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

/// A device buffer with a page-locked twin of the same size: what goes up, or what comes back.
#[derive(Debug)]
pub struct Staged {
    pub host: Host,
    pub device: Device,
}

impl Staged {
    pub(crate) fn new<Q: LaneStream<Raw = *mut c_void>>(
        memory: &mut Memory<'_, Q>,
        bytes: usize,
    ) -> Result<Staged, MemoryError> {
        Ok(Staged {
            host: memory.pinned(bytes)?,
            device: memory.device(bytes)?,
        })
    }

    pub fn of_u32<Q: LaneStream<Raw = *mut c_void>>(
        memory: &mut Memory<'_, Q>,
        values: &[u32],
    ) -> Result<Staged, MemoryError> {
        let mut staged = Staged::new(memory, values.len() * 4)?;
        staged.fill_u32(values);
        Ok(staged)
    }

    pub fn of_f32<Q: LaneStream<Raw = *mut c_void>>(
        memory: &mut Memory<'_, Q>,
        values: &[f32],
    ) -> Result<Staged, MemoryError> {
        let mut staged = Staged::new(memory, values.len() * 4)?;
        staged.fill_f32(values);
        Ok(staged)
    }

    pub fn addr(&self) -> u64 {
        self.device.addr()
    }

    pub fn fill_u32(&mut self, values: &[u32]) {
        for (slot, value) in self.host.as_mut_slice().as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *slot = value.to_le_bytes();
        }
    }

    pub fn fill_f32(&mut self, values: &[f32]) {
        for (slot, value) in self.host.as_mut_slice().as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *slot = value.to_le_bytes();
        }
    }

    pub fn fill_i32(&mut self, values: &[i32]) {
        for (slot, value) in self.host.as_mut_slice().as_chunks_mut::<4>().0.iter_mut().zip(values) {
            *slot = value.to_le_bytes();
        }
    }

    pub fn read_f32(&self) -> Vec<f32> {
        let (words, _) = self.host.as_slice().as_chunks::<4>();
        words.iter().map(|word| f32::from_le_bytes(*word)).collect()
    }

    /// The first `count` floats.
    pub fn floats(&self, count: usize) -> Vec<f32> {
        let (words, _) = self.host.as_slice().as_chunks::<4>();
        words.iter().take(count).map(|word| f32::from_le_bytes(*word)).collect()
    }

    pub fn halves(&self) -> Vec<u16> {
        let (words, _) = self.host.as_slice().as_chunks::<2>();
        words.iter().map(|word| u16::from_le_bytes(*word)).collect()
    }

    fn bytes<const N: usize>(&self, offset: usize) -> [u8; N] {
        let host = self.host.as_slice();
        from_fn(|i| host.get(offset + i).copied().unwrap_or(0))
    }

    pub fn float_at(&self, offset: usize) -> f32 {
        f32::from_le_bytes(self.bytes(offset))
    }

    pub fn int_at(&self, offset: usize) -> i32 {
        i32::from_le_bytes(self.bytes(offset))
    }

    pub fn double_at(&self, offset: usize) -> f64 {
        f64::from_le_bytes(self.bytes(offset))
    }

    /// Enqueues the copy of the host bytes to the device. The host bytes must stay as they are until it retires.
    pub(crate) fn upload<Q: LaneEnqueue>(&self, queue: &mut Q) -> Result<(), GpuError> {
        // SAFETY: both ranges are live buffers of `len` bytes owned by this value, which the chain keeps until the
        // retirement is confirmed; nothing writes the host side meanwhile.
        unsafe { queue.copy_to_device(self.device.addr(), self.host.as_ptr(), self.host.len()) }?;
        Ok(())
    }

    /// Enqueues the copy of the device bytes to the host buffer, which is read after the retirement.
    pub(crate) fn download<Q: LaneEnqueue>(&mut self, queue: &mut Q) -> Result<(), GpuError> {
        let bytes = self.host.len();
        // SAFETY: as `upload`; nothing reads the host side before the retirement.
        unsafe { queue.copy_to_host(self.host.as_mut_ptr(), self.device.addr(), bytes) }?;
        Ok(())
    }
}
