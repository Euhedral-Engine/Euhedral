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
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::fmt::{Debug, Formatter, Result as FmtResult};
use std::iter::repeat_n;
use std::sync::Arc;

use euhedral_gpu::{GpuError, LaneEnqueue, LaneStream};
use euhedral_inference_cuda::launch::ops::FreeFailure;
use euhedral_inference_cuda::released::Released as GivenUp;
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

/// A device buffer with the planner claim it is charged to, the item the queue carries.
pub struct Held {
    buffer: DeviceBuffer,
    claim: Option<Claim>,
}

/// The device buffers given up and waiting for the owner of a lane to free them in stream order.
pub type Released = GivenUp<Held>;

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
    let freed = released.drain(|held| {
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

/// The pinned staging buffers a run's steps hold at once, by size class: what [`LaneMemory::prewarm`] puts in the pool
/// before the first step, so that no step allocates page-locked memory.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct StagingNeeds {
    /// For each size class in bytes, the most buffers of it one step holds at once.
    classes: BTreeMap<usize, usize>,
}

impl StagingNeeds {
    /// No staging.
    pub fn new() -> StagingNeeds {
        StagingNeeds::default()
    }

    /// Adds one step that holds a staging buffer of each of `sizes` bytes at once. A class's count is the most any
    /// added step holds of it (steps of a lane run one after another). Refuses a size no class holds; nothing is added
    /// then.
    pub fn step(&mut self, sizes: impl IntoIterator<Item = u64>) -> Result<(), MemoryError> {
        let mut held = BTreeMap::<usize, usize>::new();
        for bytes in sizes {
            let wanted = usize::try_from(bytes.max(1)).unwrap_or(usize::MAX);
            let class = PinnedPool::class_bytes(wanted)
                .ok_or_else(|| MemoryError::Refused(format!("{bytes} bytes is too large for a pinned buffer")))?;
            *held.entry(class).or_default() += 1;
        }
        for (class, count) in held {
            let most = self.classes.entry(class).or_default();
            *most = (*most).max(count);
        }
        Ok(())
    }

    /// The pool bytes of every buffer: what a claim for the prewarm must cover.
    pub fn bytes(&self) -> u64 {
        self.classes
            .iter()
            .map(|(class, count)| *class as u64 * *count as u64)
            .fold(0, u64::saturating_add)
    }

    /// The size class of each buffer, one entry per buffer, smallest class first.
    pub fn buffers(&self) -> impl Iterator<Item = usize> + '_ {
        self.classes.iter().flat_map(|(class, count)| repeat_n(*class, *count))
    }
}

/// Holds one buffer from `acquire` for each of `needs`, then gives them all back together, so that each lives in the
/// pool afterwards. The buffers' bytes are charged to `claim` for as long as they are held. Allocating a buffer is the
/// only work `acquire` does; it is called outside any step.
fn warm<T>(
    needs: &StagingNeeds,
    claim: &mut Option<Claim>,
    mut acquire: impl FnMut(usize) -> Result<T, MemoryError>,
) -> Result<(), MemoryError> {
    let charged = charge(claim, usize::try_from(needs.bytes()).unwrap_or(usize::MAX), "pinned")?;
    let mut held = Vec::new();
    for class in needs.buffers() {
        held.push(acquire(class)?);
    }
    drop(held);
    drop(charged);
    Ok(())
}

impl LaneMemory {
    /// Puts the buffers of `needs` in the pinned pool: it holds at least that many of each class, taken from `claim`
    /// while they are acquired. Allocates (waits on the driver for each new buffer): call it outside a frame, before
    /// the first step. Fails with the pool's error, or when `claim` is missing or holds fewer bytes than
    /// [`StagingNeeds::bytes`]; buffers acquired by then stay in the pool.
    pub fn prewarm(&self, needs: &StagingNeeds, claim: &mut Option<Claim>) -> Result<(), MemoryError> {
        warm(needs, claim, |class| Ok(self.pinned.acquire(class)?))
    }
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

#[cfg(test)]
#[allow(
    clippy::unwrap_used,
    clippy::expect_used,
    reason = "a test: a failed unwrap is the failure"
)]
mod tests {
    use std::cell::RefCell;
    use std::rc::Rc;

    use euhedral_memory::{LocationKind, LocationSpec, OwnerSpec, Planner, Priority, Scope};

    use super::*;

    /// A pool of staging buffers that allocates only when it holds none of the class.
    #[derive(Default)]
    struct FakePool {
        held: RefCell<Vec<usize>>,
        allocated: RefCell<Vec<usize>>,
    }

    /// A loan: dropping it gives the buffer back to the pool.
    struct Loan {
        pool: Rc<FakePool>,
        class: usize,
    }

    impl Drop for Loan {
        fn drop(&mut self) {
            self.pool.held.borrow_mut().push(self.class);
        }
    }

    fn acquire(pool: &Rc<FakePool>, class: usize) -> Result<Loan, MemoryError> {
        let mut held = pool.held.borrow_mut();
        match held.iter().position(|have| *have == class) {
            Some(at) => {
                held.swap_remove(at);
            }
            None => pool.allocated.borrow_mut().push(class),
        }
        Ok(Loan {
            pool: Rc::clone(pool),
            class,
        })
    }

    fn claim_of(bytes: u64) -> (Arc<Planner>, Option<Claim>) {
        let planner = Planner::new([LocationSpec {
            kind: LocationKind::HostPinned,
            capacity: 1 << 30,
            reserve: 0,
        }]);
        let owner = planner.open_owner(OwnerSpec::new("test", Priority(0), Scope::Process));
        let claim = planner
            .claim(owner, euhedral_memory::LocationId(0), bytes, "test")
            .unwrap();
        (planner, Some(claim))
    }

    fn needs_of(steps: &[&[u64]]) -> StagingNeeds {
        let mut needs = StagingNeeds::new();
        for step in steps {
            needs.step(step.iter().copied()).unwrap();
        }
        needs
    }

    #[test]
    fn a_step_needs_its_size_classes_and_a_class_counts_the_most_one_step_holds() {
        let needs = needs_of(&[&[1, 4096, 5000], &[4096, 5000, 5000, 9000], &[100]]);
        assert_eq!(
            needs.buffers().collect::<Vec<_>>(),
            vec![4096, 4096, 8192, 8192, 16384],
            "two of 4096 (the first step), two of 8192 (the second), one of 16384"
        );
        assert_eq!(needs.bytes(), 2 * 4096 + 2 * 8192 + 16384);
        assert_eq!(StagingNeeds::new().bytes(), 0);
        assert_eq!(StagingNeeds::new().buffers().count(), 0);
    }

    #[test]
    fn a_size_no_class_holds_is_refused_and_adds_nothing() {
        let mut needs = needs_of(&[&[4096]]);
        let before = needs.clone();
        assert!(matches!(needs.step([4096, u64::MAX]), Err(MemoryError::Refused(_))));
        assert_eq!(needs, before);
    }

    #[test]
    fn the_prewarm_acquires_exactly_the_classes_the_steps_need_and_leaves_them_in_the_pool() {
        let needs = needs_of(&[&[4096, 4096, 20_000], &[70_000]]);
        let (_planner, mut claim) = claim_of(needs.bytes());
        let pool = Rc::new(FakePool::default());
        warm(&needs, &mut claim, |class| acquire(&pool, class)).unwrap();
        let mut allocated = pool.allocated.borrow().clone();
        allocated.sort_unstable();
        assert_eq!(allocated, vec![4096, 4096, 32_768, 131_072]);
        let mut held = pool.held.borrow().clone();
        held.sort_unstable();
        assert_eq!(held, allocated, "every buffer is back in the pool");

        // The first step then takes its buffers from the pool: nothing is allocated.
        let loans: Vec<Loan> = [4096u64, 4096, 20_000]
            .into_iter()
            .map(|bytes| acquire(&pool, PinnedPool::class_bytes(bytes as usize).unwrap()).unwrap())
            .collect();
        assert_eq!(pool.allocated.borrow().len(), 4, "the step allocated nothing");
        drop(loans);
    }

    #[test]
    fn a_prewarm_without_a_claim_that_covers_it_acquires_nothing() {
        let needs = needs_of(&[&[4096, 8192]]);
        let pool = Rc::new(FakePool::default());
        assert!(matches!(
            warm(&needs, &mut None, |class| acquire(&pool, class)),
            Err(MemoryError::Refused(_))
        ));
        let (_planner, mut small) = claim_of(needs.bytes() - 1);
        assert!(matches!(
            warm(&needs, &mut small, |class| acquire(&pool, class)),
            Err(MemoryError::Refused(_))
        ));
        assert!(pool.allocated.borrow().is_empty());
    }

    #[test]
    fn a_failed_acquisition_ends_the_prewarm_with_its_error() {
        let needs = needs_of(&[&[4096, 8192]]);
        let (_planner, mut claim) = claim_of(needs.bytes());
        let mut calls = 0;
        let result = warm(&needs, &mut claim, |_| {
            calls += 1;
            Err::<(), _>(MemoryError::Refused("the pool is out".to_string()))
        });
        assert!(matches!(result, Err(MemoryError::Refused(text)) if text == "the pool is out"));
        assert_eq!(calls, 1);
    }
}
