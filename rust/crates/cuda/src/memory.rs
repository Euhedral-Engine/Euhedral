//! Device, pinned and host-weight memory. Each owner frees its allocation on drop.

use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::slice;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};

#[cfg(any(target_os = "linux", target_os = "android", windows))]
use euhedral_io::IoMemory;

use crate::error::{CudaError, check};
use crate::library::CudaLibrary;

/// A device address, opaque to the host. Zero is null.
///
/// Safe code gets one from a [`DeviceBuffer`] (and offsets within it). Making one from a number is `unsafe`: an
/// address into mapped host memory is host memory, and the safe operations would let a kernel write it behind Rust's
/// references.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DevicePtr(u64);

impl DevicePtr {
    pub const NULL: DevicePtr = DevicePtr(0);

    /// A device pointer from its raw address.
    ///
    /// # Safety
    /// `address` must be null, or an address of device memory the caller owns for as long as the pointer is used, or
    /// mapped host memory under the contract of [`PinnedBuffer::device_address`].
    pub unsafe fn from_raw(address: u64) -> DevicePtr {
        DevicePtr(address)
    }

    /// The raw address.
    pub fn get(self) -> u64 {
        self.0
    }

    /// The address `bytes` further on.
    pub fn offset(self, bytes: u64) -> DevicePtr {
        DevicePtr(self.0 + bytes)
    }

    pub fn is_null(self) -> bool {
        self.0 == 0
    }

    /// The address as the C ABI takes it. The pointer is never dereferenced on the host.
    pub(crate) fn as_const(self) -> *const c_void {
        ptr::with_exposed_provenance(self.0 as usize)
    }

    pub(crate) fn as_mut(self) -> *mut c_void {
        ptr::with_exposed_provenance_mut(self.0 as usize)
    }
}

/// The serial of one device allocation. Serials are never reused, so a captured graph that read a buffer is told
/// apart from a later allocation at the same address.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct AllocationId(pub u64);

/// The bytes of the live [`DeviceBuffer`]s of this process: raised by an allocation that succeeded, lowered when its
/// buffer is freed. A failed allocation leaves it where it was.
#[derive(Debug, Default)]
struct LiveBytes(AtomicU64);

impl LiveBytes {
    const fn new() -> LiveBytes {
        LiveBytes(AtomicU64::new(0))
    }

    fn allocated(&self, bytes: usize) {
        self.0.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    fn freed(&self, bytes: usize) {
        self.0.fetch_sub(bytes as u64, Ordering::Relaxed);
    }

    fn get(&self) -> u64 {
        self.0.load(Ordering::Relaxed)
    }
}

static LIVE_DEVICE_BYTES: LiveBytes = LiveBytes::new();

/// A device allocation, freed on drop.
///
/// A plain drop frees the buffer at once with `cudaFree`, which waits for the whole device under the driver's lock (and
/// frees memory queued work may still use if the caller dropped it too early: nothing here tracks readers). That is for
/// startup and shutdown only. To free on a running engine, allocate with [`LaneOps::alloc_async`](crate::LaneOps::alloc_async) and free it in stream
/// order with [`LaneOps::free_async`](crate::LaneOps::free_async), which uses `cudaFreeAsync` and never waits. A buffer
/// from [`DeviceBuffer::alloc`] cannot be freed that way ([`DeviceBuffer::is_stream_ordered`]).
#[derive(Debug)]
pub struct DeviceBuffer {
    library: Arc<CudaLibrary>,
    addr: u64,
    bytes: usize,
    generation: AllocationId,
    /// Allocated with `cudaMallocAsync`: freed in stream order, so it can be freed without blocking.
    stream_ordered: bool,
}

impl DeviceBuffer {
    /// Allocates `bytes` of device memory.
    pub fn alloc(library: &Arc<CudaLibrary>, bytes: usize) -> Result<DeviceBuffer, CudaError> {
        let size = non_empty(bytes)?;
        // SAFETY: the call takes a size.
        let address = unsafe { library.api().euhedral_cuda_malloc(size) };
        Self::wrap(library, "euhedral_cuda_malloc", address, bytes, false)
    }

    /// Allocates `bytes` in stream order on the stream the calling thread has selected in the C library.
    pub(crate) fn alloc_on_selected(library: &Arc<CudaLibrary>, bytes: usize) -> Result<DeviceBuffer, CudaError> {
        let size = non_empty(bytes)?;
        // SAFETY: the caller has selected a live stream, which the call allocates on.
        let address = unsafe { library.api().euhedral_cuda_malloc_async(size) };
        Self::wrap(library, "euhedral_cuda_malloc_async", address, bytes, true)
    }

    /// Frees the buffer in stream order on the stream the calling thread has selected in the C library.
    pub(crate) fn free_on_selected(mut self) -> Result<(), CudaError> {
        // SAFETY: the address is this buffer's live allocation, and the caller has selected a live stream.
        let status = unsafe { self.library.api().euhedral_cuda_free_async(self.ptr().as_mut()) };
        // The allocation is gone either way: a failed stream-ordered free must not become a second free on drop.
        self.addr = 0;
        LIVE_DEVICE_BYTES.freed(self.bytes);
        check("euhedral_cuda_free_async", status)
    }

    fn wrap(
        library: &Arc<CudaLibrary>,
        call: &'static str,
        address: *mut c_void,
        bytes: usize,
        stream_ordered: bool,
    ) -> Result<DeviceBuffer, CudaError> {
        if address.is_null() {
            return Err(CudaError::NullAllocation {
                call,
                bytes: bytes as u64,
            });
        }
        LIVE_DEVICE_BYTES.allocated(bytes);
        Ok(DeviceBuffer {
            library: Arc::clone(library),
            addr: address.expose_provenance() as u64,
            bytes,
            generation: library.next_allocation(),
            stream_ordered,
        })
    }

    /// The bytes of every live device buffer of this process, whichever library allocated them: what the process
    /// itself holds on the device through this crate, unmoved by other processes on a shared card.
    pub fn live_bytes() -> u64 {
        LIVE_DEVICE_BYTES.get()
    }

    /// The device address.
    pub fn addr(&self) -> u64 {
        self.addr
    }

    /// The device address as a [`DevicePtr`].
    pub fn ptr(&self) -> DevicePtr {
        DevicePtr(self.addr)
    }

    /// The size in bytes.
    pub fn bytes(&self) -> usize {
        self.bytes
    }

    /// Whether the buffer was allocated stream-ordered, so that it can be freed in stream order without waiting.
    pub fn is_stream_ordered(&self) -> bool {
        self.stream_ordered
    }

    /// The allocation's serial.
    pub fn generation(&self) -> AllocationId {
        self.generation
    }

    /// Copies `host` into the buffer at `offset` and returns when the copy is done. Startup, shutdown and tests: it
    /// blocks, and unless it runs inside a lane's lent stream (`LaneTable::submit_host`) it runs on the legacy stream, which does not order against lane work.
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, synchronous copy: documented as startup, shutdown and tests only"
    )]
    pub fn copy_from_host(&self, offset: usize, host: &[u8]) -> Result<(), CudaError> {
        let destination = self.range(offset, host.len())?;
        // SAFETY: `destination` lies inside this allocation and `host` is a live slice of the stated length.
        let status = unsafe {
            self.library.api().euhedral_cuda_copy_host_to_device(
                destination.as_mut(),
                host.as_ptr().cast(),
                host.len() as u64,
            )
        };
        check("euhedral_cuda_copy_host_to_device", status)
    }

    /// Copies `host.len()` bytes from the buffer at `offset` and returns when the copy is done. See
    /// [`DeviceBuffer::copy_from_host`].
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, synchronous copy: documented as startup, shutdown and tests only"
    )]
    pub fn copy_to_host(&self, offset: usize, host: &mut [u8]) -> Result<(), CudaError> {
        let source = self.range(offset, host.len())?;
        // SAFETY: `source` lies inside this allocation and `host` is a live exclusive slice of the stated length.
        let status = unsafe {
            self.library.api().euhedral_cuda_copy_device_to_host(
                host.as_mut_ptr().cast(),
                source.as_const(),
                host.len() as u64,
            )
        };
        check("euhedral_cuda_copy_device_to_host", status)
    }

    /// The address of `length` bytes at `offset`, refused when they leave the buffer.
    fn range(&self, offset: usize, length: usize) -> Result<DevicePtr, CudaError> {
        match offset.checked_add(length) {
            Some(end) if end <= self.bytes => Ok(self.ptr().offset(offset as u64)),
            _ => Err(CudaError::Argument(format!(
                "{length} bytes at offset {offset} lie outside a {} byte buffer",
                self.bytes
            ))),
        }
    }
}

impl Drop for DeviceBuffer {
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, cudaFree waits for the device: long-lived buffers are freed at shutdown, stream-ordered ones with free_async"
    )]
    fn drop(&mut self) {
        if self.addr == 0 {
            return;
        }
        // SAFETY: the address is this buffer's live allocation, freed once. A failure leaves nothing to recover.
        let _ = unsafe { self.library.api().euhedral_cuda_free(self.ptr().as_mut()) };
        LIVE_DEVICE_BYTES.freed(self.bytes);
    }
}

fn non_empty(bytes: usize) -> Result<u64, CudaError> {
    if bytes == 0 {
        return Err(CudaError::Argument("an allocation needs at least one byte".to_string()));
    }
    Ok(bytes as u64)
}

/// Page-locked host memory (`cudaHostAlloc`), freed on drop. Copies between it and the device run on the copy
/// engines without staging, and the device can write a word in it (the lane completion word).
///
/// It is owned exclusively: the pointer and length are fixed for the buffer's life. A plain drop calls `cudaFreeHost`,
/// which waits for the whole device under the driver's lock: startup and shutdown only. On a running engine take
/// buffers from a [`PinnedPool`](crate::PinnedPool), which recycles them. A raw `PinnedBuffer` is never freed on a running engine.
#[derive(Debug)]
pub struct PinnedBuffer {
    library: Arc<CudaLibrary>,
    pointer: NonNull<u8>,
    len: usize,
}

// SAFETY: the buffer owns its allocation exclusively; the pointer is not tied to a thread, and the shared accessors
// only read through it or use atomics.
unsafe impl Send for PinnedBuffer {}
// SAFETY: see `Send`; a `&PinnedBuffer` offers reads and atomic word accesses only.
unsafe impl Sync for PinnedBuffer {}

impl PinnedBuffer {
    /// Allocates `bytes` of page-locked host memory.
    pub fn alloc(library: &Arc<CudaLibrary>, bytes: usize) -> Result<PinnedBuffer, CudaError> {
        let size = non_empty(bytes)?;
        // SAFETY: the call takes a size.
        let address = unsafe { library.api().euhedral_cuda_host_malloc(size) };
        let pointer = NonNull::new(address.cast::<u8>()).ok_or(CudaError::NullAllocation {
            call: "euhedral_cuda_host_malloc",
            bytes: size,
        })?;
        Ok(PinnedBuffer {
            library: Arc::clone(library),
            pointer,
            len: bytes,
        })
    }

    /// The start of the memory.
    pub fn as_ptr(&self) -> *const u8 {
        self.pointer.as_ptr()
    }

    /// The start of the memory, for writing.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.pointer.as_ptr()
    }

    /// The size in bytes.
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// The bytes. Read memory a queued readback fills only after the lane's work has retired.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is `len` bytes, live and initialised by the allocator or by earlier writes.
        unsafe { slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }

    /// The bytes, for writing. Write memory a queued upload reads only after the lane's work has retired.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, and the borrow is exclusive.
        unsafe { slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len) }
    }

    /// The 32-bit word `index` as an atomic, for words the device reads or writes (stream flags).
    pub fn word(&self, index: usize) -> &AtomicU32 {
        assert!(
            index
                .checked_mul(4)
                .and_then(|offset| offset.checked_add(4))
                .is_some_and(|end| end <= self.len),
            "word {index} lies outside a {} byte buffer",
            self.len
        );
        // SAFETY: the word lies inside the allocation, and page-locked allocations are page aligned.
        unsafe { AtomicU32::from_ptr(self.pointer.as_ptr().cast::<u32>().add(index)) }
    }

    /// The address at which the device reads and writes this memory.
    ///
    /// # Safety
    /// Device work through the address races with the host's references to the same bytes ([`PinnedBuffer::as_slice`],
    /// [`PinnedBuffer::as_mut_slice`] and the pointers). The caller must keep the buffer alive until that work has
    /// retired, and touch the bytes the device accesses only through atomics ([`PinnedBuffer::word`]) or after the
    /// work retires.
    pub unsafe fn device_address(&self) -> Result<DevicePtr, CudaError> {
        device_address(&self.library, self.pointer.as_ptr().cast_const().cast())
    }
}

/// A pinned buffer is a region `AsyncIo::read_into` can fill: the file's bytes land in page-locked memory, ready for
/// a copy engine, with no staging copy in between.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
// SAFETY: the pointer addresses `len` bytes of one live `cudaHostAlloc` allocation, valid for reads and writes, that
// the buffer owns exclusively and that stays where it is when the value moves: the buffer holds a raw `NonNull`
// pointer to memory outside Rust's allocator, so a move retags nothing. Every bit pattern is a valid byte, which is
// all `initialized` can mean for memory a kernel overwrites (the same claim `as_slice` makes). The region is
// exclusively the operation's from submission until the continuation gets the buffer back, because the operation owns
// the value, and the value touches the region only through this pointer. `len` never changes.
unsafe impl IoMemory for PinnedBuffer {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.pointer.as_ptr()
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// A byte range of a shared [`PinnedBuffer`]: a region `AsyncIo::read_into` can fill while other ranges of the same
/// buffer are read or copied. It keeps the buffer alive, so the memory cannot be freed under a read that has not ended.
///
/// Ranges of one buffer must not overlap while both are in use: the caller that cuts them (a window of slots) owns that.
#[cfg(any(target_os = "linux", target_os = "android", windows))]
#[derive(Debug)]
pub struct PinnedRange {
    buffer: Arc<PinnedBuffer>,
    offset: usize,
    len: usize,
}

#[cfg(any(target_os = "linux", target_os = "android", windows))]
impl PinnedRange {
    /// `len` bytes of `buffer` from `offset`; `None` when they pass the buffer's end.
    ///
    /// # Safety
    /// While this range is alive, no other range or reference touches its bytes: the caller cuts ranges of one buffer
    /// that do not overlap while both are in use.
    pub unsafe fn new(buffer: &Arc<PinnedBuffer>, offset: usize, len: usize) -> Option<PinnedRange> {
        let end = offset.checked_add(len)?;
        (end <= buffer.len()).then(|| PinnedRange {
            buffer: Arc::clone(buffer),
            offset,
            len,
        })
    }

    /// The start of the range.
    pub fn as_ptr(&self) -> *const u8 {
        // SAFETY: `new` checked that the range lies inside the buffer's allocation.
        unsafe { self.buffer.as_ptr().add(self.offset) }
    }
}

#[cfg(any(target_os = "linux", target_os = "android", windows))]
// SAFETY: the pointer addresses `len` bytes inside one live `cudaHostAlloc` allocation that this value keeps alive
// through its `Arc`, valid for reads and writes, and that stays where it is when the value moves (the allocation is
// outside Rust's allocator, reached by a raw pointer, so a move retags nothing). Every bit pattern is a valid byte.
// The region is exclusively the operation's from submission until the continuation gets the value back, provided the
// caller cut ranges that do not overlap while in use (the contract of `PinnedRange`); the value touches the region only
// through this pointer. `len` never changes.
unsafe impl IoMemory for PinnedRange {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.as_ptr().cast_mut()
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl Drop for PinnedBuffer {
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, freeing pinned memory waits for the device: pinned buffers are freed at shutdown"
    )]
    fn drop(&mut self) {
        // SAFETY: the pointer is this buffer's live allocation, freed once. A failure leaves nothing to recover.
        let _ = unsafe { self.library.api().euhedral_cuda_host_free(self.pointer.as_ptr().cast()) };
    }
}

/// Pinned, huge-page-backed host memory for weights staged to the device on use (`host_weights_malloc`): 2 MiB
/// aligned and mapped into the device's address space, so kernels read it in place. One arena per backing store; many
/// per-tensor allocations fall to 4 KiB pages. Freed on drop, which waits for the device (startup and shutdown only).
#[derive(Debug)]
pub struct HostWeightsArena {
    library: Arc<CudaLibrary>,
    pointer: NonNull<u8>,
    len: usize,
}

// SAFETY: as `PinnedBuffer`: exclusive ownership of a thread-independent allocation.
unsafe impl Send for HostWeightsArena {}
// SAFETY: as `PinnedBuffer`; the shared accessors do not write.
unsafe impl Sync for HostWeightsArena {}

impl HostWeightsArena {
    /// Allocates `bytes` (rounded up to 2 MiB by the allocator; [`HostWeightsArena::len`] stays `bytes`).
    pub fn alloc(library: &Arc<CudaLibrary>, bytes: usize) -> Result<HostWeightsArena, CudaError> {
        let size = non_empty(bytes)?;
        // SAFETY: the call takes a size.
        let address = unsafe { library.api().euhedral_cuda_host_weights_malloc(size) };
        let pointer = NonNull::new(address.cast::<u8>()).ok_or(CudaError::NullAllocation {
            call: "euhedral_cuda_host_weights_malloc",
            bytes: size,
        })?;
        Ok(HostWeightsArena {
            library: Arc::clone(library),
            pointer,
            len: bytes,
        })
    }

    pub fn as_ptr(&self) -> *const u8 {
        self.pointer.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.pointer.as_ptr()
    }

    /// The start of the arena as a pointer the device may write through. The arena's owner promises not to read or
    /// write the bytes a queued device copy writes until it has retired; this crate's copy operations are the only
    /// callers.
    pub(crate) fn device_writable_ptr(&self) -> *mut u8 {
        self.pointer.as_ptr()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is at least `len` bytes, live, and zero-filled by the allocator.
        unsafe { slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, and the borrow is exclusive.
        unsafe { slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len) }
    }

    /// The address at which kernels read this memory in place.
    ///
    /// # Safety
    /// Kernels read the arena through the address while the host may write it ([`HostWeightsArena::as_mut_slice`]).
    /// The caller must keep the arena alive until every such kernel has retired, and not write bytes a pending kernel
    /// reads.
    pub unsafe fn device_address(&self) -> Result<DevicePtr, CudaError> {
        device_address(&self.library, self.pointer.as_ptr().cast_const().cast())
    }
}

impl HostWeightsArena {
    /// The `len` bytes at `offset` as a region a read fills in place, or `None` when they do not lie inside the arena
    /// or are none.
    ///
    /// # Safety
    /// The region must be disjoint from every other [`ArenaRange`] of this arena that is alive, and from the host's
    /// own references to the arena ([`HostWeightsArena::as_slice`] is only called with no range being read), and the
    /// device must not read it until the read has ended.
    pub unsafe fn range(self: &Arc<Self>, offset: usize, len: usize) -> Option<ArenaRange> {
        let end = offset.checked_add(len)?;
        (len > 0 && end <= self.len).then(|| ArenaRange {
            arena: Arc::clone(self),
            offset,
            len,
        })
    }
}

/// A region of a [`HostWeightsArena`] that `AsyncIo::read_into` fills: the file's bytes land in the arena, where
/// kernels read them in place, with no staging copy in between. It keeps the arena alive.
#[derive(Debug)]
pub struct ArenaRange {
    arena: Arc<HostWeightsArena>,
    offset: usize,
    len: usize,
}

impl ArenaRange {
    /// The region's bytes. Only for a region no read is filling.
    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the region lies inside the arena's live allocation (checked by `range`), and the caller reads it
        // when no operation writes it.
        unsafe { slice::from_raw_parts(self.arena.pointer.as_ptr().add(self.offset), self.len) }
    }

    /// Where the region starts in the arena.
    pub fn offset(&self) -> usize {
        self.offset
    }
}

#[cfg(any(target_os = "linux", target_os = "android", windows))]
// SAFETY: the pointer addresses `len` bytes inside one live allocation (the arena, kept alive by the `Arc` the range
// holds, whose memory stays where it is when the range moves), valid for reads and writes and zero-filled by the
// allocator, so initialized. The region is exclusively the operation's: `HostWeightsArena::range` makes its caller
// promise that no other range or reference overlaps it. The range touches the bytes only through this pointer, and
// `len` never changes.
unsafe impl IoMemory for ArenaRange {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        // SAFETY: `offset` lies inside the allocation.
        unsafe { self.arena.pointer.as_ptr().add(self.offset) }
    }

    fn len(&self) -> usize {
        self.len
    }
}

impl Drop for HostWeightsArena {
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, freeing pinned memory waits for the device: the arena is freed at shutdown"
    )]
    fn drop(&mut self) {
        // SAFETY: the pointer is this arena's live allocation. A failed unregistration keeps the memory pinned and
        // the C side does not release it, so there is nothing more to do.
        let _ = unsafe {
            self.library
                .api()
                .euhedral_cuda_host_weights_free(self.pointer.as_ptr().cast())
        };
    }
}

/// The device address of registered host memory. Taking the address is harmless; the callers' `unsafe` is about what
/// the device does through it.
fn device_address(library: &CudaLibrary, host: *const c_void) -> Result<DevicePtr, CudaError> {
    let mut device = 0u64;
    // SAFETY: `host` is the start of a live registered allocation and `device` is a live local.
    let status = unsafe {
        library
            .api()
            .euhedral_cuda_host_weights_device_pointer(host, &mut device)
    };
    check("euhedral_cuda_host_weights_device_pointer", status)?;
    Ok(DevicePtr(device))
}

#[cfg(test)]
mod tests {
    use super::LiveBytes;

    #[test]
    fn an_allocation_raises_the_count_by_its_size_and_its_free_lowers_it() {
        let live = LiveBytes::new();
        live.allocated(4096);
        live.allocated(1 << 20);
        assert_eq!(live.get(), 4096 + (1 << 20));
        live.freed(4096);
        assert_eq!(live.get(), 1 << 20);
        live.freed(1 << 20);
        assert_eq!(live.get(), 0);
    }
}
