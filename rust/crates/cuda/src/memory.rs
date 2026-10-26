//! Device, pinned and host-weight memory. Each owner frees its allocation on drop.

use std::ffi::c_void;
use std::ptr::{self, NonNull};
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use euhedral_graph::Marker;

use crate::error::{CudaError, check};
use crate::library::CudaLibrary;
use crate::submission::{Submission, capture_selected};

/// A device address, opaque to the host. Zero is null.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct DevicePtr(pub u64);

impl DevicePtr {
    pub const NULL: DevicePtr = DevicePtr(0);

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

/// A device allocation, freed on drop.
///
/// Dropping a buffer frees it at once with `cudaFree`, which waits for the device (and the memory may still be in use
/// by queued work only if the caller dropped it too early: nothing here tracks its readers). Free long-lived buffers
/// at shutdown. To free without waiting, retire the buffer with the marker of its last use
/// ([`RetireList`](crate::RetireList)), or free it in stream order with [`DeviceBuffer::free_async`].
#[derive(Debug)]
pub struct DeviceBuffer {
    library: Arc<CudaLibrary>,
    addr: u64,
    bytes: usize,
    generation: AllocationId,
}

impl DeviceBuffer {
    /// Allocates `bytes` of device memory.
    pub fn alloc(library: &Arc<CudaLibrary>, bytes: usize) -> Result<DeviceBuffer, CudaError> {
        let size = non_empty(bytes)?;
        // SAFETY: the call takes a size.
        let address = unsafe { library.api().euhedral_cuda_malloc(size) };
        Self::wrap(library, "euhedral_cuda_malloc", address, bytes)
    }

    /// Allocates `bytes` in stream order on the submission's lane: the memory is usable by work queued after this
    /// call on that lane, and the host never waits. Not allowed while the lane is captured into a graph.
    pub fn alloc_async(submission: &Submission<'_>, bytes: usize) -> Result<DeviceBuffer, CudaError> {
        if submission.lane().is_capturing() {
            return Err(CudaError::Capturing("a stream-ordered allocation"));
        }
        let size = non_empty(bytes)?;
        let library = submission.lane().library();
        // SAFETY: the submission has selected its lane's stream, which the call allocates on.
        let address = unsafe { library.api().euhedral_cuda_malloc_async(size) };
        Self::wrap(library, "euhedral_cuda_malloc_async", address, bytes)
    }

    fn wrap(
        library: &Arc<CudaLibrary>,
        call: &'static str,
        address: *mut c_void,
        bytes: usize,
    ) -> Result<DeviceBuffer, CudaError> {
        if address.is_null() {
            return Err(CudaError::NullAllocation {
                call,
                bytes: bytes as u64,
            });
        }
        Ok(DeviceBuffer {
            library: Arc::clone(library),
            addr: address.expose_provenance() as u64,
            bytes,
            generation: library.next_allocation(),
        })
    }

    /// Frees the buffer in stream order on the submission's lane: the memory returns to the pool once the lane
    /// reaches this point, after every reader queued before it on that lane. `readers` are the markers of the last
    /// uses on other lanes; the free waits for each on the device first, so a reader on another lane never sees
    /// reused memory. Pass every lane that still reads the buffer.
    pub fn free_async(mut self, submission: &Submission<'_>, readers: &[Marker]) -> Result<(), CudaError> {
        if submission.lane().is_capturing() {
            return Err(CudaError::Capturing("a stream-ordered free"));
        }
        for reader in readers {
            submission.wait(*reader)?;
        }
        // SAFETY: the address is this buffer's live allocation, and the submission has selected a stream.
        let status = unsafe { self.library.api().euhedral_cuda_free_async(self.ptr().as_mut()) };
        // The allocation is gone either way: a failed stream-ordered free must not become a second free on drop.
        self.addr = 0;
        check("euhedral_cuda_free_async", status)
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

    /// The allocation's serial.
    pub fn generation(&self) -> AllocationId {
        self.generation
    }

    /// Copies `host` into the buffer at `offset` and returns when the copy is done. Startup, shutdown and tests: it
    /// blocks, and outside a submission it runs on the legacy stream, which does not order against lane work.
    pub fn copy_from_host(&self, offset: usize, host: &[u8]) -> Result<(), CudaError> {
        if capture_selected() {
            return Err(CudaError::Capturing("a blocking host copy"));
        }
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
    pub fn copy_to_host(&self, offset: usize, host: &mut [u8]) -> Result<(), CudaError> {
        if capture_selected() {
            return Err(CudaError::Capturing("a blocking host copy"));
        }
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
    fn drop(&mut self) {
        if self.addr == 0 {
            return;
        }
        // SAFETY: the address is this buffer's live allocation, freed once. A failure leaves nothing to recover.
        let _ = unsafe { self.library.api().euhedral_cuda_free(self.ptr().as_mut()) };
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
/// It is owned exclusively: the pointer and length are fixed for the buffer's life. Freeing pinned memory waits for
/// the device, so long-lived buffers are freed at shutdown.
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
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }

    /// The bytes, for writing. Write memory a queued upload reads only after the lane's work has retired.
    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, and the borrow is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len) }
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

impl Drop for PinnedBuffer {
    fn drop(&mut self) {
        // SAFETY: the pointer is this buffer's live allocation, freed once. A failure leaves nothing to recover.
        let _ = unsafe { self.library.api().euhedral_cuda_host_free(self.pointer.as_ptr().cast()) };
    }
}

/// Pinned, huge-page-backed host memory for weights staged to the device on use (`host_weights_malloc`): 2 MiB
/// aligned and mapped into the device's address space, so kernels read it in place. One arena per backing store; many
/// per-tensor allocations fall to 4 KiB pages. Freed on drop.
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

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: the allocation is at least `len` bytes, live, and zero-filled by the allocator.
        unsafe { std::slice::from_raw_parts(self.pointer.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as `as_slice`, and the borrow is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.pointer.as_ptr(), self.len) }
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

impl Drop for HostWeightsArena {
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
