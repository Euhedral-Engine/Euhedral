//! Host memory in page-aligned slots, and the leases on them.
//!
//! A lease is the right to touch a slot: one writer, or any number of readers. A lease that conflicts is refused at
//! once (nothing waits), so a slot a load is filling cannot be read, nor one being read be overwritten. A write lease
//! moves with the work: the read of a record cuts it into [`WritePart`]s, each the memory of one asynchronous read, and
//! the lease ends when the last part is dropped. A part the I/O engine leaks (a ring that broke) keeps the lease for
//! good, which keeps the slot out of use: the kernel may still write it.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::ptr::NonNull;
use std::slice::{from_raw_parts, from_raw_parts_mut};
use std::sync::Arc;
use std::sync::atomic::{AtomicI32, Ordering};

use euhedral_inference_cuda::PinnedBuffer;
use euhedral_io::IoMemory;

use super::layout::ExpertError;
use crate::residency::offload::HOST_ALIGNMENT;

/// The lease state of a slot while a writer holds it; a positive state counts the readers.
const WRITING: i32 = -1;

/// `slots` slots of `slot_bytes` each, in one page-aligned block of ordinary memory, zeroed.
#[derive(Debug)]
pub struct HostArena {
    base: NonNull<u8>,
    layout: Layout,
    slots: usize,
    slot_bytes: usize,
    leases: Box<[AtomicI32]>,
    /// The page-locked buffer the block is, when it is one: the block is then not this arena's to free.
    pinned: Option<PinnedBuffer>,
}

// SAFETY: the block is reached only through leases, which admit one writer or many readers per slot; the pointer is
// never handed out otherwise.
unsafe impl Send for HostArena {}
// SAFETY: as above.
unsafe impl Sync for HostArena {}

impl HostArena {
    /// An arena of `slots` slots, each `slot_bytes` (a whole number of pages). Refuses a slot that is not, no slot, and a
    /// size that overflows or cannot be had, naming the values.
    pub fn new(slots: usize, slot_bytes: usize) -> Result<Arc<HostArena>, ExpertError> {
        if slot_bytes == 0 || !(slot_bytes as u64).is_multiple_of(HOST_ALIGNMENT) {
            return Err(ExpertError::SlotNotPaged {
                bytes: slot_bytes,
                block: HOST_ALIGNMENT,
            });
        }
        let size = slots
            .checked_mul(slot_bytes)
            .filter(|&size| size > 0 && size <= isize::MAX as usize)
            .ok_or(ExpertError::ArenaSize {
                slots,
                bytes: slot_bytes,
            })?;
        let layout = Layout::from_size_align(size, HOST_ALIGNMENT as usize).map_err(|_| ExpertError::ArenaSize {
            slots,
            bytes: slot_bytes,
        })?;
        // SAFETY: the layout has a non-zero size.
        let base = NonNull::new(unsafe { alloc_zeroed(layout) }).ok_or(ExpertError::ArenaSize {
            slots,
            bytes: slot_bytes,
        })?;
        Ok(Arc::new(HostArena {
            base,
            layout,
            slots,
            slot_bytes,
            leases: (0..slots).map(|_| AtomicI32::new(0)).collect(),
            pinned: None,
        }))
    }

    /// An arena of `slots` slots of `slot_bytes` over the page-locked `buffer`, which a device copy can read from
    /// directly. Refuses a slot that is not a whole number of pages, a buffer that is too small or not page-aligned,
    /// naming the values.
    pub fn over(slots: usize, slot_bytes: usize, mut buffer: PinnedBuffer) -> Result<Arc<HostArena>, ExpertError> {
        if slot_bytes == 0 || !(slot_bytes as u64).is_multiple_of(HOST_ALIGNMENT) {
            return Err(ExpertError::SlotNotPaged {
                bytes: slot_bytes,
                block: HOST_ALIGNMENT,
            });
        }
        let size = slots
            .checked_mul(slot_bytes)
            .filter(|&size| size > 0 && size <= buffer.len())
            .ok_or(ExpertError::ArenaSize {
                slots,
                bytes: slot_bytes,
            })?;
        let layout = Layout::from_size_align(size, HOST_ALIGNMENT as usize).map_err(|_| ExpertError::ArenaSize {
            slots,
            bytes: slot_bytes,
        })?;
        let address = buffer.as_mut_ptr();
        if !(address as usize as u64).is_multiple_of(HOST_ALIGNMENT) {
            return Err(ExpertError::Unaligned {
                address: address as usize,
                block: HOST_ALIGNMENT,
            });
        }
        let base = NonNull::new(address).ok_or(ExpertError::ArenaSize {
            slots,
            bytes: slot_bytes,
        })?;
        Ok(Arc::new(HostArena {
            base,
            layout,
            slots,
            slot_bytes,
            leases: (0..slots).map(|_| AtomicI32::new(0)).collect(),
            pinned: Some(buffer),
        }))
    }

    pub fn slots(&self) -> usize {
        self.slots
    }

    pub fn slot_bytes(&self) -> usize {
        self.slot_bytes
    }

    /// The write lease of `slot`, or why it is refused (out of range, or leased already).
    pub fn write(self: &Arc<Self>, slot: usize) -> Result<SlotWrite, ExpertError> {
        self.check(slot)?;
        self.leases[slot]
            .compare_exchange(0, WRITING, Ordering::Acquire, Ordering::Relaxed)
            .map_err(|_| ExpertError::SlotBusy {
                slot,
                wanted: "writing",
            })?;
        Ok(SlotWrite {
            guard: Arc::new(WriteGuard {
                arena: Arc::clone(self),
                slot,
            }),
            cut: 0,
        })
    }

    /// A read lease of `slot`, or why it is refused (out of range, or being written).
    pub fn read(self: &Arc<Self>, slot: usize) -> Result<SlotRead, ExpertError> {
        self.check(slot)?;
        self.leases[slot]
            .try_update(Ordering::Acquire, Ordering::Relaxed, |state| {
                (state >= 0).then(|| state + 1)
            })
            .map_err(|_| ExpertError::SlotBusy {
                slot,
                wanted: "reading",
            })?;
        Ok(SlotRead {
            arena: Arc::clone(self),
            slot,
        })
    }

    /// Whether anyone holds a lease on `slot`.
    pub fn is_leased(&self, slot: usize) -> bool {
        self.leases
            .get(slot)
            .is_some_and(|state| state.load(Ordering::Acquire) != 0)
    }

    fn check(&self, slot: usize) -> Result<(), ExpertError> {
        if slot >= self.slots {
            return Err(ExpertError::SlotOutOfRange {
                slot,
                slots: self.slots,
            });
        }
        Ok(())
    }

    fn pointer(&self, slot: usize, from: usize) -> *mut u8 {
        // SAFETY: `slot` is below `slots` and `from` is within the slot, so the offset is inside the block.
        unsafe { self.base.as_ptr().add(slot * self.slot_bytes + from) }
    }
}

impl Drop for HostArena {
    fn drop(&mut self) {
        if self.pinned.is_none() {
            // SAFETY: the block was allocated with this layout; nothing leases it, since every lease holds the arena.
            unsafe { dealloc(self.base.as_ptr(), self.layout) };
        }
    }
}

/// The write lease of one slot; the lease ends when the last holder drops.
#[derive(Debug)]
struct WriteGuard {
    arena: Arc<HostArena>,
    slot: usize,
}

impl Drop for WriteGuard {
    fn drop(&mut self) {
        self.arena.leases[self.slot].store(0, Ordering::Release);
    }
}

/// The write lease of a slot: its bytes, or parts of it cut out for reads.
#[derive(Debug)]
pub struct SlotWrite {
    guard: Arc<WriteGuard>,
    /// Where the next part starts: parts are cut in order and never overlap.
    cut: usize,
}

impl SlotWrite {
    pub fn slot(&self) -> usize {
        self.guard.slot
    }

    /// The slot's bytes.
    pub fn capacity(&self) -> usize {
        self.guard.arena.slot_bytes
    }

    /// The next `len` bytes of the slot as memory for one read, or `None` when fewer remain. A part holds the lease.
    pub fn cut(&mut self, len: usize) -> Option<WritePart> {
        let end = self.cut.checked_add(len).filter(|&end| end <= self.capacity())?;
        let part = WritePart {
            guard: Arc::clone(&self.guard),
            from: self.cut,
            len,
        };
        self.cut = end;
        Some(part)
    }

    /// The slot's bytes, while no part is out (`None` after a part was leaked or is still held).
    pub fn bytes_mut(&mut self) -> Option<&mut [u8]> {
        let guard = Arc::get_mut(&mut self.guard)?;
        // SAFETY: the lease is exclusive and no part exists (`get_mut` proved it), so nothing else touches the slot.
        Some(unsafe { from_raw_parts_mut(guard.arena.pointer(guard.slot, 0), guard.arena.slot_bytes) })
    }
}

/// A range of a leased slot: the memory of one read. Holds the write lease.
#[derive(Debug)]
pub struct WritePart {
    guard: Arc<WriteGuard>,
    from: usize,
    len: usize,
}

impl WritePart {
    /// The bytes written so far.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: the lease is exclusive and the parts are disjoint, so this range has this part as its only user.
        unsafe { from_raw_parts(self.guard.arena.pointer(self.guard.slot, self.from), self.len) }
    }

    /// The bytes to fill.
    pub fn bytes_mut(&mut self) -> &mut [u8] {
        // SAFETY: as in `bytes`, and `&mut self` makes this the only reference.
        unsafe { from_raw_parts_mut(self.guard.arena.pointer(self.guard.slot, self.from), self.len) }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Keeps the first `len` bytes of the part; a longer `len` changes nothing.
    pub fn shorten(&mut self, len: usize) {
        self.len = self.len.min(len);
    }
}

// SAFETY: the pointer is the part's range of a block that lives as long as the lease the part holds; the range is
// exclusive (a write lease, parts cut disjoint); it is initialized (the arena is zeroed and bytes are plain); `len` is
// fixed except through `shorten`, which a read never sees (the memory is moved into the read).
unsafe impl IoMemory for WritePart {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        self.guard.arena.pointer(self.guard.slot, self.from)
    }

    fn len(&self) -> usize {
        self.len
    }
}

/// A read lease of one slot, until dropped.
#[derive(Debug)]
pub struct SlotRead {
    arena: Arc<HostArena>,
    slot: usize,
}

impl SlotRead {
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// The page-locked buffer the slot is in and the slot's offset in it, for a device copy; `None` when the arena is
    /// ordinary memory.
    pub fn pinned(&self) -> Option<(&PinnedBuffer, usize)> {
        self.arena
            .pinned
            .as_ref()
            .map(|buffer| (buffer, self.slot * self.arena.slot_bytes))
    }

    /// The slot's bytes.
    pub fn bytes(&self) -> &[u8] {
        // SAFETY: a read lease excludes writers for as long as it lives, and the arena is alive through it.
        unsafe { from_raw_parts(self.arena.pointer(self.slot, 0), self.arena.slot_bytes) }
    }
}

impl Drop for SlotRead {
    fn drop(&mut self) {
        self.arena.leases[self.slot].fetch_sub(1, Ordering::Release);
    }
}
