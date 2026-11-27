//! The disk window and the owner that reads disk-backed weights into it.
//!
//! A weight that lives only in the artifact file is read, whole aligned blocks at a time, into a slot of the pinned
//! *window*; a copy then moves it from the slot to a staging slot on the device. The [`DiskReader`] is the owner that
//! decides which read goes into which slot and when. It is driven by messages, never by a poll:
//!
//! - a quantum that starts posts its uses ([`DiskReader::enqueue`]), in the order its device work consumes them;
//! - a read's completion posts [`DiskMessage::Arrived`] (from the frame an `AsyncIo` continuation returns);
//! - a copy's marker completion posts [`DiskMessage::SlotFreed`] with its quantum's ticket, which frees the slot for the
//!   next read (a free that names another ticket than the one holding the slot is ignored).
//!
//! No device work writes a host flag and nothing waits for a read. The device never waits for the host either: a copy
//! stage holds a [`Gate`] (taken in `pass_gate`, through [`DiskReader::claim`]) until its read is in the slot, so the
//! stage is published, and its copy queued, only once the bytes are there. A copy that finds its read in claims it at
//! once ([`Claimed::Ready`]) and takes no gate.
//!
//! **Slots.** Each slot is read one use at a time: its uses wait in a queue, in admission order and, within a quantum, in
//! use order, which is the order the copies of that slot run. A slot is `Idle`, `Reading`, `Filled` (read, or failed, no copy
//! claimed it yet) or `Claimed` (its copy owns the bytes until [`DiskMessage::SlotFreed`] of the same ticket). A read is never started
//! into a slot before the previous read of it has arrived, whoever it belonged to: the kernel may still be writing.
//!
//! **Abandoned quanta.** [`DiskReader::abandon`] ends a quantum's uses: those not read are dropped, a read in flight is
//! discarded when it arrives, a read that arrived and no copy claimed is dropped at once, and a copy that waits for it
//! has its gate failed (so a released gate always means the copy owns the slot). A slot a copy already claimed stays
//! with that copy until it frees it. A quantum abandoned before its uses were posted reads nothing. A claim that comes
//! after its ticket was abandoned (the stage was published between the quantum's end and its `pass_gate`) finds no use
//! of that ticket on the slot and fails its gate at once; the slot is left alone for the quanta that follow.
//!
//! **Failures.** A failed (or short) read fails its quantum through the [`FailHandle`] first and then still hands the
//! slot to its copy, so the copy of stale bytes belongs to a quantum that cannot commit. A direct read the file
//! system refuses (`EINVAL`, or error 87 on Windows) is repeated once through the page cache.
//!
//! **Gates.** A gate is spent exactly once: released when its read is in, failed when its quantum is abandoned or the
//! reader closes. Gates are spent after the owner's claim is released, in the order the owner decided.
//!
//! **Close.** [`DiskReader::close`] stops new reads, fails the gates still waiting and returns a hand-off that is sent
//! when no read is in flight, once for each close. The hand-off waits for reads, not for copies that claimed a slot, so
//! the window is safe to free only when every quantum has retired as well. The caller keeps the [`DiskReader`] alive
//! until the hand-off arrives: the reads reach the owner through a weak address, and a dropped reader loses them.

#[cfg(test)]
use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::{BTreeSet, VecDeque};
use std::fmt::{self, Formatter};
use std::fs::{File, OpenOptions};
use std::io;
use std::mem;
use std::path::Path;
#[cfg(test)]
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use euhedral_data_structures::queues::{Consumer, MpscQueue, Single};
use euhedral_graph::{Gate, StageError};
use euhedral_inference_cuda::{PinnedBuffer, PinnedRange};
use euhedral_inference_settings::{Auto, ByteSize};
use euhedral_io::{AsyncIo, IoDoneOwned, IoMemory, MAX_PIECE, OwnedFile, ReadOptions};

use crate::frames::{Kind, step};
use crate::generation::{Address, Mailbox, Spawn, Turn};
use crate::handoff::Handoff;
use crate::residency::offload::{HOST_ALIGNMENT, WindowGeometry};
use crate::runtime::raw_file;

/// What the owner tells the quantum whose read failed (the text says which object and why). It must not block: it runs
/// where the read's arrival is handled, and the gate of the quantum's copy is released after it.
pub type FailHandle = Arc<dyn Fn(String) + Send + Sync>;

/// Why a disk window or a use of it is refused.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DiskError {
    #[error("the disk window's memory is {have} bytes and its {slots} slots of {slot_bytes} bytes need {need}")]
    Memory {
        have: usize,
        need: u64,
        slots: usize,
        slot_bytes: u64,
    },
    #[error("a window slot is {0} bytes: not a multiple of {HOST_ALIGNMENT}")]
    SlotSize(u64),
    #[error("the disk window's memory does not start on a {HOST_ALIGNMENT} byte boundary")]
    Unaligned,
    #[error("the disk use of '{name}' names slot {slot} of {slots}")]
    NoSuchSlot { name: String, slot: usize, slots: usize },
    #[error("the disk use of '{name}' reads {bytes} bytes into a slot of {slot_bytes}")]
    TooLarge { name: String, bytes: u64, slot_bytes: u64 },
    #[error("the disk use of '{name}' has no bytes or ends past the largest file offset")]
    Empty { name: String },
}

/// One use of a disk-backed weight in a view: its payload in the artifact (`bytes` at `file_offset`) and the window
/// slot it is read into. The read covers the whole aligned blocks around the payload, which starts [`lead`](Self::lead)
/// bytes into the slot.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DiskUse {
    pub name: String,
    pub file_offset: u64,
    pub bytes: u64,
    pub slot: usize,
}

impl DiskUse {
    /// The file offset the read starts at: the payload's, rounded down to a block.
    pub fn aligned_offset(&self) -> u64 {
        self.file_offset & !(HOST_ALIGNMENT - 1)
    }

    /// Where the payload starts within the slot.
    pub fn lead(&self) -> u64 {
        self.file_offset - self.aligned_offset()
    }

    /// The bytes of the aligned read: whole blocks from [`aligned_offset`](Self::aligned_offset) past the payload's end.
    pub fn read_bytes(&self) -> u64 {
        (self.lead() + self.bytes).next_multiple_of(HOST_ALIGNMENT)
    }
}

/// An aligned allocation on the heap that stands in for pinned memory in tests: a window needs no device to be read into.
#[cfg(test)]
#[derive(Debug)]
pub struct HeapRegion {
    pointer: NonNull<u8>,
    len: usize,
}

#[cfg(test)]
impl HeapRegion {
    fn new(len: usize) -> HeapRegion {
        let layout = Layout::from_size_align(len, HOST_ALIGNMENT as usize).expect("a layout");
        // SAFETY: the layout has a non-zero size (the tests allocate at least a block).
        let pointer = unsafe { alloc_zeroed(layout) };
        HeapRegion {
            pointer: NonNull::new(pointer).expect("memory"),
            len,
        }
    }
}

#[cfg(test)]
impl Drop for HeapRegion {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len, HOST_ALIGNMENT as usize).expect("a layout");
        // SAFETY: allocated in `new` with this layout and freed once.
        unsafe { dealloc(self.pointer.as_ptr(), layout) };
    }
}

// SAFETY: the region is plain memory the owner of the window hands out in disjoint slots.
#[cfg(test)]
unsafe impl Send for HeapRegion {}
// SAFETY: as above.
#[cfg(test)]
unsafe impl Sync for HeapRegion {}

/// The memory behind a window.
enum Backing {
    Pinned(Arc<PinnedBuffer>),
    #[cfg(test)]
    Heap(Arc<HeapRegion>),
}

/// A range of a window's memory a read fills: `IoMemory` over a slot (or its first bytes).
#[derive(Debug)]
pub enum SlotMemory {
    Pinned(PinnedRange),
    #[cfg(test)]
    Heap {
        region: Arc<HeapRegion>,
        offset: usize,
        len: usize,
    },
}

// SAFETY: a `PinnedRange` is `IoMemory` (see its impl), and the heap variant is the same claim over a live heap
// allocation its `Arc` keeps alive: `len` bytes valid for reads and writes at a stable address, touched only through
// the pointer. The window cuts one range per slot and the reader keeps one read per slot in flight, so the region is
// exclusively the operation's.
unsafe impl IoMemory for SlotMemory {
    fn as_mut_ptr(&mut self) -> *mut u8 {
        match self {
            SlotMemory::Pinned(range) => IoMemory::as_mut_ptr(range),
            #[cfg(test)]
            // SAFETY: `offset` lies inside the region (`slot_memory` checked it).
            SlotMemory::Heap { region, offset, .. } => unsafe { region.pointer.as_ptr().add(*offset) },
        }
    }

    fn len(&self) -> usize {
        match self {
            SlotMemory::Pinned(range) => IoMemory::len(range),
            #[cfg(test)]
            SlotMemory::Heap { len, .. } => *len,
        }
    }
}

/// The pinned host memory the artifact is read into: `slots` slots of `slot_bytes`, each a multiple of 4096 bytes on a
/// 4096 byte boundary, so that a direct read can land in any of them.
pub struct DiskWindow {
    backing: Backing,
    geometry: WindowGeometry,
}

impl fmt::Debug for DiskWindow {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiskWindow")
            .field("geometry", &self.geometry)
            .finish()
    }
}

impl DiskWindow {
    /// A window over `buffer`, laid out as `geometry` says.
    pub fn new(buffer: Arc<PinnedBuffer>, geometry: WindowGeometry) -> Result<DiskWindow, DiskError> {
        Self::check(buffer.as_ptr().addr(), buffer.len(), geometry)?;
        Ok(DiskWindow {
            backing: Backing::Pinned(buffer),
            geometry,
        })
    }

    fn check(address: usize, have: usize, geometry: WindowGeometry) -> Result<(), DiskError> {
        if !geometry.slot_bytes.is_multiple_of(HOST_ALIGNMENT) {
            return Err(DiskError::SlotSize(geometry.slot_bytes));
        }
        if !address.is_multiple_of(HOST_ALIGNMENT as usize) {
            return Err(DiskError::Unaligned);
        }
        let need = geometry.bytes();
        if (have as u64) < need {
            return Err(DiskError::Memory {
                have,
                need,
                slots: geometry.slots,
                slot_bytes: geometry.slot_bytes,
            });
        }
        Ok(())
    }

    /// A window over heap memory, for tests.
    #[cfg(test)]
    fn on_heap(geometry: WindowGeometry) -> DiskWindow {
        let region = Arc::new(HeapRegion::new(geometry.bytes() as usize));
        DiskWindow {
            backing: Backing::Heap(region),
            geometry,
        }
    }

    pub fn geometry(&self) -> WindowGeometry {
        self.geometry
    }

    /// The start of slot `slot`: where the copy that follows its read reads from (plus the use's lead).
    pub fn slot_ptr(&self, slot: usize) -> *const u8 {
        assert!(slot < self.geometry.slots, "slot {slot} of {}", self.geometry.slots);
        let offset = slot * self.geometry.slot_bytes as usize;
        match &self.backing {
            // SAFETY: `check` made sure the slots lie inside the buffer.
            Backing::Pinned(buffer) => unsafe { buffer.as_ptr().add(offset) },
            #[cfg(test)]
            // SAFETY: as above, for the heap region.
            Backing::Heap(region) => unsafe { region.pointer.as_ptr().add(offset) },
        }
    }

    /// The first `len` bytes of slot `slot`, as the memory a read fills.
    fn slot_memory(&self, slot: usize, len: usize) -> Option<SlotMemory> {
        assert!(slot < self.geometry.slots && len as u64 <= self.geometry.slot_bytes);
        let offset = slot * self.geometry.slot_bytes as usize;
        match &self.backing {
            Backing::Pinned(buffer) => {
                // SAFETY: the slots are disjoint ranges of the buffer, and the reader reads one use of a slot at a
                // time, so no two ranges are in use over the same bytes.
                unsafe { PinnedRange::new(buffer, offset, len) }.map(SlotMemory::Pinned)
            }
            #[cfg(test)]
            Backing::Heap(region) => Some(SlotMemory::Heap {
                region: Arc::clone(region),
                offset,
                len,
            }),
        }
    }
}

/// One read the owner asks of its [`ReadPort`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ReadSpec {
    /// The window slot the bytes land at the start of.
    pub slot: usize,
    /// The file offset to read from.
    pub offset: u64,
    /// The bytes to read.
    pub len: usize,
    /// Through the handle that bypasses the page cache (the offset, length and slot are blocks then).
    pub direct: bool,
}

/// Called once, from wherever a read ended, with the bytes moved or the error.
pub type ReadDone = Box<dyn FnOnce(io::Result<usize>) + Send>;

/// Where the reader's reads go: the window's file and the ring that reads it. A fake stands in for it in tests.
pub trait ReadPort: Send + Sync + 'static {
    /// Starts `spec`; `done` is called once when the read ended (maybe before `start` returns). An `Err` started nothing
    /// and never calls `done`.
    fn start(&self, spec: ReadSpec, done: ReadDone) -> io::Result<()>;

    /// Whether a direct read can be asked for.
    fn has_direct(&self) -> bool;

    /// The file's length in bytes: a direct read that would pass it is made buffered.
    fn file_len(&self) -> u64;
}

/// The artifact, open twice: through the page cache, and (where the platform and the file system allow) past it.
#[derive(Debug)]
pub struct ArtifactFile {
    buffered: OwnedFile,
    direct: Option<OwnedFile>,
    len: u64,
}

impl ArtifactFile {
    /// Opens `path` for the reads of the window. Blocking: call it at startup, outside any frame.
    pub fn open(path: &Path) -> io::Result<ArtifactFile> {
        Self::open_checked(path, open_direct, |_| Ok(()))
    }

    /// As [`open`](Self::open), with each handle it opens (the buffered one and the direct one) shown to `admit` first:
    /// an `Err` from it fails the open. The engine passes the check that the handle is the file it holds.
    pub fn open_admitted(path: &Path, admit: impl Fn(&File) -> io::Result<()>) -> io::Result<ArtifactFile> {
        Self::open_checked(path, open_direct, admit)
    }

    /// As [`open_admitted`](Self::open_admitted) with `direct` as the page-cache-bypassing open (tests force it to
    /// fail). A file system that refuses the direct open (tmpfs does) leaves every read buffered.
    #[allow(
        clippy::disallowed_methods,
        reason = "no-wait, blocking I/O: one fstat at startup, before any frame runs"
    )]
    fn open_checked(
        path: &Path,
        direct: impl FnOnce(&Path) -> io::Result<File>,
        admit: impl Fn(&File) -> io::Result<()>,
    ) -> io::Result<ArtifactFile> {
        let buffered = open_buffered(path)?;
        admit(&buffered)?;
        let len = buffered.metadata()?.len();
        let direct = direct(path)
            .map_err(|refused| log::info!("the artifact is read through the page cache: {refused}"))
            .ok();
        if let Some(direct) = &direct {
            admit(direct)?;
        }
        Ok(ArtifactFile {
            buffered: buffered.into(),
            direct: direct.map(OwnedFile::from),
            len,
        })
    }

    /// The handle reads go through: the one that bypasses the page cache when `direct` and there is one.
    pub(crate) fn handle(&self, direct: bool) -> &OwnedFile {
        match &self.direct {
            Some(direct_handle) if direct => direct_handle,
            _ => &self.buffered,
        }
    }

    /// Whether reads can bypass the page cache.
    pub fn has_direct(&self) -> bool {
        self.direct.is_some()
    }

    pub fn len(&self) -> u64 {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: opening a file is one short call at startup, on the calling thread before any frame runs; the reads go through AsyncIo"
)]
/// The artifact opened for reads through the page cache. On Windows every handle `AsyncIo` reads is overlapped.
fn open_buffered(path: &Path) -> io::Result<File> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
        options.custom_flags(FILE_FLAG_OVERLAPPED);
    }
    options.open(path)
}

/// `O_DIRECT` on the architectures whose value is known (it differs: ARM's is `0o200000`, x86's `0o40000`).
#[cfg(all(
    any(target_os = "linux", target_os = "android"),
    any(target_arch = "x86", target_arch = "x86_64")
))]
const O_DIRECT: Option<i32> = Some(0o40000);
#[cfg(all(
    any(target_os = "linux", target_os = "android"),
    any(target_arch = "arm", target_arch = "aarch64")
))]
const O_DIRECT: Option<i32> = Some(0o200000);
#[cfg(all(
    any(target_os = "linux", target_os = "android"),
    not(any(
        target_arch = "x86",
        target_arch = "x86_64",
        target_arch = "arm",
        target_arch = "aarch64"
    ))
))]
const O_DIRECT: Option<i32> = None;

/// The artifact opened to bypass the page cache: `O_DIRECT` on Linux, `FILE_FLAG_NO_BUFFERING` (overlapped) on Windows.
/// Refused elsewhere. The caller falls back to the buffered handle when this fails.
#[cfg(any(target_os = "linux", target_os = "android"))]
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: opening a file is one short call at startup, on the calling thread before any frame runs; the reads go through AsyncIo"
)]
fn open_direct(path: &Path) -> io::Result<File> {
    use std::os::unix::fs::OpenOptionsExt;

    let flag =
        O_DIRECT.ok_or_else(|| io::Error::new(io::ErrorKind::Unsupported, "no direct reads on this architecture"))?;
    OpenOptions::new().read(true).custom_flags(flag).open(path)
}

#[cfg(windows)]
#[allow(
    clippy::disallowed_methods,
    reason = "no-wait, blocking I/O: opening a file is one short call at startup, on the calling thread before any frame runs; the reads go through AsyncIo"
)]
fn open_direct(path: &Path) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;

    const FILE_FLAG_NO_BUFFERING: u32 = 0x2000_0000;
    const FILE_FLAG_OVERLAPPED: u32 = 0x4000_0000;
    OpenOptions::new()
        .read(true)
        .custom_flags(FILE_FLAG_NO_BUFFERING | FILE_FLAG_OVERLAPPED)
        .open(path)
}

#[cfg(not(any(target_os = "linux", target_os = "android", windows)))]
fn open_direct(_path: &Path) -> io::Result<File> {
    Err(io::Error::new(
        io::ErrorKind::Unsupported,
        "no direct reads on this platform",
    ))
}

/// The reads of the real window: `AsyncIo` reads the artifact straight into the slots.
impl fmt::Debug for FilePort {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("FilePort")
            .field("window", &self.window)
            .finish_non_exhaustive()
    }
}

pub struct FilePort {
    io: Arc<AsyncIo>,
    file: Arc<ArtifactFile>,
    window: DiskWindow,
    /// The options of every read: the ring is the core's, shared, so the piece is set per read and not on the ring.
    options: ReadOptions,
}

/// The most bytes one piece of a read of the artifact moves (`memory.disk-read-piece`): from one block to what the ring
/// allows, a whole number of blocks, so that a direct read's every piece is aligned. A value of this type is valid.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DiskReadPiece(usize);

impl DiskReadPiece {
    /// The piece the setting `asked` names; `auto` is the most the ring allows, which reads each slot whole.
    /// Refuses a size that is not a whole number of blocks, or above the ring's limit, naming the value and the limit.
    pub fn from_setting(asked: Auto<ByteSize>) -> Result<DiskReadPiece, String> {
        let Some(asked) = asked.into_value() else {
            return Ok(DiskReadPiece(MAX_PIECE));
        };
        let bytes = asked.bytes();
        let block = HOST_ALIGNMENT;
        if bytes == 0 || bytes > MAX_PIECE as u64 || !bytes.is_multiple_of(block) {
            return Err(format!(
                "memory.disk-read-piece is {bytes} bytes: it must be a multiple of {block} bytes, from {block} to {MAX_PIECE}"
            ));
        }
        Ok(DiskReadPiece(bytes as usize))
    }

    /// The bytes of one piece.
    pub fn bytes(self) -> usize {
        self.0
    }

    /// The options of a read in pieces of this size.
    pub(crate) fn options(self) -> ReadOptions {
        ReadOptions::with_piece(self.0)
    }
}

impl FilePort {
    pub fn new(io: Arc<AsyncIo>, file: Arc<ArtifactFile>, window: DiskWindow, piece: DiskReadPiece) -> FilePort {
        FilePort {
            io,
            file,
            window,
            options: piece.options(),
        }
    }

    /// The window, for the copies that read from it.
    pub fn window(&self) -> &DiskWindow {
        &self.window
    }
}

impl ReadPort for FilePort {
    fn start(&self, spec: ReadSpec, done: ReadDone) -> io::Result<()> {
        let handle = match (&self.file.direct, spec.direct) {
            (Some(direct), true) => direct,
            _ => &self.file.buffered,
        };
        let memory = self
            .window
            .slot_memory(spec.slot, spec.len)
            .ok_or_else(|| io::Error::other("the read does not lie in the window"))?;
        let file = Arc::clone(&self.file);
        let slot = spec.slot;
        let then = Box::new(move |ended: IoDoneOwned<SlotMemory>| {
            step(Kind::DiskRead, slot, move || {
                // The file and the memory are let go before the arrival is told: nothing touches the range after it.
                let IoDoneOwned { buf, result, .. } = ended;
                drop((buf, file));
                done(result);
            })
        });
        self.io
            .read_into_with(raw_file(handle), spec.offset, memory, self.options, then)
    }

    fn has_direct(&self) -> bool {
        self.file.direct.is_some()
    }

    fn file_len(&self) -> u64 {
        self.file.len
    }
}

/// What [`DiskReader::claim`] found.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Claimed {
    /// The read is in the slot and the copy owns it: no gate was taken; the stage may be published.
    Ready,
    /// The read has not arrived: the gate was taken and is released when it does (or failed if the reader closes).
    Held,
}

/// What the owner is told.
pub enum DiskMessage {
    /// A quantum's uses, in the order its device work consumes them.
    Enqueue {
        ticket: u64,
        uses: Vec<DiskUse>,
        fail: FailHandle,
    },
    /// The quantum ended before its device work consumed all its uses.
    Abandon { ticket: u64 },
    /// `ticket` ends when `kill` is set: a later [`DiskMessage::Killed`] of the same flag abandons it. Forgotten once
    /// nothing else holds the flag.
    Watch { ticket: u64, kill: Arc<AtomicBool> },
    /// `kill` was set: the tickets watching it are abandoned.
    Killed { kill: Arc<AtomicBool> },
    /// A read of `slot` ended.
    Arrived {
        slot: usize,
        ticket: u64,
        result: io::Result<usize>,
    },
    /// The copy of `ticket` that consumed `slot` ended: the slot may be read into again.
    SlotFreed { slot: usize, ticket: u64 },
    /// Stop reading; `closed` is sent when no read is in flight.
    Close { closed: Arc<Handoff<()>> },
}

impl fmt::Debug for DiskMessage {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        match self {
            DiskMessage::Enqueue { ticket, uses, .. } => write!(formatter, "Enqueue({ticket}, {} uses)", uses.len()),
            DiskMessage::Abandon { ticket } => write!(formatter, "Abandon({ticket})"),
            DiskMessage::Watch { ticket, .. } => write!(formatter, "Watch({ticket})"),
            DiskMessage::Killed { .. } => write!(formatter, "Killed"),
            DiskMessage::Arrived { slot, ticket, .. } => write!(formatter, "Arrived({slot}, {ticket})"),
            DiskMessage::SlotFreed { slot, ticket } => write!(formatter, "SlotFreed({slot}, {ticket})"),
            DiskMessage::Close { .. } => write!(formatter, "Close"),
        }
    }
}

/// What a turn leaves to be done once the owner's claim is released, in order.
pub(crate) enum Finished {
    /// A quantum hears that a read failed.
    Fail(FailHandle, String),
    /// A copy's read is in: its stage is published.
    Release(Gate),
    /// A copy's stage will never get its read: its quantum fails.
    FailGate(Gate, StageError),
}

impl Finished {
    fn spend(self) {
        match self {
            Finished::Fail(handle, why) => handle(why),
            Finished::Release(gate) => gate.release(),
            Finished::FailGate(gate, error) => gate.fail(error),
        }
    }
}

#[derive(Debug)]
struct Shared {
    in_flight: AtomicUsize,
}

struct Pending {
    ticket: u64,
    use_: DiskUse,
    fail: FailHandle,
}

enum State {
    Idle,
    /// A read is out; `direct` says which handle it uses and `discard` that its quantum was abandoned.
    Reading {
        pending: Pending,
        direct: bool,
        discard: bool,
    },
    /// Read, and no copy has claimed it. `failure` is why the read failed: the claim of the copy fails with it, so no
    /// copy is ever handed a slot whose bytes are wrong, however it races the quantum being told.
    Filled {
        ticket: u64,
        failure: Option<String>,
    },
    /// The copy of `ticket` owns the bytes until it frees the slot.
    Claimed {
        ticket: u64,
    },
}

struct Slot {
    state: State,
    queue: VecDeque<Pending>,
    /// Copies waiting for their read, in the order the reads are made.
    claims: VecDeque<(u64, Gate)>,
}

/// The owner: the state of every slot, changed only by its turns.
pub(crate) struct DiskOwner {
    inbox: Consumer<DiskMessage, Single>,
    port: Arc<dyn ReadPort>,
    address: Arc<Address<DiskOwner>>,
    shared: Arc<Shared>,
    slots: Vec<Slot>,
    /// Tickets abandoned before their uses were posted.
    early: BTreeSet<u64>,
    /// The highest ticket registered or abandoned before registration: a ticket at or below it that the slots do not
    /// hold is over.
    highest: Option<u64>,
    direct_works: bool,
    closed: bool,
    closed_signals: Vec<Arc<Handoff<()>>>,
    told: Vec<Finished>,
    /// The tickets that end when their kill flag is set.
    watching: Vec<(u64, Arc<AtomicBool>)>,
}

fn abandoned_error() -> StageError {
    StageError::Fatal("the quantum of the copy ended before its read".into())
}

fn closed_error() -> StageError {
    StageError::Fatal("the disk reader is closed".into())
}

impl DiskOwner {
    fn in_flight(&self) -> usize {
        self.shared.in_flight.load(Ordering::SeqCst)
    }

    fn handle(&mut self, message: DiskMessage) {
        match message {
            DiskMessage::Enqueue { ticket, uses, fail } => self.enqueue(ticket, uses, fail),
            DiskMessage::Abandon { ticket } => self.abandon(ticket),
            DiskMessage::Watch { ticket, kill } => {
                // A flag nothing else holds can no longer be set.
                self.watching.retain(|(_, flag)| Arc::strong_count(flag) > 1);
                if kill.load(Ordering::Acquire) {
                    self.abandon(ticket);
                } else {
                    self.watching.push((ticket, kill));
                }
            }
            DiskMessage::Killed { kill } => {
                let (killed, kept): (Vec<_>, Vec<_>) = mem::take(&mut self.watching)
                    .into_iter()
                    .partition(|(_, flag)| Arc::ptr_eq(flag, &kill));
                self.watching = kept;
                for (ticket, _) in killed {
                    self.abandon(ticket);
                }
            }
            DiskMessage::Arrived { slot, ticket, result } => self.arrive(slot, ticket, result),
            DiskMessage::SlotFreed { slot, ticket } => {
                if let Some(state) = self.slots.get_mut(slot).map(|slot| &mut slot.state)
                    && matches!(state, State::Claimed { ticket: claimed } if *claimed == ticket)
                {
                    *state = State::Idle;
                }
            }
            DiskMessage::Close { closed } => self.close(closed),
        }
    }

    fn enqueue(&mut self, ticket: u64, uses: Vec<DiskUse>, fail: FailHandle) {
        self.highest = self.highest.max(Some(ticket));
        let abandoned = self.early.remove(&ticket);
        self.early.retain(|early| *early > ticket);
        if abandoned {
            return;
        }
        if self.closed {
            self.told.push(Finished::Fail(fail, "the disk reader is closed".into()));
            return;
        }
        for use_ in uses {
            let slot = use_.slot;
            self.slots[slot].queue.push_back(Pending {
                ticket,
                use_,
                fail: Arc::clone(&fail),
            });
        }
    }

    fn abandon(&mut self, ticket: u64) {
        let known = self.slots.iter().any(|slot| {
            slot.queue.iter().any(|pending| pending.ticket == ticket)
                || slot.claims.iter().any(|(claimed, _)| *claimed == ticket)
                || matches!(&slot.state, State::Reading { pending, .. } if pending.ticket == ticket)
                || matches!(&slot.state, State::Filled { ticket: filled, .. } if *filled == ticket)
        });
        // A ticket the owner has never heard of is either to come (remembered) or over (nothing to do): tickets grow,
        // so one below every ticket posted that is not here is over. Remembering it too costs one entry, pruned by the
        // next enqueue above it.
        if !known {
            self.early.insert(ticket);
            self.highest = self.highest.max(Some(ticket));
        }
        for slot in &mut self.slots {
            slot.queue.retain(|pending| pending.ticket != ticket);
            let mut kept = VecDeque::new();
            for (claimed, gate) in mem::take(&mut slot.claims) {
                if claimed == ticket {
                    self.told.push(Finished::FailGate(gate, abandoned_error()));
                } else {
                    kept.push_back((claimed, gate));
                }
            }
            slot.claims = kept;
            match &mut slot.state {
                State::Reading { pending, discard, .. } if pending.ticket == ticket => *discard = true,
                State::Filled { ticket: filled, .. } if *filled == ticket => slot.state = State::Idle,
                _ => {}
            }
        }
    }

    /// Starts the next read of `slot` if it is idle and has a use waiting.
    fn dispatch(&mut self, slot: usize) {
        while !self.closed && matches!(self.slots[slot].state, State::Idle) {
            let Some(pending) = self.slots[slot].queue.pop_front() else {
                return;
            };
            let use_ = &pending.use_;
            let direct = self.direct_works
                && self.port.has_direct()
                && use_.aligned_offset() + use_.read_bytes() <= self.port.file_len();
            self.launch(slot, pending, direct);
        }
    }

    fn launch(&mut self, slot: usize, pending: Pending, direct: bool) {
        let use_ = &pending.use_;
        // Whole blocks past the page cache, else exactly the payload (and its lead) through it.
        let len = if direct {
            use_.read_bytes()
        } else {
            use_.lead() + use_.bytes
        };
        let spec = ReadSpec {
            slot,
            offset: use_.aligned_offset(),
            len: len as usize,
            direct,
        };
        let ticket = pending.ticket;
        self.slots[slot].state = State::Reading {
            pending,
            direct,
            discard: false,
        };
        self.shared.in_flight.fetch_add(1, Ordering::SeqCst);
        let address = Arc::clone(&self.address);
        let done: ReadDone = Box::new(move |result| address.post(DiskMessage::Arrived { slot, ticket, result }));
        if let Err(refused) = self.port.start(spec, done) {
            // Nothing was started: the use ends as a failed read.
            self.arrive(slot, ticket, Err(refused));
        }
    }

    fn arrive(&mut self, slot: usize, ticket: u64, result: io::Result<usize>) {
        let State::Reading {
            pending,
            direct,
            discard,
        } = mem::replace(&mut self.slots[slot].state, State::Idle)
        else {
            log::error!("a read of window slot {slot} ended with none out");
            return;
        };
        self.shared.in_flight.fetch_sub(1, Ordering::SeqCst);
        debug_assert_eq!(pending.ticket, ticket);
        let needed = (pending.use_.lead() + pending.use_.bytes) as usize;
        let result = result.and_then(|moved| {
            if moved >= needed {
                Ok(())
            } else {
                Err(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    format!("the artifact ended after {moved} of {needed} bytes"),
                ))
            }
        });
        if discard || self.closed {
            return;
        }
        if let Err(refused) = &result
            && direct
            && refused.raw_os_error().is_some_and(refuses_alignment)
        {
            // The file system takes the direct open and refuses the read: every read goes through the page cache now.
            log::info!("direct reads of the artifact are refused ({refused}); reading through the page cache");
            self.direct_works = false;
            self.launch(slot, pending, false);
            return;
        }
        let failure = result.err().map(|failed| {
            let why = format!("reading '{}' from the artifact failed: {failed}", pending.use_.name);
            log::error!("{why}");
            self.told.push(Finished::Fail(Arc::clone(&pending.fail), why.clone()));
            why
        });
        let state = &mut self.slots[slot];
        match state.claims.front() {
            Some((claimed, _)) if *claimed == ticket => {
                state.state = State::Claimed { ticket };
                if let Some((_, gate)) = state.claims.pop_front() {
                    self.told.push(match failure {
                        Some(why) => Finished::FailGate(gate, StageError::Fatal(why)),
                        None => Finished::Release(gate),
                    });
                }
            }
            // No copy of this read has claimed it yet (a copy waiting is another quantum's, and ordered after).
            _ => state.state = State::Filled { ticket, failure },
        }
    }

    /// Whether the bytes `ticket`'s copy of `slot` wants are in (`Ok(true)`: the copy owns them now) or not yet
    /// (`Ok(false)`: the copy must wait), or why it can never have them.
    fn judge(&mut self, slot: usize, ticket: u64) -> Result<bool, StageError> {
        if self.closed {
            return Err(closed_error());
        }
        let Some(state) = self.slots.get_mut(slot) else {
            return Err(StageError::Fatal(format!(
                "a copy claimed an unknown window slot {slot}"
            )));
        };
        if let State::Filled {
            ticket: filled,
            failure,
        } = &mut state.state
            && *filled == ticket
        {
            let failure = failure.take();
            // The copy owns the slot either way: a failed one hands it back when its quantum ends.
            state.state = State::Claimed { ticket };
            return failure.map_or(Ok(true), |why| Err(StageError::Fatal(why)));
        }
        let held = state.queue.iter().any(|pending| pending.ticket == ticket)
            || matches!(&state.state, State::Reading { pending, discard: false, .. } if pending.ticket == ticket);
        if !held && self.highest.is_some_and(|highest| ticket <= highest) {
            // The quantum ended before its copy asked: nothing of it is on the slot, and the slot is left alone.
            return Err(abandoned_error());
        }
        Ok(false)
    }

    /// The copy of `ticket`'s next use of `slot` wants its bytes: `Ready` if they are in, else `hold` is called for the
    /// gate that is kept until they arrive.
    fn claim(&mut self, slot: usize, ticket: u64, hold: impl FnOnce() -> Gate) -> Claimed {
        match self.judge(slot, ticket) {
            Ok(true) => Claimed::Ready,
            Ok(false) => {
                self.wait(slot, ticket, hold());
                Claimed::Held
            }
            Err(error) => {
                self.told.push(Finished::FailGate(hold(), error));
                Claimed::Held
            }
        }
    }

    /// As [`claim`](Self::claim) for a copy whose gate is taken already: it is released when the bytes are in.
    fn claim_held(&mut self, slot: usize, ticket: u64, gate: Gate) {
        match self.judge(slot, ticket) {
            Ok(true) => self.told.push(Finished::Release(gate)),
            Ok(false) => self.wait(slot, ticket, gate),
            Err(error) => self.told.push(Finished::FailGate(gate, error)),
        }
    }

    fn wait(&mut self, slot: usize, ticket: u64, gate: Gate) {
        self.slots[slot].claims.push_back((ticket, gate));
        self.dispatch(slot);
    }

    fn close(&mut self, closed: Arc<Handoff<()>>) {
        self.closed = true;
        self.closed_signals.push(closed);
        self.fail_waiting(&closed_error());
    }

    /// Fails the gates still waiting and drops the uses not read.
    fn fail_waiting(&mut self, error: &StageError) {
        for slot in &mut self.slots {
            slot.queue.clear();
            for (_, gate) in mem::take(&mut slot.claims) {
                self.told.push(Finished::FailGate(gate, error.clone()));
            }
        }
    }

    fn signal_closed(&mut self) {
        if self.closed && self.in_flight() == 0 {
            for closed in mem::take(&mut self.closed_signals) {
                closed.send(());
            }
        }
    }
}

/// Whether an OS error is the one a direct read gets for a request the file system cannot align: `EINVAL` on Unix,
/// `ERROR_INVALID_PARAMETER` on Windows.
pub(crate) fn refuses_alignment(code: i32) -> bool {
    const EINVAL: i32 = 22;
    const ERROR_INVALID_PARAMETER: i32 = 87;
    if cfg!(windows) {
        code == ERROR_INVALID_PARAMETER
    } else {
        code == EINVAL
    }
}

impl Turn for DiskOwner {
    type Message = DiskMessage;
    type Finished = Finished;

    fn turn(&mut self) {
        while let Some(message) = self.inbox.poll() {
            self.handle(message);
        }
        for slot in 0..self.slots.len() {
            self.dispatch(slot);
        }
        self.signal_closed();
    }

    fn take_finished(&mut self) -> Vec<Finished> {
        mem::take(&mut self.told)
    }

    fn abandon(&mut self, why: &str) -> Vec<Finished> {
        self.fail_waiting(&StageError::Fatal(why.to_string()));
        self.signal_closed();
        mem::take(&mut self.told)
    }

    fn deliver(finished: Finished) {
        finished.spend();
    }

    /// Stops reading; the hand-off is sent when no read is in flight.
    fn shut(&mut self, idle: Arc<Handoff<()>>) {
        self.close(idle);
        self.signal_closed();
    }

    fn into_barrier(finished: Finished) -> Result<Arc<Handoff<()>>, Finished> {
        Err(finished)
    }

    fn idle(&self) -> bool {
        self.in_flight() == 0
            && self.slots.iter().all(|slot| {
                slot.queue.is_empty() && slot.claims.is_empty() && !matches!(slot.state, State::Reading { .. })
            })
    }
}

/// The reader: its owner in a mailbox, and what is read without it.
#[derive(Clone)]
pub struct DiskReader {
    mailbox: Arc<Mailbox<DiskOwner>>,
    shared: Arc<Shared>,
    geometry: WindowGeometry,
}

impl fmt::Debug for DiskReader {
    fn fmt(&self, formatter: &mut Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiskReader")
            .field("geometry", &self.geometry)
            .finish_non_exhaustive()
    }
}

impl DiskReader {
    /// A reader of `port` into a window of `geometry`; its hand-offs to a worker go through `spawn`.
    pub fn new(port: Arc<dyn ReadPort>, geometry: WindowGeometry, spawn: Spawn) -> DiskReader {
        let queue = MpscQueue::<DiskMessage>::new(INBOX_CHUNK);
        let (producer, inbox) = (queue.producer(), queue.consumer());
        let address = Address::new();
        let shared = Arc::new(Shared {
            in_flight: AtomicUsize::new(0),
        });
        let owner = DiskOwner {
            inbox,
            port,
            address: Arc::clone(&address),
            shared: Arc::clone(&shared),
            slots: (0..geometry.slots)
                .map(|_| Slot {
                    state: State::Idle,
                    queue: VecDeque::new(),
                    claims: VecDeque::new(),
                })
                .collect(),
            early: BTreeSet::new(),
            highest: None,
            direct_works: true,
            closed: false,
            closed_signals: Vec::new(),
            told: Vec::new(),
            watching: Vec::new(),
        };
        let mailbox = Arc::new(Mailbox::new(owner, producer, spawn));
        address.set(&mailbox);
        DiskReader {
            mailbox,
            shared,
            geometry,
        }
    }

    /// Registers `ticket`'s uses, in the order its device work consumes them. Called when the quantum starts, in start
    /// order, with tickets that grow with it. `fail` is told once for each read that fails.
    pub fn enqueue(&self, ticket: u64, uses: Vec<DiskUse>, fail: FailHandle) -> Result<(), DiskError> {
        for use_ in &uses {
            if use_.slot >= self.geometry.slots {
                return Err(DiskError::NoSuchSlot {
                    name: use_.name.clone(),
                    slot: use_.slot,
                    slots: self.geometry.slots,
                });
            }
            if use_.bytes == 0 || use_.file_offset.checked_add(use_.bytes).is_none() {
                return Err(DiskError::Empty {
                    name: use_.name.clone(),
                });
            }
            if use_.read_bytes() > self.geometry.slot_bytes {
                return Err(DiskError::TooLarge {
                    name: use_.name.clone(),
                    bytes: use_.read_bytes(),
                    slot_bytes: self.geometry.slot_bytes,
                });
            }
        }
        self.mailbox.post(DiskMessage::Enqueue { ticket, uses, fail });
        Ok(())
    }

    /// `ticket` stopped before its device work consumed all its uses. At any time, before or after its uses were posted.
    pub fn abandon(&self, ticket: u64) {
        self.mailbox.post(DiskMessage::Abandon { ticket });
    }

    /// `ticket` ends when `kill` is set (see [`DiskReader::killed`]). Posted with its uses, so a flag set before the
    /// quantum started abandons it at once.
    pub fn watch(&self, ticket: u64, kill: Arc<AtomicBool>) {
        self.mailbox.post(DiskMessage::Watch { ticket, kill });
    }

    /// `kill` was set: every ticket watching it is abandoned, so a copy waiting for a read of its own gives its gate up
    /// and its quantum can end. Called by whoever sets the flag, after setting it.
    pub fn killed(&self, kill: &Arc<AtomicBool>) {
        self.mailbox.post(DiskMessage::Killed { kill: Arc::clone(kill) });
    }

    /// The copy of `ticket` that consumed `slot` ended (its marker completed): the slot may be read into again. A free
    /// of a slot that `ticket` does not hold is ignored.
    pub fn slot_freed(&self, slot: usize, ticket: u64) {
        self.mailbox.post(DiskMessage::SlotFreed { slot, ticket });
    }

    /// Stops reading. The hand-off is sent once no read is in flight (it waits for reads, not for the copies that
    /// claimed a slot: free the window only when every quantum has retired too). Keep the reader alive until then.
    pub fn close(&self) -> Arc<Handoff<()>> {
        let closed = Handoff::new();
        self.mailbox.post(DiskMessage::Close {
            closed: Arc::clone(&closed),
        });
        closed
    }

    /// Reads submitted and not yet ended.
    pub fn reads_in_flight(&self) -> usize {
        self.shared.in_flight.load(Ordering::SeqCst)
    }

    /// The copy of `ticket`'s next use of `slot`, from the transfer stage's `pass_gate`. Never waits.
    ///
    /// - `Ready`: the read is in the slot and the copy owns it; `hold` was not called, and the stage may be published
    ///   (`pass_gate` returns `true`).
    /// - `Held`: `hold` was called (it must be `GateRef::hold`, which takes the stage's gate) and `pass_gate` must
    ///   return `false`. The gate is released when the read arrives (the stage is then published and its copy queued)
    ///   or when `ticket` is abandoned; it is failed if the reader closes. It may be released before `pass_gate`
    ///   returns, from another thread.
    ///
    /// Called once per use, in the order the device consumes the slot's uses. A claim runs in `pass_gate`, a hot scope:
    /// when the owner is busy on another thread the gate is handed to it and the call returns `Held` at once.
    pub fn claim(&self, slot: usize, ticket: u64, hold: impl FnOnce() -> Gate) -> Claimed {
        let mut hold = Some(hold);
        let outcome = self.mailbox.try_with(|owner| match hold.take() {
            Some(hold) => {
                let claimed = owner.claim(slot, ticket, hold);
                (claimed, mem::take(&mut owner.told))
            }
            None => (Claimed::Held, Vec::new()),
        });
        match outcome {
            Ok(Some((claimed, told))) => {
                told.into_iter().for_each(Finished::spend);
                claimed
            }
            // Closed: the owner is gone, and the gate fails.
            Ok(None) => {
                if let Some(hold) = hold.take() {
                    hold().fail(closed_error());
                }
                Claimed::Held
            }
            // The owner is busy on another thread: it takes the gate and decides.
            Err(work) => {
                drop(work);
                if let Some(hold) = hold.take() {
                    let gate = hold();
                    self.mailbox.defer(move |owner| match owner {
                        Some(owner) => {
                            owner.claim_held(slot, ticket, gate);
                            mem::take(&mut owner.told)
                        }
                        None => vec![Finished::FailGate(gate, closed_error())],
                    });
                }
                Claimed::Held
            }
        }
    }
}

/// The slots of a chunk of the inbox: messages are a few per use.
const INBOX_CHUNK: usize = 64;

#[cfg(test)]
#[allow(
    clippy::disallowed_methods,
    clippy::disallowed_types,
    reason = "the fake port's reads are shared between the test and the closures the reader is given, and the scratch file of the real ring is made and found outside the lattice"
)]
mod tests;
