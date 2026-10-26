//! Lanes: one CUDA stream, an event ring and a host-mapped completion word.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};

use euhedral_graph::{DeviceQueue, LaneId, Marker};

use crate::error::{CudaError, check};
use crate::library::CudaLibrary;
use crate::memory::{DevicePtr, PinnedBuffer};
use crate::sys::EuhedralCuda;

/// The events a lane records its markers on, in a ring: marker `value` (counting from 1) uses slot
/// `(value - 1) % len`. A marker older than the ring is stale: the lane's later work has reused its event, so a wait
/// on it is refused. The ring is sized to the most markers the plan keeps in flight.
///
/// The ring sits in the library, not in the lane, so a submission on one lane can wait for a marker of another. Lane
/// ids are never reused and the events are destroyed with the library, so a wait never meets a destroyed event.
#[derive(Debug)]
pub(crate) struct EventRing {
    handles: Box<[AtomicU64]>,
    /// Markers issued so far.
    issued: AtomicU64,
}

impl EventRing {
    fn new(handles: Vec<u64>) -> EventRing {
        EventRing {
            handles: handles.into_iter().map(AtomicU64::new).collect(),
            issued: AtomicU64::new(0),
        }
    }

    /// Claims the next marker value, counting from 1.
    pub(crate) fn claim(&self) -> u64 {
        self.issued.fetch_add(1, Ordering::AcqRel) + 1
    }

    /// The event marker `value` records on.
    pub(crate) fn event(&self, value: u64) -> Option<u64> {
        let slot = usize::try_from(value.checked_sub(1)? % self.handles.len() as u64).ok()?;
        Some(self.handles[slot].load(Ordering::Acquire))
    }

    /// The event to wait on for `marker`: `None` for value 0 (always reached), an error for a marker that is not
    /// issued yet or is older than the ring.
    pub(crate) fn waitable(&self, marker: Marker) -> Result<Option<u64>, CudaError> {
        if marker.value == 0 {
            return Ok(None);
        }
        let issued = self.issued.load(Ordering::Acquire);
        if marker.value > issued {
            return Err(CudaError::FutureMarker { marker, issued });
        }
        if marker.value + self.handles.len() as u64 <= issued {
            return Err(CudaError::StaleMarker {
                marker,
                ring: self.handles.len(),
                issued,
            });
        }
        Ok(self.event(marker.value))
    }

    /// Destroys every event. Run once, when the library goes.
    pub(crate) fn destroy(&self, library: &EuhedralCuda) {
        for handle in &self.handles {
            let event = handle.swap(0, Ordering::AcqRel);
            if event != 0 {
                // SAFETY: the event was created for this ring and is destroyed once.
                let _ = unsafe { library.euhedral_cuda_completion_event_destroy(event) };
            }
        }
    }
}

/// No read yet. A count of 2^64 - 1 is unreachable.
const UNREAD: u64 = u64::MAX;

/// The host-mapped 32-bit word a lane's completed work writes, and its widening to 64 bits.
///
/// The device writes only the low 32 bits of a marker value, so the word wraps; [`CompletionWord::completed`] keeps
/// the high half by serial-number arithmetic: the first read is taken as it is, and a later read less than 2^31 ahead
/// of the last one moves the count forward (across a wrap, by the difference); anything else is a read that raced with
/// a newer one and is ignored.
pub(crate) struct CompletionWord {
    word: *const AtomicU32,
    widened: AtomicU64,
}

// SAFETY: the word is only read through an atomic load, and the memory it points into outlives the lane that owns
// this value (the lane keeps the pinned buffer in a field declared after it).
unsafe impl Send for CompletionWord {}
// SAFETY: see `Send`; `widened` is an atomic.
unsafe impl Sync for CompletionWord {}

impl CompletionWord {
    /// Reads `word` as the completion word.
    ///
    /// # Safety
    /// `word` must stay valid for reads for as long as this value is used.
    pub(crate) unsafe fn over(word: *const AtomicU32) -> CompletionWord {
        CompletionWord {
            word,
            widened: AtomicU64::new(UNREAD),
        }
    }

    /// The highest value the device has written, widened: one atomic read of the word.
    pub(crate) fn completed(&self) -> u64 {
        // SAFETY: `over`'s contract keeps the word valid.
        let word = unsafe { &*self.word }.load(Ordering::Acquire);
        let mut previous = self.widened.load(Ordering::Relaxed);
        loop {
            let next = if previous == UNREAD {
                u64::from(word)
            } else {
                widen(previous, word)
            };
            if next == previous {
                return previous;
            }
            match self
                .widened
                .compare_exchange_weak(previous, next, Ordering::Relaxed, Ordering::Relaxed)
            {
                Ok(_) => return next,
                Err(current) => previous = current,
            }
        }
    }
}

/// `previous` advanced to the value whose low 32 bits are `word`, when that value is less than 2^31 ahead.
fn widen(previous: u64, word: u32) -> u64 {
    let ahead = word.wrapping_sub(previous as u32);
    if ahead < 1 << 31 {
        previous + u64::from(ahead)
    } else {
        previous
    }
}

/// One CUDA stream with an event ring and a completion word: a lane of a `LanePool`.
///
/// Work reaches a lane through a [`Submission`](crate::Submission), one at a time: a second `begin` while one is live
/// fails with `LaneBusy`, never blocks. Its markers (event ring) order work across lanes
/// on the device; its completion word, written by [`Submission::signal`](crate::Submission::signal), is what
/// [`DeviceQueue::completed`] reads, so the host learns of progress without a call into CUDA.
pub struct CudaLane {
    library: Arc<CudaLibrary>,
    id: LaneId,
    name: String,
    stream: u64,
    /// Whether a submission or a capture holds the lane.
    owned: AtomicBool,
    capturing: AtomicBool,
    word: CompletionWord,
    word_device: DevicePtr,
    // Declared last: the word above reads this memory until the lane is dropped.
    _word_memory: PinnedBuffer,
}

impl fmt::Debug for CudaLane {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CudaLane")
            .field("id", &self.id)
            .field("stream", &self.stream)
            .finish_non_exhaustive()
    }
}

impl CudaLane {
    /// Creates lane `id` on `library` with a ring of `event_ring` events: the most markers the lane keeps in flight
    /// (not yet waited for by their last consumer). Each id exists once per library.
    pub fn new(library: &Arc<CudaLibrary>, id: LaneId, event_ring: usize) -> Result<Arc<CudaLane>, CudaError> {
        if event_ring == 0 {
            return Err(CudaError::Argument(
                "a lane's event ring needs at least one event".to_string(),
            ));
        }
        let api = library.api();
        // SAFETY: the call takes no arguments.
        let stream = unsafe { api.euhedral_cuda_stream_create() };
        if stream == 0 {
            return Err(CudaError::NullAllocation {
                call: "euhedral_cuda_stream_create",
                bytes: 0,
            });
        }
        let made = Self::attach(library, id, event_ring, stream);
        if made.is_err() {
            // SAFETY: the stream was created above and nothing else holds it.
            let _ = unsafe { api.euhedral_cuda_stream_destroy(stream) };
        }
        made
    }

    fn attach(
        library: &Arc<CudaLibrary>,
        id: LaneId,
        event_ring: usize,
        stream: u64,
    ) -> Result<Arc<CudaLane>, CudaError> {
        let memory = PinnedBuffer::alloc(library, 64)?;
        // SAFETY: the word is only ever accessed through atomic loads and stores, on the host here and by the device
        // through stream memory operations, and `memory` outlives every use of the address (the lane keeps both).
        let word_device = unsafe { memory.device_address()? };
        let word = memory.word(0);
        word.store(0, Ordering::Release);
        // SAFETY: the word lies in `memory`, which the lane keeps for as long as it keeps `word`.
        let word = unsafe { CompletionWord::over(std::ptr::from_ref(word)) };

        let api = library.api();
        let mut handles = Vec::with_capacity(event_ring);
        for _ in 0..event_ring {
            // SAFETY: the call takes no arguments.
            let event = unsafe { api.euhedral_cuda_completion_event_create() };
            if event == 0 {
                EventRing::new(handles).destroy(api);
                return Err(CudaError::NullAllocation {
                    call: "euhedral_cuda_completion_event_create",
                    bytes: 0,
                });
            }
            handles.push(event);
        }
        if let Err((error, ring)) = library.register_ring(id, EventRing::new(handles)) {
            ring.destroy(api);
            return Err(error);
        }
        Ok(Arc::new(CudaLane {
            library: Arc::clone(library),
            id,
            name: format!("cuda lane {}", id.0),
            stream,
            owned: AtomicBool::new(false),
            capturing: AtomicBool::new(false),
            word,
            word_device,
            _word_memory: memory,
        }))
    }

    pub fn id(&self) -> LaneId {
        self.id
    }

    /// Blocks until the lane's queued work finishes. For startup, shutdown and tests.
    /// Refused while the lane is captured into a graph (synchronizing a capturing stream invalidates the capture).
    pub fn synchronize(&self) -> Result<(), CudaError> {
        if self.is_capturing() {
            return Err(CudaError::Capturing("a lane synchronization"));
        }
        // SAFETY: the stream is this lane's live stream.
        check("euhedral_cuda_stream_synchronize", unsafe {
            self.library.api().euhedral_cuda_stream_synchronize(self.stream)
        })
    }

    pub(crate) fn library(&self) -> &Arc<CudaLibrary> {
        &self.library
    }

    pub(crate) fn stream(&self) -> u64 {
        self.stream
    }

    pub(crate) fn word_device(&self) -> DevicePtr {
        self.word_device
    }

    /// Claims the next marker value.
    pub(crate) fn next_value(&self) -> Result<u64, CudaError> {
        Ok(self.library.ring(self.id)?.claim())
    }

    /// Takes the lane for one submission or capture, without blocking.
    pub(crate) fn try_own(&self) -> Result<(), CudaError> {
        self.owned
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| ())
            .map_err(|_| CudaError::LaneBusy(self.id))
    }

    /// Gives the lane back.
    pub(crate) fn disown(&self) {
        self.owned.store(false, Ordering::Release);
    }

    pub(crate) fn is_capturing(&self) -> bool {
        self.capturing.load(Ordering::Acquire)
    }

    /// Marks the lane captured (or not); returns whether it was already.
    pub(crate) fn set_capturing(&self, capturing: bool) -> bool {
        self.capturing.swap(capturing, Ordering::AcqRel)
    }
}

impl DeviceQueue for CudaLane {
    fn completed(&self) -> u64 {
        self.word.completed()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

impl Drop for CudaLane {
    fn drop(&mut self) {
        // SAFETY: the stream is this lane's; destroying it lets queued work finish first. The lane's events stay: a
        // wait on another lane may still be using one, and the library destroys them when it goes.
        let _ = unsafe { self.library.api().euhedral_cuda_stream_destroy(self.stream) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A completion word over a fake mapped word.
    fn fake(initial: u32) -> (Box<AtomicU32>, CompletionWord) {
        let mapped = Box::new(AtomicU32::new(initial));
        // SAFETY: the box outlives the returned word in every test (they drop together).
        let word = unsafe { CompletionWord::over(std::ptr::from_ref(&*mapped)) };
        (mapped, word)
    }

    #[test]
    fn completed_widens_wrapping_u32() {
        let (mapped, word) = fake(0xFFFF_FFF0);
        assert_eq!(word.completed(), 0xFFFF_FFF0);
        mapped.store(0x10, Ordering::Release);
        assert_eq!(word.completed(), (1 << 32) + 0x10);
        mapped.store(0x20, Ordering::Release);
        assert_eq!(word.completed(), (1 << 32) + 0x20);
    }

    #[test]
    fn completed_starts_at_zero_and_never_goes_back() {
        let (mapped, word) = fake(0);
        assert_eq!(word.completed(), 0);
        mapped.store(7, Ordering::Release);
        assert_eq!(word.completed(), 7);
        // A read that raced behind the latest one must not move the count back, or across a phantom wrap.
        mapped.store(5, Ordering::Release);
        assert_eq!(word.completed(), 7);
    }

    #[test]
    fn the_ring_reuses_events_in_order_and_value_zero_has_none() {
        let ring = EventRing::new(vec![10, 11, 12]);
        assert_eq!(ring.event(0), None);
        assert_eq!(
            [1, 2, 3, 4].map(|value| ring.event(value)),
            [Some(10), Some(11), Some(12), Some(10)]
        );
    }

    #[test]
    fn a_wait_is_refused_for_a_marker_not_issued_or_older_than_the_ring() {
        let ring = EventRing::new(vec![10, 11, 12]);
        let marker = |value| Marker { lane: LaneId(0), value };
        assert!(matches!(
            ring.waitable(marker(1)),
            Err(CudaError::FutureMarker { issued: 0, .. })
        ));
        assert_eq!(ring.waitable(marker(0)).unwrap(), None);
        for expected in 1..=4 {
            assert_eq!(ring.claim(), expected);
        }
        // Four issued on a ring of three: marker 1 shares its event with marker 4.
        assert!(matches!(
            ring.waitable(marker(1)),
            Err(CudaError::StaleMarker { ring: 3, issued: 4, .. })
        ));
        assert_eq!(ring.waitable(marker(2)).unwrap(), Some(11));
        assert_eq!(ring.waitable(marker(4)).unwrap(), Some(10));
        assert!(matches!(
            ring.waitable(marker(5)),
            Err(CudaError::FutureMarker { issued: 4, .. })
        ));
    }
}
