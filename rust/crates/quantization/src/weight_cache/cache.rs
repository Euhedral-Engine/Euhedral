//! The shared objects: one slot per distinct object, loaded once, alive while any lease on it is.
//!
//! A slot is an `Arc`: the leases are its references, the table holds it only weakly, and the last lease to drop frees
//! the object (the backend's handle drops) and takes the slot out of the table. A load is a frame of its own; engines
//! that ask for an object while it loads leave a callback that the load's completion calls.

use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::mem;
use std::panic::{self, AssertUnwindSafe};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, Weak};
use std::thread;

use euhedral_data_structures::sync::{HotScope, LifecycleMutex};

use super::frames::Frames;

/// The identity of one object: the same file, name and offset are the same bytes.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Key {
    pub file: PathBuf,
    pub name: String,
    pub offset: u64,
}

/// Why an object did not load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LoadError(pub String);

impl fmt::Display for LoadError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.0)
    }
}

impl Error for LoadError {}

/// Where objects come from: the host's memory in production, a counting fake in tests. An object ends when its handle
/// drops, on the thread that dropped the last lease.
pub trait Backend: Send + Sync + 'static {
    type Handle: Send + Sync + 'static;

    /// Reads the `bytes` of the object `key` names. Runs in a frame of its own, with no lock held.
    fn load(&self, key: &Key, bytes: u64) -> Result<Self::Handle, LoadError>;
}

/// What an acquisition ends with: a lease, or why the object did not load.
pub type Outcome<B> = Result<Lease<B>, Arc<LoadError>>;

type Done<B> = Box<dyn FnOnce(Outcome<B>) + Send>;

/// What is called after an object's memory is back.
type Freed = Box<dyn Fn() + Send + Sync>;

struct Inner<B: Backend> {
    backend: B,
    /// Where the free of an object runs: a pinned or device free must not block the worker that dropped the last lease.
    frames: Arc<dyn Frames>,
    freed: Option<Freed>,
    frees: AtomicUsize,
    table: LifecycleMutex<HashMap<Key, Weak<Slot<B>>>>,
    resident: AtomicU64,
    loads: AtomicUsize,
}

struct Slot<B: Backend> {
    cache: Arc<Inner<B>>,
    key: Key,
    bytes: u64,
    result: OnceLock<Result<B::Handle, Arc<LoadError>>>,
    /// The callbacks of the engines that asked while the object loaded. The result is set while this is held, so an
    /// engine either finds the result or is called by the load's completion.
    waiting: LifecycleMutex<Vec<Done<B>>>,
}

impl<B: Backend> Slot<B> {
    fn wait(self: &Arc<Self>, done: Done<B>) {
        let _cold = HotScope::leave();
        let mut waiting = self.waiting.lock();
        if self.result.get().is_none() {
            waiting.push(done);
            return;
        }
        drop(waiting);
        self.deliver(done);
    }

    fn complete(self: &Arc<Self>, result: Result<B::Handle, Arc<LoadError>>) {
        let _cold = HotScope::leave();
        let waiting = {
            let mut waiting = self.waiting.lock();
            if self.result.set(result).is_err() {
                return;
            }
            mem::take(&mut *waiting)
        };
        // One callback that panics must not leave the others uncalled: each runs, and the first panic is raised after.
        let mut first = None;
        for done in waiting {
            if let Err(payload) = panic::catch_unwind(AssertUnwindSafe(|| self.deliver(done))) {
                first.get_or_insert(payload);
            }
        }
        // A completion that runs while a panic unwinds (a load that panicked) raises nothing more.
        if let Some(payload) = first
            && !thread::panicking()
        {
            panic::resume_unwind(payload);
        }
    }

    fn deliver(self: &Arc<Self>, done: Done<B>) {
        match self.result.get() {
            Some(Ok(_)) => done(Ok(Lease { slot: Arc::clone(self) })),
            Some(Err(error)) => done(Err(Arc::clone(error))),
            None => {}
        }
    }
}

impl<B: Backend> Drop for Slot<B> {
    fn drop(&mut self) {
        // The last lease is gone: the object ends in a frame of its own. Until it does, its bytes still count as held,
        // so that an engine is not admitted into memory that is not yet back.
        let free = Free {
            cache: Arc::clone(&self.cache),
            key: self.key.clone(),
            bytes: self.bytes,
            object: self.result.take(),
        };
        self.cache.frees.fetch_add(1, Ordering::AcqRel);
        // A frame source that refuses the task drops it, which frees the object all the same.
        self.cache.frames.submit(Box::new(move || drop(free)));
    }
}

/// The end of one object: dropping it frees the object (with no lock held: its drop may do anything, a new acquisition
/// of the same key included, which finds the slot dead and makes its own), gives its bytes back, takes its key out of
/// the table and tells the cache's observer.
struct Free<B: Backend> {
    cache: Arc<Inner<B>>,
    key: Key,
    bytes: u64,
    object: Option<Result<B::Handle, Arc<LoadError>>>,
}

impl<B: Backend> Drop for Free<B> {
    fn drop(&mut self) {
        let _cold = HotScope::leave();
        drop(self.object.take());
        self.cache.resident.fetch_sub(self.bytes, Ordering::AcqRel);
        {
            let mut table = self.cache.table.lock();
            if table.get(&self.key).is_some_and(|slot| slot.strong_count() == 0) {
                table.remove(&self.key);
            }
        }
        self.cache.frees.fetch_sub(1, Ordering::AcqRel);
        if let Some(freed) = &self.cache.freed {
            freed();
        }
    }
}

/// The load of one slot, run as a frame. A loader dropped without running (a lake that refused the frame) fails the
/// slot, so no engine waits for a load that will not happen.
struct Loader<B: Backend> {
    slot: Option<Arc<Slot<B>>>,
}

impl<B: Backend> Loader<B> {
    fn run(mut self) {
        // The slot stays here while the backend loads: a load that panics drops this loader, which fails the waiters.
        let Some(slot) = &self.slot else { return };
        let result = slot.cache.backend.load(&slot.key, slot.bytes).map_err(Arc::new);
        if result.is_ok() {
            slot.cache.loads.fetch_add(1, Ordering::AcqRel);
        }
        if let Some(slot) = self.slot.take() {
            slot.complete(result);
        }
    }
}

impl<B: Backend> Drop for Loader<B> {
    fn drop(&mut self) {
        if let Some(slot) = self.slot.take() {
            let error = LoadError(format!("the load of {} was never run", slot.key.name));
            slot.complete(Err(Arc::new(error)));
        }
    }
}

/// One reference to a loaded object. The object ends with the last lease.
pub struct Lease<B: Backend> {
    slot: Arc<Slot<B>>,
}

impl<B: Backend> Lease<B> {
    pub fn handle(&self) -> &B::Handle {
        match self.slot.result.get() {
            Some(Ok(handle)) => handle,
            // A lease is made only from a slot whose load succeeded, and the object leaves with the last lease.
            _ => unreachable!("a lease holds a loaded object"),
        }
    }

    pub fn key(&self) -> &Key {
        &self.slot.key
    }

    pub fn bytes(&self) -> u64 {
        self.slot.bytes
    }
}

impl<B: Backend> Clone for Lease<B> {
    fn clone(&self) -> Self {
        Lease {
            slot: Arc::clone(&self.slot),
        }
    }
}

impl<B: Backend> fmt::Debug for Lease<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("Lease")
            .field("key", &self.slot.key)
            .finish_non_exhaustive()
    }
}

/// The objects of the engines scored together, shared: each distinct object is loaded once and held for as long as any
/// engine using it runs.
pub struct WeightCache<B: Backend> {
    inner: Arc<Inner<B>>,
}

impl<B: Backend> WeightCache<B> {
    /// A cache over `backend`; the objects that end are freed in frames of `frames`.
    pub fn new(backend: B, frames: Arc<dyn Frames>) -> WeightCache<B> {
        Self::build(backend, frames, None)
    }

    /// As [`new`](Self::new), and `freed` is called each time an object's memory is back (after its handle dropped and
    /// its bytes left [`resident_bytes`](Self::resident_bytes)): the moment to run admission again.
    pub fn with_freed(backend: B, frames: Arc<dyn Frames>, freed: impl Fn() + Send + Sync + 'static) -> WeightCache<B> {
        Self::build(backend, frames, Some(Box::new(freed)))
    }

    fn build(backend: B, frames: Arc<dyn Frames>, freed: Option<Freed>) -> WeightCache<B> {
        WeightCache {
            inner: Arc::new(Inner {
                backend,
                frames,
                freed,
                frees: AtomicUsize::new(0),
                table: LifecycleMutex::new(HashMap::new()),
                resident: AtomicU64::new(0),
                loads: AtomicUsize::new(0),
            }),
        }
    }

    /// Asks for the object `key` names, `bytes` long. `done` is called once, with a lease on the object when it is
    /// loaded (at once when it already is) or with why it did not load. The first engine to ask for an object that no
    /// lease holds starts its load as a frame of `frames`; the others share it. The caller's lease is its reference:
    /// dropping it gives the reference up.
    pub fn acquire(&self, key: Key, bytes: u64, frames: &dyn Frames, done: impl FnOnce(Outcome<B>) + Send + 'static) {
        let (slot, created) = self.join_or_create(key, bytes);
        slot.wait(Box::new(done));
        if created {
            let loader = Loader { slot: Some(slot) };
            frames.submit(Box::new(move || loader.run()));
        }
    }

    fn join_or_create(&self, key: Key, bytes: u64) -> (Arc<Slot<B>>, bool) {
        let _cold = HotScope::leave();
        let mut table = self.inner.table.lock();
        if let Some(slot) = table.get(&key).and_then(Weak::upgrade) {
            return (slot, false);
        }
        let slot = Arc::new(Slot {
            cache: Arc::clone(&self.inner),
            key: key.clone(),
            bytes,
            result: OnceLock::new(),
            waiting: LifecycleMutex::new(Vec::new()),
        });
        self.inner.resident.fetch_add(bytes, Ordering::AcqRel);
        table.insert(key, Arc::downgrade(&slot));
        (slot, true)
    }

    /// Whether a lease holds the object (still loading included).
    pub fn contains(&self, key: &Key) -> bool {
        let _cold = HotScope::leave();
        self.inner
            .table
            .lock()
            .get(key)
            .is_some_and(|slot| slot.strong_count() > 0)
    }

    /// Bytes of the objects held, those still loading included, until the memory of an object that ended is back.
    pub fn resident_bytes(&self) -> u64 {
        self.inner.resident.load(Ordering::Acquire)
    }

    /// Objects loaded so far.
    pub fn loads(&self) -> usize {
        self.inner.loads.load(Ordering::Acquire)
    }

    /// Objects that ended and whose memory is not back yet.
    pub fn frees_in_flight(&self) -> usize {
        self.inner.frees.load(Ordering::Acquire)
    }

    /// Objects held.
    pub fn size(&self) -> usize {
        let _cold = HotScope::leave();
        self.inner
            .table
            .lock()
            .values()
            .filter(|slot| slot.strong_count() > 0)
            .count()
    }
}

impl<B: Backend> fmt::Debug for WeightCache<B> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("WeightCache")
            .field("resident", &self.resident_bytes())
            .field("loads", &self.loads())
            .finish_non_exhaustive()
    }
}
