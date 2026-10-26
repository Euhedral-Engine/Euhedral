//! The graphs a sequence's quanta have been captured into, by capture key.
//!
//! A key is seen once and its quantum runs stage by stage. On the second sight the quantum's launches are recorded
//! into a graph, which is launched for that quantum, and kept: every later quantum with the key is one graph launch.
//! The cache belongs to the sequence's owner, which admits one quantum at a time ([`GraphCache::admit`]); the slots it
//! hands out are shared with the quanta in flight, and a slot's graph is written once, by the quantum that records
//! it, so no lock is needed.
//!
//! A graph repeats the allocations it captured, so a key names them ([`CaptureKey::buffer_generations`]) and a cache
//! serves one sequence: another sequence's allocations key differently. Its capacity is the plan's
//! [`capture_key_space`](crate::plan::Plan::capture_key_space), the keys one sequence can produce; a key beyond it
//! evicts the key admitted longest ago, whose graph is freed once the quanta that hold it have retired.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};

use euhedral_inference_cuda::{CudaError, CudaLane, GraphExec, GraphRecorder, Submission, SubmitMode};

use super::capture_key::CaptureKey;

/// One key's graph, once it is recorded.
#[derive(Debug)]
pub struct GraphSlot {
    graph: OnceLock<GraphExec>,
    recording: AtomicBool,
    broken: AtomicBool,
    replays: AtomicU64,
    /// The hash of the submissions the graph was recorded from, for the test oracle (0: none).
    hash: AtomicU64,
}

impl GraphSlot {
    fn new() -> GraphSlot {
        GraphSlot {
            graph: OnceLock::new(),
            recording: AtomicBool::new(false),
            broken: AtomicBool::new(false),
            replays: AtomicU64::new(0),
            hash: AtomicU64::new(0),
        }
    }

    /// The recorded graph, if there is one.
    pub fn graph(&self) -> Option<&GraphExec> {
        self.graph.get()
    }

    /// Whether the key's quanta can no longer be captured (the graph could not be recorded or instantiated); they run
    /// stage by stage.
    pub fn is_broken(&self) -> bool {
        self.broken.load(Ordering::Acquire)
    }

    /// Launches of the recorded graph so far, the launch that recorded it included.
    pub fn launches(&self) -> u64 {
        self.replays.load(Ordering::Relaxed)
    }

    pub(crate) fn launched(&self) {
        self.replays.fetch_add(1, Ordering::Relaxed);
    }

    /// Claims the right to record; false when another quantum is recording this key.
    pub(crate) fn claim(&self) -> bool {
        self.recording
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .is_ok()
    }

    pub(crate) fn release(&self) {
        self.recording.store(false, Ordering::Release);
    }

    /// Keeps the recorded graph. The claim is released.
    pub(crate) fn publish(&self, graph: GraphExec) -> &GraphExec {
        let kept = self.graph.get_or_init(|| graph);
        self.release();
        kept
    }

    /// The key cannot be captured: its quanta run stage by stage from now on. The claim is released.
    pub(crate) fn abandon(&self) {
        self.broken.store(true, Ordering::Release);
        self.release();
    }

    /// The hash the graph was recorded from (the test oracle), or 0.
    pub fn recorded_hash(&self) -> u64 {
        self.hash.load(Ordering::Acquire)
    }

    pub(crate) fn set_recorded_hash(&self, hash: u64) {
        self.hash.store(hash, Ordering::Release);
    }
}

/// What the cache decided for a quantum.
#[derive(Clone, Debug)]
pub enum Admission {
    /// Not captured: its key is new (the first sight), it has none, or its graph could not be recorded. It runs stage
    /// by stage.
    Eager,
    /// Captured: it records the slot's graph if the slot has none, and launches it.
    Captured(Arc<GraphSlot>),
}

#[derive(Debug)]
struct Entry {
    slot: Arc<GraphSlot>,
    sightings: u32,
    admitted: u64,
}

/// The captured graphs of one sequence.
#[derive(Debug)]
pub struct GraphCache {
    capacity: usize,
    entries: HashMap<CaptureKey, Entry>,
    clock: u64,
    evictions: u64,
}

impl GraphCache {
    /// A cache that captures nothing: every quantum runs stage by stage. The reference that a capturing cache is
    /// compared with.
    pub fn disabled() -> GraphCache {
        GraphCache {
            capacity: 0,
            entries: HashMap::new(),
            clock: 0,
            evictions: 0,
        }
    }

    /// A cache for up to `capacity` keys (the plan's `capture_key_space`).
    pub fn with_capacity(capacity: usize) -> GraphCache {
        GraphCache {
            capacity: capacity.max(1),
            entries: HashMap::new(),
            clock: 0,
            evictions: 0,
        }
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Keys the cache holds.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Keys with a recorded graph.
    pub fn recorded(&self) -> usize {
        self.entries
            .values()
            .filter(|entry| entry.slot.graph().is_some())
            .count()
    }

    /// Keys evicted to make room.
    pub fn evictions(&self) -> u64 {
        self.evictions
    }

    /// Admits a quantum with `key` (`None`: it is not capturable). The first sight of a key runs stage by stage; from
    /// the second the quantum is captured.
    pub fn admit(&mut self, key: Option<CaptureKey>) -> Admission {
        let Some(key) = key else { return Admission::Eager };
        if self.capacity == 0 {
            return Admission::Eager;
        }
        self.clock += 1;
        if !self.entries.contains_key(&key) && self.entries.len() >= self.capacity {
            self.evict();
        }
        let clock = self.clock;
        let entry = self.entries.entry(key).or_insert_with(|| Entry {
            slot: Arc::new(GraphSlot::new()),
            sightings: 0,
            admitted: clock,
        });
        entry.admitted = clock;
        entry.sightings = entry.sightings.saturating_add(1);
        if entry.sightings < 2 || entry.slot.is_broken() {
            Admission::Eager
        } else {
            Admission::Captured(Arc::clone(&entry.slot))
        }
    }

    /// Records the graph of `key` on `lane` on its second sight, or finds it: the first call with a key returns
    /// [`Sighting::First`] and records nothing, the second records the launches `record` queues on a submission of
    /// `lane`, and every later call returns the same slot. The graph is not launched. The cache's [`admit`] and the
    /// quantum's own stage do this for a running engine; this is the same step for a caller that holds the lane.
    ///
    /// [`admit`]: GraphCache::admit
    pub fn get_or_record(
        &mut self,
        key: CaptureKey,
        lane: &CudaLane,
        mode: SubmitMode,
        record: impl FnOnce(&Submission<'_>) -> Result<(), CudaError>,
    ) -> Result<Sighting, CudaError> {
        let slot = match self.admit(Some(key)) {
            Admission::Eager => return Ok(Sighting::First),
            Admission::Captured(slot) => slot,
        };
        if slot.graph().is_none() && slot.claim() {
            match record_on(lane, mode, record) {
                Ok(graph) => {
                    slot.publish(graph);
                }
                Err(error) => {
                    slot.abandon();
                    return Err(error);
                }
            }
        }
        Ok(Sighting::Graph(slot))
    }

    fn evict(&mut self) {
        if let Some(oldest) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.admitted)
            .map(|(key, _)| key.clone())
        {
            self.entries.remove(&oldest);
            self.evictions += 1;
        }
    }
}

/// What [`GraphCache::get_or_record`] found.
#[derive(Debug)]
pub enum Sighting {
    /// The key is new: the quantum runs stage by stage.
    First,
    /// The key's slot, with its graph recorded.
    Graph(Arc<GraphSlot>),
}

/// Records the launches `record` queues on a capture of `lane`, and instantiates the graph.
pub(crate) fn record_on<E>(
    lane: &CudaLane,
    mode: SubmitMode,
    record: impl FnOnce(&Submission<'_>) -> Result<(), E>,
) -> Result<GraphExec, E>
where
    E: From<CudaError>,
{
    let recorder = GraphRecorder::begin(lane)?;
    {
        let submission = recorder.submission(mode)?;
        record(&submission)?;
    }
    Ok(recorder.end()?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::plan::View;
    use crate::stages::QuantumKind;

    fn key(geometry: u64) -> CaptureKey {
        CaptureKey {
            view: View::Decode,
            kind: QuantumKind::Decode,
            rows: 1,
            logits: 1,
            selects: true,
            seeds_draft: false,
            draft_seed: None,
            draft_read: None,
            draft_write: None,
            writes_current_token: false,
            split_geometry: geometry,
            reserved_host_pages: [0, 0],
            replay_rows: Vec::new(),
            buffer_generations: Vec::new(),
        }
    }

    #[test]
    fn a_key_runs_stage_by_stage_once_and_is_captured_from_its_second_sight() {
        let mut cache = GraphCache::with_capacity(8);
        assert!(matches!(cache.admit(Some(key(1))), Admission::Eager), "the first sight");
        let Admission::Captured(slot) = cache.admit(Some(key(1))) else {
            panic!("the second sight is captured")
        };
        assert!(slot.graph().is_none(), "no graph until a quantum records one");
        let Admission::Captured(again) = cache.admit(Some(key(1))) else {
            panic!("the third sight is captured")
        };
        assert!(Arc::ptr_eq(&slot, &again), "one slot per key");
        assert!(
            matches!(cache.admit(Some(key(2))), Admission::Eager),
            "another key starts over"
        );
        assert!(
            matches!(cache.admit(None), Admission::Eager),
            "a quantum with no key is never captured"
        );
        assert_eq!(cache.len(), 2);
    }

    #[test]
    fn a_key_that_cannot_be_captured_runs_stage_by_stage_for_good() {
        let mut cache = GraphCache::with_capacity(8);
        let _ = cache.admit(Some(key(1)));
        let Admission::Captured(slot) = cache.admit(Some(key(1))) else {
            panic!("captured")
        };
        assert!(slot.claim(), "one quantum records");
        assert!(!slot.claim(), "not two at once");
        slot.abandon();
        assert!(slot.is_broken());
        assert!(matches!(cache.admit(Some(key(1))), Admission::Eager));
    }

    #[test]
    fn a_full_cache_evicts_the_key_admitted_longest_ago() {
        let mut cache = GraphCache::with_capacity(2);
        let _ = cache.admit(Some(key(1)));
        let _ = cache.admit(Some(key(2)));
        let _ = cache.admit(Some(key(1)));
        let _ = cache.admit(Some(key(3)));
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.evictions(), 1);
        // Key 2 was evicted, so it is new again; key 1 kept its sightings.
        assert!(matches!(cache.admit(Some(key(1))), Admission::Captured(_)));
        assert!(matches!(cache.admit(Some(key(2))), Admission::Eager));
    }
}
